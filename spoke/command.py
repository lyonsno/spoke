"""Command dispatch to a local OMLX server.

Sends voice command utterances to a local model via the OpenAI-compatible
chat completions API, streams the response, and maintains a ring buffer
of recent exchanges for conversational context.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Generator

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND_URL = "http://localhost:8001"
_DEFAULT_COMMAND_MODEL = "qwen3p5-35B-A3B"
_DEFAULT_RING_BUFFER_SIZE = 10

_SYSTEM_PROMPT = (
    "You are a local voice assistant invoked by a spoken command. "
    "Be concise. The user spoke this aloud at their desktop.\n\n"
    "You have tools to interact with the user's screen:\n"
    "- capture_context: captures the frontmost window and returns OCR text "
    "blocks with refs. Use when the user refers to something visible on screen "
    '(e.g. "read that", "what does this say", "read the tab title").\n'
    "- read_aloud: resolves a source ref to exact text and speaks it via TTS. "
    "Use scene_block refs from capture_context, or clipboard/selection/literal refs.\n\n"
    "Prefer refs over regenerated text. If the user asks you to read something "
    "visible, call capture_context first, then read_aloud with a scene_block ref. "
    "If the user asks to read selected text or the clipboard, use read_aloud directly "
    "with selection:frontmost or clipboard:current. Do not restate visible text in "
    "your response when a ref can be spoken instead."
)


class CommandClient:
    """Streaming chat client for voice commands via OMLX."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_history: int | None = None,
    ):
        self._base_url = (
            base_url
            or os.environ.get("SPOKE_COMMAND_URL", _DEFAULT_COMMAND_URL)
        ).rstrip("/")
        self._model = (
            model
            or os.environ.get("SPOKE_COMMAND_MODEL", _DEFAULT_COMMAND_MODEL)
        )
        self._api_key = (
            api_key
            or os.environ.get("SPOKE_COMMAND_API_KEY", "")
        )
        self._max_history = (
            max_history
            if max_history is not None
            else int(os.environ.get("SPOKE_COMMAND_HISTORY", str(_DEFAULT_RING_BUFFER_SIZE)))
        )
        # Thinking: enabled by default, disable with SPOKE_COMMAND_THINKING=0
        self._enable_thinking = os.environ.get("SPOKE_COMMAND_THINKING", "1") != "0"
        # Ring buffer: list of (user_utterance, assistant_response) pairs
        self._history: list[tuple[str, str]] = []

    @property
    def history(self) -> list[tuple[str, str]]:
        return list(self._history)

    def _build_messages(self, utterance: str) -> list[dict]:
        """Assemble the messages array: system + history + current utterance.

        The history is ordered oldest-first so the stable prefix stays
        KV-cache-friendly — new pairs are appended at the end.
        """
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user_text, assistant_text in self._history:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": utterance})
        return messages

    def stream_command(
        self,
        utterance: str,
        *,
        tools: list[dict] | None = None,
        tool_executor: Callable[..., str] | None = None,
    ) -> Generator[str, None, str]:
        """Send a command utterance and yield content tokens as they arrive.

        Returns the full assembled response text when the stream ends.
        The response is automatically added to the ring buffer.

        Parameters
        ----------
        tools : list[dict], optional
            OpenAI function-calling tool schemas to include in the request.
        tool_executor : callable, optional
            Function that executes a tool call locally. Called as
            ``tool_executor(name=..., arguments=...)``. Required if
            tools are provided and the model calls one.

        Yields
        ------
        str
            Individual content token strings as they stream in.

        Returns
        -------
        str
            The complete response text (via generator return value).
        """
        messages = self._build_messages(utterance)
        full_response = ""

        # Allow up to 5 tool call round-trips to prevent infinite loops
        max_tool_rounds = 5

        for _round in range(max_tool_rounds + 1):
            body: dict = {
                "model": self._model,
                "messages": messages,
                "stream": True,
            }
            if not self._enable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if tools:
                body["tools"] = tools

            payload = json.dumps(body).encode()

            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            url = f"{self._base_url}/v1/chat/completions"
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

            # Track tool call deltas for this round
            tool_call_acc = None
            if tools:
                from spoke.tool_dispatch import ToolCallAccumulator
                tool_call_acc = ToolCallAccumulator()

            finish_reason = None

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        if line.startswith(": "):
                            continue
                        if not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping malformed SSE chunk: %s", data_str[:80])
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})

                        # Track finish reason
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr

                        # Content tokens
                        token = delta.get("content")
                        if token is not None:
                            full_response += token
                            yield token

                        # Tool call deltas — accumulate
                        tool_calls_delta = delta.get("tool_calls")
                        if tool_calls_delta and tool_call_acc is not None:
                            for tc_delta in tool_calls_delta:
                                tool_call_acc.feed(tc_delta)

            except urllib.error.URLError as exc:
                logger.error("Command request failed: %s", exc)
                raise
            except Exception:
                logger.exception("Command stream error")
                raise

            # If the model called tools, execute them and loop
            if (
                finish_reason == "tool_calls"
                and tool_call_acc is not None
                and tool_call_acc.has_calls
                and tool_executor is not None
            ):
                completed_calls = tool_call_acc.finish()
                logger.info(
                    "Executing %d tool call(s): %s",
                    len(completed_calls),
                    ", ".join(c["function"]["name"] for c in completed_calls),
                )

                # Add the assistant's tool-call message to the conversation
                messages.append({
                    "role": "assistant",
                    "tool_calls": completed_calls,
                })

                # Execute each tool and add results
                for call in completed_calls:
                    fn_name = call["function"]["name"]
                    try:
                        fn_args = json.loads(call["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_result = tool_executor(name=fn_name, arguments=fn_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    })

                # Continue the loop — next round sends tool results
                continue

            # No tool calls (or no tools configured) — we're done
            break

        # Add to ring buffer (content only, no reasoning)
        self._history.append((utterance, full_response))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return full_response

    def clear_history(self) -> None:
        """Clear the conversation ring buffer."""
        self._history.clear()
