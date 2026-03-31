"""Command dispatch to a local OMLX server.

Sends voice command utterances to a local model via the OpenAI-compatible
chat completions API, streams the response, and maintains a ring buffer
of recent exchanges for conversational context.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Callable, Generator, Literal

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
    "Pass block refs from capture_context directly (e.g., 'scene-abc:block-1'). "
    "Other ref formats: 'clipboard:current', 'selection:frontmost', "
    "'last_response:current', 'literal:text to speak'.\n\n"
    "Prefer refs over regenerated text. If the user asks you to read something "
    "visible, call capture_context first, then read_aloud with a block ref. "
    "If the user asks to read selected text or the clipboard, use read_aloud directly "
    "with selection:frontmost or clipboard:current. Do not restate visible text in "
    "your response when a ref can be spoken instead."
)


@dataclass(frozen=True)
class CommandStreamEvent:
    """Semantic event emitted while streaming a command response."""

    kind: Literal["assistant_delta", "assistant_final", "tool_call"]
    text: str = ""
    tool_name: str | None = None
    tool_arguments: str | None = None


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

    def list_models(self) -> list[str]:
        """Return model ids exposed by the OMLX OpenAI-compatible endpoint."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/v1/models",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [
            model["id"]
            for model in payload.get("data", [])
            if isinstance(model, dict) and model.get("id")
        ]

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
        """Compatibility wrapper yielding only assistant content deltas.

        New code should prefer ``stream_command_events()`` so it can
        distinguish provisional deltas, final assistant content, and tool
        calls without guessing from raw text alone.
        """
        final_response = ""
        for event in self.stream_command_events(
            utterance,
            tools=tools,
            tool_executor=tool_executor,
        ):
            if event.kind == "assistant_delta":
                yield event.text
            elif event.kind == "assistant_final":
                final_response = event.text

        return final_response

    def stream_command_events(
        self,
        utterance: str,
        *,
        tools: list[dict] | None = None,
        tool_executor: Callable[..., str] | None = None,
    ) -> Generator[CommandStreamEvent, None, str]:
        """Send a command utterance and yield semantic stream events.

        Returns the full assembled response text when the stream ends.
        The response is automatically added to the ring buffer.

        Events
        ------
        assistant_delta:
            Provisional visible assistant text from the current round.
        assistant_final:
            Final canonical assistant text that should be treated as durable.
        tool_call:
            Completed function call emitted by the model for the current round.
        """
        messages = self._build_messages(utterance)
        full_response = ""

        # Allow up to 5 tool call round-trips to prevent infinite loops
        max_tool_rounds = 5

        for _round in range(max_tool_rounds + 1):
            if _round > 0:
                logger.info("Tool follow-up round %d starting", _round)
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
            first_token_logged = False
            # Content accumulated during this round only (may be
            # intermediate text during a tool-call turn)
            round_content = ""

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
                            if not first_token_logged:
                                logger.info("First content token on round %d: %r", _round, token[:50] if len(token) > 50 else token)
                                first_token_logged = True
                            round_content += token
                            yield CommandStreamEvent(
                                kind="assistant_delta",
                                text=token,
                            )

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
                for call in completed_calls:
                    yield CommandStreamEvent(
                        kind="tool_call",
                        tool_name=call["function"]["name"],
                        tool_arguments=call["function"]["arguments"],
                    )

                # Add the assistant's tool-call message with any content
                # from this round (required by OpenAI API spec)
                messages.append({
                    "role": "assistant",
                    "content": round_content or None,
                    "tool_calls": completed_calls,
                })

                # Execute each tool and add results
                for call in completed_calls:
                    fn_name = call["function"]["name"]
                    try:
                        fn_args = json.loads(call["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    logger.info("Executing tool %s with args: %s", fn_name, str(fn_args)[:200])
                    tool_result = tool_executor(name=fn_name, arguments=fn_args)
                    logger.info(
                        "Tool %s result: %d chars (preview: %s)",
                        fn_name, len(tool_result), tool_result[:200],
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    })

                logger.info(
                    "Sending tool results back to model (round %d, %d messages total)",
                    _round + 1, len(messages),
                )
                # Continue the loop — next round sends tool results
                continue

            # No tool calls — this round's content is the final response
            full_response = round_content
            yield CommandStreamEvent(kind="assistant_final", text=full_response)
            break

        # Add to ring buffer (content only, no reasoning)
        self._history.append((utterance, full_response))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return full_response

    def clear_history(self) -> None:
        """Clear the conversation ring buffer."""
        self._history.clear()
