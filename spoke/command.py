"""Command dispatch to a local OMLX server.

Sends voice command utterances to a local model via the OpenAI-compatible
chat completions API, streams the response, and maintains a ring buffer
of recent exchanges for conversational context.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Generator

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND_URL = "http://localhost:8001"
_DEFAULT_COMMAND_MODEL = "qwen3p5-35B-A3B"
_DEFAULT_RING_BUFFER_SIZE = 10

_SYSTEM_PROMPT = (
    "You are a local voice assistant invoked by a spoken command. "
    "Be concise. The user spoke this aloud at their desktop."
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
            or os.environ.get("SPOKE_COMMAND_API_KEY")
            or os.environ.get("OMLX_SERVER_API_KEY", "")
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

    def stream_command(self, utterance: str) -> Generator[str, None, str]:
        """Send a command utterance and yield content tokens as they arrive.

        Returns the full assembled response text when the stream ends.
        The response is automatically added to the ring buffer.

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
        body: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if not self._enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        payload = json.dumps(body).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/v1/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        full_response = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith(": "):
                        # SSE comment (keep-alive)
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
                    delta = choices[0].get("delta", {})
                    # Only surface content tokens — skip reasoning_content
                    token = delta.get("content")
                    if token is not None:
                        full_response += token
                        yield token
                    # Surface tool calls as visible text
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name")
                            if name:
                                tool_text = f"\n[calling {name}…]\n"
                                full_response += tool_text
                                yield tool_text
        except urllib.error.URLError as exc:
            logger.error("Command request failed: %s", exc)
            raise
        except Exception:
            logger.exception("Command stream error")
            raise

        # Add to ring buffer (content only, no reasoning)
        self._history.append((utterance, full_response))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return full_response

    def clear_history(self) -> None:
        """Clear the conversation ring buffer."""
        self._history.clear()
