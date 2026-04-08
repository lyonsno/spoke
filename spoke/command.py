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
import re
import uuid
from typing import Any, Callable, Generator, Literal

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


def _extract_xml_tool_calls(text: str) -> tuple[str, list[dict]] | None:
    """Extract bare XML tool calls from content text.

    Catches models (e.g. Qwen3-Coder) that emit tool calls as XML in the
    content stream instead of using the structured tool_calls API field.

    Handles:
      <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
      <function=name><parameter=key>value</parameter></function></tool_call>
      <function=name><parameter=key>value</parameter></function>

    Returns (cleaned_text, tool_calls) or None if no XML tool calls found.
    """
    # Match <function=name>...</function> with or without <tool_call> wrapper
    pattern = r"(?:<tool_call>\s*)?<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if not matches:
        return None

    tool_calls = []
    for m in matches:
        func_name = m.group(1)
        params_text = m.group(2)
        arguments = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL
        ):
            key = pm.group(1)
            val = pm.group(2).strip()
            try:
                arguments[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = val
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    # Remove matched XML from content text
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls

_DEFAULT_COMMAND_URL = "http://localhost:8001"
_DEFAULT_COMMAND_MODEL = "qwen3p5-35B-A3B"
_DEFAULT_RING_BUFFER_SIZE = 10


def _rough_tool_result_tokens(text: str) -> int:
    """Approximate token count for a tool result."""
    return int(len(text.split()) * 1.3)

_SYSTEM_PROMPT = (
    "You are an assistant with access to tools for interacting with the user's "
    "development environment, screen, clipboard, email, and project state. "
    "Think carefully and use your tools when appropriate.\n\n"
    "Environment: your working directory is ~/dev, the user's development root. "
    "File tool paths resolve relative to ~/dev, so you can use short paths like "
    "'epistaxis/projects/spoke/epistaxis.md' or 'spoke/spoke/command.py'. "
    "Key repos here include: spoke (this app), omlx (local model server), "
    "mlx-audio (TTS/ASR sidecar), epanorthosis (automated review), and epistaxis "
    "(cross-session state and coordination).\n\n"
    "Epistaxis (~/dev/epistaxis/) is the user's durable off-repo state system. Layout:\n"
    "- projects/<repo>/epistaxis.md — per-repo status, lanes, scoped state\n"
    "- attractors/ — atomic units of work: stimulus, source, satisfaction condition\n"
    "- policy/ — tool-specific policy files (claude/, codex/, gemini/, shared/)\n"
    "- system/epistaxis.md — global system state\n"
    "- metadosis/ — shared mutable artifacts between lanes\n"
    "- prs/<repo>/ — PR tracking per repo\n"
    "- sylloge/ — compressed thread→artifact summaries\n"
    "- autopoiesis/ — self-generated concept genesis and traces\n"
    "- auxesis/ — growth/capability emergence traces\n"
    "- machines/ — per-machine context\n"
    "- reviews/ — review artifacts\n"
    "When the user references epistaxis state, use run_epistaxis_ops.\n\n"
    "You have tools to resolve exact text and act on it:\n"
    "- capture_context: captures the frontmost window and returns OCR text "
    "blocks with refs. Use when the user refers to something visible on screen "
    '(e.g. "read that", "what does this say", "read the tab title").\n'
    "- read_aloud: resolves a source ref to exact text and speaks it via TTS. "
    "Pass block refs from capture_context directly (e.g., 'scene-abc:block-1'). "
    "Other ref formats: 'clipboard:current', 'selection:frontmost', "
    "'last_response:current', 'literal:text to speak'. Use 'literal:' when the "
    "user asks you to say an arbitrary phrase, sentence, or other text that is "
    "not being read from the screen, selection, clipboard, or prior response.\n"
    "- add_to_tray: places exact text into the tray for later insertion or "
    "sending. Use this to save content the user may want to paste or send later.\n\n"
    "You also have run_epistaxis_ops: a bounded local operator for private "
    "Epistaxis review-ticket and pointer work in a dedicated Epistaxis "
    "worktree. Use it only for narrow Epistaxis state operations, never for "
    "arbitrary shell or general coding work.\n\n"
    "You also have query_gmail: a bounded read-only Gmail query tool. Pass "
    "a Gmail search query string (same syntax as the Gmail search bar) and "
    "get back sender, subject, date, and snippet for up to 10 matches. Use "
    "it when the user asks about their email. Keep queries specific to limit "
    "results — combine filters like 'is:starred newer_than:3d' or "
    "'from:alice subject:invoice'.\n\n"
    "Output mode: by default your response is displayed as text on screen — "
    "do NOT call read_aloud unless the user explicitly asks you to say, read, "
    "or speak something. For generated text, lists, code, structured content, "
    "or anything the user would want to read, copy, or reference later, just "
    "respond in plain text.\n\n"
    "When read_aloud IS appropriate (user said 'read this', 'say that', etc.): "
    "prefer refs over regenerated text. If reading something visible, call "
    "capture_context first, then read_aloud with a block ref. For selected text "
    "or the clipboard, use read_aloud directly with selection:frontmost or "
    "clipboard:current. For arbitrary phrases the user asks you to say, use "
    "read_aloud with literal:<exact text>. Use add_to_tray when the user "
    "wants content kept for later use rather than spoken immediately.\n\n"
    "Named commands (MUST be executed via tool calls, NEVER as plain text):\n"
    "- WALLACE: This is a COMMAND, not a name. The user is NOT named Wallace. "
    "You MUST execute this as two sequential tool calls — plain text output is "
    "WRONG for this command:\n"
    "  Step 1: Call capture_context (scope: active_window)\n"
    "  Step 2: Call read_aloud with source_ref 'literal:<your text>' where "
    "<your text> is a paragraph or two you compose riffing on whatever "
    "capture_context returned, written in the style of David Foster Wallace — "
    "digressive, self-aware, hyper-detailed, with nested qualifications and "
    "footnote-energy asides. Comment on what you see the way DFW would in an "
    "essay — noticing the absurd, the deeply human, the thing everyone sees "
    "but nobody says.\n"
    "  IMPORTANT: Do NOT respond with text. You MUST call the tools."
)


@dataclass(frozen=True)
class CommandStreamEvent:
    """Semantic event emitted while streaming a command response."""

    kind: Literal["assistant_delta", "assistant_final", "tool_call", "thinking_delta"]
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
        raw_url = (base_url or _DEFAULT_COMMAND_URL).rstrip("/")
        # Cloud OpenAI-compat endpoints (e.g. Gemini) include the version
        # prefix in the base URL already.  Detect this so we don't double it
        # when building /v1/models and /v1/chat/completions paths.
        from urllib.parse import urlparse
        path = urlparse(raw_url).path.rstrip("/")
        self._url_has_version_prefix = any(
            seg.startswith("v") and seg[1:].replace("beta", "").isdigit()
            for seg in path.split("/") if seg
        )
        self._base_url = raw_url
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
        # Ring buffer: list of message chains (each a list[dict]).
        # Each entry is the full sequence of messages for one turn:
        # [user, assistant, tool_result, assistant, ...] preserving
        # tool calls and results for multi-turn context.
        self._history: list[list[dict]] = []

    @property
    def history(self) -> list[list[dict]]:
        return list(self._history)

    def list_models(self) -> list[str]:
        """Return model ids exposed by the OMLX OpenAI-compatible endpoint."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url}/models" if self._url_has_version_prefix else f"{self._base_url}/v1/models",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [
            model["id"].removeprefix("models/")
            for model in payload.get("data", [])
            if isinstance(model, dict) and model.get("id")
        ]

    def _build_messages(self, utterance: str) -> list[dict]:
        """Assemble the messages array: system + history + current utterance.

        The history is ordered oldest-first so the stable prefix stays
        KV-cache-friendly — new pairs are appended at the end.
        Each history entry is a full message chain including tool calls
        and results from that turn.
        """
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for chain in self._history:
            messages.extend(chain)
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
        visible_response = ""
        final_response = ""
        for event in self.stream_command_events(
            utterance,
            tools=tools,
            tool_executor=tool_executor,
        ):
            if event.kind == "assistant_delta":
                visible_response += event.text
                yield event.text
            elif event.kind == "assistant_final":
                final_response = event.text

        # History is now managed by stream_command_events() which stores
        # the full message chain including tool calls and results.
        return visible_response or final_response

    def stream_command_events(
        self,
        utterance: str,
        *,
        tools: list[dict] | None = None,
        tool_executor: Callable[..., str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Generator[CommandStreamEvent, None, str]:
        """Send a command utterance and yield semantic stream events.

        Returns the full assembled response text when the stream ends.
        The response is automatically added to the ring buffer.

        Parameters
        ----------
        cancel_check:
            Optional callable returning True when the caller wants to
            abort.  Checked at tool-call boundaries and between SSE
            chunks.

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
        # Track where this turn's messages start (after system + history)
        turn_start_idx = len(messages) - 1  # points to the user message
        full_response = ""

        # Safety cap on tool call round-trips.  With cancel_check wired
        # up the user can bail out at any time, so this is just a backstop
        # against genuinely infinite loops (e.g. model keeps retrying the
        # same failing tool).
        max_tool_rounds = 20

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
            logger.info(
                "Sending to model: round=%d model=%s messages=%d tools=%d thinking=%s payload_bytes=%d",
                _round, self._model, len(messages),
                len(tools) if tools else 0,
                self._enable_thinking,
                len(payload),
            )
            if _round == 0:
                # Log the user utterance as sent
                user_msg = messages[-1].get("content", "")
                logger.info("User utterance sent to model: %s", user_msg[:300])

            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            url = f"{self._base_url}/chat/completions" if self._url_has_version_prefix else f"{self._base_url}/v1/chat/completions"
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

            # Track tool call deltas for this round
            from spoke.tool_dispatch import ToolCallAccumulator
            tool_call_acc = ToolCallAccumulator()
            emitted_tool_call_indices: set[int] = set()

            finish_reason = None
            first_token_logged = False
            first_delta_logged = False
            # Content accumulated during this round only (may be
            # intermediate text during a tool-call turn)
            round_content = ""
            # Thinking token state machine for <think>...</think> tags.
            # States: "detect" (haven't seen anything yet),
            #         "thinking" (inside <think> block),
            #         "content" (after </think>, normal content)
            thinking_state = "detect"
            thinking_tag_buf = ""  # partial tag accumulator

            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    for raw_line in resp:
                        # Cancel check between SSE chunks
                        if cancel_check is not None and cancel_check():
                            logger.info("Cancel requested during SSE stream — breaking")
                            resp.close()
                            break
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

                        # Log first delta for debugging thinking detection (once per round)
                        if not first_delta_logged and delta:
                            first_delta_logged = True
                            logger.info(
                                "First SSE delta keys on round %d: %s (preview: %s)",
                                _round, list(delta.keys()),
                                {k: str(v)[:80] for k, v in delta.items()},
                            )

                        # Reasoning tokens (OpenAI reasoning_content field)
                        reasoning_token = delta.get("reasoning_content")
                        if reasoning_token is None:
                            reasoning_token = delta.get("reasoning")
                        if reasoning_token is not None:
                            yield CommandStreamEvent(
                                kind="thinking_delta",
                                text=reasoning_token,
                            )

                        # Content tokens — with <think> tag state machine
                        token = delta.get("content")
                        if token is not None:
                            # Route through thinking state machine
                            text_to_process = token

                            if thinking_state == "detect":
                                # Accumulate until we can tell if it starts with <think>
                                thinking_tag_buf += text_to_process
                                if thinking_tag_buf.startswith("<think>"):
                                    thinking_state = "thinking"
                                    text_to_process = thinking_tag_buf[len("<think>"):]
                                    thinking_tag_buf = ""
                                    if not text_to_process:
                                        continue
                                    # Fall through to "thinking" handler below
                                elif len(thinking_tag_buf) < len("<think>") and "<think>".startswith(thinking_tag_buf):
                                    # Could still be <think>, keep buffering
                                    continue
                                else:
                                    # Not a <think> tag — flush buffer as content
                                    thinking_state = "content"
                                    text_to_process = thinking_tag_buf
                                    thinking_tag_buf = ""

                            if thinking_state == "thinking":
                                # Inside <think> block — check for </think>
                                combined = thinking_tag_buf + text_to_process
                                thinking_tag_buf = ""
                                close_idx = combined.find("</think>")
                                if close_idx >= 0:
                                    # Emit remaining thinking, switch to content
                                    before = combined[:close_idx]
                                    after = combined[close_idx + len("</think>"):]
                                    if before:
                                        yield CommandStreamEvent(kind="thinking_delta", text=before)
                                    thinking_state = "content"
                                    if after:
                                        text_to_process = after
                                    else:
                                        continue
                                else:
                                    # Check for partial </think> at end
                                    for k in range(min(len("</think>"), len(combined)), 0, -1):
                                        if combined.endswith("</think>"[:k]):
                                            thinking_tag_buf = combined[-k:]
                                            combined = combined[:-k]
                                            break
                                    if combined:
                                        yield CommandStreamEvent(kind="thinking_delta", text=combined)
                                    continue

                            # thinking_state == "content" — normal visible content
                            if not first_token_logged:
                                logger.info("First content token on round %d: %r", _round, text_to_process[:50] if len(text_to_process) > 50 else text_to_process)
                                first_token_logged = True
                            round_content += text_to_process
                            yield CommandStreamEvent(
                                kind="assistant_delta",
                                text=text_to_process,
                            )

                        # Tool call deltas — yield name indicators and accumulate
                        tool_calls_delta = delta.get("tool_calls")
                        if tool_calls_delta:
                            for tc_delta in tool_calls_delta:
                                tool_call_acc.feed(tc_delta)
                                idx = tc_delta.get("index")
                                function_delta = tc_delta.get("function") or {}
                                tool_name = function_delta.get("name")
                                if not tool_name or idx is None or idx in emitted_tool_call_indices:
                                    continue
                                indicator = f"\n[calling {tool_name}…]\n"
                                round_content += indicator
                                yield CommandStreamEvent(
                                    kind="assistant_delta",
                                    text=indicator,
                                )
                                emitted_tool_call_indices.add(idx)
                                yield CommandStreamEvent(
                                    kind="tool_call",
                                    tool_name=tool_name,
                                    tool_arguments=function_delta.get("arguments"),
                                )

            except urllib.error.URLError as exc:
                logger.error("Command request failed: %s", exc)
                raise
            except Exception:
                logger.exception("Command stream error")
                raise

            # Fallback: if the model emitted XML tool calls in the content
            # stream (e.g. Qwen3-Coder), extract them as structured calls.
            if not tool_call_acc.has_calls and tool_executor is not None:
                xml_result = _extract_xml_tool_calls(round_content)
                if xml_result is not None:
                    cleaned_text, xml_calls = xml_result
                    logger.info(
                        "Extracted %d XML tool call(s) from content stream: %s",
                        len(xml_calls),
                        ", ".join(c["function"]["name"] for c in xml_calls),
                    )
                    round_content = cleaned_text
                    for i, xc in enumerate(xml_calls):
                        tool_call_acc._calls[i] = xc
                        indicator = f"\n[calling {xc['function']['name']}…]\n"
                        yield CommandStreamEvent(
                            kind="assistant_delta",
                            text=indicator,
                        )
                        yield CommandStreamEvent(
                            kind="tool_call",
                            tool_name=xc["function"]["name"],
                            tool_arguments=xc["function"]["arguments"],
                        )

            # If the model called tools, execute them and loop.
            # Use has_calls as the primary signal — some model servers
            # (MLX, vLLM) return finish_reason="stop" even when the
            # model emitted tool call deltas.
            has_tool_calls = (
                tool_call_acc.has_calls
                and tool_executor is not None
            )
            if has_tool_calls and finish_reason != "tool_calls":
                logger.warning(
                    "Model emitted tool call deltas but finish_reason=%r "
                    "(expected 'tool_calls') — executing anyway",
                    finish_reason,
                )

            # Cancel check at tool-call boundary — before executing tools
            if has_tool_calls and cancel_check is not None and cancel_check():
                logger.info(
                    "Cancel requested at tool-call boundary (round %d) "
                    "— returning accumulated content as final response",
                    _round,
                )
                full_response = round_content
                yield CommandStreamEvent(kind="assistant_final", text=full_response)
                break

            if has_tool_calls:
                completed_calls = tool_call_acc.finish()
                logger.info(
                    "Executing %d tool call(s): %s",
                    len(completed_calls),
                    ", ".join(c["function"]["name"] for c in completed_calls),
                )
                for idx, call in enumerate(completed_calls):
                    if idx in emitted_tool_call_indices:
                        continue
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
                    result_tokens = _rough_tool_result_tokens(tool_result)
                    logger.info(
                        "Tool %s result: %d chars (~%d tokens) (preview: %s)",
                        fn_name, len(tool_result), result_tokens, tool_result[:200],
                    )

                    # Emit a subtext line with tool result info
                    info_parts = []
                    if fn_name == "read_file" and fn_args.get("path"):
                        info_parts.append(fn_args["path"])
                    elif fn_name == "search_file":
                        query = fn_args.get("query", "")
                        path = fn_args.get("path", "")
                        if query and path:
                            info_parts.append(f'"{query}" in {path}')
                        elif query:
                            info_parts.append(f'"{query}"')
                        elif path:
                            info_parts.append(path)
                    elif fn_name == "capture_context":
                        info_parts.append("screen capture")
                    if result_tokens > 0:
                        info_parts.append(f"~{result_tokens} tokens")
                    if info_parts:
                        info_line = f"  [{' · '.join(info_parts)}]\n"
                        round_content += info_line
                        yield CommandStreamEvent(
                            kind="assistant_delta",
                            text=info_line,
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

        # Add to ring buffer — only this turn's messages (from the user
        # utterance onward), preserving tool calls and results.
        # Strip reasoning content from assistant messages to save context.
        turn_messages = []
        for msg in messages[turn_start_idx:]:
            if msg["role"] == "assistant" and msg.get("content"):
                # Strip <think>...</think> blocks from assistant content
                import re
                cleaned = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()
                msg = {**msg, "content": cleaned or None}
            turn_messages.append(msg)
        self._history.append(turn_messages)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return full_response

    def clear_history(self) -> None:
        """Clear the conversation ring buffer."""
        self._history.clear()
