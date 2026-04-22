"""Command dispatch to the local command endpoint.

Sends voice command utterances to the local OpenAI-compatible command stack,
streams the response, and maintains a ring buffer of recent exchanges for
conversational context.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import logging
import os
import re
import uuid
from typing import Any, Callable, Generator, Literal

from pathlib import Path
import urllib.request
import urllib.error
from urllib.parse import urlparse

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


def _extract_reasoning_tokens(delta: dict[str, Any]) -> list[str]:
    """Flatten provider-specific reasoning shapes into displayable text chunks."""
    tokens: list[str] = []

    # OpenAI-compatible servers sometimes expose plaintext reasoning directly.
    for field in ("reasoning_content", "reasoning"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            tokens.append(value)

    # OpenRouter streams structured reasoning in delta.reasoning_details.
    reasoning_details = delta.get("reasoning_details")
    if isinstance(reasoning_details, list):
        for detail in reasoning_details:
            if not isinstance(detail, dict):
                continue
            summary = detail.get("summary")
            if isinstance(summary, str) and summary:
                tokens.append(summary)
            text = detail.get("text")
            if isinstance(text, str) and text:
                tokens.append(text)

    return tokens

_DEFAULT_COMMAND_URL = "http://localhost:8090"
_DEFAULT_COMMAND_MODEL = "qwen3p5-35B-A3B"
_DEFAULT_RING_BUFFER_SIZE = 20
_HISTORY_PATH = Path.home() / ".config" / "spoke" / "history.json"


def _rough_tool_result_tokens(text: str) -> int:
    """Approximate token count for a tool result."""
    return int(len(text.split()) * 1.3)

_SYSTEM_PROMPT = (
    "Environment: your working directory is ~/dev, the user's development root. "
    "File tool paths resolve relative to ~/dev, so you can use short paths like "
    "'epistaxis/projects/spoke/epistaxis.md' or 'spoke/spoke/command.py'. "
    "Key repos: spoke (this app), omlx (local model server), "
    "mlx-audio (TTS/ASR sidecar), epanorthosis (automated review), epistaxis "
    "(cross-session state and coordination), grapheus (inference traffic proxy).\n\n"
    "Epistaxis (~/dev/epistaxis/) is the user's durable off-repo state system. Layout:\n"
    "- projects/<repo>/epistaxis.md — per-repo status, lanes, scoped state\n"
    "- attractors/ — atomic units of work: stimulus, source, satisfaction condition\n"
    "- system/epistaxis.md — global system state\n"
    "- sylloge/ — compressed thread→artifact summaries\n"
    "- autopoiesis/ — self-generated concept genesis and traces\n"
    "- auxesis/ — growth/capability emergence traces\n"
    "- machines/ — per-machine context\n"
    "- reviews/ — review artifacts\n"
    "- metadosis/ — shared mutable artifacts between lanes\n"
    "When the user references epistaxis state, use file tools or run_epistaxis_ops.\n\n"
    "You have tools to resolve exact text and act on it:\n"
    "- capture_context: captures the frontmost window and returns OCR text "
    "blocks with refs. On vision-capable backends it can also attach a "
    "downscaled screenshot for direct visual inspection. Use when the user "
    'refers to something visible on screen (e.g. "read that", "what does this say", "read the tab title").\n'
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
    "You also have search_web: a bounded read-only public web search via "
    "Brave Search for lightweight fact lookup (up to 10 results).\n\n"
    "You also have launch_subagent, list_subagents, get_subagent_result, and "
    "cancel_subagent for operator-owned background jobs. Current support is "
    "kind='search' for bounded local file/code search. Subagents are "
    "asynchronous. After launch, do not spin on get_subagent_result in a "
    "tight loop. If a job is queued or running, continue the main conversation "
    "and check again later only when useful or when the user asks.\n\n"
    "You also have compact_history to reduce context size. Modes: "
    "drop_tool_results (strip tool call/result messages from the oldest N "
    "turns while keeping user and assistant text), summarize (replace the "
    "oldest N turns with your summary), and guided (return attractor-aware "
    "retention flags for the oldest N turns, then follow up with summarize "
    "using those flags as a safety net).\n\n"
    "You also have query_gmail: a bounded read-only Gmail query tool. Pass "
    "a Gmail search query string (same syntax as the Gmail search bar) and "
    "get back sender, subject, date, and snippet for up to 10 matches. Use "
    "it when the user asks about their email. Keep queries specific to limit "
    "results — combine filters like 'is:starred newer_than:3d' or "
    "'from:alice subject:invoice'.\n\n"
    "You also have run_terminal_command: a bounded local terminal command "
    "runner. Pass argv as a list of tokens, not raw shell text. Do not use "
    "shell syntax like pipes, redirects, chaining, glob expansion, env-var "
    "assignment, or shell interpolation. The tool may allow, deny, or "
    "require approval depending on the command family.\n\n"
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
    "read_aloud with literal:<exact text to speak>. Do not pretend read_aloud "
    "is limited to visible text. Use add_to_tray when the user "
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

    kind: Literal[
        "assistant_delta",
        "assistant_final",
        "tool_call",
        "thinking_delta",
        "approval_request",
    ]
    text: str = ""
    tool_name: str | None = None
    tool_arguments: str | None = None
    approval_request: dict[str, Any] | None = None


@dataclass(frozen=True)
class PendingToolApproval:
    """Paused tool-call turn waiting for a host-side user approval gesture."""

    utterance: str
    messages: list[dict[str, Any]]
    call: dict[str, Any]
    remaining_calls: list[dict[str, Any]]
    round_index: int
    tools: list[dict] | None
    turn_start_idx: int


class CommandClient:
    """Streaming chat client for voice commands via OMLX."""

    _SENTINEL = object()

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_history: int | None = None,
        history_path: Path | None | object = _SENTINEL,
        system_prompt: str | None = None,
    ):
        raw_url = (base_url or _DEFAULT_COMMAND_URL).rstrip("/")
        # Cloud OpenAI-compat endpoints (e.g. Gemini) include the version
        # prefix in the base URL already.  Detect this so we don't double it
        # when building /v1/models and /v1/chat/completions paths.
        path = urlparse(raw_url).path.rstrip("/")
        host = urlparse(raw_url).netloc.lower()
        self._url_has_version_prefix = any(
            seg.startswith("v") and seg[1:].replace("beta", "").isdigit()
            for seg in path.split("/") if seg
        )
        self._is_openrouter = "openrouter.ai" in host
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
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        # Ring buffer: list of message chains (each a list[dict]).
        # Each entry is the full sequence of messages for one turn:
        # [user, assistant, tool_result, assistant, ...] preserving
        # tool calls and results for multi-turn context.
        self._history_path = _HISTORY_PATH if history_path is self._SENTINEL else history_path
        self._history: list[list[dict]] = self._load_history()
        self._pending_tool_approval: PendingToolApproval | None = None

    def _load_history(self) -> list[list[dict]]:
        """Load persisted history from disk, or return empty list.

        Handles migration from old format (list of [user_str, assistant_str]
        pairs) to new format (list of message chains).
        """
        if self._history_path is None:
            return []
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            converted = []
            for entry in data:
                if not isinstance(entry, list) or not entry:
                    continue
                if isinstance(entry[0], str):
                    # Old format: ["user text", "assistant text"]
                    chain = [{"role": "user", "content": entry[0]}]
                    if len(entry) > 1 and entry[1]:
                        chain.append({"role": "assistant", "content": entry[1]})
                    converted.append(chain)
                elif isinstance(entry[0], dict):
                    # New format: list of message dicts
                    converted.append(entry)
            return converted
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return []

    def _save_history(self) -> None:
        """Persist current history to disk."""
        if self._history_path is None:
            return
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._history_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._history, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self._history_path)

    @property
    def history(self) -> list[tuple[str, str]]:
        """Backward-compatible pair view of the stored turn history."""
        pairs: list[tuple[str, str]] = []
        for chain in self._history:
            user_text = ""
            assistant_parts: list[str] = []
            for message in chain:
                role = message.get("role")
                content = message.get("content")
                if role == "user" and isinstance(content, str) and not user_text:
                    user_text = content
                elif role == "assistant" and isinstance(content, str) and content:
                    assistant_parts.append(content)
            pairs.append((user_text, "".join(assistant_parts)))
        return pairs

    def _normalized_history_turn(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize a turn before storing it in durable history."""
        turn_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("content"):
                cleaned = re.sub(
                    r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL
                ).strip()
                msg = {**msg, "content": cleaned or None}
            turn_messages.append(msg)
        return turn_messages

    def append_history_turn(self, messages: list[dict[str, Any]]) -> None:
        """Append one normalized turn to the bounded history ring."""
        self._history.append(self._normalized_history_turn(messages))
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._save_history()

    def append_history_pair(self, user_text: str, assistant_text: str) -> None:
        """Append a minimal user/assistant exchange to history."""
        self.append_history_turn([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])

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
        messages: list[dict] = [{"role": "system", "content": self._system_prompt}]
        for chain in self._history:
            messages.extend(chain)
        messages.append({"role": "user", "content": utterance})
        return messages

    def _supports_multimodal_tool_content(self) -> bool:
        """Whether the current backend is likely to accept image tool content."""
        model = self._model.lower()
        base_url = self._base_url.lower()
        return (
            "googleapis.com" in base_url
            or "gemini" in model
            or "gpt-4.1" in model
            or "gpt-4o" in model
        )

    def _normalize_tool_result(self, tool_result: Any) -> tuple[Any, str]:
        """Return (message_content, log_preview_text) for a tool result."""
        if isinstance(tool_result, dict) and "content" in tool_result:
            content = tool_result["content"]
            log_text = tool_result.get("log_text")
            if not isinstance(log_text, str):
                if isinstance(content, str):
                    log_text = content
                else:
                    log_text = json.dumps(content)
            return content, log_text
        if isinstance(tool_result, str):
            return tool_result, tool_result
        serialized = json.dumps(tool_result)
        return serialized, serialized

    def _tool_executor_supports_output_mode(
        self,
        tool_executor: Callable[..., Any],
    ) -> bool:
        """Whether tool_executor accepts the tool_output_mode kwarg."""
        return self._tool_executor_supports_kwarg(tool_executor, "tool_output_mode")

    def _tool_executor_supports_kwarg(
        self,
        tool_executor: Callable[..., Any],
        kwarg_name: str,
    ) -> bool:
        """Whether tool_executor accepts a specific keyword argument."""
        try:
            signature = inspect.signature(tool_executor)
        except (TypeError, ValueError):
            return True
        return any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            or param.name == kwarg_name
            for param in signature.parameters.values()
        )

    def _invoke_tool_call(
        self,
        call: dict[str, Any],
        *,
        tool_executor: Callable[..., Any],
        approval_granted: bool = False,
    ) -> Any:
        fn_name = call["function"]["name"]
        try:
            fn_args = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError:
            fn_args = {}

        logger.info("Executing tool %s with args: %s", fn_name, str(fn_args)[:200])
        tool_kwargs: dict[str, Any] = {
            "name": fn_name,
            "arguments": fn_args,
        }
        if self._tool_executor_supports_output_mode(tool_executor):
            tool_kwargs["tool_output_mode"] = (
                "multimodal"
                if self._supports_multimodal_tool_content()
                else "text"
            )
        if approval_granted and self._tool_executor_supports_kwarg(
            tool_executor, "approval_granted"
        ):
            tool_kwargs["approval_granted"] = True
        return tool_executor(**tool_kwargs)

    def _is_pending_approval_result(self, tool_result: Any) -> bool:
        parsed = self._tool_result_mapping(tool_result)
        return (
            isinstance(parsed, dict)
            and bool(parsed.get("pending_approval"))
            and isinstance(parsed.get("approval_request"), dict)
        )

    def _tool_result_mapping(self, tool_result: Any) -> dict[str, Any] | None:
        if isinstance(tool_result, dict):
            return tool_result
        if not isinstance(tool_result, str):
            return None
        try:
            parsed = json.loads(tool_result)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _execute_tool_calls(
        self,
        *,
        utterance: str,
        messages: list[dict[str, Any]],
        visible_response: str,
        completed_calls: list[dict[str, Any]],
        tool_executor: Callable[..., Any],
        round_index: int,
        tools: list[dict] | None,
        turn_start_idx: int,
        approval_granted_call_id: str | None = None,
    ) -> Generator[CommandStreamEvent, None, tuple[list[dict[str, Any]], str, bool]]:
        for idx, call in enumerate(completed_calls):
            fn_name = call["function"]["name"]
            try:
                fn_args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}
            tool_result = self._invoke_tool_call(
                call,
                tool_executor=tool_executor,
                approval_granted=(approval_granted_call_id == call["id"]),
            )
            if self._is_pending_approval_result(tool_result):
                approval_result = self._tool_result_mapping(tool_result) or {}
                approval_request = approval_result["approval_request"]
                self._pending_tool_approval = PendingToolApproval(
                    utterance=utterance,
                    messages=list(messages),
                    call=call,
                    remaining_calls=completed_calls[idx + 1:],
                    round_index=round_index,
                    tools=tools,
                    turn_start_idx=turn_start_idx,
                )
                yield CommandStreamEvent(
                    kind="approval_request",
                    text=approval_request.get("message", "Approval needed"),
                    approval_request=approval_request,
                )
                return messages, visible_response, True

            tool_content, tool_preview = self._normalize_tool_result(tool_result)
            result_tokens = _rough_tool_result_tokens(tool_preview)
            logger.info(
                "Tool %s result: %d chars (~%d tokens) (preview: %s)",
                fn_name,
                len(tool_preview),
                result_tokens,
                tool_preview[:200],
            )

            info_parts = []
            if fn_name == "read_file" and fn_args.get("file_path"):
                info_parts.append(fn_args["file_path"])
            elif fn_name == "search_file":
                pattern = fn_args.get("pattern", "")
                dir_path = fn_args.get("dir_path", "")
                if pattern and dir_path:
                    info_parts.append(f'"{pattern}" in {dir_path}')
                elif pattern:
                    info_parts.append(f'"{pattern}"')
                elif dir_path:
                    info_parts.append(dir_path)
            elif fn_name == "capture_context":
                info_parts.append("screen capture")
            if result_tokens > 0:
                info_parts.append(f"~{result_tokens} tokens")
            if info_parts:
                info_line = f"  [{' · '.join(info_parts)}]\n"
                visible_response += info_line
                yield CommandStreamEvent(
                    kind="assistant_delta",
                    text=info_line,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_content,
                }
            )

        return messages, visible_response, False

    def stream_command(
        self,
        utterance: str,
        *,
        tools: list[dict] | None = None,
        tool_executor: Callable[..., Any] | None = None,
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
        tool_executor: Callable[..., Any] | None = None,
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
        approval_request:
            Host-side approval is required before the exact pending tool call
            can be replayed and the assistant turn may continue.
        """
        self._pending_tool_approval = None
        messages = self._build_messages(utterance)
        # Track where this turn's messages start (after system + history)
        turn_start_idx = len(messages) - 1  # points to the user message
        full_response = ""
        visible_response = ""

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
            if self._enable_thinking and self._is_openrouter:
                body["reasoning"] = {"enabled": True}
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

                        # Reasoning tokens can arrive as plaintext fields
                        # or structured reasoning_details, depending on provider.
                        for reasoning_token in _extract_reasoning_tokens(delta):
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
                                candidate = thinking_tag_buf.lstrip()
                                if candidate.startswith("<think>"):
                                    thinking_state = "thinking"
                                    text_to_process = candidate[len("<think>"):]
                                    thinking_tag_buf = ""
                                    if not text_to_process:
                                        continue
                                    # Fall through to "thinking" handler below
                                elif not candidate:
                                    continue
                                elif len(candidate) < len("<think>") and "<think>".startswith(candidate):
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
                            visible_response += text_to_process
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
                                visible_response += indicator
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
                    original_round_content = round_content
                    cleaned_text, xml_calls = xml_result
                    logger.info(
                        "Extracted %d XML tool call(s) from content stream: %s",
                        len(xml_calls),
                        ", ".join(c["function"]["name"] for c in xml_calls),
                    )
                    round_content = cleaned_text
                    if (
                        original_round_content
                        and visible_response.endswith(original_round_content)
                    ):
                        visible_response = (
                            visible_response[: -len(original_round_content)] + cleaned_text
                        )
                    for i, xc in enumerate(xml_calls):
                        tool_call_acc._calls[i] = xc
                        indicator = f"\n[calling {xc['function']['name']}…]\n"
                        visible_response += indicator
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
                full_response = visible_response or round_content
                messages.append({"role": "assistant", "content": full_response or None})
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

                messages, visible_response, paused_for_approval = yield from self._execute_tool_calls(
                    utterance=utterance,
                    messages=messages,
                    visible_response=visible_response,
                    completed_calls=completed_calls,
                    tool_executor=tool_executor,
                    round_index=_round,
                    tools=tools,
                    turn_start_idx=turn_start_idx,
                )
                if paused_for_approval:
                    return full_response

                logger.info(
                    "Sending tool results back to model (round %d, %d messages total)",
                    _round + 1, len(messages),
                )
                # Continue the loop — next round sends tool results
                continue

            # No tool calls — this round's content is the final response
            full_response = round_content
            messages.append({"role": "assistant", "content": full_response or None})
            yield CommandStreamEvent(kind="assistant_final", text=full_response)
            break

        # Add to ring buffer — only this turn's messages (from the user
        # utterance onward), preserving tool calls and results.
        self.append_history_turn(messages[turn_start_idx:])

        return full_response

    def approve_pending_tool_call(
        self,
        *,
        tool_executor: Callable[..., Any],
        cancel_check: Callable[[], bool] | None = None,
    ) -> Generator[CommandStreamEvent, None, str]:
        """Resume a paused tool-call turn after the user approved the pending call."""
        pending = self._pending_tool_approval
        if pending is None:
            raise RuntimeError("No pending tool approval to resume")

        self._pending_tool_approval = None
        messages = list(pending.messages)
        completed_calls = [pending.call, *pending.remaining_calls]
        messages, visible_response, paused_for_approval = yield from self._execute_tool_calls(
            utterance=pending.utterance,
            messages=messages,
            visible_response="",
            completed_calls=completed_calls,
            tool_executor=tool_executor,
            round_index=pending.round_index,
            tools=pending.tools,
            turn_start_idx=pending.turn_start_idx,
            approval_granted_call_id=pending.call["id"],
        )
        if paused_for_approval:
            return ""

        full_response = ""
        max_tool_rounds = 20
        for _round in range(pending.round_index + 1, max_tool_rounds + 1):
            body: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "stream": True,
            }
            if self._enable_thinking and self._is_openrouter:
                body["reasoning"] = {"enabled": True}
            if not self._enable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if pending.tools:
                body["tools"] = pending.tools

            payload = json.dumps(body).encode()
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            url = (
                f"{self._base_url}/chat/completions"
                if self._url_has_version_prefix
                else f"{self._base_url}/v1/chat/completions"
            )
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

            from spoke.tool_dispatch import ToolCallAccumulator

            tool_call_acc = ToolCallAccumulator()
            emitted_tool_call_indices: set[int] = set()
            finish_reason = None
            round_content = ""
            thinking_state = "detect"
            thinking_tag_buf = ""

            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    if cancel_check is not None and cancel_check():
                        resp.close()
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(": ") or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    for reasoning_token in _extract_reasoning_tokens(delta):
                        yield CommandStreamEvent(
                            kind="thinking_delta",
                            text=reasoning_token,
                        )

                    token = delta.get("content")
                    if token is not None:
                        text_to_process = token
                        if thinking_state == "detect":
                            thinking_tag_buf += text_to_process
                            if thinking_tag_buf.startswith("<think>"):
                                thinking_state = "thinking"
                                text_to_process = thinking_tag_buf[len("<think>"):]
                                thinking_tag_buf = ""
                                if not text_to_process:
                                    continue
                            elif len(thinking_tag_buf) < len("<think>") and "<think>".startswith(thinking_tag_buf):
                                continue
                            else:
                                thinking_state = "content"
                                text_to_process = thinking_tag_buf
                                thinking_tag_buf = ""

                        if thinking_state == "thinking":
                            combined = thinking_tag_buf + text_to_process
                            thinking_tag_buf = ""
                            close_idx = combined.find("</think>")
                            if close_idx >= 0:
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
                                for k in range(min(len("</think>"), len(combined)), 0, -1):
                                    if combined.endswith("</think>"[:k]):
                                        thinking_tag_buf = combined[-k:]
                                        combined = combined[:-k]
                                        break
                                if combined:
                                    yield CommandStreamEvent(kind="thinking_delta", text=combined)
                                continue

                        round_content += text_to_process
                        visible_response += text_to_process
                        yield CommandStreamEvent(kind="assistant_delta", text=text_to_process)

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
                            visible_response += indicator
                            yield CommandStreamEvent(kind="assistant_delta", text=indicator)
                            emitted_tool_call_indices.add(idx)
                            yield CommandStreamEvent(
                                kind="tool_call",
                                tool_name=tool_name,
                                tool_arguments=function_delta.get("arguments"),
                            )

            has_tool_calls = tool_call_acc.has_calls and tool_executor is not None
            if has_tool_calls and finish_reason != "tool_calls":
                logger.warning(
                    "Model emitted tool call deltas but finish_reason=%r "
                    "(expected 'tool_calls') — executing anyway",
                    finish_reason,
                )
            if has_tool_calls and cancel_check is not None and cancel_check():
                full_response = visible_response or round_content
                messages.append({"role": "assistant", "content": full_response or None})
                yield CommandStreamEvent(kind="assistant_final", text=full_response)
                break
            if has_tool_calls:
                completed_calls = tool_call_acc.finish()
                for idx, call in enumerate(completed_calls):
                    if idx in emitted_tool_call_indices:
                        continue
                    yield CommandStreamEvent(
                        kind="tool_call",
                        tool_name=call["function"]["name"],
                        tool_arguments=call["function"]["arguments"],
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": round_content or None,
                        "tool_calls": completed_calls,
                    }
                )
                messages, visible_response, paused_for_approval = yield from self._execute_tool_calls(
                    utterance=pending.utterance,
                    messages=messages,
                    visible_response=visible_response,
                    completed_calls=completed_calls,
                    tool_executor=tool_executor,
                    round_index=_round,
                    tools=pending.tools,
                    turn_start_idx=pending.turn_start_idx,
                )
                if paused_for_approval:
                    return full_response
                continue

            full_response = round_content
            messages.append({"role": "assistant", "content": full_response or None})
            yield CommandStreamEvent(kind="assistant_final", text=full_response)
            break

        self.append_history_turn(messages[pending.turn_start_idx:])
        return full_response

    def cancel_pending_tool_call(self) -> None:
        """Discard a paused tool-call turn without mutating history."""
        self._pending_tool_approval = None

    def clear_history(self) -> None:
        """Clear the conversation ring buffer."""
        self._history.clear()
        self._save_history()
