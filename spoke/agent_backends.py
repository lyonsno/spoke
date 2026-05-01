"""Local-auth coding-agent backend sessions for the operator shell."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent_shell_identity import AgentShellIdentity, resolve_agent_shell_identity
from .agent_thread_cards import (
    build_agent_thread_card,
    extract_thread_waypoints_from_text,
)


_ALLOWED_PROVIDERS = {"codex", "claude-code", "gemini-cli"}
_BILLING_CREDENTIAL_ENV = (
    "OPENAI_" + "API_KEY",
    "CODEX_" + "API_KEY",
    "ANTHROPIC_" + "API_KEY",
    "GEMINI_" + "API_KEY",
    "GOOGLE_" + "API_KEY",
    "GOOGLE_GENAI_" + "API_KEY",
)
_CODEX_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/codex",
    "/usr/local/bin/codex",
    os.path.expanduser("~/.local/bin/codex"),
)
_CLAUDE_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
    os.path.expanduser("~/.local/bin/claude"),
)
_GEMINI_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/gemini",
    "/usr/local/bin/gemini",
    os.path.expanduser("~/.local/bin/gemini"),
)
_GEMINI_OPERATOR_PROMPT_PREFIX = """You are running inside Spoke's compact operator shell.
Answer the user's latest instruction directly and concisely.
Do not print an implementation plan, checklist, file-by-file itinerary, or "I will..." reconnaissance log unless the user explicitly asks for one.
If tools are useful, use them and report the result rather than narrating every intended step first.

User instruction:
"""
_CODEX_SESSION_LOG_SCAN_LIMIT = 250_000


class AgentBackendUnavailable(RuntimeError):
    """Raised when a local agent backend is not installed or not ready."""


@dataclass(frozen=True)
class AgentBackendEvent:
    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBackendRunResult:
    provider: str
    session_id: str | None
    final_response: str
    events: tuple[AgentBackendEvent, ...] = ()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _subscription_only_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in _BILLING_CREDENTIAL_ENV:
        env.pop(name, None)
    return env


def _executable_file(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _resolve_codex_path(env: dict[str, str] | None = None) -> str | None:
    env = env or os.environ
    override = env.get("SPOKE_CODEX_PATH")
    if isinstance(override, str) and override.strip():
        candidate = override.strip()
        if _executable_file(candidate):
            return candidate

    path_value = env.get("PATH")
    found = (
        shutil.which("codex", path=path_value)
        if isinstance(path_value, str) and path_value
        else shutil.which("codex")
    )
    if found:
        return found

    for candidate in _CODEX_CANDIDATE_PATHS:
        if _executable_file(candidate):
            return candidate

    try:
        result = subprocess.run(
            ["/bin/zsh", "-lc", "command -v codex"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = (result.stdout or "").strip().splitlines()[0:1]
    if candidate and _executable_file(candidate[0]):
        return candidate[0]
    return None


def _resolve_claude_path(env: dict[str, str] | None = None) -> str | None:
    env = env or os.environ
    override = env.get("SPOKE_CLAUDE_CODE_PATH")
    if isinstance(override, str) and override.strip():
        candidate = override.strip()
        if _executable_file(candidate):
            return candidate

    path_value = env.get("PATH")
    found = (
        shutil.which("claude", path=path_value)
        if isinstance(path_value, str) and path_value
        else shutil.which("claude")
    )
    if found:
        return found

    for candidate in _CLAUDE_CANDIDATE_PATHS:
        if _executable_file(candidate):
            return candidate

    try:
        result = subprocess.run(
            ["/bin/zsh", "-lc", "command -v claude"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = (result.stdout or "").strip().splitlines()[0:1]
    if candidate and _executable_file(candidate[0]):
        return candidate[0]
    return None


def _resolve_gemini_path(env: dict[str, str] | None = None) -> str | None:
    env = env or os.environ
    override = env.get("SPOKE_GEMINI_CLI_PATH")
    if isinstance(override, str) and override.strip():
        candidate = override.strip()
        if _executable_file(candidate):
            return candidate

    path_value = env.get("PATH")
    found = (
        shutil.which("gemini", path=path_value)
        if isinstance(path_value, str) and path_value
        else shutil.which("gemini")
    )
    if found:
        return found

    for candidate in _GEMINI_CANDIDATE_PATHS:
        if _executable_file(candidate):
            return candidate

    try:
        result = subprocess.run(
            ["/bin/zsh", "-lc", "command -v gemini"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = (result.stdout or "").strip().splitlines()[0:1]
    if candidate and _executable_file(candidate[0]):
        return candidate[0]
    return None


def _codex_login_status(codex_path: str, env: dict[str, str]) -> str:
    try:
        result = subprocess.run(
            [codex_path, "login", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentBackendUnavailable(f"Codex login status failed: {exc}") from exc
    return result.stdout or ""


def _require_codex_subscription_login(codex_path: str, env: dict[str, str]) -> None:
    status = _codex_login_status(codex_path, env)
    if "ChatGPT" in status:
        return
    raise AgentBackendUnavailable(
        "Codex CLI is not logged in with ChatGPT subscription auth; "
        "billing-backed Codex credentials are disabled for Spoke Agent Shell."
    )


def _claude_auth_status(claude_path: str, env: dict[str, str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [claude_path, "auth", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentBackendUnavailable(f"Claude Code auth status failed: {exc}") from exc
    output = result.stdout or ""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        raise AgentBackendUnavailable(
            "Claude Code auth status did not return machine-readable subscription state"
        ) from exc
    if not isinstance(parsed, dict):
        raise AgentBackendUnavailable(
            "Claude Code auth status did not return a subscription state object"
        )
    return parsed


def _require_claude_subscription_login(claude_path: str, env: dict[str, str]) -> None:
    status = _claude_auth_status(claude_path, env)
    if (
        status.get("loggedIn") is True
        and status.get("authMethod") == "claude.ai"
        and status.get("apiProvider") == "firstParty"
        and isinstance(status.get("subscriptionType"), str)
        and bool(status.get("subscriptionType"))
    ):
        return
    raise AgentBackendUnavailable(
        "Claude Code CLI is not logged in with claude.ai subscription auth; "
        "billing-backed Anthropic credentials are disabled for Spoke Agent Shell."
    )


def _codex_command(
    *,
    codex_path: str,
    prompt: str,
    cwd: str,
    resume_id: str | None,
) -> list[str]:
    if resume_id:
        return [codex_path, "exec", "resume", "--json", resume_id, prompt]
    return [codex_path, "exec", "--json", "--cd", cwd, prompt]


def _claude_command(
    *,
    claude_path: str,
    resume_id: str | None,
) -> list[str]:
    command = [
        claude_path,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "dontAsk",
    ]
    if resume_id:
        command.extend(["--resume", resume_id])
    return command


def _gemini_command(
    *,
    gemini_path: str,
    prompt: str,
    resume_id: str | None,
) -> list[str]:
    operator_prompt = f"{_GEMINI_OPERATOR_PROMPT_PREFIX}{prompt}"
    command = [
        gemini_path,
        "-p",
        operator_prompt,
        "--output-format",
        "stream-json",
        "--approval-mode",
        "yolo",
    ]
    if resume_id:
        command.extend(["--resume", resume_id])
    return command


def _event_from_codex_item(item: dict[str, Any]) -> AgentBackendEvent | None:
    item_type = item.get("type")
    if not isinstance(item_type, str) or not item_type:
        return None
    text = ""
    if item_type == "agent_message":
        raw = item.get("text")
        text = raw if isinstance(raw, str) else ""
    elif item_type == "reasoning":
        raw = item.get("text")
        text = raw if isinstance(raw, str) else ""
    elif item_type == "command_execution":
        command = item.get("command")
        output = item.get("aggregated_output")
        text = command if isinstance(command, str) else ""
        if isinstance(output, str) and output:
            text = f"{text}\n{output}" if text else output
    elif item_type == "file_change":
        changes = item.get("changes")
        if isinstance(changes, list):
            paths = [
                change.get("path")
                for change in changes
                if isinstance(change, dict) and isinstance(change.get("path"), str)
            ]
            text = ", ".join(paths)
    elif item_type == "error":
        raw = item.get("message")
        text = raw if isinstance(raw, str) else ""
    return AgentBackendEvent(kind=item_type, text=text, data=item)


def _text_from_codex_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _percent_for_window(rate_limits: dict[str, Any], window_minutes: int) -> float | None:
    for value in rate_limits.values():
        if not isinstance(value, dict):
            continue
        if value.get("window_minutes") != window_minutes:
            continue
        percent = value.get("used_percent")
        if isinstance(percent, (int, float)):
            return float(percent)
    return None


def _events_from_codex_stream_event(event: dict[str, Any]) -> list[AgentBackendEvent]:
    event_type = event.get("type")
    payload = event.get("payload")
    if event_type in {"session_meta", "turn_context"} and isinstance(payload, dict):
        data: dict[str, Any] = {}
        session_id = payload.get("id")
        if isinstance(session_id, str) and session_id:
            data["provider_session_id"] = session_id
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd:
            data["cwd"] = cwd
        model = payload.get("model")
        if isinstance(model, str) and model:
            data["model"] = model
        cli_version = payload.get("cli_version")
        if isinstance(cli_version, str) and cli_version:
            data["cli_version"] = cli_version
        model_provider = payload.get("model_provider")
        if isinstance(model_provider, str) and model_provider:
            data["model_provider"] = model_provider
        if data:
            return [
                AgentBackendEvent(
                    kind="session_metadata",
                    text=data.get("cwd", ""),
                    data=data,
                )
            ]
        return []

    if event_type == "event_msg" and isinstance(payload, dict):
        payload_type = payload.get("type")
        if payload_type == "token_count":
            rate_limits = payload.get("rate_limits")
            if not isinstance(rate_limits, dict):
                return []
            data: dict[str, Any] = {}
            five_hour = _percent_for_window(rate_limits, 300)
            seven_day = _percent_for_window(rate_limits, 10080)
            if five_hour is not None:
                data["five_hour_percent"] = five_hour
            if seven_day is not None:
                data["seven_day_percent"] = seven_day
            plan_type = rate_limits.get("plan_type")
            if isinstance(plan_type, str) and plan_type:
                data["plan_type"] = plan_type
            limit_id = rate_limits.get("limit_id")
            if isinstance(limit_id, str) and limit_id:
                data["limit_id"] = limit_id
            if data:
                return [AgentBackendEvent(kind="usage_limits", data=data)]
            return []

    if event_type == "response_item" and isinstance(payload, dict):
        item_type = payload.get("type")
        role = payload.get("role")
        if item_type == "message" and role == "assistant":
            text = _text_from_codex_response_content(payload.get("content"))
            if text:
                data = {
                    "id": payload.get("id", ""),
                    "type": "agent_message",
                    "text": text,
                }
                return [AgentBackendEvent(kind="agent_message", text=text, data=data)]
    return []


def _text_from_claude_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _events_from_claude_stream_event(event: dict[str, Any]) -> list[AgentBackendEvent]:
    event_type = event.get("type")
    if event_type == "system" and event.get("subtype") == "init":
        data: dict[str, Any] = {}
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            data["provider_session_id"] = session_id
        cwd = event.get("cwd")
        if isinstance(cwd, str) and cwd:
            data["cwd"] = cwd
        model = event.get("model")
        if isinstance(model, str) and model:
            data["model"] = model
        cli_version = event.get("claude_code_version")
        if isinstance(cli_version, str) and cli_version:
            data["cli_version"] = cli_version
        credential_source = event.get("apiKeySource")
        if isinstance(credential_source, str) and credential_source:
            data["credential_source"] = credential_source
        if data:
            return [
                AgentBackendEvent(
                    kind="session_metadata",
                    text=data.get("cwd", ""),
                    data=data,
                )
            ]
        return []

    if event_type == "assistant":
        message = event.get("message")
        if not isinstance(message, dict):
            return []
        text = _text_from_claude_content(message.get("content"))
        if not text:
            return []
        return [
            AgentBackendEvent(
                kind="agent_message",
                text=text,
                data={"type": "agent_message", "text": text},
            )
        ]

    if event_type == "rate_limit_event":
        info = event.get("rate_limit_info")
        if not isinstance(info, dict):
            return []
        data: dict[str, Any] = {}
        rate_limit_type = info.get("rateLimitType")
        if isinstance(rate_limit_type, str) and rate_limit_type:
            data["rate_limit_type"] = rate_limit_type
        status = info.get("status")
        if isinstance(status, str) and status:
            data["status"] = status
        if isinstance(info.get("isUsingOverage"), bool):
            data["is_using_overage"] = info["isUsingOverage"]
        if data:
            return [AgentBackendEvent(kind="usage_limits", data=data)]
    return []


def _events_from_gemini_stream_event(
    event: dict[str, Any],
    *,
    cwd: str,
) -> list[AgentBackendEvent]:
    event_type = event.get("type")
    if event_type == "init":
        data: dict[str, Any] = {"cwd": cwd}
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            data["provider_session_id"] = session_id
        model = event.get("model")
        if isinstance(model, str) and model:
            data["model"] = model
        return [
            AgentBackendEvent(
                kind="session_metadata",
                text=cwd,
                data=data,
            )
        ]

    if event_type == "message" and event.get("role") == "assistant":
        content = event.get("content")
        if not isinstance(content, str) or not content:
            return []
        return [
            AgentBackendEvent(
                kind="agent_message",
                text=content,
                data={
                    "type": "agent_message",
                    "text": content,
                    "delta": bool(event.get("delta")),
                },
            )
        ]

    if event_type == "tool_use":
        tool_name = event.get("tool_name")
        tool_name = tool_name if isinstance(tool_name, str) else ""
        parameters = event.get("parameters")
        command = ""
        if isinstance(parameters, dict):
            raw_command = parameters.get("command")
            if isinstance(raw_command, str):
                command = raw_command
        text = tool_name
        if command:
            text = f"{tool_name}: {command}" if tool_name else command
        return [
            AgentBackendEvent(
                kind="tool_use",
                text=text,
                data={
                    "tool_name": tool_name,
                    "tool_id": event.get("tool_id", ""),
                    "parameters": parameters if isinstance(parameters, dict) else {},
                },
            )
        ]

    if event_type == "tool_result":
        status = event.get("status")
        text = status if isinstance(status, str) else ""
        return [
            AgentBackendEvent(
                kind="tool_result",
                text=text,
                data={
                    "tool_id": event.get("tool_id", ""),
                    "status": text,
                },
            )
        ]

    if event_type == "result":
        stats = event.get("stats")
        data: dict[str, Any] = {}
        if isinstance(stats, dict):
            for key in (
                "duration_ms",
                "tool_calls",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "cached",
                "input",
            ):
                value = stats.get(key)
                if isinstance(value, (int, float)):
                    data[key] = value
            models = stats.get("models")
            if isinstance(models, dict):
                data["models"] = models
        status = event.get("status")
        if isinstance(status, str) and status:
            data["status"] = status
        if data:
            return [AgentBackendEvent(kind="usage_limits", data=data)]
    return []


def _thread_waypoint_events_from_text(
    text: str,
    *,
    sequence: int = 0,
    source: str,
) -> list[AgentBackendEvent]:
    return [
        AgentBackendEvent(
            kind="thread_waypoint",
            text=waypoint.text,
            data=waypoint.to_event_data(),
        )
        for waypoint in extract_thread_waypoints_from_text(
            text,
            sequence=sequence,
            source=source,
        )
    ]


def _codex_sessions_root(
    env: dict[str, str] | None = None,
    codex_home: Path | None = None,
) -> Path:
    if codex_home is not None:
        return codex_home / "sessions"
    env = env or os.environ
    configured = env.get("CODEX_HOME")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser() / "sessions"
    return Path.home() / ".codex" / "sessions"


def _codex_session_log_path(
    session_id: str,
    *,
    env: dict[str, str] | None = None,
    codex_home: Path | None = None,
) -> Path | None:
    session_id = session_id.strip()
    if not session_id:
        return None
    root = _codex_sessions_root(env=env, codex_home=codex_home)
    if not root.exists():
        return None
    matches = list(root.rglob(f"*{session_id}.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _events_from_codex_session_log(
    session_id: str,
    *,
    env: dict[str, str] | None = None,
    codex_home: Path | None = None,
) -> list[AgentBackendEvent]:
    path = _codex_session_log_path(session_id, env=env, codex_home=codex_home)
    if path is None:
        return []
    session_meta_event: AgentBackendEvent | None = None
    turn_context_event: AgentBackendEvent | None = None
    usage_event: AgentBackendEvent | None = None
    waypoint_events: list[AgentBackendEvent] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= _CODEX_SESSION_LOG_SCAN_LIMIT:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                parsed = _events_from_codex_stream_event(event)
                if not parsed:
                    continue
                event_type = event.get("type")
                if event_type == "session_meta" and session_meta_event is None:
                    session_meta_event = parsed[0]
                elif event_type == "turn_context":
                    turn_context_event = parsed[0]
                elif event_type == "event_msg":
                    usage_event = parsed[0]
                for parsed_event in parsed:
                    if parsed_event.kind == "agent_message" and parsed_event.text:
                        waypoint_events.extend(
                            _thread_waypoint_events_from_text(
                                parsed_event.text,
                                sequence=index,
                                source="codex-log",
                            )
                        )
    except OSError:
        return []
    return [
        event
        for event in (session_meta_event, turn_context_event, usage_event)
        if event is not None
    ] + waypoint_events


def _append_event(
    events: list[AgentBackendEvent],
    event: AgentBackendEvent,
    event_sink: Callable[[AgentBackendEvent], None] | None,
) -> None:
    events.append(event)
    if event_sink is not None:
        event_sink(event)


def _emit_process_started(
    process: subprocess.Popen,
    event_sink: Callable[[AgentBackendEvent], None] | None,
) -> None:
    if event_sink is None:
        return
    data: dict[str, Any] = {"pid": process.pid}
    try:
        data["pgid"] = os.getpgid(process.pid)
    except OSError:
        data["pgid"] = process.pid
    event_sink(AgentBackendEvent(kind="_process_started", data=data))


def _agent_shell_identity_event(
    *,
    provider: str,
    provider_session_id: str | None,
    cwd: str,
    final_response: str,
    events: list[AgentBackendEvent],
    identity_resolver: Callable[..., AgentShellIdentity | None] = resolve_agent_shell_identity,
) -> AgentBackendEvent | None:
    transcript_parts = [final_response]
    transcript_parts.extend(event.text for event in events if event.text)
    identity = identity_resolver(
        provider=provider,
        provider_session_id=provider_session_id,
        cwd=cwd,
        transcript_text="\n".join(part for part in transcript_parts if part),
    )
    if identity is None:
        return None
    data = {
        "name": identity.topos_name,
        "source": identity.source,
        "confidence": identity.confidence,
    }
    if provider_session_id:
        data["provider_session_id"] = provider_session_id
    if cwd:
        data["cwd"] = cwd
    return AgentBackendEvent(
        kind="topos_identity",
        text=identity.topos_name,
        data=data,
    )


def _run_codex_cli(
    *,
    prompt: str,
    cwd: str,
    resume_id: str | None,
    cancel_check: Callable[[], bool] | None,
    event_sink: Callable[[AgentBackendEvent], None] | None = None,
) -> AgentBackendRunResult:
    env = _subscription_only_env()
    codex_path = _resolve_codex_path(env)
    if not codex_path:
        raise AgentBackendUnavailable("Codex CLI is not installed or not on PATH")
    _require_codex_subscription_login(codex_path, env)

    command = _codex_command(
        codex_path=codex_path,
        prompt=prompt,
        cwd=cwd,
        resume_id=resume_id,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        raise AgentBackendUnavailable(f"Codex CLI failed to start: {exc}") from exc
    _emit_process_started(process, event_sink)

    provider_session_id = resume_id
    final_response = ""
    events: list[AgentBackendEvent] = []
    stream_error = ""
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if cancel_check is not None and cancel_check():
                process.terminate()
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                _append_event(
                    events,
                    AgentBackendEvent(kind="raw_output", text=stripped),
                    event_sink,
                )
                continue
            event_type = event.get("type")
            if event_type == "thread.started":
                thread_id = event.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    provider_session_id = thread_id
                    _append_event(
                        events,
                        AgentBackendEvent(
                            kind="thread_started",
                            text=thread_id,
                            data=event,
                        ),
                        event_sink,
                    )
            elif event_type in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if isinstance(item, dict):
                    backend_event = _event_from_codex_item(item)
                    if backend_event is not None:
                        _append_event(events, backend_event, event_sink)
                    if event_type == "item.completed" and item.get("type") == "agent_message":
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            final_response = text
            elif event_type == "turn.failed":
                error = event.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str):
                        stream_error = message
            elif event_type == "error":
                message = event.get("message")
                if isinstance(message, str):
                    stream_error = message
            for backend_event in _events_from_codex_stream_event(event):
                session_id = backend_event.data.get("provider_session_id")
                if isinstance(session_id, str) and session_id:
                    provider_session_id = session_id
                _append_event(events, backend_event, event_sink)
                if backend_event.kind == "agent_message" and backend_event.text:
                    final_response = backend_event.text
    finally:
        if process.stdout is not None:
            process.stdout.close()

    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return_code = process.wait()
    if cancel_check is not None and cancel_check():
        return AgentBackendRunResult(
            provider="codex",
            session_id=provider_session_id,
            final_response=final_response,
            events=tuple(events),
        )
    if return_code != 0:
        detail = stream_error or stderr.strip() or f"exit status {return_code}"
        raise RuntimeError(f"Codex CLI failed: {detail}")
    if provider_session_id:
        for backend_event in _events_from_codex_session_log(provider_session_id, env=env):
            _append_event(events, backend_event, event_sink)
    identity_event = _agent_shell_identity_event(
        provider="codex",
        provider_session_id=provider_session_id,
        cwd=cwd,
        final_response=final_response,
        events=events,
    )
    if identity_event is not None:
        _append_event(events, identity_event, event_sink)
    return AgentBackendRunResult(
        provider="codex",
        session_id=provider_session_id,
        final_response=final_response,
        events=tuple(events),
    )


def _run_claude_cli(
    *,
    prompt: str,
    cwd: str,
    resume_id: str | None,
    cancel_check: Callable[[], bool] | None,
    event_sink: Callable[[AgentBackendEvent], None] | None = None,
) -> AgentBackendRunResult:
    env = _subscription_only_env()
    claude_path = _resolve_claude_path(env)
    if not claude_path:
        raise AgentBackendUnavailable("Claude Code CLI is not installed or not on PATH")
    _require_claude_subscription_login(claude_path, env)

    command = _claude_command(claude_path=claude_path, resume_id=resume_id)
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        raise AgentBackendUnavailable(f"Claude Code CLI failed to start: {exc}") from exc
    _emit_process_started(process, event_sink)

    provider_session_id = resume_id
    final_response = ""
    events: list[AgentBackendEvent] = []
    stream_error = ""
    if process.stdin is not None:
        process.stdin.write(prompt)
        if not prompt.endswith("\n"):
            process.stdin.write("\n")
        process.stdin.close()

    assert process.stdout is not None
    try:
        for line in process.stdout:
            if cancel_check is not None and cancel_check():
                process.terminate()
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                _append_event(
                    events,
                    AgentBackendEvent(kind="raw_output", text=stripped),
                    event_sink,
                )
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            session_id = event.get("session_id")
            if isinstance(session_id, str) and session_id:
                provider_session_id = session_id
            if event_type == "result":
                result_text = event.get("result")
                if isinstance(result_text, str) and result_text:
                    final_response = result_text
                if event.get("is_error") is True:
                    stream_error = final_response or "Claude Code returned an error"
            for backend_event in _events_from_claude_stream_event(event):
                event_session_id = backend_event.data.get("provider_session_id")
                if isinstance(event_session_id, str) and event_session_id:
                    provider_session_id = event_session_id
                _append_event(events, backend_event, event_sink)
                if backend_event.kind == "agent_message" and backend_event.text:
                    final_response = backend_event.text
    finally:
        if process.stdout is not None:
            process.stdout.close()

    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return_code = process.wait()
    if cancel_check is not None and cancel_check():
        return AgentBackendRunResult(
            provider="claude-code",
            session_id=provider_session_id,
            final_response=final_response,
            events=tuple(events),
        )
    if return_code != 0 or stream_error:
        detail = stream_error or stderr.strip() or f"exit status {return_code}"
        raise RuntimeError(f"Claude Code CLI failed: {detail}")
    identity_event = _agent_shell_identity_event(
        provider="claude-code",
        provider_session_id=provider_session_id,
        cwd=cwd,
        final_response=final_response,
        events=events,
    )
    if identity_event is not None:
        _append_event(events, identity_event, event_sink)
    return AgentBackendRunResult(
        provider="claude-code",
        session_id=provider_session_id,
        final_response=final_response,
        events=tuple(events),
    )


def _run_gemini_cli(
    *,
    prompt: str,
    cwd: str,
    resume_id: str | None,
    cancel_check: Callable[[], bool] | None,
    event_sink: Callable[[AgentBackendEvent], None] | None = None,
) -> AgentBackendRunResult:
    env = _subscription_only_env()
    gemini_path = _resolve_gemini_path(env)
    if not gemini_path:
        raise AgentBackendUnavailable("Gemini CLI is not installed or not on PATH")

    command = _gemini_command(
        gemini_path=gemini_path,
        prompt=prompt,
        resume_id=resume_id,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        raise AgentBackendUnavailable(f"Gemini CLI failed to start: {exc}") from exc
    _emit_process_started(process, event_sink)

    provider_session_id = resume_id
    final_response_parts: list[str] = []
    events: list[AgentBackendEvent] = []
    stream_error = ""
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if cancel_check is not None and cancel_check():
                process.terminate()
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                _append_event(
                    events,
                    AgentBackendEvent(kind="raw_output", text=stripped),
                    event_sink,
                )
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "init":
                session_id = event.get("session_id")
                if isinstance(session_id, str) and session_id:
                    provider_session_id = session_id
            elif event_type == "message" and event.get("role") == "assistant":
                content = event.get("content")
                if isinstance(content, str) and content:
                    final_response_parts.append(content)
            elif event_type == "result":
                if event.get("status") not in {None, "success"}:
                    stream_error = str(event.get("status") or "Gemini CLI returned an error")
            for backend_event in _events_from_gemini_stream_event(event, cwd=cwd):
                event_session_id = backend_event.data.get("provider_session_id")
                if isinstance(event_session_id, str) and event_session_id:
                    provider_session_id = event_session_id
                _append_event(events, backend_event, event_sink)
    finally:
        if process.stdout is not None:
            process.stdout.close()

    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return_code = process.wait()
    final_response = "".join(final_response_parts).strip()
    if cancel_check is not None and cancel_check():
        return AgentBackendRunResult(
            provider="gemini-cli",
            session_id=provider_session_id,
            final_response=final_response,
            events=tuple(events),
        )
    if return_code != 0 or stream_error:
        detail = stream_error or stderr.strip() or f"exit status {return_code}"
        raise RuntimeError(f"Gemini CLI failed: {detail}")
    identity_event = _agent_shell_identity_event(
        provider="gemini-cli",
        provider_session_id=provider_session_id,
        cwd=cwd,
        final_response=final_response,
        events=events,
    )
    if identity_event is not None:
        _append_event(events, identity_event, event_sink)
    return AgentBackendRunResult(
        provider="gemini-cli",
        session_id=provider_session_id,
        final_response=final_response,
        events=tuple(events),
    )


def run_agent_backend_session(
    provider: str,
    prompt: str,
    cwd: str,
    resume_id: str | None,
    cancel_check: Callable[[], bool] | None = None,
    event_sink: Callable[[AgentBackendEvent], None] | None = None,
) -> AgentBackendRunResult:
    provider = provider.strip().lower()
    if provider == "codex":
        return _run_codex_cli(
            prompt=prompt,
            cwd=cwd,
            resume_id=resume_id,
            cancel_check=cancel_check,
            event_sink=event_sink,
        )
    if provider == "claude-code":
        return _run_claude_cli(
            prompt=prompt,
            cwd=cwd,
            resume_id=resume_id,
            cancel_check=cancel_check,
            event_sink=event_sink,
        )
    if provider == "gemini-cli":
        return _run_gemini_cli(
            prompt=prompt,
            cwd=cwd,
            resume_id=resume_id,
            cancel_check=cancel_check,
            event_sink=event_sink,
        )
    raise ValueError(f"Unsupported agent backend: {provider}")


class AgentBackendManager:
    """Track operator-owned local agent backend sessions."""

    def __init__(
        self,
        *,
        backend_runner: Callable[
            [
                str,
                str,
                str,
                str | None,
                Callable[[], bool],
                Callable[[AgentBackendEvent], None],
            ],
            AgentBackendRunResult,
        ] = run_agent_backend_session,
        thread_factory: Callable[..., Any] = threading.Thread,
    ):
        self._backend_runner = backend_runner
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._counter = 0
        self._sessions: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def launch(
        self,
        *,
        provider: str,
        prompt: str,
        cwd: str,
        resume_id: str | None = None,
    ) -> dict[str, Any]:
        provider = provider.strip().lower() if isinstance(provider, str) else ""
        if provider not in _ALLOWED_PROVIDERS:
            raise ValueError(f"Unsupported agent backend: {provider}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError("cwd must be a non-empty string")
        resume_id = (
            resume_id.strip()
            if isinstance(resume_id, str) and resume_id.strip()
            else None
        )

        with self._lock:
            self._counter += 1
            session_id = f"agent-backend-{provider}-{self._counter}"
            cancel_event = threading.Event()
            session = {
                "id": session_id,
                "provider": provider,
                "prompt": prompt.strip(),
                "cwd": cwd.strip(),
                "resume_id": resume_id,
                "state": "queued",
                "created_at": _iso_now(),
                "started_at": None,
                "finished_at": None,
                "provider_session_id": None,
                "result": None,
                "events": [],
                "event_counter": 0,
                "error": None,
                "backend_unavailable": False,
                "_process_pid": None,
                "_process_pgid": None,
                "_process_terminated": False,
                "_cancel_event": cancel_event,
            }
            self._sessions[session_id] = session
            self._order.insert(0, session_id)

        thread = self._thread_factory(
            target=self._run_session,
            args=(session_id,),
            daemon=True,
        )
        thread.start()
        return self._public_session(session)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._public_session(self._sessions[session_id])
                for session_id in self._order
            ]

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": f"Unknown agent backend session: {session_id}"}
            return self._public_session(session)

    def _append_session_event(self, session_id: str, event: AgentBackendEvent) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if event.kind == "_process_started":
                pid = event.data.get("pid")
                pgid = event.data.get("pgid")
                if isinstance(pid, int):
                    session["_process_pid"] = pid
                if isinstance(pgid, int):
                    session["_process_pgid"] = pgid
                return
            session["event_counter"] += 1
            session["events"].append(
                {
                    "sequence": session["event_counter"],
                    "kind": event.kind,
                    "text": event.text,
                    "data": event.data,
                }
            )

    def _terminate_session_process_locked(self, session: dict[str, Any]) -> None:
        if session.get("_process_terminated") is True:
            return
        pgid = session.get("_process_pgid")
        pid = session.get("_process_pid")
        try:
            if isinstance(pgid, int):
                os.killpg(pgid, signal.SIGTERM)
            elif isinstance(pid, int):
                os.kill(pid, signal.SIGTERM)
            else:
                return
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
        session["_process_terminated"] = True

    def cancel(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": f"Unknown agent backend session: {session_id}"}
            if session["state"] in {"completed", "failed", "cancelled"}:
                return self._public_session(session)
            session["_cancel_event"].set()
            self._terminate_session_process_locked(session)
            session["state"] = "cancelling"
            return self._public_session(session)

    def _run_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions[session_id]
            if session["_cancel_event"].is_set():
                session["state"] = "cancelled"
                session["finished_at"] = _iso_now()
                return
            session["state"] = "running"
            session["started_at"] = _iso_now()
            provider = session["provider"]
            prompt = session["prompt"]
            cwd = session["cwd"]
            resume_id = session["resume_id"]
            cancel_event = session["_cancel_event"]

        try:
            result = self._backend_runner(
                provider,
                prompt,
                cwd,
                resume_id,
                cancel_event.is_set,
                lambda event: self._append_session_event(session_id, event),
            )
        except AgentBackendUnavailable as exc:
            with self._lock:
                session = self._sessions[session_id]
                if session["_cancel_event"].is_set():
                    session["state"] = "cancelled"
                else:
                    session["state"] = "failed"
                    session["error"] = str(exc)
                    session["backend_unavailable"] = True
                session["finished_at"] = _iso_now()
            return
        except Exception as exc:
            with self._lock:
                session = self._sessions[session_id]
                if session["_cancel_event"].is_set():
                    session["state"] = "cancelled"
                else:
                    session["state"] = "failed"
                    session["error"] = str(exc)
                session["finished_at"] = _iso_now()
            return

        with self._lock:
            session = self._sessions[session_id]
            if session["_cancel_event"].is_set():
                session["state"] = "cancelled"
            else:
                session["state"] = "completed"
                session["provider_session_id"] = result.session_id
                session["result"] = result.final_response
                if not session["events"]:
                    for event in result.events:
                        session["event_counter"] += 1
                        session["events"].append(
                            {
                                "sequence": session["event_counter"],
                                "kind": event.kind,
                                "text": event.text,
                                "data": event.data,
                            }
                        )
            session["finished_at"] = _iso_now()

    @staticmethod
    def _public_session(session: dict[str, Any]) -> dict[str, Any]:
        result = session.get("result")
        preview = None
        if isinstance(result, str) and result:
            preview = result[:160]
        poll_hint = None
        if session["state"] in {"queued", "running", "cancelling"}:
            poll_hint = (
                "Agent backend still in flight. Continue other work and check "
                "again later when useful."
            )
        return {
            "id": session["id"],
            "provider": session["provider"],
            "prompt": session["prompt"],
            "cwd": session["cwd"],
            "resume_id": session["resume_id"],
            "state": session["state"],
            "created_at": session["created_at"],
            "started_at": session["started_at"],
            "finished_at": session["finished_at"],
            "provider_session_id": session["provider_session_id"],
            "result": result,
            "result_preview": preview,
            "backend_events": list(session.get("events") or []),
            "error": session.get("error"),
            "backend_unavailable": bool(session.get("backend_unavailable")),
            "poll_hint": poll_hint,
            "thread_card": build_agent_thread_card(session).to_dict(),
        }
