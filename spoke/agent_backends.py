"""Local-auth coding-agent backend sessions for the operator shell."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


_ALLOWED_PROVIDERS = {"codex"}
_BILLING_CREDENTIAL_ENV = ("OPENAI_" + "API_KEY", "CODEX_" + "API_KEY")
_CODEX_CANDIDATE_PATHS = (
    "/opt/homebrew/bin/codex",
    "/usr/local/bin/codex",
    os.path.expanduser("~/.local/bin/codex"),
)


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


def _append_event(
    events: list[AgentBackendEvent],
    event: AgentBackendEvent,
    event_sink: Callable[[AgentBackendEvent], None] | None,
) -> None:
    events.append(event)
    if event_sink is not None:
        event_sink(event)


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
        )
    except OSError as exc:
        raise AgentBackendUnavailable(f"Codex CLI failed to start: {exc}") from exc

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
    return AgentBackendRunResult(
        provider="codex",
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
        raise AgentBackendUnavailable(
            "Claude Code CLI backend is reserved but not wired yet; "
            "Anthropic Agent SDK is excluded from this no-billing design."
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
            session["event_counter"] += 1
            session["events"].append(
                {
                    "sequence": session["event_counter"],
                    "kind": event.kind,
                    "text": event.text,
                    "data": event.data,
                }
            )

    def cancel(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": f"Unknown agent backend session: {session_id}"}
            if session["state"] in {"completed", "failed", "cancelled"}:
                return self._public_session(session)
            session["_cancel_event"].set()
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
        }
