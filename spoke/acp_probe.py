"""Small ACP reconnaissance client for Agent Shell backend probes."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


PROTOCOL_VERSION = 1
_KNOWN_PROVIDERS = {"codex", "claude-code", "gemini-cli"}


class AcpProbeError(RuntimeError):
    """Raised when an ACP probe cannot complete."""


@dataclass(frozen=True)
class AcpSessionSupport:
    live_session_switching: bool
    session_catalog: bool
    load_existing_session: bool
    resume_existing_session: bool
    fork_checkpoint: bool
    close_session: bool


class JsonRpcLineCodec:
    """NDJSON JSON-RPC codec used by the ACP SDK stdio helpers."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def encode(self, message: dict[str, Any]) -> bytes:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        return payload + b"\n"

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        self._buffer.extend(data)
        messages: list[dict[str, Any]] = []
        while True:
            try:
                newline = self._buffer.index(0x0A)
            except ValueError:
                break
            raw = bytes(self._buffer[:newline]).strip()
            del self._buffer[: newline + 1]
            if raw:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    messages.append(parsed)
        return messages


def provider_command(provider: str) -> tuple[str, ...]:
    """Return the default ACP command for a known provider."""
    normalized = provider.strip().casefold()
    if normalized == "gemini-cli":
        return ("gemini", "--acp")
    if normalized == "codex":
        return ("codex-acp",)
    if normalized == "claude-code":
        return ("claude-agent-acp",)
    raise ValueError(f"unknown ACP provider: {provider}")


def provider_default_mode(provider: str) -> str:
    """Return Spoke's default ACP permission mode for a provider."""
    normalized = provider.strip().casefold()
    if normalized == "codex":
        return "full-access"
    if normalized == "claude-code":
        return "bypassPermissions"
    if normalized == "gemini-cli":
        return "default"
    raise ValueError(f"unknown ACP provider: {provider}")


def classify_session_support(init_response: dict[str, Any]) -> AcpSessionSupport:
    capabilities = init_response.get("agentCapabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    session_capabilities = capabilities.get("sessionCapabilities")
    session_capabilities = (
        session_capabilities if isinstance(session_capabilities, dict) else {}
    )
    return AcpSessionSupport(
        live_session_switching=True,
        session_catalog="list" in session_capabilities,
        load_existing_session=capabilities.get("loadSession") is True,
        resume_existing_session="resume" in session_capabilities,
        fork_checkpoint="fork" in session_capabilities,
        close_session="close" in session_capabilities,
    )


def _text_content(update: dict[str, Any]) -> str:
    content = update.get("content")
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def summarize_session_update(update: dict[str, Any]) -> dict[str, Any]:
    """Project an ACP session update into the bearing/card questions we care about."""
    update_kind = update.get("sessionUpdate")
    if update_kind == "agent_message_chunk":
        return {
            "kind": "agent_message",
            "bearing_relevant": False,
            "text": _text_content(update),
        }
    if update_kind == "agent_thought_chunk":
        return {
            "kind": "thought",
            "bearing_relevant": True,
            "text": _text_content(update),
        }
    if update_kind == "plan":
        entries = update.get("entries")
        return {
            "kind": "plan",
            "bearing_relevant": True,
            "entries": entries if isinstance(entries, list) else [],
        }
    if update_kind == "tool_call":
        return {
            "kind": "tool_call",
            "bearing_relevant": True,
            "tool_call_id": str(update.get("toolCallId") or ""),
            "title": str(update.get("title") or ""),
            "status": str(update.get("status") or ""),
            "tool_kind": str(update.get("kind") or ""),
        }
    if update_kind == "tool_call_update":
        return {
            "kind": "tool_call_update",
            "bearing_relevant": True,
            "tool_call_id": str(update.get("toolCallId") or ""),
            "status": str(update.get("status") or ""),
            "has_content": bool(update.get("content")),
            "has_locations": bool(update.get("locations")),
        }
    if update_kind == "usage_update":
        return {
            "kind": "usage",
            "bearing_relevant": False,
            "usage": update.get("usage") if isinstance(update.get("usage"), dict) else {},
        }
    if update_kind == "current_mode_update":
        return {
            "kind": "mode",
            "bearing_relevant": False,
            "modes": update.get("modes") if isinstance(update.get("modes"), dict) else {},
        }
    if update_kind == "config_option_update":
        return {
            "kind": "config_option",
            "bearing_relevant": False,
            "option": update.get("option") if isinstance(update.get("option"), dict) else {},
        }
    if update_kind == "available_commands_update":
        return {
            "kind": "available_commands",
            "bearing_relevant": False,
            "commands": update.get("commands") if isinstance(update.get("commands"), list) else [],
        }
    if update_kind == "session_info_update":
        return {
            "kind": "session_info",
            "bearing_relevant": False,
            "title": update.get("title"),
            "updated_at": update.get("updatedAt"),
        }
    return {"kind": str(update_kind or "unknown"), "bearing_relevant": False}


class AcpJsonRpcClient:
    def __init__(
        self,
        command: tuple[str, ...],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.cwd = cwd
        self.env = env
        self.codec = JsonRpcLineCodec()
        self.process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            list(self.command),
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=self.env,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        while True:
            chunk = process.stdout.readline()
            if not chunk:
                break
            for message in self.codec.feed(chunk):
                self._messages.put(message)

    def _write(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise AcpProbeError("ACP process is not running")
        process.stdin.write(self.codec.encode(message))
        process.stdin.flush()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
        notification_sink: Callable[[dict[str, Any]], None] | None = None,
        request_handler: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        request_id = self._next_id
        self._next_id += 1
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        notifications: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                message = self._messages.get(timeout=min(0.25, remaining))
            except queue.Empty:
                process = self.process
                if process is not None and process.poll() is not None:
                    raise AcpProbeError(f"ACP process exited with {process.returncode}")
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise AcpProbeError(f"{method} failed: {message['error']}")
                result = message.get("result")
                return (result if isinstance(result, dict) else {}, notifications)
            if (
                "id" in message
                and isinstance(message.get("method"), str)
                and request_handler is not None
            ):
                response = request_handler(message)
                if response is not None:
                    self._write(
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": response,
                        }
                    )
                continue
            notifications.append(message)
            if notification_sink is not None:
                notification_sink(message)
        raise AcpProbeError(f"timed out waiting for ACP response to {method}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})


def run_probe(
    command: tuple[str, ...],
    *,
    cwd: str,
    prompt: str | None = None,
    timeout: float = 30.0,
    cancel_after: float | None = None,
) -> dict[str, Any]:
    client = AcpJsonRpcClient(command, cwd=cwd)
    client.start()
    try:
        init, init_notifications = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "_meta": {"terminal_output": True},
                },
            },
            timeout=timeout,
        )
        session, session_notifications = client.request(
            "session/new",
            {"cwd": str(Path(cwd).resolve()), "mcpServers": []},
            timeout=timeout,
        )
        session_id = session.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AcpProbeError("session/new did not return a sessionId")

        prompt_result: dict[str, Any] | None = None
        prompt_notifications: list[dict[str, Any]] = []
        if prompt:
            if cancel_after is None:
                prompt_result, prompt_notifications = client.request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": prompt}],
                        "messageId": str(uuid.uuid4()),
                    },
                    timeout=timeout,
                )
            else:
                holder: dict[str, Any] = {}

                def _prompt() -> None:
                    try:
                        result, notifications = client.request(
                            "session/prompt",
                            {
                                "sessionId": session_id,
                                "prompt": [{"type": "text", "text": prompt}],
                                "messageId": str(uuid.uuid4()),
                            },
                            timeout=timeout,
                        )
                        holder["result"] = result
                        holder["notifications"] = notifications
                    except BaseException as exc:  # noqa: BLE001 - probe should report failures.
                        holder["error"] = repr(exc)

                thread = threading.Thread(target=_prompt, daemon=True)
                thread.start()
                time.sleep(cancel_after)
                client.notify("session/cancel", {"sessionId": session_id})
                thread.join(timeout=max(1.0, timeout))
                prompt_result = holder.get("result")
                prompt_notifications = holder.get("notifications", [])
                if "error" in holder:
                    prompt_result = {"error": holder["error"]}

        updates = [
            message.get("params", {}).get("update")
            for message in [*init_notifications, *session_notifications, *prompt_notifications]
            if message.get("method") == "session/update"
            and isinstance(message.get("params"), dict)
            and isinstance(message.get("params", {}).get("update"), dict)
        ]
        summarized_updates = [summarize_session_update(update) for update in updates]
        return {
            "command": list(command),
            "initialize": init,
            "session_support": asdict(classify_session_support(init)),
            "session": session,
            "prompt_result": prompt_result,
            "update_kinds": sorted(
                {str(update.get("sessionUpdate")) for update in updates}
            ),
            "bearing_relevant_updates": [
                update for update in summarized_updates if update.get("bearing_relevant")
            ],
            "summarized_updates": summarized_updates,
        }
    finally:
        client.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an ACP agent over stdio")
    parser.add_argument(
        "provider",
        choices=sorted(_KNOWN_PROVIDERS),
        help="Known ACP provider command to launch",
    )
    parser.add_argument("--cwd", default=os.getcwd(), help="Session working directory")
    parser.add_argument("--prompt", help="Optional prompt turn to send")
    parser.add_argument(
        "--cancel-after",
        type=float,
        help="Send session/cancel this many seconds after starting the prompt",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="Override command after --, e.g. --command -- node ./agent.js",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    command = tuple(args.command) if args.command else provider_command(args.provider)
    if command and command[0] == "--":
        command = command[1:]
    try:
        result = run_probe(
            command,
            cwd=args.cwd,
            prompt=args.prompt,
            timeout=args.timeout,
            cancel_after=args.cancel_after,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
