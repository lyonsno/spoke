"""Opt-in trace breadcrumbs for assistant overlay gesture debugging."""

from __future__ import annotations

import atexit
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import threading

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_IDENTITY = None
_SOURCE_IDENTITY_LOCK = threading.Lock()
_TRACE_QUEUE: queue.Queue[tuple[str, dict[str, object]]] = queue.Queue()
_TRACE_SEQUENCE = 0
_TRACE_SEQUENCE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _trace_writer() -> None:
    while True:
        event, details = _TRACE_QUEUE.get()
        try:
            _write_command_overlay_trace(event, dict(details))
        except Exception as exc:
            try:
                _write_trace_failure(event, details, exc)
            except Exception:
                logger.exception("Trace event %s could not be written or receipted", event)
        finally:
            _TRACE_QUEUE.task_done()


_TRACE_WRITER = threading.Thread(
    target=_trace_writer, name="spoke-command-overlay-trace", daemon=True
)
_TRACE_WRITER.start()


def _source_identity() -> dict[str, object]:
    global _SOURCE_IDENTITY
    if _SOURCE_IDENTITY is not None:
        return dict(_SOURCE_IDENTITY)
    with _SOURCE_IDENTITY_LOCK:
        if _SOURCE_IDENTITY is not None:
            return dict(_SOURCE_IDENTITY)
        try:
            revision = subprocess.run(
                ["git", "-C", str(_SOURCE_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(_SOURCE_ROOT),
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                        "--",
                        ".",
                        ":(exclude).spoke-smoke-env",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision = None
            dirty = None
        try:
            smoke_env_sha256 = hashlib.sha256(
                (_SOURCE_ROOT / ".spoke-smoke-env").read_bytes()
            ).hexdigest()
        except OSError:
            smoke_env_sha256 = None
        _SOURCE_IDENTITY = {
            "source_root": str(_SOURCE_ROOT),
            "source_revision": revision,
            "source_dirty": dirty,
            "smoke_env_sha256": smoke_env_sha256,
        }
    return dict(_SOURCE_IDENTITY)


def _write_command_overlay_trace(event: str, details: dict[str, object]) -> None:
    path_text = str(details.pop("trace_path", "") or "").strip()
    if not path_text:
        path_text = os.environ.get("SPOKE_COMMAND_OVERLAY_TRACE_PATH", "").strip()
    if not path_text:
        return
    payload = {
        "timestamp": details.pop("timestamp", datetime.now().astimezone().isoformat(timespec="milliseconds")),
        "write_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        "pid": details.pop("pid", os.getpid()),
        "thread": details.pop("event_thread", threading.current_thread().name),
        "launch_id": details.pop("launch_id", os.environ.get("SPOKE_LAUNCH_ID")),
        "launch_target_id": details.pop("launch_target_id", os.environ.get("SPOKE_LAUNCH_TARGET_ID")),
        **_source_identity(),
    }
    payload.update({key: value for key, value in details.items() if value is not None})
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_trace_failure(event: str, details: dict[str, object], error: Exception) -> None:
    path_text = str(details.get("trace_path") or "").strip()
    if not path_text:
        return
    failure_path = Path(f"{Path(path_text).expanduser()}.failures.jsonl")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "trace.write.failed",
        "source_event": event,
        "timestamp": details.get("timestamp"),
        "trace_sequence": details.get("trace_sequence"),
        "trace_path": path_text,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def enqueue_command_overlay_trace(event: str, **details) -> None:
    """Queue trace I/O so render callbacks never wait on git or the filesystem."""
    global _TRACE_SEQUENCE
    path = os.environ.get("SPOKE_COMMAND_OVERLAY_TRACE_PATH", "").strip()
    if not path:
        return
    details.setdefault("timestamp", datetime.now().astimezone().isoformat(timespec="milliseconds"))
    details.setdefault("event_thread", threading.current_thread().name)
    details.setdefault("pid", os.getpid())
    details.setdefault("launch_id", os.environ.get("SPOKE_LAUNCH_ID"))
    details.setdefault("launch_target_id", os.environ.get("SPOKE_LAUNCH_TARGET_ID"))
    details.setdefault("trace_path", path)
    with _TRACE_SEQUENCE_LOCK:
        _TRACE_SEQUENCE += 1
        details.setdefault("trace_sequence", _TRACE_SEQUENCE)
    _TRACE_QUEUE.put((event, details))


def record_command_overlay_trace(event: str, **details) -> None:
    """Compatibility entry point; trace writes are asynchronous."""
    enqueue_command_overlay_trace(event, **details)


def flush_command_overlay_trace() -> None:
    """Wait for queued trace writes; never call from an animation callback."""
    _TRACE_QUEUE.join()


atexit.register(flush_command_overlay_trace)
