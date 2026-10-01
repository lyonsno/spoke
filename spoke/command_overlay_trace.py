"""Opt-in trace breadcrumbs for assistant overlay gesture debugging."""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_IDENTITY = None
_SOURCE_IDENTITY_LOCK = threading.Lock()
_TRACE_QUEUE: queue.Queue[tuple[str, dict[str, object]]] = queue.Queue()
_TRACE_SEQUENCE = 0
_TRACE_SEQUENCE_LOCK = threading.Lock()
_TRACE_FILES = threading.local()
logger = logging.getLogger(__name__)


def _trace_writer() -> None:
    while True:
        batch = [_TRACE_QUEUE.get()]
        # Drain the backlog present at this boundary, without dropping records
        # or letting continuously arriving producers postpone the first write.
        for _ in range(_TRACE_QUEUE.qsize()):
            try:
                batch.append(_TRACE_QUEUE.get_nowait())
            except queue.Empty:
                break
        try:
            with ExitStack() as files:
                _TRACE_FILES.handles = {}
                _TRACE_FILES.stack = files
                for event, details in batch:
                    try:
                        _write_command_overlay_trace(event, dict(details))
                    except Exception as exc:
                        _receipt_failure(event, details, exc)
        except Exception as exc:
            # A buffered close/flush failure can affect every record in a batch.
            for event, details in batch:
                _receipt_failure(event, details, exc)
        finally:
            _TRACE_FILES.handles = None
            _TRACE_FILES.stack = None
            for _ in batch:
                _TRACE_QUEUE.task_done()


def _receipt_failure(event, details, error):
    try:
        _write_trace_failure(event, details, error)
    except Exception:
        logger.exception("Trace event %s could not be written or receipted", event)


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
    event_time = details.pop("event_time_unix_seconds", None)
    timestamp = details.pop("timestamp", None)
    if timestamp is None:
        timestamp = datetime.fromtimestamp(event_time if event_time is not None else time.time()).astimezone().isoformat(timespec="milliseconds")
    thread_id = details.pop("event_thread_id", None)
    event_thread = details.pop("event_thread", None)
    if event_thread is None:
        event_thread = f"thread-{thread_id}" if thread_id is not None else threading.current_thread().name
    payload = {
        "timestamp": timestamp,
        "write_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        "pid": details.pop("pid", os.getpid()),
        "thread": event_thread,
        "event_thread_id": thread_id,
        "launch_id": details.pop("launch_id", os.environ.get("SPOKE_LAUNCH_ID")),
        "launch_target_id": details.pop("launch_target_id", os.environ.get("SPOKE_LAUNCH_TARGET_ID")),
        **_source_identity(),
    }
    payload.update({key: value for key, value in details.items() if value is not None})
    path = Path(path_text).expanduser()
    line = json.dumps(payload, sort_keys=True) + "\n"
    handles = getattr(_TRACE_FILES, "handles", None)
    if handles is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    else:
        if path not in handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[path] = _TRACE_FILES.stack.enter_context(path.open("a", encoding="utf-8"))
        handles[path].write(line)


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
        "event_time_unix_seconds": details.get("event_time_unix_seconds"),
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
    details.setdefault("event_time_unix_seconds", time.time())
    details.setdefault("event_thread_id", threading.get_ident())
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
