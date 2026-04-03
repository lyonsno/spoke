"""Process heartbeat and model TTL for zombie prevention.

Spoke writes a heartbeat file every HEARTBEAT_INTERVAL_S seconds.  On startup
the zombie sweep reads the heartbeat and kills any stale spoke process before
the single-instance lock is acquired.  The same timer that writes the heartbeat
also checks model-last-use timestamps and returns a list of model IDs that
have exceeded the TTL so the caller can evict them.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = os.environ.get(
    "SPOKE_HEARTBEAT_PATH",
    os.path.expanduser("~/Library/Logs/.spoke-heartbeat.json"),
)
HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_MODEL_TTL_S = 600.0  # 10 minutes
STALE_THRESHOLD_S = 120.0  # 2 minutes without heartbeat = stale


def _is_process_alive(pid: int) -> bool:
    """Check whether *pid* refers to a running process (same user)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive but owned by another user — shouldn't happen for spoke.
        return True


def _is_spoke_process(pid: int) -> bool:
    """Best-effort check that *pid* is actually a spoke process."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode("utf-8", errors="replace")
        return "-m" in cmdline and "spoke" in cmdline
    except FileNotFoundError:
        pass
    # macOS: use ps
    try:
        import subprocess

        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cmd = result.stdout.strip()
        return "spoke" in cmd and ("-m" in cmd or "spoke/" in cmd)
    except Exception:
        return False


def zombie_sweep(heartbeat_path: str = HEARTBEAT_PATH) -> None:
    """Kill stale spoke processes discovered via the heartbeat file.

    Runs once at startup, before the single-instance lock is acquired.
    """
    try:
        raw = Path(heartbeat_path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return

    pid = data.get("pid")
    ts_raw = data.get("timestamp")
    if not isinstance(pid, int) or not ts_raw:
        return

    my_pid = os.getpid()
    if pid == my_pid:
        return

    if not _is_process_alive(pid):
        logger.info("Zombie sweep: heartbeat pid %d is already dead — cleaning up", pid)
        _remove_heartbeat(heartbeat_path)
        return

    # Process is alive — check staleness
    try:
        last_beat = datetime.fromisoformat(ts_raw)
        if last_beat.tzinfo is None:
            last_beat = last_beat.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - last_beat).total_seconds()
    except (ValueError, TypeError):
        age_s = float("inf")

    if age_s <= STALE_THRESHOLD_S:
        # Fresh heartbeat — process is healthy, defer to single-instance guard.
        logger.info(
            "Zombie sweep: heartbeat pid %d is fresh (%.0fs old) — deferring to lock guard",
            pid,
            age_s,
        )
        return

    # Stale heartbeat — verify it's actually a spoke process before killing.
    if not _is_spoke_process(pid):
        logger.warning(
            "Zombie sweep: pid %d has stale heartbeat but is not a spoke process — skipping",
            pid,
        )
        _remove_heartbeat(heartbeat_path)
        return

    logger.warning(
        "Zombie sweep: stale heartbeat (pid=%d, age=%.0fs) — sending SIGTERM",
        pid,
        age_s,
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        _remove_heartbeat(heartbeat_path)
        return

    # Give it 2 seconds to exit.
    for _ in range(20):
        time.sleep(0.1)
        if not _is_process_alive(pid):
            logger.info("Zombie sweep: pid %d exited after SIGTERM", pid)
            _remove_heartbeat(heartbeat_path)
            return

    logger.warning("Zombie sweep: pid %d still alive after SIGTERM — sending SIGKILL", pid)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    _remove_heartbeat(heartbeat_path)


def _remove_heartbeat(heartbeat_path: str) -> None:
    try:
        os.unlink(heartbeat_path)
    except OSError:
        pass


class HeartbeatManager:
    """Writes a periodic heartbeat and checks model TTLs.

    The caller is responsible for scheduling ``tick()`` on a timer.  Each tick
    writes the heartbeat file and returns a list of model IDs that have
    exceeded their TTL so the caller can evict them.
    """

    def __init__(
        self,
        heartbeat_path: str = HEARTBEAT_PATH,
        model_ttl_s: float = DEFAULT_MODEL_TTL_S,
    ) -> None:
        self._path = heartbeat_path
        self._ttl = model_ttl_s
        self._pid = os.getpid()
        self._models: dict[str, float] = {}  # model_id -> last-use monotonic
        self._launch_target: str | None = None
        self._worktree: str | None = None
        self._on_evict: Callable[[str], None] | None = None

    # ── Public API ──────────────────────────────────────────────

    def set_context(
        self,
        launch_target: str | None = None,
        worktree: str | None = None,
    ) -> None:
        """Set contextual metadata included in the heartbeat file."""
        if launch_target is not None:
            self._launch_target = launch_target
        if worktree is not None:
            self._worktree = worktree

    def set_evict_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when a model should be evicted.

        The callback receives the model_id and should unload the model,
        then call :meth:`unregister_model`.
        """
        self._on_evict = callback

    def register_model(self, model_id: str) -> None:
        """Record that *model_id* was just loaded."""
        self._models[model_id] = time.monotonic()

    def unregister_model(self, model_id: str) -> None:
        """Record that *model_id* was just unloaded."""
        self._models.pop(model_id, None)

    def touch(self, model_id: str) -> None:
        """Update last-use time for *model_id*."""
        if model_id in self._models:
            self._models[model_id] = time.monotonic()

    def tick(self) -> list[str]:
        """Write heartbeat and return model IDs that exceeded their TTL.

        The caller should evict the returned models (unload weights, free GPU
        memory) and then call :meth:`unregister_model` for each.
        """
        expired = self._check_ttls()
        self._write_heartbeat()
        return expired

    def remove(self) -> None:
        """Remove the heartbeat file (called during clean shutdown)."""
        _remove_heartbeat(self._path)

    # ── Internals ───────────────────────────────────────────────

    def _check_ttls(self) -> list[str]:
        now = time.monotonic()
        expired = [
            model_id
            for model_id, last_use in self._models.items()
            if (now - last_use) > self._ttl
        ]
        if expired and self._on_evict is not None:
            for model_id in expired:
                try:
                    self._on_evict(model_id)
                except Exception:
                    logger.exception("Failed to evict model %s", model_id)
        return expired

    def _write_heartbeat(self) -> None:
        now_wall = datetime.now(timezone.utc).isoformat()
        now_mono = time.monotonic()
        data = {
            "pid": self._pid,
            "timestamp": now_wall,
            "models_loaded": sorted(self._models.keys()),
            "model_last_use": {
                mid: f"{now_wall[:-6]}-{int(now_mono - last_use):d}s-ago"
                for mid, last_use in self._models.items()
            },
            "launch_target": self._launch_target,
            "worktree": self._worktree,
        }
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            logger.warning("Failed to write heartbeat to %s", self._path)

    def clear_metal_cache(self) -> None:
        """Best-effort MLX Metal cache clear after model eviction."""
        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass
        gc.collect()
