"""Asynchronous Spoke client for GPU Greenroom interactive leases."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid


logger = logging.getLogger(__name__)

_REPORT_SCHEMA = "spoke.gpu-interactive-lease.v1"
_GREENROOM_SCHEMA = "gpu-greenroom.interactive-lease.v1"
_FALSE_VALUES = {"0", "false", "no", "off"}


def _enabled_from_environment() -> bool:
    return (
        os.environ.get("SPOKE_GPU_INTERACTIVE_LEASE", "0").strip().lower()
        not in _FALSE_VALUES
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


class GPUInteractiveLease:
    """One per-recording holder process and its truthful Spoke-side report."""

    def __init__(
        self,
        *,
        lease_id: str,
        binary: Path | None,
        queue_dir: Path,
        receipt_dir: Path,
        route_id: str | None,
        popen_factory,
        thread_factory,
        on_finished,
    ) -> None:
        self.lease_id = lease_id
        self.binary = binary
        self.queue_dir = queue_dir
        self.greenroom_receipt_path = receipt_dir / f"{lease_id}.greenroom.json"
        self.report_path = receipt_dir / f"{lease_id}.spoke.json"
        self.stderr_path = receipt_dir / f"{lease_id}.stderr.log"
        self.route_id = route_id
        self._popen_factory = popen_factory
        self._thread_factory = thread_factory
        self._on_finished = on_finished
        self._process = None
        self._lock = threading.Lock()
        self._release_requested = False
        self._report = {
            "schema": _REPORT_SCHEMA,
            "lease_id": lease_id,
            "requested": True,
            "effective": False,
            "state": "launching",
            "scheduling_posture": "scheduler-unverified",
            "requested_at": time.time(),
            "updated_at": time.time(),
            "holder_pid": None,
            "requested_route": {
                "binary": str(binary) if binary is not None else None,
                "queue_dir": str(queue_dir),
                "purpose": "final-asr",
                "route_id": route_id,
            },
            "effective_route": None,
            "greenroom_receipt_path": str(self.greenroom_receipt_path),
            "stderr_path": str(self.stderr_path),
            "report_path": str(self.report_path),
            "failure_phase": None,
            "error": None,
            "last_trustworthy_event": "spoke-request-persisted",
            "current_authority": "requires-live-holder-process-and-effective-greenroom-event",
        }
        self._write_report()

    def start(self) -> GPUInteractiveLease:
        if self.binary is None or not self.binary.is_file() or not os.access(
            self.binary, os.X_OK
        ):
            self._fail(
                "launch",
                "launch-rejected-missing-binary",
                f"GPU Greenroom binary is unavailable: {self.binary}",
            )
            return self

        command = [
            str(self.binary),
            "--queue-dir",
            str(self.queue_dir),
            "lease",
            "acquire",
            "--lease-id",
            self.lease_id,
            "--holder",
            "spoke",
            "--purpose",
            "final-asr",
            "--receipt-path",
            str(self.greenroom_receipt_path),
        ]
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.stderr_path.open("a", encoding="utf-8") as stderr_file:
                self._process = self._popen_factory(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    bufsize=1,
                )
        except Exception as exc:
            self._fail("launch", "holder-process-launch-failed", str(exc))
            return self

        with self._lock:
            self._report["holder_pid"] = self._process.pid
            self._report["updated_at"] = time.time()
            self._write_report_locked()
        thread = self._thread_factory(
            target=self._read_events,
            daemon=True,
            name=f"spoke-gpu-lease-{self.lease_id}",
        )
        thread.start()
        return self

    def snapshot(self) -> dict:
        with self._lock:
            report = dict(self._report)
            process = self._process
            if report["state"] == "effective" and (
                process is None or process.poll() is not None
            ):
                report["state"] = "stale-effective"
                report["effective"] = False
                report["scheduling_posture"] = "scheduler-unverified"
            return report

    def release(self) -> None:
        with self._lock:
            if self._release_requested:
                return
            self._release_requested = True
            process = self._process
            if self._report["state"] not in {"failed", "released", "released-unacquired"}:
                self._report["state"] = "release-requested"
                self._report["effective"] = False
                self._report["scheduling_posture"] = "scheduler-unverified"
                self._report["updated_at"] = time.time()
                self._report["last_trustworthy_event"] = "spoke-release-requested"
                self._write_report_locked()
        if process is not None and process.poll() is None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                logger.warning("GPU lease stdin already closed: %s", self.lease_id)

    def _read_events(self) -> None:
        process = self._process
        assert process is not None
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        self._fail(
                            "protocol",
                            "invalid-jsonl-event",
                            "GPU Greenroom emitted invalid JSONL",
                        )
                        continue
                    self._handle_event(event)
            return_code = process.wait()
            with self._lock:
                if self._report["state"] not in {
                    "released",
                    "released-unacquired",
                    "failed",
                }:
                    self._report["state"] = "failed"
                    self._report["effective"] = False
                    self._report["scheduling_posture"] = "scheduler-unverified"
                    self._report["failure_phase"] = "holder-exit"
                    self._report["error"] = f"holder exited with code {return_code}"
                    self._report["last_trustworthy_event"] = "holder-exited-without-terminal-event"
                    self._report["updated_at"] = time.time()
                    self._write_report_locked()
        finally:
            self._on_finished(self)

    def _handle_event(self, event: dict) -> None:
        if event.get("schema") != _GREENROOM_SCHEMA or event.get("lease_id") != self.lease_id:
            self._fail(
                "protocol",
                "route-identity-mismatch",
                "GPU Greenroom event schema or lease identity did not match",
            )
            return
        state = event.get("state")
        if state not in {"requested", "effective", "released", "released-unacquired"}:
            self._fail(
                "protocol",
                "invalid-state-event",
                f"GPU Greenroom emitted unsupported lease state: {state!r}",
            )
            return
        if state == "effective" and event.get("effective_at") is None:
            self._fail(
                "protocol",
                "invalid-effective-event",
                "GPU Greenroom effective event omitted effective_at",
            )
            return
        with self._lock:
            current_state = self._report["state"]
            if current_state in {"failed", "released", "released-unacquired"}:
                logger.warning(
                    "Ignoring GPU lease event after terminal state: lease=%s "
                    "current=%s event=%s",
                    self.lease_id,
                    current_state,
                    state,
                )
                return
            if self._release_requested and state not in {
                "released",
                "released-unacquired",
            }:
                logger.warning(
                    "Ignoring nonterminal GPU lease event after release request: "
                    "lease=%s event=%s",
                    self.lease_id,
                    state,
                )
                return
            self._report["state"] = state
            self._report["effective"] = state == "effective"
            self._report["scheduling_posture"] = (
                "lease-effective" if state == "effective" else "scheduler-unverified"
            )
            if state == "effective":
                self._report["effective_route"] = {
                    **event,
                    "binary": str(self.binary),
                    "queue_dir": str(self.queue_dir),
                    "route_id": self.route_id,
                }
            self._report["updated_at"] = time.time()
            self._report["last_trustworthy_event"] = f"greenroom-{state}"
            self._write_report_locked()
        logger.info(
            "GPU lease %s: requested=True effective=%s state=%s pid=%s receipt=%s",
            self.lease_id,
            state == "effective",
            state,
            getattr(self._process, "pid", None),
            self.greenroom_receipt_path,
        )

    def _fail(self, phase: str, event: str, error: str) -> None:
        with self._lock:
            self._report["state"] = "failed"
            self._report["effective"] = False
            self._report["scheduling_posture"] = "scheduler-unverified"
            self._report["failure_phase"] = phase
            self._report["error"] = error
            self._report["updated_at"] = time.time()
            self._report["last_trustworthy_event"] = event
            self._write_report_locked()
        logger.warning("GPU lease %s failed in %s: %s", self.lease_id, phase, error)

    def _write_report(self) -> None:
        with self._lock:
            self._write_report_locked()

    def _write_report_locked(self) -> None:
        _atomic_write_json(self.report_path, self._report)


class GPUInteractiveLeaseManager:
    """Create and drain independent lease holders for overlapping recordings."""

    def __init__(
        self,
        *,
        enabled: bool,
        binary: str | Path | None,
        queue_dir: str | Path,
        receipt_dir: str | Path,
        route_id: str | None = None,
        popen_factory=subprocess.Popen,
        thread_factory=threading.Thread,
    ) -> None:
        self.enabled = enabled
        self.binary = Path(binary).expanduser() if binary is not None else None
        self.queue_dir = Path(queue_dir).expanduser()
        self.receipt_dir = Path(receipt_dir).expanduser()
        self.route_id = route_id
        self._popen_factory = popen_factory
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._sessions: set[GPUInteractiveLease] = set()

    @classmethod
    def from_environment(cls) -> GPUInteractiveLeaseManager:
        binary_value = os.environ.get("SPOKE_GPU_GREENROOM_BINARY", "").strip()
        binary = Path(binary_value) if binary_value else shutil.which("gpu-greenroom")
        return cls(
            enabled=_enabled_from_environment(),
            binary=binary,
            queue_dir=os.environ.get(
                "SPOKE_GPU_GREENROOM_DIR",
                str(Path.home() / ".local/state/gpu-greenroom"),
            ),
            receipt_dir=os.environ.get(
                "SPOKE_GPU_LEASE_RECEIPT_DIR",
                str(Path.home() / "Library/Application Support/Spoke/gpu-lease-receipts"),
            ),
            route_id=os.environ.get("SPOKE_GPU_GREENROOM_ROUTE_ID", "").strip() or None,
        )

    def request(self, lease_id: str) -> GPUInteractiveLease | None:
        if not self.enabled:
            return None
        lease = GPUInteractiveLease(
            lease_id=lease_id,
            binary=self.binary,
            queue_dir=self.queue_dir,
            receipt_dir=self.receipt_dir,
            route_id=self.route_id,
            popen_factory=self._popen_factory,
            thread_factory=self._thread_factory,
            on_finished=self._discard,
        )
        with self._lock:
            self._sessions.add(lease)
        return lease.start()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions)
        for lease in sessions:
            lease.release()

    def _discard(self, lease: GPUInteractiveLease) -> None:
        with self._lock:
            self._sessions.discard(lease)
