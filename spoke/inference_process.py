"""Owned, reusable inference process with explicit user cancellation."""

from contextlib import ExitStack, contextmanager
import importlib
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Connection

logger = logging.getLogger(__name__)

def _reap(process, connection=None):
    """Reaping a killed native worker must not retain inference or AppKit."""
    def finish():
        try:
            process.wait()
        finally:
            if connection is not None:
                connection.close()
    threading.Thread(target=finish, daemon=True, name="spoke-inference-reap").start()


@contextmanager
def _request_activity(method):
    receipt = {"schema": "spoke.inference_activity.v1", "platform": sys.platform,
               "status": "not_requested"}
    if method != "transcribe":
        yield receipt
        return
    if sys.platform != "darwin":
        receipt["status"] = "not_applicable"
        yield receipt
        return

    info = token = None
    try:
        from Foundation import NSActivityUserInitiatedAllowingIdleSystemSleep, NSProcessInfo
        info = NSProcessInfo.processInfo()
        options = NSActivityUserInitiatedAllowingIdleSystemSleep
        receipt["options"] = int(options)
        token = info.beginActivityWithOptions_reason_(options, "Transcribing requested dictation")
        if token is None:
            raise RuntimeError("macOS returned no activity token")
        receipt["status"] = "active"
    except Exception as exc:
        receipt.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        logger.warning("Inference user activity unavailable: %s", receipt["error"])

    try:
        yield receipt
    finally:
        if token is not None:
            try:
                info.endActivity_(token)
                receipt["status"] = "ended"
            except Exception as exc:
                receipt.update(status="end_failed", error=f"{type(exc).__name__}: {exc}")
                logger.warning("Inference user activity release failed: %s", receipt["error"])


class InferenceCancelled(RuntimeError):
    def __init__(self):
        super().__init__("Stopped by user for re-transcription; original audio retained")


class InferenceProcess:
    def __init__(self, module, client_class, kwargs):
        self._config = (module, client_class, kwargs)
        self._calls = threading.Lock()
        self._state = threading.Lock()
        self._generation = 0
        self._process = self._connection = None

    @property
    def pid(self):
        with self._state:
            return self._process.pid if self._process is not None else None

    @property
    def generation(self):
        with self._state:
            return self._generation

    def call(self, method, *args, generation=None):
        if generation is None:
            generation = self.generation
        with self._calls:
            with self._state:
                if generation != self._generation:
                    raise InferenceCancelled()
                if self._process is None:
                    parent, child = socket.socketpair()
                    try:
                        root = Path(__file__).resolve().parents[1]
                        process = subprocess.Popen(
                            [sys.executable, "-m", "spoke.inference_process", str(child.fileno()), str(os.getpid())],
                            cwd=root, pass_fds=(child.fileno(),), start_new_session=True,
                        )
                        connection = Connection(parent.detach())
                        self._process, self._connection = process, connection
                    finally:
                        parent.close()
                        child.close()
                process, connection = self._process, self._connection
            request_id = uuid.uuid4().hex
            transferred = threading.Event()
            reply = {}
            def exchange():
                try:
                    connection.send((request_id, self._config, method, args))
                    reply["response"] = connection.recv()
                except BaseException as exc:
                    reply["error"] = exc
                finally:
                    transferred.set()
            try:
                threading.Thread(target=exchange, daemon=True,
                                 name="spoke-inference-transfer").start()
                while not transferred.wait(0.05):
                    if generation != self.generation:
                        raise InferenceCancelled()
                if generation != self.generation:
                    raise InferenceCancelled()
                if "error" in reply:
                    raise reply["error"]
                response = reply["response"]
                with self._state:
                    if generation != self._generation:
                        raise InferenceCancelled()
                if not isinstance(response, dict) or response.get("request_id") != request_id:
                    raise RuntimeError("Inference worker returned an invalid request identity")
                worker = response.get("worker", {})
                if (worker.get("pid") != process.pid
                        or worker.get("source_root") != str(Path(__file__).resolve().parents[1])
                        or not isinstance(response.get("effective"), dict)
                        or ("value" not in response and not response.get("error"))):
                    raise RuntimeError("Inference worker returned an unverified route or partial result")
                return response
            except (EOFError, OSError) as exc:
                with self._state:
                    cancelled = generation != self._generation
                if cancelled:
                    raise InferenceCancelled() from exc
                raise RuntimeError("Inference worker exited without a result") from exc
            finally:
                with self._state:
                    detached = self._process is not process
                    dead = process.poll() is not None
                    if dead and not detached:
                        self._process = self._connection = None
                if detached or dead:
                    if transferred.is_set():
                        connection.close()
                        _reap(process)
                    else:
                        _reap(process, connection)

    def cancel(self):
        with self._state:
            self._generation += 1
            process, connection = self._process, self._connection
            self._process = self._connection = None
            if process is None:
                return False
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # The blocked caller reaps its process. Idle workers have no caller.
        if self._calls.acquire(blocking=False):
            try:
                connection.close()
                _reap(process)
            finally:
                self._calls.release()
        return True

    close = cancel


class InferenceControl:
    """Invalidate queued work and stop only a cancellation-capable active client."""

    def __init__(self):
        self._state = threading.RLock()
        self._epoch = 0
        self._active = {}

    @property
    def epoch(self):
        with self._state:
            return self._epoch

    @property
    def can_cancel(self):
        with self._state:
            return all(getattr(client, "supports_cancellation", False) is True
                       for client in self._active.values())

    def cancel(self, invalidate=None):
        with self._state:
            if not self.can_cancel:
                return False
            self._epoch += 1
            if invalidate is not None:
                invalidate()
            for client in {id(c): c for c in self._active.values()}.values():
                client.cancel()
            return True

    @contextmanager
    def context(self, lock, client, *, epoch=None, timings=None):
        from .recording_history import timed_inference_context
        if epoch is None:
            epoch = self.epoch
        with timed_inference_context(lock, timings,
                cancel_check=lambda: epoch != self.epoch) as timing, ExitStack() as scope:
            with self._state:
                if epoch != self._epoch:
                    raise InferenceCancelled()
                if getattr(client, "supports_cancellation", False) is True:
                    scope.enter_context(client.cancellation_scope())
                reservation = object()
                self._active[reservation] = client
            try:
                yield timing
                if epoch != self.epoch:
                    raise InferenceCancelled()
            finally:
                with self._state:
                    self._active.pop(reservation, None)


def _serve(fd, parent_pid):
    from .recording_history import client_receipt
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(1)
        os._exit(0)
    threading.Thread(target=watch_parent, name="spoke-worker-parent", daemon=True).start()
    connection = Connection(fd)
    client = None
    try:
        while True:
            try:
                request_id, config, method, args = connection.recv()
            except EOFError:
                return
            response = {"request_id": request_id, "worker": {
                "pid": os.getpid(), "source_root": str(Path(__file__).resolve().parents[1]),
                "python": sys.executable,
            }}
            try:
                if method not in {"prepare", "transcribe"}:
                    raise ValueError("Unsupported inference worker operation")
                with _request_activity(method) as activity:
                    response["worker"]["activity"] = activity
                    if client is None:
                        module, name, kwargs = config
                        client = getattr(importlib.import_module(module), name)(**kwargs)
                    response["value"] = getattr(client, method)(*args)
            except Exception as exc:
                response["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                response["effective"] = client_receipt(client) if client is not None else {}
                response["loaded"] = bool(getattr(client, "is_loaded", False))
            connection.send(response)
    finally:
        connection.close()
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    _serve(int(sys.argv[1]), int(sys.argv[2]))
