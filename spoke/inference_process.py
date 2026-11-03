"""Owned, reusable inference process with explicit user cancellation."""

from contextlib import contextmanager
import importlib
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

    def call(self, method, *args):
        with self._state:
            generation = self._generation
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
            try:
                connection.send((request_id, self._config, method, args))
                response = connection.recv()
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
                    connection.close()
                    process.wait()

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
                process.wait()
            finally:
                self._calls.release()
        return True

    close = cancel


class InferenceControl:
    """Invalidate queued work and stop only a cancellation-capable active client."""

    def __init__(self):
        self._state = threading.RLock()
        self._epoch = 0
        self._active = None

    @property
    def epoch(self):
        with self._state:
            return self._epoch

    @property
    def can_cancel(self):
        with self._state:
            return self._active is None or getattr(self._active, "supports_cancellation", False) is True

    def cancel(self):
        with self._state:
            if not self.can_cancel:
                return False
            self._epoch += 1
            if self._active is not None:
                self._active.cancel()
            return True

    @contextmanager
    def context(self, lock, client, *, epoch=None, timings=None):
        from .recording_history import timed_inference_context
        if epoch is None:
            epoch = self.epoch
        with timed_inference_context(lock, timings) as timing:
            with self._state:
                if epoch != self._epoch:
                    raise InferenceCancelled()
                self._active = client
            try:
                yield timing
                if epoch != self.epoch:
                    raise InferenceCancelled()
            finally:
                with self._state:
                    self._active = None


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
                if client is None:
                    module, name, kwargs = config
                    client = getattr(importlib.import_module(module), name)(**kwargs)
                if method not in {"prepare", "transcribe"}:
                    raise ValueError("Unsupported inference worker operation")
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
