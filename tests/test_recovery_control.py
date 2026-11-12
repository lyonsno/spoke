"""Recovery ownership must not depend on a cooperative backend or child reap."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
import signal
import threading
import time

import pytest


@pytest.mark.parametrize("blocked_operation", ["send", "recv"])
def test_cancel_releases_caller_when_pipe_transfer_is_blocked(monkeypatch, blocked_operation):
    from spoke import inference_process as mod
    worker = mod.InferenceProcess("fixture", "Client", {})
    entered, release, finished = (threading.Event() for _ in range(3))
    process = MagicMock(pid=123456)
    process.poll.return_value = None
    process.wait.side_effect = release.wait
    connection = MagicMock()
    connection.poll.return_value = True
    def blocked(*args):
        entered.set()
        release.wait()
        raise EOFError()
    getattr(connection, blocked_operation).side_effect = blocked
    worker._process, worker._connection = process, connection
    monkeypatch.setattr(mod.os, "killpg", lambda *args: None)
    errors = []
    def call():
        try:
            worker.call("transcribe", b"retained")
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()
    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert entered.wait(1)
        worker.cancel()
        assert finished.wait(1), f"Cancelled caller still owns blocked {blocked_operation}"
        assert isinstance(errors[0], mod.InferenceCancelled)
    finally:
        release.set()
        thread.join(2)


def test_cancelled_call_releases_its_caller_before_child_reaping(monkeypatch):
    from spoke import inference_process as mod
    worker = mod.InferenceProcess("fixture", "Client", {})
    sent, unblock_recv, reaping, release_reap, finished = (threading.Event() for _ in range(5))
    process = MagicMock(pid=123456)
    process.poll.return_value = None
    process.wait.side_effect = lambda: (reaping.set(), release_reap.wait())
    connection = MagicMock()
    connection.send.side_effect = lambda _: sent.set()
    connection.poll.side_effect = lambda timeout: (time.sleep(timeout), False)[1]
    connection.recv.side_effect = lambda: (unblock_recv.wait(), (_ for _ in ()).throw(EOFError()))[-1]
    worker._process, worker._connection = process, connection
    killed = []
    monkeypatch.setattr(mod.os, "killpg", lambda *args: killed.append(args))
    errors = []
    def run():
        try:
            worker.call("transcribe", b"preserved")
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert sent.wait(1)
        assert worker.cancel()
        assert killed == [(123456, signal.SIGKILL)]
        assert finished.wait(1), "Cancelled caller still owns recovery until recv/reap cooperates"
        assert isinstance(errors[0], mod.InferenceCancelled)
    finally:
        unblock_recv.set()
        release_reap.set()
        thread.join(2)


def test_idle_cancel_does_not_wait_for_child_reaping_on_ui_thread(monkeypatch):
    from spoke import inference_process as mod
    worker = mod.InferenceProcess("fixture", "Client", {})
    release, done = threading.Event(), threading.Event()
    process = MagicMock(pid=123456)
    process.wait.side_effect = release.wait
    worker._process, worker._connection = process, MagicMock()
    monkeypatch.setattr(mod.os, "killpg", lambda *args: None)
    thread = threading.Thread(target=lambda: (worker.cancel(), done.set()))
    thread.start()
    try:
        assert done.wait(1), "Idle cancellation waits synchronously for native process death"
    finally:
        release.set()
        thread.join(2)


def test_queued_old_epoch_exits_before_a_stuck_active_call_releases_lock():
    from spoke.inference_process import InferenceControl, InferenceCancelled
    control, lock = InferenceControl(), threading.Lock()
    lock.acquire()
    epoch = control.epoch
    done = threading.Event()
    errors = []
    def queued():
        try:
            with control.context(lock, object(), epoch=epoch):
                pytest.fail("Cancelled queued inference ran")
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()
    thread = threading.Thread(target=queued)
    thread.start()
    try:
        control.cancel()
        assert done.wait(1), "Cancelled queued work still occupies the History worker"
        assert isinstance(errors[0], InferenceCancelled)
    finally:
        lock.release()
        thread.join(2)


def test_remote_request_is_registered_without_taking_gpu_lock(main_module, monkeypatch):
    from tests.test_delegate import _make_delegate
    from spoke.transcribe import TranscriptionClient
    from spoke.inference_process import InferenceControl
    d = _make_delegate(main_module, monkeypatch)
    d._local_inference_lock = threading.Lock()
    d._inference_control = InferenceControl()
    client = TranscriptionClient("http://fixture")
    client.supports_cancellation = True
    cancelled = []
    client.cancel = lambda: cancelled.append(True)
    client.cancellation_scope = nullcontext
    d._local_inference_lock.acquire()
    try:
        from spoke.inference_process import InferenceCancelled
        with pytest.raises(InferenceCancelled):
            with d._local_inference_context(client):
                assert d._inference_control.cancel()
                assert cancelled == [True], "Remote request bypassed cancellation ownership"
    finally:
        d._local_inference_lock.release()
        client.close()


def test_history_action_stops_http_and_retries_preserved_audio_on_same_queue(
        tmp_path, main_module, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    foundation = sys.modules["Foundation"]
    for name in ("NSIndexSet", "NSURL"):
        monkeypatch.setattr(foundation, name, MagicMock(), raising=False)
    monkeypatch.setitem(sys.modules, "spoke.house_window",
                        SimpleNamespace(HouseWindowShell=foundation.NSObject))
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from spoke.inference_process import InferenceControl
    from spoke.recording_history import run_retranscription
    from spoke.recording_history_window import RecordingHistoryWindow
    from spoke.transcribe import TranscriptionClient
    from tests.test_delegate import _make_delegate, _silent_wav
    entered, release = threading.Event(), threading.Event()
    requests, clients = [], []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            if len(requests) == 1:
                entered.set()
                release.wait()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"text":"Recovered exact recording."}')
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    client = TranscriptionClient(f"http://127.0.0.1:{server.server_port}",
                                 model="fixture-http", isolated=True)
    clients.append(client)
    control = InferenceControl()
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    audio = _silent_wav(1)
    capture = spool.spool_capture(audio, metadata={"preserve_audio": True})
    route = {"backend": "cloud", "model": "fixture-http"}
    attempt = spool.start_attempt(capture.capture_id, requested=route)
    delegate = _make_delegate(main_module, monkeypatch)
    delegate._inference_control = control
    delegate._history_model = route
    delegate._capture.is_recording = False
    delegate._local_inference_lock = threading.Lock()
    delegate._preview_session_token = 0
    def build(route):
        result = TranscriptionClient(f"http://127.0.0.1:{server.server_port}",
                                      model=route["model"], isolated=True)
        clients.append(result)
        return result
    delegate._build_history_client = build
    owner = RecordingHistoryWindow.alloc().init()
    owner._delegate, owner._spool = delegate, spool
    owner._selected = lambda: spool.read_recording_metadata(capture.capture_id)
    owner._model_routes = [{"route": route, "available": True}]
    owner._busy, owner._replay_generation = {capture.capture_id: {}}, 0
    owner._status = MagicMock()
    for name in ("_render_detail", "refresh_", "_update_controls"):
        setattr(owner, name, MagicMock())
    completed = threading.Event()
    owner.performSelectorOnMainThread_withObject_waitUntilDone_ = (
        lambda selector, payload, wait: completed.set() if selector == "replayFinished:" else None)
    executor = owner._replays = ThreadPoolExecutor(max_workers=1)
    original = executor.submit(run_retranscription, spool, capture.capture_id, attempt, route,
        lambda _: client, lambda c: control.context(None, c, epoch=0),
        cancel_check=lambda: owner._replay_generation != 0)
    try:
        assert entered.wait(10), "Real HTTP worker did not submit"
        old_pid = client._worker.pid
        owner.forceRetranscribe_(None)
        assert completed.wait(10), "History retry remained behind the cancelled request"
        original.result(timeout=1)
        record = spool.read_recording_metadata(capture.capture_id)
        attempts = record["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["status"] == "cancelled"
        assert attempts[1]["status"] == "success"
        assert attempts[1]["text"] == "Recovered exact recording."
        assert attempts[1]["requested"] == route
        receipt = attempts[1]["effective"]["worker_receipt"]
        assert receipt["pid"] != old_pid
        assert receipt["effective"]["model"] == "fixture-http"
        assert len(requests) == 2 and all(audio in body for body in requests)
        assert hashlib.sha256(spool.read_recording_audio(capture.capture_id)).hexdigest() == capture.sha256
        assert delegate._transcription_token == 1
    finally:
        release.set()
        for item in clients:
            item.close()
        executor.shutdown(wait=True)
        server.shutdown()
        server.server_close()
        serving.join(2)


def test_pending_microphone_is_visibly_not_recording(main_module, monkeypatch):
    from concurrent.futures import Future
    from tests.test_delegate import _make_delegate
    d = _make_delegate(main_module, monkeypatch)
    d._capture.start_async.side_effect = None
    d._capture.start_async.return_value = Future()
    d._capture.stop.return_value = b""
    d._on_hold_start()
    d._overlay.set_text.assert_called_with("Starting microphone - not recording yet")
    d._on_hold_end()
    assert "No audio was recorded" in d._overlay.flash_notice.call_args.args[0]
