import threading

import pytest


def test_owned_process_can_be_cancelled_and_reused_without_old_result(tmp_path):
    from spoke.inference_process import InferenceCancelled, InferenceProcess

    worker = InferenceProcess("tests.fixtures.inference_client", "FixtureClient", {})
    errors = []
    marker = tmp_path / "entered"

    def stalled():
        try:
            worker.call("transcribe", str(marker).encode())
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=stalled)
    thread.start()
    try:
        assert _wait_for(marker.exists), "Fixture never entered its blocking operation"
        old_pid = worker.pid
        assert worker.cancel() is True
        thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], InferenceCancelled)
        result = worker.call("transcribe", b"recover")
        assert result["value"] == "recovered"
        assert result["effective"]["model"] == "fixture-model"
        assert result["worker"]["pid"] != old_pid
        from pathlib import Path
        assert result["worker"]["source_root"] == str(Path(__file__).resolve().parents[1])
    finally:
        worker.close()
        thread.join(5)


def _wait_for(predicate):
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_cancelled_whisper_does_not_start_automatic_fallback(monkeypatch):
    from spoke.inference_process import InferenceCancelled
    from spoke.transcribe_local import LocalTranscriptionClient

    client = LocalTranscriptionClient(model="fixture", isolated=True)
    monkeypatch.setattr(client._worker, "call", lambda *args, **kwargs: (_ for _ in ()).throw(InferenceCancelled()))
    with pytest.raises(InferenceCancelled):
        client.transcribe(b"exact audio")
    assert client._last_receipt["status"] == "cancelled"


def test_cancel_rejects_waiting_old_epoch_before_model_runs():
    from spoke.inference_process import InferenceCancelled, InferenceControl

    control = InferenceControl()
    epoch = control.epoch
    assert control.cancel() is True
    with pytest.raises(InferenceCancelled):
        with control.context(threading.Lock(), object(), epoch=epoch):
            pytest.fail("Superseded queued work ran")


def test_cancel_does_not_claim_to_interrupt_unmanaged_inference():
    from spoke.inference_process import InferenceControl

    control = InferenceControl()
    with control.context(threading.Lock(), object()):
        assert control.can_cancel is False
        assert control.cancel() is False


def test_worker_reuses_prepared_process_and_returns_failure_identity():
    from spoke.inference_process import InferenceProcess

    worker = InferenceProcess("tests.fixtures.inference_client", "FixtureClient", {})
    try:
        prepared = worker.call("prepare")
        success = worker.call("transcribe", b"recover")
        failure = worker.call("transcribe", b"fail")
        assert prepared["worker"]["pid"] == success["worker"]["pid"] == failure["worker"]["pid"]
        assert failure["error"] == "ValueError: fixture failure"
        assert failure["effective"]["model"] == "fixture-model"
    finally:
        worker.close()


def test_control_rejects_a_result_that_finishes_after_cancel():
    from spoke.inference_process import InferenceCancelled, InferenceControl
    from types import SimpleNamespace

    control = InferenceControl()
    cancelled = []
    from contextlib import nullcontext
    client = SimpleNamespace(supports_cancellation=True, cancel=lambda: cancelled.append(True),
                             cancellation_scope=nullcontext)
    with pytest.raises(InferenceCancelled):
        with control.context(threading.Lock(), client):
            assert control.cancel() is True
    assert cancelled == [True]


def test_cancel_before_worker_submission_cannot_resurrect_old_work():
    from spoke.inference_process import InferenceCancelled, InferenceControl, InferenceProcess
    from spoke.transcribe_local import LocalTranscriptionClient

    control = InferenceControl()
    client = LocalTranscriptionClient(model="fixture", isolated=True)
    client._worker = InferenceProcess("tests.fixtures.inference_client", "FixtureClient", {})
    try:
        with pytest.raises(InferenceCancelled):
            with control.context(threading.Lock(), client):
                control.cancel()
                with pytest.raises(InferenceCancelled):
                    client.transcribe(b"recover")
        assert client._worker.pid is None
    finally:
        client.close()


def _run_activity_request(monkeypatch, *, method="transcribe", platform="darwin",
                          backend_error=False, begin_error=False, nil_token=False,
                          end_error=False, foundation_missing=False):
    import sys
    from types import SimpleNamespace
    import spoke.inference_process as worker_module

    events = []
    active = []
    token = object()

    class ProcessInfo:
        def beginActivityWithOptions_reason_(self, options, reason):
            assert options == 0x1234
            assert reason
            events.append("begin")
            if begin_error:
                raise RuntimeError("activity service unavailable")
            if nil_token:
                return None
            active.append(token)
            return token

        def endActivity_(self, actual):
            assert actual is token
            events.append("end")
            active.remove(token)
            if end_error:
                raise RuntimeError("activity end failed")

    class Client:
        _model = "activity-fixture"
        is_loaded = True

        def transcribe(self, audio):
            events.append("transcribe")
            assert audio == b"preserved audio"
            if platform == "darwin" and not begin_error and not nil_token and not foundation_missing:
                assert active == [token], "Transcription ran without its user-initiated activity"
            if backend_error:
                raise ValueError("fixture decode failed")
            return "heard"

        def prepare(self):
            events.append("prepare")
            assert not active

        def close(self):
            assert not active
            events.append("close")

    requests = iter([("activity-request", ("spoke_activity_fixture", "Client", {}),
                      method, (b"preserved audio",) if method == "transcribe" else ())])
    responses = []

    def receive():
        try:
            return next(requests)
        except StopIteration:
            raise EOFError from None

    connection = SimpleNamespace(recv=receive, send=responses.append, close=lambda: None)
    info = ProcessInfo()
    foundation = SimpleNamespace(
        NSProcessInfo=SimpleNamespace(processInfo=lambda: info),
        NSActivityUserInitiatedAllowingIdleSystemSleep=0x1234,
    )
    monkeypatch.setitem(sys.modules, "Foundation", None if foundation_missing else foundation)
    monkeypatch.setitem(sys.modules, "spoke_activity_fixture", SimpleNamespace(Client=Client))
    monkeypatch.setattr(worker_module.sys, "platform", platform)
    monkeypatch.setattr(worker_module, "Connection", lambda fd: connection)
    monkeypatch.setattr(worker_module.threading, "Thread",
                        lambda **kwargs: SimpleNamespace(start=lambda: None))
    worker_module._serve(17, 1)
    assert len(responses) == 1
    return events, responses[0]


def test_explicit_transcription_owns_activity_until_backend_finishes(monkeypatch):
    events, result = _run_activity_request(monkeypatch)
    assert result.get("value") == "heard", result
    assert events == ["begin", "transcribe", "end", "close"]
    assert result["worker"]["activity"]["status"] == "ended"


def test_failed_transcription_releases_activity_and_preserves_error(monkeypatch):
    events, result = _run_activity_request(monkeypatch, backend_error=True)
    assert result["error"] == "ValueError: fixture decode failed"
    assert events == ["begin", "transcribe", "end", "close"]
    assert result["worker"]["activity"]["status"] == "ended"


@pytest.mark.parametrize("platform,method,status", [
    ("darwin", "prepare", "not_requested"),
    ("linux", "transcribe", "not_applicable"),
])
def test_no_activity_for_idle_preparation_or_other_platforms(monkeypatch, platform, method, status):
    events, result = _run_activity_request(monkeypatch, platform=platform, method=method)
    assert "error" not in result
    assert events == [method, "close"]
    assert result["worker"]["activity"]["status"] == status


@pytest.mark.parametrize("options", [{"begin_error": True}, {"nil_token": True}])
def test_activity_admission_failure_does_not_discard_audio(monkeypatch, options):
    events, result = _run_activity_request(monkeypatch, **options)
    assert result.get("value") == "heard"
    assert events == ["begin", "transcribe", "close"]
    assert result["worker"]["activity"]["status"] == "unavailable"
    assert result["worker"]["activity"]["error"]


def test_unavailable_native_bridge_does_not_discard_audio(monkeypatch):
    events, result = _run_activity_request(monkeypatch, foundation_missing=True)
    assert result["value"] == "heard"
    assert events == ["transcribe", "close"]
    assert result["worker"]["activity"]["status"] == "unavailable"
    assert result["worker"]["activity"]["error"]


@pytest.mark.parametrize("backend_error", [False, True])
def test_activity_release_failure_does_not_replace_backend_outcome(monkeypatch, backend_error):
    events, result = _run_activity_request(monkeypatch, end_error=True, backend_error=backend_error)
    if backend_error:
        assert result["error"] == "ValueError: fixture decode failed"
    else:
        assert result["value"] == "heard"
    assert events == ["begin", "transcribe", "end", "close"]
    assert result["worker"]["activity"]["status"] == "end_failed"
    assert "activity end failed" in result["worker"]["activity"]["error"]

