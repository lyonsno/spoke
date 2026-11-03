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

