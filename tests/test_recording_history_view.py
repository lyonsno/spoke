from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

class Text:
    def __init__(self, value=""):
        self.value = value

    def setStringValue_(self, value):
        self.value = str(value)

    def stringValue(self):
        return self.value


def view(record):
    from spoke.recording_history_window import RecordingHistoryWindow
    owner = RecordingHistoryWindow.alloc().init()
    route = {"backend": "local", "model": "chosen"}
    owner._delegate = SimpleNamespace(_history_model=route, _history_live_busy=lambda: False,
        _history_can_take_over=lambda: True, _local_inference_context=lambda client: nullcontext(),
        _build_history_client=MagicMock())
    owner._records = owner._filtered = [deepcopy(record)]
    owner._selected_id = record["capture_id"]
    owner._attempt_id = record["attempts"][0]["attempt_id"]
    owner._busy, owner._replay_generation = {}, 0
    owner._replay_outcomes = {}
    owner._loading_partial = owner._load_failed = False
    owner._load_generation, owner._record_overrides = 7, {}
    owner._refreshing = False
    owner._count, owner._table, owner._search = MagicMock(), MagicMock(), Text()
    owner._spool = SimpleNamespace(start_attempt=MagicMock(return_value="new-attempt"))
    owner._model_routes = [{"route": route, "available": True}]
    owner._replays = MagicMock()
    owner.refresh_ = MagicMock()
    owner._status = Text("Retention: manual")
    for name in ("_title", "_meta", "_attempts", "_text", "_route", "_copy", "_insert", "_play", "_trash", "_retry", "_force_retry"):
        setattr(owner, name, MagicMock())
    owner.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock(
        side_effect=lambda selector, payload, wait: getattr(owner, selector.replace(":", "_"))(payload))
    return owner


def original():
    return {"capture_id": "capture", "audio_available": True, "pathway": "text",
        "attempts": [{"attempt_id": "original", "kind": "live", "status": "cancelled",
                      "error": "Stopped", "requested": {"model": "preferred"}}]}


def test_starting_replay_cannot_render_back_to_cancelled_original():
    owner = view(original())
    owner._start_retranscription()
    owner._render_detail()
    assert owner._attempt_id == "new-attempt", "Stale snapshot reset a new durable replay to Original1"
    assert owner._attempt()["requested"]["model"] == "chosen"


def test_selected_unrequested_record_does_not_claim_another_replay_is_queued():
    owner = view(original())
    owner._busy = {"different-record": {"attempt_id": "other", "phase": "transcribing", "generation": 0}}
    owner._status.setStringValue_("Re-transcription queued")
    owner._render_detail()
    assert owner._status.stringValue() == "Re-transcribing another recording"


def test_phase_callbacks_are_exact_attempt_and_generation_bound():
    owner = view(original())
    owner._start_retranscription()
    for attempt, generation in (("foreign", 0), ("new-attempt", 1)):
        owner.replayPhase_({"capture_id": "capture", "attempt_id": attempt,
                            "generation": generation, "phase": "transcribing"})
        assert owner._busy["capture"]["phase"] == "queued"
    owner.replayPhase_({"capture_id": "capture", "attempt_id": "new-attempt",
                        "generation": 0, "phase": "transcribing"})
    assert owner._status.stringValue() == "Re-transcribing..."


def test_terminal_result_updates_selected_record_before_slow_library_scan():
    owner = view(original())
    owner._start_retranscription()
    latest = original()
    latest["attempts"].append({"attempt_id": "new-attempt", "kind": "retranscription",
        "requested": {"model": "chosen"}, "status": "success", "text": "recovered"})
    owner.replayFinished_({"capture_id": "capture", "attempt_id": "new-attempt",
                          "generation": 0, "record": latest})
    assert owner._attempt()["text"] == "recovered"
    assert not owner._busy
    owner.loaded_({"records": [original()], "complete": True, "load_generation": 7})
    assert owner._attempt()["text"] == "recovered", "Older full scan undid a durable terminal result"
    owner._load_generation = 8
    owner.loaded_({"records": [original()], "complete": False, "load_generation": 8})
    assert owner._attempt()["text"] == "recovered", "Partial scan hid a verified terminal result"
    owner.loaded_({"records": [latest], "complete": True, "load_generation": 8})
    assert not owner._record_overrides
    assert owner._attempt()["text"] == "recovered"


def test_stale_completion_cannot_clear_a_newer_job():
    owner = view(original())
    owner._start_retranscription()
    for attempt, generation in (("old-attempt", 0), ("new-attempt", -1)):
        owner.replayFinished_({"capture_id": "capture", "attempt_id": attempt,
                              "generation": generation, "record": original()})
        assert owner._busy["capture"]["attempt_id"] == "new-attempt"
        assert owner._attempt_id == "new-attempt"


def test_render_filter_and_refresh_results_never_request_a_replay():
    owner = view(original())
    owner._render_detail()
    owner._filter()
    owner.loaded_({"records": [original()], "complete": True, "load_generation": 7})
    owner._spool.start_attempt.assert_not_called()
    owner._replays.submit.assert_not_called()


def test_submit_failure_does_not_leave_a_phantom_busy_job():
    owner = view(original())
    owner._spool.finish_attempt = MagicMock()
    owner._replays.submit.side_effect = RuntimeError("executor closed")
    owner._start_retranscription()
    assert not owner._busy
    assert "Could not queue" in owner._status.stringValue()
    assert owner._spool.finish_attempt.call_args.kwargs["evidence"]["phase"] == "queue_submission"


def test_worker_reports_transcribing_and_returns_current_metadata(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    attempt = spool.start_attempt(capture.capture_id, requested={"model": "preferred"})
    spool.finish_attempt(capture.capture_id, attempt, text=None, error="Stopped", cancelled=True,
                         effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    observed = []
    class Client:
        def transcribe(self, audio):
            observed.append(owner._status.stringValue())
            return "current result"
    owner._delegate._build_history_client.return_value = Client()
    owner._start_retranscription()
    owner._replays.submit.call_args.args[0]()
    assert observed == ["Re-transcribing..."]
    assert owner._attempt()["text"] == "current result"
    assert "finished" in owner._status.stringValue()


def test_completion_cannot_claim_success_for_a_different_selected_record():
    owner = view(original())
    owner._start_retranscription()
    other = original()
    other["capture_id"] = "other-record"
    owner._records.append(other)
    owner._selected_id, owner._attempt_id = "other-record", "original"
    owner._filter()
    latest = original()
    latest["attempts"].append({"attempt_id": "new-attempt", "kind": "retranscription",
        "requested": {"model": "chosen"}, "status": "success", "text": "recovered"})
    owner.replayFinished_({"capture_id": "capture", "attempt_id": "new-attempt",
                          "generation": 0, "record": latest})
    owner._render_detail()
    assert owner._selected_id == "other-record"
    assert "finished" not in owner._status.stringValue()
    owner._selected_id, owner._attempt_id = "capture", "new-attempt"
    owner._render_detail()
    assert owner._attempt()["text"] == "recovered"
    assert "finished" in owner._status.stringValue()
    owner._attempt_id = "original"
    owner._render_attempt()
    assert "finished" not in owner._status.stringValue()


def test_rejected_executor_installs_failed_attempt_without_archive_publication(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    original_id = spool.start_attempt(capture.capture_id, requested={"model": "preferred"})
    spool.finish_attempt(capture.capture_id, original_id, text="original words", effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    owner._replays = ThreadPoolExecutor(max_workers=1)
    owner._replays.shutdown()
    owner._refreshing = True
    owner._start_retranscription()
    assert not owner._busy
    assert owner._attempt()["status"] == "failed"
    assert "Could not queue" in owner._text.setString_.call_args.args[0]
    assert "Could not queue" in owner._status.stringValue()
    owner._attempt_id = original_id
    assert owner._attempt()["text"] == "original words"


def test_rejected_worker_start_cannot_revive_on_a_later_retry(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    original_id = spool.start_attempt(capture.capture_id, requested={})
    spool.finish_attempt(capture.capture_id, original_id, text="original", effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    owner._replays = ThreadPoolExecutor(max_workers=1)
    calls = []
    class Client:
        def transcribe(self, audio):
            calls.append(audio)
            return "authorized replay"
    owner._delegate._build_history_client.return_value = Client()
    try:
        with monkeypatch.context() as patch:
            def fail_start(thread):
                raise RuntimeError("cannot start worker")
            patch.setattr(threading.Thread, "start", fail_start)
            owner._start_retranscription()
        rejected = owner._attempt_id
        owner._start_retranscription()
    finally:
        owner._replays.shutdown(wait=True)
    assert calls == [capture.wav_path.read_bytes()], "Rejected work entered the backend on a later retry"
    attempts = spool.read_recording_metadata(capture.capture_id)["attempts"]
    assert next(a for a in attempts if a["attempt_id"] == rejected)["status"] == "failed"
    assert attempts[-1]["text"] == "authorized replay"


def test_metadata_read_failure_keeps_exact_replay_selected_until_reconciliation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    original_id = spool.start_attempt(capture.capture_id, requested={})
    spool.finish_attempt(capture.capture_id, original_id, text="original", effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    owner._replays = ThreadPoolExecutor(max_workers=1)
    owner._replays.shutdown()
    with monkeypatch.context() as patch:
        def unavailable(capture_id):
            raise OSError("metadata temporarily unreadable")
        patch.setattr(spool, "read_recording_metadata", unavailable)
        owner._start_retranscription()
    durable = spool.read_recording_metadata(capture.capture_id)
    replay_id = durable["attempts"][-1]["attempt_id"]
    assert owner._attempt_id == replay_id
    assert owner._attempt()["status"] == "unverified"
    assert "Could not queue" in owner._status.stringValue()
    owner._load_generation = 8
    owner.loaded_({"records": [durable], "complete": True, "load_generation": 8})
    assert owner._attempt_id == replay_id
    assert owner._attempt()["status"] == "failed"
    assert owner._status.stringValue() == "Re-transcription failed; original retained"
    owner._attempt_id = original_id
    owner._render_attempt()
    assert owner._attempt()["text"] == "original"
    assert "Re-transcription" not in owner._status.stringValue()


def test_callbacks_without_required_identity_cannot_mutate_current_work():
    owner = view(original())
    owner._start_retranscription()
    owner.replayFinished_({"capture_id": "capture", "attempt_id": "new-attempt"})
    assert owner._busy["capture"]["attempt_id"] == "new-attempt"
    owner.loaded_({"records": [], "complete": True})
    assert owner._selected_id == "capture"
    owner.replayPhase_({"attempt_id": "new-attempt", "generation": 0, "phase": "transcribing"})
    owner.replayPhase_({"capture_id": "capture", "attempt_id": "new-attempt", "generation": 0})
    assert owner._busy["capture"]["phase"] == "queued"


def test_scan_start_failure_releases_refresh_and_allows_the_next_scan(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from spoke.recording_history_window import RecordingHistoryWindow
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    original_id = spool.start_attempt(capture.capture_id, requested={})
    spool.finish_attempt(capture.capture_id, original_id, text="original", effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    owner._replays = ThreadPoolExecutor(max_workers=1)
    owner.refresh_ = lambda sender: RecordingHistoryWindow.refresh_(owner, sender)
    try:
        with monkeypatch.context() as patch:
            def unavailable(thread):
                raise RuntimeError("cannot start worker")
            patch.setattr(threading.Thread, "start", unavailable)
            owner._start_retranscription()
        assert not owner._refreshing
        assert owner._attempt()["status"] == "failed"
        generation = owner._load_generation
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", lambda thread: thread.run())
            owner.refresh_(None)
        assert owner._load_generation == generation + 1
        assert not owner._refreshing
    finally:
        owner._replays.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("queue_rejected", [True, False])
def test_unsaved_ended_attempt_keeps_its_error_after_scans_and_later_retry(tmp_path, monkeypatch, queue_rejected):
    from concurrent.futures import ThreadPoolExecutor
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    from tests.test_audio_spool import _wav_bytes
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    original_id = spool.start_attempt(capture.capture_id, requested={})
    spool.finish_attempt(capture.capture_id, original_id, text="original", effective={}, wall_seconds=0)
    owner = view(spool.read_recording_metadata(capture.capture_id))
    owner._spool = spool
    owner._delegate._build_history_client.return_value = SimpleNamespace(transcribe=lambda audio: "new result")
    if queue_rejected:
        owner._replays = ThreadPoolExecutor(max_workers=1)
        owner._replays.shutdown()
    with monkeypatch.context() as patch:
        def unwritable(*args, **kwargs):
            raise OSError("result cannot be saved")
        patch.setattr(spool, "finish_attempt", unwritable)
        owner._start_retranscription()
        if not queue_rejected:
            owner._replays.submit.call_args.args[0]()
    unsaved = owner._attempt_id
    assert owner._attempt()["status"] == "unverified"
    assert not owner._busy
    owner._load_generation += 1
    owner.loaded_({"records": spool.list_recordings(), "complete": True,
                   "load_generation": owner._load_generation})
    assert owner._attempt_id == unsaved
    assert owner._attempt()["status"] == "unverified"
    assert owner._attempt()["error"]
    owner._replays = MagicMock()
    owner._start_retranscription()
    owner._replays.submit.call_args.args[0]()
    assert owner._attempt()["text"] == "new result"
    owner._load_generation += 1
    owner.loaded_({"records": spool.list_recordings(), "complete": True,
                   "load_generation": owner._load_generation})
    owner._attempt_id = unsaved
    assert owner._attempt()["status"] == "unverified"
    assert owner._attempt()["error"]


@pytest.mark.parametrize("payload", [{"load_generation": 7}, {"load_generation": 7, "records": []}])
def test_incomplete_current_scan_envelope_cannot_discard_partial_selection(payload):
    owner = view(original())
    owner._refreshing = owner._loading_partial = True
    owner.loaded_(payload)
    assert owner._selected_id == "capture"
    assert owner._refreshing and owner._loading_partial
