import importlib.util
from contextlib import nullcontext

from spoke.audio_spool import AudioSpool, AudioSpoolConfig
from tests.test_audio_spool import _wav_bytes


def _service():
    assert importlib.util.find_spec("spoke.recording_history"), "Independent re-transcription service is absent"
    from spoke.recording_history import run_retranscription
    return run_retranscription


def test_retranscription_uses_exact_audio_and_requested_route(tmp_path):
    run = _service()
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    route = {"backend": "local", "model": "model-two"}
    attempt = spool.start_attempt(capture.capture_id, requested=route, kind="retranscription")
    received = []
    class Client:
        _model = "model-two"
        def transcribe(self, audio):
            received.append(audio)
            return "A new version"
        def close(self):
            received.append("closed")
    run(spool, capture.capture_id, attempt, route, lambda r: Client(), lambda c: nullcontext())
    result = spool.list_recordings()[0]["attempts"][0]
    assert received == [capture.wav_path.read_bytes(), "closed"]
    assert result["text"] == "A new version"
    assert result["requested"] == route
    assert result["effective"]["model"] == "model-two"


def test_retranscription_failure_before_client_is_durable(tmp_path):
    run = _service()
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    route = {"backend": "local", "model": "unavailable"}
    attempt = spool.start_attempt(capture.capture_id, requested=route, kind="retranscription")
    def factory(route):
        raise RuntimeError("Requested model is not installed")
    run(spool, capture.capture_id, attempt, route, factory, lambda c: nullcontext())
    result = spool.list_recordings()[0]["attempts"][0]
    assert result["status"] == "failed"
    assert "not installed" in result["error"]
    assert result["evidence"]["phase"] == "client_creation"
    assert result["effective"] == {}


def test_retranscription_integrity_failure_never_loads_model(tmp_path):
    run = _service()
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    route = {"backend": "local", "model": "test"}
    attempt = spool.start_attempt(capture.capture_id, requested=route, kind="retranscription")
    capture.wav_path.write_bytes(b"damaged")
    called = []
    run(spool, capture.capture_id, attempt, route, lambda r: called.append(r), lambda c: nullcontext())
    assert called == []
    result = spool.list_recordings()[0]["attempts"][0]
    assert result["status"] == "failed"
    assert result["evidence"]["phase"] == "audio_validation"


def test_history_list_preview_keeps_original_after_successful_retry():
    from spoke.recording_history import first_line
    assert first_line({"attempts": [
        {"kind": "live", "status": "success", "text": "Original words"},
        {"kind": "retranscription", "status": "success", "text": "Different words"},
    ]}) == "Original words"


def test_history_preview_uses_only_first_physical_line(tmp_path):
    from spoke.recording_history import first_line
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    attempt = spool.start_attempt(capture.capture_id, requested={"model": "original"})
    text = "First  line\nSecond line\n\nFinal paragraph"
    spool.finish_attempt(capture.capture_id, attempt, text=text, effective={}, wall_seconds=1)
    record = spool.list_recordings()[0]
    assert first_line(record) == "First line"
    assert record["attempts"][0]["text"] == text


def test_empty_error_still_marks_attempt_failed(tmp_path):
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    attempt = spool.start_attempt(capture.capture_id, requested={})
    spool.finish_attempt(capture.capture_id, attempt, text=None, error="", effective={}, wall_seconds=0)
    assert spool.list_recordings()[0]["attempts"][0]["status"] == "failed"


def test_delivery_events_are_chronological_not_uuid_order(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from spoke import audio_spool
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    attempt = spool.start_attempt(capture.capture_id, requested={})
    ids = iter(["z", "z-temp", "a", "a-temp"])
    monkeypatch.setattr(audio_spool.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    spool.record_delivery(capture.capture_id, attempt, state="insert_requested")
    spool.record_delivery(capture.capture_id, attempt, state="clipboard_restored")
    assert [d["state"] for d in spool.list_recordings()[0]["attempts"][0]["deliveries"]] == [
        "insert_requested", "clipboard_restored"]


def test_command_delivery_labels_do_not_overstate_request_success():
    from spoke.recording_history import delivery_label

    assert delivery_label({"deliveries": [{"state": "command_requested"}]}) == (
        "Delivery: Assistant request started; completion unverified"
    )
    assert delivery_label({"deliveries": [{"state": "command_response_started"}]}) == (
        "Delivery: Assistant response started; completion unverified"
    )
    assert delivery_label({"deliveries": [{"state": "command_failed"}]}) == (
        "Delivery: Assistant request failed; retained in history"
    )


def test_history_prompt_snapshot_is_private_and_does_not_reread_source(tmp_path):
    from types import SimpleNamespace
    from spoke.recording_history import client_receipt
    from spoke.transcription_prompt import TranscriptionPromptProvider

    path = tmp_path / "prompt.txt"
    path.write_text("Private original context")
    provider = TranscriptionPromptProvider(path=path, include_builtin=False)
    prompt = provider.resolve()
    receipt = prompt.receipt(supported=True, payload_constructed=True,
                             submission_attempted=True, runtime_accepted=True)
    client = SimpleNamespace(_prompt_provider=provider, _last_prompt_receipt=receipt)
    path.write_text("Later context")
    history = client_receipt(client)
    assert history["prompt_snapshot"]["text"] == "Private original context"
    assert history["prompt_snapshot"]["sha256"] == receipt["sha256"]
    assert "text" not in receipt


def test_history_omits_stale_unmatched_prompt_snapshot():
    from types import SimpleNamespace
    from spoke.recording_history import client_receipt
    from spoke.transcription_prompt import TranscriptionPromptProvider

    provider = TranscriptionPromptProvider(inline="Earlier context", include_builtin=False)
    provider.resolve()
    assert "prompt_snapshot" not in client_receipt(SimpleNamespace(_prompt_provider=provider))


def test_history_action_buttons_have_legible_symbols_and_click_targets():
    from spoke.recording_history_window import AK, NSMakeRect, RecordingHistoryWindow

    owner = RecordingHistoryWindow.alloc().init()
    parent = AK.NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 100))
    buttons = [
        owner._tool(parent, symbol, tooltip, action, index * 40, 0, 0)
        for index, (symbol, tooltip, action) in enumerate((
            ("play.fill", "Play original audio", "playAudio:"),
            ("doc.on.doc", "Copy selected transcript", "copyText:"),
            ("arrow.up.doc", "Insert selected transcript", "insertText:"),
            ("trash", "Move recording and attempts to Trash", "trashRecording:"),
        ))
    ]

    for button in buttons:
        frame = button.frame()
        image = button.image()
        assert button.bezelStyle() == AK.NSBezelStyleCircular
        assert button.controlSize() == AK.NSControlSizeLarge
        assert frame.size.width >= 60
        assert frame.size.height >= 60
        assert image.size().width >= 42
        assert image.size().height >= 42
        assert image.symbolConfiguration().pointSize() >= 32
