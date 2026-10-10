import io
import json
import wave
from pathlib import Path

import pytest


def _wav_bytes(sample_count: int = 1600, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * sample_count)
    return buf.getvalue()


def test_audio_spool_writes_wav_and_metadata(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(
        AudioSpoolConfig(
            root=tmp_path,
            max_recordings=10,
            max_bytes=10_000_000,
            max_age_seconds=3600,
        )
    )

    record = spool.spool_capture(
        _wav_bytes(sample_count=3200),
        metadata={
            "pathway": "text",
            "backend": "local",
            "model": "mlx-community/whisper-medium.en-mlx-8bit",
        },
    )

    assert record is not None
    assert record.wav_path.exists()
    assert record.metadata_path.exists()
    assert record.wav_path.read_bytes().startswith(b"RIFF")
    payload = json.loads(record.metadata_path.read_text())
    assert payload["schema"] == "spoke.audio_spool.capture.v1"
    assert payload["pathway"] == "text"
    assert payload["backend"] == "local"
    assert payload["model"] == "mlx-community/whisper-medium.en-mlx-8bit"
    assert payload["byte_count"] == record.byte_count == record.wav_path.stat().st_size
    assert payload["duration_seconds"] == pytest.approx(0.2)
    assert payload["wav_path"] == str(record.wav_path)
    assert len(payload["sha256"]) == 64


def test_audio_spool_prunes_by_count_and_total_bytes(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(
        AudioSpoolConfig(
            root=tmp_path,
            max_recordings=3,
            max_bytes=6500,
            max_age_seconds=3600,
        )
    )

    records = [
        spool.spool_capture(_wav_bytes(sample_count=1000), metadata={"sequence": i})
        for i in range(5)
    ]

    surviving_payloads = [
        json.loads(path.read_text())
        for path in sorted(Path(tmp_path).glob("*.json"))
    ]
    surviving_sequences = {payload["sequence"] for payload in surviving_payloads}
    surviving_bytes = sum(path.stat().st_size for path in Path(tmp_path).glob("*.wav"))

    assert records[-1] is not None
    assert len(surviving_payloads) <= 3
    assert surviving_bytes <= 6500
    assert max(surviving_sequences) == 4
    assert min(surviving_sequences) >= 2


def test_audio_spool_disabled_does_not_write(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, enabled=False))

    record = spool.spool_capture(_wav_bytes(), metadata={"pathway": "text"})

    assert record is None
    assert list(tmp_path.iterdir()) == []


def test_default_spool_preserves_recordings_beyond_old_black_box_limit(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    records = [spool.spool_capture(_wav_bytes(), metadata={"sequence": n}) for n in range(15)]

    assert all(record.wav_path.exists() for record in records)
    assert len(list(tmp_path.glob("*.json"))) == 15


def test_history_appends_attempts_without_overwriting_audio_or_first_result(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    record = spool.spool_capture(_wav_bytes(), metadata={"transcription_model": "first"})
    original_audio = record.wav_path.read_bytes()
    assert callable(getattr(spool, "start_attempt", None)), "Spool lacks transcription attempt lineage"
    first = spool.start_attempt(record.capture_id, requested={"backend": "local", "model": "first"})
    spool.finish_attempt(record.capture_id, first, text="Original text", effective={"model": "first"}, wall_seconds=1.0)
    before = spool.list_recordings()[0]["attempts"][0]
    second = spool.start_attempt(record.capture_id, requested={"backend": "local", "model": "second"})
    spool.finish_attempt(record.capture_id, second, text="Corrected text", effective={"model": "second"}, wall_seconds=2.0)

    item = spool.list_recordings()[0]
    assert item["attempts"][0] == before
    assert [attempt["text"] for attempt in item["attempts"]] == ["Original text", "Corrected text"]
    assert record.wav_path.read_bytes() == original_audio
    with pytest.raises(FileExistsError):
        spool.finish_attempt(record.capture_id, first, text="Overwritten", effective={}, wall_seconds=3.0)


def test_history_keeps_pending_blank_and_failed_attempts(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    record = spool.spool_capture(_wav_bytes())
    assert callable(getattr(spool, "start_attempt", None)), "Spool lacks visible attempt states"
    pending = spool.start_attempt(record.capture_id, requested={"model": "pending"})
    blank = spool.start_attempt(record.capture_id, requested={"model": "blank"})
    spool.finish_attempt(record.capture_id, blank, text="", effective={}, wall_seconds=1.0)
    failed = spool.start_attempt(record.capture_id, requested={"model": "failed"})
    spool.finish_attempt(record.capture_id, failed, text=None, error="decoder stopped", effective={}, wall_seconds=2.0)

    attempts = spool.list_recordings()[0]["attempts"]
    assert [(a["attempt_id"], a["status"]) for a in attempts] == [
        (pending, "pending"), (blank, "blank"), (failed, "failed")]


def test_history_audio_hash_is_verified_before_retranscription(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    record = spool.spool_capture(_wav_bytes())
    record.wav_path.write_bytes(_wav_bytes(3200))
    assert callable(getattr(spool, "read_recording_audio", None)), "History lacks audio identity validation"
    with pytest.raises(ValueError, match="SHA-256"):
        spool.read_recording_audio(record.capture_id)


def test_history_recording_is_protected_from_opted_in_spool_pruning(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, max_recordings=1))
    record = spool.spool_capture(_wav_bytes())
    assert callable(getattr(spool, "start_attempt", None)), "History lacks retention protection"
    attempt = spool.start_attempt(record.capture_id, requested={"model": "test"})
    spool.finish_attempt(record.capture_id, attempt, text="Keep this", effective={}, wall_seconds=1.0)
    spool.spool_capture(_wav_bytes(3200))
    assert record.wav_path.exists()
    assert len(spool.list_recordings()) == 2


def test_pending_capture_is_preserved_before_an_attempt_starts(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, max_bytes=1))
    record = spool.spool_capture(_wav_bytes(), metadata={"preserve_audio": True})
    assert record.wav_path.exists()


def test_corrupt_history_metadata_is_visible_not_fatal(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    record = spool.spool_capture(_wav_bytes())
    record.metadata_path.write_text("[]")
    rows = spool.list_recordings()
    assert rows[0]["status"] == "unverified"
    assert rows[0]["audio_available"] is True


def test_new_history_capture_does_not_trigger_legacy_pruning(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    legacy = spool.spool_capture(_wav_bytes())
    limited = AudioSpool(AudioSpoolConfig(root=tmp_path, max_bytes=1))
    limited.spool_capture(_wav_bytes(), metadata={"preserve_audio": True})
    assert legacy.wav_path.exists()


def test_malformed_terminal_result_is_not_presented_as_success(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path))
    capture = spool.spool_capture(_wav_bytes())
    attempt = spool.start_attempt(capture.capture_id, requested={"model": "test"})
    spool.finish_attempt(capture.capture_id, attempt, text=" ", effective=[], wall_seconds=1)
    assert spool.list_recordings()[0]["attempts"][0]["status"] == "unverified"
