import io
import json
import logging
from unittest.mock import patch
import wave

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

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, enabled=True))

    record = spool.spool_capture(
        _wav_bytes(sample_count=3200),
        metadata={
            "pathway": "text",
            "requested_transcription_model": "whisperkit/medium.en",
            "transcription_route_state": "requested_pre_transcription",
        },
    )

    assert record is not None
    assert record.wav_path.read_bytes().startswith(b"RIFF")
    payload = json.loads(record.metadata_path.read_text())
    assert payload["schema"] == "spoke.audio_spool.capture.v1"
    assert payload["pathway"] == "text"
    assert payload["requested_transcription_model"] == "whisperkit/medium.en"
    assert payload["transcription_route_state"] == "requested_pre_transcription"
    assert payload["byte_count"] == record.byte_count == record.wav_path.stat().st_size
    assert payload["duration_seconds"] == pytest.approx(0.2)
    assert payload["wav_path"] == str(record.wav_path)
    assert len(payload["sha256"]) == 64


def test_audio_spool_overload_warns_without_deleting_evidence(tmp_path, caplog):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(
        AudioSpoolConfig(
            root=tmp_path,
            enabled=True,
            warn_recordings=2,
            warn_bytes=1,
        )
    )

    with caplog.at_level(logging.WARNING):
        records = [
            spool.spool_capture(_wav_bytes(sample_count=1000), metadata={"sequence": i})
            for i in range(3)
        ]

    assert all(record is not None for record in records)
    assert len(list(tmp_path.glob("*.wav"))) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert "AUDIO SPOOL OVERLOAD" in caplog.text
    assert "no evidence was deleted" in caplog.text


def test_audio_spool_disabled_does_not_write(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, enabled=False))

    record = spool.spool_capture(_wav_bytes(), metadata={"pathway": "text"})

    assert record is None
    assert list(tmp_path.iterdir()) == []


def test_audio_spool_collision_never_replaces_existing_evidence(tmp_path):
    from spoke.audio_spool import AudioSpool, AudioSpoolConfig

    original_wav = tmp_path / "occupied.wav"
    original_json = tmp_path / "occupied.json"
    original_wav.write_bytes(b"irreplaceable audio")
    original_json.write_text('{"irreplaceable": true}')
    spool = AudioSpool(AudioSpoolConfig(root=tmp_path, enabled=True))

    with patch(
        "spoke.audio_spool._capture_id",
        side_effect=["occupied", "fresh"],
    ):
        record = spool.spool_capture(_wav_bytes(), metadata={"pathway": "text"})

    assert record is not None
    assert record.capture_id == "fresh"
    assert original_wav.read_bytes() == b"irreplaceable audio"
    assert original_json.read_text() == '{"irreplaceable": true}'
    assert record.wav_path.name == "fresh.wav"
    assert record.metadata_path.name == "fresh.json"
