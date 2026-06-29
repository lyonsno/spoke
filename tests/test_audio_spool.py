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
