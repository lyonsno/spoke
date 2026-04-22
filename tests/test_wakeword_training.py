import json
import wave
import zipfile
from pathlib import Path

from spoke.wakeword_training import export_training_packs


def _write_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16)


def _write_manifest(batch_dir: Path) -> None:
    rows = [
        {
            "text": "tessera",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 16000,
            "num_samples": 16,
            "relative_path": "0001_tessera__af-heart__sidecar.wav",
        },
        {
            "text": "tessera",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "am_adam",
            "sample_rate": 16000,
            "num_samples": 16,
            "relative_path": "0002_tessera__am-adam__sidecar.wav",
        },
        {
            "text": "alpha",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 16000,
            "num_samples": 16,
            "relative_path": "0003_alpha__af-heart__sidecar.wav",
        },
    ]
    for row in rows:
        _write_test_wav(batch_dir / row["relative_path"])
    (batch_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_export_training_packs_groups_samples_by_keyword(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_manifest(batch_dir)

    packs = export_training_packs(batch_dir, tmp_path / "training")

    assert [pack.keyword for pack in packs] == ["alpha", "tessera"]

    index = json.loads((tmp_path / "training" / "manifest.json").read_text())
    assert [entry["keyword"] for entry in index["packs"]] == ["alpha", "tessera"]
    assert [entry["sample_count"] for entry in index["packs"]] == [1, 2]

    tessera_dir = tmp_path / "training" / "tessera"
    copied = sorted(path.name for path in (tessera_dir / "samples").glob("*.wav"))
    assert copied == [
        "0001_tessera__af-heart__sidecar.wav",
        "0002_tessera__am-adam__sidecar.wav",
    ]

    tessera_manifest = json.loads((tessera_dir / "manifest.json").read_text())
    assert tessera_manifest["keyword"] == "tessera"
    assert [row["voice"] for row in tessera_manifest["records"]] == ["af_heart", "am_adam"]

    with zipfile.ZipFile(tessera_dir / "tessera-samples.zip") as archive:
        assert sorted(archive.namelist()) == copied


def test_export_training_packs_can_filter_keywords(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_manifest(batch_dir)

    packs = export_training_packs(
        batch_dir,
        tmp_path / "training",
        keywords=["tessera"],
    )

    assert [pack.keyword for pack in packs] == ["tessera"]
    assert (tmp_path / "training" / "tessera").exists()
    assert not (tmp_path / "training" / "alpha").exists()


def test_export_training_packs_removes_stale_samples_on_rerun(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_manifest(batch_dir)

    training_dir = tmp_path / "training"
    export_training_packs(batch_dir, training_dir, keywords=["tessera"])

    batch_rows = [
        {
            "text": "tessera",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 16000,
            "num_samples": 16,
            "relative_path": "0001_tessera__af-heart__sidecar.wav",
        }
    ]
    (batch_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in batch_rows)
    )

    export_training_packs(batch_dir, training_dir, keywords=["tessera"])

    copied = sorted(path.name for path in (training_dir / "tessera" / "samples").glob("*.wav"))
    assert copied == ["0001_tessera__af-heart__sidecar.wav"]
