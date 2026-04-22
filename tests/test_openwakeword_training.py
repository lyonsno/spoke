import json
import sys
import types
import wave
from pathlib import Path

import numpy as np

from spoke.openwakeword_training import (
    _clip_scores,
    _json_ready,
    _read_wav_pcm16,
    build_keyword_dataset,
    recommend_total_length,
    suggest_threshold,
)


def _write_test_wav(path: Path, nframes: int, *, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * nframes)


def _write_manifest(batch_dir: Path) -> None:
    rows = [
        {
            "text": "tessera",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 16000,
            "num_samples": 12000,
            "relative_path": "0001_tessera__af-heart__sidecar.wav",
        },
        {
            "text": "tessera",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "am_adam",
            "sample_rate": 16000,
            "num_samples": 14000,
            "relative_path": "0002_tessera__am-adam__sidecar.wav",
        },
        {
            "text": "alpha",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 16000,
            "num_samples": 11000,
            "relative_path": "0003_alpha__af-heart__sidecar.wav",
        },
        {
            "text": "omega",
            "backend": "sidecar",
            "model": "kokoro",
            "voice": "am_adam",
            "sample_rate": 16000,
            "num_samples": 15000,
            "relative_path": "0004_omega__am-adam__sidecar.wav",
        },
    ]
    for row in rows:
        _write_test_wav(batch_dir / row["relative_path"], row["num_samples"])
    (batch_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_build_keyword_dataset_partitions_positive_and_negative_clips(tmp_path):
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    _write_manifest(batch_dir)

    dataset = build_keyword_dataset(batch_dir, keyword="tessera", test_ratio=0.5)

    assert sorted(path.name for path in dataset.positive_train) == [
        "0001_tessera__af-heart__sidecar.wav",
    ]
    assert sorted(path.name for path in dataset.positive_test) == [
        "0002_tessera__am-adam__sidecar.wav",
    ]
    assert sorted(path.name for path in dataset.negative_train) == [
        "0003_alpha__af-heart__sidecar.wav",
    ]
    assert sorted(path.name for path in dataset.negative_test) == [
        "0004_omega__am-adam__sidecar.wav",
    ]


def test_recommend_total_length_rounds_and_enforces_minimum():
    assert recommend_total_length([9000, 11000, 10000]) == 32000
    assert recommend_total_length([25000, 26000, 27000]) == 38000


def test_suggest_threshold_prefers_recall_then_lower_false_positive_rate():
    threshold, metrics = suggest_threshold(
        positive_scores=[0.92, 0.77, 0.55],
        negative_scores=[0.61, 0.42, 0.11],
        thresholds=[0.5, 0.6, 0.7],
    )

    assert threshold == 0.7
    assert metrics == {
        "recall": 2 / 3,
        "false_positive_rate": 0.0,
        "false_positives": 0,
        "true_positives": 2,
        "true_negatives": 3,
        "false_negatives": 1,
    }


def test_read_wav_pcm16_resamples_mouthfeel_style_24khz_clips(tmp_path):
    wav_path = tmp_path / "mouthfeel.wav"
    _write_test_wav(wav_path, 24000, sample_rate=24000)

    pcm = _read_wav_pcm16(wav_path)

    assert pcm.dtype.name == "int16"
    assert pcm.shape == (16000,)


def test_json_ready_converts_numpy_scalars():
    payload = {
        "threshold": np.float32(0.75),
        "metrics": {"true_positives": np.int64(3)},
    }

    assert json.loads(json.dumps(_json_ready(payload))) == {
        "threshold": 0.75,
        "metrics": {"true_positives": 3},
    }


def test_clip_scores_resamples_audio_before_runtime_scoring(tmp_path, monkeypatch):
    wav_path = tmp_path / "mouthfeel.wav"
    _write_test_wav(wav_path, 24000, sample_rate=24000)
    seen = {}

    class FakeModel:
        def __init__(self, wakeword_model_paths):
            seen["model_paths"] = wakeword_model_paths

        def predict_clip(self, clip):
            seen["clip_type"] = type(clip).__name__
            seen["clip_shape"] = tuple(clip.shape)
            return [{"tessera": 0.5}]

    import openwakeword

    fake_module = types.SimpleNamespace(Model=FakeModel)
    monkeypatch.setattr(openwakeword, "model", fake_module, raising=False)
    monkeypatch.setitem(sys.modules, "openwakeword.model", fake_module)

    scores = _clip_scores(tmp_path / "tessera.onnx", [wav_path], "tessera")

    assert scores == [0.5]
    assert seen == {
        "model_paths": [str(tmp_path / "tessera.onnx")],
        "clip_type": "ndarray",
        "clip_shape": (16000,),
    }
