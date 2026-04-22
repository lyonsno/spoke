"""Local openWakeWord training helpers for Mouthfeel-style keyword batches."""

from __future__ import annotations

import argparse
import json
import random
import wave
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np

from .wakeword_samples import WakewordSampleRecord


@dataclass(frozen=True, slots=True)
class KeywordDataset:
    keyword: str
    positive_train: list[Path]
    positive_test: list[Path]
    negative_train: list[Path]
    negative_test: list[Path]


@dataclass(frozen=True, slots=True)
class KeywordTrainingResult:
    keyword: str
    model_path: str
    threshold: float
    total_length: int
    positive_test_scores: list[float]
    negative_test_scores: list[float]
    metrics: dict[str, float | int]


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def load_sample_manifest(batch_dir: str | Path) -> list[WakewordSampleRecord]:
    manifest_path = Path(batch_dir) / "manifest.jsonl"
    rows: list[WakewordSampleRecord] = []
    for raw in manifest_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(WakewordSampleRecord(**json.loads(raw)))
    return rows


def _split_paths(paths: list[Path], test_ratio: float) -> tuple[list[Path], list[Path]]:
    if len(paths) <= 1:
        return paths[:], []
    n_test = max(1, int(round(len(paths) * test_ratio)))
    n_test = min(n_test, len(paths) - 1)
    return paths[:-n_test], paths[-n_test:]


def build_keyword_dataset(
    batch_dir: str | Path,
    *,
    keyword: str,
    test_ratio: float = 0.25,
) -> KeywordDataset:
    batch_dir = Path(batch_dir)
    keyword = keyword.strip().lower()
    positives: list[Path] = []
    negatives: list[Path] = []
    for row in sorted(load_sample_manifest(batch_dir), key=lambda item: item.relative_path):
        path = batch_dir / row.relative_path
        if row.text.strip().lower() == keyword:
            positives.append(path)
        else:
            negatives.append(path)
    if not positives:
        raise ValueError(f"No positive clips found for keyword {keyword!r}")
    if not negatives:
        raise ValueError(f"No negative clips found for keyword {keyword!r}")
    positive_train, positive_test = _split_paths(positives, test_ratio)
    negative_train, negative_test = _split_paths(negatives, test_ratio)
    return KeywordDataset(
        keyword=keyword,
        positive_train=positive_train,
        positive_test=positive_test,
        negative_train=negative_train,
        negative_test=negative_test,
    )


def recommend_total_length(sample_counts: Iterable[int]) -> int:
    counts = [int(value) for value in sample_counts]
    if not counts:
        raise ValueError("sample_counts must not be empty")
    base = int(round(float(np.median(counts)) / 1000.0) * 1000) + 12000
    if base < 32000:
        return 32000
    if abs(base - 32000) <= 4000:
        return 32000
    return base


def suggest_threshold(
    *,
    positive_scores: Iterable[float],
    negative_scores: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> tuple[float, dict[str, float | int]]:
    positives = [float(score) for score in positive_scores]
    negatives = [float(score) for score in negative_scores]
    if not positives or not negatives:
        raise ValueError("positive_scores and negative_scores must both be non-empty")
    candidate_thresholds = list(thresholds or np.linspace(0.1, 0.9, 17))

    best_threshold = candidate_thresholds[0]
    best_metrics: dict[str, float | int] | None = None
    best_score: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds:
        tp = sum(score >= threshold for score in positives)
        fn = len(positives) - tp
        fp = sum(score >= threshold for score in negatives)
        tn = len(negatives) - fp
        recall = tp / len(positives)
        false_positive_rate = fp / len(negatives)
        balanced_accuracy = (recall + (tn / len(negatives))) / 2.0
        score = (balanced_accuracy, -false_positive_rate, threshold)
        metrics = {
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "false_positives": fp,
            "true_positives": tp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def _read_wav_pcm16(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        sample_rate = wav_file.getframerate()
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)

    if sample_rate == 16000:
        return pcm

    from scipy import signal

    scale = gcd(sample_rate, 16000)
    resampled = signal.resample_poly(
        pcm.astype(np.float32),
        up=16000 // scale,
        down=sample_rate // scale,
    )
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def _pad_or_trim_clip(pcm: np.ndarray, total_length: int) -> np.ndarray:
    if pcm.shape[0] >= total_length:
        return pcm[:total_length]
    padding = np.zeros(total_length - pcm.shape[0], dtype=np.int16)
    return np.concatenate((pcm, padding))


def _load_padded_batch(paths: Iterable[Path], total_length: int) -> np.ndarray:
    clips = [_pad_or_trim_clip(_read_wav_pcm16(path), total_length) for path in paths]
    if not clips:
        raise ValueError("No clips were provided")
    return np.stack(clips, axis=0)


def _audio_features():
    from openwakeword.utils import AudioFeatures

    return AudioFeatures()


def _compute_features(paths: Iterable[Path], total_length: int) -> np.ndarray:
    clips = _load_padded_batch(paths, total_length)
    features = _audio_features().embed_clips(clips, batch_size=min(len(clips), 32), ncpu=1)
    return features.astype(np.float32)


def _train_binary_classifier(
    positive_train_features: np.ndarray,
    negative_train_features: np.ndarray,
    positive_test_features: np.ndarray,
    negative_test_features: np.ndarray,
    *,
    layer_size: int = 32,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    seed: int = 0,
):
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    input_shape = positive_train_features.shape[1:]

    class BinaryWakewordNet(torch.nn.Module):
        def __init__(self, input_shape: tuple[int, int], layer_size: int):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.linear1 = torch.nn.Linear(input_shape[0] * input_shape[1], layer_size)
            self.norm1 = torch.nn.LayerNorm(layer_size)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(layer_size, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.norm1(self.linear1(x)))
            return self.sigmoid(self.linear2(x))

    train_x = np.vstack((positive_train_features, negative_train_features))
    train_y = np.concatenate(
        (
            np.ones(len(positive_train_features), dtype=np.float32),
            np.zeros(len(negative_train_features), dtype=np.float32),
        )
    )
    val_x = np.vstack((positive_test_features, negative_test_features))
    val_y = np.concatenate(
        (
            np.ones(len(positive_test_features), dtype=np.float32),
            np.zeros(len(negative_test_features), dtype=np.float32),
        )
    )

    permutation = np.arange(len(train_x))
    rng = np.random.default_rng(seed)
    model = BinaryWakewordNet(input_shape, layer_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    best_state = None
    best_val_loss = None

    train_x_tensor = torch.from_numpy(train_x)
    train_y_tensor = torch.from_numpy(train_y[:, None])
    val_x_tensor = torch.from_numpy(val_x)
    val_y_tensor = torch.from_numpy(val_y[:, None])

    batch_size = min(32, len(train_x))
    for _epoch in range(epochs):
        rng.shuffle(permutation)
        model.train()
        for start in range(0, len(train_x), batch_size):
            ndx = permutation[start:start + batch_size]
            batch_x = train_x_tensor[ndx]
            batch_y = train_y_tensor[ndx]
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x_tensor), val_y_tensor).item()
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, input_shape


def _export_model_to_onnx(model, input_shape: tuple[int, int], output_path: str | Path) -> None:
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.rand((1,) + input_shape),
        str(output_path),
        opset_version=13,
    )


def _clip_scores(model_path: str | Path, paths: Iterable[Path], keyword: str) -> list[float]:
    from openwakeword.model import Model as RuntimeModel

    runtime = RuntimeModel(wakeword_model_paths=[str(model_path)])
    scores: list[float] = []
    for path in paths:
        frame_predictions = runtime.predict_clip(_read_wav_pcm16(path))
        scores.append(max(float(frame.get(keyword, 0.0)) for frame in frame_predictions))
    return scores


def train_keyword_model(
    batch_dir: str | Path,
    *,
    keyword: str,
    output_dir: str | Path,
    test_ratio: float = 0.25,
    layer_size: int = 32,
    epochs: int = 250,
    seed: int = 0,
) -> KeywordTrainingResult:
    dataset = build_keyword_dataset(batch_dir, keyword=keyword, test_ratio=test_ratio)
    sample_counts = [_read_wav_pcm16(path).shape[0] for path in dataset.positive_train + dataset.positive_test]
    total_length = recommend_total_length(sample_counts)

    positive_train_features = _compute_features(dataset.positive_train, total_length)
    negative_train_features = _compute_features(dataset.negative_train, total_length)
    positive_test_features = _compute_features(dataset.positive_test or dataset.positive_train, total_length)
    negative_test_features = _compute_features(dataset.negative_test or dataset.negative_train, total_length)

    model, input_shape = _train_binary_classifier(
        positive_train_features,
        negative_train_features,
        positive_test_features,
        negative_test_features,
        layer_size=layer_size,
        epochs=epochs,
        seed=seed,
    )

    output_dir = Path(output_dir)
    model_path = output_dir / f"{keyword}.onnx"
    _export_model_to_onnx(model, input_shape, model_path)

    positive_scores = _clip_scores(model_path, dataset.positive_test or dataset.positive_train, keyword)
    negative_scores = _clip_scores(model_path, dataset.negative_test or dataset.negative_train, keyword)
    threshold, metrics = suggest_threshold(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    )

    result = KeywordTrainingResult(
        keyword=keyword,
        model_path=str(model_path),
        threshold=threshold,
        total_length=total_length,
        positive_test_scores=positive_scores,
        negative_test_scores=negative_scores,
        metrics=metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{keyword}.metrics.json").write_text(
        json.dumps(_json_ready(asdict(result)), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / f"{keyword}.dataset.json").write_text(
        json.dumps(
            _json_ready({
                "keyword": keyword,
                "positive_train": [str(path) for path in dataset.positive_train],
                "positive_test": [str(path) for path in dataset.positive_test],
                "negative_train": [str(path) for path in dataset.negative_train],
                "negative_test": [str(path) for path in dataset.negative_test],
            }),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train simple local openWakeWord models from a wakeword sample batch."
    )
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--layer-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    batch_dir = Path(args.batch_dir)
    if args.keyword:
        keywords = [value.strip().lower() for value in args.keyword if value.strip()]
    else:
        keywords = sorted({row.text.strip().lower() for row in load_sample_manifest(batch_dir)})

    results = [
        train_keyword_model(
            batch_dir,
            keyword=keyword,
            output_dir=Path(args.output_dir),
            test_ratio=args.test_ratio,
            layer_size=args.layer_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        for keyword in keywords
    ]
    print(json.dumps(_json_ready([asdict(result) for result in results]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
