"""Batch generation of wakeword training samples via spoke's TTS backends."""

from __future__ import annotations

import argparse
import json
import os
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .tts import (
    _DEFAULT_MODEL_ID,
    _DEFAULT_VOICE,
    _DEFAULT_GEMINI_TTS_MODEL,
    CloudTTSClient,
    RemoteTTSClient,
    TTSClient,
)


@dataclass(frozen=True, slots=True)
class WakewordSampleSpec:
    text: str
    backend: str
    model: str
    voice: str


@dataclass(frozen=True, slots=True)
class WakewordSampleRecord:
    text: str
    backend: str
    model: str
    voice: str
    sample_rate: int
    num_samples: int
    relative_path: str


def load_phrase_lines(path: str | Path) -> list[str]:
    lines: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-") or "sample"


def _audio_matrix(audio: np.ndarray) -> np.ndarray:
    matrix = np.asarray(audio, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def _float_audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def write_wav_file(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    matrix = _audio_matrix(audio)
    pcm = _float_audio_to_pcm16(matrix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(matrix.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def write_sample_batch(
    specs: Iterable[WakewordSampleSpec],
    output_dir: str | Path,
    synthesize: Callable[[WakewordSampleSpec], object],
) -> list[WakewordSampleRecord]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[WakewordSampleRecord] = []
    for idx, spec in enumerate(specs, start=1):
        rendered = synthesize(spec)
        audio = _audio_matrix(np.asarray(rendered.audio, dtype=np.float32))
        sample_rate = int(rendered.sample_rate)
        filename = (
            f"{idx:04d}_"
            f"{_slugify(spec.text)}__{_slugify(spec.voice)}__{_slugify(spec.backend)}.wav"
        )
        relative_path = filename
        write_wav_file(output_dir / filename, audio, sample_rate)
        records.append(
            WakewordSampleRecord(
                text=spec.text,
                backend=spec.backend,
                model=spec.model,
                voice=spec.voice,
                sample_rate=sample_rate,
                num_samples=int(audio.shape[0]),
                relative_path=relative_path,
            )
        )

    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(asdict(record)) + "\n" for record in records))
    return records


def _default_model_for_backend(backend: str) -> str:
    if backend == "cloud":
        return _DEFAULT_GEMINI_TTS_MODEL
    return _DEFAULT_MODEL_ID


def _default_voice_for_backend(backend: str) -> str:
    if backend == "cloud":
        return "Aoede"
    return _DEFAULT_VOICE


def _resolve_cloud_api_key(explicit: str | None, env_name: str) -> str:
    if explicit:
        return explicit
    for name in (env_name, "GEMINI_API_KEY_INACTIVE", "GEMINI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        f"Cloud TTS requested but no API key was found in --api-key, {env_name}, "
        "GEMINI_API_KEY_INACTIVE, or GEMINI_API_KEY"
    )


def _build_synthesizer(args: argparse.Namespace) -> Callable[[WakewordSampleSpec], object]:
    cache: dict[tuple[str, str, str], object] = {}

    def _client_for(spec: WakewordSampleSpec):
        key = (spec.backend, spec.model, spec.voice)
        client = cache.get(key)
        if client is not None:
            return client

        if spec.backend == "local":
            client = TTSClient(
                model_id=spec.model,
                voice=spec.voice,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
        elif spec.backend == "cloud":
            client = CloudTTSClient(
                api_key=_resolve_cloud_api_key(args.api_key, args.api_key_env),
                model=spec.model,
                voice=spec.voice,
                timeout=args.timeout,
            )
        elif spec.backend == "sidecar":
            if not args.sidecar_url:
                raise RuntimeError("--sidecar-url is required when backend=sidecar")
            client = RemoteTTSClient(
                base_url=args.sidecar_url,
                model_id=spec.model,
                voice=spec.voice,
                timeout=args.timeout,
            )
        else:
            raise RuntimeError(f"Unsupported backend: {spec.backend}")

        cache[key] = client
        return client

    def _synthesize(spec: WakewordSampleSpec):
        client = _client_for(spec)
        return client.synthesize_audio(spec.text)

    return _synthesize


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate wakeword training WAVs via local, cloud, or sidecar TTS."
    )
    parser.add_argument("--backend", choices=("local", "cloud", "sidecar"), default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--voice", action="append", default=[])
    parser.add_argument("--voice-file", action="append", default=[])
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--text-file", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY_INACTIVE")
    parser.add_argument("--sidecar-url", default=os.environ.get("SPOKE_TTS_URL", ""))
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    texts = list(args.text)
    for text_file in args.text_file:
        texts.extend(load_phrase_lines(text_file))
    if not texts:
        raise SystemExit("Provide at least one --text or --text-file")

    voices = list(args.voice)
    for voice_file in args.voice_file:
        voices.extend(load_phrase_lines(voice_file))
    if not voices:
        voices = [_default_voice_for_backend(args.backend)]
    model = args.model or _default_model_for_backend(args.backend)
    specs = [
        WakewordSampleSpec(text=text, backend=args.backend, model=model, voice=voice)
        for text in texts
        for voice in voices
    ]

    synthesize = _build_synthesizer(args)
    records = write_sample_batch(specs, args.output_dir, synthesize)
    print(f"Wrote {len(records)} wakeword samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
