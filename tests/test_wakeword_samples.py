import json
import types
import wave

import numpy as np

from spoke.wakeword_samples import (
    WakewordSampleSpec,
    _build_synthesizer,
    _parse_args,
    load_phrase_lines,
    write_sample_batch,
)


def test_load_phrase_lines_ignores_blanks_and_comments(tmp_path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("\n# comment\n\ntessera\n  alpha  \n")

    assert load_phrase_lines(phrases) == ["tessera", "alpha"]


def test_write_sample_batch_writes_wavs_and_manifest(tmp_path):
    calls = []

    def synthesize(spec: WakewordSampleSpec):
        calls.append((spec.text, spec.voice, spec.backend, spec.model))
        return types.SimpleNamespace(
            audio=np.array([[0.1], [-0.1], [0.05]], dtype=np.float32),
            sample_rate=16000,
        )

    specs = [
        WakewordSampleSpec(
            text="tessera",
            backend="local",
            model="local-model",
            voice="casual_female",
        ),
        WakewordSampleSpec(
            text="tessera",
            backend="cloud",
            model="gemini-2.5-flash-preview-tts",
            voice="Aoede",
        ),
    ]

    records = write_sample_batch(specs, tmp_path, synthesize)

    assert calls == [
        ("tessera", "casual_female", "local", "local-model"),
        ("tessera", "Aoede", "cloud", "gemini-2.5-flash-preview-tts"),
    ]
    assert len(records) == 2

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert [row["voice"] for row in manifest_rows] == ["casual_female", "Aoede"]
    assert manifest_rows[0]["backend"] == "local"
    assert manifest_rows[1]["backend"] == "cloud"

    for row in manifest_rows:
        wav_path = tmp_path / row["relative_path"]
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnchannels() == 1
            assert wav_file.getnframes() == 3


def test_build_synthesizer_passes_max_tokens_to_local_tts(monkeypatch):
    captured = {}

    class FakeTTSClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def synthesize_audio(self, text: str):
            assert text == "tessera"
            return types.SimpleNamespace(
                audio=np.array([[0.1], [-0.1]], dtype=np.float32),
                sample_rate=24000,
            )

    monkeypatch.setattr("spoke.wakeword_samples.TTSClient", FakeTTSClient)
    args = _parse_args(
        [
            "--backend", "local",
            "--text", "tessera",
            "--voice", "casual_female",
            "--output-dir", "/tmp/ignored",
            "--max-tokens", "64",
        ]
    )

    synthesize = _build_synthesizer(args)
    synthesize(
        WakewordSampleSpec(
            text="tessera",
            backend="local",
            model="local-model",
            voice="casual_female",
        )
    )

    assert captured == {
        "model_id": "local-model",
        "voice": "casual_female",
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.95,
        "max_tokens": 64,
    }


def test_parse_args_accepts_voice_file(tmp_path):
    voices = tmp_path / "voices.txt"
    voices.write_text("\n# comment\n\naf_heart\nam_adam\n")

    args = _parse_args(
        [
            "--text", "tessera",
            "--voice-file", str(voices),
            "--output-dir", "/tmp/ignored",
        ]
    )

    resolved_voices = list(args.voice)
    for voice_file in args.voice_file:
        resolved_voices.extend(load_phrase_lines(voice_file))

    assert resolved_voices == ["af_heart", "am_adam"]
