from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import wave

import pytest

import spoke.transcribe_nemotron as nemotron_module
from spoke.transcribe_nemotron import NemotronCPUClient, NemotronCPUError
from spoke.transcription_prompt import TranscriptionPromptProvider


def _wav_bytes(seconds: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(16000 * seconds))
    return output.getvalue()


def _seated_paths(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "nemo-speech"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "nemotron.gguf"
    model.write_bytes(b"model")
    return binary, model


_TEST_MODEL_SHA256 = hashlib.sha256(b"model").hexdigest()


def test_available_requires_executable_and_model(tmp_path):
    binary, model = _seated_paths(tmp_path)

    assert NemotronCPUClient.available(binary=binary, model_path=model)
    model.unlink()
    assert not NemotronCPUClient.available(binary=binary, model_path=model)


def test_transcribe_uses_cpu_full_buffer_and_uncapped_decoder_phrases(tmp_path):
    binary, model = _seated_paths(tmp_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Kaminos, Epistaxis\nTrellis2MLX", encoding="utf-8")
    prompt_provider = TranscriptionPromptProvider(
        path=prompt_path,
        include_builtin=False,
    )
    wav_bytes = _wav_bytes()
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        seen["wav_bytes"] = Path(cmd[3]).read_bytes()
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "file": cmd[3],
                    "text": "hello from nemotron",
                    "confidence": 1.0,
                    "duration": 0.1,
                    "languages": ["en"],
                    "words": [],
                    "future_additive_field": True,
                }
            ),
            stderr="",
        )

    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=prompt_provider,
        failure_dir=tmp_path / "failures",
    )
    with patch("spoke.transcribe_nemotron.subprocess.run", side_effect=fake_run):
        text = client.transcribe(wav_bytes)

    assert text == "hello from nemotron"
    assert seen["wav_bytes"] == wav_bytes
    assert seen["kwargs"]["timeout"] is None
    cmd = seen["cmd"]
    assert cmd[:3] == [str(binary), "--json", "transcribe"]
    assert cmd[cmd.index("--model") + 1] == str(model)
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--format") + 1] == "json"
    assert "--no-batching" in cmd
    assert [cmd[index + 1] for index, value in enumerate(cmd) if value == "--speech-context"] == [
        "Kaminos",
        "Epistaxis",
        "Trellis2MLX",
    ]
    assert "--speech-context-boost" in cmd
    assert "--stream" not in cmd
    assert "--endpointing" not in cmd
    assert "--vad-model" not in cmd
    assert "--vad-masking" not in cmd
    assert client._last_receipt["effective_device"] == "cpu"
    assert client._last_receipt["vad"] is False
    assert client._last_receipt["prompt"]["effective"] is True


def test_process_failure_writes_replayable_route_report(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )
    failed = subprocess.CompletedProcess(
        [str(binary)],
        9,
        stdout="",
        stderr="decoder failed before output",
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=failed):
        with pytest.raises(NemotronCPUError, match="report="):
            client.transcribe(_wav_bytes())

    reports = list(failure_dir.glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["failure_phase"] == "transcribe_process"
    assert report["effective_device"] == "cpu"
    assert report["streaming"] is False
    assert report["endpointing"] is False
    assert report["vad"] is False
    assert report["audio_sha256"]
    assert report["audio_bytes"] == len(_wav_bytes())
    assert report["exit_code"] == 9


def test_invalid_success_payload_fails_loud_and_preserves_report(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )
    malformed = subprocess.CompletedProcess(
        [str(binary)],
        0,
        stdout=json.dumps({"confidence": 1.0}),
        stderr="",
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=malformed):
        with pytest.raises(NemotronCPUError, match="invalid JSON output"):
            client.transcribe(_wav_bytes())

    report = json.loads(next(failure_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["failure_phase"] == "parse_output"


def test_prepare_rejects_model_hash_drift(tmp_path):
    binary, model = _seated_paths(tmp_path)
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256="0" * 64,
    )

    with pytest.raises(NemotronCPUError, match="SHA-256 mismatch"):
        client.prepare()


def test_replay_harness_rejects_false_cpu_route_and_still_writes_report(tmp_path):
    assert hasattr(nemotron_module, "run_replay")

    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(_wav_bytes())
    output_path = tmp_path / "replay.json"

    class WrongRouteClient:
        _last_receipt = None

        def transcribe(self, wav_bytes):
            self._last_receipt = {
                "requested_model": nemotron_module._NEMOTRON_CPU_MODEL_ID,
                "effective_device": "metal",
                "audio_sha256": "wrong",
            }
            return "false success"

    with pytest.raises(NemotronCPUError, match="route identity"):
        nemotron_module.run_replay(
            wav_path,
            output_path,
            client=WrongRouteClient(),
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["failure_phase"] == "route_identity"
    assert report["effective_device"] == "metal"
    assert "transcript" not in report
