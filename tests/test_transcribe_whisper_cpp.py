"""Tests for the whisper.cpp CoreML transcription client."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch
import wave

import pytest


def _wav_bytes() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return out.getvalue()


def test_coreml_companion_path_for_standard_ggml_model(tmp_path):
    from spoke.transcribe_whisper_cpp import _coreml_companion_path

    model = tmp_path / "ggml-base.en.bin"

    assert _coreml_companion_path(model) == tmp_path / "ggml-base.en-encoder.mlmodelc"


def test_client_requires_binary_model_and_coreml_companion(tmp_path):
    from spoke.transcribe_whisper_cpp import WhisperCppCoreMLClient

    binary = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-base.en.bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model.write_bytes(b"model")

    assert not WhisperCppCoreMLClient.available(binary=binary, model_path=model)

    (tmp_path / "ggml-base.en-encoder.mlmodelc").mkdir()
    assert WhisperCppCoreMLClient.available(binary=binary, model_path=model)


def test_transcribe_invokes_whisper_cli_and_reads_text_output(tmp_path):
    from spoke.transcribe_whisper_cpp import WhisperCppCoreMLClient

    binary = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-base.en.bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model.write_bytes(b"model")
    (tmp_path / "ggml-base.en-encoder.mlmodelc").mkdir()

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        output_prefix = Path(cmd[cmd.index("-of") + 1])
        output_prefix.with_suffix(".txt").write_text("  hello from ane  \n")
        class Result:
            returncode = 0
            stdout = "system_info: COREML = 1\n"
            stderr = ""
        return Result()

    client = WhisperCppCoreMLClient(binary=binary, model_path=model, timeout=9.0)
    with patch("spoke.transcribe_whisper_cpp.subprocess.run", side_effect=fake_run):
        text = client.transcribe(_wav_bytes())

    assert text == "hello from ane"
    cmd = seen["cmd"]
    assert cmd[:4] == [str(binary), "-m", str(model), "-f"]
    assert "-otxt" in cmd
    assert "-np" in cmd
    assert seen["kwargs"]["timeout"] == 9.0
    assert seen["kwargs"]["check"] is False


def test_transcribe_raises_with_phase_when_binary_fails(tmp_path):
    from spoke.transcribe_whisper_cpp import (
        WhisperCppCoreMLClient,
        WhisperCppCoreMLError,
    )

    binary = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-base.en.bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model.write_bytes(b"model")
    (tmp_path / "ggml-base.en-encoder.mlmodelc").mkdir()

    class Result:
        returncode = 13
        stdout = ""
        stderr = "bad coreml day"

    client = WhisperCppCoreMLClient(binary=binary, model_path=model)
    with patch("spoke.transcribe_whisper_cpp.subprocess.run", return_value=Result()):
        with pytest.raises(WhisperCppCoreMLError, match="whisper.cpp failed"):
            client.transcribe(_wav_bytes())
