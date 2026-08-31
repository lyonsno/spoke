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


def _complete_route_receipt(audio_sha256: str) -> dict:
    return {
        "schema": "spoke.nemotron-cpu-transcription.v2",
        "status": "success",
        "requested_model": nemotron_module._NEMOTRON_CPU_MODEL_ID,
        "effective_binary": "/pinned/nemo-speech",
        "effective_model_path": "/pinned/nemotron.gguf",
        "model_sha256_expected": "a" * 64,
        "model_sha256_actual": "a" * 64,
        "effective_device": "cpu",
        "audio_sha256": audio_sha256,
        "streaming": False,
        "endpointing": False,
        "vad": False,
        "recognizer_configuration": {
            "authority": "explicit_cli_with_nemo_environment_cleared",
            "streaming": False,
            "endpointing": False,
            "vad": False,
        },
    }


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
    assert client._last_receipt["prompt"]["payload_constructed"] is True
    assert client._last_receipt["prompt"]["submission_attempted"] is True
    assert client._last_receipt["prompt"]["runtime_accepted"] is True
    assert client._last_receipt["prompt"]["semantic_effect_observed"] is None
    assert client._last_receipt["recognizer_configuration"] == {
        "authority": "explicit_cli_with_nemo_environment_cleared",
        "streaming": False,
        "endpointing": False,
        "vad": False,
    }
    assert "env" in seen["kwargs"]
    assert not any(
        key == "NEMO_SPEECH" or key.startswith("NEMO_SPEECH_")
        for key in seen["kwargs"]["env"]
    )


def test_transcribe_strips_inherited_nemo_configuration(tmp_path, monkeypatch):
    binary, model = _seated_paths(tmp_path)
    monkeypatch.setenv("NEMO_SPEECH_VAD_MODEL", "/tmp/hidden-vad.nemo")
    monkeypatch.setenv("NEMO_SPEECH_ENDPOINTING_ENABLED", "true")
    monkeypatch.setenv("NEMO_SPEECH", "hidden-config")
    monkeypatch.setenv("UNRELATED_PARENT_VALUE", "preserved")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"text": "controlled route"}), stderr=""
        )

    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=tmp_path / "failures",
    )
    with patch("spoke.transcribe_nemotron.subprocess.run", side_effect=fake_run):
        assert client.transcribe(_wav_bytes()) == "controlled route"

    assert seen["env"]["UNRELATED_PARENT_VALUE"] == "preserved"
    assert not any(
        key == "NEMO_SPEECH" or key.startswith("NEMO_SPEECH_")
        for key in seen["env"]
    )


@pytest.mark.parametrize("runtime_text", ["", "   "])
def test_blank_runtime_output_is_durable_failure(tmp_path, runtime_text):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )
    completed = subprocess.CompletedProcess(
        [str(binary)], 0, stdout=json.dumps({"text": runtime_text}), stderr=""
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=completed):
        with pytest.raises(NemotronCPUError, match="blank transcript"):
            client.transcribe(_wav_bytes())

    report = json.loads(next(failure_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["failure_phase"] == "validate_output"
    assert report["audio_sha256"]


@pytest.mark.parametrize(
    "runtime_text",
    ["thank you", "repeat this repeat this repeat this"],
)
def test_nemotron_output_is_not_mutated_by_whisper_filters(tmp_path, runtime_text):
    binary, model = _seated_paths(tmp_path)
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=tmp_path / "failures",
    )
    completed = subprocess.CompletedProcess(
        [str(binary)], 0, stdout=json.dumps({"text": runtime_text}), stderr=""
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=completed):
        assert client.transcribe(_wav_bytes()) == runtime_text


def test_actual_adapter_passes_every_prompt_phrase_without_a_cap(tmp_path):
    binary, model = _seated_paths(tmp_path)
    phrases = [f"private-term-{index:03d}" for index in range(257)]
    provider = TranscriptionPromptProvider(
        inline=", ".join(phrases),
        include_builtin=False,
    )
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"text": "all phrases submitted"}), stderr=""
        )

    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=provider,
        failure_dir=tmp_path / "failures",
    )
    with patch("spoke.transcribe_nemotron.subprocess.run", side_effect=fake_run):
        client.transcribe(_wav_bytes())

    submitted = [
        seen["cmd"][index + 1]
        for index, value in enumerate(seen["cmd"])
        if value == "--speech-context"
    ]
    assert submitted == phrases
    assert client._last_receipt["speech_context_count"] == len(phrases)


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
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256="0" * 64,
        failure_dir=failure_dir,
    )

    with pytest.raises(NemotronCPUError, match="SHA-256 mismatch"):
        client.transcribe(_wav_bytes())

    report = json.loads(next(failure_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["failure_phase"] == "prepare_runtime"
    assert report["audio_sha256"]
    assert report["model_sha256_expected"] == "0" * 64
    assert report["model_sha256_actual"] == _TEST_MODEL_SHA256


def test_process_launch_failure_clears_stale_receipt_and_writes_report(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )
    success = subprocess.CompletedProcess(
        [str(binary)], 0, stdout=json.dumps({"text": "first success"}), stderr=""
    )
    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=success):
        client.transcribe(_wav_bytes())
    assert client._last_receipt["status"] == "success"

    with patch(
        "spoke.transcribe_nemotron.subprocess.run",
        side_effect=OSError("exec format error"),
    ):
        with pytest.raises(NemotronCPUError, match="process launch failed"):
            client.transcribe(_wav_bytes())

    assert client._last_receipt["status"] == "failure"
    assert client._last_receipt["failure_phase"] == "process_launch"
    report = json.loads(sorted(failure_dir.glob("*.json"))[-1].read_text(encoding="utf-8"))
    assert report["failure_phase"] == "process_launch"


def test_temporary_input_failure_writes_phase_report(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )

    with patch.object(Path, "write_bytes", side_effect=OSError("read-only volume")):
        with pytest.raises(NemotronCPUError, match="temporary input write failed"):
            client.transcribe(_wav_bytes())

    report = json.loads(next(failure_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["failure_phase"] == "write_temporary_input"
    assert report["audio_sha256"]


def test_invalid_utf8_output_writes_decode_failure(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(include_builtin=False),
        failure_dir=failure_dir,
    )
    completed = subprocess.CompletedProcess(
        [str(binary)], 0, stdout=b"\xff\xfe", stderr=b""
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=completed):
        with pytest.raises(NemotronCPUError, match="output decoding failed"):
            client.transcribe(_wav_bytes())

    report = json.loads(next(failure_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["failure_phase"] == "decode_output"
    assert report["evidence"]["stdout_bytes"] == 2
    assert report["evidence"]["stdout_sha256"]


def test_timeout_report_and_error_do_not_retain_private_phrases(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    private_phrase = "PRIVATE-SENTINEL-NEVER-PERSIST"
    private_stdout = b"PRIVATE-PARTIAL-STDOUT"
    private_stderr = b"PRIVATE-PARTIAL-STDERR"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        timeout=0.01,
        prompt_provider=TranscriptionPromptProvider(
            inline=private_phrase,
            include_builtin=False,
        ),
        failure_dir=failure_dir,
    )

    def time_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd,
            kwargs["timeout"],
            output=private_stdout,
            stderr=private_stderr,
        )

    with patch("spoke.transcribe_nemotron.subprocess.run", side_effect=time_out):
        with pytest.raises(NemotronCPUError) as exc_info:
            client.transcribe(_wav_bytes())

    report_text = next(failure_dir.glob("*.json")).read_text(encoding="utf-8")
    assert private_phrase not in report_text
    assert private_stdout.decode() not in report_text
    assert private_stderr.decode() not in report_text
    assert private_phrase not in str(exc_info.value)
    report = json.loads(report_text)
    assert report["failure_phase"] == "transcribe_timeout"
    assert report["prompt"]["sha256"]
    assert report["speech_context_count"] == 1
    assert report["evidence"] == {
        "stdout_bytes": len(private_stdout),
        "stdout_sha256": hashlib.sha256(private_stdout).hexdigest(),
        "stderr_bytes": len(private_stderr),
        "stderr_sha256": hashlib.sha256(private_stderr).hexdigest(),
    }


def test_process_failure_report_hashes_diagnostics_without_private_bodies(tmp_path):
    binary, model = _seated_paths(tmp_path)
    failure_dir = tmp_path / "failures"
    private_phrase = "PRIVATE-CONTEXT-MUST-NOT-APPEAR"
    private_stderr = f"decoder rejected {private_phrase}"
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256=_TEST_MODEL_SHA256,
        prompt_provider=TranscriptionPromptProvider(
            inline=private_phrase,
            include_builtin=False,
        ),
        failure_dir=failure_dir,
    )
    failed = subprocess.CompletedProcess(
        [str(binary)], 7, stdout="", stderr=private_stderr
    )

    with patch("spoke.transcribe_nemotron.subprocess.run", return_value=failed):
        with pytest.raises(NemotronCPUError) as exc_info:
            client.transcribe(_wav_bytes())

    report_text = next(failure_dir.glob("*.json")).read_text(encoding="utf-8")
    assert private_phrase not in report_text
    assert private_phrase not in str(exc_info.value)
    report = json.loads(report_text)
    assert report["evidence"]["stderr_bytes"] == len(private_stderr.encode())
    assert report["evidence"]["stderr_sha256"] == hashlib.sha256(
        private_stderr.encode()
    ).hexdigest()


def test_failure_report_publication_failure_remains_explicit(tmp_path):
    binary, model = _seated_paths(tmp_path)
    client = NemotronCPUClient(
        binary=binary,
        model_path=model,
        expected_model_sha256="0" * 64,
        failure_dir=tmp_path / "failures",
    )

    with patch.object(client, "_write_failure", side_effect=OSError("disk unavailable")):
        with pytest.raises(NemotronCPUError, match="failure report publication failed"):
            client.transcribe(_wav_bytes())

    assert client._last_receipt["status"] == "failure"
    assert client._last_receipt["failure_phase"] == "prepare_runtime"
    assert client._last_receipt["report_publication"]["status"] == "failure"


def test_replay_harness_rejects_false_cpu_route_and_still_writes_report(tmp_path):
    assert hasattr(nemotron_module, "run_replay")

    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(_wav_bytes())
    output_path = tmp_path / "replay.json"

    class WrongRouteClient:
        _last_receipt = None

        def transcribe(self, wav_bytes):
            self._last_receipt = _complete_route_receipt(
                hashlib.sha256(wav_bytes).hexdigest()
            )
            self._last_receipt["effective_device"] = "metal"
            return "false success"

    with pytest.raises(NemotronCPUError, match="route identity"):
        nemotron_module.run_replay(
            wav_path,
            output_path,
            client=WrongRouteClient(),
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["schema"] == "spoke.nemotron-cpu-replay.v2"
    assert report["failure_phase"] == "route_identity"
    assert report["receipt"]["status"] == "success"
    assert report["receipt"]["effective_device"] == "metal"
    assert "transcript" not in report


def test_replay_harness_reports_missing_input_before_transcription(tmp_path):
    missing_path = tmp_path / "vanished.wav"
    output_path = tmp_path / "replay.json"

    with pytest.raises(NemotronCPUError, match="input is unavailable"):
        nemotron_module.run_replay(missing_path, output_path)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report == {
        "schema": "spoke.nemotron-cpu-replay.v2",
        "status": "failure",
        "failure_phase": "read_input",
        "input_path": str(missing_path),
        "error_type": "FileNotFoundError",
    }


@pytest.mark.parametrize(
    "missing_field",
    ["effective_binary", "effective_model_path", "model_sha256_actual"],
)
def test_replay_harness_requires_complete_effective_identity(tmp_path, missing_field):
    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(_wav_bytes())
    output_path = tmp_path / "replay.json"
    expected_audio_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()

    class IncompleteRouteClient:
        _last_receipt = None

        def transcribe(self, wav_bytes):
            self._last_receipt = _complete_route_receipt(expected_audio_sha256)
            self._last_receipt.pop(missing_field)
            return "false success"

    with pytest.raises(NemotronCPUError, match="route identity"):
        nemotron_module.run_replay(
            wav_path,
            output_path,
            client=IncompleteRouteClient(),
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "failure"
    assert report["schema"] == "spoke.nemotron-cpu-replay.v2"
    assert report["failure_phase"] == "route_identity"
    assert report["receipt"]["status"] == "success"
    assert missing_field not in report["receipt"]


@pytest.mark.parametrize(
    ("field_path", "false_value"),
    [
        ("audio_sha256", "f" * 64),
        ("model_sha256_actual", "b" * 64),
        ("recognizer_configuration.authority", "inherited_environment"),
        ("recognizer_configuration.streaming", True),
        ("recognizer_configuration.endpointing", True),
        ("recognizer_configuration.vad", True),
    ],
)
def test_replay_harness_rejects_each_false_effective_identity_independently(
    tmp_path,
    field_path,
    false_value,
):
    wav_path = tmp_path / "input.wav"
    wav_path.write_bytes(_wav_bytes())
    output_path = tmp_path / "replay.json"
    expected_audio_sha256 = hashlib.sha256(wav_path.read_bytes()).hexdigest()

    class OneFalseIdentityClient:
        _last_receipt = None

        def transcribe(self, wav_bytes):
            self._last_receipt = _complete_route_receipt(expected_audio_sha256)
            target = self._last_receipt
            parts = field_path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = false_value
            return "false success"

    with pytest.raises(NemotronCPUError, match="route identity"):
        nemotron_module.run_replay(
            wav_path,
            output_path,
            client=OneFalseIdentityClient(),
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema"] == "spoke.nemotron-cpu-replay.v2"
    assert report["status"] == "failure"
    assert report["failure_phase"] == "route_identity"
    assert report["input_path"] == str(wav_path)
    assert report["expected_audio_sha256"] == expected_audio_sha256
    assert report["wall_seconds"] >= 0
    assert report["receipt"]["schema"] == "spoke.nemotron-cpu-transcription.v2"
    assert report["receipt"]["status"] == "success"
    observed = report["receipt"]
    for part in field_path.split("."):
        observed = observed[part]
    assert observed == false_value
