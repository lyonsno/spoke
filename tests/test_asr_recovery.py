from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


def test_serial_recovery_uses_remote_then_whisperkit(tmp_path) -> None:
    from spoke.asr_recovery import ASRRouteReporter, SerialASRRecovery

    events: list[str] = []
    remote = MagicMock()
    remote.route_identity.return_value = {
        "route": "remote",
        "url": "http://whisper-box:7001",
        "model": "large-v3-turbo",
    }
    remote.transcribe.side_effect = lambda _wav: (
        events.append("remote") or (_ for _ in ()).throw(ConnectionError("down"))
    )
    whisperkit = MagicMock()
    whisperkit.route_identity.return_value = {
        "route": "whisperkit-cli",
        "model": "medium.en",
        "audio_encoder_compute_units": "cpuAndNeuralEngine",
        "text_decoder_compute_units": "cpuOnly",
        "chunking_strategy": "none",
    }
    whisperkit.transcribe.side_effect = lambda _wav: (
        events.append("whisperkit") or "recovered text"
    )
    reporter = ASRRouteReporter(
        tmp_path / "utterance.asr.json",
        wav_bytes=b"authoritative wav",
        requested_route={
            "route": "local-mlx-whisper",
            "model": "whisper-large-v3-turbo",
        },
    )
    recovery = SerialASRRecovery(
        [("remote", remote), ("whisperkit", whisperkit)]
    )

    text = recovery.recover(
        b"authoritative wav",
        primary_failure=TimeoutError("deadline"),
        reporter=reporter,
    )

    assert text == "recovered text"
    assert events == ["remote", "whisperkit"]
    report = json.loads((tmp_path / "utterance.asr.json").read_text())
    assert report["status"] == "succeeded"
    assert report["requested_route"]["route"] == "local-mlx-whisper"
    assert report["effective_route"]["route"] == "whisperkit-cli"
    assert [attempt["status"] for attempt in report["recovery_attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert report["audio"]["sha256"]
    assert report["audio"]["byte_count"] == len(b"authoritative wav")


def test_blank_recovery_output_is_not_success(tmp_path) -> None:
    from spoke.asr_recovery import (
        ASRRecoveryError,
        ASRRouteReporter,
        SerialASRRecovery,
    )

    whisperkit = MagicMock()
    whisperkit.route_identity.return_value = {"route": "whisperkit-cli"}
    whisperkit.transcribe.return_value = "   "
    report_path = tmp_path / "blank.asr.json"
    reporter = ASRRouteReporter(
        report_path,
        wav_bytes=b"wav",
        requested_route={"route": "local-mlx-whisper"},
    )

    with pytest.raises(ASRRecoveryError, match="blank"):
        SerialASRRecovery([("whisperkit", whisperkit)]).recover(
            b"wav",
            primary_failure=TimeoutError("deadline"),
            reporter=reporter,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["effective_route"] is None
    assert report["recovery_attempts"][0]["status"] == "failed"
    assert report["failure_phase"] == "serial_recovery"


def test_recovery_routes_are_serial_across_utterances(tmp_path) -> None:
    from spoke.asr_recovery import ASRRouteReporter, SerialASRRecovery

    first_entered = threading.Event()
    release_first = threading.Event()
    concurrent_entry = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    class BlockingRoute:
        def route_identity(self):
            return {"route": "blocking-test"}

        def transcribe(self, _wav):
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                concurrent_entry.set()
            return f"text {call_number}"

    recovery = SerialASRRecovery([("blocking", BlockingRoute())])

    def run(index: int) -> None:
        reporter = ASRRouteReporter(
            tmp_path / f"{index}.json",
            wav_bytes=f"wav {index}".encode(),
            requested_route={"route": "local-mlx-whisper"},
        )
        recovery.recover(
            f"wav {index}".encode(),
            primary_failure=TimeoutError("deadline"),
            reporter=reporter,
        )

    first = threading.Thread(target=run, args=(1,))
    second = threading.Thread(target=run, args=(2,))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    try:
        assert not concurrent_entry.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert calls == 2


def test_report_survives_failure_before_any_recovery_route(tmp_path) -> None:
    from spoke.asr_recovery import (
        ASRRecoveryError,
        ASRRouteReporter,
        SerialASRRecovery,
    )

    report_path = tmp_path / "no-route.asr.json"
    reporter = ASRRouteReporter(
        report_path,
        wav_bytes=b"wav",
        requested_route={"route": "local-mlx-whisper"},
    )

    with pytest.raises(ASRRecoveryError, match="no serial ASR recovery routes"):
        SerialASRRecovery([]).recover(
            b"wav",
            primary_failure=TimeoutError("deadline"),
            reporter=reporter,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "route_selection"
    assert report["last_trustworthy_evidence"] == "primary_failure"


def test_remote_escape_route_identity_includes_prompt_receipt() -> None:
    prompt_receipt = {
        "schema": "spoke.transcription-prompt.v1",
        "requested": True,
        "supported": True,
        "effective": True,
        "sha256": "abc",
        "char_count": 123,
        "sources": ["builtin:spoke-vocabulary-v1"],
    }

    class FakeTranscriptionClient:
        def __init__(self, *_args, **_kwargs):
            self._last_prompt_receipt = None

        def transcribe(self, _wav):
            self._last_prompt_receipt = prompt_receipt
            return "remote text"

    from spoke.asr_recovery import RemoteASREscapeClient

    client = RemoteASREscapeClient("http://whisper-sidecar:7001")
    client._client = FakeTranscriptionClient()

    assert client.transcribe(b"wav") == "remote text"
    assert client.route_identity()["prompt"] == prompt_receipt


def test_whisperkit_escape_command_is_vad_free(monkeypatch) -> None:
    from spoke.asr_recovery import WhisperKitEscapeClient

    run = MagicMock(
        return_value=MagicMock(returncode=0, stdout="transcript", stderr="")
    )
    monkeypatch.setattr("spoke.asr_recovery.subprocess.run", run)
    client = WhisperKitEscapeClient(
        cli_path="/opt/homebrew/bin/whisperkit-cli",
        model="medium.en",
    )

    assert client.transcribe(b"wav") == "transcript"

    command = run.call_args.args[0]
    assert command[command.index("--chunking-strategy") + 1] == "none"
    assert (
        command[command.index("--audio-encoder-compute-units") + 1]
        == "cpuAndNeuralEngine"
    )
    assert command[command.index("--text-decoder-compute-units") + 1] == "cpuOnly"
    assert "vad" not in command
    assert "--prompt" in command
