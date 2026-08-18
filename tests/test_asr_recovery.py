"""Tests for the independent WhisperKit final-transcription recovery route."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import threading
import time


def test_whisperkit_recovery_records_effective_route_and_returns_stdout():
    from spoke.asr_recovery import WhisperKitRecoveryClient

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "recovered words\n", "")

    client = WhisperKitRecoveryClient(binary="/opt/test/whisperkit-cli", runner=run)

    assert client.transcribe(b"RIFF-wav") == "recovered words"
    command, kwargs = calls[0]
    assert command[0] == "/opt/test/whisperkit-cli"
    assert command[1:4] == ["transcribe", "--model", "medium.en"]
    assert "--audio-encoder-compute-units" in command
    assert command[command.index("--audio-encoder-compute-units") + 1] == (
        "cpuAndNeuralEngine"
    )
    assert command[command.index("--text-decoder-compute-units") + 1] == "cpuOnly"
    assert command[command.index("--chunking-strategy") + 1] == "none"
    assert command[command.index("--concurrent-worker-count") + 1] == "1"
    assert "timeout" not in kwargs
    assert client.last_route == (
        "whisperkit-cli[/opt/test/whisperkit-cli]:medium.en:"
        "encoder=cpuAndNeuralEngine:decoder=cpuOnly"
    )


def test_whisperkit_recovery_honors_explicit_binary_env(monkeypatch):
    from spoke.asr_recovery import WhisperKitRecoveryClient

    monkeypatch.setenv(
        "SPOKE_WHISPERKIT_RECOVERY_BINARY", "/opt/custom/bin/whisperkit-cli"
    )
    monkeypatch.setattr("spoke.asr_recovery.shutil.which", lambda _name: None)

    client = WhisperKitRecoveryClient()

    assert client._binary == "/opt/custom/bin/whisperkit-cli"


def test_whisperkit_recovery_finds_known_install_with_minimal_path(monkeypatch):
    import spoke.asr_recovery as asr_recovery
    from spoke.asr_recovery import WhisperKitRecoveryClient

    known_binary = "/opt/test-known/bin/whisperkit-cli"
    monkeypatch.delenv("SPOKE_WHISPERKIT_RECOVERY_BINARY", raising=False)
    monkeypatch.setattr(asr_recovery.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        asr_recovery, "_KNOWN_BINARY_PATHS", (known_binary,), raising=False
    )
    monkeypatch.setattr(
        Path, "is_file", lambda path: str(path) == known_binary
    )
    monkeypatch.setattr(
        os, "access", lambda path, mode: str(path) == known_binary and mode == os.X_OK
    )

    client = WhisperKitRecoveryClient()

    assert client._binary == known_binary


def test_whisperkit_recovery_fails_loud_on_blank_output():
    from spoke.asr_recovery import WhisperKitRecoveryClient

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "   \n", "")

    client = WhisperKitRecoveryClient(binary="/opt/test/whisperkit-cli", runner=run)

    try:
        client.transcribe(b"RIFF-wav")
    except RuntimeError as exc:
        assert "blank transcript" in str(exc)
    else:
        raise AssertionError("blank WhisperKit output was accepted")


def test_whisperkit_recovery_serializes_processes():
    from spoke.asr_recovery import WhisperKitRecoveryClient

    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def run(command, **kwargs):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0, "recovered\n", "")

    first = WhisperKitRecoveryClient(binary="/opt/test/whisperkit-cli", runner=run)
    second = WhisperKitRecoveryClient(binary="/opt/test/whisperkit-cli", runner=run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda client: client.transcribe(b"RIFF-wav"), (first, second)))

    assert results == ["recovered", "recovered"]
    assert maximum_active == 1
