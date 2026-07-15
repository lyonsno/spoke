"""Tests for WhisperKitClient."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class TestWhisperKitClientInterface:
    """Contract tests — no real whisperkit-cli required."""

    def test_empty_bytes_returns_empty_string(self):
        from spoke.transcribe_whisperkit import WhisperKitClient

        client = WhisperKitClient(model="base.en")
        assert client.transcribe(b"") == ""

    def test_unload_is_noop(self):
        from spoke.transcribe_whisperkit import WhisperKitClient

        client = WhisperKitClient()
        client.unload()  # Should not raise

    def test_is_loaded_always_false(self):
        from spoke.transcribe_whisperkit import WhisperKitClient

        client = WhisperKitClient()
        assert client.is_loaded is False

    def test_close_is_safe(self):
        from spoke.transcribe_whisperkit import WhisperKitClient

        client = WhisperKitClient()
        client.close()  # Should not raise

    def test_default_model_is_medium_en(self):
        from spoke.transcribe_whisperkit import WhisperKitClient

        client = WhisperKitClient()
        assert client._model == "medium.en"

    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_prepare_starts_resident_server_before_first_transcription(
        self,
        mock_find,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        client = WhisperKitClient(model="medium.en")

        with patch.object(
            client,
            "_ensure_resident_server",
            return_value=("http://localhost:51232", client._cli_path, 4240),
        ) as ensure_server:
            client.prepare()

        ensure_server.assert_called_once_with()

    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_prepare_failure_preserves_cli_fallback_for_app_warmup(
        self,
        mock_find,
        monkeypatch,
        caplog,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        client = WhisperKitClient(model="medium.en")

        with patch.object(
            client,
            "_ensure_resident_server",
            side_effect=TimeoutError("listener ownership timed out"),
        ):
            client.prepare()

        assert "resident preload failed" in caplog.text.lower()
        assert client.last_route_report == {
            "requested_route": "resident-server",
            "effective_route": None,
            "fallback_reason": (
                "preload:TimeoutError:listener ownership timed out"
            ),
            "status": "preload_failed",
            "model": "medium.en",
        }


class TestWhisperKitClientSubprocess:
    """Test subprocess invocation without actually calling whisperkit-cli."""

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_transcribe_calls_cli_with_correct_args(self, mock_find, mock_run, monkeypatch):
        from spoke.transcribe_whisperkit import WhisperKitClient
        from spoke.transcription_prompt import TranscriptionPromptProvider

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        prompt_provider = TranscriptionPromptProvider(
            path=None,
            inline="Kaminos, Trellis2MLX.",
            include_builtin=False,
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello world",
            stderr="",
        )

        client = WhisperKitClient(model="medium.en", prompt_provider=prompt_provider)
        result = client.transcribe(_make_wav_bytes())

        assert result == "Hello world"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "/usr/local/bin/whisperkit-cli"
        assert "transcribe" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "medium.en"
        assert "--audio-encoder-compute-units" in cmd
        assert "cpuAndNeuralEngine" in cmd
        assert "--text-decoder-compute-units" in cmd
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuOnly"
        assert "--chunking-strategy" in cmd
        chunking_idx = cmd.index("--chunking-strategy")
        assert cmd[chunking_idx + 1] == "none"
        assert cmd[cmd.index("--prompt") + 1] == "Kaminos, Trellis2MLX."
        assert "--skip-special-tokens" in cmd
        assert client.last_prompt_receipt["requested"] is True
        assert client.last_prompt_receipt["supported"] is True
        assert client.last_prompt_receipt["effective"] is True

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_transcribe_nonzero_exit_fails_loud_after_preserving_audio(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Model not found",
        )

        client = WhisperKitClient(model="nonexistent")
        with pytest.raises(RuntimeError, match="whisperkit-cli failed with exit 1"):
            client.transcribe(_make_wav_bytes())

        report = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert report["status"] == "failed"
        assert report["terminal_error"] == "cli_exit:1:Model not found"

    @patch(
        "spoke.transcribe_whisperkit._record_whisperkit_terminal_failure_bundle",
        return_value=None,
    )
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch(
        "spoke.transcribe_whisperkit._find_whisperkit_cli",
        return_value="/usr/local/bin/whisperkit-cli",
    )
    def test_terminal_identity_survives_failure_bundle_persistence_failure(
        self,
        mock_find,
        mock_run,
        mock_record_bundle,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        mock_run.side_effect = subprocess.TimeoutExpired("whisperkit-cli", 30.0)

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            client.transcribe(_make_wav_bytes(duration_s=30.0))

        mock_record_bundle.assert_called_once()
        assert caught.value.terminal_asr_failure is True
        assert caught.value.failure_bundle_persisted is False
        assert getattr(caught.value, "failure_bundle_path", None) is None

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_suspicious_cli_output_is_terminal_without_identical_retry(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        mock_run.return_value = MagicMock(returncode=0, stdout="Too short.", stderr="")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="suspicious output"):
            client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert mock_run.call_count == 1
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["status"] == "failed"
        assert report["effective_model"] == "medium.en"
        assert report["effective_cli_path"] == "/usr/local/bin/whisperkit-cli"
        assert report["audio_bytes"] > 0
        assert report["duration_seconds"] == pytest.approx(30.0)
        assert report["chosen_output"] == "Too short."
        assert len(report["attempts"]) == 1
        assert report["attempts"][0]["suspicious"] is True
        assert Path(report["audio_path"]).exists()

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch(
        "spoke.transcribe_whisperkit._find_whisperkit_cli",
        return_value="/usr/local/bin/whisperkit-cli",
    )
    def test_long_audio_suspicious_output_fails_loud_with_one_authoritative_bundle(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_run.return_value = MagicMock(returncode=0, stdout="Too short.", stderr="")
        wav_bytes = _make_wav_bytes(duration_s=30.0)

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="suspicious output") as exc_info:
            client.transcribe(wav_bytes)

        assert mock_run.call_count == 1
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["status"] == "failed"
        assert report["terminal_error"].startswith("cli_suspicious_output:")
        assert report["chosen_output"] == "Too short."
        assert report["attempts"][0]["suspicion_reason"]
        assert Path(report["audio_path"]).read_bytes() == wav_bytes
        assert exc_info.value.failure_bundle_path == reports[0]

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_compute_unit_overrides_reach_one_shot_cli(
        self,
        mock_find,
        mock_run,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS", "all")
        monkeypatch.setenv("SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS", "cpuAndGPU")
        mock_run.return_value = MagicMock(returncode=0, stdout="Configured route", stderr="")

        client = WhisperKitClient(model="medium.en")
        result = client.transcribe(_make_wav_bytes())

        assert result == "Configured route"
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("--audio-encoder-compute-units") + 1] == "all"
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuAndGPU"

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_explicit_timeout_reaches_cli_and_route_receipt(
        self,
        mock_find,
        mock_run,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", "17")
        mock_run.return_value = MagicMock(returncode=0, stdout="Bounded route", stderr="")

        client = WhisperKitClient(model="medium.en")
        assert client.transcribe(_make_wav_bytes()) == "Bounded route"

        assert mock_run.call_args.kwargs["timeout"] == 17.0
        assert client.last_route_report["timeout_seconds"] == 17.0

    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value=None)
    def test_transcribe_missing_cli_fails_loud_after_preserving_audio(
        self,
        mock_find,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        client = WhisperKitClient()
        with pytest.raises(RuntimeError, match="whisperkit-cli not found"):
            client.transcribe(_make_wav_bytes())

        report = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert report["status"] == "failed"
        assert report["terminal_error"] == "cli_missing"

    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_available_when_cli_found(self, mock_find):
        from spoke.transcribe_whisperkit import WhisperKitClient

        assert WhisperKitClient.available() is True

    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value=None)
    def test_not_available_when_cli_missing(self, mock_find):
        from spoke.transcribe_whisperkit import WhisperKitClient

        assert WhisperKitClient.available() is False


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self._body


class TestWhisperKitResidentServer:
    def test_tcp_readiness_rejects_listener_owned_by_another_process(self):
        from spoke.transcribe_whisperkit import _wait_for_tcp_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("localhost", 0))
            listener.listen()
            port = listener.getsockname()[1]
            spawned_process = MagicMock(pid=os.getpid() + 100_000)
            spawned_process.poll.return_value = None

            assert _wait_for_tcp_port(
                "localhost",
                port,
                0,
                expected_pid=spawned_process.pid,
                process=spawned_process,
            ) is False

    @patch("spoke.transcribe_whisperkit._tcp_listener_owner_pids", return_value=set())
    def test_tcp_readiness_fails_closed_when_listener_owner_lookup_fails(
        self,
        mock_owners,
    ):
        from spoke.transcribe_whisperkit import _wait_for_tcp_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("localhost", 0))
            listener.listen()
            port = listener.getsockname()[1]
            process = MagicMock(pid=os.getpid())
            process.poll.return_value = None

            assert _wait_for_tcp_port(
                "localhost",
                port,
                0,
                expected_pid=process.pid,
                process=process,
            ) is False

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_prepare_rejects_listener_when_spawned_server_has_exited(
        self,
        mock_find,
        mock_popen,
        mock_wait,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51231")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_LOG", str(tmp_path / "server.log"))
        mock_popen.return_value = MagicMock(pid=4239, poll=MagicMock(return_value=48))

        client = WhisperKitClient(model="medium.en")

        with pytest.raises(RuntimeError, match="exited before owning listener"):
            client._ensure_resident_server()

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_server_is_default_for_whisperkit_client(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.delenv("SPOKE_WHISPERKIT_RESIDENT", raising=False)
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51233")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "0")
        mock_popen.return_value = MagicMock(pid=4241, poll=MagicMock(return_value=None))
        mock_urlopen.return_value = _FakeHTTPResponse({"text": "Default resident"})

        client = WhisperKitClient(model="medium.en")
        result = client.transcribe(_make_wav_bytes())

        assert result == "Default resident"
        mock_run.assert_not_called()
        serve_cmd = mock_popen.call_args.args[0]
        assert serve_cmd[:2] == ["/usr/local/bin/whisperkit-cli", "serve"]

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_mode_starts_server_and_posts_audio_without_transcribe_cli(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient
        from spoke.transcription_prompt import TranscriptionPromptProvider

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51234")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_CHUNKING_STRATEGY", "none")
        monkeypatch.setenv(
            "SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS",
            "cpuAndNeuralEngine",
        )
        monkeypatch.setenv("SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS", "cpuOnly")
        mock_popen.return_value = MagicMock(pid=4242, poll=MagicMock(return_value=None))
        mock_urlopen.return_value = _FakeHTTPResponse({"text": "Hello resident"})
        prompt_provider = TranscriptionPromptProvider(
            path=None,
            inline="Kaminos, WhisperKit.",
            include_builtin=False,
        )

        client = WhisperKitClient(model="medium.en", prompt_provider=prompt_provider)
        result = client.transcribe(_make_wav_bytes())

        assert result == "Hello resident"
        mock_run.assert_not_called()
        mock_popen.assert_called_once()
        serve_cmd = mock_popen.call_args.args[0]
        assert serve_cmd[:2] == ["/usr/local/bin/whisperkit-cli", "serve"]
        assert "--model" in serve_cmd
        assert serve_cmd[serve_cmd.index("--model") + 1] == "medium.en"
        assert "--port" in serve_cmd
        assert serve_cmd[serve_cmd.index("--port") + 1] == "51234"
        assert "--chunking-strategy" in serve_cmd
        assert serve_cmd[serve_cmd.index("--chunking-strategy") + 1] == "none"
        assert (
            serve_cmd[serve_cmd.index("--audio-encoder-compute-units") + 1]
            == "cpuAndNeuralEngine"
        )
        assert (
            serve_cmd[serve_cmd.index("--text-decoder-compute-units") + 1]
            == "cpuOnly"
        )
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:51234/v1/audio/transcriptions"
        assert request.get_method() == "POST"
        assert b'name="prompt"' in request.data
        assert b"Kaminos, WhisperKit." in request.data
        assert client.last_route_report["requested_route"] == "resident-server"
        assert client.last_route_report["effective_route"] == "resident-server"
        assert client.last_route_report["effective_chunking_strategy"] == "none"
        assert (
            client.last_route_report["audio_encoder_compute_units"]
            == "cpuAndNeuralEngine"
        )
        assert client.last_route_report["text_decoder_compute_units"] == "cpuOnly"
        assert client.last_prompt_receipt["effective"] is True

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_compute_unit_overrides_reach_resident_serve_command(
        self,
        mock_find,
        mock_popen,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS", "all")
        monkeypatch.setenv("SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS", "cpuAndGPU")
        mock_popen.return_value = MagicMock(pid=4245, poll=MagicMock(return_value=None))

        client = WhisperKitClient(model="medium.en")
        client.prepare()

        cmd = mock_popen.call_args.args[0]
        assert cmd[cmd.index("--audio-encoder-compute-units") + 1] == "all"
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuAndGPU"

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_invalid_compute_units_warn_and_use_split_defaults(
        self,
        mock_find,
        mock_popen,
        mock_wait,
        monkeypatch,
        caplog,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS", "turbo")
        monkeypatch.setenv("SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS", "warp")
        mock_popen.return_value = MagicMock(pid=4246, poll=MagicMock(return_value=None))

        client = WhisperKitClient(model="medium.en")
        client.prepare()

        cmd = mock_popen.call_args.args[0]
        assert cmd[cmd.index("--audio-encoder-compute-units") + 1] == "cpuAndNeuralEngine"
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuOnly"
        assert "Invalid SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS" in caplog.text
        assert "Invalid SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS" in caplog.text

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_terminal_close_serializes_with_inflight_resident_spawn(
        self,
        mock_find,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51238")
        spawn_entered = threading.Event()
        release_spawn = threading.Event()
        proc = MagicMock(pid=5003)
        proc.poll.return_value = None
        proc.wait.return_value = 0

        def blocked_spawn(*args, **kwargs):
            spawn_entered.set()
            assert release_spawn.wait(timeout=2)
            return proc

        client = WhisperKitClient(model="medium.en")
        prepare_errors = []

        def ensure_server():
            try:
                client._ensure_resident_server()
            except Exception as exc:  # pragma: no cover - asserted below
                prepare_errors.append(exc)

        with patch("spoke.transcribe_whisperkit.subprocess.Popen", side_effect=blocked_spawn):
            prepare_thread = threading.Thread(target=ensure_server)
            prepare_thread.start()
            assert spawn_entered.wait(timeout=2)

            close_thread = threading.Thread(target=client.close)
            close_thread.start()
            time.sleep(0.05)
            release_spawn.set()

            prepare_thread.join(timeout=2)
            close_thread.join(timeout=2)

        assert prepare_errors == []
        assert not prepare_thread.is_alive()
        assert not close_thread.is_alive()
        proc.terminate.assert_called_once_with()
        assert client._server_proc is None

        with pytest.raises(RuntimeError, match="client is closed"):
            client._ensure_resident_server()

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_unload_stops_resident_but_allows_lazy_restart(
        self,
        mock_find,
        mock_popen,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51239")
        first_proc = MagicMock(pid=5004)
        first_proc.poll.return_value = None
        first_proc.wait.return_value = 0
        second_proc = MagicMock(pid=5005)
        second_proc.poll.return_value = None
        mock_popen.side_effect = [first_proc, second_proc]

        client = WhisperKitClient(model="medium.en")
        client.prepare()
        client.unload()
        client.prepare()

        first_proc.terminate.assert_called_once_with()
        assert mock_popen.call_count == 2
        assert client._server_proc is second_proc

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_suspicious_output_is_terminal_without_identical_retry(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51237")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_popen.return_value = MagicMock(pid=4244, poll=MagicMock(return_value=None))
        mock_urlopen.return_value = _FakeHTTPResponse({"text": "Too short."})

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="suspicious output"):
            client.transcribe(_make_wav_bytes(duration_s=30.0))

        mock_urlopen.assert_called_once()
        mock_run.assert_not_called()
        report_path = next(tmp_path.glob("*.json"))
        report = json.loads(report_path.read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["effective_route"] == "resident-server"
        assert report["status"] == "failed"
        assert len(report["attempts"]) == 1

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch(
        "spoke.transcribe_whisperkit._find_whisperkit_cli",
        return_value="/usr/local/bin/whisperkit-cli",
    )
    def test_resident_long_audio_empty_output_is_terminal_failure(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51242")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_popen.return_value = MagicMock(pid=4250, poll=MagicMock(return_value=None))
        mock_urlopen.return_value = _FakeHTTPResponse({"text": ""})
        wav_bytes = _make_wav_bytes(duration_s=30.0)

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="suspicious output"):
            client.transcribe(wav_bytes)

        mock_urlopen.assert_called_once()
        mock_run.assert_not_called()
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["effective_route"] == "resident-server"
        assert report["chosen_output"] == ""
        assert Path(report["audio_path"]).read_bytes() == wav_bytes

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_failure_falls_back_to_cli_with_loud_reason(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
        caplog,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51235")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "0")
        mock_popen.return_value = MagicMock(pid=4243, poll=MagicMock(return_value=None))
        mock_urlopen.side_effect = urllib.error.URLError("server unavailable")
        mock_run.return_value = MagicMock(returncode=0, stdout="Fallback text", stderr="")

        client = WhisperKitClient(model="medium.en")
        result = client.transcribe(_make_wav_bytes())

        assert result == "Fallback text"
        assert "falling back to CLI subprocess" in caplog.text
        assert "requested_route=resident-server" in caplog.text
        assert "effective_route=cli-subprocess" in caplog.text
        assert "server unavailable" in caplog.text
        mock_run.assert_called_once()
        fallback_cmd = mock_run.call_args.args[0]
        assert fallback_cmd[:2] == ["/usr/local/bin/whisperkit-cli", "transcribe"]
        assert client.last_route_report["requested_route"] == "resident-server"
        assert client.last_route_report["effective_route"] == "cli-subprocess"
        assert "server unavailable" in client.last_route_report["fallback_reason"]

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_timeout_cancels_owned_server_without_duplicate_cli_decode(
        self,
        mock_find,
        mock_urlopen,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", "30")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_urlopen.side_effect = TimeoutError("timed out")
        mock_run.side_effect = AssertionError("duplicate CLI decode must not start")
        proc = MagicMock(pid=4249)
        proc.poll.return_value = None
        proc.wait.return_value = 0

        client = WhisperKitClient(model="medium.en")
        client._server_proc = proc
        client._server_url = "http://localhost:51240"
        client._server_port = 51240
        client._server_ready = True
        wav_bytes = _make_wav_bytes(duration_s=30.0)

        with pytest.raises(TimeoutError, match="resident request exceeded"):
            client.transcribe(wav_bytes)

        mock_run.assert_not_called()
        proc.terminate.assert_called_once_with()
        report_path = next(tmp_path.glob("*.json"))
        report = json.loads(report_path.read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["status"] == "failed"
        assert report["requested_route"] == "resident-server"
        assert report["effective_route"] == "resident-server"
        assert report["fallback_attempted"] is False
        assert "TimeoutError:timed out" in report["terminal_error"]
        assert Path(report["audio_path"]).read_bytes() == wav_bytes

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit.time.monotonic")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_fast_resident_failure_shares_total_deadline_with_cli_fallback(
        self,
        mock_find,
        mock_monotonic,
        mock_urlopen,
        mock_run,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_URL", "http://localhost:51241")
        monkeypatch.setenv("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", "30")
        mock_monotonic.side_effect = [100.0, 105.0, 105.0, 105.0, 105.0]
        mock_urlopen.side_effect = urllib.error.URLError("server unavailable")
        mock_run.return_value = MagicMock(returncode=0, stdout="Fallback text", stderr="")

        client = WhisperKitClient(model="medium.en")
        assert client.transcribe(_make_wav_bytes()) == "Fallback text"

        assert mock_run.call_args.kwargs["timeout"] == pytest.approx(25.0)

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port")
    @patch(
        "spoke.transcribe_whisperkit._find_whisperkit_cli",
        return_value="/usr/local/bin/whisperkit-cli",
    )
    def test_resident_startup_is_capped_by_total_transcription_deadline(
        self,
        mock_find,
        mock_wait,
        mock_popen,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        clock = [100.0]

        def exhaust_deadline(*args, **kwargs):
            clock[0] = 105.0
            return False

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", "5")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "30")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_LOG", str(tmp_path / "server.log"))
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path / "failures"))
        monkeypatch.setattr("spoke.transcribe_whisperkit.time.monotonic", lambda: clock[0])
        mock_wait.side_effect = exhaust_deadline
        proc = MagicMock(pid=4251)
        proc.poll.return_value = None
        proc.wait.return_value = 0
        mock_popen.return_value = proc

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(TimeoutError, match="deadline was exhausted"):
            client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert mock_wait.call_args.args[2] == pytest.approx(5.0)
        mock_run.assert_not_called()
        report = json.loads(next((tmp_path / "failures").glob("*.json")).read_text())
        assert report["requested_route"] == "resident-server"
        assert report["effective_route"] == "resident-server-startup"
        assert report["fallback_attempted"] is False
        assert report["effective_server_url"]
        assert report["server_pid"] == 4251

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_terminal_cli_timeout_preserves_audio_and_route_report(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", "30")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["whisperkit-cli", "transcribe"],
            timeout=30.0,
        )
        wav_bytes = _make_wav_bytes(duration_s=30.0)

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(subprocess.TimeoutExpired):
            client.transcribe(wav_bytes)

        lease_snapshot = {
            "lease_id": "spoke-live-timeout",
            "requested": True,
            "effective": True,
            "state": "effective",
            "scheduling_posture": "lease-effective",
        }
        client.augment_last_failure_bundle(
            pathway="text",
            lease_snapshot=lease_snapshot,
        )

        report = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["requested_route"] == "cli-subprocess"
        assert report["effective_route"] == "cli-subprocess"
        assert report["fallback_attempted"] is False
        assert "TimeoutExpired" in report["terminal_error"]
        assert report["pathway"] == "text"
        assert report["lease"] == lease_snapshot
        assert Path(report["audio_path"]).read_bytes() == wav_bytes

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_successful_next_utterance_clears_stale_failure_bundle_pointer(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_run.side_effect = [
            subprocess.TimeoutExpired(["whisperkit-cli", "transcribe"], timeout=30.0),
            MagicMock(returncode=0, stdout="Recovered", stderr=""),
        ]
        client = WhisperKitClient(model="medium.en")

        with pytest.raises(subprocess.TimeoutExpired):
            client.transcribe(_make_wav_bytes())
        assert client.last_failure_bundle_path is not None

        assert client.transcribe(_make_wav_bytes()) == "Recovered"
        assert client.last_failure_bundle_path is None

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch(
        "spoke.transcribe_whisperkit._find_whisperkit_cli",
        return_value="/usr/local/bin/whisperkit-cli",
    )
    def test_failure_context_is_attached_by_utterance_owned_report_path(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_run.side_effect = [
            subprocess.TimeoutExpired(["whisperkit-cli", "transcribe"], timeout=30.0),
            subprocess.TimeoutExpired(["whisperkit-cli", "transcribe"], timeout=30.0),
        ]
        client = WhisperKitClient(model="medium.en")

        with pytest.raises(subprocess.TimeoutExpired) as first_failure:
            client.transcribe(_make_wav_bytes(duration_s=30.0))
        with pytest.raises(subprocess.TimeoutExpired) as second_failure:
            client.transcribe(_make_wav_bytes(duration_s=31.0))

        first_path = first_failure.value.failure_bundle_path
        second_path = second_failure.value.failure_bundle_path
        assert first_path != second_path
        client.augment_failure_bundle(
            first_path,
            pathway="text",
            lease_snapshot={"lease_id": "lease-one"},
        )
        client.augment_failure_bundle(
            second_path,
            pathway="command",
            lease_snapshot={"lease_id": "lease-two"},
        )

        first_report = json.loads(first_path.read_text())
        second_report = json.loads(second_path.read_text())
        assert first_report["pathway"] == "text"
        assert first_report["lease"]["lease_id"] == "lease-one"
        assert second_report["pathway"] == "command"
        assert second_report["lease"]["lease_id"] == "lease-two"

    def test_recovery_outcome_is_attached_to_exact_terminal_report(self, tmp_path):
        from spoke.transcribe_whisperkit import WhisperKitClient

        first_path = tmp_path / "first.json"
        second_path = tmp_path / "second.json"
        first_path.write_text('{"phase": "whisperkit_terminal_failure"}\n')
        second_path.write_text('{"phase": "whisperkit_terminal_failure"}\n')
        client = WhisperKitClient.__new__(WhisperKitClient)
        recovery = {
            "requested_route": "local-mlx-whisper",
            "effective_route": "local-mlx-whisper",
            "model": "mlx-community/whisper-large-v3-turbo",
            "decode_timeout_seconds": None,
            "eager_eval": True,
            "status": "succeeded",
            "elapsed_seconds": 17.25,
            "transcript_chars": 143,
        }

        client.record_recovery_outcome(first_path, recovery)

        assert json.loads(first_path.read_text())["recovery"] == recovery
        assert "recovery" not in json.loads(second_path.read_text())

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_failed_cli_fallback_preserves_terminal_route_identity(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_popen.return_value = MagicMock(pid=4247, poll=MagicMock(return_value=None))
        mock_urlopen.side_effect = urllib.error.URLError("resident unavailable")
        mock_run.return_value = MagicMock(returncode=7, stdout="", stderr="cli failed")

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="whisperkit-cli failed with exit 7"):
            client.transcribe(_make_wav_bytes())

        assert client.last_route_report["requested_route"] == "resident-server"
        assert client.last_route_report["effective_route"] == "cli-subprocess"
        assert client.last_route_report["status"] == "failed"
        assert "resident unavailable" in client.last_route_report["fallback_reason"]
        assert client.last_route_report["terminal_error"] == "cli_exit:7:cli failed"

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_suspicious_cli_fallback_bundle_preserves_route_identity(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))
        mock_popen.return_value = MagicMock(pid=4248, poll=MagicMock(return_value=None))
        mock_urlopen.side_effect = urllib.error.URLError("resident unavailable")
        mock_run.return_value = MagicMock(returncode=0, stdout="Too short.", stderr="")

        client = WhisperKitClient(model="medium.en")
        with pytest.raises(RuntimeError, match="suspicious output"):
            client.transcribe(_make_wav_bytes(duration_s=30.0))

        report = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert report["phase"] == "whisperkit_terminal_failure"
        assert report["status"] == "failed"
        assert report["requested_route"] == "resident-server"
        assert report["effective_route"] == "cli-subprocess"
        assert "resident unavailable" in report["fallback_reason"]

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_owned_resident_server_is_restarted_if_process_dies(
        self,
        mock_find,
        mock_urlopen,
        mock_popen,
        mock_run,
        mock_wait,
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51236")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "0")
        first_proc = MagicMock(pid=5001)
        first_proc.poll.return_value = None
        second_proc = MagicMock(pid=5002)
        second_proc.poll.return_value = None
        mock_popen.side_effect = [first_proc, second_proc]
        mock_urlopen.side_effect = [
            _FakeHTTPResponse({"text": "First"}),
            _FakeHTTPResponse({"text": "Second"}),
        ]

        client = WhisperKitClient(model="medium.en")
        assert client.transcribe(_make_wav_bytes()) == "First"
        first_proc.poll.return_value = 1
        assert client.transcribe(_make_wav_bytes()) == "Second"

        mock_run.assert_not_called()
        assert mock_popen.call_count == 2


def _import_delegate():
    """Import SpokeAppDelegate, skipping if ObjC class collision occurs."""
    try:
        from spoke.__main__ import SpokeAppDelegate
        return SpokeAppDelegate
    except Exception:
        pytest.skip("spoke.__main__ import failed (ObjC class re-registration)")


class TestWhisperKitRouting:
    """Test that the delegate routes whisperkit/ model IDs correctly."""

    def test_model_allowed_for_whisperkit(self):
        SpokeAppDelegate = _import_delegate()
        from spoke.transcribe_whisperkit import WhisperKitClient

        with patch.object(WhisperKitClient, "available", return_value=True):
            assert SpokeAppDelegate._model_allowed("whisperkit/medium.en") is True

    def test_model_disallowed_when_cli_missing(self):
        SpokeAppDelegate = _import_delegate()
        from spoke.transcribe_whisperkit import WhisperKitClient

        with patch.object(WhisperKitClient, "available", return_value=False):
            assert SpokeAppDelegate._model_allowed("whisperkit/medium.en") is False

    def test_build_client_returns_whisperkit_client(self):
        SpokeAppDelegate = _import_delegate()
        from spoke.transcribe_whisperkit import WhisperKitClient

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        client = delegate._build_client("", "whisperkit/medium.en")
        assert isinstance(client, WhisperKitClient)
        assert client._model == "medium.en"

    def test_build_client_extracts_variant(self):
        SpokeAppDelegate = _import_delegate()
        from spoke.transcribe_whisperkit import WhisperKitClient

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        client = delegate._build_client("", "whisperkit/base.en")
        assert isinstance(client, WhisperKitClient)
        assert client._model == "base.en"

    def test_role_env_overrides_saved_model_preferences(
        self,
        main_module,
        monkeypatch,
    ):
        SpokeAppDelegate = main_module.SpokeAppDelegate

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        delegate._load_model_preferences = MagicMock(
            return_value={
                "preview_model": "mlx-community/whisper-tiny.en-mlx",
                "transcription_model": "mlx-community/whisper-small.en-mlx",
            }
        )
        delegate._model_allowed = MagicMock(return_value=True)
        monkeypatch.setenv("SPOKE_TRANSCRIPTION_MODEL", "whisperkit/medium.en")
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)

        preview_model, transcription_model = delegate._resolve_model_ids()

        assert preview_model == "mlx-community/whisper-tiny.en-mlx"
        assert transcription_model == "whisperkit/medium.en"

    def test_explicit_unavailable_whisperkit_does_not_silently_become_mlx(
        self,
        main_module,
        monkeypatch,
    ):
        SpokeAppDelegate = main_module.SpokeAppDelegate
        from spoke.transcribe_whisperkit import WhisperKitClient

        delegate = SpokeAppDelegate.__new__(SpokeAppDelegate)
        monkeypatch.setenv("SPOKE_TRANSCRIPTION_MODEL", "whisperkit/medium.en")

        with patch.object(WhisperKitClient, "available", return_value=False):
            effective = delegate._sanitize_model_id(
                "whisperkit/medium.en",
                role="transcription",
            )

        assert effective == "whisperkit/medium.en"


class TestWhisperKitSmokeEnv:
    """Smoke-surface contract for the controlled split-compute comparison."""

    def test_whisperkit_smoke_env_selects_split_compute_without_vad(self):
        smoke_env = Path(".spoke-smoke-env").read_text()

        assert 'SPOKE_TRANSCRIPTION_MODEL="whisperkit/medium.en"' in smoke_env
        assert 'SPOKE_WHISPERKIT_RESIDENT="1"' in smoke_env
        assert 'SPOKE_WHISPERKIT_CHUNKING_STRATEGY="none"' in smoke_env
        assert (
            'SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS="cpuAndNeuralEngine"'
            in smoke_env
        )
        assert 'SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS="cpuOnly"' in smoke_env
        assert 'SPOKE_WHISPERKIT_TIMEOUT_SECONDS="30"' in smoke_env
        assert (
            'SPOKE_WHISPERKIT_TERMINAL_RECOVERY_MODEL='
            '"mlx-community/whisper-large-v3-turbo"'
        ) in smoke_env
        assert 'SPOKE_VAD_ENABLED="0"' in smoke_env
