"""Tests for WhisperKitClient."""

from __future__ import annotations

import io
import json
import os
import socket
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
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuAndNeuralEngine"
        assert "--chunking-strategy" in cmd
        chunking_idx = cmd.index("--chunking-strategy")
        assert cmd[chunking_idx + 1] == "vad"
        assert cmd[cmd.index("--prompt") + 1] == "Kaminos, Trellis2MLX."
        assert "--skip-special-tokens" in cmd
        assert client.last_prompt_receipt["requested"] is True
        assert client.last_prompt_receipt["supported"] is True
        assert client.last_prompt_receipt["effective"] is True

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_transcribe_returns_empty_on_failure(self, mock_find, mock_run, monkeypatch):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Model not found",
        )

        client = WhisperKitClient(model="nonexistent")
        result = client.transcribe(_make_wav_bytes())

        assert result == ""

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_suspicious_success_is_preserved_without_identical_retry(
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
        result = client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert result == "Too short."
        assert mock_run.call_count == 1
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["status"] == "suspicious_unrecovered"
        assert report["effective_model"] == "medium.en"
        assert report["effective_cli_path"] == "/usr/local/bin/whisperkit-cli"
        assert report["audio_bytes"] > 0
        assert report["duration_seconds"] == pytest.approx(30.0)
        assert report["chosen_output"] == "Too short."
        assert len(report["attempts"]) == 1
        assert report["attempts"][0]["suspicious"] is True
        assert Path(report["audio_path"]).exists()

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
    def test_transcribe_returns_empty_when_cli_missing(self, mock_find, monkeypatch):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        client = WhisperKitClient()
        result = client.transcribe(_make_wav_bytes())

        assert result == ""

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
    def test_invalid_compute_units_warn_and_use_ane_defaults(
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
        assert cmd[cmd.index("--text-decoder-compute-units") + 1] == "cpuAndNeuralEngine"
        assert "Invalid SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS" in caplog.text
        assert "Invalid SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS" in caplog.text

    @patch("spoke.transcribe_whisperkit._wait_for_tcp_port", return_value=True)
    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit.subprocess.Popen")
    @patch("urllib.request.urlopen")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_resident_suspicious_success_is_not_identically_retried(
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
        result = client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert result == "Too short."
        mock_urlopen.assert_called_once()
        mock_run.assert_not_called()
        report_path = next(tmp_path.glob("*.json"))
        report = json.loads(report_path.read_text())
        assert report["mode"] == "resident-server"
        assert report["status"] == "suspicious_unrecovered"
        assert len(report["attempts"]) == 1

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
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        mock_popen.return_value = MagicMock(pid=4247, poll=MagicMock(return_value=None))
        mock_urlopen.side_effect = urllib.error.URLError("resident unavailable")
        mock_run.return_value = MagicMock(returncode=7, stdout="", stderr="cli failed")

        client = WhisperKitClient(model="medium.en")
        assert client.transcribe(_make_wav_bytes()) == ""

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
        assert client.transcribe(_make_wav_bytes(duration_s=30.0)) == "Too short."

        report = json.loads(next(tmp_path.glob("*.json")).read_text())
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
        assert 'SPOKE_VAD_ENABLED="0"' in smoke_env
