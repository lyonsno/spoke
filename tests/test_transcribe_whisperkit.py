"""Tests for WhisperKitClient."""

from __future__ import annotations

import io
import json
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


class TestWhisperKitClientSubprocess:
    """Test subprocess invocation without actually calling whisperkit-cli."""

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_transcribe_calls_cli_with_correct_args(self, mock_find, mock_run, monkeypatch):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello world",
            stderr="",
        )

        client = WhisperKitClient(model="medium.en")
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
        assert "--chunking-strategy" in cmd
        chunking_idx = cmd.index("--chunking-strategy")
        assert cmd[chunking_idx + 1] == "vad"
        assert "--skip-special-tokens" in cmd

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
    def test_suspicious_success_retries_and_returns_recovered_text(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        recovered_text = (
            "This recovered retry contains enough words to be plausible for a "
            "medium-length operator dictation, so the first tiny success should "
            "not be trusted as the final output."
        )
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Too short.", stderr=""),
            MagicMock(returncode=0, stdout=recovered_text, stderr=""),
        ]
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))

        client = WhisperKitClient(model="medium.en")
        result = client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert result == recovered_text
        assert mock_run.call_count == 2
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["status"] == "recovered_by_retry"
        assert report["effective_model"] == "medium.en"
        assert report["effective_cli_path"] == "/usr/local/bin/whisperkit-cli"
        assert report["audio_bytes"] > 0
        assert report["duration_seconds"] == pytest.approx(30.0)
        assert report["chosen_output"] == recovered_text
        assert report["attempts"][0]["suspicious"] is True
        assert report["attempts"][1]["suspicious"] is False
        assert Path(report["audio_path"]).exists()

    @patch("spoke.transcribe_whisperkit.subprocess.run")
    @patch("spoke.transcribe_whisperkit._find_whisperkit_cli", return_value="/usr/local/bin/whisperkit-cli")
    def test_suspicious_success_preserves_replay_bundle_when_retry_still_short(
        self,
        mock_find,
        mock_run,
        monkeypatch,
        tmp_path,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "0")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Tiny.", stderr=""),
            MagicMock(returncode=0, stdout="Still tiny.", stderr=""),
        ]
        monkeypatch.setenv("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR", str(tmp_path))

        client = WhisperKitClient(model="medium.en")
        result = client.transcribe(_make_wav_bytes(duration_s=30.0))

        assert result == "Still tiny."
        assert mock_run.call_count == 2
        reports = list(tmp_path.glob("*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["status"] == "suspicious_unrecovered"
        assert report["chosen_output"] == "Still tiny."
        assert report["attempts"][0]["stdout_len"] == 5
        assert report["attempts"][1]["stdout_len"] == 11
        assert report["attempts"][0]["suspicion_reason"] == "too_short_for_audio_duration"
        assert Path(report["audio_path"]).read_bytes() == _make_wav_bytes(duration_s=30.0)

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
        monkeypatch,
    ):
        from spoke.transcribe_whisperkit import WhisperKitClient

        monkeypatch.setenv("SPOKE_WHISPERKIT_RESIDENT", "1")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_PORT", "51234")
        monkeypatch.setenv("SPOKE_WHISPERKIT_SERVER_START_TIMEOUT", "0")
        mock_popen.return_value = MagicMock(pid=4242, poll=MagicMock(return_value=None))
        mock_urlopen.return_value = _FakeHTTPResponse({"text": "Hello resident"})

        client = WhisperKitClient(model="medium.en")
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
        assert serve_cmd[serve_cmd.index("--chunking-strategy") + 1] == "vad"
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "http://localhost:51234/v1/audio/transcriptions"
        assert request.get_method() == "POST"

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
        mock_run.assert_called_once()
        fallback_cmd = mock_run.call_args.args[0]
        assert fallback_cmd[:2] == ["/usr/local/bin/whisperkit-cli", "transcribe"]

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
    """Smoke-surface contract for replayable WhisperKit-vs-standard autopsy."""

    def test_whisperkit_smoke_env_preserves_raw_audio_for_replay(self):
        smoke_env = Path(".spoke-smoke-env").read_text()

        assert 'SPOKE_TRANSCRIPTION_MODEL="whisperkit/medium.en"' in smoke_env
        assert 'SPOKE_WHISPERKIT_RESIDENT="1"' in smoke_env
        assert 'SPOKE_AUDIO_SPOOL_ENABLED="1"' in smoke_env
