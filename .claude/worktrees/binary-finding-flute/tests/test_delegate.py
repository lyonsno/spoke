"""Tests for DontTypeAppDelegate orchestration and error paths.

Tests the wiring between layers: hold callbacks, transcription lifecycle,
generation-based stale result rejection, and env var validation.
"""

import os
from unittest.mock import MagicMock, patch


def _make_delegate(main_module, monkeypatch):
    """Create a DictateAppDelegate with mocked sub-components."""
    monkeypatch.setenv("DICTATE_WHISPER_URL", "http://test:8000")

    delegate = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
    delegate._capture = MagicMock()
    delegate._client = MagicMock()
    delegate._detector = MagicMock()
    delegate._menubar = MagicMock()
    delegate._glow = MagicMock()
    delegate._overlay = MagicMock()
    delegate._transcribing = False
    delegate._transcription_token = 0
    delegate._preview_active = False
    delegate._preview_thread = None
    delegate._preview_client = MagicMock()
    delegate._local_mode = False
    delegate._record_start_time = 0.0
    delegate._cap_fired = False
    # Stub performSelectorOnMainThread so we can call callbacks directly
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()
    return delegate


class TestHoldCallbacks:
    """Test _on_hold_start and _on_hold_end orchestration."""

    def test_hold_start_begins_recording(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._on_hold_start()

        d._capture.start.assert_called_once()
        d._menubar.set_recording.assert_called_with(True)

    def test_hold_end_stops_capture_and_spawns_thread(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"fake-wav"

        with patch.object(main_module.threading, "Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            d._on_hold_end()

        d._capture.stop.assert_called_once()
        d._menubar.set_recording.assert_called_with(False)
        MockThread.assert_called_once()
        mock_thread.start.assert_called_once()
        assert d._transcribing is True

    def test_hold_end_with_empty_audio_skips_transcription(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""

        with patch.object(main_module.threading, "Thread") as MockThread:
            d._on_hold_end()

        MockThread.assert_not_called()
        assert d._transcribing is False
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")


class TestTranscriptionToken:
    """Test generation-based stale result rejection."""

    def test_hold_end_increments_token(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"wav"
        assert d._transcription_token == 0

        with patch.object(main_module.threading, "Thread"):
            d._on_hold_end()
        assert d._transcription_token == 1

        with patch.object(main_module.threading, "Thread"):
            d._on_hold_end()
        assert d._transcription_token == 2

    def test_stale_transcription_is_discarded(self, main_module, monkeypatch):
        """If token doesn't match, result should be silently discarded."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 5
        d._transcribing = True

        # Simulate a result from an older generation
        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionComplete_({"token": 3, "text": "stale result"})

        mock_inject.assert_not_called()
        # _transcribing should NOT be cleared — current generation is still in flight
        assert d._transcribing is True

    def test_current_token_is_accepted(self, main_module, monkeypatch):
        """Result with matching token should be injected."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 5
        d._transcribing = True

        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionComplete_({"token": 5, "text": "hello world"})

        mock_inject.assert_called_once()
        assert mock_inject.call_args[0][0] == "hello world"
        assert d._transcribing is False

    def test_stale_failure_is_ignored(self, main_module, monkeypatch):
        """Failed transcription with old token should be silently ignored."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 3
        d._transcribing = True

        d.transcriptionFailed_({"token": 1})

        # Should still be transcribing — the stale failure doesn't clear state
        assert d._transcribing is True

    def test_current_failure_updates_state(self, main_module, monkeypatch):
        """Failed transcription with current token should clear state and show error."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 3
        d._transcribing = True

        d.transcriptionFailed_({"token": 3})

        assert d._transcribing is False
        d._menubar.set_status_text.assert_called_with("Error — try again")

    def test_empty_text_not_injected(self, main_module, monkeypatch):
        """Empty transcription result should not call inject_text."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 1

        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionComplete_({"token": 1, "text": ""})

        mock_inject.assert_not_called()


class TestServerCrashResilience:
    """Test that the app survives server errors without crashing."""

    def test_transcribe_worker_catches_connection_error(self, main_module, monkeypatch):
        """Server going down mid-request should not crash the app."""
        import httpx

        d = _make_delegate(main_module, monkeypatch)
        d._client.transcribe.side_effect = httpx.ConnectError("Connection refused")

        # Call the worker directly (normally runs on background thread)
        d._transcribe_worker(b"wav", token=1)

        # Should have posted failure to main thread, not crashed
        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once()
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionFailed:"
        assert call_args[0][1]["token"] == 1

    def test_transcribe_worker_catches_http_error(self, main_module, monkeypatch):
        """HTTP 500 from server should not crash the app."""
        import httpx

        d = _make_delegate(main_module, monkeypatch)
        d._client.transcribe.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=MagicMock()
        )

        d._transcribe_worker(b"wav", token=1)

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once()
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionFailed:"

    def test_transcribe_worker_catches_json_decode_error(self, main_module, monkeypatch):
        """Server returning invalid JSON should not crash the app."""
        import json

        d = _make_delegate(main_module, monkeypatch)
        d._client.transcribe.side_effect = json.JSONDecodeError("", "", 0)

        d._transcribe_worker(b"wav", token=1)

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once()
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionFailed:"


class TestHoldMsBounds:
    """Test that hold_ms rejects zero and negative values."""

    def test_zero_hold_ms_exits(self, main_module, monkeypatch):
        """DICTATE_HOLD_MS=0 should be rejected."""
        monkeypatch.setenv("DICTATE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("DICTATE_HOLD_MS", "0")
        import pytest

        with pytest.raises(SystemExit) as exc_info:
            d = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
            d.init()
        assert exc_info.value.code == 1

    def test_negative_hold_ms_exits(self, main_module, monkeypatch):
        """DICTATE_HOLD_MS=-500 should be rejected."""
        monkeypatch.setenv("DICTATE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("DICTATE_HOLD_MS", "-500")
        import pytest

        with pytest.raises(SystemExit) as exc_info:
            d = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
            d.init()
        assert exc_info.value.code == 1


class TestEnvValidation:
    """Test environment variable validation in DictateAppDelegate.init."""

    def test_missing_whisper_url_uses_local(self, main_module, monkeypatch):
        """Missing DICTATE_WHISPER_URL should fall back to local transcription."""
        monkeypatch.delenv("DICTATE_WHISPER_URL", raising=False)
        d = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
        result = d.init()
        assert result is not None
        assert isinstance(d._client, main_module.LocalTranscriptionClient)

    def test_invalid_hold_ms_exits(self, main_module, monkeypatch):
        """Non-integer DICTATE_HOLD_MS should sys.exit(1)."""
        monkeypatch.setenv("DICTATE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("DICTATE_HOLD_MS", "not-a-number")
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            d = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
            d.init()
        assert exc_info.value.code == 1

    def test_valid_config_succeeds(self, main_module, monkeypatch):
        """Valid env vars should create a delegate without error."""
        monkeypatch.setenv("DICTATE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("DICTATE_HOLD_MS", "300")
        d = main_module.DontTypeAppDelegate.__new__(main_module.DontTypeAppDelegate)
        result = d.init()
        assert result is not None
