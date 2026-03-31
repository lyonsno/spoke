"""Tests for SpokeAppDelegate orchestration and error paths.

Tests the wiring between layers: hold callbacks, transcription lifecycle,
generation-based stale result rejection, and env var validation.
"""

import logging
import os
import json
import time
from unittest.mock import MagicMock, call, patch


def _make_delegate(main_module, monkeypatch):
    """Create a SpokeAppDelegate with mocked sub-components."""
    monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")

    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._capture = MagicMock()
    delegate._client = MagicMock(supports_streaming=False)
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
    delegate._transcribe_start = time.monotonic()
    delegate._last_preview_text = ""
    delegate._command_client = None
    delegate._command_overlay = None
    delegate._scene_cache = None
    delegate._tts_client = None
    delegate._command_tool_used_tts = False
    # Tray state
    delegate._tray_stack = []
    delegate._tray_index = 0
    delegate._tray_active = False
    # Recovery mode state
    delegate._pre_paste_clipboard = None
    delegate._verify_paste_text = None
    delegate._verify_paste_attempt = 0
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_pending_insert = None
    delegate._recovery_hold_active = False
    delegate._recovery_retry_pending = False
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

    def test_hold_start_capture_failure_restores_idle_ui(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._capture.start.side_effect = RuntimeError("audio dead")

        d._on_hold_start()

        d._glow.hide.assert_called_once_with()
        d._overlay.hide.assert_called_once_with()
        d._menubar.set_recording.assert_any_call(True)
        d._menubar.set_recording.assert_any_call(False)
        d._menubar.set_status_text.assert_called_with("Audio input error — try again")
        assert d._preview_active is False

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
        assert d._preview_cancelled_on_release is True

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


class TestPreviewFinalizationContract:
    """Test preview/final handoff behavior for dual-model mode."""

    def test_preview_text_update_populates_last_preview_text(
        self, main_module, monkeypatch
    ):
        """Accepted preview text should be stored for final-failure fallback."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = True

        d.previewTextUpdate_("fresh preview")

        assert d._last_preview_text == "fresh preview"
        d._overlay.set_text.assert_called_once_with("fresh preview")

    def test_preview_text_update_ignored_after_release(self, main_module, monkeypatch):
        """Late preview text should not overwrite the post-release UI state."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = False

        d.previewTextUpdate_("late preview")

        d._overlay.set_text.assert_not_called()

    def test_transcription_failed_uses_latest_preview_text_as_fallback(
        self, main_module, monkeypatch
    ):
        """If final transcription fails, the latest preview text should be pasted."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 7
        d._transcribing = True
        d._last_preview_text = "usable preview text"

        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionFailed_({"token": 7})

        mock_inject.assert_called_once()
        assert mock_inject.call_args[0][0] == "usable preview text"
        assert d._transcribing is False

    def test_transcribe_worker_streaming_preview_to_batch_final_uses_batch_final(
        self, main_module, monkeypatch
    ):
        """Streaming preview with a different batch final client should use batch finalization."""
        d = _make_delegate(main_module, monkeypatch)
        preview_thread = MagicMock()
        d._preview_thread = preview_thread
        d._preview_client = MagicMock(supports_streaming=True, has_active_stream=True)
        d._client = MagicMock(supports_streaming=False)
        d._client.transcribe.return_value = "batch final text"

        d._transcribe_worker(b"wav", token=11)

        preview_thread.join.assert_called_once_with(timeout=2.0)
        d._preview_client.finish_stream.assert_not_called()
        d._client.transcribe.assert_called_once_with(b"wav")
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionComplete:"
        assert call_args[0][1]["text"] == "batch final text"

    def test_transcribe_worker_release_cutover_skips_preview_wait_and_join(
        self, main_module, monkeypatch
    ):
        """Release-time cutover should not block on preview thread shutdown."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_cancelled_on_release = True
        d._preview_thread = MagicMock()
        d._preview_done = MagicMock()
        d._client = MagicMock(supports_streaming=False)
        d._client.transcribe.return_value = "final text"

        d._transcribe_worker(b"wav", token=21)

        d._preview_done.wait.assert_not_called()
        d._preview_thread.join.assert_not_called()
        d._client.transcribe.assert_called_once_with(b"wav")

    def test_transcribe_worker_same_streaming_client_uses_finish_stream(
        self, main_module, monkeypatch
    ):
        """Same-client streaming finalization should use finish_stream(), not batch transcribe."""
        d = _make_delegate(main_module, monkeypatch)
        preview_thread = MagicMock()
        d._preview_thread = preview_thread
        streaming_client = MagicMock(supports_streaming=True, has_active_stream=True)
        streaming_client.finish_stream.return_value = "stream final text"
        d._client = streaming_client
        d._preview_client = streaming_client

        d._transcribe_worker(b"wav", token=12)

        preview_thread.join.assert_called_once_with(timeout=2.0)
        streaming_client.finish_stream.assert_called_once_with()
        streaming_client.transcribe.assert_not_called()
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionComplete:"
        assert call_args[0][1]["text"] == "stream final text"

    def test_transcribe_worker_release_cutover_cancels_shared_preview_stream(
        self, main_module, monkeypatch
    ):
        """Release-time cutover should cancel the shared preview stream and run batch finalization."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_cancelled_on_release = True
        d._preview_thread = MagicMock()
        streaming_client = MagicMock(supports_streaming=True, has_active_stream=True)
        streaming_client.transcribe.return_value = "batch final text"
        d._client = streaming_client
        d._preview_client = streaming_client

        d._transcribe_worker(b"wav", token=22)

        streaming_client.cancel_stream.assert_called_once_with()
        streaming_client.finish_stream.assert_not_called()
        streaming_client.transcribe.assert_called_once_with(b"wav")
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "transcriptionComplete:"
        assert call_args[0][1]["text"] == "batch final text"


class TestPreviewStreamCleanup:
    """Test preview loop streaming cleanup paths (dual-model findings 4-6)."""

    def test_preview_loop_streaming_calls_finish_stream_on_separate_preview_client(
        self, main_module, monkeypatch
    ):
        """When preview and final clients differ, the preview loop's finally
        block should call finish_stream() on the preview client."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = True
        d._preview_done = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)

        preview_client = MagicMock(supports_streaming=True, has_active_stream=True)
        preview_client.feed.return_value = "preview text"
        final_client = MagicMock(supports_streaming=False)
        d._preview_client = preview_client
        d._client = final_client
        d._capture.get_new_frames.return_value = b"\x00" * 100

        call_count = 0

        def _feed(_frames):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                d._preview_active = False
            return "preview"

        preview_client.feed.side_effect = _feed

        with patch.object(main_module.time, "sleep"):
            d._preview_loop_streaming()

        preview_client.finish_stream.assert_called_once()
        d._preview_done.set.assert_called_once()

    def test_preview_loop_streaming_calls_cancel_stream_on_release(
        self, main_module, monkeypatch
    ):
        """When _preview_cancelled_on_release is True, the finally block
        should call cancel_stream() instead of finish_stream()."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = True
        d._preview_cancelled_on_release = True
        d._preview_done = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)

        preview_client = MagicMock(supports_streaming=True, has_active_stream=True)
        final_client = MagicMock(supports_streaming=False)
        d._preview_client = preview_client
        d._client = final_client
        d._capture.get_new_frames.return_value = b"\x00" * 100

        def _feed(_frames):
            d._preview_active = False
            return "preview"

        preview_client.feed.side_effect = _feed

        with patch.object(main_module.time, "sleep"):
            d._preview_loop_streaming()

        preview_client.cancel_stream.assert_called_once()
        preview_client.finish_stream.assert_not_called()
        d._preview_done.set.assert_called_once()

    def test_preview_loop_streaming_skips_cleanup_when_same_client(
        self, main_module, monkeypatch
    ):
        """When preview and final are the same client, the finally block
        should NOT call finish_stream() — the transcribe worker handles it."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = True
        d._preview_done = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)

        shared_client = MagicMock(supports_streaming=True, has_active_stream=True)
        d._preview_client = shared_client
        d._client = shared_client
        d._capture.get_new_frames.return_value = b"\x00" * 100

        def _feed(_frames):
            d._preview_active = False
            return "preview"

        shared_client.feed.side_effect = _feed

        with patch.object(main_module.time, "sleep"):
            d._preview_loop_streaming()

        shared_client.finish_stream.assert_not_called()
        shared_client.cancel_stream.assert_not_called()
        d._preview_done.set.assert_called_once()

    def test_preview_loop_streaming_signals_done_on_start_failure(
        self, main_module, monkeypatch
    ):
        """If start_stream() raises, the finally block should still signal done
        (via the batch fallback's finally)."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_active = False  # Batch fallback exits immediately
        d._preview_done = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)

        preview_client = MagicMock(supports_streaming=True)
        preview_client.start_stream.side_effect = RuntimeError("backend unavailable")
        d._preview_client = preview_client
        d._client = MagicMock(supports_streaming=False)

        with patch.object(main_module.time, "sleep"):
            d._preview_loop_streaming()

        # Even though streaming failed, the batch fallback's finally should signal done
        d._preview_done.set.assert_called()


class TestModelPreferencePersistence:
    """Test model preference save/load round-trip (dual-model finding 5)."""

    def test_save_and_load_model_preferences_round_trip(
        self, main_module, monkeypatch, tmp_path
    ):
        """Saving model preferences should create a file that loads back correctly."""
        prefs_file = tmp_path / "model_preferences.json"
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d._save_model_preferences(
            "mlx-community/whisper-base.en-mlx",
            "mlx-community/whisper-medium.en-mlx-8bit",
        )

        assert prefs_file.exists()
        loaded = d._load_model_preferences()
        assert loaded["preview_model"] == "mlx-community/whisper-base.en-mlx"
        assert loaded["transcription_model"] == "mlx-community/whisper-medium.en-mlx-8bit"

    def test_save_model_preferences_uses_atomic_write(
        self, main_module, monkeypatch, tmp_path
    ):
        """Preferences should be written via tmp+rename for atomicity."""
        prefs_file = tmp_path / "model_preferences.json"
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d._save_model_preferences("model-a", "model-b")

        # The .tmp file should not linger after a successful write
        assert not prefs_file.with_suffix(".tmp").exists()
        assert prefs_file.exists()

    def test_save_preserves_existing_local_whisper_preferences(
        self, main_module, monkeypatch, tmp_path
    ):
        """Saving model prefs should not clobber co-located local Whisper prefs."""
        import json

        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(json.dumps({
            "local_whisper_decode_timeout": 30.0,
            "local_whisper_eager_eval": True,
        }))
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d._save_model_preferences("model-a", "model-b")

        loaded = json.loads(prefs_file.read_text())
        assert loaded["preview_model"] == "model-a"
        assert loaded["transcription_model"] == "model-b"
        assert loaded["local_whisper_decode_timeout"] == 30.0
        assert loaded["local_whisper_eager_eval"] is True

    def test_save_preserves_existing_command_model_preference(
        self, main_module, monkeypatch, tmp_path
    ):
        """Saving Whisper model prefs should not clobber the assistant model choice."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(json.dumps({
            "command_model": "qwen3p5-35B-A3B",
        }))
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d._save_model_preferences("model-a", "model-b")

        loaded = json.loads(prefs_file.read_text())
        assert loaded["preview_model"] == "model-a"
        assert loaded["transcription_model"] == "model-b"
        assert loaded["command_model"] == "qwen3p5-35B-A3B"


class TestConcurrencyContract:
    """Test thread handoff and local-inference serialization."""

    def test_preview_loop_batch_uses_faster_local_preview_cadence(
        self, main_module, monkeypatch
    ):
        """Local batch preview should use the tighter startup and steady-state cadence."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_active = True
        d._capture.get_buffer.return_value = b"wav"
        call_count = 0

        def _transcribe(_wav_bytes):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                d._preview_active = False
            return "preview"

        d._preview_client.transcribe.side_effect = _transcribe

        sleeps = []

        def _sleep(seconds):
            sleeps.append(seconds)

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            with patch.object(
                main_module.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.0]
            ):
                d._preview_loop_batch()

        assert sleeps[:2] == [0.15, 0.2]

    def test_preview_loop_batch_uses_local_inference_lock_and_signals_done(
        self, main_module, monkeypatch
    ):
        """Batch preview should serialize local inference and signal completion on exit."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_active = True
        d._capture.get_buffer.return_value = b"wav"
        d._preview_done = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)

        def _transcribe(_wav_bytes):
            d._preview_active = False
            return "preview"

        d._preview_client.transcribe.side_effect = _transcribe

        with patch.object(main_module.time, "sleep"):
            d._preview_loop_batch()

        d._local_inference_lock.__enter__.assert_called_once()
        d._preview_done.set.assert_called_once_with()

    def test_preview_loop_batch_signals_done_when_transcription_raises(
        self, main_module, monkeypatch
    ):
        """Preview completion should be signaled even if the last preview pass raises."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_active = True
        d._capture.get_buffer.return_value = b"wav"
        d._preview_done = MagicMock()
        d._preview_client.transcribe.side_effect = RuntimeError("preview failed")

        def _sleep(_seconds):
            d._preview_active = False

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            d._preview_loop_batch()

        d._preview_done.set.assert_called_once_with()

    def test_transcribe_worker_waits_for_preview_done_and_uses_inference_lock(
        self, main_module, monkeypatch
    ):
        """Final transcription should wait for preview completion before local batch inference."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_thread = MagicMock()
        d._preview_done = MagicMock()
        d._preview_done.wait.return_value = True
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=None)
        d._client = MagicMock(supports_streaming=False)
        d._client.transcribe.return_value = "final text"

        d._transcribe_worker(b"wav", token=3)

        d._preview_done.wait.assert_called_once_with(timeout=2.0)
        d._local_inference_lock.__enter__.assert_called_once()
        d._client.transcribe.assert_called_once_with(b"wav")


class TestHoldMsBounds:
    """Test that hold_ms rejects zero and negative values."""

    def test_zero_hold_ms_exits(self, main_module, monkeypatch):
        """SPOKE_HOLD_MS=0 should be rejected."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("SPOKE_HOLD_MS", "0")
        import pytest

        with pytest.raises(SystemExit) as exc_info:
            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            d.init()
        assert exc_info.value.code == 1

    def test_negative_hold_ms_exits(self, main_module, monkeypatch):
        """SPOKE_HOLD_MS=-500 should be rejected."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("SPOKE_HOLD_MS", "-500")
        import pytest

        with pytest.raises(SystemExit) as exc_info:
            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            d.init()
        assert exc_info.value.code == 1


class TestModelPicker:
    """Test model selection menu and RAM guard."""

    def test_model_allowed_blocks_large_below_16gb(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 15.0)
        assert d._model_allowed("mlx-community/whisper-large-v3-turbo") is False

    def test_model_allowed_permits_large_at_16gb(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
        assert d._model_allowed("mlx-community/whisper-medium.en-mlx-8bit") is True
        assert d._model_allowed("Qwen/Qwen3-ASR-0.6B") is True
        assert d._model_allowed("mlx-community/whisper-large-v3-turbo") is True

    def test_model_allowed_permits_large_at_32gb(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 32.0)
        assert d._model_allowed("mlx-community/whisper-large-v3-turbo") is True

    def test_select_model_none_returns_list(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 15.0)
        models = d._select_model(None)
        model_ids = [m[0] for m in models]
        assert "mlx-community/whisper-tiny.en-mlx" in model_ids
        assert "mlx-community/whisper-base.en-mlx" in model_ids
        assert "mlx-community/whisper-base.en-mlx-8bit" in model_ids
        assert "mlx-community/whisper-small.en-mlx" in model_ids
        assert "mlx-community/whisper-small.en-mlx-8bit" in model_ids
        assert "mlx-community/whisper-medium.en-mlx-8bit" in model_ids
        assert "Qwen/Qwen3-ASR-0.6B" in model_ids
        assert "mlx-community/whisper-large-v3-turbo" not in model_ids

    def test_select_model_none_exposes_float16_laptop_tiers_with_labels(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
        models = d._select_model(None)

        labels_by_id = {model_id: label for model_id, label, _enabled in models}

        assert labels_by_id["mlx-community/whisper-tiny.en-mlx"] == "Tiny.en (float16)"
        assert labels_by_id["mlx-community/whisper-base.en-mlx"] == "Base.en (float16)"
        assert labels_by_id["mlx-community/whisper-small.en-mlx"] == "Small.en (float16)"
        assert labels_by_id["mlx-community/whisper-medium.en-mlx"] == "Medium.en (float16)"

    def test_select_model_none_includes_large_on_high_ram(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
        models = d._select_model(None)
        labels_by_id = {model_id: label for model_id, label, _enabled in models}
        assert "mlx-community/whisper-large-v3-turbo" in labels_by_id
        assert labels_by_id["mlx-community/whisper-large-v3-turbo"] == "v3 Large Turbo (float16)"


class TestDualModelConfiguration:
    """Test separate preview/final model selection and persistence hooks."""

    def test_init_uses_separate_preview_and_transcription_model_env_vars(
        self, main_module, monkeypatch
    ):
        """Role-specific env vars should create distinct clients when models differ."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {},
            raising=False,
        )
        monkeypatch.setenv(
            "SPOKE_PREVIEW_MODEL", "mlx-community/whisper-medium.en-mlx-8bit"
        )
        monkeypatch.setenv(
            "SPOKE_TRANSCRIPTION_MODEL", "mlx-community/whisper-large-v3-turbo"
        )

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            final_client = MagicMock(name="final_client")
            preview_client = MagicMock(name="preview_client")
            MockLocal.side_effect = [final_client, preview_client]

            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._client is final_client
        assert d._preview_client is preview_client
        assert MockLocal.call_count == 2
        assert (
            MockLocal.call_args_list[0].kwargs["model"]
            == "mlx-community/whisper-large-v3-turbo"
        )
        assert (
            MockLocal.call_args_list[1].kwargs["model"]
            == "mlx-community/whisper-medium.en-mlx-8bit"
        )

    def test_init_accepts_new_float16_whisper_model_ids(
        self, main_module, monkeypatch
    ):
        """New tiny/base/small float16 IDs should flow through the normal local init path."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.setenv("SPOKE_PREVIEW_MODEL", "mlx-community/whisper-small.en-mlx")
        monkeypatch.setenv("SPOKE_TRANSCRIPTION_MODEL", "mlx-community/whisper-tiny.en-mlx")

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            final_client = MagicMock(name="final_client")
            preview_client = MagicMock(name="preview_client")
            MockLocal.side_effect = [final_client, preview_client]

            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._client is final_client
        assert d._preview_client is preview_client
        assert MockLocal.call_count == 2
        assert (
            MockLocal.call_args_list[0].kwargs["model"]
            == "mlx-community/whisper-tiny.en-mlx"
        )
        assert (
            MockLocal.call_args_list[1].kwargs["model"]
            == "mlx-community/whisper-small.en-mlx"
        )

    def test_init_loads_persisted_local_whisper_preferences_when_env_vars_absent(
        self, main_module, monkeypatch
    ):
        """Persisted local Whisper controls should restore when env vars are unset."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {
                "decode_timeout": None,
                "eager_eval": True,
            },
            raising=False,
        )

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        result = d.init()

        assert result is not None
        assert d._local_whisper_decode_timeout is None
        assert d._local_whisper_eager_eval is True

    def test_handle_model_menu_none_exposes_local_whisper_settings_in_local_mode(
        self, main_module, monkeypatch
    ):
        """Local mode should surface local Whisper guard/affordance controls."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = False
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: True)

        model_state = d._handle_model_menu_action(None)

        assert model_state["local_whisper"]["title"] == "Local Whisper"
        assert model_state["local_whisper"]["items"] == [
            ("decode_timeout", "Decode timeout guard (30s)", True, True),
            ("eager_eval", "Stability mode (eager eval)", False, True),
        ]

    def test_handle_model_menu_none_marks_eager_eval_unavailable_when_backend_lacks_it(
        self, main_module, monkeypatch
    ):
        """Unsupported eager_eval should render as disabled instead of looking live."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = True
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: False)

        model_state = d._handle_model_menu_action(None)

        assert model_state["local_whisper"]["items"] == [
            ("decode_timeout", "Decode timeout guard (30s)", True, True),
            (
                "eager_eval",
                "Stability mode (eager eval) [mlx-whisper update needed]",
                False,
                False,
            ),
        ]

    def test_handle_model_menu_none_hides_local_whisper_settings_in_sidecar_mode(
        self, main_module, monkeypatch
    ):
        """Remote sidecar mode should not show local-only Whisper controls."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = False

        model_state = d._handle_model_menu_action(None)

        assert "local_whisper" not in model_state

    def test_handle_model_menu_none_exposes_assistant_models_when_command_enabled(
        self, main_module, monkeypatch
    ):
        """Command mode should surface an Assistant submenu with the selected OMLX model."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_model_id = "qwen3p5-35B-A3B"
        d._command_model_options = [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", True),
        ]

        model_state = d._handle_model_menu_action(None)

        assert model_state["assistant"] == {
            "selected": "qwen3p5-35B-A3B",
            "models": [
                ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
                ("qwen3-14b", "qwen3-14b", True),
            ],
        }

    def test_handle_model_menu_none_exposes_launch_targets_from_registry(
        self, main_module, monkeypatch
    ):
        """A populated launch-target registry should surface a dedicated submenu."""
        d = _make_delegate(main_module, monkeypatch)
        d._launch_target_menu_state = MagicMock(
            return_value={
                "title": "Launch Target",
                "selected": "main",
                "items": [
                    ("main", "Main", True),
                    ("smoke", "Smoke", True),
                ],
            }
        )

        model_state = d._handle_model_menu_action(None)

        assert model_state["launch_target"] == {
            "title": "Launch Target",
            "selected": "main",
            "items": [
                ("main", "Main", True),
                ("smoke", "Smoke", True),
            ],
        }

    def test_launch_target_menu_state_prefers_persisted_registry_selection(
        self, main_module, monkeypatch
    ):
        """Launch target menu should show the persisted registry selection, not the current checkout."""
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(
            main_module,
            "iter_launch_targets",
            lambda: [
                {"id": "main", "label": "Main", "enabled": True},
                {"id": "dev", "label": "Dev", "enabled": True},
            ],
        )
        monkeypatch.setattr(
            main_module,
            "load_launch_target_registry",
            lambda: {"selected": "main", "targets": []},
        )
        monkeypatch.setattr(
            main_module,
            "current_launch_target_id",
            lambda _path: "dev",
        )

        assert d._launch_target_menu_state() == {
            "title": "Launch Target",
            "selected": "main",
            "items": [
                ("main", "Main", True),
                ("dev", "Dev", True),
            ],
        }

    def test_selecting_launch_target_persists_choice_and_invokes_helper(
        self, main_module, monkeypatch
    ):
        """Choosing a new launch target should save it and hand off to the launcher helper."""
        d = _make_delegate(main_module, monkeypatch)
        d._persist_launch_target_selection = MagicMock(return_value=True)
        d._invoke_launch_target_helper = MagicMock(return_value=True)

        d._handle_model_menu_action(("launch_target", "smoke"))

        d._persist_launch_target_selection.assert_called_once_with("smoke")
        d._invoke_launch_target_helper.assert_called_once_with("smoke")

    def test_toggle_local_whisper_eager_eval_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Toggling eager-eval should persist, update env, and relaunch."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = False
        d._save_local_whisper_preferences = MagicMock()
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: True)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("local_whisper", "eager_eval"))

        d._save_local_whisper_preferences.assert_called_once_with(30.0, True)
        assert os.environ["SPOKE_LOCAL_WHISPER_EAGER_EVAL"] == "1"
        assert os.environ["SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT"] == "30"
        mock_execv.assert_called_once()

    def test_toggle_local_whisper_eager_eval_is_ignored_when_backend_lacks_support(
        self, main_module, monkeypatch
    ):
        """Unsupported eager_eval should not relaunch or mutate persisted settings."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = False
        d._save_local_whisper_preferences = MagicMock()
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: False)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("local_whisper", "eager_eval"))

        d._save_local_whisper_preferences.assert_not_called()
        mock_execv.assert_not_called()

    def test_toggle_local_whisper_decode_timeout_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Toggling the timeout guard should flip between default and disabled."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = False
        d._save_local_whisper_preferences = MagicMock()

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("local_whisper", "decode_timeout"))

        d._save_local_whisper_preferences.assert_called_once_with(None, False)
        assert os.environ["SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT"] == "off"
        assert os.environ["SPOKE_LOCAL_WHISPER_EAGER_EVAL"] == "0"
        mock_execv.assert_called_once()

    def test_selecting_assistant_model_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Choosing a different assistant model should persist it and relaunch."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_model_id = "qwen3p5-35B-A3B"
        d._command_model_options = [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", True),
        ]
        d._save_command_model_preference = MagicMock(return_value=True)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("assistant", "qwen3-14b"))

        d._save_command_model_preference.assert_called_once_with("qwen3-14b")
        assert os.environ["SPOKE_COMMAND_MODEL"] == "qwen3-14b"
        mock_execv.assert_called_once()

    def test_discover_command_models_merges_server_and_local_inventory(
        self, main_module, monkeypatch, tmp_path
    ):
        """Assistant discovery should preserve server models and append curated local ones."""
        model_root = tmp_path / "models"
        curated = model_root / "lmstudio-community" / "Qwen3-4B-Instruct-2507-MLX-6bit"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors.index.json").write_text("{}")
        (model_root / "unsloth" / "Qwen3-4B-Instruct-2507-GGUF").mkdir(parents=True)
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.list_models.return_value = ["qwen3p5-35B-A3B", "qwen3-14b"]

        options = d._discover_command_models("qwen3p5-35B-A3B")

        assert options == [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", False),
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            ),
        ]

    def test_discover_command_models_drops_stale_selected_model_when_server_is_unavailable(
        self, main_module, monkeypatch, tmp_path
    ):
        """A stale selected assistant model should disappear when it is not actually available."""
        model_root = tmp_path / "models"
        curated = model_root / "lmstudio-community" / "Qwen2.5-Coder-3B-Instruct-MLX-8bit"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors").write_text("weights")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.list_models.side_effect = RuntimeError("offline")

        options = d._discover_command_models("qwen3p5-35B-A3B")

        assert options == [
            (
                "lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-8bit",
                "lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-8bit",
                False,
            ),
        ]

    def test_seed_command_model_options_prefers_curated_local_inventory(
        self, main_module, monkeypatch, tmp_path
    ):
        """Initial Assistant menu should seed from the local curated shortlist without /v1/models."""
        model_root = tmp_path / "models"
        curated = model_root / "alexgusevski" / "LFM2.5-1.2B-Nova-Function-Calling-mlx"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors.index.json").write_text("{}")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)

        options = d._seed_command_model_options("qwen3p5-35B-A3B")

        assert options == [
            (
                "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
                "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
                False,
            ),
        ]

    def test_reselecting_current_assistant_model_repairs_stale_preference_without_relaunch(
        self, main_module, monkeypatch, tmp_path
    ):
        """Re-selecting the running assistant model should heal stale prefs without relaunch."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(json.dumps({
            "command_model": "qwen3-14b",
        }))
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._command_model_id = "qwen3p5-35B-A3B"
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_command_model_preference = main_module.SpokeAppDelegate._load_command_model_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_command_model_preference = main_module.SpokeAppDelegate._save_command_model_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._relaunch = MagicMock()

        d._apply_command_model_selection("qwen3p5-35B-A3B")

        loaded = json.loads(prefs_file.read_text())
        assert loaded["command_model"] == "qwen3p5-35B-A3B"
        d._relaunch.assert_not_called()

    def test_init_shares_client_when_preview_and_transcription_models_match(
        self, main_module, monkeypatch
    ):
        """Matching role-specific selections should reuse a single client instance."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {},
            raising=False,
        )
        monkeypatch.setenv(
            "SPOKE_PREVIEW_MODEL", "mlx-community/whisper-medium.en-mlx-8bit"
        )
        monkeypatch.setenv(
            "SPOKE_TRANSCRIPTION_MODEL", "mlx-community/whisper-medium.en-mlx-8bit"
        )

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            shared_client = MagicMock(name="shared_client")
            MockLocal.return_value = shared_client

            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._client is shared_client
        assert d._preview_client is shared_client
        MockLocal.assert_called_once_with(
            model="mlx-community/whisper-medium.en-mlx-8bit",
            decode_timeout=30.0,
            eager_eval=False,
        )

    def test_init_loads_persisted_model_preferences_when_env_vars_absent(
        self, main_module, monkeypatch
    ):
        """Persisted selections should be used when role-specific env vars are unset."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {},
            raising=False,
        )
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_model_preferences",
            lambda self: {
                "preview_model": "mlx-community/whisper-medium.en-mlx-8bit",
                "transcription_model": "mlx-community/whisper-large-v3-turbo",
            },
            raising=False,
        )

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            final_client = MagicMock(name="final_client")
            preview_client = MagicMock(name="preview_client")
            MockLocal.side_effect = [final_client, preview_client]

            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._client is final_client
        assert d._preview_client is preview_client
        assert MockLocal.call_count == 2
        assert (
            MockLocal.call_args_list[0].kwargs["model"]
            == "mlx-community/whisper-large-v3-turbo"
        )
        assert (
            MockLocal.call_args_list[1].kwargs["model"]
            == "mlx-community/whisper-medium.en-mlx-8bit"
        )

    def test_init_falls_back_from_unsupported_persisted_transcription_model(
        self, main_module, monkeypatch, tmp_path
    ):
        """Unsupported saved transcription models should not abort startup."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(
            '{\n'
            '  "preview_model": "mlx-community/whisper-tiny.en-mlx",\n'
            '  "transcription_model": "mlx-community/whisper-large-v3-turbo"\n'
            '}\n'
        )
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)
        monkeypatch.setattr(main_module, "_RAM_GB", 15.0)

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            final_client = MagicMock(name="final_client")
            preview_client = MagicMock(name="preview_client")
            MockLocal.side_effect = [final_client, preview_client]

            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._transcription_model_id == "mlx-community/whisper-medium.en-mlx-8bit"
        assert d._preview_model_id == "mlx-community/whisper-tiny.en-mlx"
        assert (
            MockLocal.call_args_list[0].kwargs["model"]
            == "mlx-community/whisper-medium.en-mlx-8bit"
        )
        loaded = json.loads(prefs_file.read_text())
        assert loaded["transcription_model"] == "mlx-community/whisper-medium.en-mlx-8bit"

    def test_init_loads_persisted_command_model_when_env_var_absent(
        self, main_module, monkeypatch
    ):
        """Persisted assistant model should bootstrap the OMLX command client."""
        monkeypatch.setenv("SPOKE_COMMAND_URL", "http://omlx:8001")
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )
        with patch.object(main_module, "CommandClient") as MockCommand:
            MockCommand.return_value = MagicMock()
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[("lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit", "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit", False)],
            ) as mock_seed:
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        MockCommand.assert_called_once_with(model="qwen3-14b")
        assert d._command_model_id == "qwen3-14b"
        mock_seed.assert_called_once_with("qwen3-14b")
        assert d._command_model_options == [
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            )
        ]

    def test_init_seeds_command_model_options_without_sync_discovery(
        self, main_module, monkeypatch
    ):
        """Startup should not block on /v1/models just to seed the Assistant menu."""
        monkeypatch.setenv("SPOKE_COMMAND_URL", "http://omlx:8001")
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )

        with patch.object(main_module, "CommandClient") as MockCommand:
            command_client = MagicMock()
            MockCommand.return_value = command_client
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[("lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit", "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit", False)],
            ) as mock_seed:
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        MockCommand.assert_called_once_with(model="qwen3-14b")
        command_client.list_models.assert_not_called()
        mock_seed.assert_called_once_with("qwen3-14b")
        assert d._command_model_options == [
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            )
        ]

    def test_handle_model_menu_none_sanitizes_unsupported_selected_model(
        self, main_module, monkeypatch
    ):
        """Menu state should not advertise an unsupported current selection."""
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 15.0)
        d._preview_model_id = "mlx-community/whisper-tiny.en-mlx"
        d._transcription_model_id = "mlx-community/whisper-large-v3-turbo"

        model_state = d._handle_model_menu_action(None)

        assert model_state["transcription"]["selected"] == "mlx-community/whisper-medium.en-mlx-8bit"
        transcription_ids = [model_id for model_id, _label, _enabled in model_state["transcription"]["models"]]
        assert "mlx-community/whisper-large-v3-turbo" not in transcription_ids

    def test_switching_away_from_qwen_persists_models_across_relaunch(
        self, main_module, monkeypatch, tmp_path
    ):
        """Switching away from Qwen should survive relaunch and re-enable local Whisper controls."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(
            '{\n'
            '  "preview_model": "Qwen/Qwen3-ASR-0.6B",\n'
            '  "transcription_model": "Qwen/Qwen3-ASR-0.6B"\n'
            '}\n'
        )
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._preview_model_id = "Qwen/Qwen3-ASR-0.6B"
        d._transcription_model_id = "Qwen/Qwen3-ASR-0.6B"
        d._local_mode = True
        d._save_model_preferences = main_module.SpokeAppDelegate._save_model_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._relaunch = MagicMock()

        d._apply_model_selection(
            "mlx-community/whisper-base.en-mlx-8bit",
            "mlx-community/whisper-medium.en-mlx-8bit",
        )

        assert d._relaunch.called

        with patch.object(main_module, "LocalTranscriptionClient") as MockLocal:
            final_client = MagicMock(name="final_client")
            preview_client = MagicMock(name="preview_client")
            MockLocal.side_effect = [final_client, preview_client]

            relaunched = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = relaunched.init()

        assert result is not None
        assert (
            MockLocal.call_args_list[0].kwargs["model"]
            == "mlx-community/whisper-medium.en-mlx-8bit"
        )
        assert (
            MockLocal.call_args_list[1].kwargs["model"]
            == "mlx-community/whisper-base.en-mlx-8bit"
        )
        assert relaunched._local_whisper_controls_available() is True

    def test_sequential_model_switches_persist_across_multiple_relaunches(
        self, main_module, monkeypatch, tmp_path
    ):
        """A series of menu-driven model changes should survive each relaunch boundary."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(
            '{\n'
            '  "preview_model": "Qwen/Qwen3-ASR-0.6B",\n'
            '  "transcription_model": "Qwen/Qwen3-ASR-0.6B"\n'
            '}\n'
        )
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)

        def _fresh_delegate():
            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            d._capture = MagicMock()
            d._client_cache = {}
            d._local_inference_lock = MagicMock()
            d._local_mode = True
            d._command_client = None
            d._preview_done = MagicMock()
            d._preview_done.set = MagicMock()
            d._preview_active = False
            d._preview_thread = None
            d._transcribing = False
            d._detector = MagicMock()
            d._close_clients = MagicMock()
            return d

        # Step 1: switch transcription off Qwen.
        first = _fresh_delegate()
        first.init()
        with patch.object(main_module.os, "execv"):
            first._handle_model_menu_action(
                ("transcription", "mlx-community/whisper-small.en-mlx")
            )

        # Step 2: relaunch and switch preview off Qwen.
        second = _fresh_delegate()
        second.init()
        with patch.object(main_module.os, "execv"):
            second._handle_model_menu_action(
                ("preview", "mlx-community/whisper-tiny.en-mlx")
            )

        # Step 3: relaunch and switch transcription again.
        third = _fresh_delegate()
        third.init()
        with patch.object(main_module.os, "execv"):
            third._handle_model_menu_action(
                ("transcription", "mlx-community/whisper-medium.en-mlx")
            )

        loaded = json.loads(prefs_file.read_text())
        assert loaded["preview_model"] == "mlx-community/whisper-tiny.en-mlx"
        assert loaded["transcription_model"] == "mlx-community/whisper-medium.en-mlx"

    def test_model_switch_does_not_relaunch_on_preference_save_failure(
        self, main_module, monkeypatch
    ):
        """A failed save should not be masked by relaunching into env-only model state."""

        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)

        d = _make_delegate(main_module, monkeypatch)
        d._preview_model_id = "Qwen/Qwen3-ASR-0.6B"
        d._transcription_model_id = "Qwen/Qwen3-ASR-0.6B"
        d._save_model_preferences = MagicMock(return_value=False)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._apply_model_selection(
                "mlx-community/whisper-base.en-mlx-8bit",
                "mlx-community/whisper-medium.en-mlx-8bit",
            )

        d._save_model_preferences.assert_called_once_with(
            "mlx-community/whisper-base.en-mlx-8bit",
            "mlx-community/whisper-medium.en-mlx-8bit",
        )
        mock_execv.assert_not_called()
        assert "SPOKE_PREVIEW_MODEL" not in os.environ
        assert "SPOKE_TRANSCRIPTION_MODEL" not in os.environ
        assert "SPOKE_WHISPER_MODEL" not in os.environ

    def test_reselecting_current_models_repairs_stale_qwen_preferences(
        self, main_module, monkeypatch, tmp_path
    ):
        """Re-selecting the current runtime models should heal stale on-disk Qwen prefs."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(
            '{\n'
            '  "preview_model": "Qwen/Qwen3-ASR-0.6B",\n'
            '  "transcription_model": "Qwen/Qwen3-ASR-0.6B"\n'
            '}\n'
        )
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)
        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._preview_model_id = "mlx-community/whisper-base.en-mlx-8bit"
        d._transcription_model_id = "mlx-community/whisper-medium.en-mlx-8bit"
        d._save_model_preferences = main_module.SpokeAppDelegate._save_model_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_model_preferences = main_module.SpokeAppDelegate._load_model_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._relaunch = MagicMock()

        d._apply_model_selection(
            "mlx-community/whisper-base.en-mlx-8bit",
            "mlx-community/whisper-medium.en-mlx-8bit",
        )

        loaded = json.loads(prefs_file.read_text())
        assert loaded["preview_model"] == "mlx-community/whisper-base.en-mlx-8bit"
        assert loaded["transcription_model"] == "mlx-community/whisper-medium.en-mlx-8bit"
        d._relaunch.assert_not_called()


class TestWarmupContract:
    """Test non-blocking warmup behavior."""

    def test_setup_event_tap_starts_background_warmup_before_ready(
        self, main_module, monkeypatch
    ):
        """The app should not block the main thread on first-run model warmup."""
        d = _make_delegate(main_module, monkeypatch)
        d._detector.install.return_value = True
        d._prepare_clients = MagicMock()
        d._warmup_in_flight = False

        with patch.object(main_module.threading, "Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            d._setup_event_tap()

        d._prepare_clients.assert_not_called()
        mock_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once_with()
        d._menubar.set_status_text.assert_called_with("Loading models…")
        d._overlay.show.assert_called_once_with()
        d._overlay.set_text.assert_called_once_with(
            "Loading models...\nFirst launch may download selected models."
        )

    def test_background_warmup_dispatches_success_to_main_thread(
        self, main_module, monkeypatch
    ):
        """Warmup completion should be handed back to the main thread."""
        d = _make_delegate(main_module, monkeypatch)
        d._prepare_clients = MagicMock()

        d._prepare_clients_in_background()

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "clientWarmupSucceeded:", None, False
        )

    def test_background_warmup_dispatches_failure_to_main_thread(
        self, main_module, monkeypatch
    ):
        """Warmup failures should be surfaced back on the main thread."""
        d = _make_delegate(main_module, monkeypatch)
        d._prepare_clients = MagicMock(side_effect=RuntimeError("warm failed"))

        d._prepare_clients_in_background()

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "clientWarmupFailed:", None, False
        )
        assert isinstance(d._warm_error, RuntimeError)

    def test_warmup_success_hides_startup_indicator(
        self, main_module, monkeypatch
    ):
        """Successful warmup should remove the loading overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = None
        d.clientWarmupSucceeded_(None)

        d._overlay.hide.assert_called_once_with()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")

    def test_warmup_failure_updates_startup_indicator(
        self, main_module, monkeypatch
    ):
        """Failed warmup should keep a visible on-screen failure message."""
        d = _make_delegate(main_module, monkeypatch)
        d._warm_error = RuntimeError("warm failed")
        d._show_model_load_alert = MagicMock()

        d.clientWarmupFailed_(None)

        d._overlay.show.assert_called_once_with()
        d._overlay.set_text.assert_called_once_with(
            "Model load failed.\nChoose another model from the menu."
        )
        d._menubar.set_status_text.assert_called_with(
            "Model load failed — choose another model"
        )

    def test_warmup_failure_still_allows_model_selection_recovery(
        self, main_module, monkeypatch, tmp_path
    ):
        """A failed warmup should still leave model selection available for recovery."""
        d = _make_delegate(main_module, monkeypatch)
        d._warm_error = RuntimeError("warm failed")
        d._show_model_load_alert = MagicMock()
        monkeypatch.setenv(
            "SPOKE_MODEL_PREFERENCES_PATH", str(tmp_path / "model_preferences.json")
        )
        monkeypatch.setenv(
            "SPOKE_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
        )
        with patch.object(main_module.os, "execv") as mock_execv:
            d.clientWarmupFailed_(None)
            d._select_model("Qwen/Qwen3-ASR-0.6B")

        mock_execv.assert_called_once()

    def test_application_launch_kicks_off_command_model_refresh_async(
        self, main_module, monkeypatch
    ):
        """Assistant model discovery should start after launch, not inside init()."""
        d = _make_delegate(main_module, monkeypatch)
        d._quit = MagicMock()
        d._command_client = MagicMock()
        d._refresh_command_model_options_async = MagicMock()
        d._request_mic_permission = MagicMock()

        menubar = MagicMock()
        menubar.setup = MagicMock()
        glow = MagicMock()
        glow.setup = MagicMock()
        overlay = MagicMock()
        overlay.setup = MagicMock()
        command_overlay = MagicMock()
        command_overlay.setup = MagicMock()

        with patch.object(main_module.MenuBarIcon, "alloc") as mock_menubar_alloc, \
            patch.object(main_module.GlowOverlay, "alloc") as mock_glow_alloc, \
            patch.object(main_module.TranscriptionOverlay, "alloc") as mock_overlay_alloc:
            mock_menubar_alloc.return_value.initWithQuitCallback_selectModelCallback_.return_value = menubar
            mock_glow_alloc.return_value.initWithScreen_.return_value = glow
            mock_overlay_alloc.return_value.initWithScreen_.return_value = overlay
            import sys
            sys.modules["spoke.command_overlay"] = MagicMock()
            sys.modules["spoke.command_overlay"].CommandOverlay.alloc.return_value.initWithScreen_.return_value = command_overlay

            d.applicationDidFinishLaunching_(None)

        d._refresh_command_model_options_async.assert_called_once_with()

    def test_refresh_command_model_options_async_spawns_background_thread(
        self, main_module, monkeypatch
    ):
        """Refreshing assistant model options should happen on a background thread."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_model_id = "qwen3p5-35B-A3B"
        d._command_models_refresh_in_flight = False

        with patch.object(main_module.threading, "Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread

            d._refresh_command_model_options_async()

        MockThread.assert_called_once()
        mock_thread.start.assert_called_once_with()
        assert d._command_models_refresh_in_flight is True

    def test_command_models_discovered_updates_options_and_refreshes_menu(
        self, main_module, monkeypatch
    ):
        """Async completion should publish the discovered models and rebuild the menu."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_model_id = "qwen3p5-35B-A3B"
        d._command_models_refresh_in_flight = True

        d.commandModelsDiscovered_(
            {
                "options": [
                    ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
                    ("qwen3-14b", "qwen3-14b", True),
                ]
            }
        )

        assert d._command_models_refresh_in_flight is False
        assert d._command_model_options == [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", True),
        ]
        d._menubar.refresh_menu.assert_called_once_with()

    def test_command_models_discovered_does_not_reinsert_missing_current_model(
        self, main_module, monkeypatch
    ):
        """Async completion should not put a stale saved assistant model back into the menu."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_model_id = "qwen3p5-35B-A3B"
        d._command_models_refresh_in_flight = True

        d.commandModelsDiscovered_(
            {
                "options": [
                    (
                        "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                        "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                        False,
                    ),
                ]
            }
        )

        assert d._command_models_refresh_in_flight is False
        assert d._command_model_options == [
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            ),
        ]
        d._menubar.refresh_menu.assert_called_once_with()


class TestWarmupHoldGuard:
    """Test that pre-ready holds cannot push the UI back to a false ready state."""

    def test_hold_end_before_models_ready_keeps_loading_status(
        self, main_module, monkeypatch
    ):
        """Warmup should stay authoritative until models are actually ready."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = False
        d._warm_error = None

        d._on_hold_start()
        d._on_hold_end()

        d._capture.stop.assert_not_called()
        d._overlay.hide.assert_not_called()
        assert "Ready — hold spacebar" not in [
            call.args[0] for call in d._menubar.set_status_text.call_args_list
        ]
        assert d._menubar.set_status_text.call_args_list[-1].args[0] == "Loading models…"
        assert d._overlay.set_text.call_args_list[-1].args[0] == (
            "Loading models...\nFirst launch may download selected models."
        )

    def test_hold_end_after_warmup_failure_keeps_failure_status(
        self, main_module, monkeypatch
    ):
        """Model-load failure should remain visible even if the user taps hold/release."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = False
        d._warm_error = RuntimeError("warm failed")

        d._on_hold_start()
        d._on_hold_end()

        d._capture.stop.assert_not_called()
        d._overlay.hide.assert_not_called()
        assert d._menubar.set_status_text.call_args_list[-1].args[0] == (
            "Model load failed — choose another model"
        )
        assert d._overlay.set_text.call_args_list[-1].args[0] == (
            "Model load failed.\nChoose another model from the menu."
        )

    def test_hold_end_after_warmup_success_still_ignores_rejected_hold(
        self, main_module, monkeypatch
    ):
        """A hold that began during warmup stays invalid even if warmup finishes before release."""
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = None

        d._models_ready = False
        d._warm_error = None
        d._on_hold_start()
        d.clientWarmupSucceeded_(None)
        d._on_hold_end()

        d._capture.stop.assert_not_called()
        d._menubar.set_recording.assert_not_called()
        assert d._menubar.set_status_text.call_args_list[-1].args[0] == (
            "Ready — hold spacebar"
        )

    def test_hold_end_after_warmup_failure_transition_still_ignores_rejected_hold(
        self, main_module, monkeypatch
    ):
        """A rejected hold should not mutate UI state after warmup flips into failure."""
        d = _make_delegate(main_module, monkeypatch)
        d._show_model_load_alert = MagicMock()

        d._models_ready = False
        d._warm_error = None
        d._on_hold_start()
        d._warm_error = RuntimeError("warm failed")
        d.clientWarmupFailed_(None)
        d._on_hold_end()

        d._capture.stop.assert_not_called()
        d._overlay.hide.assert_not_called()
        assert d._menubar.set_status_text.call_args_list[-1].args[0] == (
            "Model load failed — choose another model"
        )


class TestEnvValidation:
    """Test environment variable validation in SpokeAppDelegate.init."""

    def test_missing_whisper_url_uses_local(self, main_module, monkeypatch):
        """Missing SPOKE_WHISPER_URL should fall back to local transcription."""
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_WHISPER_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_PREVIEW_MODEL", raising=False)
        monkeypatch.delenv("SPOKE_TRANSCRIPTION_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_model_preferences",
            lambda self: {},
            raising=False,
        )
        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        result = d.init()
        assert result is not None
        assert isinstance(d._client, main_module.LocalTranscriptionClient)

    def test_invalid_hold_ms_exits(self, main_module, monkeypatch):
        """Non-integer SPOKE_HOLD_MS should sys.exit(1)."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("SPOKE_HOLD_MS", "not-a-number")
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            d.init()
        assert exc_info.value.code == 1

    def test_valid_config_succeeds(self, main_module, monkeypatch):
        """Valid env vars should create a delegate without error."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
        monkeypatch.setenv("SPOKE_HOLD_MS", "300")
        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        result = d.init()
        assert result is not None

    def test_default_hold_ms_is_200(self, main_module, monkeypatch):
        """Default hold timing should match the faster interaction cadence."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
        monkeypatch.delenv("SPOKE_HOLD_MS", raising=False)
        detector_instance = MagicMock(name="detector_instance")

        with patch.object(main_module.SpacebarHoldDetector, "alloc") as mock_alloc:
            mock_alloc.return_value.initWithHoldStart_holdEnd_holdMs_.return_value = (
                detector_instance
            )
            d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
            result = d.init()

        assert result is not None
        assert d._detector is detector_instance
        assert (
            mock_alloc.return_value.initWithHoldStart_holdEnd_holdMs_.call_args.args[2]
            == 200
        )


class TestRecordingCap:
    """Test the recording cap countdown and force_end trigger."""

    def test_max_record_secs_uses_16gb_cutoff(self, main_module):
        assert main_module._max_record_secs_for_ram(15.0) == 20.0
        assert main_module._max_record_secs_for_ram(16.0) is None
        assert main_module._max_record_secs_for_ram(31.0) is None
        assert main_module._max_record_secs_for_ram(32.0) is None

    def test_cap_fires_after_max_seconds(self, main_module, monkeypatch):
        """When elapsed >= _MAX_RECORD_SECS, cap should fire and call force_end."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._cap_fired = False
        d._record_start_time = time.monotonic() - 21.0  # 21s elapsed, cap is 20s
        d._glow._cap_factor = 1.0
        d._glow._smoothed_amplitude = 0.0

        # Patch module-level _MAX_RECORD_SECS
        monkeypatch.setattr(main_module, "_MAX_RECORD_SECS", 20.0)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        assert d._cap_fired is True
        d._detector.force_end.assert_called_once()

    def _setup_glow_mock(self, d):
        """Configure glow mock so amplitudeUpdate_ doesn't crash past the cap check."""
        d._glow._cap_factor = 1.0
        d._glow._smoothed_amplitude = 0.0
        glow_layer = MagicMock()
        glow_layer.opacity.return_value = 0.1
        d._glow._glow_layer = glow_layer

    def test_cap_ramps_glow_in_last_3_seconds(self, main_module, monkeypatch):
        """In the last 3s before cap, _cap_factor should ramp from 1.0 toward 0.0."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._cap_fired = False
        # 18.5s elapsed — 1.5s into the 3s warning window (cap at 20s)
        d._record_start_time = time.monotonic() - 18.5
        self._setup_glow_mock(d)

        monkeypatch.setattr(main_module, "_MAX_RECORD_SECS", 20.0)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        # 1.5s into 3s warning = 50% progress, cap_factor should be ~0.5
        assert d._glow._cap_factor < 0.6
        assert d._glow._cap_factor > 0.4
        assert d._cap_fired is False

    def test_cap_noop_when_already_fired(self, main_module, monkeypatch):
        """Once cap has fired, subsequent amplitude updates should not re-fire."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._cap_fired = True  # already fired
        d._record_start_time = time.monotonic() - 25.0
        self._setup_glow_mock(d)

        monkeypatch.setattr(main_module, "_MAX_RECORD_SECS", 20.0)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        d._detector.force_end.assert_not_called()

    def test_cap_noop_in_sidecar_mode(self, main_module, monkeypatch):
        """Recording cap should not fire in sidecar mode (inference is remote)."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = False  # sidecar
        d._cap_fired = False
        d._record_start_time = time.monotonic() - 25.0
        self._setup_glow_mock(d)

        monkeypatch.setattr(main_module, "_MAX_RECORD_SECS", 20.0)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        d._detector.force_end.assert_not_called()
        assert d._cap_fired is False

    def test_cap_noop_when_max_record_secs_none(self, main_module, monkeypatch):
        """At 32GB+, _MAX_RECORD_SECS is None — cap is disabled."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._cap_fired = False
        d._record_start_time = time.monotonic() - 25.0
        self._setup_glow_mock(d)

        monkeypatch.setattr(main_module, "_MAX_RECORD_SECS", None)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        d._detector.force_end.assert_not_called()
        assert d._cap_fired is False

    def test_amplitude_update_syncs_command_overlay_brightness(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._glow._brightness = 0.61
        self._setup_glow_mock(d)

        d.amplitudeUpdate_(MagicMock(return_value=0.01))

        d._command_overlay.set_brightness.assert_called_with(0.61, immediate=False)


class TestCommandTranscribeWorker:
    """Test _command_transcribe_worker branching and dispatch."""

    def _make_command_delegate(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_overlay = MagicMock()
        d._preview_thread = None
        d._preview_done = MagicMock()
        d._preview_done.wait = MagicMock()
        d._preview_done.set = MagicMock()
        d._local_inference_lock = MagicMock()
        d._local_inference_lock.__enter__ = MagicMock(return_value=None)
        d._local_inference_lock.__exit__ = MagicMock(return_value=False)
        d._transcribe_start = time.monotonic()
        d._transcription_token = 1
        d._command_first_token = False
        d._scene_cache = None
        d._tool_schemas = None
        return d

    def test_successful_transcribe_and_stream(self, main_module, monkeypatch):
        """Happy path: transcribe → stream tokens → complete."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.return_value = "open the file"
        d._client.supports_streaming = False
        d._command_client.stream_command_events.return_value = iter([
            MagicMock(kind="assistant_delta", text="Hello"),
            MagicMock(kind="assistant_delta", text=" world"),
            MagicMock(kind="assistant_final", text="Hello world"),
        ])

        d._command_transcribe_worker(b"wav-data", 1)

        # Should have dispatched: utteranceReady, 2 tokens, complete
        calls = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        selectors = [c[0][0] for c in calls]
        assert "commandUtteranceReady:" in selectors
        assert selectors.count("commandToken:") == 2
        assert "commandComplete:" in selectors
        complete_call = next(c for c in calls if c[0][0] == "commandComplete:")
        assert complete_call[0][1]["response"] == "Hello world"

    def test_empty_utterance_recalls_last_response(self, main_module, monkeypatch):
        """Empty transcription with shift = recall last response."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.return_value = ""
        d._client.supports_streaming = False

        d._command_transcribe_worker(b"wav-data", 1)

        calls = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        selectors = [c[0][0] for c in calls]
        assert "_recallLastResponse:" in selectors
        # Should NOT have tried to stream
        d._command_client.stream_command_events.assert_not_called()

    def test_transcription_failure_dispatches_error(self, main_module, monkeypatch):
        """Transcription exception → commandFailed."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.side_effect = RuntimeError("model crashed")
        d._client.supports_streaming = False

        d._command_transcribe_worker(b"wav-data", 1)

        calls = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        selectors = [c[0][0] for c in calls]
        assert "commandFailed:" in selectors
        # Should NOT have tried to stream
        d._command_client.stream_command_events.assert_not_called()

    def test_stream_failure_dispatches_error(self, main_module, monkeypatch):
        """Streaming exception → commandFailed after utterance dispatched."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.return_value = "do something"
        d._client.supports_streaming = False
        d._command_client.stream_command_events.side_effect = ConnectionError("OMLX down")

        d._command_transcribe_worker(b"wav-data", 1)

        calls = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        selectors = [c[0][0] for c in calls]
        assert "commandUtteranceReady:" in selectors
        assert "commandFailed:" in selectors
        assert "commandComplete:" not in selectors

    def test_stale_token_breaks_stream(self, main_module, monkeypatch):
        """If transcription_token changes mid-stream, stop dispatching tokens."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.return_value = "do something"
        d._client.supports_streaming = False

        def token_gen(utterance, **kwargs):
            yield MagicMock(kind="assistant_delta", text="first")
            d._transcription_token = 99  # simulate new recording invalidating
            yield MagicMock(kind="assistant_delta", text="should not appear")
            yield MagicMock(kind="assistant_final", text="also should not appear")

        d._command_client.stream_command_events.side_effect = token_gen

        d._command_transcribe_worker(b"wav-data", 1)

        calls = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
        token_calls = [c for c in calls if c[0][0] == "commandToken:"]
        # Only the first token should have been dispatched
        assert len(token_calls) == 1
        assert token_calls[0][0][1]["text"] == "first"

    def test_streaming_preview_finalize_path(self, main_module, monkeypatch):
        """When client is preview client with active stream, use finish_stream."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.supports_streaming = True
        d._client.has_active_stream = True
        d._preview_client = d._client  # same object
        d._client.finish_stream.return_value = "streamed utterance"
        d._command_client.stream_command_events.return_value = iter([
            MagicMock(kind="assistant_delta", text="ok"),
            MagicMock(kind="assistant_final", text="ok"),
        ])

        d._command_transcribe_worker(b"wav-data", 1)

        d._client.finish_stream.assert_called_once()
        d._client.transcribe.assert_not_called()

    def test_non_streaming_uses_transcribe(self, main_module, monkeypatch):
        """When client doesn't support streaming, use transcribe."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.supports_streaming = False
        d._client.transcribe.return_value = "hello"
        d._command_client.stream_command_events.return_value = iter([])

        d._command_transcribe_worker(b"wav-data", 1)

        d._client.transcribe.assert_called_once_with(b"wav-data")

    def test_waits_for_preview_thread(self, main_module, monkeypatch):
        """Should wait for preview thread to finish before transcribing."""
        d = self._make_command_delegate(main_module, monkeypatch)
        mock_thread = MagicMock()
        d._preview_thread = mock_thread
        d._client.transcribe.return_value = ""
        d._client.supports_streaming = False

        d._command_transcribe_worker(b"wav-data", 1)

        d._preview_done.wait.assert_called_once_with(timeout=2.0)
        mock_thread.join.assert_called_once_with(timeout=2.0)


class TestCommandCallbacks:
    """Test main-thread command pathway callbacks."""

    def test_command_utterance_ready_hides_input_shows_command(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1

        d.commandUtteranceReady_({"token": 1, "utterance": "open file"})

        d._overlay.hide.assert_called()
        d._command_overlay.show.assert_called()
        d._command_overlay.set_utterance.assert_called_with("open file")

    def test_command_utterance_ready_primes_command_overlay_brightness(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._glow._brightness = 0.73
        d._transcription_token = 1

        d.commandUtteranceReady_({"token": 1, "utterance": "open file"})

        assert d._command_overlay.method_calls[:3] == [
            call.set_brightness(0.73, immediate=True),
            call.show(),
            call.set_utterance("open file"),
        ]

    def test_command_utterance_ready_stale_token_ignored(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 2

        d.commandUtteranceReady_({"token": 1, "utterance": "stale"})

        d._command_overlay.show.assert_not_called()

    def test_command_token_first_inverts_thinking(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._command_first_token = True

        d.commandToken_({"token": 1, "text": "first"})

        d._command_overlay.invert_thinking_timer.assert_called_once()
        d._command_overlay.append_token.assert_called_with("first")
        assert d._command_first_token is False

    def test_command_token_subsequent_no_invert(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._command_first_token = False

        d.commandToken_({"token": 1, "text": "more"})

        d._command_overlay.invert_thinking_timer.assert_not_called()
        d._command_overlay.append_token.assert_called_with("more")

    def test_command_token_invert_failure_still_appends_token(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay.invert_thinking_timer.side_effect = RuntimeError("flip")
        d._transcription_token = 1
        d._command_first_token = True

        with caplog.at_level(logging.ERROR):
            d.commandToken_({"token": 1, "text": "first"})

        d._command_overlay.append_token.assert_called_with("first")
        assert d._command_first_token is False
        assert "Command overlay failed to invert thinking timer" in caplog.text

    def test_command_complete_finishes_overlay(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True

        d.commandComplete_({"token": 1})

        assert d._transcribing is False
        d._command_overlay.finish.assert_called_once()
        d._glow.hide.assert_called()

    def test_command_complete_replaces_overlay_with_final_response(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True

        d.commandComplete_({"token": 1, "response": "Done."})

        d._command_overlay.set_response_text.assert_called_once_with("Done.")
        d._command_overlay.finish.assert_called_once()

    def test_command_complete_finish_failure_still_starts_autoplay(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay.finish.side_effect = RuntimeError("finish")
        d._transcription_token = 1
        d._transcribing = True
        d._tts_client = MagicMock()

        with caplog.at_level(logging.ERROR):
            d.commandComplete_({"token": 1, "response": "Hello there"})

        assert d._transcribing is False
        d._tts_client.speak_async.assert_called_once()
        d._command_overlay.tts_start.assert_called_once()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")
        assert "Command overlay finish failed" in caplog.text

    def test_command_complete_autoplay_failure_is_suppressed(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True
        d._tts_client = MagicMock()
        d._tts_client.speak_async.side_effect = RuntimeError("tts launch")

        with caplog.at_level(logging.ERROR):
            d.commandComplete_({"token": 1, "response": "Hello there"})

        assert d._transcribing is False
        d._command_overlay.tts_stop.assert_called_once()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")
        assert "Command autoplay failed to start" in caplog.text

    def test_tool_executor_marks_tool_tts_usage(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._tts_client = MagicMock()

        executor = d._make_tool_executor()
        result = executor("read_aloud", {"source_ref": "literal:hello world"})

        assert result == "Speaking: hello world"
        assert d._command_tool_used_tts is True
        d._tts_client.speak_async.assert_called_once_with("hello world")

    def test_tool_executor_routes_add_to_tray_through_delegate_bridge(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._add_assistant_content_to_tray = MagicMock(
            return_value={"status": "added", "tray_visible": False, "stack_size": 1}
        )

        executor = d._make_tool_executor()
        result = executor("add_to_tray", {"text": "save this"})

        d._add_assistant_content_to_tray.assert_called_once_with("save this")
        parsed = json.loads(result)
        assert parsed["status"] == "added"

    def test_tool_executor_does_not_mark_tool_tts_usage_on_launch_failure(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._tts_client = MagicMock()
        d._tts_client.speak_async.side_effect = RuntimeError("device unavailable")

        executor = d._make_tool_executor()
        result = executor("read_aloud", {"source_ref": "literal:hello world"})

        assert result == "Error speaking text: TTS playback failed"
        assert d._command_tool_used_tts is False
        d._tts_client.speak_async.assert_called_once_with("hello world")

    def test_command_failed_shows_error_in_overlay(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay._visible = False
        d._transcription_token = 1
        d._transcribing = True

        d.commandFailed_({"token": 1, "error": "OMLX down"})

        assert d._transcribing is False
        d._command_overlay.show.assert_called()
        d._command_overlay.append_token.assert_called()
        d._command_overlay.finish.assert_called()

    def test_tts_amplitude_update_failure_is_suppressed(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay.update_tts_amplitude.side_effect = RuntimeError("amp")

        with caplog.at_level(logging.ERROR):
            d.ttsAmplitudeUpdate_(0.5)

        assert "Command overlay amplitude update failed" in caplog.text

    def test_tts_finished_failure_is_suppressed(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay.tts_stop.side_effect = RuntimeError("stop")

        with caplog.at_level(logging.ERROR):
            d.ttsFinished_(None)

        assert "Command overlay TTS stop failed" in caplog.text

    def test_command_failed_overlay_failure_is_suppressed(
        self, main_module, monkeypatch, caplog
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay._visible = False
        d._command_overlay.append_token.side_effect = RuntimeError("append")
        d._transcription_token = 1
        d._transcribing = True

        with caplog.at_level(logging.ERROR):
            d.commandFailed_({"token": 1, "error": "OMLX down"})

        assert d._transcribing is False
        d._command_overlay.show.assert_called()
        assert "Command overlay append failed during error presentation" in caplog.text

    def test_recall_last_response_shows_history(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = [("what time is it", "It's 3pm")]
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True

        d._recallLastResponse_({"token": 1})

        assert d._transcribing is False
        d._command_overlay.show.assert_called()
        d._command_overlay.set_utterance.assert_called_with("what time is it")
        d._command_overlay.finish.assert_called()

    def test_recall_no_history(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True

        d._recallLastResponse_({"token": 1})

        d._command_overlay.show.assert_not_called()


class TestResultInjection:
    """Test timing of the post-injection overlay cleanup."""

    def test_inject_result_text_orders_out_overlay_before_focus_check(
        self, main_module, monkeypatch
    ):
        """Overlay should be ordered out (not faded) before the focus check
        runs so the AX system sees the underlying text field, not the overlay."""
        d = _make_delegate(main_module, monkeypatch)

        with patch.object(main_module, "inject_text"):
            d._inject_result_text("hello", "Ready")

        d._overlay.order_out.assert_called()


class TestHoldStartDuringTranscription:
    """Test interrupt-and-restart when hold starts during active transcription."""

    def test_hold_during_transcription_increments_token(self, main_module, monkeypatch):
        """Starting a new hold while transcribing should invalidate the old generation."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 5
        d._models_ready = True

        d._on_hold_start()

        assert d._transcription_token == 6
        assert d._transcribing is False
        # Should have fallen through to start recording
        d._capture.start.assert_called_once()

    def test_hold_during_transcription_starts_new_recording(self, main_module, monkeypatch):
        """After cancelling the old transcription, recording should proceed normally."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 0
        d._models_ready = True

        d._on_hold_start()

        d._menubar.set_recording.assert_called_with(True)
        d._glow.show.assert_called_once()


class TestShortShiftHold:
    """Test the instant recall/dismiss path for short shift-holds."""

    def test_short_shift_hold_discards_audio(self, main_module, monkeypatch):
        """Shift-release under 800ms should force empty-audio path."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"real-audio-data"
        d._record_start_time = time.monotonic() - 0.3  # 300ms ago
        d._command_client = MagicMock()
        d._command_client.history = []
        d._command_overlay = MagicMock(_visible=False)

        with patch.object(main_module.threading, "Thread") as MockThread:
            d._on_hold_end(shift_held=True)

        # Should not spawn transcription thread — audio was discarded
        MockThread.assert_not_called()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")

    def test_long_shift_hold_keeps_audio(self, main_module, monkeypatch):
        """Shift-release over 800ms should proceed to transcription."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"real-audio-data"
        d._record_start_time = time.monotonic() - 1.0  # 1s ago

        with patch.object(main_module.threading, "Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            d._on_hold_end(shift_held=True)

        MockThread.assert_called_once()
        mock_thread.start.assert_called_once()

    def test_short_shift_hold_recalls_tray(self, main_module, monkeypatch):
        """Short shift-hold with tray entries should recall into tray."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.1  # 100ms
        d._tray_stack = ["previous text"]
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        d._on_hold_end(shift_held=True)

        # Should enter tray, not command overlay
        assert d._tray_active is True
        assert d._tray_index == 0
        d._overlay.show_tray.assert_called()

    def test_short_shift_enter_hold_recalls_command_overlay(self, main_module, monkeypatch):
        """When both Shift and Enter were used, Enter should win for assistant recall."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.1  # 100ms
        d._tray_stack = ["previous text"]
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        d._on_hold_end(shift_held=True, enter_held=True)

        assert d._tray_active is False
        d._command_overlay.show.assert_called_once()
        d._command_overlay.set_utterance.assert_called_once_with("hello")
        d._command_overlay.append_token.assert_called()
        d._command_overlay.finish.assert_called_once()
        d._overlay.show_tray.assert_not_called()

    def test_short_shift_enter_hold_dismisses_visible_command_overlay(
        self, main_module, monkeypatch
    ):
        """The same combined gesture should dismiss the assistant overlay with Shift still held."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.1  # 100ms
        d._tray_stack = ["previous text"]
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)

        d._on_hold_end(shift_held=True, enter_held=True)

        assert d._tray_active is False
        d._command_overlay.cancel_dismiss.assert_called_once()
        d._overlay.show_tray.assert_not_called()


class TestCoerceSettings:
    """Test _coerce_decode_timeout_setting and _coerce_eager_eval_setting parsers."""

    def test_decode_timeout_none_returns_default(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting(None) == 30.0

    def test_decode_timeout_positive_int(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting(15) == 15.0

    def test_decode_timeout_zero_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting(0) is None

    def test_decode_timeout_negative_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting(-1) is None

    def test_decode_timeout_positive_float(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting(20.5) == 20.5

    def test_decode_timeout_string_off_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("off") is None

    def test_decode_timeout_string_none_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("none") is None

    def test_decode_timeout_string_false_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("false") is None

    def test_decode_timeout_string_default_returns_default(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("default") == 30.0

    def test_decode_timeout_empty_string_returns_default(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("") == 30.0

    def test_decode_timeout_numeric_string_parsed(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("15.5") == 15.5

    def test_decode_timeout_invalid_string_returns_default(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("xyz") == 30.0

    def test_decode_timeout_string_zero_disables(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("0") is None

    def test_decode_timeout_whitespace_stripped(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_decode_timeout_setting("  OFF  ") is None

    def test_eager_eval_none_returns_default(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_eager_eval_setting(None) is False

    def test_eager_eval_bool_passthrough(self, main_module):
        assert main_module.SpokeAppDelegate._coerce_eager_eval_setting(True) is True
        assert main_module.SpokeAppDelegate._coerce_eager_eval_setting(False) is False

    def test_eager_eval_string_true_variants(self, main_module):
        for val in ("1", "true", "yes", "on", "  TRUE  ", "On"):
            assert main_module.SpokeAppDelegate._coerce_eager_eval_setting(val) is True

    def test_eager_eval_string_false_variants(self, main_module):
        for val in ("0", "false", "no", "off", "", "anything"):
            assert main_module.SpokeAppDelegate._coerce_eager_eval_setting(val) is False


class TestBuildClientRouting:
    """Test _build_client client-type routing."""

    def test_sidecar_url_returns_transcription_client(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        client = d._build_client("http://localhost:8000", "any-model")
        assert isinstance(client, main_module.TranscriptionClient)

    def test_qwen_prefix_returns_local_qwen_client(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        client = d._build_client("", "Qwen/Qwen3-ASR-0.6B")
        assert isinstance(client, main_module.LocalQwenClient)

    def test_default_returns_local_transcription_client(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        client = d._build_client("", "mlx-community/whisper-base.en-mlx-8bit")
        assert isinstance(client, main_module.LocalTranscriptionClient)

    def test_sidecar_takes_precedence_over_qwen_prefix(self, main_module, monkeypatch):
        """When URL is set, sidecar wins even if model starts with Qwen/."""
        d = _make_delegate(main_module, monkeypatch)
        client = d._build_client("http://localhost:8000", "Qwen/Qwen3-ASR-0.6B")
        assert isinstance(client, main_module.TranscriptionClient)


class TestGetClipboardPreviewText:
    """Test _get_clipboard_preview_text extraction."""

    def test_none_returns_empty(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        assert d._get_clipboard_preview_text(None) == "(empty)"

    def test_empty_list_returns_empty(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        assert d._get_clipboard_preview_text([]) == "(empty)"

    def test_utf8_text_decoded(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        saved = [("public.utf8-plain-text", b"hello world")]
        assert d._get_clipboard_preview_text(saved) == "hello world"

    def test_string_ptype_decoded(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        saved = [("NSStringPboardType", b"test")]
        assert d._get_clipboard_preview_text(saved) == "test"

    def test_non_text_content(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        saved = [("public.png", b"\x89PNG")]
        assert d._get_clipboard_preview_text(saved) == "(non-text)"

    def test_invalid_utf8_falls_through(self, main_module, monkeypatch):
        """Invalid UTF-8 in first text item should try next, then return non-text."""
        d = _make_delegate(main_module, monkeypatch)
        saved = [("public.utf8-plain-text", b"\xff\xfe")]
        assert d._get_clipboard_preview_text(saved) == "(non-text)"

    def test_first_invalid_second_valid(self, main_module, monkeypatch):
        """Should skip invalid UTF-8 and return the next valid text item."""
        d = _make_delegate(main_module, monkeypatch)
        saved = [
            ("public.utf8-plain-text", b"\xff\xfe"),
            ("NSStringPboardType", b"fallback"),
        ]
        assert d._get_clipboard_preview_text(saved) == "fallback"
