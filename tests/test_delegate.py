"""Tests for SpokeAppDelegate orchestration and error paths.

Tests the wiring between layers: hold callbacks, transcription lifecycle,
generation-based stale result rejection, and env var validation.
"""

import logging
import os
import json
import io
import time
import threading
import urllib.error
from unittest.mock import MagicMock, call, patch


def _make_delegate(main_module, monkeypatch):
    """Create a SpokeAppDelegate with mocked sub-components."""
    monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")

    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._capture = MagicMock()
    delegate._client = MagicMock(supports_streaming=False)
    delegate._detector = MagicMock()
    delegate._detector.command_overlay_active = False
    delegate._menubar = MagicMock()
    delegate._glow = MagicMock()
    delegate._overlay = MagicMock()
    delegate._transcribing = False
    delegate._transcription_token = 0
    delegate._preview_active = False
    delegate._preview_thread = None
    delegate._preview_client = MagicMock()
    delegate._preview_model_id = "preview-model"
    delegate._transcription_model_id = "transcription-model"
    delegate._local_mode = False
    delegate._record_start_time = 0.0
    delegate._cap_fired = False
    delegate._transcribe_start = time.monotonic()
    delegate._last_preview_text = ""
    delegate._mic_ready = True
    delegate._command_client = None
    delegate._command_backend = "local"
    delegate._command_url = "http://localhost:8001"
    delegate._command_sidecar_url = None
    delegate._command_model_id = None
    delegate._command_model_options = []
    delegate._parallel_insert_token = 0
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
    delegate._whisper_backend = "local"
    delegate._preview_backend = "local"
    delegate._segment_accumulator = main_module.SegmentAccumulator()
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

    def test_hold_start_shows_preview_before_starting_capture(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        call_order: list[str] = []
        d._glow.show.side_effect = lambda: call_order.append("glow")
        d._overlay.show.side_effect = lambda: call_order.append("overlay")
        d._capture.start.side_effect = lambda **kwargs: call_order.append("capture")

        d._on_hold_start()

        assert call_order[:3] == ["glow", "overlay", "capture"]

    def test_hold_start_capture_failure_restores_idle_ui(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._capture.start.side_effect = RuntimeError("audio dead")

        d._on_hold_start()

        d._glow.hide.assert_called_once_with()
        d._overlay.flash_notice.assert_called_once()
        d._menubar.set_recording.assert_any_call(True)
        d._menubar.set_recording.assert_any_call(False)
        d._menubar.set_status_text.assert_called_with(
            "Audio unavailable — memory pressure"
        )
        assert d._preview_active is False

    def test_hold_start_suspends_wakeword_listener_before_capture_start(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._handsfree = MagicMock()
        d._handsfree.state = main_module.HandsFreeState.LISTENING
        d._handsfree.is_dictating = False
        call_order: list[str] = []
        d._handsfree.disable.side_effect = lambda: call_order.append("disable")
        d._capture.start.side_effect = lambda **kwargs: call_order.append("capture")

        d._on_hold_start()

        assert call_order[:2] == ["disable", "capture"]
        assert d._handsfree_resume_state_for_hold == main_module.HandsFreeState.LISTENING

    def test_hold_end_with_empty_audio_restores_wakeword_listener(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._handsfree = MagicMock()
        d._handsfree_resume_state_for_hold = main_module.HandsFreeState.LISTENING
        d._capture.stop.return_value = b""

        d._on_hold_end()

        d._handsfree.enable.assert_called_once_with()
        assert d._handsfree_resume_state_for_hold is None

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

    def test_result_inject_delayed_restores_wakeword_listener_after_hold(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._handsfree = MagicMock()
        d._handsfree_resume_state_for_hold = main_module.HandsFreeState.LISTENING
        d._result_pending_inject = ("hello", "Pasted!")

        def fake_inject_text(text, on_restored=None):
            assert text == "hello"
            if on_restored is not None:
                on_restored()

        with patch.object(main_module, "inject_text", side_effect=fake_inject_text):
            d.resultInjectDelayed_(None)

        d._handsfree.enable.assert_called_once_with()
        assert d._handsfree_resume_state_for_hold is None


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
        assert d._transcription_token == 1
        assert d._parallel_insert_token == 1

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
        """Result with matching token should be injected after grace window."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 5
        d._transcribing = True

        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionComplete_({"token": 5, "text": "hello world"})
            # Fire the grace window timer, then the deferred inject timer
            d.graceTimerFired_(None)
            d.resultInjectDelayed_(None)

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

    def test_current_failure_surfaces_specific_error_when_present(
        self, main_module, monkeypatch
    ):
        """A concrete finalization error should reach the menubar instead of generic text."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 7
        d._transcribing = True

        d.transcriptionFailed_(
            {"token": 7, "error": "Local transcription timed out after bounded retries"}
        )

        assert d._transcribing is False
        d._menubar.set_status_text.assert_called_with(
            "Local transcription timed out after bounded retries"
        )

    def test_empty_text_not_injected(self, main_module, monkeypatch):
        """Empty transcription result should not call inject_text."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcription_token = 1

        with patch.object(main_module, "inject_text") as mock_inject:
            d.transcriptionComplete_({"token": 1, "text": ""})

        mock_inject.assert_not_called()

    def test_parallel_insert_result_is_accepted_without_touching_active_turn(
        self, main_module, monkeypatch
    ):
        """Parallel plain-space insertions should inject text without clearing the assistant turn."""
        d = _make_delegate(main_module, monkeypatch)
        d._parallel_insert_token = 2
        d._transcription_token = 5
        d._transcribing = True

        with patch.object(main_module, "inject_text") as mock_inject:
            d.parallelTranscriptionComplete_({"token": 2, "text": "hello world"})
            d.graceTimerFired_(None)
            d.resultInjectDelayed_(None)

        mock_inject.assert_called_once()
        assert mock_inject.call_args[0][0] == "hello world"
        assert d._transcribing is True
        assert d._transcription_token == 5

    def test_stale_parallel_insert_result_is_discarded(self, main_module, monkeypatch):
        """Parallel insertion results should respect their own token lane."""
        d = _make_delegate(main_module, monkeypatch)
        d._parallel_insert_token = 4
        d._transcription_token = 7
        d._transcribing = True

        with patch.object(main_module, "inject_text") as mock_inject:
            d.parallelTranscriptionComplete_({"token": 3, "text": "stale"})

        mock_inject.assert_not_called()
        assert d._transcribing is True
        assert d._transcription_token == 7


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
            # Fire the deferred inject timer callback
            d.resultInjectDelayed_(None)

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

    def test_parallel_insert_release_cutover_cancels_shared_stt_stream_only(
        self, main_module, monkeypatch
    ):
        """Parallel insert release cutover should finalize STT only and leave the assistant turn untouched."""
        d = _make_delegate(main_module, monkeypatch)
        d._preview_cancelled_on_release = True
        d._preview_thread = MagicMock()
        d._preview_done = MagicMock()
        d._transcribing = True
        d._transcription_token = 17
        d._command_client = MagicMock()
        streaming_client = MagicMock(supports_streaming=True, has_active_stream=True)
        streaming_client.transcribe.return_value = "parallel insert text"
        d._client = streaming_client
        d._preview_client = streaming_client

        d._parallel_insert_worker(b"wav", token=3)

        d._preview_done.wait.assert_not_called()
        d._preview_thread.join.assert_not_called()
        streaming_client.cancel_stream.assert_called_once_with()
        streaming_client.finish_stream.assert_not_called()
        streaming_client.transcribe.assert_called_once_with(b"wav")
        d._command_client.assert_not_called()
        call_args = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args
        assert call_args[0][0] == "parallelTranscriptionComplete:"
        assert call_args[0][1]["token"] == 3
        assert call_args[0][1]["text"] == "parallel insert text"
        assert d._transcribing is True
        assert d._transcription_token == 17


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
        monkeypatch.setattr(main_module, "_RAM_GB", 32.0)
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {},
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
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
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
            lambda self: {},
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
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
            lambda self: {},
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
            ("qwen3-14b", "qwen3-14b", False),
        ]

        model_state = d._handle_model_menu_action(None)

        assert model_state["assistant"] == {
            "title": "Assistant Model",
            "selected": "qwen3p5-35B-A3B",
            "models": [
                ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
                ("qwen3-14b", "qwen3-14b", True),
            ],
        }

    def test_handle_model_menu_none_exposes_assistant_backend_controls(
        self, main_module, monkeypatch
    ):
        """Command mode should surface persisted local-vs-sidecar backend controls."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_backend = "sidecar"
        d._command_sidecar_url = "http://other-box:8001"

        model_state = d._handle_model_menu_action(None)

        assert model_state["assistant_backend"] == {
            "title": "Assistant Backend",
            "items": [
                ("local", "Local OMLX", False, True),
                ("sidecar", "Sidecar OMLX", True, True),
                ("cloud_google", "Google Cloud", False, True),
                ("cloud_openrouter", "OpenRouter", False, True),
                ("configure", "Set Sidecar URL…", False, True),
                ("configure_cloud_google", "Set Google Cloud Endpoint…", False, True),
                ("configure_cloud_openrouter", "Set OpenRouter Endpoint…", False, True),
            ],
        }

    def test_handle_model_menu_none_surfaces_tts_backend_and_endpoint(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = MagicMock()
        d._tts_client._model_id = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._load_preference = lambda key: None
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("SPOKE_TTS_VOICE", "casual_female")

        model_state = d._handle_model_menu_action(None)

        assert model_state["tts_backend"] == {
            "title": "TTS Backend: Local",
            "items": [
                ("local", "Local runtime", True),
                ("sidecar", "Sidecar (not configured)", False, False),
                ("cloud", "Cloud (no API key)", False, False),
                ("configure_tts", "Set TTS Sidecar URL\u2026", False, True),
            ],
        }
        assert model_state["tts_endpoint"] == {
            "title": "TTS Endpoint: local runtime",
            "note": "Routing source: local runtime",
        }
        assert any(
            model_id == "k2-fsa/OmniVoice"
            for model_id, _label, _enabled in model_state["tts"]["models"]
        )

    def test_handle_model_menu_none_keeps_tts_controls_for_saved_model_without_live_client(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._tts_client = None
        saved = {"tts_model": "k2-fsa/OmniVoice"}
        d._load_preference = lambda key: saved.get(key)
        monkeypatch.delenv("SPOKE_TTS_VOICE", raising=False)
        monkeypatch.delenv("SPOKE_TTS_MODEL", raising=False)

        model_state = d._handle_model_menu_action(None)

        assert model_state["tts_backend"]["title"] == "TTS Backend: Local"
        assert model_state["tts"]["selected"] == "k2-fsa/OmniVoice"
        assert model_state["tts_voice"] == {
            "type": "choice",
            "title": "TTS Voice: Auto voice",
            "selected": "",
            "models": [
                ("", "Auto voice", True),
                ("female, child", "Female, child", True),
                ("male, high pitch, indian accent", "Male, high pitch, Indian", True),
                ("female, elderly, british accent", "Female, elderly, British", True),
                ("female, young adult, whisper", "Female, young adult, whisper", True),
                ("male, middle-aged, very low pitch", "Male, middle-aged, very low pitch", True),
                ("female, low pitch, british accent", "Female, low pitch, British", True),
                ("male, british accent", "Male, British", True),
                ("female, whisper, british accent", "Female whisper, British", True),
                ("female, high pitch, american accent", "Female, high pitch, American", True),
                ("male, low pitch, american accent", "Male, low pitch, American", True),
                ("configure_voice", "Set Custom Voice…", True),
            ],
        }

    def test_handle_model_menu_none_labels_saved_omnivoice_prompt_value_as_prompt(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._tts_client = None
        saved = {
            "tts_model": "k2-fsa/OmniVoice",
            "tts_voice": "female, british accent",
        }
        d._load_preference = lambda key: saved.get(key)
        monkeypatch.delenv("SPOKE_TTS_VOICE", raising=False)
        monkeypatch.delenv("SPOKE_TTS_MODEL", raising=False)

        model_state = d._handle_model_menu_action(None)

        assert model_state["tts_voice"] == {
            "type": "choice",
            "title": "TTS Voice: female, british accent",
            "selected": "female, british accent",
            "models": [
                ("female, british accent", "Custom: female, british accent", True),
                ("", "Auto voice", True),
                ("female, child", "Female, child", True),
                ("male, high pitch, indian accent", "Male, high pitch, Indian", True),
                ("female, elderly, british accent", "Female, elderly, British", True),
                ("female, young adult, whisper", "Female, young adult, whisper", True),
                ("male, middle-aged, very low pitch", "Male, middle-aged, very low pitch", True),
                ("female, low pitch, british accent", "Female, low pitch, British", True),
                ("male, british accent", "Male, British", True),
                ("female, whisper, british accent", "Female whisper, British", True),
                ("female, high pitch, american accent", "Female, high pitch, American", True),
                ("male, low pitch, american accent", "Male, low pitch, American", True),
                ("configure_voice", "Set Custom Voice…", True),
            ],
        }

    def test_handle_model_menu_none_surfaces_local_omnivoice_prompt_presets(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._tts_client = None
        saved = {"tts_model": "k2-fsa/OmniVoice"}
        d._load_preference = lambda key: saved.get(key)
        monkeypatch.delenv("SPOKE_TTS_VOICE", raising=False)
        monkeypatch.delenv("SPOKE_TTS_MODEL", raising=False)

        model_state = d._handle_model_menu_action(None)

        assert model_state["tts_voice"] == {
            "type": "choice",
            "title": "TTS Voice: Auto voice",
            "selected": "",
            "models": [
                ("", "Auto voice", True),
                ("female, child", "Female, child", True),
                ("male, high pitch, indian accent", "Male, high pitch, Indian", True),
                ("female, elderly, british accent", "Female, elderly, British", True),
                ("female, young adult, whisper", "Female, young adult, whisper", True),
                ("male, middle-aged, very low pitch", "Male, middle-aged, very low pitch", True),
                ("female, low pitch, british accent", "Female, low pitch, British", True),
                ("male, british accent", "Male, British", True),
                ("female, whisper, british accent", "Female whisper, British", True),
                ("female, high pitch, american accent", "Female, high pitch, American", True),
                ("male, low pitch, american accent", "Male, low pitch, American", True),
                ("configure_voice", "Set Custom Voice…", True),
            ],
        }

    def test_handle_model_menu_none_marks_missing_sidecar_voice_discovery(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = MagicMock()
        d._tts_client._model_id = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
        d._tts_client._voice = "casual_female"
        d._tts_backend = "sidecar"
        d._tts_sidecar_url = "http://other-box:9001"
        d._discover_tts_sidecar_models = MagicMock(
            return_value=[
                ("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "Voxtral 4B (4-bit)", True)
            ]
        )
        d._discover_tts_sidecar_voices = MagicMock(return_value=[])

        model_state = d._handle_model_menu_action(None)

        assert model_state["tts_voice"] == {
            "type": "choice",
            "title": "TTS Voice: Casual Female",
            "selected": "casual_female",
            "models": [
                ("casual_female", "Casual Female", True),
                ("casual_male", "Casual Male", True),
                ("cheerful_female", "Cheerful Female", True),
                ("neutral_female", "Neutral Female", True),
                ("neutral_male", "Neutral Male", True),
                ("fr_female", "French Female", True),
                ("fr_male", "French Male", True),
                ("es_female", "Spanish Female", True),
                ("es_male", "Spanish Male", True),
                ("de_female", "German Female", True),
                ("de_male", "German Male", True),
                ("it_female", "Italian Female", True),
                ("it_male", "Italian Male", True),
                ("pt_female", "Portuguese Female", True),
                ("pt_male", "Portuguese Male", True),
                ("nl_female", "Dutch Female", True),
                ("nl_male", "Dutch Male", True),
                ("ar_male", "Arabic Male", True),
                ("hi_female", "Hindi Female", True),
                ("hi_male", "Hindi Male", True),
                ("configure_voice", "Set Custom Voice…", True),
            ],
        }

    def test_build_tts_client_local_uses_saved_preferences_without_env(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._local_inference_lock = object()
        saved = {
            "tts_voice": "calm_female",
            "tts_model": "mlx-community/Kokoro-82M-bf16",
        }
        d._load_preference = lambda key: saved.get(key)
        monkeypatch.delenv("SPOKE_TTS_VOICE", raising=False)
        monkeypatch.delenv("SPOKE_TTS_MODEL", raising=False)

        with patch.object(main_module, "TTSClient") as MockTTS:
            result = d._build_tts_client()

        MockTTS.assert_called_once_with(
            model_id="mlx-community/Kokoro-82M-bf16",
            voice="calm_female",
            gpu_lock=d._local_inference_lock,
        )
        assert result is MockTTS.return_value

    def test_build_tts_client_local_omnivoice_without_voice_skips_startup_client(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_backend = "local"
        d._tts_sidecar_url = ""
        d._local_inference_lock = object()
        saved = {"tts_model": "k2-fsa/OmniVoice"}
        d._load_preference = lambda key: saved.get(key)
        monkeypatch.delenv("SPOKE_TTS_VOICE", raising=False)
        monkeypatch.delenv("SPOKE_TTS_MODEL", raising=False)

        with patch.object(main_module, "TTSClient") as MockTTS:
            result = d._build_tts_client()

        MockTTS.assert_not_called()
        assert result is None

    def test_handle_model_menu_none_surfaces_transcription_and_preview_backends(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._preview_backend = "local"
        d._whisper_sidecar_url = ""
        d._whisper_cloud_url = ""
        d._whisper_cloud_api_key = ""

        model_state = d._handle_model_menu_action(None)

        expected_items = [
            ("local", "Local Whisper", True),
            ("sidecar", "Sidecar (not configured)", False, False),
            ("cloud", "Cloud (not configured)", False, False),
            ("configure_whisper", "Set Whisper Sidecar URL\u2026", False, True),
            ("configure_whisper_cloud", "Set Cloud API Key\u2026", False, True),
        ]
        assert model_state["transcription_backend"] == {
            "title": "Final: Local",
            "items": expected_items,
        }
        assert model_state["preview_backend"] == {
            "title": "Preview: Local",
            "items": expected_items,
        }

    def test_handle_model_menu_independent_preview_and_transcription_backends(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "cloud"
        d._preview_backend = "local"
        d._whisper_sidecar_url = ""
        d._whisper_cloud_url = "https://api.openai.com"
        d._whisper_cloud_api_key = "sk-test"

        model_state = d._handle_model_menu_action(None)

        assert model_state["transcription_backend"]["title"] == "Final: Cloud (OpenAI)"
        assert model_state["preview_backend"]["title"] == "Preview: Local"
        final_items = model_state["transcription_backend"]["items"]
        assert final_items[2] == ("cloud", "Cloud (OpenAI)", True, True)
        preview_items = model_state["preview_backend"]["items"]
        assert preview_items[0] == ("local", "Local Whisper", True)

    def test_selecting_transcription_backend_sidecar_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._whisper_sidecar_url = "http://my-server:8080"
        d._save_preference = MagicMock()

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("transcription_backend", "sidecar"))

        d._save_preference.assert_called_once_with("whisper_backend", "sidecar")
        mock_execv.assert_called_once()

    def test_selecting_transcription_backend_sidecar_blocked_without_url(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._whisper_sidecar_url = ""
        d._menubar = MagicMock()
        d._save_preference = MagicMock()

        d._handle_model_menu_action(("transcription_backend", "sidecar"))

        d._save_preference.assert_not_called()
        d._menubar.set_status_text.assert_called_once_with(
            "No Whisper sidecar URL configured"
        )

    def test_selecting_transcription_backend_cloud_blocked_without_key(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._whisper_cloud_api_key = ""
        d._menubar = MagicMock()
        d._save_preference = MagicMock()

        d._handle_model_menu_action(("transcription_backend", "cloud"))

        d._save_preference.assert_not_called()
        d._menubar.set_status_text.assert_called_once_with(
            "No cloud API key configured"
        )

    def test_selecting_transcription_backend_noop_when_already_selected(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._save_preference = MagicMock()

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("transcription_backend", "local"))

        d._save_preference.assert_not_called()
        mock_execv.assert_not_called()

    def test_selecting_preview_backend_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._preview_backend = "local"
        d._whisper_sidecar_url = "http://my-server:8080"
        d._save_preference = MagicMock()

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("preview_backend", "sidecar"))

        d._save_preference.assert_called_once_with("preview_backend", "sidecar")
        mock_execv.assert_called_once()

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

    def test_selecting_launch_target_persists_choice_and_invokes_helper(
        self, main_module, monkeypatch
    ):
        """Choosing a new launch target should flow through the launch-target selector."""
        d = _make_delegate(main_module, monkeypatch)
        d._apply_launch_target_selection = MagicMock()

        d._handle_model_menu_action(("launch_target", "smoke"))

        d._apply_launch_target_selection.assert_called_once_with("smoke")
    def test_toggle_local_whisper_eager_eval_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Toggling eager-eval should persist and relaunch without shadowing prefs via env."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._local_whisper_decode_timeout = 30.0
        d._local_whisper_eager_eval = False
        d._save_local_whisper_preferences = MagicMock()
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: True)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("local_whisper", "eager_eval"))

        d._save_local_whisper_preferences.assert_called_once_with(30.0, True)
        assert "SPOKE_LOCAL_WHISPER_EAGER_EVAL" not in os.environ
        assert "SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT" not in os.environ
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
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("local_whisper", "decode_timeout"))

        d._save_local_whisper_preferences.assert_called_once_with(None, False)
        assert "SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT" not in os.environ
        assert "SPOKE_LOCAL_WHISPER_EAGER_EVAL" not in os.environ
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
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)

        with patch.object(main_module.os, "execv") as mock_execv:
            d._handle_model_menu_action(("assistant", "qwen3-14b"))

        d._save_command_model_preference.assert_called_once_with("qwen3-14b")
        assert "SPOKE_COMMAND_MODEL" not in os.environ
        assert os.environ.get("SPOKE_RELAUNCH_COMMAND_MODEL") == "qwen3-14b"
        mock_execv.assert_called_once()

    def test_selecting_assistant_model_survives_relaunch_via_saved_preferences(
        self, main_module, monkeypatch, tmp_path
    ):
        """Assistant model changes should survive relaunch from prefs without env shadowing."""
        prefs_file = tmp_path / "model_preferences.json"
        prefs_file.write_text(
            '{\n'
            '  "command_model": "qwen3p5-35B-A3B"\n'
            '}\n'
        )
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)

        first = _make_delegate(main_module, monkeypatch)
        first._command_client = MagicMock()
        first._command_model_id = "qwen3p5-35B-A3B"
        first._command_model_options = [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", True),
        ]
        first._save_command_model_preference = (
            main_module.SpokeAppDelegate._save_command_model_preference.__get__(
                first, main_module.SpokeAppDelegate
            )
        )
        first._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            first, main_module.SpokeAppDelegate
        )
        first._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            first, main_module.SpokeAppDelegate
        )
        first._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            first, main_module.SpokeAppDelegate
        )

        with patch.object(main_module.os, "execv"):
            first._handle_model_menu_action(("assistant", "qwen3-14b"))

        reloaded = json.loads(prefs_file.read_text())
        assert reloaded["command_model"] == "qwen3-14b"

        second = _make_delegate(main_module, monkeypatch)
        second._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            second, main_module.SpokeAppDelegate
        )

        assert second._load_command_model_preference() == "qwen3-14b"

    def test_discover_command_models_merges_server_and_local_inventory(
        self, main_module, monkeypatch, tmp_path
    ):
        """Assistant discovery should keep only curated installed local MLX models."""
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
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            ),
        ]

    def test_discover_command_models_prefers_server_inventory_for_sidecar(
        self, main_module, monkeypatch, tmp_path
    ):
        """Sidecar discovery should show the server's model list instead of local disk inventory."""
        model_root = tmp_path / "models"
        curated = model_root / "lmstudio-community" / "Qwen3-4B-Instruct-2507-MLX-6bit"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors.index.json").write_text("{}")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "sidecar"
        d._command_client = MagicMock()
        d._command_client.list_models.return_value = ["qwen3p5-35B-A3B", "qwen3-14b"]

        options = d._discover_command_models("qwen3p5-35B-A3B")

        assert options == [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", False),
        ]

    def test_discover_command_models_alphabetizes_openrouter_inventory(
        self, main_module, monkeypatch
    ):
        """OpenRouter cloud discovery should sort provider models alphabetically for menu usability."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "cloud"
        d._command_cloud_provider = "openrouter"
        d._command_client = MagicMock()
        d._command_client.list_models.return_value = [
            "z-ai/glm-4.5-air:free",
            "anthropic/claude-3.7-sonnet",
            "stepfun/step-3.5-flash:free",
        ]

        options = d._discover_command_models("stepfun/step-3.5-flash:free")

        assert options == [
            ("anthropic/claude-3.7-sonnet", "anthropic/claude-3.7-sonnet", False),
            ("stepfun/step-3.5-flash:free", "stepfun/step-3.5-flash:free", True),
            ("z-ai/glm-4.5-air:free", "z-ai/glm-4.5-air:free", False),
        ]

    def test_discover_command_models_returns_empty_when_server_is_unavailable(
        self, main_module, monkeypatch, tmp_path
    ):
        """Local-disk models must not appear when the server is unreachable."""
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

        assert options == []

    def test_seed_command_model_options_empty_when_server_unreachable(
        self, main_module, monkeypatch, tmp_path
    ):
        """Initial Assistant menu should be empty when the local server is unreachable,
        even if curated models exist on disk."""
        model_root = tmp_path / "models"
        curated = model_root / "alexgusevski" / "LFM2.5-1.2B-Nova-Function-Calling-mlx"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors.index.json").write_text("{}")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        # No _command_client set → server unreachable

        options = d._seed_command_model_options("qwen3p5-35B-A3B")

        assert options == []

    def test_seed_command_model_options_keeps_sidecar_seed_to_selected_model(
        self, main_module, monkeypatch, tmp_path
    ):
        """Sidecar startup should avoid seeding the Assistant menu from local disk."""
        model_root = tmp_path / "models"
        curated = model_root / "alexgusevski" / "LFM2.5-1.2B-Nova-Function-Calling-mlx"
        curated.mkdir(parents=True)
        (curated / "config.json").write_text("{}")
        (curated / "tokenizer.json").write_text("{}")
        (curated / "model.safetensors.index.json").write_text("{}")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "sidecar"

        options = d._seed_command_model_options(
            "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx"
        )

        assert options == [
            (
                "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
                "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
                True,
            ),
        ]

    def test_seed_command_model_options_uses_server_inventory_when_local_curated_list_empty(
        self, main_module, monkeypatch, tmp_path
    ):
        """Local startup seed should use live server models when no curated disk models are present."""
        model_root = tmp_path / "models"
        model_root.mkdir()
        monkeypatch.setenv("SPOKE_COMMAND_MODEL_DIR", str(model_root))

        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "local"
        d._command_client = MagicMock()
        d._command_client.list_models.return_value = ["qwen3-14b", "step-3p5-flash-mixedp-final"]

        options = d._seed_command_model_options("step-3p5-flash-mixedp-final")

        assert options == [
            ("qwen3-14b", "qwen3-14b", False),
            ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", True),
        ]

    def test_command_models_discovered_heals_stale_sidecar_selection_without_relaunch(
        self, main_module, monkeypatch
    ):
        """A stale local-only assistant model should heal to the first sidecar model after refresh."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "sidecar"
        d._command_model_id = "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx"
        d._command_client = MagicMock()
        d._menubar = MagicMock()
        d._save_command_model_preference = MagicMock(return_value=True)

        d.commandModelsDiscovered_(
            {
                "options": [
                    ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", False),
                    ("qwen3-14b", "qwen3-14b", False),
                ]
            }
        )

        assert d._command_model_id == "qwen3p5-35B-A3B"
        assert d._command_client._model == "qwen3p5-35B-A3B"
        d._save_command_model_preference.assert_called_once_with("qwen3p5-35B-A3B")
        assert d._command_model_options == [
            ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
            ("qwen3-14b", "qwen3-14b", False),
        ]
        d._menubar.refresh_menu.assert_called_once_with()

    def test_command_models_discovered_heals_local_selection_back_to_persisted_model(
        self, main_module, monkeypatch
    ):
        """Local startup fallback should heal back to the persisted assistant model once options arrive."""
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "step-3p5-flash-mixedp-final")
        d = _make_delegate(main_module, monkeypatch)
        d._command_backend = "local"
        d._command_model_id = "step-3p5-flash-mixedp-final"
        d._command_client = MagicMock()
        d._menubar = MagicMock()
        d._save_command_model_preference = MagicMock(return_value=True)
        d._load_command_model_preference = MagicMock(return_value="qwen3-14b")

        d.commandModelsDiscovered_(
            {
                "options": [
                    ("qwen3-14b", "qwen3-14b", False),
                    ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", True),
                ]
            }
        )

        assert d._command_model_id == "qwen3-14b"
        assert d._command_client._model == "qwen3-14b"
        d._save_command_model_preference.assert_not_called()
        assert d._command_model_options == [
            ("qwen3-14b", "qwen3-14b", True),
            ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", False),
        ]
        d._menubar.refresh_menu.assert_called_once_with()

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

    def test_provider_scoped_cloud_preferences_persist_independently(
        self, main_module, monkeypatch, tmp_path
    ):
        """Google Cloud and OpenRouter should keep separate saved URL/key/model state."""
        prefs_file = tmp_path / "model_preferences.json"
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_cloud_preferences = main_module.SpokeAppDelegate._save_cloud_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_cloud_url_preference = main_module.SpokeAppDelegate._load_cloud_url_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_cloud_api_key_preference = main_module.SpokeAppDelegate._load_cloud_api_key_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_cloud_model_preference = main_module.SpokeAppDelegate._load_cloud_model_preference.__get__(
            d, main_module.SpokeAppDelegate
        )

        assert d._save_cloud_preferences(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "google-key",
            "gemini-2.5-flash",
        )
        assert d._save_cloud_preferences(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "openrouter-key",
            "stepfun/step-3.5-flash:free",
        )

        assert d._load_cloud_url_preference("google") == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert d._load_cloud_api_key_preference("google") == "google-key"
        assert d._load_cloud_model_preference("google") == "gemini-2.5-flash"
        assert d._load_cloud_url_preference("openrouter") == "https://openrouter.ai/api/v1"
        assert d._load_cloud_api_key_preference("openrouter") == "openrouter-key"
        assert d._load_cloud_model_preference("openrouter") == "stepfun/step-3.5-flash:free"

    def test_resolve_command_cloud_api_key_prefers_provider_specific_env_over_generic(
        self, main_module, monkeypatch, tmp_path
    ):
        """Provider-specific env keys should beat the generic cloud fallback."""
        prefs_file = tmp_path / "model_preferences.json"
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))
        monkeypatch.setenv("SPOKE_COMMAND_CLOUD_API_KEY", "generic-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_cloud_provider_preference = main_module.SpokeAppDelegate._load_cloud_provider_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._load_cloud_api_key_preference = main_module.SpokeAppDelegate._load_cloud_api_key_preference.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._resolve_command_cloud_api_key = main_module.SpokeAppDelegate._resolve_command_cloud_api_key.__get__(
            d, main_module.SpokeAppDelegate
        )

        assert d._resolve_command_cloud_api_key(
            "https://openrouter.ai/api/v1", "openrouter"
        ) == "openrouter-key"
        assert d._resolve_command_cloud_api_key(
            "https://generativelanguage.googleapis.com/v1beta/openai", "google"
        ) == "gemini-key"

    def test_save_google_cloud_preferences_keeps_legacy_generic_keys_aligned(
        self, main_module, monkeypatch, tmp_path
    ):
        """Saving Google Cloud settings should continue updating the legacy generic keys."""
        prefs_file = tmp_path / "model_preferences.json"
        monkeypatch.setenv("SPOKE_MODEL_PREFERENCES_PATH", str(prefs_file))

        d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
        d._load_preferences = main_module.SpokeAppDelegate._load_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._preferences_path = main_module.SpokeAppDelegate._preferences_path.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_preferences = main_module.SpokeAppDelegate._save_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )
        d._save_cloud_preferences = main_module.SpokeAppDelegate._save_cloud_preferences.__get__(
            d, main_module.SpokeAppDelegate
        )

        assert d._save_cloud_preferences(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "google-key",
            "gemini-2.5-flash",
        )

        loaded = json.loads(prefs_file.read_text())
        assert loaded["command_cloud_provider"] == "google"
        assert loaded["command_cloud_google_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert loaded["command_cloud_google_api_key"] == "google-key"
        assert loaded["command_cloud_google_model"] == "gemini-2.5-flash"
        assert loaded["command_cloud_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert loaded["command_cloud_api_key"] == "google-key"
        assert loaded["command_cloud_model"] == "gemini-2.5-flash"

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
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
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
        monkeypatch.setattr(main_module, "_RAM_GB", 32.0)
        monkeypatch.delenv("SPOKE_WHISPER_URL", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOKE_LOCAL_WHISPER_EAGER_EVAL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_local_whisper_preferences",
            lambda self: {},
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
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
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "local",
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
        assert d._command_model_id == "qwen3-14b"
        MockCommand.assert_called_once_with(
            base_url=main_module._DEFAULT_COMMAND_URL,
            model="qwen3-14b",
        )
        assert d._command_model_options == [
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            )
        ]
        mock_seed.assert_called_once_with("qwen3-14b")

    def test_init_prefers_env_command_model_over_stale_persisted_local_selection(
        self, main_module, monkeypatch
    ):
        """Smoke env should override a dead persisted local assistant model at bootstrap."""
        monkeypatch.delenv("SPOKE_RELAUNCH_COMMAND_MODEL", raising=False)
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "step-3p5-flash-mixedp-final")
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "Harmonic-27B-MLX-16bit",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "local",
            raising=False,
        )
        with patch.object(main_module, "CommandClient") as MockCommand:
            MockCommand.return_value = MagicMock()
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", True)],
            ):
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        MockCommand.assert_called_once_with(
            base_url=main_module._DEFAULT_COMMAND_URL,
            model="step-3p5-flash-mixedp-final",
        )
        assert d._command_model_id == "step-3p5-flash-mixedp-final"

    def test_init_prefers_valid_persisted_local_selection_over_smoke_env_override(
        self, main_module, monkeypatch, tmp_path
    ):
        """A valid local assistant choice should survive relaunch even when smoke env pins a default."""
        monkeypatch.delenv("SPOKE_RELAUNCH_COMMAND_MODEL", raising=False)
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "step-3p5-flash-mixedp-final")
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "local",
            raising=False,
        )
        with patch.object(main_module, "CommandClient") as MockCommand:
            MockCommand.return_value = MagicMock()
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[
                    ("qwen3-14b", "qwen3-14b", True),
                    ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", False),
                ],
            ):
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        assert d._command_model_id == "qwen3-14b"
        assert d._command_client._model == "qwen3-14b"
        assert d._command_model_options == [
            ("qwen3-14b", "qwen3-14b", True),
            ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", False),
        ]

    def test_init_prefers_relaunch_command_model_override_over_smoke_env(
        self, main_module, monkeypatch
    ):
        """The immediate relaunch override should beat the smoke-env default for the next process."""
        monkeypatch.setenv("SPOKE_RELAUNCH_COMMAND_MODEL", "qwen3-14b")
        monkeypatch.setenv("SPOKE_COMMAND_MODEL", "step-3p5-flash-mixedp-final")
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3p5-35B-A3B",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "local",
            raising=False,
        )
        with patch.object(main_module, "CommandClient") as MockCommand:
            MockCommand.return_value = MagicMock()
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[
                    ("qwen3-14b", "qwen3-14b", True),
                    ("step-3p5-flash-mixedp-final", "step-3p5-flash-mixedp-final", False),
                ],
            ):
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        assert d._command_model_id == "qwen3-14b"
        MockCommand.assert_called_once_with(
            base_url=main_module._DEFAULT_COMMAND_URL,
            model="qwen3-14b",
        )
        assert "SPOKE_RELAUNCH_COMMAND_MODEL" not in os.environ

    def test_init_seeds_command_model_options_without_sync_discovery(
        self, main_module, monkeypatch
    ):
        """Startup should not block on /v1/models just to seed the Assistant menu."""
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "local",
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
        MockCommand.assert_called_once_with(
            base_url=main_module._DEFAULT_COMMAND_URL,
            model="qwen3-14b",
        )
        command_client.list_models.assert_not_called()
        mock_seed.assert_called_once_with("qwen3-14b")
        assert d._command_model_options == [
            (
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                False,
            )
        ]

    def test_selecting_assistant_backend_sidecar_persists_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Choosing the sidecar backend should persist it and relaunch against that URL."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_backend = "local"
        d._command_url = "http://localhost:8001"
        d._command_sidecar_url = "http://other-box:8001"
        d._save_command_backend_preferences = MagicMock(return_value=True)
        d._relaunch = MagicMock()

        d._handle_model_menu_action(("assistant_backend", "sidecar"))

        d._save_command_backend_preferences.assert_called_once_with(
            "sidecar", "http://other-box:8001"
        )
        d._relaunch.assert_called_once_with()

    def test_selecting_assistant_backend_sidecar_prompts_for_url_when_missing(
        self, main_module, monkeypatch
    ):
        """Choosing sidecar with no saved URL should ask once, persist, and relaunch."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_backend = "local"
        d._command_url = "http://localhost:8001"
        d._command_sidecar_url = None
        d._prompt_for_command_sidecar_url = MagicMock(
            return_value="http://other-box:8001"
        )
        d._save_command_backend_preferences = MagicMock(return_value=True)
        d._relaunch = MagicMock()

        d._handle_model_menu_action(("assistant_backend", "sidecar"))

        d._prompt_for_command_sidecar_url.assert_called_once_with("")
        d._save_command_backend_preferences.assert_called_once_with(
            "sidecar", "http://other-box:8001"
        )
        d._relaunch.assert_called_once_with()

    def test_selecting_assistant_backend_openrouter_persists_provider_and_relaunches(
        self, main_module, monkeypatch
    ):
        """Choosing OpenRouter should persist the cloud provider and relaunch on its URL."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_backend = "local"
        d._command_url = "http://localhost:8001"
        d._save_command_backend_preferences = MagicMock(return_value=True)
        d._save_command_cloud_provider_preference = MagicMock(return_value=True)
        d._load_cloud_url_preference = MagicMock(return_value="https://openrouter.ai/api/v1")
        d._relaunch = MagicMock()

        d._handle_model_menu_action(("assistant_backend", "cloud_openrouter"))

        d._save_command_backend_preferences.assert_called_once_with("cloud", None)
        d._save_command_cloud_provider_preference.assert_called_once_with("openrouter")
        assert d._command_backend == "cloud"
        assert d._command_cloud_provider == "openrouter"
        assert d._command_url == "https://openrouter.ai/api/v1"
        d._relaunch.assert_called_once_with()

    def test_configuring_assistant_sidecar_url_persists_without_relaunch_when_local_backend_active(
        self, main_module, monkeypatch
    ):
        """Saving a sidecar URL should not force a relaunch while local backend remains active."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_backend = "local"
        d._command_url = "http://localhost:8001"
        d._command_sidecar_url = None
        d._prompt_for_command_sidecar_url = MagicMock(
            return_value="http://other-box:8001"
        )
        d._save_command_backend_preferences = MagicMock(return_value=True)
        d._relaunch = MagicMock()

        d._handle_model_menu_action(("assistant_backend", "configure"))

        d._save_command_backend_preferences.assert_called_once_with(
            "local", "http://other-box:8001"
        )
        d._menubar.set_status_text.assert_called_with("Assistant sidecar URL saved")
        d._relaunch.assert_not_called()

    def test_selecting_tts_voice_auto_clears_local_omnivoice_prompt(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = MagicMock()
        d._tts_client._voice = "female, british accent"
        d._save_preference = MagicMock(return_value=True)
        d._relaunch = MagicMock()

        d._handle_model_menu_action(("tts_voice", ""))

        d._save_preference.assert_called_once_with("tts_voice", "")
        d._relaunch.assert_called_once_with()

    def test_configuring_local_omnivoice_prompt_surfaces_upstream_lexicon_help(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = None
        d._tts_backend = "local"
        d._load_preference = lambda key: {"tts_model": "k2-fsa/OmniVoice"}.get(key)

        alert = MagicMock()
        alert.runModal.return_value = 1001
        field = MagicMock()
        field.stringValue.return_value = ""
        main_module.NSAlert.new.return_value = alert
        main_module.NSTextField.alloc.return_value.initWithFrame_.return_value = field

        d._configure_tts_voice()

        alert.setMessageText_.assert_called_once_with("TTS Prompt")
        informative_text = alert.setInformativeText_.call_args.args[0]
        assert "Gender: female, male" in informative_text
        assert "Age: child, young adult, middle-aged, elderly" in informative_text
        assert "Pitch: very low pitch, low pitch, high pitch, very high pitch" in informative_text
        assert "Style: whisper" in informative_text
        assert "English accent: american accent, british accent, indian accent" in informative_text
        assert "female, low pitch, british accent" in informative_text

    def test_init_prefers_persisted_sidecar_backend_over_launcher_default_local_url(
        self, main_module, monkeypatch
    ):
        """Saved sidecar selection should beat the persisted local default."""
        monkeypatch.delenv("SPOKE_COMMAND_MODEL", raising=False)
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_model_preference",
            lambda self: "qwen3-14b",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_backend_preference",
            lambda self: "sidecar",
            raising=False,
        )
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_command_sidecar_url_preference",
            lambda self: "http://other-box:8001",
            raising=False,
        )
        with patch.object(main_module, "CommandClient") as MockCommand:
            MockCommand.return_value = MagicMock()
            with patch.object(
                main_module.SpokeAppDelegate,
                "_seed_command_model_options",
                return_value=[
                    (
                        "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                        "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
                        False,
                    )
                ],
            ):
                d = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
                result = d.init()

        assert result is not None
        MockCommand.assert_called_once_with(
            base_url="http://other-box:8001",
            model="qwen3-14b",
        )
        assert d._command_backend == "sidecar"
        assert d._command_url == "http://other-box:8001"
        assert d._command_sidecar_url == "http://other-box:8001"


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

        with patch.object(main_module, "_record_runtime_phase") as mock_phase:
            d._prepare_clients_in_background()

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "clientWarmupSucceeded:", None, False
        )
        mock_phase.assert_any_call("client_warmup.start")
        mock_phase.assert_any_call("client_warmup.succeeded")

    def test_background_warmup_dispatches_failure_to_main_thread(
        self, main_module, monkeypatch
    ):
        """Warmup failures should be surfaced back on the main thread."""
        d = _make_delegate(main_module, monkeypatch)
        d._prepare_clients = MagicMock(side_effect=RuntimeError("warm failed"))

        with patch.object(main_module, "_record_runtime_phase") as mock_phase:
            d._prepare_clients_in_background()

        d.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_once_with(
            "clientWarmupFailed:", None, False
        )
        assert isinstance(d._warm_error, RuntimeError)
        mock_phase.assert_any_call("client_warmup.start")
        mock_phase.assert_any_call("client_warmup.failed", error="warm failed")

    def test_prepare_clients_warms_local_whisper_at_startup(
        self, main_module, monkeypatch
    ):
        """Local MLX Whisper clients are warmed eagerly at startup."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._model_allowed = MagicMock(return_value=True)
        d._transcription_model_id = "mlx-community/whisper-small.en-mlx"
        d._preview_model_id = "mlx-community/whisper-tiny.en-mlx"
        d._client = main_module.LocalTranscriptionClient(model=d._transcription_model_id)
        d._preview_client = main_module.LocalTranscriptionClient(model=d._preview_model_id)

        with patch.object(d._client, "prepare") as prep_client, patch.object(
            d._preview_client, "prepare"
        ) as prep_preview:
            d._prepare_clients()

        prep_client.assert_called_once()
        prep_preview.assert_called_once()

    def test_prepare_clients_warms_local_qwen_at_startup(
        self, main_module, monkeypatch
    ):
        """Local MLX Qwen client is warmed eagerly at startup."""
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._model_allowed = MagicMock(return_value=True)
        d._transcription_model_id = "Qwen/Qwen3-ASR-0.6B"
        d._client = main_module.LocalQwenClient(model=d._transcription_model_id)
        d._preview_model_id = None
        d._preview_client = None

        with patch.object(d._client, "prepare") as prep_client:
            d._prepare_clients()

        prep_client.assert_called_once()

    def test_warmup_success_hides_startup_indicator(
        self, main_module, monkeypatch
    ):
        """Successful warmup should remove the loading overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = None
        with patch.object(main_module, "_record_runtime_phase") as mock_phase:
            d.clientWarmupSucceeded_(None)

        d._overlay.hide.assert_called_once_with()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")
        mock_phase.assert_called_with("app.ready")

    def test_warmup_success_does_not_start_tts_warmup(
        self, main_module, monkeypatch
    ):
        """Startup must not warm TTS, because that can starve transcription on local MLX."""
        d = _make_delegate(main_module, monkeypatch)
        d._tts_client = MagicMock()

        with patch.object(main_module.threading, "Thread") as mock_thread_cls:
            d.clientWarmupSucceeded_(None)

        mock_thread_cls.assert_not_called()

    def test_warmup_success_does_not_auto_enable_handsfree(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._handsfree = MagicMock()
        monkeypatch.setenv("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY", "test-key")

        d.clientWarmupSucceeded_(None)

        d._handsfree.enable.assert_not_called()

class TestRuntimePhaseLogging:
    """Test runtime phase snapshot behavior under repeated writes."""

    def test_record_runtime_phase_handles_concurrent_updates(
        self, main_module, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(
            "SPOKE_RUNTIME_PHASE_PATH",
            str(tmp_path / "spoke-last-phase.json"),
        )

        real_write_text = main_module.Path.write_text
        barrier = threading.Barrier(2)
        tmp_names = []

        def synced_write_text(self, data, *args, **kwargs):
            result = real_write_text(self, data, *args, **kwargs)
            if self.name.startswith("spoke-last-phase.json") and self.name.endswith(".tmp"):
                tmp_names.append(self.name)
                barrier.wait(timeout=1.0)
            return result

        with patch.object(main_module.Path, "write_text", new=synced_write_text):
            with patch.object(main_module.logger, "exception") as mock_exception:
                threads = [
                    threading.Thread(
                        target=main_module._record_runtime_phase,
                        args=(f"phase-{idx}",),
                    )
                    for idx in range(2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=1.0)

        assert not any(thread.is_alive() for thread in threads)
        mock_exception.assert_not_called()
        assert len(tmp_names) == 2
        assert len(set(tmp_names)) == 2

        payload = json.loads((tmp_path / "spoke-last-phase.json").read_text())
        assert payload["phase"] in {"phase-0", "phase-1"}

    def test_record_runtime_phase_includes_visible_launch_target(self, main_module, monkeypatch, tmp_path):
        """Crash breadcrumbs should name the visible Launch Target."""
        phase_path = tmp_path / "spoke-last-phase.json"
        monkeypatch.setenv("SPOKE_RUNTIME_PHASE_PATH", str(phase_path))
        monkeypatch.setattr(
            main_module,
            "current_launch_target",
            lambda _cwd: {
                "id": "airstrike",
                "label": "Assistant Backend on Main Next Airstrike",
                "path": tmp_path,
                "enabled": True,
            },
        )

        with patch.object(main_module.logger, "info") as mock_info:
            main_module._record_runtime_phase("process.start", detail="value")

        payload = json.loads(phase_path.read_text())
        assert payload["launch_target_id"] == "airstrike"
        assert payload["launch_target_label"] == "Assistant Backend on Main Next Airstrike"
        assert mock_info.call_args[0][0] == "Runtime phase: %s (%s)"
        assert mock_info.call_args[0][1] == "process.start"
        log_detail_text = mock_info.call_args[0][2]
        assert "launch_target_id='airstrike'" in log_detail_text
        assert "launch_target_label='Assistant Backend on Main Next Airstrike'" in log_detail_text

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
        d._setup_event_tap = MagicMock()

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
            sys.modules["spoke.terraform_hud"] = MagicMock()
            sys.modules["spoke.terraform_hud"].TerraformHUD.alloc.return_value.init.return_value = MagicMock()

            d.applicationDidFinishLaunching_(None)

        d._refresh_command_model_options_async.assert_called_once_with()
        d._setup_event_tap.assert_called_once_with()

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
        monkeypatch.setattr(
            main_module.SpokeAppDelegate,
            "_load_preferences",
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
        d._command_client._history = []
        d._command_client._max_history = 5
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
        d._narrator = None
        d._ensure_tts_client = MagicMock(return_value=None)
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
        assert d._command_client._history == []

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

    def test_stream_http_error_dispatches_provider_status_detail(
        self, main_module, monkeypatch
    ):
        """HTTP failures should preserve provider status/body detail for the UI."""
        d = self._make_command_delegate(main_module, monkeypatch)
        d._client.transcribe.return_value = "do something"
        d._client.supports_streaming = False
        error_body = json.dumps(
            {"error": {"message": "This model is temporarily rate-limited."}}
        ).encode()
        d._command_client.stream_command_events.side_effect = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )

        d._command_transcribe_worker(b"wav-data", 1)

        failure_call = next(
            c
            for c in d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args_list
            if c[0][0] == "commandFailed:"
        )
        assert (
            failure_call[0][1]["error"]
            == "HTTP 429 Too Many Requests — This model is temporarily rate-limited."
        )

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
        complete_call = next(c for c in calls if c[0][0] == "commandComplete:")
        assert complete_call[0][1]["response"] == "first"
        d._command_client.append_history_pair.assert_called_once_with(
            "do something",
            "first",
        )

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

    def test_command_approval_required_shows_pending_approval_card(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True

        d.commandApprovalRequired_(
            {
                "token": 1,
                "approval_request": {
                    "kind": "terminal_command",
                    "argv": ["git", "commit", "-m", "x"],
                    "cwd": "/tmp/repo",
                    "reason": "command requires approval: git commit -m",
                    "message": "Approval needed\n\ngit commit -m x",
                },
            }
        )

        assert d._transcribing is False
        assert d._pending_command_approval_active is True
        assert d._detector.approval_active is True
        assert d._detector.command_overlay_active is True
        d._command_overlay.set_tool_active.assert_called_once_with(False)
        d._command_overlay.set_response_text.assert_called_once_with(
            "Approval needed\n\ngit commit -m x"
        )
        d._command_overlay.finish.assert_called_once_with()
        d._menubar.set_status_text.assert_called_once_with("Approval needed")

    def test_command_complete_finish_failure_hides_glow(
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
        # Autoplay removed — TTS is only triggered via read_aloud tool
        d._tts_client.speak_async.assert_not_called()
        d._glow.hide.assert_called()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")
        assert "Command overlay finish failed" in caplog.text

    def test_command_complete_no_autoplay(
        self, main_module, monkeypatch
    ):
        """commandComplete_ does not auto-play TTS — the model uses read_aloud tool instead."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._transcription_token = 1
        d._transcribing = True
        d._tts_client = MagicMock()

        d.commandComplete_({"token": 1, "response": "Hello there"})

        assert d._transcribing is False
        d._tts_client.speak_async.assert_not_called()
        d._glow.hide.assert_called()

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
        d._tts_client.speak.assert_not_called()

    def test_tool_executor_lazily_builds_tts_client_for_read_aloud(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._tts_client = None
        built_tts = MagicMock()
        d._ensure_tts_client = MagicMock(return_value=built_tts)

        executor = d._make_tool_executor()
        result = executor("read_aloud", {"source_ref": "literal:hello world"})

        d._ensure_tts_client.assert_called_once_with(allow_default_voice=True)
        built_tts.speak_async.assert_called_once_with("hello world")
        built_tts.speak.assert_not_called()
        assert result == "Speaking: hello world"
        assert d._command_tool_used_tts is True

    def test_approval_tap_runs_pending_command(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._approve_pending_command = MagicMock()
        d._pending_command_approval_active = True

        d._on_hold_end(shift_held=False, enter_held=False, approval_tap=True)

        d._approve_pending_command.assert_called_once_with()

    def test_approval_shift_tap_cancels_pending_command(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._cancel_pending_command_approval = MagicMock()
        d._pending_command_approval_active = True

        d._on_hold_end(shift_held=True, enter_held=False, approval_tap=True)

        d._cancel_pending_command_approval.assert_called_once_with()

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

    def test_tool_executor_forwards_tool_output_mode(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._ensure_tts_client = MagicMock(return_value=None)

        with patch.object(main_module, "execute_tool", return_value='{"ok": true}') as exec_tool:
            executor = d._make_tool_executor()
            result = executor(
                "capture_context",
                {"scope": "active_window"},
                tool_output_mode="multimodal",
            )

        assert result == '{"ok": true}'
        exec_tool.assert_called_once()
        assert exec_tool.call_args.kwargs["tool_output_mode"] == "multimodal"

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

        assert "Error speaking text: TTS playback failed." in result
        assert "device unavailable" in result
        assert d._command_tool_used_tts is False
        d._tts_client.speak_async.assert_called_once_with("hello world")
        d._tts_client.speak.assert_not_called()

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

    def test_command_failed_preserves_specific_provider_error_in_overlay(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._command_overlay._visible = False
        d._transcription_token = 1
        d._transcribing = True
        d._last_command_utterance = "test utterance"
        d._last_command_response = ""

        d.commandFailed_(
            {
                "token": 1,
                "error": "HTTP 402 Payment Required — This model requires paid credits.",
            }
        )

        d._command_overlay.append_token.assert_called_once_with(
            "HTTP 402 Payment Required — This model requires paid credits."
        )
        assert (
            d._last_command_response
            == "HTTP 402 Payment Required — This model requires paid credits."
        )

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

    def test_recall_last_response_uses_error_snapshot_when_history_empty(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = []
        d._command_overlay = MagicMock()
        d._last_command_utterance = "what time is it"
        d._last_command_response = "couldn't reach the model — try again in a moment"
        d._transcription_token = 1
        d._transcribing = True

        d._recallLastResponse_({"token": 1})

        d._command_overlay.show.assert_called_once()
        d._command_overlay.set_utterance.assert_called_once_with("what time is it")
        d._command_overlay.finish.assert_called_once()


class TestResultInjection:
    """Test timing of the post-injection overlay cleanup."""

    def test_inject_result_text_orders_out_overlay_before_delayed_inject(
        self, main_module, monkeypatch
    ):
        """Overlay should be ordered out before the delayed paste fires."""
        d = _make_delegate(main_module, monkeypatch)

        with patch.object(main_module, "inject_text"):
            d._inject_result_text("hello", "Ready")

        d._overlay.order_out.assert_called()


class TestCommandOverlayToggle:
    def test_toggle_command_overlay_resumes_in_progress_timer_without_reset(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_overlay = MagicMock(_visible=False)
        d._transcribing = True
        d._last_command_utterance = "open file"
        d._command_streaming_text = "still working"

        d._toggle_command_overlay()

        d._command_overlay.show.assert_called_once_with(preserve_thinking_timer=True)
        d._command_overlay.set_utterance.assert_called_once_with("open file")
        d._command_overlay.set_response_text.assert_called_once_with("still working")
        d._command_overlay.invert_thinking_timer.assert_called_once()


class TestHoldStartDuringTranscription:
    """Test interrupt-and-restart when hold starts during active transcription."""

    def test_hold_during_transcription_does_not_cancel_generation(self, main_module, monkeypatch):
        """Starting a new hold while generation is active should NOT kill the stream.
        Generation continues while the user records a new utterance."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 5
        d._models_ready = True

        d._on_hold_start()

        assert d._transcription_token == 5  # unchanged
        assert d._transcribing is True  # still generating
        # Should have fallen through to start recording
        d._capture.start.assert_called_once()

    def test_hold_during_transcription_starts_new_recording(self, main_module, monkeypatch):
        """Recording starts normally even while generation is in progress."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 0
        d._models_ready = True

        d._on_hold_start()

        d._menubar.set_recording.assert_called_with(True)
        d._glow.show.assert_called_once()

    def test_hold_during_transcription_keeps_visible_command_overlay_active(
        self, main_module, monkeypatch
    ):
        """Recording during generation must not hide the visible assistant overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._models_ready = True
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        d._on_hold_start()

        d._command_overlay.hide.assert_not_called()
        assert d._detector.command_overlay_active is True

    def test_plain_space_release_during_active_turn_uses_parallel_insert_lane(
        self, main_module, monkeypatch
    ):
        """Plain space during generation should fork into its own insertion lane."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 5
        d._capture.stop.return_value = b"ambient-noise"
        d._record_start_time = time.monotonic() - 0.6
        d._last_preview_text = ""

        with patch.object(main_module.threading, "Thread") as mock_thread:
            d._on_hold_end(shift_held=False, enter_held=False)

        assert d._transcription_token == 5
        assert d._parallel_insert_token == 1
        mock_thread.assert_called_once()
        assert mock_thread.call_args.kwargs["target"] == d._parallel_insert_worker
        assert mock_thread.call_args.kwargs["args"] == (b"ambient-noise", 1)

    def test_plain_space_release_with_preview_text_does_not_use_assistant_or_tray_path(
        self, main_module, monkeypatch
    ):
        """A real utterance during generation should stay off the assistant and tray paths."""
        d = _make_delegate(main_module, monkeypatch)
        d._transcribing = True
        d._transcription_token = 5
        d._capture.stop.return_value = b"spoken-audio"
        d._record_start_time = time.monotonic() - 0.6
        d._last_preview_text = "capture this after the file read"
        d._add_tray_entry = MagicMock()

        with patch.object(main_module.threading, "Thread") as mock_thread:
            d._on_hold_end(shift_held=False, enter_held=False)

        assert d._transcription_token == 5
        assert d._parallel_insert_token == 1
        mock_thread.assert_called_once()
        d._add_tray_entry.assert_not_called()
class TestMicNotReady:
    """Hold is rejected and status reflects mic unavailability."""

    def test_hold_rejected_when_mic_not_ready(self, main_module, monkeypatch):
        """Spacebar hold with models ready but mic unavailable should not start recording."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = True
        d._mic_ready = False

        d._on_hold_start()

        d._capture.start.assert_not_called()
        d._menubar.set_status_text.assert_called_with("Mic unavailable — retrying…")

    def test_warmup_succeeded_shows_mic_unavailable_when_mic_not_ready(self, main_module, monkeypatch):
        """clientWarmupSucceeded_ should reflect mic state in menubar status."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_ready = False
        d._warmup_in_flight = True

        d.clientWarmupSucceeded_(None)

        assert d._models_ready is True
        d._menubar.set_status_text.assert_called_with(
            "Models ready — mic unavailable, retrying…"
        )

    def test_mic_granted_after_models_ready_shows_ready(self, main_module, monkeypatch):
        """micPermissionGranted_ when models already loaded should show full ready state."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = True
        d._mic_ready = False
        d._mic_probe_in_flight = True

        d.micPermissionGranted_(None)

        assert d._mic_ready is True
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")


class TestCaptureStartFailure:
    """Capture failure at hold time shows memory-pressure guidance."""

    def test_capture_failure_shows_overlay_notice(self, main_module, monkeypatch):
        """When capture.start() throws, the overlay should flash a notice
        explaining audio is unavailable due to memory pressure."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = True
        d._mic_ready = True
        d._capture.start.side_effect = RuntimeError("PortAudio error -9986")

        d._on_hold_start()

        d._overlay.flash_notice.assert_called_once()
        msg = d._overlay.flash_notice.call_args[0][0]
        assert "audio" in msg.lower() or "microphone" in msg.lower()
        assert "memory" in msg.lower() or "pressure" in msg.lower()

    def test_capture_failure_updates_menubar(self, main_module, monkeypatch):
        """Menubar status should mention memory pressure on capture failure."""
        d = _make_delegate(main_module, monkeypatch)
        d._models_ready = True
        d._mic_ready = True
        d._capture.start.side_effect = RuntimeError("PortAudio error -9986")

        d._on_hold_start()

        d._menubar.set_recording.assert_called_with(False)
        status_text = d._menubar.set_status_text.call_args[0][0]
        assert "memory" in status_text.lower() or "pressure" in status_text.lower()


class TestMicPermissionProbe:
    """Mic probe uses AVCaptureDevice permission check, not sd.rec()."""

    def test_probe_dispatches_granted_on_authorized_status(self, main_module, monkeypatch):
        """When AVCaptureDevice says authorized, mic probe should dispatch
        micPermissionGranted_ without touching PortAudio at all."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: 3)
        # sd.rec should NOT be called — permission check is enough
        mock_sd = MagicMock()
        monkeypatch.setattr("sounddevice.rec", mock_sd)

        # Capture the selector dispatched to main thread
        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        mock_sd.assert_not_called()
        assert "micPermissionGranted:" in dispatched

    def test_probe_dispatches_denied_on_denied_status(self, main_module, monkeypatch):
        """When AVCaptureDevice says denied, probe should dispatch denial
        without attempting PortAudio recording."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: 2)

        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        assert "micPermissionDenied:" in dispatched
        assert d._mic_ready is False

    def test_probe_requests_on_not_determined(self, main_module, monkeypatch):
        """When status is not-determined, probe should request access via
        AVCaptureDevice and dispatch granted if the request succeeds."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: 0)
        monkeypatch.setattr(main_module, "_request_av_mic_access", lambda: True)

        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        assert "micPermissionGranted:" in dispatched


class TestPortAudioFallbackProbe:
    """PortAudio fallback probe fires when AVFoundation is unavailable."""

    def test_fallback_triggers_on_av_unavailable(self, main_module, monkeypatch):
        """When _get_av_auth_status returns -1, probe should fall back to
        PortAudio sd.rec() and dispatch granted on success."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: -1)
        mock_rec = MagicMock()
        monkeypatch.setattr("sounddevice.rec", mock_rec)

        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        mock_rec.assert_called_once()
        assert "micPermissionGranted:" in dispatched

    def test_fallback_dispatches_denied_on_permission_error(self, main_module, monkeypatch):
        """PortAudio fallback should dispatch denied when sd.rec() raises
        a permission-related error."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: -1)
        monkeypatch.setattr(
            "sounddevice.rec",
            MagicMock(side_effect=RuntimeError("access denied by system")),
        )

        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        assert "micPermissionDenied:" in dispatched

    def test_fallback_dispatches_failed_on_generic_error(self, main_module, monkeypatch):
        """PortAudio fallback should dispatch micProbeFailed_ when sd.rec()
        raises a non-permission error (e.g. PortAudio buffer allocation)."""
        d = _make_delegate(main_module, monkeypatch)
        d._mic_probe_in_flight = True
        d._mic_ready = False

        monkeypatch.setattr(main_module, "_get_av_auth_status", lambda: -1)
        monkeypatch.setattr(
            "sounddevice.rec",
            MagicMock(side_effect=RuntimeError("PortAudio error -9986")),
        )

        dispatched = []
        d.performSelectorOnMainThread_withObject_waitUntilDone_ = (
            lambda sel, obj, wait: dispatched.append(sel)
        )

        d._probe_mic_permission()

        assert "micProbeFailed:" in dispatched


class TestAVHelperExceptionPaths:
    """Exception paths in _get_av_auth_status and _request_av_mic_access."""

    def test_get_av_auth_status_returns_neg1_on_import_failure(self, main_module, monkeypatch):
        """If AVFoundation import fails, _get_av_auth_status should return -1."""
        def broken_import():
            raise ImportError("No module named 'AVFoundation'")

        original = main_module._get_av_auth_status

        def patched():
            # Simulate import failure inside the function
            raise ImportError("No module named 'AVFoundation'")

        # We need to patch at the point where AVFoundation is imported
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "AVFoundation":
                raise ImportError("No module named 'AVFoundation'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = main_module._get_av_auth_status()
        assert result == -1

    def test_request_av_mic_access_returns_false_on_failure(self, main_module, monkeypatch):
        """If AVFoundation import fails, _request_av_mic_access should return False."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "AVFoundation":
                raise ImportError("No module named 'AVFoundation'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        result = main_module._request_av_mic_access()
        assert result is False


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

    def test_short_shift_enter_hold_recalls_tray_not_command_overlay(
        self, main_module, monkeypatch
    ):
        """Shift+Enter short empty hold should stay on the tray path, not toggle assistant UI."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.1  # 100ms
        d._tray_stack = ["previous text"]
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        d._on_hold_end(shift_held=True, enter_held=True)

        assert d._tray_active is True
        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        d._overlay.show_tray.assert_called_once()

    def test_tray_enter_first_release_sends_current_entry_to_assistant(
        self, main_module, monkeypatch
    ):
        """A tray-visible enter-first release should send the current tray entry, not insert it."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._tray_active = True
        d._detector.tray_active = True
        d._tray_stack = ["previous text"]
        d._tray_index = 0
        d._command_client = MagicMock()
        d._send_text_as_command = MagicMock()

        d._on_hold_end(
            shift_held=False,
            enter_held=True,
        )

        d._send_text_as_command.assert_called_once_with("previous text")
        d._overlay.show_tray.assert_not_called()

    def test_short_shift_enter_hold_dismisses_visible_command_overlay(
        self, main_module, monkeypatch
    ):
        """Shift+Enter short empty hold should not dismiss a visible assistant overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.1  # 100ms
        d._tray_stack = ["previous text"]
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        d._on_hold_end(shift_held=True, enter_held=True)

        assert d._tray_active is True
        d._command_overlay.cancel_dismiss.assert_not_called()
        assert d._detector.command_overlay_active is True
        d._overlay.show_tray.assert_called_once()


class _RemovedCommandOverlayDismissRecallCycle:
    """REMOVED: Old Space+Enter chord toggle tests. Replaced by double-tap Enter.

    Kept as a dead class marker so git blame shows what was here.
    Tests in TestDoubleTapGestures (test_input_tap.py) cover the new behavior."""

    def test_enter_empty_tap_is_noop_when_overlay_not_active(self, main_module, monkeypatch):
        """An earlier Enter tap should not recall when the overlay is hidden."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False

        d._on_hold_end(shift_held=False, enter_held=False)

        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        assert d._detector.command_overlay_active is False

    def test_space_first_enter_chord_recalls_when_overlay_not_active(self, main_module, monkeypatch):
        """Space-first enter chord should recall when the overlay is hidden."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False

        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )

        d._command_overlay.show.assert_called_once()
        d._command_overlay.finish.assert_called_once()
        assert d._detector.command_overlay_active is True

    def test_space_first_enter_chord_recalls_error_snapshot_when_history_empty(
        self, main_module, monkeypatch
    ):
        """Space-first enter chord should recall the last failed assistant overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = []
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False
        d._last_command_utterance = "hello"
        d._last_command_response = "couldn't reach the model — try again in a moment"

        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )

        d._command_overlay.show.assert_called_once()
        d._command_overlay.set_utterance.assert_called_once_with("hello")
        d._command_overlay.finish.assert_called_once()
        assert d._detector.command_overlay_active is True

    def test_space_first_enter_chord_dismisses_when_overlay_active(self, main_module, monkeypatch):
        """Space-first enter chord should dismiss when command_overlay_active is True."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )

        d._command_overlay.cancel_dismiss.assert_called_once()
        assert d._detector.command_overlay_active is False

    def test_space_first_enter_chord_after_dismiss_cycle_recall_requires_toggle_route(
        self, main_module, monkeypatch
    ):
        """After dismiss, only the explicit toggle route should recall the assistant overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        # Step 1: dismiss
        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )
        assert d._detector.command_overlay_active is False
        d._command_overlay.cancel_dismiss.assert_called_once()

        # Step 2: earlier Enter tap should not recall
        d._command_overlay.reset_mock()
        d._on_hold_end(shift_held=False, enter_held=False)
        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        assert d._detector.command_overlay_active is False

        # Step 3: explicit toggle route should recall
        d._command_overlay.reset_mock()
        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )
        d._command_overlay.show.assert_called_once()
        d._command_overlay.finish.assert_called_once()
        assert d._detector.command_overlay_active is True

    def test_enter_first_empty_chord_is_noop_when_overlay_not_active(self, main_module, monkeypatch):
        """Enter-first empty chord should not recall when there is no utterance to send."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False

        d._on_hold_end(shift_held=False, enter_held=True)

        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        assert d._detector.command_overlay_active is False

    def test_empty_tap_recall_blocked_after_instant_dismiss(self, main_module, monkeypatch):
        """Plain empty tap should stay hidden if instant dismiss just fired on the same tap."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = True  # instant dismiss just fired

        d._on_hold_end(shift_held=False, enter_held=False)

        d._command_overlay.show.assert_not_called()
        assert d._detector.command_overlay_active is False

    def test_empty_tap_stays_hidden_on_fresh_tap(self, main_module, monkeypatch):
        """Plain empty tap should stay hidden when the overlay is already hidden."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = False

        d._on_hold_end(shift_held=False, enter_held=False)

        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        assert d._detector.command_overlay_active is False

    def test_hold_start_preserves_visible_overlay_active(self, main_module, monkeypatch):
        """_on_hold_start should keep assistant overlay active when it is visibly up."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = False

        d._on_hold_start()

        assert d._detector.command_overlay_active is True
        assert d._detector._command_overlay_just_dismissed is False

    def test_hold_start_keeps_dismissed_overlay_inactive(self, main_module, monkeypatch):
        """_on_hold_start preserves _just_dismissed and keeps the overlay inactive.

        _just_dismissed must survive until _on_hold_end so the empty-recording
        recall path can see it.  If hold_start cleared it, a slow dismiss tap
        (>400ms, reaching RECORDING) would recall the overlay it just dismissed.
        """
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True
        d._detector._command_overlay_just_dismissed = True

        d._on_hold_start()

        assert d._detector.command_overlay_active is False
        assert d._detector._command_overlay_just_dismissed is True

    def test_slow_dismiss_tap_does_not_recall(self, main_module, monkeypatch):
        """A dismiss tap slow enough to reach RECORDING should not recall on release.

        Sequence: instant dismiss (sets _just_dismissed=True) → hold_start
        (clears overlay_active, preserves _just_dismissed) → empty recording
        release → should NOT recall because _just_dismissed is True.
        """
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        # Step 1: instant dismiss already fired (simulated)
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = True

        # Step 2: hold_start fires (hold timer reached RECORDING)
        d._on_hold_start()

        # Step 3: empty recording release
        d._capture.stop.return_value = b""
        d._on_hold_end(shift_held=False, enter_held=False)

        # Should NOT have recalled
        d._command_overlay.show.assert_not_called()

    def test_command_utterance_ready_sets_overlay_active(self, main_module, monkeypatch):
        """commandUtteranceReady_ should set command_overlay_active = True."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock()
        d._detector.command_overlay_active = False
        d._transcription_token = 1

        d.commandUtteranceReady_({"token": 1, "utterance": "test"})

        assert d._detector.command_overlay_active is True

    def test_repeated_enter_first_empty_chords_stay_hidden_after_dismiss(
        self, main_module, monkeypatch
    ):
        """Repeated enter-first empty chords should not re-open assistant history."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        # Step 1: dismiss (active=True → False)
        d._on_hold_end(
            shift_held=False,
            enter_held=False,
            toggle_command_overlay=True,
        )
        assert d._detector.command_overlay_active is False
        d._command_overlay.cancel_dismiss.assert_called_once()

        # Step 2: earlier Enter tap stays hidden
        d._command_overlay.reset_mock()
        d._on_hold_end(shift_held=False, enter_held=False)
        assert d._detector.command_overlay_active is False
        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()

        # Step 3: still stays hidden
        d._command_overlay.reset_mock()
        d._on_hold_end(shift_held=False, enter_held=False)
        assert d._detector.command_overlay_active is False
        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()

    def test_instant_dismiss_then_hold_end_no_recall(self, main_module, monkeypatch):
        """Single-gesture simulation: instant dismiss sets _just_dismissed, then
        immediate _on_hold_end with empty audio must not recall (stutter prevention)."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        # Simulate instant dismiss on spacebar keyDown (event tap thread):
        # sets command_overlay_active=False, _just_dismissed=True
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = True

        # Then _on_hold_end fires with empty audio (plain tap, no enter)
        d._on_hold_end(shift_held=False, enter_held=False)

        # Must not recall — _just_dismissed blocks it
        d._command_overlay.show.assert_not_called()

    def test_hold_start_then_empty_recording_does_not_recall_without_overlay(
        self, main_module, monkeypatch
    ):
        """New bare-Enter empty recording should stay hidden when no overlay is up.

        _just_dismissed is cleared by the event tap on spacebar keyDown when the
        overlay is not active (the else branch).  _on_hold_start preserves it for
        the same-tap case but it should already be False for a new gesture. That
        still must not turn a bare Enter empty recording into assistant recall.
        """
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=False)

        # Simulate new gesture: event tap keyDown cleared _just_dismissed,
        # overlay was not active so it didn't dismiss.
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = False

        # hold_start preserves _just_dismissed (False in this case)
        d._on_hold_start()
        assert d._detector.command_overlay_active is False
        assert d._detector._command_overlay_just_dismissed is False

        # Now empty recording with no Enter held at release: should remain hidden
        d._on_hold_end(shift_held=False, enter_held=False)
        d._command_overlay.show.assert_not_called()
        d._command_overlay.finish.assert_not_called()
        assert d._detector.command_overlay_active is False

    def test_enter_first_empty_chord_while_overlay_active_is_noop(self, main_module, monkeypatch):
        """Enter-first empty chord should not dismiss an already visible overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._capture.stop.return_value = b""
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]
        d._command_overlay = MagicMock(_visible=True)
        d._detector.command_overlay_active = True

        # Enter-first empty recording with visible overlay → no-op
        d._on_hold_end(shift_held=False, enter_held=True)

        d._command_overlay.cancel_dismiss.assert_not_called()
        assert d._detector.command_overlay_active is True
        d._command_overlay.show.assert_not_called()


    def test_dismiss_tap_does_not_start_recording(self, main_module, monkeypatch):
        """A dismiss tap should not show glow, start capture, or set 'Recording...' status.

        When the instant dismiss fires on spacebar keyDown, the hold timer
        should be suppressed (_just_dismissed check) so _on_hold_start never
        runs and no recording side effects are visible.
        """
        d = _make_delegate(main_module, monkeypatch)
        d._command_overlay = MagicMock(_visible=True)
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "world")]

        # Simulate: instant dismiss already fired
        d._detector.command_overlay_active = False
        d._detector._command_overlay_just_dismissed = True

        # If hold_start ran, it would show glow and start capture
        d._on_hold_start()

        # Glow should NOT have been shown (no recording started)
        # But _on_hold_start always runs — the suppression is in the
        # hold timer, not in _on_hold_start itself. This test verifies
        # that even if _on_hold_start runs, the dismiss state persists
        # so _on_hold_end can suppress recall.
        d._capture.stop.return_value = b""
        d._on_hold_end(shift_held=False, enter_held=False)
        d._command_overlay.show.assert_not_called()


class TestOverlayRecallSnapshots:
    """Recall should prefer the delegate snapshot over stale ring history."""

    def test_last_command_overlay_snapshot_prefers_delegate_snapshot_over_stale_history(
        self, main_module, monkeypatch
    ):
        """Snapshot lookup should restore the last rendered transcript before ring history."""
        d = _make_delegate(main_module, monkeypatch)
        d._command_client = MagicMock()
        d._command_client.history = [("hello", "Done.")]
        d._last_command_utterance = "hello"
        d._last_command_response = "Let me check. \n[calling capture_context…]\nDone."

        assert d._last_command_overlay_snapshot() == (
            "hello",
            "Let me check. \n[calling capture_context…]\nDone.",
        )


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


class TestVADPreviewGating:
    """Test that the preview loop respects VAD silence to save compute."""

    def test_preview_loop_batch_skips_inference_on_silence(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_session_token = 1
        d._preview_active = True
        d._capture = MagicMock()
        d._capture.get_buffer.return_value = b"wav"
        d._preview_client = MagicMock()
        
        # Simulate silence
        d._is_speech = False
        d._force_preview_update = False
        
        sleep_count = [0]
        def _sleep(*args):
            sleep_count[0] += 1
            if sleep_count[0] > 1:
                d._preview_active = False

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            d._preview_loop_batch()

        # Should not have called transcribe because of silence
        d._preview_client.transcribe.assert_not_called()

    def test_preview_loop_batch_forces_update_on_silence_transition(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_session_token = 1
        d._preview_active = True
        d._capture = MagicMock()
        d._capture.get_buffer.return_value = b"wav"
        d._preview_client = MagicMock()
        
        # Simulate transition to silence (force update)
        d._is_speech = False
        d._force_preview_update = True
        
        sleep_count = [0]
        def _sleep(*args):
            sleep_count[0] += 1
            if sleep_count[0] > 1:
                d._preview_active = False

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            d._preview_loop_batch()

        # Should have called transcribe exactly once
        d._preview_client.transcribe.assert_called_once()
        assert d._force_preview_update is False

    def test_preview_loop_streaming_skips_inference_on_silence(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        d._local_mode = True
        d._preview_session_token = 1
        d._preview_active = True
        d._capture = MagicMock()
        d._capture.get_new_frames.return_value = MagicMock(size=1024)
        d._preview_client = MagicMock()
        
        d._is_speech = False
        d._force_preview_update = False
        
        sleep_count = [0]
        def _sleep(*args):
            sleep_count[0] += 1
            if sleep_count[0] > 1:
                d._preview_active = False

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            d._preview_loop_streaming()

        d._preview_client.feed.assert_not_called()


class TestSegmentAccumulator:
    """Test the SegmentAccumulator for opportunistic segment transcription."""

    def test_empty_accumulator(self, main_module, monkeypatch):
        acc = main_module.SegmentAccumulator()
        assert acc.count == 0
        assert acc.text == ""
        assert not acc.has_results

    def test_dispatch_accumulates_results(self, main_module, monkeypatch):
        acc = main_module.SegmentAccumulator()
        client = MagicMock()
        client.transcribe.side_effect = ["hello", "world"]

        acc.dispatch(b"seg1", client)
        acc.dispatch(b"seg2", client)
        acc.wait(timeout=5.0)

        assert acc.count == 2
        assert acc.text == "hello world"
        assert acc.has_results

    def test_dispatch_handles_failure_gracefully(self, main_module, monkeypatch):
        acc = main_module.SegmentAccumulator()
        client = MagicMock()
        client.transcribe.side_effect = [Exception("server down"), "world"]

        acc.dispatch(b"seg1", client)
        acc.dispatch(b"seg2", client)
        acc.wait(timeout=5.0)

        assert acc.count == 2
        assert acc.text == "world"  # first segment failed, second succeeded

    def test_reset_clears_state(self, main_module, monkeypatch):
        acc = main_module.SegmentAccumulator()
        client = MagicMock()
        client.transcribe.return_value = "hello"

        acc.dispatch(b"seg1", client)
        acc.wait(timeout=5.0)
        assert acc.count == 1

        acc.reset()
        assert acc.count == 0
        assert acc.text == ""


class TestSegmentAcceleratedTranscription:
    """Test the segment-accelerated final transcription path."""

    def test_transcribe_worker_uses_segments_when_available(self, main_module, monkeypatch):
        """When segments are cached, _transcribe_worker should use them + tail."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "sidecar"
        d._transcribe_start = time.monotonic()

        # Pre-populate the segment accumulator with cached results.
        acc = main_module.SegmentAccumulator()
        d._segment_accumulator = acc
        # Simulate two already-completed segments.
        d._client.transcribe.side_effect = ["seg one", "seg two", "tail text"]
        acc.dispatch(b"s1", d._client)
        acc.dispatch(b"s2", d._client)
        acc.wait(timeout=5.0)

        # Simulate pre-stop tail capture (done by _on_hold_end before stop()).
        d._pre_stop_tail_wav = b"tail_wav"
        d._pre_stop_segment_count = acc.count  # 2 segments, no final flush
        d._client.transcribe.side_effect = ["tail text"]

        d._transcribe_worker(b"full_wav_unused", token=1)

        # The final client.transcribe call should be for the tail only.
        last_call = d._client.transcribe.call_args
        assert last_call[0][0] == b"tail_wav"

        # Result should be segments + tail joined.
        result_call = d.performSelectorOnMainThread_withObject_waitUntilDone_
        payload = result_call.call_args[0][1]
        assert "seg one" in payload["text"]
        assert "seg two" in payload["text"]
        assert "tail text" in payload["text"]

    def test_transcribe_worker_falls_back_without_segments(self, main_module, monkeypatch):
        """Without segments, _transcribe_worker should use full buffer."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._transcribe_start = time.monotonic()
        d._segment_accumulator = main_module.SegmentAccumulator()  # empty

        d._client.transcribe.return_value = "full buffer text"

        d._transcribe_worker(b"full_wav", token=1)

        d._client.transcribe.assert_called_once_with(b"full_wav")

    def test_transcribe_worker_retries_local_whisper_after_initial_failure(
        self, main_module, monkeypatch
    ):
        """Local Whisper finalization should retry from cached audio with bounded settings."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"
        d._transcribe_start = time.monotonic()
        d._segment_accumulator = main_module.SegmentAccumulator()
        d._client = main_module.LocalTranscriptionClient(
            model="mlx-community/whisper-large-v3-turbo",
            decode_timeout=30.0,
            eager_eval=False,
        )
        monkeypatch.setattr(main_module, "supports_eager_eval", lambda: True)
        attempts: list[tuple[float | None, bool]] = []

        def fake_transcribe(self, wav_bytes):
            attempts.append((self._decode_timeout, self._eager_eval))
            if len(attempts) == 1:
                raise TimeoutError("decode timed out")
            return "recovered text"

        monkeypatch.setattr(
            main_module.LocalTranscriptionClient,
            "transcribe",
            fake_transcribe,
            raising=False,
        )

        d._transcribe_worker(b"full_wav", token=1)

        payload = d.performSelectorOnMainThread_withObject_waitUntilDone_.call_args[0][1]
        assert payload["text"] == "recovered text"
        assert attempts == [(8.0, False), (8.0, True)]

    def test_hold_start_wires_segment_callback_for_sidecar(self, main_module, monkeypatch):
        """_on_hold_start should wire segment_callback when backend is sidecar."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "sidecar"

        d._on_hold_start()

        # capture.start should have been called with a segment_callback.
        call_kwargs = d._capture.start.call_args[1]
        assert call_kwargs.get("segment_callback") is not None

    def test_hold_start_no_segment_callback_for_local(self, main_module, monkeypatch):
        """_on_hold_start should not wire segment_callback for local backend."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "local"

        d._on_hold_start()

        call_kwargs = d._capture.start.call_args[1]
        assert call_kwargs.get("segment_callback") is None

    def test_preview_batch_uses_cached_segments_plus_tail(self, main_module, monkeypatch):
        """Preview loop should use cached segment text + tail when segments exist."""
        d = _make_delegate(main_module, monkeypatch)
        d._whisper_backend = "sidecar"
        d._preview_active = True
        d._is_speech = True
        d._force_preview_update = False
        d._preview_done = threading.Event()

        # Pre-populate segment accumulator.
        acc = main_module.SegmentAccumulator()
        client = MagicMock()
        client.transcribe.return_value = "cached segment"
        acc.dispatch(b"s1", client)
        acc.wait(timeout=5.0)
        d._segment_accumulator = acc

        d._capture.get_tail_buffer.return_value = b"tail_wav"
        d._preview_client.transcribe.return_value = "tail preview"

        call_count = [0]
        def _sleep(*args):
            call_count[0] += 1
            if call_count[0] > 1:
                d._preview_active = False

        with patch.object(main_module.time, "sleep", side_effect=_sleep):
            d._preview_loop_batch()

        # Preview client should have been called with tail, not full buffer.
        d._preview_client.transcribe.assert_called_with(b"tail_wav")
        # Full buffer should NOT have been requested.
        d._capture.get_buffer.assert_not_called()
