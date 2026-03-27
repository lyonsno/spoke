"""Tests for SpokeAppDelegate orchestration and error paths.

Tests the wiring between layers: hold callbacks, transcription lifecycle,
generation-based stale result rejection, and env var validation.
"""

import os
import time
from unittest.mock import MagicMock, patch


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

    def test_model_allowed_blocks_large_on_low_ram(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
        assert d._model_allowed("mlx-community/whisper-medium.en-mlx-8bit") is True
        assert d._model_allowed("Qwen/Qwen3-ASR-0.6B") is True
        assert d._model_allowed("mlx-community/whisper-large-v3-turbo") is False

    def test_model_allowed_permits_large_on_high_ram(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 36.0)
        assert d._model_allowed("mlx-community/whisper-large-v3-turbo") is True

    def test_select_model_none_returns_list(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
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

    def test_select_model_none_exposes_bf16_laptop_tiers_with_labels(
        self, main_module, monkeypatch
    ):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 16.0)
        models = d._select_model(None)

        labels_by_id = {model_id: label for model_id, label, _enabled in models}

        assert labels_by_id["mlx-community/whisper-tiny.en-mlx"] == "Tiny.en (bf16)"
        assert labels_by_id["mlx-community/whisper-base.en-mlx"] == "Base.en (bf16)"
        assert labels_by_id["mlx-community/whisper-small.en-mlx"] == "Small.en (bf16)"

    def test_select_model_none_includes_large_on_high_ram(self, main_module, monkeypatch):
        d = _make_delegate(main_module, monkeypatch)
        monkeypatch.setattr(main_module, "_RAM_GB", 36.0)
        models = d._select_model(None)
        model_ids = [m[0] for m in models]
        assert "mlx-community/whisper-large-v3-turbo" in model_ids


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

    def test_init_accepts_new_bf16_whisper_model_ids(
        self, main_module, monkeypatch
    ):
        """New tiny/base/small bf16 IDs should flow through the normal local init path."""
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


class TestWarmupContract:
    """Test explicit warm-before-ready behavior."""

    def test_setup_event_tap_prepares_clients_before_ready(
        self, main_module, monkeypatch
    ):
        """The app should warm selected clients before advertising readiness."""
        d = _make_delegate(main_module, monkeypatch)
        d._detector.install.return_value = True
        d._prepare_clients = MagicMock()

        d._setup_event_tap()

        d._prepare_clients.assert_called_once_with()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")

    def test_setup_event_tap_surfaces_prepare_failure(
        self, main_module, monkeypatch
    ):
        """Warmup failures should block ready-state and surface a model error."""
        d = _make_delegate(main_module, monkeypatch)
        d._detector.install.return_value = True
        d._prepare_clients = MagicMock(side_effect=RuntimeError("warm failed"))
        d._show_model_load_alert = MagicMock()

        d._setup_event_tap()

        d._show_model_load_alert.assert_called_once()
        d._menubar.set_status_text.assert_called_with(
            "Model load failed — choose another model"
        )

    def test_prepare_failure_still_allows_model_selection_recovery(
        self, main_module, monkeypatch
    ):
        """A failed warmup should still leave model selection available for recovery."""
        d = _make_delegate(main_module, monkeypatch)
        d._detector.install.return_value = True
        d._prepare_clients = MagicMock(side_effect=RuntimeError("warm failed"))
        d._show_model_load_alert = MagicMock()
        monkeypatch.setenv(
            "SPOKE_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
        )

        with patch.object(main_module.os, "execv") as mock_execv:
            d._setup_event_tap()
            d._select_model("Qwen/Qwen3-ASR-0.6B")

        mock_execv.assert_called_once()


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


class TestResultInjection:
    """Test timing of the post-injection overlay cleanup."""

    def test_inject_result_text_hides_overlay_soon_after_injection(
        self, main_module, monkeypatch
    ):
        """Final text should not leave the overlay hanging around after injection."""
        d = _make_delegate(main_module, monkeypatch)

        with patch.object(main_module, "inject_text"):
            with patch("Foundation.NSTimer") as MockTimer:
                d._inject_result_text("hello", "Ready")

        MockTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        assert (
            MockTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args.args[0]
            == 0.12
        )
