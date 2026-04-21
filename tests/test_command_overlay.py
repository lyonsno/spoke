"""Tests for CommandOverlay state machines and timer logic.

Tests cover: thinking timer lifecycle, dismiss animation phases,
show/finish/hide state transitions, and timer cancellation.
All tests use mocked PyObjC — no GUI runtime required.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )

def _make_overlay(mock_pyobjc):
    """Create a CommandOverlay with mocked internals."""
    sys.modules.pop("spoke.command_overlay", None)
    mod = importlib.import_module("spoke.command_overlay")
    overlay = mod.CommandOverlay.__new__(mod.CommandOverlay)
    overlay._window = MagicMock()
    overlay._window.alphaValue.return_value = 1.0
    overlay._wrapper_view = MagicMock()
    overlay._wrapper_view.layer.return_value = MagicMock()
    overlay._content_view = MagicMock()
    overlay._content_view.layer.return_value = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._text_view = MagicMock()
    overlay._text_view.textStorage.return_value = MagicMock()
    overlay._text_view.textStorage.return_value.length.return_value = 0
    overlay._thinking_label = MagicMock()
    overlay._thinking_label.isHidden.return_value = False
    overlay._narrator_label = MagicMock()
    overlay._screen = MagicMock()
    overlay._screen.frame.return_value = _make_rect(0.0, 0.0, 1920.0, 1080.0)
    overlay._visible = False
    overlay._streaming = False
    overlay._response_text = ""
    overlay._utterance_text = ""
    overlay._fade_timer = None
    overlay._pulse_timer = None
    overlay._linger_timer = None
    overlay._thinking_timer = None
    overlay._cancel_timer_anim = None
    overlay._thinking_seconds = 0.0
    overlay._thinking_inverted = False
    overlay._fade_step = 0
    overlay._fade_from = 0.0
    overlay._fade_direction = 0
    overlay._pulse_phase_asst = 0.0
    overlay._pulse_phase_user = 0.0
    overlay._color_phase = 0.0
    overlay._color_velocity_phase = 0.0
    overlay._tts_amplitude = 0.0
    overlay._tts_active = False
    overlay._tts_blend = 0.0
    overlay._tool_mode = False
    overlay._brightness = 0.0
    overlay._brightness_target = 0.0
    overlay._fill_layer = MagicMock()
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.capture_blurred_image.return_value = None
    overlay._backdrop_blur_radius_points = 9.0
    overlay._backdrop_capture_overscan_points = 42.519685
    overlay._backdrop_capture_rect = None
    overlay._backdrop_capture_pixel_size = None
    overlay._backdrop_timer = None
    overlay._cancel_spring = 0.0
    overlay._cancel_spring_target = 0.0
    overlay._cancel_spring_fired = False
    overlay._cancel_step = 0
    overlay._cancel_phase = ""
    return overlay, mod


class _FakeLayoutManager:
    def __init__(self, height):
        self.height = height

    def ensureLayoutForTextContainer_(self, container):
        self._container = container

    def usedRectForTextContainer_(self, container):
        return _make_rect(0.0, 0.0, 0.0, self.height)


class TestThinkingTimer:
    """Test the thinking timer state machine."""

    def test_start_sets_initial_state(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()

        assert overlay._thinking_seconds == 0.0
        assert overlay._thinking_inverted is False
        assert overlay._thinking_timer is not None
        overlay._thinking_label.setHidden_.assert_called_with(False)
        overlay._thinking_label.setStringValue_.assert_called_with("0.0s")

    def test_tick_increments_time(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()

        for _ in range(10):
            overlay.thinkingTick_(None)

        assert abs(overlay._thinking_seconds - 1.0) < 0.01
        overlay._thinking_label.setStringValue_.assert_called_with("1.0s")

    def test_invert_sets_flag(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()
        assert overlay._thinking_inverted is False

        overlay.invert_thinking_timer()
        assert overlay._thinking_inverted is True

    def test_stop_clears_timer_and_hides_label(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()
        assert overlay._thinking_timer is not None

        overlay._stop_thinking_timer()
        assert overlay._thinking_timer is None
        overlay._thinking_label.setHidden_.assert_called_with(True)

    def test_stop_is_safe_when_no_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        assert overlay._thinking_timer is None
        overlay._stop_thinking_timer()  # should not raise

    def test_init_uses_shared_backdrop_renderer_factory(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        sentinel = object()
        factory = MagicMock(return_value=sentinel)
        monkeypatch.setattr(mod, "make_backdrop_renderer", factory)

        overlay = mod.CommandOverlay.alloc().initWithScreen_(MagicMock())

        assert overlay._backdrop_renderer is sentinel
        factory.assert_called_once()

    def test_install_backdrop_frame_callback_pushes_live_frames_into_layer(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_layer = MagicMock()

        overlay._install_backdrop_frame_callback()

        callback = overlay._backdrop_renderer.set_frame_callback.call_args[0][0]
        callback("live-frame")

        overlay._backdrop_layer.setContents_.assert_called_once_with("live-frame")

    def test_install_backdrop_frame_callback_skips_image_path_for_sample_buffer_layer(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_layer_is_sample_buffer_display = True

        overlay._install_backdrop_frame_callback()

        overlay._backdrop_renderer.set_frame_callback.assert_called_once_with(None)

    def test_install_backdrop_frame_callback_installs_when_optical_shell_enabled(
        self, mock_pyobjc, monkeypatch
    ):
        """When optical shell is enabled, frame callback is always installed
        because SCK optical shell routes through CGImage path."""
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_layer = MagicMock()

        overlay._install_backdrop_frame_callback()

        args = overlay._backdrop_renderer.set_frame_callback.call_args[0]
        assert args[0] is not None

    def test_install_backdrop_sample_buffer_callback_enqueues_live_samples(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_layer = MagicMock()

        overlay._install_backdrop_sample_buffer_callback()

        callback = overlay._backdrop_renderer.set_sample_buffer_callback.call_args[0][0]
        callback("live-sample")

        overlay._backdrop_layer.enqueueSampleBuffer_.assert_called_once_with("live-sample")

    def test_choose_backdrop_layer_uses_display_layer_when_renderer_supports_sample_buffers(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        sentinel_layer_class = MagicMock()
        monkeypatch.setattr(mod, "_backdrop_display_layer_class", lambda: sentinel_layer_class)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_renderer.supports_sample_buffer_presentation.return_value = True
        overlay._backdrop_blur_radius_points = 5.4

        assert overlay._choose_backdrop_layer_class() is sentinel_layer_class


class TestDismissAnimation:
    """Test the pop-then-shrink dismiss animation state machine."""

    def test_cancel_dismiss_initializes_grow_phase(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True

        overlay.cancel_dismiss()

        assert overlay._cancel_elapsed == pytest.approx(0.0)
        assert overlay._cancel_phase == "grow"
        assert overlay._streaming is False
        overlay._window.setAlphaValue_.assert_called_with(1.0)

    def test_grow_phase_expands_overlay_for_first_60ms(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        phase, scale, alpha, done = mod._dismiss_animation_state(0.05)

        assert phase == "grow"
        assert scale > 1.0
        assert alpha == pytest.approx(1.0)
        assert done is False

    def test_shrink_phase_starts_after_60ms(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        phase, scale, alpha, done = mod._dismiss_animation_state(0.12)

        assert phase == "shrink"
        assert scale < mod._DISMISS_GROW_SCALE
        assert alpha < 1.0
        assert done is False

    def test_animation_completes_after_200ms(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        for _ in range(12):
            overlay._cancelAnimStep_(None)

        assert overlay._visible is False
        overlay._window.orderOut_.assert_called()
        overlay._wrapper_view.layer.return_value.setValue_forKeyPath_.assert_called_with(
            1.0, "transform.scale"
        )

    def test_cancel_dismiss_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.cancel_dismiss()  # should not raise

    def test_cancel_all_timers_clears_dismiss_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.cancel_dismiss()

        overlay._cancel_all_timers()

        assert overlay._cancel_timer_anim is None


class TestShowFinishHide:
    """Test overlay lifecycle state transitions."""

    def test_show_sets_visible_and_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()

        assert overlay._visible is True
        assert overlay._streaming is True
        assert overlay._response_text == ""

    def test_show_starts_thinking_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()

        assert overlay._thinking_timer is not None
        assert overlay._thinking_seconds == 0.0

    def test_show_clears_stale_backdrop_before_reuse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.capture_blurred_image.return_value = None

        overlay.show()

        overlay._backdrop_layer.setContents_.assert_called_with(None)
        overlay._backdrop_layer.setMask_.assert_called_with(None)

    def test_show_flushes_sample_buffer_backdrop_before_reuse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.capture_blurred_image.return_value = None
        overlay._backdrop_layer.flushAndRemoveImage = MagicMock()
        overlay._backdrop_layer_is_sample_buffer_display = True

        overlay.show()

        overlay._backdrop_layer.flushAndRemoveImage.assert_called_once_with()

    def test_apply_backdrop_pulse_style_updates_quantized_blur_mask_and_opacity(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 160.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(1.0)

        assert overlay._backdrop_blur_radius_points < 5.4
        assert overlay._backdrop_mask_width_multiplier < 9.0
        overlay._backdrop_renderer.set_live_blur_radius_points.assert_called_once_with(
            overlay._backdrop_blur_radius_points
        )
        overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)
        overlay._backdrop_layer.setOpacity_.assert_called_with(1.0)

    def test_apply_backdrop_pulse_style_keeps_backdrop_opaque_in_airy_phase(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 160.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(0.0)

        overlay._backdrop_layer.setOpacity_.assert_called_with(1.0)

    def test_apply_backdrop_pulse_style_pushes_optical_shell_config_when_enabled(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_renderer.set_live_optical_shell_config = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 160.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(1.0)

        config = overlay._backdrop_renderer.set_live_optical_shell_config.call_args[0][0]
        assert config["enabled"] is True
        # Warp capsule inflated by half-radius (_OVERLAY_HEIGHT / 4)
        capsule_r = mod._OVERLAY_HEIGHT / 4.0
        assert config["content_width_points"] == pytest.approx(mod._OVERLAY_WIDTH + capsule_r)
        assert config["content_height_points"] == pytest.approx(mod._OVERLAY_HEIGHT + capsule_r)
        assert config["ring_amplitude_points"] == pytest.approx(
            mod._COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS
        )
        assert config["tail_amplitude_points"] == pytest.approx(
            mod._COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS
        )
        assert config["cleanup_blur_radius_points"] == pytest.approx(
            mod._COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS
        )

    def test_command_overlay_size_can_be_overridden_for_smoke_surface(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_WIDTH", "1200.0")
        monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_HEIGHT", "160.0")
        monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_CORNER_RADIUS", "32.0")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_renderer.set_live_optical_shell_config = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 1280.0, 240.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(1.0)

        config = overlay._backdrop_renderer.set_live_optical_shell_config.call_args[0][0]
        assert mod._OVERLAY_WIDTH == pytest.approx(1200.0)
        assert mod._OVERLAY_HEIGHT == pytest.approx(160.0)
        assert mod._OVERLAY_CORNER_RADIUS == pytest.approx(32.0)
        # Warp capsule is inflated by half-radius (_OVERLAY_HEIGHT / 4 = 40)
        assert config["content_width_points"] == pytest.approx(1200.0 + 40.0)
        assert config["content_height_points"] == pytest.approx(160.0 + 40.0)
        assert config["corner_radius_points"] == pytest.approx(40.0)  # _OVERLAY_HEIGHT / 4

    def test_apply_backdrop_pulse_style_uses_current_content_height_for_optical_shell(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._content_view.frame.return_value = _make_rect(0.0, 0.0, mod._OVERLAY_WIDTH, 196.0)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_renderer.set_live_optical_shell_config = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 196.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(1.0)

        config = overlay._backdrop_renderer.set_live_optical_shell_config.call_args[0][0]
        # Warp capsule inflated by half-radius (_OVERLAY_HEIGHT / 4 = 20)
        assert config["content_height_points"] == pytest.approx(196.0 + 20.0)

    def test_show_starts_low_rate_backdrop_refresh_timer(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)

        backdrop_timer = object()
        def _timer(*args):
            selector = args[2]
            if selector == "backdropRefreshTick:":
                return backdrop_timer
            return object()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.side_effect = _timer

        overlay.show()

        assert overlay._backdrop_timer is backdrop_timer

    def test_hide_teardown_stops_live_backdrop_stream_when_fade_finishes(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.stop_live_stream = MagicMock()
        overlay._fade_direction = -1
        overlay._fade_step = mod._FADE_STEPS - 1
        overlay._fade_from = 1.0

        overlay.fadeStep_(None)

        overlay._backdrop_renderer.stop_live_stream.assert_called_once_with()

    def test_show_skips_backdrop_refresh_timer_in_optical_shell_debug_visualization(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        overlay, mod = _make_overlay(mock_pyobjc)

        overlay.show()

        selectors = [
            call_args[0][2]
            for call_args in mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args_list
        ]
        assert "backdropRefreshTick:" not in selectors, (
            "Optical-shell debug visualization should seed a static diagnostic field once, "
            "not rebuild it at the normal live-backdrop refresh cadence."
        )
        assert overlay._backdrop_timer is None

    def test_debug_visualization_uses_plain_calayer_backdrop(self, mock_pyobjc, monkeypatch):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        sentinel_layer_class = MagicMock()
        monkeypatch.setattr(mod, "_backdrop_display_layer_class", lambda: sentinel_layer_class)
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_renderer.supports_sample_buffer_presentation.return_value = True

        layer_class = overlay._choose_backdrop_layer_class()

        assert layer_class is mod.CALayer, (
            "Debug visualization should force a plain CALayer so direct diagnostic images "
            "are actually visible instead of disappearing into the sample-buffer presenter."
        )

    def test_backdrop_refresh_timer_is_added_to_scroll_surviving_run_loop_modes(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        run_loop = MagicMock()
        mod.NSRunLoop.currentRunLoop.return_value = run_loop
        timer = object()
        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer

        overlay._start_backdrop_refresh_timer()

        run_loop.addTimer_forMode_.assert_any_call(timer, mod._RUN_LOOP_COMMON_MODE)
        run_loop.addTimer_forMode_.assert_any_call(timer, mod._EVENT_TRACKING_RUN_LOOP_MODE)

    def test_show_can_resume_thinking_timer_without_resetting_elapsed_state(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._thinking_seconds = 4.2
        overlay._thinking_inverted = True

        overlay.show(preserve_thinking_timer=True)

        assert overlay._thinking_timer is not None
        assert overlay._thinking_seconds == 4.2
        assert overlay._thinking_inverted is True
        overlay._thinking_label.setStringValue_.assert_called_with("4.2s")

    def test_finish_clears_streaming_and_stops_thinking(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()
        assert overlay._streaming is True

        overlay.finish()
        assert overlay._streaming is False
        assert overlay._thinking_timer is None


class TestWindowLayering:
    """Command overlay should stack independently from the preview overlay."""

    def test_setup_places_command_overlay_above_preview_overlay_level(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        sys.modules.pop("spoke.command_overlay", None)
        overlay_mod = importlib.import_module("spoke.overlay")
        command_mod = importlib.import_module("spoke.command_overlay")

        assert command_mod._COMMAND_OVERLAY_WINDOW_LEVEL == overlay_mod._OVERLAY_WINDOW_LEVEL + 1

    def test_hide_clears_visible_and_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True

        overlay.hide()
        assert overlay._visible is False
        assert overlay._streaming is False

    def test_show_resets_text(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._response_text = "old response"
        overlay._utterance_text = "old utterance"

        overlay.show()
        assert overlay._response_text == ""
        assert overlay._utterance_text == ""

    def test_show_clears_attributed_text_storage_before_reuse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.show()

        overlay._text_view.textStorage().setAttributedString_.assert_called_once()

    def test_show_rebuilds_default_fill_geometry_before_reuse(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._fill_image_brightness = 0.0
        overlay._apply_ridge_masks = MagicMock()

        overlay.show()

        overlay._apply_ridge_masks.assert_called_once_with(
            mod._OVERLAY_WIDTH,
            mod._OVERLAY_HEIGHT,
        )

    def test_show_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.show()
        assert overlay._visible is False

    def test_set_response_text_replaces_existing_response(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = ""
        overlay._response_text = "Let me check."

        def _append(text):
            overlay._response_text += text

        overlay.append_token = MagicMock(side_effect=_append)

        overlay.set_response_text("Done.")

        assert overlay._response_text == "Done."
        # New path uses setAttributedString_ to rebuild in one shot, not setString_("")
        overlay._text_view.textStorage().setAttributedString_.assert_called_once()
        overlay.append_token.assert_called_once_with("Done.")

    def test_set_response_text_with_utterance_calls_layout_once(self, mock_pyobjc):
        """set_response_text must not trigger an intermediate layout with only the
        utterance text — that shrinks the window before growing it, causing visible flicker."""
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "What is the capital of France?"
        overlay._response_text = "Paris."

        layout_calls = []
        overlay._update_layout = MagicMock(side_effect=lambda: layout_calls.append(1))

        overlay.set_response_text("Paris is the capital of France.")

        assert len(layout_calls) == 1, (
            f"set_response_text called _update_layout {len(layout_calls)} time(s); "
            "expected exactly 1 — intermediate calls shrink the window causing flicker"
        )

    def test_hide_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.hide()  # should not raise


class TestTimerCancellation:
    """Test timer cleanup methods."""

    def test_cancel_all_timers_clears_everything(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        # Start all timers
        overlay.show()
        assert overlay._fade_timer is not None
        assert overlay._pulse_timer is not None
        assert overlay._thinking_timer is not None

        overlay._cancel_all_timers()
        assert overlay._fade_timer is None
        assert overlay._pulse_timer is None
        assert overlay._linger_timer is None
        assert overlay._thinking_timer is None
        assert overlay._backdrop_timer is None

    def test_cancel_fade_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._fade_timer = None
        overlay._cancel_fade()  # should not raise

    def test_cancel_pulse_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._pulse_timer = None
        overlay._cancel_pulse()  # should not raise

    def test_cancel_linger_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._linger_timer = None
        overlay._cancel_linger()  # should not raise


class TestLingerDone:
    """Test the linger completion callback."""

    def test_linger_done_hides_when_visible_and_not_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = False

        overlay.lingerDone_(None)
        # hide() sets _visible = False
        assert overlay._visible is False

    def test_linger_done_noop_when_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True

        overlay.lingerDone_(None)
        assert overlay._visible is True  # still visible — streaming

    def test_linger_done_noop_when_not_visible(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = False
        overlay._streaming = False

        overlay.lingerDone_(None)
        assert overlay._visible is False

class TestAdaptiveCompositing:
    """Test brightness-adaptive command overlay styling."""

    def test_set_brightness_immediate_snaps(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.set_brightness(0.8, immediate=True)

        assert overlay._brightness == pytest.approx(0.8)
        assert overlay._brightness_target == pytest.approx(0.8)

    def test_show_keeps_content_background_clear_after_brightness_snap(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay.set_brightness(1.0, immediate=True)

        overlay._content_view.layer.return_value.setBackgroundColor_.reset_mock()
        overlay.show()

        overlay._content_view.layer.return_value.setBackgroundColor_.assert_called_with(None)

    def test_background_fill_tracks_preview_overlay_chrome(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        sys.modules.pop("spoke.overlay", None)
        command_mod = importlib.import_module("spoke.command_overlay")
        overlay_mod = importlib.import_module("spoke.overlay")
        try:
            assert command_mod._background_color_for_brightness(0.0) == pytest.approx(
                overlay_mod._BG_COLOR_DARK
            )
            assert command_mod._background_color_for_brightness(1.0) == pytest.approx(
                overlay_mod._BG_COLOR_LIGHT
            )
        finally:
            sys.modules.pop("spoke.command_overlay", None)
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_fill_uses_additive_experiment(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._fill_compositing_filter_for_brightness(0.0) == "plusL"
            assert mod._fill_compositing_filter_for_brightness(1.0) is None
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_apply_surface_theme_updates_fill_compositing_filter(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._content_view.frame.return_value = _make_rect(0.0, 0.0, 600.0, 80.0)

        overlay._brightness = 0.0
        overlay._apply_surface_theme()
        overlay._fill_layer.setCompositingFilter_.assert_called_with("plusL")

        overlay._fill_layer.setCompositingFilter_.reset_mock()
        overlay._brightness = 1.0
        overlay._fill_image_brightness = -1.0
        overlay._apply_surface_theme()
        overlay._fill_layer.setCompositingFilter_.assert_called_with(None)

    def test_assistant_text_alpha_floor_and_ceiling_are_punchier(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._assistant_text_alpha_for_breath(0.0) == pytest.approx(1.0)
            assert mod._assistant_text_alpha_for_breath(1.0) == pytest.approx(1.0)
            assert mod.CommandOverlay._TTS_ALPHA_MIN == pytest.approx(1.0)
            assert mod.CommandOverlay._TTS_ALPHA_MAX == pytest.approx(1.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_response_color_darkens_for_bright_backgrounds(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._response_color_for_brightness((0.82, 0.66, 0.94), 0.0)
            light = mod._response_color_for_brightness((0.82, 0.66, 0.94), 1.0)

            assert sum(light) < sum(dark)
            assert light[2] == max(light)
            assert max(light) < 0.12
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_user_text_turns_dark_on_light_backgrounds(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._user_text_color_for_brightness(0.0)
            light = mod._user_text_color_for_brightness(1.0)

            # User text is now dark gray on both backgrounds
            assert max(dark) < 0.25
            assert max(light) < 0.17
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_peak_assistant_breath_drives_fill_into_preview_like_opacity_band(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
        assert fill_opacity >= 0.8, (
            "Assistant overlay should enter the same materially opaque band "
            "the preview overlay reaches when its amplitude is high."
        )

    def test_optical_shell_peak_assistant_breath_keeps_fill_light_enough_to_show_backdrop(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
        assert fill_opacity <= 0.45, (
            "Optical-shell mode should stop burying the backdrop under a near-solid fill body."
        )

    def test_shared_compositor_mode_softens_command_fill_without_hiding_it(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._text_view.textStorage.return_value.length.return_value = 0
        overlay._fullscreen_compositor = SimpleNamespace(
            active_client_count=2,
            update_shell_config_key=MagicMock(),
            refresh_brightness=MagicMock(),
            sampled_brightness=1.0,
        )

        overlay._pulseStepInner()

        fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
        assert fill_opacity <= 0.60, (
            "Shared fullscreen-compositor mode should ease the assistant fill enough "
            "to reveal warp under overlap instead of holding the near-solid solo band."
        )
        assert fill_opacity >= 0.42, (
            "Shared fullscreen-compositor mode should still keep the assistant body "
            "materially present instead of collapsing into an almost invisible fill."
        )

    def test_optical_shell_softens_cancel_spring_tint_so_shell_remains_visible(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        import Quartz

        monkeypatch.setattr(Quartz, "CGColorCreateSRGB", lambda *args: "cg-color", raising=False)
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._cancel_spring = 1.0
        overlay._cancel_spring_target = 1.0
        overlay._spring_tint_layer = MagicMock()
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        spring_opacity = overlay._spring_tint_layer.setOpacity_.call_args[0][0]
        assert spring_opacity <= 0.18, (
            "Optical-shell mode should keep the spring tint from washing out the shell effect."
        )

    def test_optical_shell_reveal_mode_hollows_out_fill_for_debugging(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL", "1")
        import Quartz

        monkeypatch.setattr(Quartz, "CGColorCreateSRGB", lambda *args: "cg-color", raising=False)
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._cancel_spring = 1.0
        overlay._cancel_spring_target = 1.0
        overlay._spring_tint_layer = MagicMock()
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
        spring_opacity = overlay._spring_tint_layer.setOpacity_.call_args[0][0]
        assert fill_opacity <= 0.02, (
            "Reveal mode should hollow out the fill so the shell field can be judged directly."
        )
        assert spring_opacity <= 0.001, (
            "Reveal mode should nearly eliminate the spring tint while inspecting the shell."
        )

    def test_apply_backdrop_pulse_style_debug_visualization_hardens_backdrop_mask(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_renderer.set_live_optical_shell_config = MagicMock()
        overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 160.0)
        overlay._update_backdrop_mask = MagicMock()
        overlay._backdrop_base_blur_radius_points = 5.4
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_base_mask_width_multiplier = 9.0
        overlay._backdrop_mask_width_multiplier = 9.0

        overlay._apply_backdrop_pulse_style(1.0)

        assert overlay._backdrop_mask_width_multiplier <= 0.25, (
            "Debug visualization should use a near-hard backdrop mask so the field "
            "isn't buried under a second soft falloff."
        )
        overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)


class TestBackdropGeometry:
    """Backdrop blur capture should use a bounded overscan, not the full SDF feather."""

    def test_backdrop_refresh_default_targets_live_scroll_cadence(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._COMMAND_BACKDROP_REFRESH_S == pytest.approx(1.0 / 30.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_mask_falloff_width_uses_configured_multiplier(self, mock_pyobjc, monkeypatch):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER", "9.0")
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._command_backdrop_mask_falloff_width(2.0) == pytest.approx(18.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_pulse_style_quantizes_midrange_breath_into_stable_tier(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            style_a = mod._command_backdrop_pulse_style(5.4, 9.0, 0.41)
            style_b = mod._command_backdrop_pulse_style(5.4, 9.0, 0.49)

            assert style_a == pytest.approx(style_b)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_attack_release_envelope_rises_faster_than_it_falls(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            rising = mod._advance_attack_release(0.2, 1.0, attack=0.4, release=0.1)
            falling = mod._advance_attack_release(0.8, 0.0, attack=0.4, release=0.1)

            assert rising - 0.2 > 0.8 - falling
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_pulse_style_expands_blur_and_mask_at_peak_breath(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            low = mod._command_backdrop_pulse_style(5.4, 9.0, 0.0)
            high = mod._command_backdrop_pulse_style(5.4, 9.0, 1.0)

            assert high[0] > low[0]
            assert high[1] > low[1]
            assert high[2] == pytest.approx(1.0)
            assert low[2] == pytest.approx(1.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_overscan_default_tracks_centimeter_budget(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._command_backdrop_capture_overscan_points() == pytest.approx(
                42.519685, abs=1e-6
            )
            assert mod._command_backdrop_capture_overscan_pixels(2.0) == pytest.approx(
                85.03937, abs=1e-5
            )
            assert mod._command_backdrop_capture_overscan_points() < mod._OUTER_FEATHER
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_backdrop_capture_rect_expands_content_frame_by_overscan(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            screen_frame = _make_rect(0.0, 0.0, 1920.0, 1080.0)
            win_frame = _make_rect(300.0, 80.0, 1040.0, 520.0)
            content_frame = _make_rect(220.0, 220.0, 600.0, 80.0)

            rect = mod._backdrop_capture_rect(
                screen_frame,
                win_frame,
                content_frame,
                overscan_points=40.0,
            )

            assert rect.origin.x == pytest.approx(480.0)
            assert rect.origin.y == pytest.approx(660.0)
            assert rect.size.width == pytest.approx(680.0)
            assert rect.size.height == pytest.approx(160.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_update_backdrop_capture_geometry_records_point_and_pixel_bounds(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
        overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
        overlay._ridge_scale = 2.0

        overlay._update_backdrop_capture_geometry()

        rect = overlay._backdrop_capture_rect
        px_w, px_h = overlay._backdrop_capture_pixel_size
        assert rect.origin.y == pytest.approx(657.480315, abs=1e-6)
        assert rect.size.width > 600.0
        assert rect.size.width < 720.0
        assert px_w == pytest.approx(rect.size.width * 2.0)
        assert px_h == pytest.approx(rect.size.height * 2.0)


class TestBackdropRefresh:
    def test_quartz_backdrop_renderer_seeds_debug_grid_when_visualize_enabled(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        renderer = mod._QuartzBackdropRenderer()
        context = MagicMock()
        context.createCGImage_fromRect_.return_value = "grid-image"
        renderer._context = MagicMock(return_value=context)
        monkeypatch.setattr(
            mod,
            "_debug_shell_grid_ci_image",
            MagicMock(return_value=SimpleNamespace(extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0))),
        )
        mock_pyobjc["Quartz"].CGWindowListCreateImage = MagicMock(
            side_effect=AssertionError("Debug visualize should bypass live Quartz capture")
        )

        image = renderer.capture_blurred_image(
            window_number=17,
            capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
            blur_radius_points=5.4,
        )

        assert image == "grid-image"

    def test_quartz_backdrop_renderer_debug_grid_runs_through_warp_kernel(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", "1")
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        renderer = mod._QuartzBackdropRenderer()
        context = MagicMock()
        context.createCGImage_fromRect_.return_value = "warped-grid-image"
        renderer._context = MagicMock(return_value=context)
        helper = MagicMock(return_value=SimpleNamespace(extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0)))
        monkeypatch.setattr(mod, "_debug_shell_grid_ci_image", helper)
        warp_helper = MagicMock(
            return_value=SimpleNamespace(extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0))
        )
        monkeypatch.setattr(mod, "_apply_optical_shell_warp_ci_image", warp_helper)
        mock_pyobjc["Quartz"].CGWindowListCreateImage = MagicMock(
            side_effect=AssertionError("Debug visualize should bypass live Quartz capture")
        )

        image = renderer.capture_blurred_image(
            window_number=17,
            capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
            blur_radius_points=5.4,
        )

        assert image == "warped-grid-image"
        helper.assert_called_once()
        warp_helper.assert_called_once()

    def test_refresh_backdrop_snapshot_updates_contents_frame_and_mask(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
        overlay._window.windowNumber.return_value = 17
        overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
        overlay._ridge_scale = 2.0
        overlay._backdrop_capture_overscan_points = 40.0
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_renderer.capture_blurred_image.return_value = "blurred-image"
        overlay._update_backdrop_mask = MagicMock()

        overlay._refresh_backdrop_snapshot()

        overlay._backdrop_renderer.capture_blurred_image.assert_called_once()
        call = overlay._backdrop_renderer.capture_blurred_image.call_args.kwargs
        assert call["window_number"] == 17
        assert call["blur_radius_points"] == pytest.approx(9.0)
        assert call["capture_rect"].origin.y == pytest.approx(660.0)
        assert call["capture_rect"].size.width == pytest.approx(680.0)
        assert call["capture_rect"].size.height == pytest.approx(160.0)
        overlay._backdrop_layer.setFrame_.assert_called_with(((180.0, 180.0), (680.0, 160.0)))
        overlay._backdrop_layer.setContents_.assert_called_with("blurred-image")
        overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)

    def test_refresh_backdrop_snapshot_is_noop_when_renderer_returns_none(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
        overlay._window.windowNumber.return_value = 17
        overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
        overlay._ridge_scale = 2.0
        overlay._backdrop_capture_overscan_points = 40.0
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_renderer.capture_blurred_image.return_value = None
        overlay._update_backdrop_mask = MagicMock()

        overlay._refresh_backdrop_snapshot()

        overlay._backdrop_layer.setContents_.assert_not_called()
        overlay._update_backdrop_mask.assert_not_called()

    def test_refresh_backdrop_snapshot_skips_image_seed_for_blurred_sample_buffer_path(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
        overlay._window.windowNumber.return_value = 17
        overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
        overlay._ridge_scale = 2.0
        overlay._backdrop_capture_overscan_points = 40.0
        overlay._backdrop_blur_radius_points = 5.4
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_renderer = MagicMock()
        overlay._backdrop_renderer.capture_blurred_image.return_value = None
        overlay._backdrop_renderer.uses_direct_sample_buffers.return_value = True
        overlay._update_backdrop_mask = MagicMock()

        overlay._refresh_backdrop_snapshot()

        overlay._backdrop_renderer.uses_direct_sample_buffers.assert_called_once_with(5.4)
        overlay._backdrop_layer.setFrame_.assert_called_with(((180.0, 180.0), (680.0, 160.0)))
        overlay._backdrop_layer.setContents_.assert_not_called()
        overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)

    def test_update_backdrop_mask_reuses_cached_mask_when_signature_is_unchanged(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_capture_overscan_points = 40.0
        overlay._ridge_scale = 2.0
        overlay._backdrop_mask_width_multiplier = 3.0
        overlay._backdrop_mask_signature = None
        overlay._backdrop_mask_layer = None

        import spoke.overlay as overlay_mod

        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", False)
        monkeypatch.setattr(mod, "_backdrop_mask_alpha", MagicMock(return_value="alpha-field"))
        monkeypatch.setattr(overlay_mod, "_overlay_rounded_rect_sdf", MagicMock(return_value="sdf-field"))
        monkeypatch.setattr(
            overlay_mod,
            "_fill_field_to_image",
            MagicMock(return_value=("mask-image", "mask-payload")),
        )

        overlay._update_backdrop_mask(680.0, 160.0)
        overlay._update_backdrop_mask(680.0, 160.0)

        overlay_mod._overlay_rounded_rect_sdf.assert_called_once()
        overlay_mod._fill_field_to_image.assert_called_once()

    def test_update_backdrop_mask_rebuilds_when_mask_signature_changes(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_layer = MagicMock()
        overlay._backdrop_capture_overscan_points = 40.0
        overlay._ridge_scale = 2.0
        overlay._backdrop_mask_width_multiplier = 3.0
        overlay._backdrop_mask_signature = None
        overlay._backdrop_mask_layer = None

        import spoke.overlay as overlay_mod

        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", False)
        monkeypatch.setattr(mod, "_backdrop_mask_alpha", MagicMock(return_value="alpha-field"))
        monkeypatch.setattr(overlay_mod, "_overlay_rounded_rect_sdf", MagicMock(return_value="sdf-field"))
        monkeypatch.setattr(
            overlay_mod,
            "_fill_field_to_image",
            MagicMock(return_value=("mask-image", "mask-payload")),
        )

        overlay._update_backdrop_mask(680.0, 160.0)
        overlay._backdrop_mask_width_multiplier = 4.0
        overlay._update_backdrop_mask(680.0, 160.0)

        assert overlay_mod._fill_field_to_image.call_count == 2


class TestGeometryCaps:
    def test_update_layout_can_grow_assistant_overlay_near_notch(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(1000.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj

        overlay._update_layout()

        frame = overlay._window.setFrame_display_animate_.call_args[0][0]
        expected_height = 640.0
        assert frame.size.height == pytest.approx(expected_height + 2 * mod._OUTER_FEATHER)
        assert overlay._content_view.setFrame_.call_args[0][0].size.height == pytest.approx(expected_height)

    def test_update_layout_rebuilds_fill_geometry_when_assistant_overlay_grows(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(280.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj
        overlay._apply_ridge_masks = MagicMock()

        overlay._update_layout()

        overlay._apply_ridge_masks.assert_called_once_with(
            mod._OVERLAY_WIDTH,
            pytest.approx(304.0),
        )


class TestToolState:
    """Test the tool execution visual state machine."""

    def test_set_tool_active_shows_label(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._thinking_label.setHidden_.reset_mock()
        
        overlay.set_tool_active(True)
        
        assert overlay._tool_mode is True
        overlay._thinking_label.setHidden_.assert_called_with(False)
        overlay._thinking_label.setStringValue_.assert_called_with("tool…")

    def test_thinking_tick_shows_tool_in_tool_mode(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.set_tool_active(True)
        overlay._thinking_label.setStringValue_.reset_mock()
        
        overlay.thinkingTick_(None)
        
        overlay._thinking_label.setStringValue_.assert_called_with("tool…")

    def test_set_tool_active_false_preserves_mode_until_tick(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.set_tool_active(True)
        overlay.set_tool_active(False)
        
        assert overlay._tool_mode is False
        overlay.thinkingTick_(None)
        # Now it should show seconds again
        assert "s" in overlay._thinking_label.setStringValue_.call_args[0][0]


class TestSDFCaching:
    """SDF recomputation is skipped when geometry hasn't changed."""

    def test_same_dimensions_reuses_cached_sdf(self, mock_pyobjc, monkeypatch):
        """Brightness-only changes should not recompute the SDF."""
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._spring_tint_layer = None

        import spoke.overlay as ov_mod
        call_count = 0
        original = ov_mod._overlay_rounded_rect_sdf

        def counting_sdf(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", counting_sdf)

        # First call — SDF must be computed
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        # Second call, same dimensions — SDF should be cached
        overlay._brightness = 0.5
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1, "SDF was recomputed despite identical geometry"

    def test_changed_height_recomputes_sdf(self, mock_pyobjc, monkeypatch):
        """A height change must recompute the SDF."""
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._spring_tint_layer = None

        import spoke.overlay as ov_mod
        call_count = 0
        original = ov_mod._overlay_rounded_rect_sdf

        def counting_sdf(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", counting_sdf)

        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        overlay._apply_ridge_masks(600.0, 200.0)
        assert call_count == 2, "SDF was not recomputed after height change"
