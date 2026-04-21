"""Contract tests for overlay timing constants."""

import colorsys
import importlib
import sys
from unittest.mock import MagicMock

import pytest


class TestOverlayTiming:
    """Keep the overlay tuned to the current fast-handoff UX."""

    def test_preview_backdrop_refresh_default_targets_live_scroll_cadence(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._PREVIEW_BACKDROP_REFRESH_S == pytest.approx(1.0 / 30.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_text_alpha_ceiling_stays_below_full_white(self, mock_pyobjc):
        """Text should stay legible without ever reaching fully opaque white."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._TEXT_ALPHA_MAX == pytest.approx(0.75)

            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._visible = True
            overlay._text_view = MagicMock()
            overlay._text_amplitude = 0.0
            overlay._backdrop_renderer = MagicMock()
            overlay._backdrop_base_blur_radius_points = mod._PREVIEW_BACKDROP_BLUR_RADIUS

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()

            overlay.update_text_amplitude(10.0)

            _, _, _, applied_alpha = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args[0]
            # Text is anchored at a fixed high alpha, not driven by amplitude
            assert applied_alpha == pytest.approx(0.88)
            overlay._backdrop_renderer.set_live_blur_radius_points.assert_called_once()
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_fade_out_lingers_longer_through_fast_finalization(self, mock_pyobjc):
        """Fade-out should linger a bit longer now that final injection lands quickly."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._FADE_OUT_S == 0.315
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_typewriter_starts_with_faster_interval_for_quick_preview_updates(
        self, mock_pyobjc
    ):
        """Typewriter pacing should keep up with the faster preview cadence."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._visible = True
            overlay._text_view = MagicMock()
            overlay._typewriter_target = ""
            overlay._typewriter_displayed = ""
            overlay._typewriter_hwm = 0
            overlay._typewriter_timer = None
            overlay._update_layout = MagicMock()
            overlay._cancel_typewriter = MagicMock()

            timer = object()
            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer

            overlay.set_text("abc")

            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once_with(
                0.02, overlay, "typewriterStep:", None, True
            )
            assert overlay._typewriter_timer is timer
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_preview_backdrop_refresh_timer_is_added_to_scroll_surviving_run_loop_modes(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._backdrop_timer = None
            overlay._backdrop_renderer = MagicMock()
            overlay._backdrop_layer = MagicMock()
            overlay._cancel_backdrop_refresh = MagicMock()

            timer = object()
            run_loop = MagicMock()
            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer
            mod.NSRunLoop.currentRunLoop.return_value = run_loop

            overlay._start_backdrop_refresh_timer()

            run_loop.addTimer_forMode_.assert_any_call(timer, mod._RUN_LOOP_COMMON_MODE)
            run_loop.addTimer_forMode_.assert_any_call(timer, mod._EVENT_TRACKING_RUN_LOOP_MODE)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_glow_amplitude_smoothing_converges(
        self, mock_pyobjc
    ):
        """Glow amplitude smoothing should converge toward the input signal."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._visible = True
            overlay._smoothed_glow_opacity = 0.0

            for _ in range(30):
                overlay.update_glow_amplitude(0.5)
            # Smoothing should converge near the input
            assert overlay._smoothed_glow_opacity == pytest.approx(0.5, abs=0.05)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_overlay_glow_color_is_desaturated_tint(self, mock_pyobjc):
        """The overlay glow should be a subtle tint, not a saturated neon outline."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            base_sat = colorsys.rgb_to_hsv(0.38, 0.52, 1.0)[1]
            overlay_sat = colorsys.rgb_to_hsv(*mod._GLOW_COLOR)[1]

            # ~10% of the base saturation
            assert overlay_sat == pytest.approx(base_sat * 0.13, rel=0.05)
            assert overlay_sat < 0.15
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_overlay_glow_layers_shift_saturation_from_inner_to_outer(self, mock_pyobjc):
        """The inner overlay glow should calm down while the wide outer glow gets more saturated."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            inner_color, middle_color, outer_color = mod._overlay_layer_colors(mod._GLOW_COLOR)
            base_sat = colorsys.rgb_to_hsv(*mod._GLOW_COLOR)[1]
            inner_sat = colorsys.rgb_to_hsv(*inner_color)[1]
            middle_sat = colorsys.rgb_to_hsv(*middle_color)[1]
            outer_sat = colorsys.rgb_to_hsv(*outer_color)[1]

            assert inner_sat == pytest.approx(base_sat * 0.7, rel=0.02)
            assert middle_sat == pytest.approx(base_sat, rel=0.02)
            assert outer_sat == pytest.approx(min(base_sat * 1.8, 1.0), rel=0.02)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_flash_tray_capture_schedules_ack_then_fade(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._window = MagicMock()
            overlay._tray_capture_flash_timer = None
            overlay.show_tray = MagicMock()
            overlay._pulse_tray_capture_ack = MagicMock()
            overlay._cancel_tray_capture_flash = MagicMock()

            timer = object()
            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer

            overlay.flash_tray_capture("saved text")

            overlay.show_tray.assert_called_once_with("saved text", owner="user")
            overlay._pulse_tray_capture_ack.assert_called_once_with()
            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once_with(
                mod._TRAY_CAPTURE_FLASH_ONSET_S,
                overlay,
                "trayCaptureFlashDone:",
                None,
                False,
            )
            assert overlay._tray_capture_flash_timer is timer
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_flash_tray_capture_callback_fades_overlay_out(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._tray_capture_flash_timer = object()
            overlay.hide = MagicMock()

            overlay.trayCaptureFlashDone_(None)

            assert overlay._tray_capture_flash_timer is None
            overlay.hide.assert_called_once_with(
                fade_duration=mod._TRAY_CAPTURE_FLASH_FADE_OUT_S
            )
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_preview_optical_shell_config_keeps_assistant_like_envelope_gap(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            cfg = mod._preview_optical_shell_config(600.0, 80.0)
            capsule_r = mod._OVERLAY_HEIGHT / 4.0
            assert cfg["content_width_points"] == pytest.approx(600.0 + 2.0 * capsule_r)
            assert cfg["content_height_points"] == pytest.approx(80.0 + 2.0 * capsule_r)
            assert cfg["bleed_zone_frac"] == pytest.approx(0.8)
            assert cfg["exterior_mix_width_points"] == pytest.approx(20.0)
            assert cfg["corner_radius_points"] == pytest.approx(mod._OVERLAY_CORNER_RADIUS)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_preview_optical_shell_config_supports_independent_x_y_inflation(self, mock_pyobjc, monkeypatch):
        monkeypatch.setenv("SPOKE_PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII", "1.5")
        monkeypatch.setenv("SPOKE_PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII", "0.75")
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            cfg = mod._preview_optical_shell_config(600.0, 80.0)
            capsule_r = mod._OVERLAY_HEIGHT / 4.0
            assert cfg["content_width_points"] == pytest.approx(600.0 + 1.5 * capsule_r)
            assert cfg["content_height_points"] == pytest.approx(80.0 + 0.75 * capsule_r)
        finally:
            sys.modules.pop("spoke.overlay", None)


class TestAdaptiveOverlayCompositing:
    """Overlay bg/text cross-fades between dark and light with brightness."""

    def _make_overlay(self, mod):
        overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
        overlay._visible = True
        overlay._text_view = MagicMock()
        overlay._text_amplitude = 0.0
        overlay._content_view = MagicMock()
        overlay._fill_layer = MagicMock()
        overlay._brightness = 0.0
        overlay._brightness_target = 0.0
        return overlay

    def test_set_brightness_immediate_snaps(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.7, immediate=True)
            assert overlay._brightness == pytest.approx(0.7)
            assert overlay._brightness_target == pytest.approx(0.7)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_preview_optical_shell_fill_rasters_at_backing_scale(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._fill_layer = MagicMock()
            overlay._brightness = 0.0
            overlay._ridge_scale = 2.0
            overlay._fullscreen_compositor = object()

            captured = {}

            def fake_fill_field_to_image(alpha, r, g, b):
                captured["shape"] = alpha.shape
                return "fill-image", b"payload"

            monkeypatch.setattr(mod, "_fill_field_to_image", fake_fill_field_to_image)

            overlay._apply_ridge_masks(600.0, 80.0)

            feather = mod._OPTICAL_SHELL_FEATHER
            expected_shape = (
                int((80.0 + 2 * feather) * overlay._ridge_scale),
                int((600.0 + 2 * feather) * overlay._ridge_scale),
            )
            assert captured["shape"] == expected_shape
            overlay._fill_layer.setContentsScale_.assert_called_once_with(overlay._ridge_scale)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_fill_uses_additive_experiment(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._fill_compositing_filter_for_brightness(0.0) == "plusL"
            assert mod._fill_compositing_filter_for_brightness(1.0) is None
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_set_brightness_without_immediate_chases_target(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.8)
            assert overlay._brightness_target == pytest.approx(0.8)
            assert overlay._brightness == pytest.approx(0.0)

            for _ in range(30):
                overlay.update_text_amplitude(0.0)
            assert overlay._brightness > 0.6

            for _ in range(100):
                overlay.update_text_amplitude(0.0)
            assert overlay._brightness == pytest.approx(0.8, abs=0.01)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_uses_light_text_and_sets_fill_opacity(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            # Text should be dark (dark text on light fill for dark backgrounds)
            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            text_color_args = None
            for call in color_calls:
                r, g, b, _ = call[0]
                if r < 0.1 and g < 0.1 and b < 0.1:
                    text_color_args = call[0]
            assert text_color_args is not None

            # Fill layer opacity should be set
            assert overlay._fill_layer.setOpacity_.called
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_text_is_white_on_dark_fill(self, mock_pyobjc):
        """On bright backgrounds, text is white against the dark fill."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            text_r, text_g, text_b, text_alpha = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0]
            # White text on dark fill for light backgrounds
            assert text_r > 0.7 and text_g > 0.7 and text_b > 0.7
            assert text_alpha > 0.5
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_is_opaque(self, mock_pyobjc):
        """On bright backgrounds, the fill layer becomes near-opaque to support the cutout."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._fill_layer.reset_mock()
            overlay.update_text_amplitude(10.0)

            # Fill layer opacity should be high on light backgrounds
            fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
            assert fill_opacity > 0.8  # near-opaque fill
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_preview_text_reaches_true_white(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            text_r, text_g, text_b, _ = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0]
            assert text_r == pytest.approx(1.0)
            assert text_g == pytest.approx(1.0)
            assert text_b == pytest.approx(1.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_fill_opacity_responds_to_amplitude(self, mock_pyobjc):
        """The SDF fill should breathe with amplitude — low at silence, high when speaking."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            # Low amplitude
            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(0.0)
            alpha_silent = overlay._fill_layer.setOpacity_.call_args[0][0]

            # High amplitude
            overlay._fill_layer.reset_mock()
            overlay.update_text_amplitude(10.0)
            alpha_loud = overlay._fill_layer.setOpacity_.call_args[0][0]

            assert alpha_loud > alpha_silent + 0.3
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_default_brightness_is_dark(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            assert overlay._brightness == pytest.approx(0.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_text_color_contrasts_with_fill(self, mock_pyobjc):
        """Text should be dark on light fill (dark bg) and white on dark fill (light bg)."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            # Dark background → light fill → dark text
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0, immediate=True)
            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)
            text_r = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0][0]
            assert text_r < 0.1  # dark text

            # Light background → dark fill → white text (ease-out snap converges)
            overlay.set_brightness(1.0, immediate=True)
            for _ in range(20):
                mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
                overlay.update_text_amplitude(10.0)
            text_r = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0][0]
            assert text_r > 0.9  # white text
        finally:
            sys.modules.pop("spoke.overlay", None)
