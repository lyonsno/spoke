"""Contract tests for overlay timing constants."""

import colorsys
import importlib
import math
import sys
from unittest.mock import MagicMock

import pytest


class TestOverlayTiming:
    """Keep the overlay tuned to the current fast-handoff UX."""

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

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()

            overlay.update_text_amplitude(10.0)

            overlay._text_view.setAlphaValue_.assert_called_once()
            applied_alpha = overlay._text_view.setAlphaValue_.call_args[0][0]
            # Text alpha now rides the view/layer seam rather than forcing a payload rebuild.
            assert applied_alpha == pytest.approx(0.88)
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

    def test_light_background_does_not_draw_rectangular_backstop(self, mock_pyobjc):
        """Bright backgrounds should rely on the SDF fill, not a rectangular fallback box."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._content_view.layer.return_value.setBackgroundColor_.reset_mock()
            overlay.update_text_amplitude(10.0)

            overlay._content_view.layer.return_value.setBackgroundColor_.assert_called_once_with(None)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_preview_fill_toggle_still_clears_any_backstop(self, mock_pyobjc):
        """Turning Preview Fill off should keep the content background clear as well."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            class _State:
                def is_visible(self, layer_id):
                    return False

            overlay = self._make_overlay(mod)
            overlay._visual_layer_state = _State()

            overlay._content_view.layer.return_value.setBackgroundColor_.reset_mock()
            overlay._apply_direct_fill_backstop(1.0, mod._BG_COLOR_LIGHT, 0.95)

            overlay._content_view.layer.return_value.setBackgroundColor_.assert_called_once_with(None)
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

    def test_dark_background_text_punches_through_fill(self, mock_pyobjc):
        """On dark backgrounds, preview text should act as a cutout through the light fill."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0, immediate=True)

            overlay.update_text_amplitude(10.0)

            overlay._text_view.layer.return_value.setCompositingFilter_.assert_called_with("destinationOut")
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_text_stays_normal_instead_of_cutting_out(self, mock_pyobjc):
        """On bright backgrounds, preview text should render normally rather than punching holes in the dark fill."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay.update_text_amplitude(10.0)

            overlay._text_view.layer.return_value.setCompositingFilter_.assert_called_with(None)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_text_snap_speed_is_faster_for_smokeability(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._TEXT_SNAP_SPEED == pytest.approx(0.35)
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
            assert text_alpha > 0.95
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_is_opaque(self, mock_pyobjc):
        """On bright backgrounds, the fill peak should get much darker without raising the floor again."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._fill_layer.reset_mock()
            overlay.update_text_amplitude(10.0)

            # Fill layer opacity should be high on light backgrounds
            fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
            assert 0.95 < fill_opacity < 0.995
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_is_violently_dark_now(self, mock_pyobjc):
        """Bright backgrounds should drive the preview fill substantially darker than before."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            r, g, b = mod._BG_COLOR_LIGHT
            assert r < 0.05
            assert g < 0.05
            assert b < 0.06
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_silence_floor_is_heavier_for_regression_triage(self, mock_pyobjc):
        """Even at silence, bright backgrounds should keep a visible dark fill without becoming too constant."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(0.0)

            fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
            assert 0.4 < fill_opacity < 0.6
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_keeps_headroom_for_amplitude(self, mock_pyobjc):
        """Bright-screen dark fill should leave room to breathe instead of starting pinned near 1.0."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(0.0)
            alpha_silent = overlay._fill_layer.setOpacity_.call_args[0][0]

            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(10.0)
            alpha_loud = overlay._fill_layer.setOpacity_.call_args[0][0]

            assert alpha_silent < 0.8
            assert alpha_loud > alpha_silent + 0.18
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_profile_pushes_more_signal_into_the_sdf(self, mock_pyobjc):
        """Bright-screen dark fill should get much darker at the top end from the SDF itself."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            width, interior_floor, opacity_min, opacity_max = mod._fill_profile_for_brightness(1.0)

            assert width == pytest.approx(14.5)
            assert interior_floor == pytest.approx(0.9997)
            assert opacity_min == pytest.approx(0.48)
            assert opacity_max == pytest.approx(0.98)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_fill_profile_gets_more_signal_too(self, mock_pyobjc):
        """Dark-background white fill should also get a stronger source shape, not just the dark-language variant."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            width, interior_floor, opacity_min, opacity_max = mod._fill_profile_for_brightness(0.0)

            assert width == pytest.approx(3.6)
            assert interior_floor == pytest.approx(0.72)
            assert opacity_min == pytest.approx(0.06)
            assert opacity_max == pytest.approx(0.98)
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

    def test_light_background_fill_responds_more_aggressively_to_mid_rms(self, mock_pyobjc):
        """The bright-screen fill should visibly move at mid-level RMS instead of feeling stuck."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(0.0)
            alpha_silent = overlay._fill_layer.setOpacity_.call_args[0][0]

            overlay._fill_layer.reset_mock()
            overlay._text_amplitude = 0.0
            overlay.update_text_amplitude(mod._TEXT_AMP_SATURATION * 0.5)
            alpha_mid = overlay._fill_layer.setOpacity_.call_args[0][0]

            assert alpha_mid > alpha_silent + 0.025
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

    def test_light_background_text_snap_is_faster_for_smokeability(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._TEXT_SNAP_SPEED == pytest.approx(0.35)
        finally:
            sys.modules.pop("spoke.overlay", None)


class TestPreviewTextPayloadInvalidation:
    """Preview text should only rebuild its CPU payload when the seam actually changes."""

    def _make_overlay(self, mod):
        overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
        overlay._visible = True
        overlay._text_view = MagicMock()
        overlay._text_view.layer.return_value = MagicMock()
        overlay._scroll_view = MagicMock()
        overlay._content_view = MagicMock()
        overlay._fill_layer = MagicMock()
        overlay._brightness = 0.0
        overlay._brightness_target = 0.0
        overlay._text_amplitude = 0.0
        overlay._typewriter_displayed = "Epistaxis"
        overlay._typewriter_target = "Epistaxis"
        overlay._set_text_view_content = MagicMock()
        return overlay

    def test_animation_ticks_reuse_text_payload_when_palette_stays_stable(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            overlay.update_text_amplitude(0.0)
            overlay.update_text_amplitude(0.0)

            assert overlay._set_text_view_content.call_count == 1
        finally:
            sys.modules.pop("spoke.overlay", None)


class TestMetalPreviewSurfaceSync:
    """The Metal preview surface should only resnapshot when the payload actually changes."""

    def _make_overlay(self, mod):
        overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
        overlay._visible = True
        overlay._text_view = MagicMock()
        overlay._text_view.layer.return_value = MagicMock()
        overlay._scroll_view = MagicMock()
        overlay._content_view = MagicMock()
        overlay._fill_layer = MagicMock()
        overlay._brightness = 0.0
        overlay._brightness_target = 0.0
        overlay._text_amplitude = 0.0
        overlay._typewriter_displayed = "Epistaxis"
        overlay._typewriter_target = "Epistaxis"
        overlay._set_text_view_content = MagicMock()
        overlay._sync_preview_surface = MagicMock()
        overlay._metal_preview_renderer = MagicMock()
        overlay._metal_preview_active = True
        return overlay

    def test_animation_ticks_do_not_resnapshot_metal_preview_when_payload_is_stable(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            overlay.update_text_amplitude(0.0)
            overlay.update_text_amplitude(0.0)

            assert overlay._sync_preview_surface.call_count == 1
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_palette_boundary_resnapshots_metal_preview(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            overlay.update_text_amplitude(0.0)
            overlay.set_brightness(1.0, immediate=True)
            overlay.update_text_amplitude(0.0)

            assert overlay._sync_preview_surface.call_count == 2
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_palette_boundary_still_rebuilds_text_payload(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            overlay.update_text_amplitude(0.0)
            overlay.set_brightness(1.0, immediate=True)
            overlay.update_text_amplitude(0.0)

            assert overlay._set_text_view_content.call_count == 2
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_pending_metal_handoff_keeps_appkit_preview_visible(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._scroll_view = MagicMock()
            overlay._text_view = MagicMock()
            overlay._metal_preview_renderer = MagicMock()
            overlay._metal_preview_active = True
            overlay._metal_preview_ready = False

            overlay._apply_preview_surface_alpha(0.88)

            overlay._scroll_view.setAlphaValue_.assert_called_with(1.0)
            overlay._text_view.setAlphaValue_.assert_called_with(0.88)
            overlay._metal_preview_renderer.set_opacity.assert_called_with(0.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_first_metal_snapshot_forces_scroll_view_opaque_then_restores_hidden(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._screen = MagicMock()
            overlay._screen.backingScaleFactor.return_value = 2.0
            overlay._scroll_view = MagicMock()
            overlay._text_view = MagicMock()
            overlay._preview_surface_alpha = 0.88
            overlay._text_payload_signature = ("Epistaxis", "light-text", None)
            overlay._preview_surface_signature = None
            overlay._metal_preview_renderer = MagicMock()
            overlay._metal_preview_renderer.update_cgimage.return_value = True
            overlay._metal_preview_active = True
            overlay._metal_preview_ready = False
            overlay._brightness = 0.0

            bounds = MagicMock()
            frame = MagicMock()
            frame_size = MagicMock()
            frame_size.width = 200.0
            frame_size.height = 60.0
            frame.size = frame_size
            overlay._scroll_view.bounds.return_value = bounds
            overlay._scroll_view.frame.return_value = frame
            overlay._scroll_view.alphaValue.return_value = 0.0
            rep = MagicMock()
            rep.CGImage.return_value = "cgimage"
            overlay._scroll_view.bitmapImageRepForCachingDisplayInRect_.return_value = rep

            assert overlay._sync_preview_surface(force=True) is True

            assert overlay._metal_preview_ready is True
            assert overlay._scroll_view.setAlphaValue_.call_args_list[0][0][0] == 1.0
            assert overlay._scroll_view.setAlphaValue_.call_args_list[-1][0][0] == 0.0
            overlay._metal_preview_renderer.set_hidden.assert_called_with(False)
            overlay._metal_preview_renderer.set_opacity.assert_called_with(0.88)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_typewriter_steps_resnapshot_metal_preview_surface(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay._typewriter_displayed = "Epi"
            overlay._typewriter_target = "Epistaxis"
            overlay._typewriter_hwm = len(overlay._typewriter_displayed)
            overlay._update_layout = MagicMock(
                side_effect=lambda: overlay._sync_preview_surface(force=True)
            )

            overlay.typewriterStep_(None)

            overlay._sync_preview_surface.assert_called_once_with(force=True)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_snap_text_update_resnapshots_metal_preview_surface(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay._typewriter_displayed = "Hello brave"
            overlay._typewriter_target = overlay._typewriter_displayed
            overlay._typewriter_hwm = len(overlay._typewriter_displayed)
            overlay._cancel_typewriter = MagicMock()
            overlay._update_layout = MagicMock(
                side_effect=lambda: overlay._sync_preview_surface(force=True)
            )

            overlay.set_text("Completely different partial")

            overlay._sync_preview_surface.assert_called_once_with(force=True)
        finally:
            sys.modules.pop("spoke.overlay", None)
