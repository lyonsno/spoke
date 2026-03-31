"""Contract tests for overlay timing constants."""

import colorsys
import importlib
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

            _, _, _, applied_alpha = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args[0]
            assert applied_alpha <= mod._TEXT_ALPHA_MAX
            assert applied_alpha == pytest.approx(mod._TEXT_ALPHA_MAX)
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

    def test_ridge_glow_amplitude_drives_ridge_layer(
        self, mock_pyobjc
    ):
        """Ridge layer should respond to glow amplitude with smoothing."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._visible = True
            overlay._ridge_layer = MagicMock()
            overlay._smoothed_glow_opacity = 0.0

            # Feed steady low signal to let the independent smoothing converge
            for _ in range(30):
                overlay.update_glow_amplitude(0.1)
            ridge_opacity = overlay._ridge_layer.setOpacity_.call_args[0][0]
            # Smoothed value converges near 0.1; ridge scales by 0.15
            assert ridge_opacity == pytest.approx(0.015, abs=0.005)

            overlay._ridge_layer.reset_mock()

            # Feed high signal
            for _ in range(30):
                overlay.update_glow_amplitude(1.0)
            ridge_at_peak = overlay._ridge_layer.setOpacity_.call_args[0][0]
            # Ridge capped at 0.20
            assert ridge_at_peak <= 0.20
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


class TestAdaptiveOverlayCompositing:
    """Overlay bg/text cross-fades between dark and light with brightness."""

    def _make_overlay(self, mod):
        overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
        overlay._visible = True
        overlay._text_view = MagicMock()
        overlay._text_amplitude = 0.0
        overlay._content_view = MagicMock()
        overlay._ridge_layer = MagicMock()
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

    def test_dark_background_uses_dark_bg_light_text(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            text_color_args = None
            bg_color_args = None
            for call in color_calls:
                r, g, b, _ = call[0]
                if r == pytest.approx(1.0) and g == pytest.approx(1.0) and b == pytest.approx(1.0):
                    text_color_args = call[0]
                if r < 0.2 and g < 0.2 and b < 0.2:
                    bg_color_args = call[0]
            assert text_color_args is not None
            assert bg_color_args is not None
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_text_becomes_transparent_cutout(self, mock_pyobjc):
        """On bright backgrounds, text alpha approaches zero (cutout effect)."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            text_r, text_g, text_b, text_alpha = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0]
            # Text color is dark (near black) but alpha is near zero — cutout
            assert text_r < 0.3 and text_g < 0.3 and text_b < 0.3
            assert text_alpha < 0.05  # approaching transparent
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_fill_is_opaque(self, mock_pyobjc):
        """On bright backgrounds, the dark fill becomes near-opaque to support the cutout."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            # The fill stays dark in both modes — find the bg color call
            # (dark color with high alpha on light backgrounds)
            calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            # Second call is the background (first is text)
            bg_r, bg_g, bg_b, bg_alpha = calls[-1][0]
            assert bg_r < 0.3  # fill stays dark on light backgrounds
            assert bg_alpha > 0.8  # near-opaque fill
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_preview_text_reaches_true_black(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            text_r, text_g, text_b, _ = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[0][0]
            assert text_r == pytest.approx(0.0)
            assert text_g == pytest.approx(0.0)
            assert text_b == pytest.approx(0.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_mid_brightness_crossover_bump_increases_fill_opacity(self, mock_pyobjc):
        """At the crossover point, the fill gets more opaque to maintain contrast."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)

            # Measure fill alpha at brightness 0 (dark, no crossover)
            overlay.set_brightness(0.0, immediate=True)
            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)
            _, _, _, alpha_dark = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[-1][0]

            # Measure fill alpha at brightness 0.35 (crossover center)
            overlay.set_brightness(0.35, immediate=True)
            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)
            _, _, _, alpha_crossover = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list[-1][0]

            # Crossover should be more opaque than dark due to the bump
            assert alpha_crossover > alpha_dark + 0.1
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

    def test_brightness_cross_fade_is_continuous(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            def get_bg_rgb(brightness):
                overlay = self._make_overlay(mod)
                overlay.set_brightness(brightness, immediate=True)
                mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
                overlay.update_text_amplitude(10.0)
                for call in mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list:
                    r, g, b, _ = call[0]
                    return (r, g, b)
                return None

            c1 = get_bg_rgb(0.4)
            c2 = get_bg_rgb(0.5)
            c3 = get_bg_rgb(0.6)

            assert c1 is not None and c2 is not None and c3 is not None
            for i in range(3):
                assert abs(c2[i] - c1[i]) < 0.25
                assert abs(c3[i] - c2[i]) < 0.25
        finally:
            sys.modules.pop("spoke.overlay", None)
