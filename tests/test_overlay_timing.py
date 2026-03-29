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

    def test_outer_glow_peak_is_softened_without_changing_low_level_response(
        self, mock_pyobjc
    ):
        """Low glow levels should stay intact while the peak outer bloom is capped."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
            overlay._visible = True
            overlay._inner_shadow = MagicMock()
            overlay._outer_glow_tight = MagicMock()
            overlay._outer_glow_wide = MagicMock()
            overlay._smoothed_glow_opacity = 0.0

            # Drive smoothing to convergence at 0.1
            for _ in range(80):
                overlay.update_glow_amplitude(0.1)
            inner_op = overlay._inner_shadow.setShadowOpacity_.call_args[0][0]
            tight_op = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            wide_op = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert inner_op == pytest.approx(0.1 * 1.4, rel=0.02)
            assert tight_op == pytest.approx(0.1 * 0.7, rel=0.02)
            assert wide_op == pytest.approx(0.1 * 1.12, rel=0.02)

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            # Drive to peak — outer should be capped at _OUTER_GLOW_PEAK_TARGET
            for _ in range(80):
                overlay.update_glow_amplitude(1.0)
            overlay._inner_shadow.setShadowOpacity_.assert_called_with(1.0)
            peak_tight = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            peak_wide = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert peak_tight == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 0.7, rel=0.02)
            assert peak_wide == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 1.12, rel=0.02)

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            # Above cap threshold — should still clamp at peak target
            for _ in range(80):
                overlay.update_glow_amplitude(0.81)
            capped_tight = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            capped_wide = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert capped_tight == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 0.7, rel=0.02)
            assert capped_wide == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 1.12, rel=0.02)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_overlay_glow_color_gets_much_bluer_than_the_edge_glow_base(self, mock_pyobjc):
        """The overlay can run bluer than the bezel glow so it still reads against the keyboard."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            previous_overlay_sat = colorsys.rgb_to_hsv(0.38, 0.52, 1.0)[1]
            overlay_sat = colorsys.rgb_to_hsv(*mod._GLOW_COLOR)[1]

            assert overlay_sat == pytest.approx(min(previous_overlay_sat * 1.28, 1.0), rel=0.02)
            assert overlay_sat > previous_overlay_sat
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
    """Overlay bg/text/glow cross-fades between dark and light based on brightness."""

    def _make_overlay(self, mod):
        overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
        overlay._visible = True
        overlay._text_view = MagicMock()
        overlay._text_amplitude = 0.0
        overlay._content_view = MagicMock()
        overlay._inner_shadow = MagicMock()
        overlay._outer_glow_tight = MagicMock()
        overlay._outer_glow_wide = MagicMock()
        overlay._brightness = 0.0
        return overlay

    def test_set_brightness_stores_value(self, mock_pyobjc):
        """set_brightness should store the brightness for use by amplitude updates."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.7)
            assert overlay._brightness == pytest.approx(0.7)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_uses_dark_bg_light_text(self, mock_pyobjc):
        """On a dark screen, bg should be dark and text should be white."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            # Text should be white (r=1, g=1, b=1)
            text_call = overlay._text_view.setTextColor_.call_args[0][0]
            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            # Last text color call — find the one used for setTextColor_
            text_color_args = None
            for call in color_calls:
                r, g, b, a = call[0]
                if r == pytest.approx(1.0) and g == pytest.approx(1.0) and b == pytest.approx(1.0):
                    text_color_args = call[0]
            assert text_color_args is not None, "Expected white text on dark background"

            # Background should be dark (close to 0.1, 0.1, 0.12)
            bg_call = overlay._content_view.layer().setBackgroundColor_.call_args[0][0]
            bg_color_args = None
            for call in color_calls:
                r, g, b, a = call[0]
                if r < 0.2 and g < 0.2 and b < 0.2:
                    bg_color_args = call[0]
            assert bg_color_args is not None, "Expected dark bg on dark background"
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_light_background_uses_light_bg_dark_text(self, mock_pyobjc):
        """On a bright screen, bg should be light/frosted and text should be dark."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(1.0)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            # Text should be dark
            text_color_args = None
            for call in color_calls:
                r, g, b, a = call[0]
                if r < 0.3 and g < 0.3 and b < 0.3 and a > 0.5:
                    text_color_args = call[0]
            assert text_color_args is not None, "Expected dark text on light background"

            # Background should be light (close to white/frosted)
            bg_color_args = None
            for call in color_calls:
                r, g, b, a = call[0]
                if r > 0.7 and g > 0.7 and b > 0.7:
                    bg_color_args = call[0]
            assert bg_color_args is not None, "Expected light bg on light background"
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_mid_brightness_blends_colors(self, mock_pyobjc):
        """At 50% brightness, bg and text should be intermediate — not pure dark or light."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.5)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            # Text should be mid-gray (not white, not black)
            text_r = None
            for call in color_calls:
                r, g, b, a = call[0]
                # The text color call — should be close to mid
                if 0.3 < r < 0.8 and 0.3 < g < 0.8 and a > 0.5:
                    text_r = r
            assert text_r is not None, "Expected mid-tone text at 50% brightness"

            # Background should also be intermediate
            bg_found = False
            for call in color_calls:
                r, g, b, a = call[0]
                if 0.3 < r < 0.8 and 0.3 < g < 0.8:
                    bg_found = True
            assert bg_found, "Expected mid-tone bg at 50% brightness"
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_default_brightness_is_dark(self, mock_pyobjc):
        """Without set_brightness, overlay defaults to dark-bg mode (brightness=0)."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            # Don't call set_brightness — default should be 0.0
            assert overlay._brightness == pytest.approx(0.0)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_brightness_cross_fade_is_continuous(self, mock_pyobjc):
        """Small brightness changes should produce small color changes — no discontinuities."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            def get_bg_rgb(brightness):
                overlay = self._make_overlay(mod)
                overlay.set_brightness(brightness)
                mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
                overlay.update_text_amplitude(10.0)
                # Find the bg color call (used for setBackgroundColor_)
                for call in mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list:
                    r, g, b, a = call[0]
                    return (r, g, b)
                return None

            c1 = get_bg_rgb(0.4)
            c2 = get_bg_rgb(0.5)
            c3 = get_bg_rgb(0.6)

            assert c1 is not None and c2 is not None and c3 is not None
            # Each step should be a small change, not a big jump
            for i in range(3):
                assert abs(c2[i] - c1[i]) < 0.25, "Jump too large between 0.4 and 0.5"
                assert abs(c3[i] - c2[i]) < 0.25, "Jump too large between 0.5 and 0.6"
        finally:
            sys.modules.pop("spoke.overlay", None)
