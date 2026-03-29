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

            # Feed steady signal to let the independent smoothing converge
            for _ in range(30):
                overlay.update_glow_amplitude(0.1)
            inner_opacity = overlay._inner_shadow.setShadowOpacity_.call_args[0][0]
            # Smoothed value * 1.4 — converges close to 0.1 * 1.4 = 0.14
            assert inner_opacity == pytest.approx(0.14, abs=0.01)
            tight_opacity = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            assert tight_opacity < 0.1  # capped outer glow stays low

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            for _ in range(30):
                overlay.update_glow_amplitude(1.0)
            inner_at_peak = overlay._inner_shadow.setShadowOpacity_.call_args[0][0]
            assert inner_at_peak == pytest.approx(1.0, abs=0.15)  # near 1.0 * 1.4 clamped
            peak_tight_opacity = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            peak_wide_opacity = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert peak_tight_opacity == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 0.7, abs=0.02)
            assert peak_wide_opacity == pytest.approx(min(mod._OUTER_GLOW_PEAK_TARGET * 1.12, 1.0), abs=0.02)

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            # Single additional call at high value — smoothed, capped, scaled
            overlay.update_glow_amplitude(0.81)
            capped_tight_opacity = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            capped_wide_opacity = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert capped_tight_opacity == pytest.approx(mod._OUTER_GLOW_PEAK_TARGET * 0.7, abs=0.02)
            assert capped_wide_opacity == pytest.approx(min(mod._OUTER_GLOW_PEAK_TARGET * 1.12, 1.0), abs=0.02)
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
        overlay._brightness_target = 0.0
        return overlay

    def test_set_brightness_immediate_snaps(self, mock_pyobjc):
        """set_brightness(immediate=True) should snap brightness instantly."""
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
        """set_brightness() should set target but not snap — brightness chases over time."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.8)
            # Target is set but brightness hasn't moved yet
            assert overlay._brightness_target == pytest.approx(0.8)
            assert overlay._brightness == pytest.approx(0.0)

            # Simulate ~30 amplitude frames (~0.5s at 60Hz) — should be close
            for _ in range(30):
                overlay.update_text_amplitude(0.0)
            assert overlay._brightness > 0.6, f"Expected >0.6 after 30 frames, got {overlay._brightness:.3f}"

            # After many more frames, should converge
            for _ in range(100):
                overlay.update_text_amplitude(0.0)
            assert overlay._brightness == pytest.approx(0.8, abs=0.01)
        finally:
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_uses_dark_bg_light_text(self, mock_pyobjc):
        """On a dark screen, bg should be dark and text should be white."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.0, immediate=True)

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
            overlay.set_brightness(1.0, immediate=True)

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

    def test_mid_brightness_has_contrast_gap(self, mock_pyobjc):
        """At 50% brightness, text should still be brighter than bg — no muddy collapse."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            overlay = self._make_overlay(mod)
            overlay.set_brightness(0.5, immediate=True)

            mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
            overlay.update_text_amplitude(10.0)

            color_calls = mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list
            # First call is text, second is bg (setTextColor_ before setBackgroundColor_)
            text_r, text_g, text_b, _ = color_calls[0][0]
            bg_r, bg_g, bg_b, _ = color_calls[1][0]

            text_lum = 0.299 * text_r + 0.587 * text_g + 0.114 * text_b
            bg_lum = 0.299 * bg_r + 0.587 * bg_g + 0.114 * bg_b

            # Text should be noticeably brighter than bg at mid-brightness
            # (the offset text curve keeps text whiter longer)
            assert text_lum > bg_lum + 0.15, (
                f"Contrast gap too small at 50% brightness: "
                f"text_lum={text_lum:.3f} bg_lum={bg_lum:.3f}"
            )
            # bg should be intermediate (not pure dark or light)
            assert 0.3 < bg_lum < 0.7, f"bg not intermediate: {bg_lum:.3f}"
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
                overlay.set_brightness(brightness, immediate=True)
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
