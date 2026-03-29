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
