"""Contract tests for overlay timing constants."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


class TestOverlayTiming:
    """Keep the overlay tuned to the current fast-handoff UX."""

    def test_fade_out_is_shortened_for_fast_finalization(self, mock_pyobjc):
        """Fade-out should get out of the way now that final injection lands quickly."""
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        try:
            assert mod._FADE_OUT_S == 0.18
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

            overlay.update_glow_amplitude(0.1)
            overlay._inner_shadow.setShadowOpacity_.assert_called_with(0.1)
            overlay._outer_glow_tight.setShadowOpacity_.assert_called_with(0.05)
            low_wide_opacity = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert low_wide_opacity == pytest.approx(0.08)

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            overlay.update_glow_amplitude(1.0)
            overlay._inner_shadow.setShadowOpacity_.assert_called_with(1.0)
            peak_tight_opacity = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            peak_wide_opacity = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert 0.22 <= peak_tight_opacity <= 0.28
            assert 0.36 <= peak_wide_opacity <= 0.44

            overlay._inner_shadow.reset_mock()
            overlay._outer_glow_tight.reset_mock()
            overlay._outer_glow_wide.reset_mock()

            overlay.update_glow_amplitude(0.81)
            capped_tight_opacity = overlay._outer_glow_tight.setShadowOpacity_.call_args[0][0]
            capped_wide_opacity = overlay._outer_glow_wide.setShadowOpacity_.call_args[0][0]
            assert 0.22 <= capped_tight_opacity <= 0.28
            assert 0.36 <= capped_wide_opacity <= 0.44
        finally:
            sys.modules.pop("spoke.overlay", None)
