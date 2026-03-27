"""Contract tests for screen-edge glow tuning."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


class TestGlowTuning:
    """Keep the screen-edge glow restrained at peaks without flattening quiet response."""

    def test_screen_glow_shadow_radius_is_doubled_for_softer_bloom(self, mock_pyobjc):
        """The edge glow should spread farther so lower peak opacity still reads as glow."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._GLOW_SHADOW_RADIUS == 60.0
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_glow_peak_is_softened_without_changing_quiet_levels(
        self, mock_pyobjc
    ):
        """Quiet and mid-level glow should stay intact while full-scale peaks get much dimmer."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._compress_screen_glow_peak(0.2) == pytest.approx(0.2)
            assert mod._compress_screen_glow_peak(0.245) == pytest.approx(0.245)
            assert mod._compress_screen_glow_peak(0.6) == pytest.approx(0.245)
            assert mod._compress_screen_glow_peak(0.81) == pytest.approx(0.245)

            peak_opacity = mod._compress_screen_glow_peak(1.0)
            assert 0.23 <= peak_opacity <= 0.26
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_glow_countdown_scales_border_opacity_too(self, mock_pyobjc, monkeypatch):
        """The recording-cap countdown should dim the border glow, not just recolor it."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            def _make_glow(cap_factor: float):
                glow = mod.GlowOverlay.__new__(mod.GlowOverlay)
                glow._visible = True
                glow._glow_layer = MagicMock()
                glow._fade_in_until = 0.0
                glow._update_count = 0
                glow._noise_floor = 0.0
                glow._smoothed_amplitude = 0.0
                glow._cap_factor = cap_factor
                glow._shadow_shape = MagicMock()
                glow._gradient_layers = []
                return glow

            monkeypatch.setattr(mod.time, "monotonic", lambda: 1.0)

            uncapped = _make_glow(1.0)
            uncapped.update_amplitude(1.0)
            uncapped_opacity = uncapped._glow_layer.setOpacity_.call_args[0][0]

            capped = _make_glow(0.5)
            capped.update_amplitude(1.0)
            capped_opacity = capped._glow_layer.setOpacity_.call_args[0][0]

            assert capped_opacity < uncapped_opacity
        finally:
            sys.modules.pop("spoke.glow", None)
