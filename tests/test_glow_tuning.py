"""Contract tests for screen-edge glow tuning."""
import colorsys
import importlib
import sys
from unittest.mock import MagicMock

import pytest


class TestGlowTuning:
    """Keep the screen-edge glow restrained at peaks without flattening quiet response."""

    def _make_glow(self, mod):
        glow = mod.GlowOverlay.__new__(mod.GlowOverlay)
        glow._visible = False
        glow._window = MagicMock()
        glow._glow_layer = MagicMock()
        glow._glow_layer.opacity.return_value = 0.07
        glow._fade_in_until = 0.0
        glow._update_count = 0
        glow._noise_floor = 0.0
        glow._smoothed_amplitude = 0.0
        glow._cap_factor = 1.0
        glow._shadow_shape = MagicMock()
        glow._gradient_layers = []
        glow._screen = object()
        glow._hide_timer = None
        glow._hide_generation = 0
        glow._glow_color = mod._GLOW_COLOR
        glow._glow_base_opacity = mod._GLOW_BASE_OPACITY
        glow._glow_peak_target = mod._GLOW_PEAK_TARGET
        glow._dim_layer = MagicMock()
        glow._dim_layer.opacity.return_value = 0.0
        return glow

    def test_screen_dim_fade_durations_are_shortened_for_dev_patch(self):
        """The temporary dimmer patch should keep fade timings short enough to avoid overlap."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_SHOW_FADE_S == pytest.approx(1.08)
            assert mod._DIM_HIDE_FADE_S == pytest.approx(2.4)
            assert mod._GLOW_SHOW_TIMING == "easeIn"
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_glow_style_keeps_dark_backgrounds_muted_and_softens_light_backgrounds(self, mock_pyobjc):
        """Dark backgrounds should stay muted while bright scenes ease back the rim saturation."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            dark_color, dark_base, dark_peak = mod._glow_style_for_brightness(0.0)
            light_color, light_base, light_peak = mod._glow_style_for_brightness(1.0)
            previous_dark_sat = colorsys.rgb_to_hsv(0.50, 0.59, 0.84)[1]
            previous_light_sat = colorsys.rgb_to_hsv(0.34, 0.50, 1.0)[1]
            dark_sat = colorsys.rgb_to_hsv(*dark_color)[1]
            light_sat = colorsys.rgb_to_hsv(*light_color)[1]

            assert light_base > dark_base
            assert light_peak > dark_peak
            assert dark_peak == pytest.approx(mod._GLOW_PEAK_TARGET_DARK)
            assert light_peak == pytest.approx(mod._GLOW_MAX_OPACITY)
            assert dark_sat == pytest.approx(previous_dark_sat * 0.4, rel=0.08)
            assert light_sat == pytest.approx(previous_light_sat * 0.5, rel=0.02)
            assert light_base == pytest.approx(0.2744)
            assert light_sat > dark_sat
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_edge_glow_bands_shift_saturation_from_inner_to_outer(self, mock_pyobjc):
        """The tight edge band should calm down while the outer tail gets more chromatic."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            light_color, _, _ = mod._glow_style_for_brightness(1.0)
            inner_color, middle_color, outer_color = mod._edge_band_colors(light_color)
            light_sat = colorsys.rgb_to_hsv(*light_color)[1]
            inner_sat = colorsys.rgb_to_hsv(*inner_color)[1]
            middle_sat = colorsys.rgb_to_hsv(*middle_color)[1]
            outer_sat = colorsys.rgb_to_hsv(*outer_color)[1]

            assert inner_sat == pytest.approx(light_sat * 0.7, rel=0.02)
            assert middle_sat == pytest.approx(light_sat, rel=0.02)
            assert outer_sat == pytest.approx(min(light_sat * 1.8, 1.0), rel=0.02)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_distance_field_opacity_drops_off_hard_inward(self, mock_pyobjc):
        """The procedural falloff should stay hottest at the edge and get steeper as power increases."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            edge = mod._distance_field_opacity(0.0, 18.0, 1.9)
            mid = mod._distance_field_opacity(9.0, 18.0, 1.9)
            wide = mod._distance_field_opacity(18.0, 18.0, 1.9)
            tail = mod._distance_field_opacity(36.0, 18.0, 1.9)
            softer = mod._distance_field_opacity(27.0, 18.0, 1.1)
            steeper = mod._distance_field_opacity(27.0, 18.0, 1.9)

            assert edge == pytest.approx(1.0)
            assert edge > mid > wide > tail
            assert steeper < softer
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_edge_mix_gives_light_backgrounds_extra_subtractive_presence(self, mock_pyobjc):
        """Bright scenes should get a modest subtractive boost so the edge treatment stays visible."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            dark_add, dark_sub = mod._edge_mix_for_brightness(0.1)
            mid_add, mid_sub = mod._edge_mix_for_brightness(0.5)
            light_add, light_sub = mod._edge_mix_for_brightness(1.0)

            assert dark_add > dark_sub
            assert mid_sub > 0.5
            assert light_add == pytest.approx(0.0)
            assert light_sub == pytest.approx(1.664)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_dimmer_moves_closer_to_opaque(self, mock_pyobjc):
        """Bright scenes should darken more assertively behind the border treatment."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_OPACITY_LIGHT == pytest.approx(0.424)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_vignette_tail_doubles_in_strength(self, mock_pyobjc):
        """The widest bright-scene edge tail should read much more strongly on white backgrounds."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            specs = mod._continuous_vignette_pass_specs()
            tail = next(spec for spec in specs if spec["name"] == "tail")
            assert tail["alpha"] == pytest.approx(0.19)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_show_applies_brightness_adaptive_glow_style(self, mock_pyobjc, monkeypatch):
        """Show should snapshot brightness and cache the active glow style for the session."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 1.0)

            expected_color, expected_base, expected_peak = mod._glow_style_for_brightness(1.0)

            glow.show()

            assert glow._glow_color == pytest.approx(expected_color)
            assert glow._glow_base_opacity == pytest.approx(expected_base)
            assert glow._glow_peak_target == pytest.approx(expected_peak)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_show_applies_dark_background_glow_style(self, mock_pyobjc, monkeypatch):
        """Dark backgrounds should cache the calmer, less saturated glow style."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 0.0)

            expected_color, expected_base, expected_peak = mod._glow_style_for_brightness(0.0)

            glow.show()

            assert glow._glow_color == pytest.approx(expected_color)
            assert glow._glow_base_opacity == pytest.approx(expected_base)
            assert glow._glow_peak_target == pytest.approx(expected_peak)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_show_invalidates_pending_hide_timer(self, mock_pyobjc, monkeypatch):
        """A new recording should cancel the prior teardown timer before restarting the glow."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            pending_timer = MagicMock()
            glow._hide_timer = pending_timer
            monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 0.5)

            glow.show()

            pending_timer.invalidate.assert_called_once_with()
            assert glow._hide_timer is None
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_hide_schedules_window_teardown_after_dim_fade(self, mock_pyobjc):
        """The window should stay alive until the dim fade has had time to finish."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            presentation = MagicMock()
            presentation.opacity.return_value = 0.3
            glow._dim_layer.presentationLayer.return_value = presentation

            timer = MagicMock()
            mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer

            glow.hide()

            call = mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
            assert call.args[0] == pytest.approx(2.45)
            assert call.args[1] is glow
            assert call.args[2] == "hideWindowAfterFade:"
            assert call.args[3] == 1
            assert call.args[4] is False
            assert glow._hide_timer is timer
            assert glow._hide_generation == 1
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_hide_uses_presentation_opacity_for_glow_fade_out(self, mock_pyobjc):
        """Hide should fade from the live on-screen glow opacity, not stale model state."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._dim_layer = None
            glow_presentation = MagicMock()
            glow_presentation.opacity.return_value = 0.22
            glow._glow_layer.presentationLayer.return_value = glow_presentation

            glow.hide()

            anim = mod.CABasicAnimation.animationWithKeyPath_.return_value
            anim.setFromValue_.assert_called_with(0.22)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_hide_window_after_fade_ignores_stale_timer_generation(self, mock_pyobjc):
        """An older hide timer must not tear down a later recording cycle."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._visible = False
            glow._hide_generation = 2

            stale_timer = MagicMock()
            stale_timer.userInfo.return_value = 1

            glow.hideWindowAfterFade_(stale_timer)

            glow._window.orderOut_.assert_not_called()
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_hide_window_after_fade_orders_out_current_hide_generation(self, mock_pyobjc):
        """The active hide timer should still complete teardown once the fade finishes."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._visible = False
            glow._hide_generation = 3

            timer = MagicMock()
            timer.userInfo.return_value = 3

            glow.hideWindowAfterFade_(timer)

            glow._window.orderOut_.assert_called_once_with(None)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_glow_shadow_radius_is_doubled_for_softer_bloom(self, mock_pyobjc):
        """The edge glow should spread farther so lower peak opacity still reads as glow."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._GLOW_SHADOW_RADIUS == 60.0
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_white_backgrounds_can_drive_the_glow_to_full_opacity(self, mock_pyobjc):
        """Bright scenes should be allowed to push the glow all the way to full strength."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._GLOW_MAX_OPACITY == pytest.approx(1.0)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_glow_peak_is_softened_without_changing_quiet_levels(
        self, mock_pyobjc
    ):
        """Quiet and mid-level glow should stay intact while full-scale peaks get much dimmer."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._compress_screen_glow_peak(0.1) == pytest.approx(0.1)
            assert mod._compress_screen_glow_peak(mod._GLOW_PEAK_TARGET) == pytest.approx(
                mod._GLOW_PEAK_TARGET
            )
            assert mod._compress_screen_glow_peak(0.6) == pytest.approx(mod._GLOW_PEAK_TARGET)
            assert mod._compress_screen_glow_peak(0.81) == pytest.approx(mod._GLOW_PEAK_TARGET)

            peak_opacity = mod._compress_screen_glow_peak(1.0)
            assert peak_opacity == pytest.approx(mod._GLOW_PEAK_TARGET)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_glow_countdown_scales_border_opacity_too(self, mock_pyobjc, monkeypatch):
        """The recording-cap countdown should dim the border glow, not just recolor it."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            def _make_glow(cap_factor: float):
                glow = self._make_glow(mod)
                glow._visible = True
                glow._cap_factor = cap_factor
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
