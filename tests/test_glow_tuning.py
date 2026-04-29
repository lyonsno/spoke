"""Contract tests for screen-edge glow tuning."""
import colorsys
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spoke.optical_shell_metrics import OpticalShellMetrics


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
        glow._metrics = None
        return glow

    def test_screen_dim_fade_durations_are_shortened_for_dev_patch(self, mock_pyobjc):
        """The temporary dimmer patch should keep fade timings short enough to avoid overlap."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_SHOW_FADE_S == pytest.approx(0.36)
            assert mod._DIM_HIDE_FADE_S == pytest.approx(0.8)
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
            assert light_base == pytest.approx(mod._GLOW_BASE_OPACITY_LIGHT)
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

    def test_edge_mix_keeps_dark_scenes_purely_additive_until_light_backgrounds(self, mock_pyobjc):
        """Dark scenes should not carry any subtractive vignette until the background is genuinely bright."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            dark_add, dark_sub = mod._edge_mix_for_brightness(0.1)
            mid_add, mid_sub = mod._edge_mix_for_brightness(0.5)
            bright_add, bright_sub = mod._edge_mix_for_brightness(0.8)
            light_add, light_sub = mod._edge_mix_for_brightness(1.0)

            assert dark_add == pytest.approx(1.0)
            assert dark_sub == pytest.approx(0.0)
            assert mid_add == pytest.approx(1.0)
            assert mid_sub == pytest.approx(0.0)
            assert bright_add < 1.0
            assert bright_sub > 0.0
            assert light_add == pytest.approx(0.0)
            assert light_sub == pytest.approx(1.664)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_dimmer_moves_closer_to_opaque(self, mock_pyobjc):
        """Bright scenes should darken more assertively behind the border treatment."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_OPACITY_DARK == pytest.approx(0.42)
            assert mod._DIM_OPACITY_LIGHT == pytest.approx(0.636)
            assert mod._dim_target_for_brightness(0.0) == pytest.approx(mod._DIM_OPACITY_DARK)
            assert mod._dim_target_for_brightness(0.5) == pytest.approx(0.8)
            assert mod._dim_target_for_brightness(1.0) == pytest.approx(mod._DIM_OPACITY_LIGHT)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_dimmer_uses_own_stronger_sdf_pass(self, mock_pyobjc):
        """The hold-space dimmer should use its own stronger SDF pass, not a flat wash."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            soft_bloom = next(
                spec
                for spec in mod._continuous_glow_pass_specs()
                if spec["name"] == "wide_bloom"
            )
            specs = mod._continuous_dimmer_pass_specs()
            dimmer = specs[0]

            assert specs == [dimmer]
            assert dimmer["name"] == "hold_dimmer"
            assert dimmer["path_kind"] == "distance_field"
            assert dimmer["falloff"] > soft_bloom["falloff"]
            assert dimmer["power"] < soft_bloom["power"]
            assert dimmer["alpha"] == pytest.approx(soft_bloom["fill_alpha"] * 4.5)
            assert dimmer["alpha"] < 1.0
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_screen_dimmer_tail_extends_inward_without_raising_peak_alpha(self, mock_pyobjc):
        """The hold-space dimmer should fade over a long tail without getting darker at the edge."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            soft_bloom = next(
                spec
                for spec in mod._continuous_glow_pass_specs()
                if spec["name"] == "wide_bloom"
            )
            dimmer = mod._continuous_dimmer_pass_specs()[0]
            old_tail_alpha = mod._distance_field_opacity(
                soft_bloom["falloff"] * 6.0,
                soft_bloom["falloff"] * 1.8,
                2.2,
            )
            new_tail_alpha = mod._distance_field_opacity(
                soft_bloom["falloff"] * 6.0,
                dimmer["falloff"],
                dimmer["power"],
            )

            assert dimmer["alpha"] == pytest.approx(soft_bloom["fill_alpha"] * 4.5)
            assert dimmer["falloff"] >= soft_bloom["falloff"] * 7.5
            assert dimmer["power"] < 1.6
            assert old_tail_alpha < 0.01
            assert new_tail_alpha > 0.25
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_setup_installs_masked_dimmer_pass_instead_of_flat_fill(
        self, mock_pyobjc, monkeypatch
    ):
        """The dim layer should be a container for SDF-masked pass layers."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            class FakeColor:
                def __init__(self, alpha=1.0):
                    self.alpha = alpha

                def CGColor(self):
                    return self

                def colorWithAlphaComponent_(self, alpha):
                    return FakeColor(alpha)

            class FakeNSColor:
                @staticmethod
                def colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha):
                    return FakeColor(alpha)

                @staticmethod
                def clearColor():
                    return FakeColor(0.0)

            class FakeLayer:
                def __init__(self):
                    self.background_color = None
                    self.frame = None
                    self.mask = None
                    self.opacity_value = None
                    self.sublayers = []

                @classmethod
                def alloc(cls):
                    return cls()

                def init(self):
                    return self

                def setFrame_(self, frame):
                    self.frame = frame

                def setBackgroundColor_(self, color):
                    self.background_color = color

                def setOpacity_(self, opacity):
                    self.opacity_value = opacity

                def addSublayer_(self, layer):
                    self.sublayers.append(layer)

                def setMask_(self, mask):
                    self.mask = mask

                def setContents_(self, contents):
                    self.contents = contents

                def setContentsScale_(self, scale):
                    self.contents_scale = scale

            class FakeContent:
                def __init__(self):
                    self.root_layer = FakeLayer()

                def setWantsLayer_(self, wants_layer):
                    self.wants_layer = wants_layer

                def layer(self):
                    return self.root_layer

            class FakeWindow:
                @classmethod
                def alloc(cls):
                    return cls()

                def initWithContentRect_styleMask_backing_defer_(self, *args):
                    self.content = FakeContent()
                    return self

                def contentView(self):
                    return self.content

                def setLevel_(self, level):
                    self.level = level

                def setOpaque_(self, opaque):
                    self.opaque = opaque

                def setBackgroundColor_(self, color):
                    self.background_color = color

                def setIgnoresMouseEvents_(self, ignores):
                    self.ignores_mouse = ignores

                def setHasShadow_(self, has_shadow):
                    self.has_shadow = has_shadow

                def setCollectionBehavior_(self, behavior):
                    self.collection_behavior = behavior

            monkeypatch.setattr(mod, "CALayer", FakeLayer)
            monkeypatch.setattr(mod, "NSColor", FakeNSColor)
            monkeypatch.setattr(mod, "NSWindow", FakeWindow)
            monkeypatch.setattr(mod, "_alpha_field_to_image", lambda alpha: ("image", alpha))

            frame = SimpleNamespace(
                origin=SimpleNamespace(x=0.0, y=0.0),
                size=SimpleNamespace(width=20.0, height=12.0),
            )
            screen = SimpleNamespace(
                frame=lambda: frame,
                backingScaleFactor=lambda: 1.0,
            )
            glow = mod.GlowOverlay.__new__(mod.GlowOverlay)
            glow._screen = screen
            glow._metrics = None

            glow.setup()

            assert glow._dim_layer.background_color is None
            assert len(glow._dim_pass_layers) == 1
            assert glow._dim_pass_layers[0]["spec"]["name"] == "hold_dimmer"
            assert glow._dim_pass_layers[0]["layer"].mask is not None
            assert glow._dim_pass_layers[0]["layer"] in glow._dim_layer.sublayers
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_vignette_tail_doubles_in_strength(self, mock_pyobjc):
        """The widest bright-scene edge tail should read much more strongly on white backgrounds."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            specs = mod._continuous_vignette_pass_specs()
            tail = next(spec for spec in specs if spec["name"] == "tail")
            assert tail["alpha"] == pytest.approx(0.28)
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

    def test_brightness_resample_records_sample_count(self, mock_pyobjc, monkeypatch):
        """Brightness sampling should increment the shared metrics surface."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._metrics = OpticalShellMetrics()
            glow._visible = True
            monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 0.0)

            glow.brightnessResample_(None)

            snapshot = glow._metrics.snapshot()
            assert snapshot["brightness_samples"] == 1
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
            assert call.args[0] == pytest.approx(0.816)
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
            assert mod._compress_screen_glow_peak(0.6) == pytest.approx(0.6)
            assert mod._compress_screen_glow_peak(0.81) == pytest.approx(0.81)
            assert mod._compress_screen_glow_peak(mod._GLOW_PEAK_TARGET) == pytest.approx(
                mod._GLOW_PEAK_TARGET
            )

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
