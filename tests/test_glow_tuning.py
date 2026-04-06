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

    def test_screen_dim_fade_durations_keep_the_glow_lingering_longer(self):
        """Vision Quest should let the glow hang a little longer while the dimmer still gets out of the way quickly."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_SHOW_FADE_S == pytest.approx(0.72)
            assert mod._DIM_HIDE_FADE_S == pytest.approx(1.6)
            assert mod._GLOW_SHOW_TIMING == "easeIn"
            assert mod._GLOW_SHOW_FADE_S == pytest.approx(0.132)
            assert mod._GLOW_HIDE_FADE_S == pytest.approx(2.4)
            assert mod._RISE_FACTOR == pytest.approx(0.90)
            assert mod._DECAY_FACTOR == pytest.approx(0.40)
            assert mod._VIGNETTE_RISE_FACTOR == pytest.approx(0.9292893218813453)
            assert mod._VIGNETTE_DECAY_FACTOR == pytest.approx(0.282842712474619)
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
            assert light_peak < dark_peak
            assert dark_peak == pytest.approx(mod._GLOW_PEAK_TARGET_DARK)
            assert light_peak == pytest.approx(0.75)
            assert dark_sat == pytest.approx(previous_dark_sat * 0.4, rel=0.08)
            assert light_sat == pytest.approx(previous_light_sat * 0.5, rel=0.02)
            assert dark_base == pytest.approx(0.1875)
            assert light_base == pytest.approx(0.2058)
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

    def test_rational_additive_curve_holds_more_energy_than_exponential(self, mock_pyobjc):
        """The rational Vision Quest mode should keep the outer shell fatter at the same distance."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            distance = 12.0
            falloff = 12.0
            power = 2.0

            exponential = mod._distance_field_opacity_with_mode(
                distance,
                falloff,
                power,
                mod.ADDITIVE_CURVE_MODE_EXPONENTIAL,
            )
            rational = mod._distance_field_opacity_with_mode(
                distance,
                falloff,
                power,
                mod.ADDITIVE_CURVE_MODE_RATIONAL,
            )

            assert rational > exponential
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_additive_mask_intensity_control_scales_midfield_alpha(self, mock_pyobjc):
        """Vision Quest should expose additive mask overdrive as a separate global lever."""
        import math
        import numpy as np

        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            alpha = mod._distance_field_alpha(
                np.array([[-18.0]], dtype=np.float32),
                18.0,
                1.0,
                intensity_multiplier=1.5,
            )

            assert float(alpha[0, 0]) == pytest.approx(min(math.exp(-1.0) * 1.5, 1.0))
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_wide_bloom_profile_presets_reshape_tail_energy(self, mock_pyobjc):
        """Tintilla should be able to bend the wide-bloom knee without changing the whole additive stack."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            tight = mod._continuous_glow_pass_specs(mod.WIDE_BLOOM_PROFILE_TIGHT)[2]
            quest = mod._continuous_glow_pass_specs(mod.WIDE_BLOOM_PROFILE_QUEST)[2]
            mist = mod._continuous_glow_pass_specs(mod.WIDE_BLOOM_PROFILE_MIST)[2]

            assert tight["falloff"] < quest["falloff"] < mist["falloff"]
            assert tight["power"] > quest["power"] > mist["power"]
            assert tight["falloff"] == pytest.approx(18.0)
            assert quest["falloff"] == pytest.approx(24.0)
            assert mist["falloff"] == pytest.approx(30.0)
            assert tight["power"] == pytest.approx(3.4)
            assert quest["power"] == pytest.approx(2.9)
            assert mist["power"] == pytest.approx(2.4)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_additive_stack_now_biases_wide_bloom_over_core(self, mock_pyobjc):
        """Vision Quest should let the wide bloom carry more of the additive read while the core backs off."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            specs = mod._continuous_glow_pass_specs()
            core = next(spec for spec in specs if spec["name"] == "core")
            tight = next(spec for spec in specs if spec["name"] == "tight_bloom")
            wide = next(spec for spec in specs if spec["name"] == "wide_bloom")

            assert core["falloff"] == pytest.approx(5.0)
            assert core["power"] == pytest.approx(2.1)
            assert tight["falloff"] == pytest.approx(11.5)
            assert tight["power"] == pytest.approx(2.6)
            assert core["fill_alpha"] == pytest.approx(0.14)
            assert wide["fill_alpha"] == pytest.approx(0.048)
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

    def test_visual_layer_state_can_hide_screen_dimmer(self, mock_pyobjc):
        """Tintilla should be able to isolate the screen dimmer separately from the glow stack."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            class _State:
                def is_visible(self, layer_id):
                    return layer_id != mod.SCREEN_DIMMER_LAYER_ID

            glow = self._make_glow(mod)
            glow._visual_layer_state = _State()
            glow._glow_pass_layers = []
            glow._vignette_pass_layers = []
            glow._apply_glow_color = MagicMock()

            glow._apply_visual_layer_state()

            glow._dim_layer.setHidden_.assert_called_once_with(True)
            glow._apply_glow_color.assert_called_once_with(glow._glow_color)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_dimmer_moves_closer_to_opaque(self, mock_pyobjc):
        """Hold-time dimming should back off to 50% of the current curve without changing its shape."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._DIM_OPACITY_DARK == pytest.approx(0.147)
            assert mod._DIM_OPACITY_LIGHT == pytest.approx(0.2226)
            assert mod._dim_target_for_brightness(0.0) == pytest.approx(mod._DIM_OPACITY_DARK)
            assert mod._dim_target_for_brightness(0.5) == pytest.approx(0.28)
            assert mod._dim_target_for_brightness(1.0) == pytest.approx(mod._DIM_OPACITY_LIGHT)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_vignette_core_and_mid_carry_more_of_the_darkness(self, mock_pyobjc):
        """Bright-scene vignette should read as stacked strata, not just a giant tail."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            specs = mod._continuous_vignette_pass_specs()
            core = next(spec for spec in specs if spec["name"] == "core")
            mid = next(spec for spec in specs if spec["name"] == "mid")
            tail = next(spec for spec in specs if spec["name"] == "tail")
            assert mod._VIGNETTE_OPACITY_SCALE == pytest.approx(0.78)
            assert core["falloff"] == pytest.approx(14.0)
            assert mid["falloff"] == pytest.approx(28.0)
            assert tail["falloff"] == pytest.approx(42.0)
            assert core["alpha"] == pytest.approx(1.0)
            assert mid["alpha"] == pytest.approx(1.0)
            assert tail["alpha"] == pytest.approx(0.9)
            assert core["floor_gain"] == pytest.approx(2.0)
            assert core["peak_gain"] == pytest.approx(4.0)
            assert mid["floor_gain"] == pytest.approx(1.0)
            assert mid["peak_gain"] == pytest.approx(0.7)
            assert tail["floor_gain"] == pytest.approx(0.75)
            assert tail["peak_gain"] == pytest.approx(0.5)
            assert core["power"] == pytest.approx(1.15)
            assert mid["power"] == pytest.approx(1.3)
            assert tail["power"] == pytest.approx(1.45)
            assert core["color_scale"] == pytest.approx(0.0009375)
            assert mid["color_scale"] == pytest.approx(0.00375)
            assert tail["color_scale"] == pytest.approx(0.015)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_vignette_profile_presets_reshape_tail_energy(self, mock_pyobjc):
        """Tintilla should be able to bend the vignette scoop independently of the additive glow."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            tight = mod._continuous_vignette_pass_specs(mod.WIDE_BLOOM_PROFILE_TIGHT)
            quest = mod._continuous_vignette_pass_specs(mod.WIDE_BLOOM_PROFILE_QUEST)
            mist = mod._continuous_vignette_pass_specs(mod.WIDE_BLOOM_PROFILE_MIST)

            tight_tail = next(spec for spec in tight if spec["name"] == "tail")
            quest_tail = next(spec for spec in quest if spec["name"] == "tail")
            mist_tail = next(spec for spec in mist if spec["name"] == "tail")

            assert tight_tail["falloff"] < quest_tail["falloff"] < mist_tail["falloff"]
            assert tight_tail["power"] > quest_tail["power"] > mist_tail["power"]
            assert tight_tail["falloff"] == pytest.approx(34.0)
            assert quest_tail["falloff"] == pytest.approx(42.0)
            assert mist_tail["falloff"] == pytest.approx(54.0)
            assert tight_tail["power"] == pytest.approx(1.6)
            assert quest_tail["power"] == pytest.approx(1.45)
            assert mist_tail["power"] == pytest.approx(1.3)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_vignette_intensity_control_scales_darkness(self, mock_pyobjc):
        """A stronger vignette intensity should darken the subtractive tint instead of sharing the additive multiplier."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            baseline = mod._continuous_vignette_pass_specs(
                mod.WIDE_BLOOM_PROFILE_QUEST,
                intensity_multiplier=1.0,
            )
            boosted = mod._continuous_vignette_pass_specs(
                mod.WIDE_BLOOM_PROFILE_QUEST,
                intensity_multiplier=2.0,
            )
            base_tail = next(spec for spec in baseline if spec["name"] == "tail")
            boosted_tail = next(spec for spec in boosted if spec["name"] == "tail")

            assert boosted_tail["color_scale"] == pytest.approx(base_tail["color_scale"] / 2.0)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_vignette_core_gets_a_stronger_floor_and_peak_than_other_passes(self, mock_pyobjc):
        """The inner vignette shell should carry more of the quiet and loud read than the rest of the stack."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            base_opacity = 0.2
            quiet_core = mod._vignette_pass_opacity(
                base_opacity,
                0.0,
                {"floor_gain": 2.0, "peak_gain": 4.0},
            )
            loud_core = mod._vignette_pass_opacity(
                base_opacity,
                1.0,
                {"floor_gain": 2.0, "peak_gain": 4.0},
            )
            quiet_tail = mod._vignette_pass_opacity(base_opacity, 0.0, {})
            loud_mid = mod._vignette_pass_opacity(
                base_opacity,
                1.0,
                {"floor_gain": 1.0, "peak_gain": 0.7},
            )
            quiet_tail_tuned = mod._vignette_pass_opacity(
                base_opacity,
                0.0,
                {"floor_gain": 0.75, "peak_gain": 0.5},
            )
            loud_tail_tuned = mod._vignette_pass_opacity(
                base_opacity,
                1.0,
                {"floor_gain": 0.75, "peak_gain": 0.5},
            )

            assert quiet_core == pytest.approx(0.4)
            assert loud_core == pytest.approx(0.8)
            assert quiet_core > quiet_tail
            assert loud_mid == pytest.approx(0.14)
            assert quiet_tail_tuned == pytest.approx(0.15)
            assert loud_tail_tuned == pytest.approx(0.10)
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_rational_curve_can_drive_vignette_masks_too(self, mock_pyobjc):
        """The same rational curve family should be available to the subtractive vignette masks."""
        import numpy as np

        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            signed_distance = np.array([[-20.0]], dtype=np.float32)
            exponential = mod._distance_field_alpha(
                signed_distance,
                40.0,
                1.55,
                curve_mode=mod.ADDITIVE_CURVE_MODE_EXPONENTIAL,
            )
            rational = mod._distance_field_alpha(
                signed_distance,
                40.0,
                1.55,
                curve_mode=mod.ADDITIVE_CURVE_MODE_RATIONAL,
            )

            assert float(rational[0, 0]) > float(exponential[0, 0])
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_vignette_masks_follow_the_live_vignette_controls(self, mock_pyobjc, monkeypatch):
        """The live vignette stack should rebuild from its own curve, intensity, and profile controls."""
        import numpy as np

        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            class _State:
                def additive_curve_mode(self):
                    return mod.ADDITIVE_CURVE_MODE_EXPONENTIAL

                def additive_mask_intensity(self):
                    return 1.0

                def wide_bloom_profile(self):
                    return mod.WIDE_BLOOM_PROFILE_QUEST

                def vignette_curve_mode(self):
                    return mod.ADDITIVE_CURVE_MODE_RATIONAL

                def vignette_mask_intensity(self):
                    return 2.0

                def vignette_profile(self):
                    return mod.WIDE_BLOOM_PROFILE_MIST

            def _layer_entry():
                layer = MagicMock()
                layer.mask.return_value = MagicMock()
                return {"layer": layer, "spec": {}}

            glow = mod.GlowOverlay.__new__(mod.GlowOverlay)
            glow._visual_layer_state = _State()
            glow._glow_geometry = {
                "pixel_width": 8,
                "pixel_height": 8,
                "top_radius": 1.0,
                "bottom_radius": 1.0,
                "notch": None,
                "scale": 1.0,
            }
            glow._glow_signed_distance = np.full((8, 8), -4.0, dtype=np.float32)
            glow._glow_mask_scale = 1.0
            glow._active_additive_curve_mode = mod.ADDITIVE_CURVE_MODE_EXPONENTIAL
            glow._active_vignette_curve_mode = mod.ADDITIVE_CURVE_MODE_EXPONENTIAL
            glow._active_additive_mask_intensity = 1.0
            glow._active_wide_bloom_profile = mod.WIDE_BLOOM_PROFILE_QUEST
            glow._active_vignette_mask_intensity = 1.0
            glow._active_vignette_profile = mod.WIDE_BLOOM_PROFILE_QUEST
            glow._glow_pass_layers = [_layer_entry(), _layer_entry(), _layer_entry()]
            glow._vignette_pass_layers = [_layer_entry(), _layer_entry(), _layer_entry()]
            glow._glow_mask_payloads = []
            glow._vignette_mask_payloads = []
            monkeypatch.setattr(
                mod,
                "_alpha_field_to_image",
                lambda alpha: (object(), object()),
            )

            glow._refresh_glow_masks_if_needed()

            assert glow._active_vignette_curve_mode == mod.ADDITIVE_CURVE_MODE_RATIONAL
            assert glow._active_vignette_mask_intensity == pytest.approx(2.0)
            assert glow._active_vignette_profile == mod.WIDE_BLOOM_PROFILE_MIST
            for entry in glow._vignette_pass_layers:
                entry["layer"].mask.return_value.setContents_.assert_called_once()
        finally:
            sys.modules.pop("spoke.glow", None)

    def test_light_background_vignette_keeps_headroom_for_rms_motion(self, mock_pyobjc):
        """Bright-scene vignette should not pin to the ceiling at its floor."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._visible = True
            glow._fade_in_until = 0.0
            glow._noise_floor = 0.0
            glow._smoothed_amplitude = 0.0
            glow._vignette_smoothed_amplitude = 0.0
            glow._glow_base_opacity = mod._GLOW_BASE_OPACITY_LIGHT
            glow._glow_peak_target = mod._GLOW_MAX_OPACITY
            glow._additive_mix = 0.0
            glow._subtractive_mix = mod._edge_mix_for_brightness(1.0)[1]
            glow._vignette_layer = MagicMock()
            glow._vignette_pass_layers = [
                {"layer": MagicMock(), "spec": spec}
                for spec in mod._continuous_vignette_pass_specs()
            ]

            glow.update_amplitude(0.0)
            low_parent_opacity = glow._vignette_layer.setOpacity_.call_args[0][0]
            low_core_opacity = glow._vignette_pass_layers[0]["layer"].setOpacity_.call_args[0][0]

            glow._vignette_layer.reset_mock()
            for entry in glow._vignette_pass_layers:
                entry["layer"].reset_mock()
            glow.update_amplitude(1.0)
            high_parent_opacity = glow._vignette_layer.setOpacity_.call_args[0][0]
            high_core_opacity = glow._vignette_pass_layers[0]["layer"].setOpacity_.call_args[0][0]

            assert low_parent_opacity == pytest.approx(1.0)
            assert high_parent_opacity == pytest.approx(1.0)
            assert low_core_opacity < 1.0
            assert high_core_opacity > low_core_opacity + 0.1
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

    def test_brightness_resample_samples_below_glow_window(self, mock_pyobjc, monkeypatch):
        """Recurring brightness adaptation should grade the scene under our window stack, not our own overlays."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            glow = self._make_glow(mod)
            glow._visible = True
            glow._brightness = 0.2
            glow._window.windowNumber.return_value = 321
            glow._apply_glow_color = MagicMock()
            glow._dim_layer.presentationLayer.return_value = None

            captured = {}

            def _sample(screen, excluding_window_id=None):
                captured["excluding_window_id"] = excluding_window_id
                return 0.9

            monkeypatch.setattr(mod, "_sample_screen_brightness", _sample)

            glow.brightnessResample_(None)

            assert captured["excluding_window_id"] == 321
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
            assert call.args[0] == pytest.approx(2.416)
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

    def test_white_backgrounds_now_cap_the_glow_at_three_quarters(self, mock_pyobjc):
        """Bright scenes should top out lower so we can inspect the new white-side headroom more gently."""
        sys.modules.pop("spoke.glow", None)
        mod = importlib.import_module("spoke.glow")
        try:
            assert mod._GLOW_MAX_OPACITY == pytest.approx(1.0)
            assert mod._GLOW_PEAK_TARGET_LIGHT == pytest.approx(0.75)
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
