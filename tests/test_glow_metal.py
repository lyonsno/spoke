"""Contracts for the Mecha Visor additive Metal seam."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _import_glow(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_WIDE_BLOOM", "1")
    sys.modules.pop("spoke.glow", None)
    return importlib.import_module("spoke.glow")


def _make_glow(mod):
    glow = mod.GlowOverlay.__new__(mod.GlowOverlay)
    glow._visible = False
    glow._window = MagicMock()
    glow._glow_layer = MagicMock()
    glow._glow_layer.opacity.return_value = 0.07
    glow._glow_layer.presentationLayer.return_value = None
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
    glow._dim_layer = None
    glow._brightness_timer = None
    glow._metal_frame_timer = None
    glow._cancel_pending_hide = lambda: None
    glow._glow_pass_layers = [{"layer": MagicMock(), "spec": {"fill_role": "inner", "fill_alpha": 0.28}}]
    glow._vignette_pass_layers = [{"layer": MagicMock(), "spec": {"alpha": 0.2, "color_scale": 0.1}}]
    return glow


def test_apply_glow_color_routes_additive_surface_to_metal_renderer(mock_pyobjc, monkeypatch):
    mod = _import_glow(mock_pyobjc, monkeypatch)
    try:
        glow = _make_glow(mod)
        glow._additive_renderer = MagicMock()

        glow._apply_glow_color(mod._GLOW_COLOR)

        glow._additive_renderer.set_base_color.assert_called_once_with(mod._GLOW_COLOR)
        glow._glow_pass_layers[0]["layer"].setBackgroundColor_.assert_called_once()
        glow._vignette_pass_layers[0]["layer"].setBackgroundColor_.assert_called_once()
    finally:
        sys.modules.pop("spoke.glow", None)


def test_show_starts_metal_frame_timer_when_renderer_present(mock_pyobjc, monkeypatch):
    mod = _import_glow(mock_pyobjc, monkeypatch)
    try:
        glow = _make_glow(mod)
        glow._additive_renderer = MagicMock()
        glow._visible = True
        timer = MagicMock()
        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = timer
        monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 0.0)

        glow.show()

        glow._additive_renderer.set_additive_mix.assert_called_once_with(1.0)
        glow._additive_renderer.draw_frame.assert_called()
        timer_calls = mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args_list
        metal_call = next(call for call in timer_calls if call.args[2] == "metalFrameTick:")
        assert metal_call.args[0] == pytest.approx(mod._METAL_FRAME_INTERVAL_S)
        assert metal_call.args[1] is glow
        assert metal_call.args[4] is True
        assert glow._metal_frame_timer is timer
    finally:
        sys.modules.pop("spoke.glow", None)


def test_show_falls_back_to_cpu_glow_when_metal_draw_fails(mock_pyobjc, monkeypatch):
    mod = _import_glow(mock_pyobjc, monkeypatch)
    try:
        glow = _make_glow(mod)
        glow._additive_renderer = MagicMock()
        glow._additive_renderer.draw_frame.return_value = False
        glow._visible = True
        monkeypatch.setattr(mod, "_sample_screen_brightness", lambda screen: 0.0)

        glow.show()

        glow._additive_renderer.set_additive_mix.assert_called_once_with(1.0)
        glow._additive_renderer.draw_frame.assert_called()
        glow._glow_pass_layers[0]["layer"].setBackgroundColor_.assert_called()
        timer_calls = mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args_list
        assert not any(call.args[2] == "metalFrameTick:" for call in timer_calls)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_hide_invalidates_metal_frame_timer(mock_pyobjc, monkeypatch):
    mod = _import_glow(mock_pyobjc, monkeypatch)
    try:
        glow = _make_glow(mod)
        glow._visible = True
        glow._additive_renderer = MagicMock()
        metal_timer = MagicMock()
        brightness_timer = MagicMock()
        glow._metal_frame_timer = metal_timer
        glow._brightness_timer = brightness_timer

        glow.hide()

        metal_timer.invalidate.assert_called_once_with()
        brightness_timer.invalidate.assert_called_once_with()
        assert glow._metal_frame_timer is None
        assert glow._brightness_timer is None
    finally:
        sys.modules.pop("spoke.glow", None)


def test_mecha_visor_sine_boost_can_raise_idle_signal_without_rms(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_WIDE_BLOOM", "1")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_SINE_BOOST", "0.8")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_SINE_HZ", "0.5")
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        boosted = mod._mecha_visor_signal_boost(0.0, now=0.5)
        trough = mod._mecha_visor_signal_boost(0.0, now=1.5)

        assert boosted == pytest.approx(0.8, rel=0.01)
        assert trough == pytest.approx(0.0, abs=1e-6)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_mecha_visor_wide_bloom_tuning_pushes_tail_farther_into_screen(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_WIDE_BLOOM", "1")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_WIDE_BLOOM_FALLOFF_SCALE", "2.4")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_WIDE_BLOOM_ALPHA_SCALE", "1.5")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_WIDE_BLOOM_POWER_SCALE", "0.8")
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        specs = mod._continuous_glow_pass_specs()
        wide = next(spec for spec in specs if spec["name"] == "wide_bloom")

        assert wide["falloff"] == pytest.approx(57.6)
        assert wide["fill_alpha"] == pytest.approx(0.72)
        assert wide["power"] == pytest.approx(2.32)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_mecha_visor_can_zero_the_core_pass_for_a_softer_field(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_WIDE_BLOOM", "1")
    monkeypatch.setenv("SPOKE_MECHA_VISOR_DISABLE_CORE_PASS", "1")
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        specs = mod._continuous_glow_pass_specs()
        core = next(spec for spec in specs if spec["name"] == "core")
        tight = next(spec for spec in specs if spec["name"] == "tight_bloom")

        assert core["fill_alpha"] == pytest.approx(0.0)
        assert tight["fill_alpha"] > 0.0
    finally:
        sys.modules.pop("spoke.glow", None)
