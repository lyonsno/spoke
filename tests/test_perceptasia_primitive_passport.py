from __future__ import annotations

from pathlib import Path

import pytest

from spoke.optical_field import OpticalFieldBounds
from spoke.perceptasia_primitive_passport import (
    PERCEPTASIA_PRIMITIVE_CLIENT_ID,
    build_perceptasia_primitive_env,
    build_perceptasia_primitive_request,
    compile_perceptasia_primitive_carrier_config,
)


def _bounds() -> OpticalFieldBounds:
    return OpticalFieldBounds(x=120.0, y=90.0, width=900.0, height=520.0)


def test_passport_request_is_independent_hud_sibling_not_assistant_shell():
    request = build_perceptasia_primitive_request(_bounds(), state="rest")

    assert request.caller_id == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert request.continuity_key == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert request.role == "hud"
    assert request.visibility_scope == "independent"
    assert request.presentation.layer == "hud"
    assert request.presentation.order > 20
    assert request.profile.base == "captured_scene"
    assert request.layout_recipe == "perceptasia-primitive-passport"
    assert request.motion.strategy == "continuous"
    assert request.motion.continuity == "preserve_identity"
    assert request.selected_handoff is None


def test_passport_carrier_captures_live_webview_into_primitive_with_perimeter_material_only():
    config = compile_perceptasia_primitive_carrier_config(_bounds(), state="materialize")

    assert config["client_id"] == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert config["role"] == "hud"
    assert config["presentation_layer"] == "hud"
    assert config["presentation_order"] > 20
    assert config["visible"] is True
    assert config["throughglass_content_carrier"] == "captured_webview"
    assert config["include_carrier_window_in_capture"] is True
    assert config["clip_captured_carrier_to_shell"] is True
    assert config["content_proof_required"] is True
    assert config["mip_blur_strength"] == pytest.approx(0.0)
    assert config["gpu_material_enabled"] == pytest.approx(1.0)
    assert 0.0 < config["gpu_material_opacity"] <= 0.45
    assert config["gpu_material_feather_points"] >= 90.0
    assert config["gpu_material_fill_overscan_points"] <= 8.0
    assert config["gpu_material_base_width_points"] == pytest.approx(config["content_width_points"])
    assert config["gpu_material_base_height_points"] == pytest.approx(config["content_height_points"])
    assert "progress" not in config["optical_field"]
    assert "phase" not in config["optical_field"]


def test_passport_carrier_uses_flat_captured_scene_profile_not_agent_card_lens():
    rest_config = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    for state in ("materialize", "rest", "dismiss"):
        config = compile_perceptasia_primitive_carrier_config(_bounds(), state=state)

        assert config["optical_field"]["profile"] == "captured_scene"
        assert config["core_magnification"] <= 1.01
        assert config["corner_radius_points"] == pytest.approx(52.0)
        assert config["cut_radius_points"] == pytest.approx(52.0)
        assert config["band_width_points"] <= 22.0
        assert config["tail_width_points"] <= 15.0
        assert config["ring_amplitude_points"] <= 18.0
        assert config["exterior_mix_width_points"] <= 47.0

    assert rest_config["band_width_points"] <= 13.0
    assert rest_config["tail_width_points"] <= 8.0
    assert rest_config["ring_amplitude_points"] <= 10.5
    assert rest_config["exterior_mix_width_points"] <= 31.5


def test_passport_summon_and_dismiss_claim_outer_shell_without_warping_payload():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    materialize = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize"
    )
    dismiss = compile_perceptasia_primitive_carrier_config(_bounds(), state="dismiss")

    for config in (materialize, dismiss):
        assert config["core_magnification"] == pytest.approx(rest["core_magnification"])
        assert config["mip_blur_strength"] == pytest.approx(0.0)
        assert config["gpu_material_enabled"] == pytest.approx(1.0)
        assert config["gpu_material_opacity"] >= rest["gpu_material_opacity"]
        assert config["throughglass_content_carrier"] == "captured_webview"

    assert materialize["band_width_points"] > rest["band_width_points"]
    assert materialize["tail_width_points"] > rest["tail_width_points"]
    assert materialize["ring_amplitude_points"] > rest["ring_amplitude_points"]
    assert materialize["exterior_mix_width_points"] > rest["exterior_mix_width_points"]

    assert dismiss["band_width_points"] > rest["band_width_points"]
    assert dismiss["tail_width_points"] > rest["tail_width_points"]
    assert dismiss["ring_amplitude_points"] > rest["ring_amplitude_points"]
    assert dismiss["exterior_mix_width_points"] > rest["exterior_mix_width_points"]


def test_passport_carrier_exposes_cut_radius_as_coupled_shell_contract():
    config = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")

    assert config["cut_radius_points"] == pytest.approx(config["corner_radius_points"])
    assert config["optical_field"]["cut_radius_points"] == pytest.approx(
        config["corner_radius_points"]
    )


def test_passport_hidden_state_unpublishes_carrier_without_assistant_visibility_coupling():
    config = compile_perceptasia_primitive_carrier_config(_bounds(), state="hidden", visible=True)

    assert config["visible"] is False
    assert config["visibility_scope"] == "independent"
    assert config["optical_field"]["visibility_scope"] == "independent"
    assert config["throughglass_content_carrier"] == "captured_webview"
    assert config["include_carrier_window_in_capture"] is True


def test_passport_env_overrides_never_touch_assistant_command_overlay_state():
    overrides = build_perceptasia_primitive_env(
        provider_url="http://localhost:8742",
        content_proof_required=True,
        publish_shell=False,
    )

    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_URL"] == "http://localhost:8742"
    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_REQUIRED"] == "1"
    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL"] == "0"
    assert all(not key.startswith("SPOKE_COMMAND_") for key in overrides)


def test_passport_smoke_env_publishes_through_independent_primitive_shell():
    smoke_env = Path(".spoke-smoke-env").read_text()

    assert 'SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL="1"' in smoke_env
    assert 'SPOKE_RETINA_LASSO_WATCH_TRACE="1"' in smoke_env


def test_passport_smoke_env_targets_live_perceptasia_provider_port():
    smoke_env = Path(".spoke-smoke-env").read_text()

    assert 'SPOKE_PERCEPTASIA_THROUGHGLASS_URL="http://localhost:8753"' in smoke_env
    assert 'SPOKE_PERCEPTASIA_THROUGHGLASS_URL="http://localhost:8742"' not in smoke_env
