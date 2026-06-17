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


def test_passport_carrier_captures_live_webview_as_primitive_payload_at_rest():
    config = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")

    assert config["client_id"] == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert config["role"] == "hud"
    assert config["presentation_layer"] == "hud"
    assert config["presentation_order"] > 20
    assert config["visible"] is True
    assert config["throughglass_content_carrier"] == "captured_webview_payload"
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


def test_passport_transition_shell_does_not_capture_live_webview_source_plate():
    for state, progress in (("materialize", 0.45), ("dismiss", 0.45)):
        config = compile_perceptasia_primitive_carrier_config(
            _bounds(), state=state, materialization_progress=progress
        )

        assert config["throughglass_content_carrier"] == "shell_transition_only"
        assert config["include_carrier_window_in_capture"] is False
        assert config["clip_captured_carrier_to_shell"] is False
        assert config["gpu_material_enabled"] == pytest.approx(1.0)


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

    assert rest_config["band_width_points"] <= 22.0
    assert rest_config["tail_width_points"] <= 15.0
    assert rest_config["ring_amplitude_points"] <= 18.0
    assert rest_config["exterior_mix_width_points"] <= 47.0


def test_passport_rest_payload_keeps_perimeter_pressure_after_capture_handoff():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    materialize = compile_perceptasia_primitive_carrier_config(_bounds(), state="materialize")

    assert rest["throughglass_content_carrier"] == "captured_webview_payload"
    assert rest["clip_captured_carrier_to_shell"] is True
    assert rest["mip_blur_strength"] == pytest.approx(0.0)
    assert rest["core_magnification"] == pytest.approx(1.0)

    assert rest["band_width_points"] >= 15.5
    assert rest["tail_width_points"] >= 10.5
    assert rest["ring_amplitude_points"] >= 12.0
    assert rest["exterior_mix_width_points"] >= 35.0

    assert rest["band_width_points"] >= materialize["band_width_points"]
    assert rest["tail_width_points"] >= materialize["tail_width_points"]
    assert rest["ring_amplitude_points"] >= materialize["ring_amplitude_points"]
    assert rest["exterior_mix_width_points"] >= materialize["exterior_mix_width_points"]
    assert rest["gpu_material_opacity"] < materialize["gpu_material_opacity"]


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
        assert config["throughglass_content_carrier"] == "shell_transition_only"
        assert config["include_carrier_window_in_capture"] is False
        assert config["clip_captured_carrier_to_shell"] is False

    assert materialize["band_width_points"] == pytest.approx(rest["band_width_points"])
    assert materialize["tail_width_points"] == pytest.approx(rest["tail_width_points"])
    assert materialize["ring_amplitude_points"] == pytest.approx(rest["ring_amplitude_points"])
    assert materialize["exterior_mix_width_points"] == pytest.approx(rest["exterior_mix_width_points"])

    assert dismiss["band_width_points"] >= 17.0
    assert dismiss["tail_width_points"] >= 12.0
    assert dismiss["ring_amplitude_points"] >= 14.0
    assert dismiss["exterior_mix_width_points"] >= 38.0


def test_passport_materialization_progress_compiles_transient_pressure_slit_without_public_progress():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    seed = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.0
    )
    mid = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.5
    )
    late = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.9
    )
    final = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=1.0
    )

    assert "progress" not in seed["optical_field"]
    assert "phase" not in seed["optical_field"]
    assert seed["continuous_present"] is True
    assert seed["content_width_points"] < rest["content_width_points"] * 0.20
    assert seed["content_height_points"] < rest["content_height_points"] * 0.08
    assert seed["corner_radius_points"] <= seed["content_height_points"] * 0.5
    assert seed["gpu_material_base_width_points"] == pytest.approx(rest["content_width_points"])
    assert seed["gpu_material_base_height_points"] == pytest.approx(rest["content_height_points"])
    assert 0.0 < seed["gpu_material_height_frac"] <= mid["gpu_material_height_frac"] < late["gpu_material_height_frac"] < 1.0
    assert seed["band_width_points"] < mid["band_width_points"] <= final["band_width_points"]
    assert seed["content_width_points"] < mid["content_width_points"] <= final["content_width_points"]
    assert final["content_width_points"] == pytest.approx(rest["content_width_points"])
    assert final["content_height_points"] == pytest.approx(rest["content_height_points"])


def test_passport_materialization_uses_assistant_style_quick_seam_then_slow_bloom():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    early_seam = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.20
    )
    seam_open = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.25
    )
    gathering = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.62
    )
    bloom = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.94
    )

    assert early_seam["content_width_points"] > rest["content_width_points"] * 0.45
    assert early_seam["content_height_points"] <= rest["content_height_points"] * 0.05
    assert seam_open["content_width_points"] >= rest["content_width_points"] * 0.86
    assert seam_open["content_height_points"] <= rest["content_height_points"] * 0.06
    assert seam_open["gpu_material_opacity"] < 0.005
    assert seam_open["gpu_material_height_frac"] <= 0.02
    assert gathering["content_height_points"] > seam_open["content_height_points"]
    assert gathering["content_height_points"] < rest["content_height_points"] * 0.60
    assert gathering["gpu_material_height_frac"] < 0.12
    assert bloom["content_height_points"] > rest["content_height_points"] * 0.80
    assert bloom["gpu_material_height_frac"] > 0.50
    assert bloom["content_width_points"] >= seam_open["content_width_points"]


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
    assert config["throughglass_content_carrier"] == "shell_transition_only"
    assert config["include_carrier_window_in_capture"] is False
    assert config["clip_captured_carrier_to_shell"] is False


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
