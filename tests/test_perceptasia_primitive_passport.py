from __future__ import annotations

import pytest

from spoke.optical_field import OpticalFieldBounds
from spoke.house_optical_primitive import (
    compile_house_optical_shell_config,
    materialized_house_optical_shell_config,
)
from spoke.perceptasia_primitive_passport import (
    PERCEPTASIA_PRIMITIVE_CLIENT_ID,
    build_perceptasia_primitive_env,
    build_perceptasia_primitive_request,
    compile_perceptasia_primitive_carrier_config,
)


def _bounds() -> OpticalFieldBounds:
    return OpticalFieldBounds(x=120.0, y=90.0, width=900.0, height=520.0)


def _assert_house_shell_fields(config: dict[str, object], expected: dict[str, object]) -> None:
    for key in (
        "content_width_points",
        "content_height_points",
        "corner_radius_points",
        "core_magnification",
        "band_width_points",
        "tail_width_points",
        "ring_amplitude_points",
        "tail_amplitude_points",
        "cleanup_blur_radius_points",
    ):
        assert config[key] == pytest.approx(float(expected[key]))


def test_passport_request_is_independent_hud_consumer_of_house_primitive():
    request = build_perceptasia_primitive_request(_bounds(), state="rest")

    assert request.caller_id == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert request.continuity_key == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert request.role == "hud"
    assert request.visibility_scope == "independent"
    assert request.presentation.layer == "hud"
    assert request.presentation.order > 20
    assert request.profile.base == "assistant_shell"
    assert request.layout_recipe == "perceptasia-primitive-passport"
    assert request.motion.strategy == "continuous"
    assert request.motion.continuity == "preserve_identity"
    assert request.selected_handoff is None


def test_passport_carrier_keeps_live_webview_external_at_rest():
    config = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")

    assert config["client_id"] == PERCEPTASIA_PRIMITIVE_CLIENT_ID
    assert config["role"] == "hud"
    assert config["presentation_layer"] == "hud"
    assert config["presentation_order"] > 20
    assert config["visible"] is True
    assert config["throughglass_content_carrier"] == "external_webview"
    assert config["include_carrier_window_in_capture"] is False
    assert config["clip_captured_carrier_to_shell"] is False
    assert config["content_proof_required"] is True
    assert config["mip_blur_strength"] == pytest.approx(0.0)
    assert config["gpu_material_enabled"] == pytest.approx(1.0)
    assert config["gpu_material_opacity"] == pytest.approx(0.0)
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


def test_passport_carrier_uses_extracted_house_primitive_not_sibling_scene_lens():
    expected = compile_house_optical_shell_config(_bounds().width, _bounds().height)
    rest_config = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    for state in ("materialize", "rest", "dismiss"):
        config = compile_perceptasia_primitive_carrier_config(_bounds(), state=state)

        assert config["optical_field"]["profile"] == "assistant_shell"
        _assert_house_shell_fields(config, expected)
        assert config["cut_radius_points"] == pytest.approx(
            config["corner_radius_points"]
        )

    _assert_house_shell_fields(rest_config, expected)


def test_passport_rest_payload_keeps_perimeter_pressure_after_capture_handoff():
    expected = compile_house_optical_shell_config(_bounds().width, _bounds().height)
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    materialize = compile_perceptasia_primitive_carrier_config(_bounds(), state="materialize")

    assert rest["throughglass_content_carrier"] == "external_webview"
    assert rest["include_carrier_window_in_capture"] is False
    assert rest["clip_captured_carrier_to_shell"] is False
    assert rest["mip_blur_strength"] == pytest.approx(0.0)
    _assert_house_shell_fields(rest, expected)

    assert rest["band_width_points"] >= materialize["band_width_points"]
    assert rest["tail_width_points"] >= materialize["tail_width_points"]
    assert rest["ring_amplitude_points"] >= materialize["ring_amplitude_points"]
    assert rest["gpu_material_opacity"] < materialize["gpu_material_opacity"]


def test_passport_external_carrier_rest_is_perimeter_only_not_body_material():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")

    assert rest["throughglass_content_carrier"] == "external_webview"
    assert rest["gpu_material_enabled"] == pytest.approx(1.0)
    assert rest["gpu_material_opacity"] == pytest.approx(0.0)
    assert rest["ring_amplitude_points"] > 0.0
    assert rest["tail_amplitude_points"] > 0.0
    assert rest["band_width_points"] > 0.0


def test_passport_summon_and_dismiss_claim_outer_shell_without_warping_payload():
    expected = compile_house_optical_shell_config(_bounds().width, _bounds().height)
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    materialize = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize"
    )
    dismiss = compile_perceptasia_primitive_carrier_config(_bounds(), state="dismiss")

    for config in (materialize, dismiss):
        assert config["mip_blur_strength"] == pytest.approx(0.0)
        assert config["gpu_material_enabled"] == pytest.approx(1.0)
        assert config["gpu_material_opacity"] >= rest["gpu_material_opacity"]
        assert config["throughglass_content_carrier"] == "shell_transition_only"
        assert config["include_carrier_window_in_capture"] is False
        assert config["clip_captured_carrier_to_shell"] is False

    _assert_house_shell_fields(rest, expected)
    _assert_house_shell_fields(materialize, expected)
    _assert_house_shell_fields(dismiss, expected)


def test_passport_materialization_progress_compiles_transient_pressure_slit_without_public_progress():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    expected_final = materialized_house_optical_shell_config(rest, 1.0)
    seed = compile_perceptasia_primitive_carrier_config(
        _bounds(), state="materialize", materialization_progress=0.0
    )
    expected_seed = materialized_house_optical_shell_config(rest, 0.0)
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
    _assert_house_shell_fields(seed, expected_seed)
    assert seed["content_width_points"] < rest["content_width_points"] * 0.20
    assert seed["content_height_points"] < rest["content_height_points"] * 0.08
    assert seed["corner_radius_points"] <= seed["content_height_points"] * 0.5
    assert seed["gpu_material_base_width_points"] == pytest.approx(rest["content_width_points"])
    assert seed["gpu_material_base_height_points"] == pytest.approx(rest["content_height_points"])
    assert 0.0 < seed["gpu_material_height_frac"] <= mid["gpu_material_height_frac"] < late["gpu_material_height_frac"] < 1.0
    assert seed["band_width_points"] < mid["band_width_points"] <= final["band_width_points"]
    assert seed["content_width_points"] < mid["content_width_points"] <= final["content_width_points"]
    _assert_house_shell_fields(final, expected_final)


def test_passport_materialization_uses_assistant_style_quick_seam_then_slow_bloom():
    rest = compile_perceptasia_primitive_carrier_config(_bounds(), state="rest")
    expected_early = materialized_house_optical_shell_config(rest, 0.20)
    expected_seam = materialized_house_optical_shell_config(rest, 0.25)
    expected_gathering = materialized_house_optical_shell_config(rest, 0.62)
    expected_bloom = materialized_house_optical_shell_config(rest, 0.94)
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

    _assert_house_shell_fields(early_seam, expected_early)
    _assert_house_shell_fields(seam_open, expected_seam)
    _assert_house_shell_fields(gathering, expected_gathering)
    _assert_house_shell_fields(bloom, expected_bloom)
    assert early_seam["content_width_points"] > rest["content_width_points"] * 0.45
    assert early_seam["content_height_points"] <= rest["content_height_points"] * 0.05
    assert seam_open["content_width_points"] >= rest["content_width_points"] * 0.86
    assert seam_open["content_height_points"] <= rest["content_height_points"] * 0.06
    assert seam_open["gpu_material_opacity"] < 0.005
    assert seam_open["gpu_material_height_frac"] <= 0.02
    assert gathering["content_height_points"] > seam_open["content_height_points"]
    assert gathering["content_height_points"] < rest["content_height_points"] * 0.60
    assert gathering["gpu_material_height_frac"] < 0.12
    assert bloom["content_height_points"] > rest["content_height_points"] * 0.75
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


def test_passport_env_builder_publishes_through_independent_primitive_shell():
    overrides = build_perceptasia_primitive_env(
        provider_url="http://localhost:8753",
        content_proof_required=True,
        publish_shell=True,
    )

    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL"] == "1"
    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_REQUIRED"] == "1"


def test_passport_env_builder_targets_live_perceptasia_provider_port():
    overrides = build_perceptasia_primitive_env(
        provider_url="http://localhost:8753/",
        content_proof_required=True,
        publish_shell=True,
    )

    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_URL"] == "http://localhost:8753"
    assert overrides["SPOKE_PERCEPTASIA_PRIMITIVE_URL"] != "http://localhost:8742"
