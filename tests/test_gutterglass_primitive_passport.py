from __future__ import annotations

import pytest

from spoke.gutterglass_primitive_passport import (
    GUTTERGLASS_PRIMITIVE_CLIENT_ID,
    GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID,
    build_gutterglass_primitive_request,
    compile_gutterglass_dismiss_pucker_config,
    compile_gutterglass_primitive_stage_config,
)
from spoke.house_optical_primitive import (
    compile_house_optical_shell_config,
    dismiss_pucker_house_shell_config,
    materialized_house_optical_shell_config,
)
from spoke.optical_field import OpticalFieldBounds


def _bounds() -> OpticalFieldBounds:
    return OpticalFieldBounds(x=180.0, y=120.0, width=1040.0, height=620.0)


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


def test_gutterglass_request_is_independent_hud_consumer_of_house_primitive():
    request = build_gutterglass_primitive_request(_bounds(), state="rest")

    assert request.caller_id == GUTTERGLASS_PRIMITIVE_CLIENT_ID
    assert request.continuity_key == GUTTERGLASS_PRIMITIVE_CLIENT_ID
    assert request.role == "hud"
    assert request.visibility_scope == "independent"
    assert request.presentation.layer == "hud"
    assert request.profile.base == "assistant_shell"
    assert request.layout_recipe == "gutterglass-smoke-stage"
    assert request.motion.strategy == "continuous"
    assert request.motion.continuity == "preserve_identity"


def test_gutterglass_stage_config_uses_extracted_house_primitive():
    expected = compile_house_optical_shell_config(_bounds().width, _bounds().height)

    for state in ("materialize", "rest", "dismiss"):
        config = compile_gutterglass_primitive_stage_config(_bounds(), state=state)

        assert config["client_id"] == GUTTERGLASS_PRIMITIVE_CLIENT_ID
        assert config["role"] == "hud"
        assert config["presentation_layer"] == "hud"
        assert config["visible"] is True
        assert config["optical_field"]["profile"] == "assistant_shell"
        assert config["mip_blur_strength"] == pytest.approx(0.0)
        assert config["gpu_material_enabled"] == pytest.approx(1.0)
        assert config["include_carrier_window_in_capture"] is False
        assert config["clip_captured_carrier_to_shell"] is False
        _assert_house_shell_fields(config, expected)
        assert config["cut_radius_points"] == pytest.approx(config["corner_radius_points"])


def test_gutterglass_materialization_reuses_assistant_zip_curve():
    rest = compile_gutterglass_primitive_stage_config(_bounds(), state="rest")
    expected_seed = materialized_house_optical_shell_config(rest, 0.0)
    expected_mid = materialized_house_optical_shell_config(rest, 0.5)
    seed = compile_gutterglass_primitive_stage_config(
        _bounds(), state="materialize", materialization_progress=0.0
    )
    mid = compile_gutterglass_primitive_stage_config(
        _bounds(), state="materialize", materialization_progress=0.5
    )

    _assert_house_shell_fields(seed, expected_seed)
    _assert_house_shell_fields(mid, expected_mid)
    assert seed["content_width_points"] < rest["content_width_points"] * 0.20
    assert seed["content_height_points"] < rest["content_height_points"] * 0.08
    assert seed["gpu_material_base_width_points"] == pytest.approx(rest["content_width_points"])
    assert seed["gpu_material_base_height_points"] == pytest.approx(rest["content_height_points"])
    assert mid["content_width_points"] > seed["content_width_points"]
    assert mid["content_height_points"] > seed["content_height_points"]


def test_gutterglass_dismiss_pucker_reuses_house_radial_oscillator_without_assistant_identity():
    rest = compile_gutterglass_primitive_stage_config(_bounds(), state="rest")
    expected = dismiss_pucker_house_shell_config(rest, 0.25)

    pucker = compile_gutterglass_dismiss_pucker_config(rest, 0.25)

    assert pucker["client_id"] == GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID
    assert pucker["client_id"] != "assistant.command.dismiss_radial_pucker"
    assert pucker["role"] == "hud"
    assert pucker["warp_mode"] == pytest.approx(2.0)
    assert pucker["scar_amount"] == pytest.approx(expected["scar_amount"])
    assert pucker["content_width_points"] == pytest.approx(expected["content_width_points"])
    assert pucker["content_height_points"] == pytest.approx(expected["content_height_points"])
    assert pucker["corner_radius_points"] == pytest.approx(expected["corner_radius_points"])
