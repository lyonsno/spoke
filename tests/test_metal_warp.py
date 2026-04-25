from spoke import metal_warp
import pytest


def test_metal_and_ci_warp_tuning_constants_stay_in_sync():
    from spoke import backdrop_stream

    shared_constant_names = [
        "_WARP_BLEED_ZONE_FRAC",
        "_WARP_CENTER_FLOOR",
        "_WARP_FIELD_EXPONENT",
        "_WARP_REMAP_BASE_EXP_SCALE",
        "_WARP_REMAP_BASE_EXP_FLOOR",
        "_WARP_REMAP_RIM_EXP",
        "_WARP_CURVEBOOST_CAP",
        "_WARP_CURVEBOOST_MAG_SCALE",
        "_WARP_CURVEBOOST_RING_DIVISOR",
        "_WARP_CURVEBOOST_RING_CAP",
        "_WARP_SPINE_PROXIMITY_BOOST",
        "_WARP_X_SQUEEZE",
        "_WARP_Y_SQUEEZE",
        "_WARP_EXTERIOR_MAG_STRENGTH",
        "_WARP_EXTERIOR_MAG_DECAY",
    ]

    mismatches = [
        name
        for name in shared_constant_names
        if getattr(metal_warp, name) != getattr(backdrop_stream, name)
    ]

    assert mismatches == []


def test_warp_alias_mip_bias_stays_zero_for_near_identity_warp():
    assert metal_warp._warp_alias_mip_bias(1.0, 1.0) == 0.0
    assert metal_warp._warp_alias_mip_bias(1.08, 0.96) == 0.0


def test_warp_alias_mip_bias_rises_for_violent_warp():
    assert metal_warp._warp_alias_mip_bias(2.6, 1.7) > 0.0
    assert metal_warp._warp_alias_mip_bias(0.38, 0.52) > 0.0


def test_pack_warp_params_uses_shell_specific_bleed_zone_frac():
    payload = metal_warp._pack_warp_params(
        1440.0,
        900.0,
        {
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "bleed_zone_frac": 0.8,
            "exterior_mix_width_points": 20.0,
        },
    )
    values = metal_warp.struct.unpack("21f", payload)
    assert values[17] == pytest.approx(0.8)
    assert values[18] == pytest.approx(20.0)


def test_pack_warp_params_uses_shell_specific_axis_squeeze():
    payload = metal_warp._pack_warp_params(
        1440.0,
        900.0,
        {
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "x_squeeze": 3.25,
            "y_squeeze": 1.2,
        },
    )
    values = metal_warp.struct.unpack("21f", payload)
    assert values[19] == pytest.approx(3.25)
    assert values[20] == pytest.approx(1.2)


def test_pack_warp_params_carries_bidirectional_luminance_window():
    payload = metal_warp._pack_warp_params(
        1440.0,
        900.0,
        {
            "min_brightness": 0.25,
            "max_brightness": 0.42,
            "bleed_zone_frac": 0.8,
        },
    )

    values = metal_warp.struct.unpack("21f", payload)
    assert values[15] == pytest.approx(0.25)
    assert values[16] == pytest.approx(0.42)


def test_metal_shader_applies_luminance_window():
    source = metal_warp._metal_shader_source()

    assert "float maxBrightness" in source
    assert "applyLuminanceWindow" in source


def test_warp_dispatch_box_respects_shell_specific_bleed_zone_frac():
    wide = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "bleed_zone_frac": 0.8,
        },
    )
    tight = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "bleed_zone_frac": 0.4,
        },
    )
    assert tight[0] > wide[0]
    assert tight[1] > wide[1]
    assert tight[2] < wide[2]
    assert tight[3] < wide[3]


def test_warp_dispatch_box_respects_shell_specific_corner_radius():
    rounder = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 60.0,
            "bleed_zone_frac": 0.8,
        },
    )
    squarer = metal_warp._warp_dispatch_box(
        1440.0,
        900.0,
        {
            "center_x": 720.0,
            "center_y": 450.0,
            "content_width_points": 640.0,
            "content_height_points": 120.0,
            "corner_radius_points": 20.0,
            "bleed_zone_frac": 0.8,
        },
    )
    assert squarer[0] > rounder[0]
    assert squarer[1] > rounder[1]
    assert squarer[2] < rounder[2]
    assert squarer[3] < rounder[3]


def test_warp_exterior_mix_weight_keeps_boundary_strength_but_starts_later_with_tighter_width():
    assert metal_warp._warp_exterior_mix_weight(0.0, 40.0) == pytest.approx(1.0)
    assert metal_warp._warp_exterior_mix_weight(0.0, 20.0) == pytest.approx(1.0)
    assert metal_warp._warp_exterior_mix_weight(10.0, 20.0) < metal_warp._warp_exterior_mix_weight(10.0, 40.0)
    assert metal_warp._warp_exterior_mix_weight(30.0, 20.0) == pytest.approx(0.0)
