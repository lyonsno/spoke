from spoke import metal_warp
import pytest


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
            "bleed_zone_frac": 0.4,
        },
    )
    values = metal_warp.struct.unpack("17f", payload)
    assert values[16] == pytest.approx(0.4)


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
