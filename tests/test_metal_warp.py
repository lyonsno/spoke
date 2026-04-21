from spoke import metal_warp


def test_warp_alias_mip_bias_stays_zero_for_near_identity_warp():
    assert metal_warp._warp_alias_mip_bias(1.0, 1.0) == 0.0
    assert metal_warp._warp_alias_mip_bias(1.08, 0.96) == 0.0


def test_warp_alias_mip_bias_rises_for_violent_warp():
    assert metal_warp._warp_alias_mip_bias(2.6, 1.7) > 0.0
    assert metal_warp._warp_alias_mip_bias(0.38, 0.52) > 0.0


def test_warp_exterior_edge_mip_bias_only_lifts_outer_ring_when_warp_is_strong():
    assert metal_warp._warp_exterior_edge_mip_bias(-4.0, 40.0, 2.4, 1.6) == 0.0
    assert metal_warp._warp_exterior_edge_mip_bias(18.0, 40.0, 1.03, 0.98) == 0.0

    bias = metal_warp._warp_exterior_edge_mip_bias(18.0, 40.0, 2.4, 1.6)
    assert bias > 0.0
    assert bias <= metal_warp._WARP_EXTERIOR_EDGE_MIP_BIAS_MAX
