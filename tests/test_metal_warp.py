import importlib
import sys

from spoke import metal_warp


def test_warp_alias_mip_bias_stays_zero_for_near_identity_warp():
    assert metal_warp._warp_alias_mip_bias(1.0, 1.0) == 0.0
    assert metal_warp._warp_alias_mip_bias(1.08, 0.96) == 0.0


def test_warp_alias_mip_bias_rises_for_violent_warp():
    assert metal_warp._warp_alias_mip_bias(2.6, 1.7) > 0.0
    assert metal_warp._warp_alias_mip_bias(0.38, 0.52) > 0.0


def test_metal_warp_shared_tuning_matches_backdrop_renderer():
    sys.modules.pop("spoke.backdrop_stream", None)
    backdrop_stream = importlib.import_module("spoke.backdrop_stream")
    shared_constants = [
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
    mismatches = {
        name: (
            getattr(backdrop_stream, name),
            getattr(metal_warp, name),
        )
        for name in shared_constants
        if getattr(backdrop_stream, name) != getattr(metal_warp, name)
    }

    assert mismatches == {}
