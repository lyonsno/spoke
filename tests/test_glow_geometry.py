"""Geometry contracts for the screen-edge glow architecture."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_APPLE_MASK_NOTCH_WIDTHS_14 = [
    386, 380, 376, 376, 374, 372, 372, 372, 372, 372, 372, 370, 370, 370,
    370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370,
    370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370,
    370, 370, 370, 370, 370, 370, 370, 368, 368, 366, 366, 364, 364, 362,
    360, 358, 356, 354, 352, 348, 344, 336,
]

_APPLE_MASK_NOTCH_WIDTHS_16 = [
    386, 380, 378, 376, 374, 372, 372, 372, 372, 372, 372, 372, 372, 372,
    372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372,
    372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372,
    372, 370, 370, 370, 370, 370, 370, 368, 368, 368, 366, 366, 364, 362,
    360, 360, 356, 354, 352, 348, 344, 336,
]


def _rect(x: float, y: float, width: float, height: float):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _center_gap_width(field, row: int) -> int:
    """Measure the contiguous notch gap centered on the display midpoint."""
    width = field.shape[1]
    center = width // 2
    start = center
    while start > 0 and field[row, start - 1] >= 0.0:
        start -= 1
    end = center
    while end + 1 < width and field[row, end + 1] >= 0.0:
        end += 1
    return end - start + 1


def _notch_profile_widths(field, height: int) -> list[int]:
    return [_center_gap_width(field, row) for row in range(height)]


def _shoulder_only_field(mod, geometry: dict):
    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_x = x - (width * 0.5)
    centered_y = (height - y) - (height * 0.5)

    signed = mod._asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y,
        width,
        height,
        geometry["top_radius"],
        geometry["bottom_radius"],
    )
    notch = geometry["notch"]
    assert notch is not None

    notch_center_x = notch["x"] + notch["width"] * 0.5 - width * 0.5
    notch_center_y = notch["y"] + notch["height"] * 0.5 - height * 0.5
    notch_local_x = centered_x - notch_center_x
    notch_local_y = centered_y - notch_center_y
    notch_signed = mod._notch_signed_distance_field(
        notch_local_x,
        notch_local_y,
        notch,
    )
    outline = np.maximum(signed, -notch_signed)

    shoulder_signed = mod._notch_shoulder_distance_field(
        notch_local_x,
        notch_local_y,
        notch,
    )
    helper_outline = np.maximum(signed, -shoulder_signed)
    shoulder_anchor = mod._notch_bottom_half_width(notch)
    helper_shoulder_smoothing = max(
        notch.get("helper_shoulder_smoothing", notch.get("shoulder_smoothing", 0.0)),
        notch.get("shoulder_smoothing", 0.0),
    )
    shoulder_band = helper_shoulder_smoothing * 4.0
    shoulder_proximity = np.maximum(
        shoulder_band - np.abs(np.abs(notch_local_x) - shoulder_anchor),
        0.0,
    ) / shoulder_band
    bottom_edge_y = -notch["height"] * 0.5
    shoulder_vertical = (
        (notch_local_y <= notch["height"] * 0.5)
        & (notch_local_y >= bottom_edge_y - shoulder_band)
    )
    inside_margin = 2.0
    shoulder_zone = (
        (shoulder_proximity > 0.0)
        & (shoulder_vertical > 0.0)
        & ((-outline) > inside_margin)
    )
    softened = outline.copy()
    softened[shoulder_zone] = np.minimum(
        helper_outline[shoulder_zone],
        -inside_margin,
    )
    return softened.astype(np.float32, copy=False)


def test_continuous_glow_pass_specs_use_distance_field_layers(mock_pyobjc):
    """The additive glow should be composed from continuous distance-field passes, not edge/corner tiles."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        specs = mod._continuous_glow_pass_specs()

        assert [spec["name"] for spec in specs] == ["core", "tight_bloom", "wide_bloom"]
        assert all(spec["path_kind"] == "distance_field" for spec in specs)
        assert specs[0]["falloff"] < specs[1]["falloff"] < specs[2]["falloff"]
        assert specs[0]["power"] <= specs[1]["power"] <= specs[2]["power"]
        assert not any("corner" in spec["name"] or "left" in spec["name"] for spec in specs)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_continuous_vignette_pass_specs_use_same_distance_field_architecture(mock_pyobjc):
    """The subtractive vignette should use the same continuous distance-field architecture."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        specs = mod._continuous_vignette_pass_specs()

        assert [spec["name"] for spec in specs] == ["core", "mid", "tail"]
        assert all(spec["path_kind"] == "distance_field" for spec in specs)
        assert specs[0]["falloff"] < specs[1]["falloff"] < specs[2]["falloff"]
        assert specs[0]["power"] <= specs[1]["power"] <= specs[2]["power"]
        assert not any("top" in spec["name"] or "bottom" in spec["name"] for spec in specs)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_derives_notch_from_auxiliary_areas(mock_pyobjc):
    """Live NSScreen auxiliary areas should define the notch cutout instead of a guessed hardcoded width."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1085.0, 767.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(961.0, 1085.0, 767.0, 32.0),
        )

        geometry = mod._display_shape_geometry(screen, 1728.0, 1117.0, 2.0)

        assert geometry["pixel_width"] == 3456
        assert geometry["pixel_height"] == 2234
        assert geometry["notch"] is not None
        assert geometry["notch"]["x"] == pytest.approx(1534.0)
        assert geometry["notch"]["width"] == pytest.approx(388.0)
        assert geometry["notch"]["height"] == pytest.approx(64.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_selects_14_inch_corner_radii(mock_pyobjc):
    """A 14" MacBook Pro (3024×1964 native) should get its own corner radii from the lookup table."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry = mod._display_shape_geometry(screen, 1512.0, 982.0, 2.0)

        assert geometry["pixel_width"] == 3024
        assert geometry["pixel_height"] == 1964
        expected_top, expected_bot = mod._DISPLAY_CORNER_RADII[(3024, 1964)]
        assert geometry["top_radius"] == pytest.approx(expected_top * 2.0)
        assert geometry["bottom_radius"] == pytest.approx(expected_bot * 2.0)
        assert geometry["notch"] is not None
        assert geometry["notch"]["width"] == pytest.approx(384.0)
        assert geometry["notch"]["height"] == pytest.approx(64.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_restores_14_inch_top_corners_after_coupling_smoke(mock_pyobjc):
    """The 14" path should be back on the normal top-corner baseline once the coupling smoke is done."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry = mod._display_shape_geometry(screen, 1512.0, 982.0, 2.0)

        assert geometry["top_radius"] == pytest.approx(20.0)
        assert geometry["bottom_radius"] == pytest.approx(12.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_selects_16_inch_corner_radii(mock_pyobjc):
    """A 16" MacBook Pro (3456×2234 native) should get its own corner radii from the lookup table."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1085.0, 767.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(961.0, 1085.0, 767.0, 32.0),
        )

        geometry = mod._display_shape_geometry(screen, 1728.0, 1117.0, 2.0)

        expected_top, expected_bot = mod._DISPLAY_CORNER_RADII[(3456, 2234)]
        assert geometry["top_radius"] == pytest.approx(expected_top * 2.0)
        assert geometry["bottom_radius"] == pytest.approx(expected_bot * 2.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_restores_16_inch_top_corners_after_coupling_smoke(mock_pyobjc):
    """The 16" M4 Max path should be back on the normal top-corner baseline once the coupling smoke is done."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1085.0, 767.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(961.0, 1085.0, 767.0, 32.0),
        )

        geometry = mod._display_shape_geometry(screen, 1728.0, 1117.0, 2.0)

        assert geometry["top_radius"] == pytest.approx(40.0)
        assert geometry["bottom_radius"] == pytest.approx(16.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_corner_finish_radii_calibrates_16_inch_to_20_8(mock_pyobjc):
    """The 16" finish-field table should track the calibrated 20/8 outer-corner fit."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        assert mod._DISPLAY_CORNER_FINISH_RADII[(3456, 2234)] == pytest.approx((20.0, 8.0))
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_keeps_14_inch_notch_straighter_than_16(mock_pyobjc):
    """The 14" notch should match the exact Apple mask row profile instead of a visually tuned approximation."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None
        field_14 = mod._display_signed_distance_field(geometry_14)

        assert _notch_profile_widths(field_14, 64) == _APPLE_MASK_NOTCH_WIDTHS_14
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_matches_16_inch_apple_mask_profile(mock_pyobjc):
    """The 16" notch should also follow the exact Apple mask profile rather than the legacy rounded cutout."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_16 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1085.0, 767.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(961.0, 1085.0, 767.0, 32.0),
        )

        geometry_16 = mod._display_shape_geometry(screen_16, 1728.0, 1117.0, 2.0)
        assert geometry_16["notch"] is not None
        field_16 = mod._display_signed_distance_field(geometry_16)

        assert _notch_profile_widths(field_16, 64) == _APPLE_MASK_NOTCH_WIDTHS_16
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_keeps_shoulder_smoothing_for_exact_notch_profiles(mock_pyobjc):
    """Exact Apple notch row profiles should still carry shoulder smoothing for the rendered seam path."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None
        assert list(geometry_14["notch"]["profile_widths"]) == _APPLE_MASK_NOTCH_WIDTHS_14
        assert geometry_14["notch"]["shoulder_smoothing"] > 0.0
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_keeps_legacy_helper_strength_for_exact_notch_profiles(mock_pyobjc):
    """Exact Apple notch profiles should still carry the stronger legacy helper geometry for shoulder shaping."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None
        assert geometry_14["notch"]["helper_bottom_radius"] == pytest.approx(16.0)
        assert geometry_14["notch"]["helper_shoulder_smoothing"] == pytest.approx(19.0)
        assert geometry_14["notch"]["helper_shoulder_smoothing"] > geometry_14["notch"]["shoulder_smoothing"]
    finally:
        sys.modules.pop("spoke.glow", None)


def test_softened_notch_shoulders_preserve_exact_14_inch_profile_widths(mock_pyobjc):
    """Shoulder smoothing must preserve the exact Apple-profile perimeter on the rendered field."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None

        softened_field = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )

        assert _notch_profile_widths(softened_field, 64) == _APPLE_MASK_NOTCH_WIDTHS_14
    finally:
        sys.modules.pop("spoke.glow", None)


def test_softened_notch_shoulders_do_not_pull_on_centerline_below_notch(mock_pyobjc):
    """Shoulder smoothing should stay local to the shoulders, not create a center cusp below the notch."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        plain = mod._display_signed_distance_field(geometry_14)
        softened = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )
        center = plain.shape[1] // 2

        for row in (64, 66, 68, 72):
            assert softened[row, center] == pytest.approx(plain[row, center])
    finally:
        sys.modules.pop("spoke.glow", None)


def test_softened_notch_shoulders_change_the_visible_notch_band_not_deep_screen(mock_pyobjc):
    """Shoulder smoothing should affect the notch shoulder band near the top edge, not rows deep in the screen."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        plain = mod._display_signed_distance_field(geometry_14)
        softened = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )
        delta = np.abs(softened - plain)

        assert np.count_nonzero(delta[:128] > 1e-6) > 0
        assert np.count_nonzero(delta[256:] > 1e-6) == 0
    finally:
        sys.modules.pop("spoke.glow", None)


def test_notch_shoulder_helper_uses_continuous_surrogate_not_row_profile(mock_pyobjc):
    """The seam helper should ignore the stepped Apple rows and use the continuous surrogate class."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None

        notch_with_profile = geometry_14["notch"]
        notch_without_profile = dict(notch_with_profile)
        notch_without_profile.pop("profile_widths", None)
        centered_x, centered_y = np.meshgrid(
            np.linspace(-240.0, 240.0, num=17, dtype=np.float32),
            np.linspace(-40.0, 40.0, num=17, dtype=np.float32),
        )

        with_profile = mod._notch_shoulder_distance_field(
            centered_x,
            centered_y,
            notch_with_profile,
        )
        without_profile = mod._notch_shoulder_distance_field(
            centered_x,
            centered_y,
            notch_without_profile,
        )

        assert np.array_equal(with_profile, without_profile)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_notch_top_corner_glow_mask_stays_corner_local(mock_pyobjc):
    """Top-corner glow attenuation should only affect the notch top corners, not deep rows or the center."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        assert geometry_14["notch"] is not None

        mask = mod._notch_top_corner_glow_mask(geometry_14)
        assert mask is not None

        top_half_width = float(geometry_14["notch"]["profile_widths"][0]) * 0.5
        center = mask.shape[1] * 0.5
        left_corner = int(round(center - top_half_width))
        right_corner = int(round(center + top_half_width))
        left_box = mask[:96, max(left_corner - 40, 0): min(left_corner + 80, mask.shape[1])]
        right_box = mask[:96, max(right_corner - 80, 0): min(right_corner + 40, mask.shape[1])]

        assert float(left_box.min()) < 1.0
        assert float(right_box.min()) < 1.0
        assert float(left_box.min()) > 0.0
        assert float(right_box.min()) > 0.0
        assert np.all(mask[128:] == pytest.approx(1.0))
        center_col = mask.shape[1] // 2
        assert np.all(mask[:96, center_col - 8:center_col + 8] == pytest.approx(1.0))
    finally:
        sys.modules.pop("spoke.glow", None)


def test_notch_top_corner_helper_lift_extends_beyond_attenuation_mask(mock_pyobjc):
    """The helper corner lift should reach slightly beyond the dimming mask without leaking deep into the screen."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        atten_mask = mod._notch_top_corner_glow_mask(geometry_14)
        helper_mask = mod._notch_top_corner_glow_mask(
            geometry_14,
            x_band=mod._NOTCH_TOP_CORNER_HELPER_X_BAND,
            y_band=mod._NOTCH_TOP_CORNER_HELPER_Y_BAND,
            attenuation=mod._NOTCH_TOP_CORNER_HELPER_ATTENUATION,
        )
        assert atten_mask is not None
        assert helper_mask is not None

        helper_only = ((1.0 - helper_mask) > 1e-6) & np.isclose(atten_mask, 1.0)
        assert np.count_nonzero(helper_only[:96]) > 0
        assert np.count_nonzero(helper_only[128:]) == 0
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_masks_for_specs_builds_additive_masks_with_notch_corner_attenuation(mock_pyobjc):
    """The additive mask builder should stay runnable when notch-corner attenuation is enabled."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0

        mod._alpha_field_to_image = lambda alpha: ("image", alpha)
        masks = mod._distance_field_masks_for_specs(
            geometry_14,
            mod._continuous_glow_pass_specs(),
        )

        assert [entry["spec"]["name"] for entry in masks] == [
            "core",
            "tight_bloom",
            "wide_bloom",
        ]
        assert all(entry["payload"] is not None for entry in masks)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_masks_for_specs_builds_hold_dimmer_mask(mock_pyobjc):
    """The screen dimmer should get a real SDF mask with the display silhouette."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0

        mod._alpha_field_to_image = lambda alpha: ("image", alpha)
        masks = mod._distance_field_masks_for_specs(
            geometry_14,
            mod._continuous_dimmer_pass_specs(),
        )

        assert [entry["spec"]["name"] for entry in masks] == ["hold_dimmer"]
        assert masks[0]["image"] == "image"
        assert masks[0]["payload"] is not None
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_alphas_for_specs_use_helper_corner_lift_for_additive_passes(mock_pyobjc):
    """Additive corner masks should borrow the continuous helper field rather than only dimming the exact field."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0
        signed_distance = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )
        corner_mask = mod._notch_top_corner_glow_mask(geometry_14)
        assert corner_mask is not None

        top_half_width = float(geometry_14["notch"]["profile_widths"][0]) * 0.5
        center = corner_mask.shape[1] * 0.5
        left_corner = int(round(center - top_half_width))
        right_corner = int(round(center + top_half_width))
        left_box = (
            slice(0, 96),
            slice(max(left_corner - 40, 0), min(left_corner + 80, corner_mask.shape[1])),
        )
        right_box = (
            slice(0, 96),
            slice(max(right_corner - 80, 0), min(right_corner + 40, corner_mask.shape[1])),
        )
        center_box = (
            slice(0, 96),
            slice(int(center) - 8, int(center) + 8),
        )
        deep_interior_box = (
            slice(128, None),
            slice(200, corner_mask.shape[1] - 200),
        )

        alpha_entries = mod._distance_field_alphas_for_specs(
            geometry_14,
            mod._continuous_glow_pass_specs(),
        )

        for entry in alpha_entries:
            spec = entry["spec"]
            alpha = entry["alpha"]
            baseline = mod._distance_field_alpha(
                signed_distance,
                spec["falloff"] * geometry_14["scale"],
                spec["power"],
            )
            baseline = (baseline * corner_mask).astype(np.float32, copy=False)

            assert np.count_nonzero(alpha[left_box] > (baseline[left_box] + 1e-6)) > 0
            assert np.count_nonzero(alpha[right_box] > (baseline[right_box] + 1e-6)) > 0
            assert np.all(alpha[deep_interior_box] == pytest.approx(baseline[deep_interior_box]))
            assert np.all(alpha[center_box] == pytest.approx(baseline[center_box]))
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_alphas_use_legacy_corner_field_not_raw_helper_outline(mock_pyobjc):
    """Corner lift should replay the legacy smoothed helper field, not the raw helper outline."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0
        signed_distance = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )
        corner_mask = mod._notch_top_corner_glow_mask(geometry_14)
        helper_mask = mod._notch_top_corner_glow_mask(
            geometry_14,
            x_band=mod._NOTCH_TOP_CORNER_HELPER_X_BAND,
            y_band=mod._NOTCH_TOP_CORNER_HELPER_Y_BAND,
            attenuation=mod._NOTCH_TOP_CORNER_HELPER_ATTENUATION,
        )
        assert corner_mask is not None
        assert helper_mask is not None

        width = geometry_14["pixel_width"]
        height = geometry_14["pixel_height"]
        x = np.arange(width, dtype=np.float32)[None, :] + 0.5
        y = np.arange(height, dtype=np.float32)[:, None] + 0.5
        centered_x = x - (width * 0.5)
        centered_y = (height - y) - (height * 0.5)
        notch = geometry_14["notch"]
        assert notch is not None
        notch_center_x = notch["x"] + notch["width"] * 0.5 - width * 0.5
        notch_center_y = notch["y"] + notch["height"] * 0.5 - height * 0.5
        notch_local_x = centered_x - notch_center_x
        notch_local_y = centered_y - notch_center_y
        helper_notch_signed = mod._notch_shoulder_distance_field(
            notch_local_x, notch_local_y, notch
        )
        display_signed = mod._asymmetric_rounded_rect_sdf(
            centered_x,
            centered_y,
            width,
            height,
            geometry_14["top_radius"],
            geometry_14["bottom_radius"],
        )
        raw_helper_signed = np.maximum(display_signed, -helper_notch_signed).astype(
            np.float32, copy=False
        )
        helper_corner_lift = (1.0 - helper_mask).astype(np.float32, copy=False)

        alpha_entries = mod._distance_field_alphas_for_specs(
            geometry_14,
            mod._continuous_glow_pass_specs(),
        )

        top_half_width = float(notch["profile_widths"][0]) * 0.5
        center = corner_mask.shape[1] * 0.5
        left_corner = int(round(center - top_half_width))
        left_box = (
            slice(0, 96),
            slice(max(left_corner - 40, 0), min(left_corner + 80, corner_mask.shape[1])),
        )

        for entry in alpha_entries:
            spec = entry["spec"]
            baseline = mod._distance_field_alpha(
                signed_distance,
                spec["falloff"] * geometry_14["scale"],
                spec["power"],
            )
            baseline = (baseline * corner_mask).astype(np.float32, copy=False)
            raw_helper_alpha = mod._distance_field_alpha(
                raw_helper_signed,
                spec["falloff"]
                * geometry_14["scale"]
                * mod._NOTCH_TOP_CORNER_HELPER_FALLOFF_SCALE,
                spec["power"],
            )
            raw_outline_blend = (
                (baseline * (1.0 - helper_corner_lift))
                + (raw_helper_alpha * helper_corner_lift)
            ).astype(np.float32, copy=False)

            assert np.count_nonzero(
                np.abs(entry["alpha"][left_box] - raw_outline_blend[left_box]) > 1e-6
            ) > 0
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_alphas_use_screen_corner_helper_lift_for_additive_passes(mock_pyobjc):
    """Outer screen corners should be able to borrow a softer helper field without affecting the center."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0
        # Pin the helper-lift contract to the normal live outline geometry so
        # temporary smoke-only top-corner overshoots don't erase the mechanism.
        geometry_14["top_radius"] = 20.0
        geometry_14["bottom_radius"] = 12.0
        signed_distance = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )

        alpha_entries = mod._distance_field_alphas_for_specs(
            geometry_14,
            mod._continuous_glow_pass_specs(),
        )

        top_left = (slice(0, 120), slice(0, 140))
        top_right = (slice(0, 120), slice(-140, None))
        bottom_left = (slice(-140, None), slice(0, 140))
        bottom_right = (slice(-140, None), slice(-140, None))
        center_box = (slice(0, 120), slice((geometry_14["pixel_width"] // 2) - 16, (geometry_14["pixel_width"] // 2) + 16))

        for entry in alpha_entries:
            spec = entry["spec"]
            alpha = entry["alpha"]
            baseline = mod._distance_field_alpha(
                signed_distance,
                spec["falloff"] * geometry_14["scale"],
                spec["power"],
            ).astype(np.float32, copy=False)

            assert np.count_nonzero(np.abs(alpha[top_left] - baseline[top_left]) > 1e-6) > 0
            assert np.count_nonzero(np.abs(alpha[top_right] - baseline[top_right]) > 1e-6) > 0
            assert np.count_nonzero(np.abs(alpha[bottom_left] - baseline[bottom_left]) > 1e-6) > 0
            assert np.count_nonzero(np.abs(alpha[bottom_right] - baseline[bottom_right]) > 1e-6) > 0
            assert np.all(alpha[center_box] == pytest.approx(baseline[center_box]))
    finally:
        sys.modules.pop("spoke.glow", None)


def test_distance_field_alphas_use_screen_corner_helper_lift_for_vignette_passes(mock_pyobjc):
    """The softer outer-corner field should shape vignette passes too, not only the additive glow."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen_14 = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 950.0, 660.0, 32.0),
            auxiliaryTopRightArea=lambda: _rect(852.0, 950.0, 660.0, 32.0),
        )

        geometry_14 = mod._display_shape_geometry(screen_14, 1512.0, 982.0, 2.0)
        geometry_14["scale"] = 2.0
        # Pin the helper-lift contract to the normal live outline geometry so
        # temporary smoke-only top-corner overshoots don't erase the mechanism.
        geometry_14["top_radius"] = 20.0
        geometry_14["bottom_radius"] = 12.0
        signed_distance = mod._display_signed_distance_field(
            geometry_14, soften_notch_shoulders=True
        )

        alpha_entries = mod._distance_field_alphas_for_specs(
            geometry_14,
            mod._continuous_vignette_pass_specs(),
        )

        top_left = (slice(0, 120), slice(0, 140))
        center_box = (
            slice(0, 120),
            slice((geometry_14["pixel_width"] // 2) - 16, (geometry_14["pixel_width"] // 2) + 16),
        )

        for entry in alpha_entries:
            spec = entry["spec"]
            alpha = entry["alpha"]
            baseline = mod._distance_field_alpha(
                signed_distance,
                spec["falloff"] * geometry_14["scale"],
                spec["power"],
            ).astype(np.float32, copy=False)

            assert np.count_nonzero(np.abs(alpha[top_left] - baseline[top_left]) > 1e-6) > 0
            assert np.all(alpha[center_box] == pytest.approx(baseline[center_box]))
    finally:
        sys.modules.pop("spoke.glow", None)




def test_soft_display_corner_signed_distance_field_uses_finish_radii(mock_pyobjc):
    """The screen-corner helper field should use the broader finish radii, not the tighter live outline radii."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        geometry = {
            "pixel_width": 200,
            "pixel_height": 120,
            "top_radius": 20.0,
            "bottom_radius": 8.0,
            "notch": None,
        }

        y = np.arange(geometry["pixel_height"], dtype=np.float32)[:, None] + 0.5
        x = np.arange(geometry["pixel_width"], dtype=np.float32)[None, :] + 0.5
        centered_x = x - (geometry["pixel_width"] * 0.5)
        centered_y = (geometry["pixel_height"] - y) - (geometry["pixel_height"] * 0.5)

        helper = mod._soft_display_corner_signed_distance_field(geometry)
        base_top = mod._CORNER_RADIUS_TOP_DEFAULT
        base_bottom = mod._CORNER_RADIUS_BOTTOM_DEFAULT
        top_scale = geometry["top_radius"] / base_top
        bottom_scale = geometry["bottom_radius"] / base_bottom
        finish_top_radius = max(
            geometry["top_radius"],
            mod._CORNER_FINISH_RADIUS_TOP_DEFAULT * top_scale,
        )
        finish_bottom_radius = max(
            geometry["bottom_radius"],
            mod._CORNER_FINISH_RADIUS_BOTTOM_DEFAULT * bottom_scale,
        )
        expected = mod._asymmetric_rounded_rect_sdf(
            centered_x,
            centered_y,
            geometry["pixel_width"],
            geometry["pixel_height"],
            finish_top_radius,
            finish_bottom_radius,
            corner_smoothing=mod._SCREEN_CORNER_HELPER_CORNER_SMOOTHING,
        ).astype(np.float32, copy=False)

        assert np.allclose(helper, expected)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_falls_back_for_unknown_display(mock_pyobjc):
    """An unrecognized display should get the default corner radii."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace()

        geometry = mod._display_shape_geometry(screen, 1920.0, 1080.0, 2.0)

        assert geometry["pixel_width"] == 3840
        assert geometry["pixel_height"] == 2160
        assert geometry["top_radius"] == pytest.approx(mod._CORNER_RADIUS_TOP_DEFAULT * 2.0)
        assert geometry["bottom_radius"] == pytest.approx(mod._CORNER_RADIUS_BOTTOM_DEFAULT * 2.0)
        assert geometry["notch"] is None
    finally:
        sys.modules.pop("spoke.glow", None)
