"""Geometry contracts for the screen-edge glow architecture."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _rect(x: float, y: float, width: float, height: float):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


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
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1101.0, 736.0, 16.0),
            auxiliaryTopRightArea=lambda: _rect(992.0, 1101.0, 736.0, 16.0),
        )

        geometry = mod._display_shape_geometry(screen, 1728.0, 1117.0, 2.0)

        assert geometry["pixel_width"] == 3456
        assert geometry["pixel_height"] == 2234
        assert geometry["notch"] is not None
        assert geometry["notch"]["x"] == pytest.approx(1472.0)
        assert geometry["notch"]["width"] == pytest.approx(512.0)
        assert geometry["notch"]["height"] == pytest.approx(32.0)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_selects_14_inch_corner_radii(mock_pyobjc):
    """A 14" MacBook Pro (3024×1964 native) should get its own corner radii from the lookup table."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 966.0, 624.0, 16.0),
            auxiliaryTopRightArea=lambda: _rect(888.0, 966.0, 624.0, 16.0),
        )

        geometry = mod._display_shape_geometry(screen, 1512.0, 982.0, 2.0)

        assert geometry["pixel_width"] == 3024
        assert geometry["pixel_height"] == 1964
        expected_top, expected_bot = mod._DISPLAY_CORNER_RADII[(3024, 1964)]
        assert geometry["top_radius"] == pytest.approx(expected_top * 2.0)
        assert geometry["bottom_radius"] == pytest.approx(expected_bot * 2.0)
        assert geometry["notch"] is not None
    finally:
        sys.modules.pop("spoke.glow", None)


def test_display_shape_geometry_selects_16_inch_corner_radii(mock_pyobjc):
    """A 16" MacBook Pro (3456×2234 native) should get its own corner radii from the lookup table."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        screen = SimpleNamespace(
            auxiliaryTopLeftArea=lambda: _rect(0.0, 1101.0, 736.0, 16.0),
            auxiliaryTopRightArea=lambda: _rect(992.0, 1101.0, 736.0, 16.0),
        )

        geometry = mod._display_shape_geometry(screen, 1728.0, 1117.0, 2.0)

        expected_top, expected_bot = mod._DISPLAY_CORNER_RADII[(3456, 2234)]
        assert geometry["top_radius"] == pytest.approx(expected_top * 2.0)
        assert geometry["bottom_radius"] == pytest.approx(expected_bot * 2.0)
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


def test_float_pass_composition_preserves_subquantum_alpha_until_final_encode(
    mock_pyobjc,
):
    """Two faint passes should survive if we compose in float before the final 8-bit encode."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        alpha = np.full((1, 1), 0.49 / 255.0, dtype=np.float32)
        rgba = mod._compose_premultiplied_rgba_fields(
            [
                {"alpha": alpha, "rgb": (1.0, 1.0, 1.0), "opacity": 1.0},
                {"alpha": alpha, "rgb": (1.0, 1.0, 1.0), "opacity": 1.0},
            ]
        )
        encoded = mod._encode_premultiplied_rgba_u8(rgba)

        assert encoded[0, 0, 3] == 1
        assert encoded[0, 0, 0] == 1
    finally:
        sys.modules.pop("spoke.glow", None)


def test_pass_composition_matches_source_over_alpha_math(mock_pyobjc):
    """Layer stacking should be resolved in float with standard premultiplied source-over."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        rgba = mod._compose_premultiplied_rgba_fields(
            [
                {"alpha": np.full((1, 1), 0.25, dtype=np.float32), "rgb": (1.0, 0.0, 0.0), "opacity": 1.0},
                {"alpha": np.full((1, 1), 0.5, dtype=np.float32), "rgb": (0.0, 0.0, 1.0), "opacity": 1.0},
            ]
        )

        assert rgba[0, 0, 3] == pytest.approx(0.625)
        assert rgba[0, 0, 0] == pytest.approx(0.125)
        assert rgba[0, 0, 1] == pytest.approx(0.0)
        assert rgba[0, 0, 2] == pytest.approx(0.5)
    finally:
        sys.modules.pop("spoke.glow", None)


def test_split_precision_compare_uses_legacy_left_and_current_right(mock_pyobjc):
    """Split-compare payload should show legacy left, a divider, and highlighted current on the right."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        legacy = np.array(
            [[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]],
            dtype=np.uint8,
        )
        current = np.array(
            [[[11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14]]],
            dtype=np.uint8,
        )

        split = mod._split_precision_compare_encoded(legacy, current)

        assert split[0, 0].tolist() == [1, 1, 1, 1]
        assert split[0, 1].tolist() == [255, 255, 255, 255]
        assert split[0, 2, 1] > current[0, 2, 1]
        assert split[0, 3, 1] > current[0, 3, 1]
    finally:
        sys.modules.pop("spoke.glow", None)
