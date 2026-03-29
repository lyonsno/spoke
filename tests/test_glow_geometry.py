"""Geometry contracts for the screen-edge glow architecture."""

import importlib
import sys
from types import SimpleNamespace

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


def test_continuous_texture_pass_specs_define_first_three_layer_slice(mock_pyobjc):
    """The texture stack should start with the agreed macro, mist, and mesa layers."""
    sys.modules.pop("spoke.glow", None)
    mod = importlib.import_module("spoke.glow")
    try:
        specs = mod._continuous_texture_pass_specs()

        assert [spec["name"] for spec in specs] == ["macro_drift", "ionized_mist", "mesa_breakup"]
        assert [spec["style"] for spec in specs] == ["macro", "mist", "mesa"]
        assert specs[0]["mask_falloff"] > specs[1]["mask_falloff"] > specs[2]["mask_falloff"]
        assert all(spec["additive_alpha"] > spec["subtractive_alpha"] > 0.0 for spec in specs)
        assert all(spec["grid_scale"] > 0.0 for spec in specs)
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
