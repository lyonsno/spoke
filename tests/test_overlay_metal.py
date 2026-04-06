"""Contracts for the Mecha Visor preview-fill Metal seam."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _import_overlay(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_PREVIEW_FILL", "1")
    sys.modules.pop("spoke.overlay", None)
    return importlib.import_module("spoke.overlay")


def _make_overlay(mod):
    overlay = mod.TranscriptionOverlay.__new__(mod.TranscriptionOverlay)
    overlay._visible = True
    overlay._text_view = MagicMock()
    overlay._text_amplitude = 0.0
    overlay._brightness = 0.0
    overlay._brightness_target = 0.0
    overlay._fill_override_rgb = None
    overlay._fill_override_opacity = None
    overlay._fill_image_brightness = -1.0
    overlay._typewriter_displayed = ""
    overlay._set_text_view_content = MagicMock()
    overlay._fill_layer = MagicMock()
    overlay._fill_layer.opacity.return_value = mod._BG_ALPHA_MIN
    overlay._fill_renderer = MagicMock()
    overlay._content_view = MagicMock()
    overlay._content_view.frame.return_value = SimpleNamespace(
        size=SimpleNamespace(width=mod._OVERLAY_WIDTH, height=mod._OVERLAY_HEIGHT)
    )
    overlay._fill_sdf = object()
    overlay._fill_scale = 2.0
    return overlay


def test_update_fill_image_routes_preview_surface_to_metal_renderer(mock_pyobjc, monkeypatch):
    mod = _import_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)

        def _unexpected(*args, **kwargs):
            raise AssertionError("baked CGImage fill path should not run")

        monkeypatch.setattr(mod, "_glow_fill_alpha", _unexpected)
        monkeypatch.setattr(mod, "_fill_field_to_image", _unexpected)

        overlay._update_fill_image(680.0, 160.0)

        overlay._fill_renderer.set_geometry.assert_called_once_with(
            680.0, 160.0, overlay._fill_sdf, overlay._fill_scale
        )
        overlay._fill_renderer.set_fill_state.assert_called_once()
        overlay._fill_renderer.draw_frame.assert_called_once_with()
        overlay._fill_layer.setContents_.assert_not_called()
    finally:
        sys.modules.pop("spoke.overlay", None)


def test_brightness_chase_refreshes_metal_fill_without_recomputing_sdf(
    mock_pyobjc, monkeypatch
):
    mod = _import_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay._brightness_target = 1.0
        overlay._apply_ridge_masks = MagicMock(
            side_effect=AssertionError("brightness chase should not rebuild SDF")
        )

        overlay.update_text_amplitude(0.5)

        overlay._fill_renderer.set_fill_state.assert_called()
        overlay._fill_renderer.draw_frame.assert_called()
        overlay._apply_ridge_masks.assert_not_called()
    finally:
        sys.modules.pop("spoke.overlay", None)


def test_fill_override_refreshes_metal_fill_without_geometry_rebuild(
    mock_pyobjc, monkeypatch
):
    mod = _import_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay._apply_ridge_masks = MagicMock(
            side_effect=AssertionError("fill override should not rebuild SDF")
        )

        overlay._set_fill_override((0.1, 0.2, 0.3), 0.35)

        overlay._fill_renderer.set_fill_state.assert_called()
        overlay._fill_renderer.draw_frame.assert_called()
        overlay._apply_ridge_masks.assert_not_called()
    finally:
        sys.modules.pop("spoke.overlay", None)


def test_update_fill_image_falls_back_to_baked_fill_when_metal_draw_fails(
    mock_pyobjc, monkeypatch
):
    mod = _import_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay._fill_renderer.draw_frame.return_value = False
        monkeypatch.setattr(mod, "_glow_fill_alpha", lambda *args, **kwargs: "fill-alpha")
        monkeypatch.setattr(mod, "_fill_field_to_image", lambda *args, **kwargs: ("fill-image", "payload"))

        overlay._update_fill_image(680.0, 160.0)

        overlay._fill_renderer.set_geometry.assert_called_once_with(
            680.0, 160.0, overlay._fill_sdf, overlay._fill_scale
        )
        overlay._fill_renderer.draw_frame.assert_called_once_with()
        overlay._fill_layer.setContents_.assert_called_once_with("fill-image")
    finally:
        sys.modules.pop("spoke.overlay", None)
