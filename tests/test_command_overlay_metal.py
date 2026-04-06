"""Contracts for the Mecha Visor command-fill Metal seam."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _import_command_overlay(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_METAL_COMMAND_FILL", "1")
    sys.modules.pop("spoke.command_overlay", None)
    return importlib.import_module("spoke.command_overlay")


def _make_overlay(mod):
    overlay = mod.CommandOverlay.__new__(mod.CommandOverlay)
    overlay._brightness = 0.0
    overlay._brightness_target = 0.0
    overlay._fill_image_brightness = -1.0
    overlay._fill_layer = MagicMock()
    overlay._fill_layer.opacity.return_value = mod._BG_ALPHA
    overlay._fill_renderer = MagicMock()
    overlay._spring_tint_layer = MagicMock()
    overlay._content_view = MagicMock()
    overlay._content_view.frame.return_value = SimpleNamespace(
        size=SimpleNamespace(width=mod._OVERLAY_WIDTH, height=mod._OVERLAY_HEIGHT)
    )
    overlay._fill_sdf = object()
    overlay._fill_scale = 2.0
    overlay._apply_thinking_label_theme = MagicMock()
    return overlay


def test_apply_ridge_masks_routes_command_fill_to_metal_renderer(
    mock_pyobjc, monkeypatch
):
    mod = _import_command_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay_mod = importlib.import_module("spoke.overlay")

        monkeypatch.setattr(overlay_mod, "_overlay_rounded_rect_sdf", lambda *a, **k: object())
        monkeypatch.setattr(overlay_mod, "_glow_fill_alpha", lambda *a, **k: object())
        monkeypatch.setattr(overlay_mod, "_fill_field_to_image", lambda *a, **k: ("fill-image", None))

        overlay._apply_ridge_masks(mod._OVERLAY_WIDTH, mod._OVERLAY_HEIGHT)

        overlay._fill_renderer.set_geometry.assert_called_once()
        overlay._fill_renderer.set_fill_state.assert_called_once()
        overlay._fill_renderer.draw_frame.assert_called_once_with()
        overlay._fill_layer.setContents_.assert_not_called()
        overlay._spring_tint_layer.setMask_.assert_called_once()
    finally:
        sys.modules.pop("spoke.command_overlay", None)


def test_surface_theme_refreshes_metal_fill_without_geometry_rebuild(
    mock_pyobjc, monkeypatch
):
    mod = _import_command_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay._brightness = 1.0
        overlay._fill_image_brightness = 0.0
        overlay._apply_ridge_masks = MagicMock(
            side_effect=AssertionError("brightness theme should not rebuild command SDF")
        )

        overlay._apply_surface_theme()

        overlay._fill_renderer.set_geometry.assert_called_once()
        overlay._fill_renderer.set_fill_state.assert_called_once()
        overlay._fill_renderer.draw_frame.assert_called_once_with()
        overlay._apply_ridge_masks.assert_not_called()
    finally:
        sys.modules.pop("spoke.command_overlay", None)


def test_apply_ridge_masks_falls_back_to_baked_fill_when_metal_draw_fails(
    mock_pyobjc, monkeypatch
):
    mod = _import_command_overlay(mock_pyobjc, monkeypatch)
    try:
        overlay = _make_overlay(mod)
        overlay._fill_renderer.draw_frame.return_value = False
        overlay_mod = importlib.import_module("spoke.overlay")

        monkeypatch.setattr(overlay_mod, "_overlay_rounded_rect_sdf", lambda *a, **k: object())
        monkeypatch.setattr(overlay_mod, "_glow_fill_alpha", lambda *a, **k: object())
        monkeypatch.setattr(overlay_mod, "_fill_field_to_image", lambda *a, **k: ("fill-image", None))

        overlay._apply_ridge_masks(mod._OVERLAY_WIDTH, mod._OVERLAY_HEIGHT)

        overlay._fill_renderer.set_geometry.assert_called_once()
        overlay._fill_renderer.draw_frame.assert_called_once_with()
        overlay._fill_layer.setContents_.assert_called_once_with("fill-image")
    finally:
        sys.modules.pop("spoke.command_overlay", None)
