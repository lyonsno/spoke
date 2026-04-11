"""Tests for the ScreenCaptureKit backdrop renderer seam."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _import_module():
    sys.modules.pop("spoke.backdrop_stream", None)
    return importlib.import_module("spoke.backdrop_stream")


def test_make_backdrop_renderer_prefers_screencapturekit_when_available(monkeypatch):
    mod = _import_module()

    class FakeRenderer:
        def __init__(self, screen, fallback_factory):
            self.screen = screen
            self.fallback_factory = fallback_factory

    screen = object()
    fallback_factory = object()
    monkeypatch.setattr(mod, "_screen_capture_kit_available", lambda: True)
    monkeypatch.setattr(mod, "_ScreenCaptureKitBackdropRenderer", FakeRenderer)

    renderer = mod.make_backdrop_renderer(screen, fallback_factory)

    assert isinstance(renderer, FakeRenderer)
    assert renderer.screen is screen
    assert renderer.fallback_factory is fallback_factory


def test_make_backdrop_renderer_falls_back_when_screencapturekit_is_unavailable(
    monkeypatch,
):
    mod = _import_module()
    sentinel = object()
    monkeypatch.setattr(mod, "_screen_capture_kit_available", lambda: False)

    renderer = mod.make_backdrop_renderer(object(), lambda: sentinel)

    assert renderer is sentinel


def test_display_local_capture_rect_rebases_global_capture_to_display():
    mod = _import_module()
    display_frame = _make_rect(1440.0, 0.0, 1728.0, 1117.0)
    capture_rect = _make_rect(1560.0, 120.0, 680.0, 160.0)

    rect = mod._display_local_capture_rect(display_frame, capture_rect)

    assert rect.origin.x == pytest.approx(120.0)
    assert rect.origin.y == pytest.approx(120.0)
    assert rect.size.width == pytest.approx(680.0)
    assert rect.size.height == pytest.approx(160.0)


def test_configure_stream_geometry_uses_bounded_rect_and_backing_scale():
    mod = _import_module()
    display_frame = _make_rect(0.0, 0.0, 1920.0, 1080.0)
    capture_rect = _make_rect(480.0, 660.0, 680.0, 160.0)

    class FakeConfig:
        def __init__(self):
            self.calls = {}

        def setWidth_(self, value):
            self.calls["width"] = value

        def setHeight_(self, value):
            self.calls["height"] = value

        def setQueueDepth_(self, value):
            self.calls["queue_depth"] = value

        def setShowsCursor_(self, value):
            self.calls["shows_cursor"] = value

        def setSourceRect_(self, value):
            self.calls["source_rect"] = value

        def setDestinationRect_(self, value):
            self.calls["destination_rect"] = value

    config = FakeConfig()

    mod._configure_stream_geometry(
        config,
        display_frame=display_frame,
        capture_rect=capture_rect,
        backing_scale=2.0,
    )

    assert config.calls["width"] == 1360
    assert config.calls["height"] == 320
    assert config.calls["queue_depth"] == 3
    assert config.calls["shows_cursor"] is False
    assert config.calls["source_rect"].origin.x == pytest.approx(480.0)
    assert config.calls["source_rect"].origin.y == pytest.approx(660.0)
    assert config.calls["destination_rect"].origin.x == pytest.approx(0.0)
    assert config.calls["destination_rect"].origin.y == pytest.approx(0.0)
