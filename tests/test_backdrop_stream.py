"""Tests for the ScreenCaptureKit backdrop renderer seam."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_content_local_capture_rect_rebases_global_capture_to_filter_content():
    mod = _import_module()
    content_rect = _make_rect(1440.0, 23.0, 1728.0, 1094.0)
    capture_rect = _make_rect(1560.0, 120.0, 680.0, 160.0)

    rect = mod._content_local_capture_rect(content_rect, capture_rect)

    assert rect.origin.x == pytest.approx(120.0)
    assert rect.origin.y == pytest.approx(97.0)
    assert rect.size.width == pytest.approx(680.0)
    assert rect.size.height == pytest.approx(160.0)


def test_configure_stream_geometry_uses_filter_content_rect_and_point_pixel_scale():
    mod = _import_module()
    content_rect = _make_rect(0.0, 23.0, 1920.0, 1057.0)
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
        content_rect=content_rect,
        capture_rect=capture_rect,
        point_pixel_scale=1.5,
    )

    assert config.calls["width"] == 1020
    assert config.calls["height"] == 240
    assert config.calls["queue_depth"] == 1
    assert config.calls["shows_cursor"] is False
    assert config.calls["source_rect"].origin.x == pytest.approx(480.0)
    assert config.calls["source_rect"].origin.y == pytest.approx(637.0)
    assert config.calls["destination_rect"].origin.x == pytest.approx(0.0)
    assert config.calls["destination_rect"].origin.y == pytest.approx(0.0)


def test_publish_live_image_caches_frame_and_invokes_callback():
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._lock = mod.threading.Lock()
    renderer._latest_image = None
    callback = MagicMock()
    renderer._frame_callback = callback

    renderer._publish_live_image("fresh-frame")

    assert renderer._latest_image == "fresh-frame"
    callback.assert_called_once_with("fresh-frame")


def test_request_stream_start_passes_dedicated_sample_handler_queue(monkeypatch):
    mod = _import_module()
    sentinel_queue = object()
    monkeypatch.setattr(mod, "_make_stream_handler_queue", lambda label: sentinel_queue)

    class FakeDisplay:
        def frame(self):
            return _make_rect(0.0, 0.0, 1728.0, 1117.0)

    fake_display = FakeDisplay()

    class FakeContent:
        def displays(self):
            return [fake_display]

        def windows(self):
            return []

    fake_content = FakeContent()
    captured = {}

    class FakeStream:
        def addStreamOutput_type_sampleHandlerQueue_error_(self, output, output_type, queue, error):
            captured["queue"] = queue
            captured["output_type"] = output_type
            return True, None

        def startCaptureWithCompletionHandler_(self, callback):
            captured["started"] = True

    fake_stream = FakeStream()

    class FakeSCStream:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithFilter_configuration_delegate_(self, content_filter, config, delegate):
            captured["content_filter"] = content_filter
            captured["config"] = config
            return fake_stream

    class FakeSCShareableContent:
        @staticmethod
        def getShareableContentWithCompletionHandler_(callback):
            callback(fake_content)

    class FakeOutput:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithRenderer_(self, renderer):
            captured["output_renderer"] = renderer
            return self

    monkeypatch.setattr(
        mod,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCShareableContent": FakeSCShareableContent,
            "SCStream": FakeSCStream,
            "SCStreamOutputTypeScreen": 7,
        },
    )
    monkeypatch.setattr(mod, "_ScreenCaptureKitStreamOutput", FakeOutput)

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._screen = object()
    renderer._fallback_factory = lambda: None
    renderer._fallback = None
    renderer._stream = None
    renderer._stream_output = None
    renderer._stream_started = False
    renderer._startup_requested = False
    renderer._pending_signature = None
    renderer._applied_signature = None
    renderer._latest_image = None
    renderer._frame_callback = None
    renderer._blur_radius_points = 0.0
    renderer._current_display = None
    renderer._current_display_frame = None
    renderer._current_content = None
    renderer._window_number = None
    renderer._lock = mod.threading.Lock()
    renderer._ci_context = None
    renderer._stream_handler_queue = None
    renderer._match_display = lambda content: fake_display
    renderer._build_filter = lambda content, display, window_number: "filter"
    renderer._build_configuration = lambda content_filter, capture_rect: "config"
    renderer._current_backing_scale = lambda: 2.0
    renderer._signature_for = lambda window_number, capture_rect, backing_scale: ("sig",)

    renderer._request_stream_start(window_number=99, capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0))

    assert captured["queue"] is sentinel_queue
    assert captured["output_type"] == 7
    assert captured["started"] is True
    assert renderer._stream_handler_queue is sentinel_queue
