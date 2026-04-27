"""Tests for the shared fullscreen compositor adapter."""

import threading
import time
from types import SimpleNamespace


def test_overlay_compositor_session_exposes_refresh_brightness():
    from spoke.fullscreen_compositor import _OverlayCompositorSession

    calls = []

    host = SimpleNamespace(
        refresh_brightness_for_client=lambda client_id: calls.append(client_id),
        sampled_brightness_for_client=lambda client_id: 0.73,
    )
    session = _OverlayCompositorSession(host, "overlay:123")

    session.refresh_brightness()

    assert calls == ["overlay:123"]
    assert session.sampled_brightness == 0.73


def test_fullscreen_compositor_seeds_sampled_brightness_from_shell_config():
    from spoke.fullscreen_compositor import FullScreenCompositor

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._running = False
    compositor._pipeline = SimpleNamespace(reset_temporal_state=lambda: None)
    compositor._sampled_brightness = 0.5
    compositor._create_fullscreen_window = lambda: None
    compositor._start_capture = lambda: None
    compositor._start_display_link = lambda: None

    assert compositor.start({"initial_brightness": 0.08})

    assert compositor.sampled_brightness == 0.08


def test_fullscreen_compositor_start_returns_before_capture_completion():
    from spoke.fullscreen_compositor import FullScreenCompositor

    capture_entered = threading.Event()
    release_capture = threading.Event()

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._running = False
    compositor._pipeline = SimpleNamespace(reset_temporal_state=lambda: None)
    compositor._sampled_brightness = 0.5
    compositor._create_fullscreen_window = lambda: None
    compositor._start_display_link = lambda: None

    def blocking_capture_start():
        capture_entered.set()
        release_capture.wait(timeout=0.5)

    compositor._start_capture = blocking_capture_start

    started_at = time.monotonic()
    try:
        assert compositor.start({"initial_brightness": 0.08})
        elapsed = time.monotonic() - started_at
        assert capture_entered.wait(timeout=0.1)
        assert elapsed < 0.1
    finally:
        release_capture.set()


def test_fullscreen_compositor_skips_display_link_when_frame_and_config_unchanged():
    from spoke.fullscreen_compositor import FullScreenCompositor

    class Layer:
        def __init__(self):
            self.drawable_calls = 0

        def nextDrawable(self):
            self.drawable_calls += 1
            return object()

    pipeline_calls = []
    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._running = True
    compositor._lock = threading.Lock()
    compositor._latest_iosurface = object()
    compositor._latest_width = 100
    compositor._latest_height = 50
    compositor._latest_frame_generation = 1
    compositor._shell_configs = [{"content_width_points": 40, "center_x": 20}]
    compositor._config_generation = 1
    compositor._rendered_frame_generation = 1
    compositor._rendered_config_generation = 1
    compositor._frame_count = 0
    compositor._interval_frame_count = 0
    compositor._interval_presented = 0
    compositor._last_report_time = time.monotonic()
    compositor._last_drawable_size = (100, 50)
    compositor._presented_count = 0
    compositor._metal_layer = Layer()
    compositor._pipeline = SimpleNamespace(
        warp_to_drawable=lambda *args, **kwargs: pipeline_calls.append((args, kwargs)) or True
    )

    compositor._on_display_link()

    assert compositor._metal_layer.drawable_calls == 0
    assert pipeline_calls == []
    assert compositor._frame_count == 0


def test_fullscreen_compositor_renders_when_config_changes_without_new_frame():
    from spoke.fullscreen_compositor import FullScreenCompositor

    class Layer:
        def __init__(self):
            self.drawable_calls = 0
            self.drawable_size = None

        def setDrawableSize_(self, size):
            self.drawable_size = size

        def nextDrawable(self):
            self.drawable_calls += 1
            return object()

    pipeline_calls = []
    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._running = True
    compositor._lock = threading.Lock()
    compositor._latest_iosurface = object()
    compositor._latest_width = 100
    compositor._latest_height = 50
    compositor._latest_frame_generation = 1
    compositor._shell_configs = [{"content_width_points": 40, "center_x": 20}]
    compositor._config_generation = 2
    compositor._rendered_frame_generation = 1
    compositor._rendered_config_generation = 1
    compositor._frame_count = 0
    compositor._interval_frame_count = 0
    compositor._interval_presented = 0
    compositor._last_report_time = time.monotonic()
    compositor._last_drawable_size = (0, 0)
    compositor._presented_count = 0
    compositor._metal_layer = Layer()
    compositor._pipeline = SimpleNamespace(
        warp_to_drawable=lambda *args, **kwargs: pipeline_calls.append((args, kwargs)) or True
    )

    compositor._on_display_link()

    assert compositor._metal_layer.drawable_calls == 1
    assert len(pipeline_calls) == 1
    assert compositor._rendered_frame_generation == 1
    assert compositor._rendered_config_generation == 2
