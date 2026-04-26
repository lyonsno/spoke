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
