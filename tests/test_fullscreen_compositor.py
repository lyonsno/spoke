"""Tests for the shared fullscreen compositor adapter."""

import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


class _FakeWindow:
    def __init__(self, window_number):
        self._window_number = window_number

    def windowNumber(self):
        return self._window_number


class _FakeFullScreenCompositor:
    instances = []

    def __init__(self, screen):
        self.screen = screen
        self.started_configs = []
        self.updated_configs = []
        self.excluded_window_ids = []
        self.stop_calls = 0
        self.sampled_configs = []
        self.sampled_brightness = 0.55
        _FakeFullScreenCompositor.instances.append(self)

    def start(self, shell_config):
        self.started_configs.append(dict(shell_config))
        return True

    def update_shell_configs(self, shell_configs):
        self.updated_configs.append([dict(config) for config in shell_configs])

    def set_excluded_window_ids(self, window_ids):
        self.excluded_window_ids = list(window_ids)

    def sample_brightness_for_config(self, shell_config):
        self.sampled_configs.append(dict(shell_config))
        return float(shell_config["initial_brightness"])

    def refresh_brightness(self):
        return None

    def stop(self):
        self.stop_calls += 1


def _reset_fake_compositor(monkeypatch):
    import spoke.fullscreen_compositor as fullscreen_compositor

    _FakeFullScreenCompositor.instances.clear()
    fullscreen_compositor._shared_overlay_hosts.clear()
    monkeypatch.setattr(
        fullscreen_compositor,
        "FullScreenCompositor",
        _FakeFullScreenCompositor,
    )
    return fullscreen_compositor


def _identity(client_id, display_id="display-1", role="assistant"):
    from spoke.fullscreen_compositor import OverlayClientIdentity

    return OverlayClientIdentity(client_id=client_id, display_id=display_id, role=role)


def _snapshot(client_id, generation=1, *, role="assistant", brightness=0.25, z_index=0):
    from spoke.fullscreen_compositor import (
        OpticalShellGeometrySnapshot,
        OpticalShellMaterialSnapshot,
        OverlayRenderSnapshot,
    )

    return OverlayRenderSnapshot(
        identity=_identity(client_id, role=role),
        generation=generation,
        visible=True,
        geometry=OpticalShellGeometrySnapshot(
            center_x=10.0 + generation,
            center_y=20.0,
            content_width_points=300.0,
            content_height_points=80.0,
            corner_radius_points=16.0,
            band_width_points=8.0,
            tail_width_points=12.0,
        ),
        material=OpticalShellMaterialSnapshot(initial_brightness=brightness),
        z_index=z_index,
    )


def test_overlay_render_snapshot_is_immutable(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    client = host.register_client(
        _identity("assistant.command"),
        window=_FakeWindow(101),
        content_view=object(),
    )
    snapshot = _snapshot("assistant.command", brightness=0.41)

    with pytest.raises(FrozenInstanceError):
        snapshot.geometry.center_x = 99.0
    with pytest.raises(FrozenInstanceError):
        snapshot.material.initial_brightness = 0.99

    assert client.publish(snapshot)
    legacy_config = {"center_x": 1.0, "initial_brightness": 0.2}
    client.update_shell_config(legacy_config)
    legacy_config["center_x"] = 999.0

    rendered = host.render_snapshots()[0]
    assert rendered.geometry.center_x == pytest.approx(1.0)


def test_registry_reuses_one_host_per_display_for_distinct_clients(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(201),
        content_view=object(),
    )
    preview = registry.host_for_screen(screen).register_client(
        _identity("preview.transcription", host.display_id, "preview"),
        window=_FakeWindow(202),
        content_view=object(),
    )

    assert registry.host_for_screen(screen) is host
    assert len(_FakeFullScreenCompositor.instances) == 1
    assert assistant.publish(_snapshot("assistant.command", z_index=10))
    assert preview.publish(_snapshot("preview.transcription", role="preview", z_index=5))

    assert [s.identity.client_id for s in host.render_snapshots()] == [
        "preview.transcription",
        "assistant.command",
    ]
    assert [config["client_id"] for config in _FakeFullScreenCompositor.instances[0].updated_configs[-1]] == [
        "preview.transcription",
        "assistant.command",
    ]


def test_release_one_client_keeps_host_running_until_last_client_releases(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(301),
        content_view=object(),
    )
    preview = host.register_client(
        _identity("preview.transcription", host.display_id, "preview"),
        window=_FakeWindow(302),
        content_view=object(),
    )
    assistant.publish(_snapshot("assistant.command"))
    preview.publish(_snapshot("preview.transcription", role="preview"))
    compositor = _FakeFullScreenCompositor.instances[0]

    assistant.release()

    assert compositor.stop_calls == 0
    assert [s.identity.client_id for s in host.render_snapshots()] == ["preview.transcription"]
    assert registry.host_for_screen(screen) is host

    preview.release()

    assert compositor.stop_calls == 1
    assert registry.host_for_screen(screen) is not host


def test_brightness_sampling_uses_requesting_client_snapshot(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    host = fullscreen_compositor.OverlayCompositorRegistry().host_for_screen(object())
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(401),
        content_view=object(),
    )
    preview = host.register_client(
        _identity("preview.transcription", host.display_id, "preview"),
        window=_FakeWindow(402),
        content_view=object(),
    )
    assistant.publish(_snapshot("assistant.command", brightness=0.17))
    preview.publish(_snapshot("preview.transcription", role="preview", brightness=0.83))

    assert preview.sample_brightness() == pytest.approx(0.83)
    assert _FakeFullScreenCompositor.instances[0].sampled_configs[-1]["client_id"] == "preview.transcription"


def test_duplicate_client_id_on_same_display_replaces_snapshot_without_second_host(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    host = registry.host_for_screen(object())
    first = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(501),
        content_view=object(),
    )
    second = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(502),
        content_view=object(),
    )

    assert len(_FakeFullScreenCompositor.instances) == 1
    assert first.identity == second.identity
    first.publish(_snapshot("assistant.command", generation=1, brightness=0.1))
    second.publish(_snapshot("assistant.command", generation=2, brightness=0.9))

    rendered = host.render_snapshots()
    assert len(rendered) == 1
    assert rendered[0].generation == 2
    assert rendered[0].material.initial_brightness == pytest.approx(0.9)


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
