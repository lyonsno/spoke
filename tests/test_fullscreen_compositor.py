"""Tests for the shared fullscreen compositor adapter."""

import sys
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _mock_fullscreen_pyobjc(mock_pyobjc):
    sys.modules.pop("spoke.fullscreen_compositor", None)
    yield
    sys.modules.pop("spoke.fullscreen_compositor", None)


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
        self.presented_count = 0
        self.diagnostics = {
            "capture_fps": 30.0,
            "display_link_ticks": 7,
            "presented_frames": 5,
            "duplicate_frames": 1,
            "skipped_frames": 2,
            "brightness_samples": 3,
            "avg_warp_to_drawable_ms": 0.4,
        }
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

    def diagnostics_snapshot(self):
        return dict(self.diagnostics)


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


def test_snapshot_round_trip_preserves_preview_warp_shape_controls():
    from spoke.fullscreen_compositor import (
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    identity = _identity("preview.transcription", role="preview")
    snapshot = _snapshot_from_shell_config(
        identity,
        {
            "center_x": 123.0,
            "center_y": 456.0,
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "band_width_points": 11.3,
            "tail_width_points": 8.5,
            "initial_brightness": 0.37,
            "bleed_zone_frac": 0.8,
            "exterior_mix_width_points": 20.0,
            "x_squeeze": 2.5,
            "y_squeeze": 1.5,
        },
        generation=7,
    )

    assert snapshot.material.bleed_zone_frac == pytest.approx(0.8)
    assert snapshot.material.exterior_mix_width_points == pytest.approx(20.0)
    assert snapshot.material.x_squeeze == pytest.approx(2.5)
    assert snapshot.material.y_squeeze == pytest.approx(1.5)

    round_trip = _snapshot_to_shell_config(snapshot)
    assert round_trip["bleed_zone_frac"] == pytest.approx(0.8)
    assert round_trip["exterior_mix_width_points"] == pytest.approx(20.0)
    assert round_trip["x_squeeze"] == pytest.approx(2.5)
    assert round_trip["y_squeeze"] == pytest.approx(1.5)


def test_snapshot_round_trip_preserves_shell_mip_blur_strength():
    from spoke.fullscreen_compositor import (
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    identity = _identity("assistant.command", role="assistant")
    snapshot = _snapshot_from_shell_config(
        identity,
        {
            "center_x": 123.0,
            "center_y": 456.0,
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "band_width_points": 11.3,
            "tail_width_points": 8.5,
            "initial_brightness": 0.37,
            "mip_blur_strength": 0.0,
        },
        generation=7,
    )

    assert snapshot.material.mip_blur_strength == pytest.approx(0.0)
    round_trip = _snapshot_to_shell_config(snapshot)
    assert round_trip["mip_blur_strength"] == pytest.approx(0.0)


def test_snapshot_round_trip_preserves_scar_warp_controls():
    from spoke.fullscreen_compositor import (
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    identity = _identity("assistant.command", role="assistant")
    snapshot = _snapshot_from_shell_config(
        identity,
        {
            "center_x": 123.0,
            "center_y": 456.0,
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "band_width_points": 11.3,
            "tail_width_points": 8.5,
            "initial_brightness": 0.37,
            "warp_mode": 1.0,
            "scar_amount": -0.25,
            "scar_seam_length_frac": 0.61,
            "scar_seam_thickness_frac": 0.42,
            "scar_seam_focus_frac": 0.23,
            "scar_vertical_grip": 0.52,
            "scar_horizontal_grip": 0.18,
            "scar_axis_rotation": 1.0,
            "scar_mirrored_lip": 1.0,
        },
        generation=7,
    )

    assert snapshot.material.warp_mode == pytest.approx(1.0)
    assert snapshot.material.scar_amount == pytest.approx(-0.25)
    assert snapshot.material.scar_seam_length_frac == pytest.approx(0.61)
    assert snapshot.material.scar_seam_thickness_frac == pytest.approx(0.42)
    assert snapshot.material.scar_seam_focus_frac == pytest.approx(0.23)
    assert snapshot.material.scar_vertical_grip == pytest.approx(0.52)
    assert snapshot.material.scar_horizontal_grip == pytest.approx(0.18)
    assert snapshot.material.scar_axis_rotation == pytest.approx(1.0)
    assert snapshot.material.scar_mirrored_lip == pytest.approx(1.0)
    round_trip = _snapshot_to_shell_config(snapshot)
    assert round_trip["warp_mode"] == pytest.approx(1.0)
    assert round_trip["scar_amount"] == pytest.approx(-0.25)
    assert round_trip["scar_seam_length_frac"] == pytest.approx(0.61)
    assert round_trip["scar_seam_thickness_frac"] == pytest.approx(0.42)
    assert round_trip["scar_seam_focus_frac"] == pytest.approx(0.23)
    assert round_trip["scar_vertical_grip"] == pytest.approx(0.52)
    assert round_trip["scar_horizontal_grip"] == pytest.approx(0.18)
    assert round_trip["scar_axis_rotation"] == pytest.approx(1.0)
    assert round_trip["scar_mirrored_lip"] == pytest.approx(1.0)


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


def test_host_batches_multi_client_updates_into_one_publish(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    host = registry.host_for_screen(object())
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(201),
        content_view=object(),
    )
    seam = host.register_client(
        _identity("assistant.command.dismiss_seam", host.display_id, "assistant"),
        window=_FakeWindow(201),
        content_view=object(),
    )
    assistant.update_shell_config(
        {"center_x": 100.0, "center_y": 200.0, "content_width_points": 300.0}
    )
    compositor = _FakeFullScreenCompositor.instances[0]
    compositor.updated_configs.clear()

    assert host.update_client_configs(
        {
            "assistant.command": {
                "center_x": 101.0,
                "center_y": 200.0,
                "content_width_points": 300.0,
            },
            "assistant.command.dismiss_seam": {
                "center_x": 101.0,
                "center_y": 200.0,
                "content_width_points": 300.0,
                "z_index": 10,
                "warp_mode": 1.0,
            },
        }
    )

    assert seam is not assistant
    assert len(compositor.updated_configs) == 1
    assert [config["client_id"] for config in compositor.updated_configs[0]] == [
        "assistant.command",
        "assistant.command.dismiss_seam",
    ]
    assert compositor.updated_configs[0][0]["center_x"] == pytest.approx(101.0)
    assert compositor.updated_configs[0][1]["warp_mode"] == pytest.approx(1.0)


def test_shell_config_preserves_agent_thread_card_payload(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    client = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(251),
        content_view=object(),
    )
    cards = [
        {
            "provider_session_id": "codex-thread-1",
            "title": "first thread",
            "readiness": "ready",
            "selected": False,
        }
    ]
    hud = {
        "surface_kind": "agent_shell_partyline",
        "cards": [
            {
                "thread_id": "codex-thread-1",
                "role": "inactive_card",
                "text": "first thread",
                "show_latest_response": False,
                "frame": {"x": 0.0, "y": 0.0, "width": 144.0, "height": 44.0},
            }
        ],
    }

    assert client.update_shell_config(
        {
            "center_x": 10.0,
            "center_y": 20.0,
            "content_width_points": 300.0,
            "content_height_points": 90.0,
            "corner_radius_points": 16.0,
            "band_width_points": 8.0,
            "tail_width_points": 12.0,
            "initial_brightness": 0.4,
            "agent_thread_cards": cards,
            "agent_thread_hud": hud,
            "surface_kind": "agent_shell",
        }
    )

    config = _FakeFullScreenCompositor.instances[0].updated_configs[-1][0]
    assert config["agent_thread_cards"] == cards
    assert config["agent_thread_hud"] == hud
    assert config["surface_kind"] == "agent_shell"


def test_shell_config_preserves_agent_shell_primitive_render_payload(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    client = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(252),
        content_view=object(),
    )
    primitives = [
        {
            "id": "codex-thread-1",
            "kind": "thread_card",
            "provider_session_id": "codex-thread-1",
            "selected": False,
        }
    ]
    renderer = {
        "surface_kind": "agent_shell_card_primitives",
        "cards": [
            {
                "primitive_id": "codex-thread-1",
                "frame": {"x": 12.0, "y": 64.0, "width": 144.0, "height": 44.0},
                "material": {"style": "quiet_chip", "prominence": "inactive"},
            }
        ],
    }
    optical_fields = {
        "surface_kind": "agent_shell_card_optical_fields",
        "requests": [
            {
                "caller_id": "agent.card.codex-thread-1",
                "profile": "agent_card",
            }
        ],
    }

    assert client.update_shell_config(
        {
            "center_x": 10.0,
            "center_y": 20.0,
            "content_width_points": 300.0,
            "content_height_points": 120.0,
            "corner_radius_points": 16.0,
            "band_width_points": 8.0,
            "tail_width_points": 12.0,
            "initial_brightness": 0.4,
            "agent_shell_primitives": primitives,
            "agent_shell_card_renderer": renderer,
            "agent_shell_card_optical_fields": optical_fields,
            "surface_kind": "agent_shell",
        }
    )

    config = _FakeFullScreenCompositor.instances[0].updated_configs[-1][0]
    assert config["agent_shell_primitives"] == primitives
    assert config["agent_shell_card_renderer"] == renderer
    assert config["agent_shell_card_optical_fields"] == optical_fields
    assert config["surface_kind"] == "agent_shell"


def test_host_keeps_agent_shell_card_surfaces_independent_of_parent_motion_and_visibility(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)
    client = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(253),
        content_view=object(),
    )

    def _config(*, center_x, center_y, visible=True):
        return {
            "center_x": center_x,
            "center_y": center_y,
            "visible": visible,
            "content_width_points": 300.0,
            "content_height_points": 120.0,
            "corner_radius_points": 16.0,
            "band_width_points": 8.0,
            "tail_width_points": 12.0,
            "initial_brightness": 0.4,
            "agent_shell_card_optical_fields": {
                "surface_kind": "agent_shell_card_optical_fields",
                "requests": [
                    {
                        "caller_id": "agent.card.codex-thread-1",
                        "compiled_shell_config": {
                            "client_id": "agent.card.codex-thread-1",
                            "role": "agent_card",
                            "center_x": 80.0,
                            "center_y": 90.0,
                            "content_width_points": 144.0,
                            "content_height_points": 44.0,
                            "corner_radius_points": 8.0,
                            "band_width_points": 3.0,
                            "tail_width_points": 2.0,
                            "initial_brightness": 0.4,
                            "z_index": 101,
                            "optical_field": {
                                "caller_id": "agent.card.codex-thread-1",
                                "profile": "agent_card",
                            },
                        },
                        "text": {
                            "primary": "Codex lane",
                            "secondary": "ready",
                            "latest_response": "",
                        },
                    }
                ],
            },
            "surface_kind": "agent_shell",
        }

    assert client.update_shell_config(_config(center_x=10.0, center_y=20.0))

    configs = _FakeFullScreenCompositor.instances[0].updated_configs[-1]
    assert [config["client_id"] for config in configs] == [
        "assistant.command",
        "agent.card.codex-thread-1",
    ]
    assert configs[1]["role"] == "agent_card"
    assert configs[1]["center_x"] == 10.0 - 150.0 + 80.0
    assert configs[1]["center_y"] == 20.0 - 60.0 + 90.0
    assert configs[1]["surface_attachment"] == "sibling"
    assert configs[1]["movable"] is True
    assert configs[1]["optical_field"]["profile"] == "agent_card"
    assert configs[1]["text"]["primary"] == "Codex lane"
    first_card_center = (configs[1]["center_x"], configs[1]["center_y"])

    assert client.update_shell_config(_config(center_x=500.0, center_y=500.0))
    moved_parent_configs = _FakeFullScreenCompositor.instances[0].updated_configs[-1]
    assert moved_parent_configs[0]["center_x"] == 500.0
    assert (
        moved_parent_configs[1]["center_x"],
        moved_parent_configs[1]["center_y"],
    ) == first_card_center

    assert client.update_shell_config(
        _config(center_x=900.0, center_y=900.0, visible=False)
    )
    hidden_parent_configs = _FakeFullScreenCompositor.instances[0].updated_configs[-1]
    assert [config["client_id"] for config in hidden_parent_configs] == [
        "agent.card.codex-thread-1",
    ]
    assert (
        hidden_parent_configs[0]["center_x"],
        hidden_parent_configs[0]["center_y"],
    ) == first_card_center


def test_agent_shell_card_text_overlay_specs_use_card_bounds_and_text_payload(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)

    specs = fullscreen_compositor._agent_shell_card_text_overlay_specs(
        [
            {
                "client_id": "agent.card.codex-thread-1",
                "role": "agent_card",
                "center_x": 400.0,
                "center_y": 240.0,
                "content_width_points": 300.0,
                "content_height_points": 72.0,
                "text": {
                    "primary": "Codex lane",
                    "secondary": "ready",
                },
            },
            {
                "client_id": "assistant.command",
                "role": "assistant",
                "center_x": 400.0,
                "center_y": 240.0,
                "content_width_points": 300.0,
                "content_height_points": 72.0,
                "text": {"primary": "ignored"},
            },
        ],
        screen_width_points=800.0,
        screen_height_points=600.0,
        scale=2.0,
    )

    assert specs == [
        {
            "client_id": "agent.card.codex-thread-1",
            "text": "Codex lane\nready",
            "font_size": 13.0,
            "frame": {
                "x": 265.0,
                "y": 339.0,
                "width": 270.0,
                "height": 42.0,
            },
        }
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


def test_refresh_brightness_uses_requesting_client_snapshot(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    host = fullscreen_compositor.OverlayCompositorRegistry().host_for_screen(object())
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(405),
        content_view=object(),
    )
    preview = host.register_client(
        _identity("preview.transcription", host.display_id, "preview"),
        window=_FakeWindow(406),
        content_view=object(),
    )
    assistant.publish(_snapshot("assistant.command", brightness=0.17))
    preview.publish(_snapshot("preview.transcription", role="preview", brightness=0.83))

    preview.refresh_brightness()

    compositor = _FakeFullScreenCompositor.instances[0]
    assert compositor.sampled_configs
    assert compositor.sampled_configs[-1]["client_id"] == "preview.transcription"


def test_fullscreen_compositor_brightness_uses_latest_pixel_buffer_not_windowserver(monkeypatch):
    import Quartz
    import spoke.fullscreen_compositor as fullscreen_compositor

    from spoke.fullscreen_compositor import FullScreenCompositor

    pixel_buffer = object()
    pixel_calls = []
    windowserver_calls = []

    def sample_pixel_buffer(pb, width, height, config, screen):
        pixel_calls.append((pb, width, height, dict(config), screen))
        return 0.82

    monkeypatch.setattr(
        fullscreen_compositor,
        "_sample_pixel_buffer_brightness",
        sample_pixel_buffer,
        raising=False,
    )
    monkeypatch.setattr(
        Quartz,
        "CGWindowListCreateImage",
        lambda *args: windowserver_calls.append(args) or None,
        raising=False,
    )

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._lock = threading.Lock()
    compositor._screen = object()
    compositor._window = None
    compositor._latest_iosurface = object()
    compositor._latest_pixel_buffer = pixel_buffer
    compositor._latest_width = 120
    compositor._latest_height = 80
    compositor._sampled_brightness = 0.5
    compositor._presented_count = 0

    brightness = compositor.sample_brightness_for_config(
        {
            "client_id": "preview.transcription",
            "center_x": 20.0,
            "content_width_points": 40.0,
        }
    )

    assert brightness == pytest.approx(0.82)
    assert pixel_calls == [
        (
            pixel_buffer,
            120,
            80,
            {
                "client_id": "preview.transcription",
                "center_x": 20.0,
                "content_width_points": 40.0,
            },
            compositor._screen,
        )
    ]
    assert windowserver_calls == []
    diagnostics = compositor.diagnostics_snapshot()
    assert diagnostics["brightness_samples"] == 1
    assert diagnostics["windowserver_brightness_samples"] == 0


def test_client_reports_shared_compositor_presented_count(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    host = fullscreen_compositor.OverlayCompositorRegistry().host_for_screen(object())
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(411),
        content_view=object(),
    )
    assistant.publish(_snapshot("assistant.command", brightness=0.17))
    compositor = _FakeFullScreenCompositor.instances[0]

    compositor.presented_count = 3

    assert host.presented_count == 3
    assert assistant.presented_count == 3


def test_shared_host_and_client_expose_residency_diagnostics(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    host = fullscreen_compositor.OverlayCompositorRegistry().host_for_screen(object())
    assistant = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(421),
        content_view=object(),
    )
    assert assistant.publish(_snapshot("assistant.command", brightness=0.17))

    diagnostics = host.diagnostics_snapshot()
    client_diagnostics = assistant.diagnostics_snapshot()

    assert diagnostics["capture_fps"] == pytest.approx(30.0)
    assert diagnostics["display_link_ticks"] == 7
    assert diagnostics["presented_frames"] == 5
    assert diagnostics["duplicate_frames"] == 1
    assert diagnostics["skipped_frames"] == 2
    assert diagnostics["brightness_samples"] == 3
    assert diagnostics["avg_warp_to_drawable_ms"] == pytest.approx(0.4)
    assert client_diagnostics == diagnostics
    assert host.debug_snapshot()["diagnostics"] == diagnostics


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


def test_duplicate_client_registration_returns_same_live_handle(monkeypatch):
    fullscreen_compositor = _reset_fake_compositor(monkeypatch)
    registry = fullscreen_compositor.OverlayCompositorRegistry()
    screen = object()
    host = registry.host_for_screen(screen)

    first = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(511),
        content_view=object(),
    )
    second = host.register_client(
        _identity("assistant.command", host.display_id, "assistant"),
        window=_FakeWindow(512),
        content_view=object(),
    )

    assert second is first
    assert second.publish(_snapshot("assistant.command", generation=2, brightness=0.9))
    second.release()

    assert _FakeFullScreenCompositor.instances[0].stop_calls == 1
    assert registry.host_for_screen(screen) is not host


def test_legacy_assistant_shell_config_preserves_absent_preview_warp_controls():
    from spoke.fullscreen_compositor import (
        OverlayClientIdentity,
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    identity = OverlayClientIdentity(
        client_id="assistant.command",
        display_id="display-1",
        role="assistant",
    )
    snapshot = _snapshot_from_shell_config(
        identity,
        {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 616.0,
            "content_height_points": 96.0,
            "corner_radius_points": 16.0,
            "core_magnification": 1.55,
            "ring_amplitude_points": 29.48031496063,
            "tail_amplitude_points": 6.377952755906,
        },
        generation=3,
    )

    round_tripped = _snapshot_to_shell_config(snapshot)

    for key in (
        "bleed_zone_frac",
        "exterior_mix_width_points",
        "x_squeeze",
        "y_squeeze",
    ):
        assert key not in round_tripped


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


def test_fullscreen_compositor_renders_duplicate_frames_when_shell_requests_continuity():
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
    compositor._shell_configs = [
        {
            "content_width_points": 40,
            "center_x": 20,
            "continuous_present": True,
        }
    ]
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

    assert compositor._metal_layer.drawable_calls == 1
    assert len(pipeline_calls) == 1
    assert compositor._presented_count == 1


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


def test_fullscreen_compositor_configures_bounded_sck_frame_interval():
    from spoke.fullscreen_compositor import _configure_stream_frame_interval

    class Config:
        def __init__(self):
            self.frame_intervals = []

        def setMinimumFrameInterval_(self, value):
            self.frame_intervals.append(value)

    config = Config()

    _configure_stream_frame_interval(config)

    assert config.frame_intervals == [(1, 30, 0, 0)]


def test_fullscreen_compositor_ignores_duplicate_shell_config_updates():
    from spoke.fullscreen_compositor import FullScreenCompositor

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._lock = threading.Lock()
    compositor._shell_configs = [{"center_x": 12.0, "min_brightness": 0.25}]
    compositor._config_generation = 3

    compositor.update_shell_configs([{"center_x": 12.0, "min_brightness": 0.25}])
    compositor.update_shell_config_key("min_brightness", 0.25)

    assert compositor._config_generation == 3


def test_fullscreen_compositor_records_residency_diagnostics(monkeypatch):
    import spoke.fullscreen_compositor as fullscreen_compositor

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

    class Pipeline:
        def __init__(self):
            self.warp_calls = 0

        def warp_to_drawable(self, *args, **kwargs):
            self.warp_calls += 1
            return True

        def diagnostics_snapshot(self):
            return {
                "drawable_copy_frames": self.warp_calls,
                "drawable_copy_pixels": self.warp_calls * 5_000,
                "mip_generation_frames": self.warp_calls,
                "mip_generation_source_pixels": self.warp_calls * 5_000,
                "warp_dispatch_pixels": self.warp_calls * 231,
            }

    now = [1.0]

    def monotonic():
        now[0] += 0.01
        return now[0]

    monkeypatch.setattr(fullscreen_compositor.time, "monotonic", monotonic)

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._running = True
    compositor._lock = threading.Lock()
    compositor._latest_iosurface = None
    compositor._latest_pixel_buffer = None
    compositor._latest_width = 0
    compositor._latest_height = 0
    compositor._latest_frame_generation = 0
    compositor._shell_configs = [{"content_width_points": 40, "center_x": 20}]
    compositor._config_generation = 1
    compositor._rendered_frame_generation = 0
    compositor._rendered_config_generation = 1
    compositor._frame_count = 0
    compositor._interval_frame_count = 0
    compositor._interval_presented = 0
    compositor._last_report_time = 0.0
    compositor._last_drawable_size = (0, 0)
    compositor._presented_count = 0
    compositor._sampled_brightness = 0.5
    compositor._metal_layer = Layer()
    compositor._pipeline = Pipeline()
    compositor._sample_iosurface_brightness = (
        lambda _iosurface, _w, _h, _config, _pixel_buffer=None: setattr(
            compositor, "_sampled_brightness", 0.7
        )
    )

    compositor.submit_iosurface(object(), width=100, height=50, pixel_buffer=object())
    compositor._on_display_link()
    compositor._on_display_link()
    compositor.submit_iosurface(object(), width=100, height=50, pixel_buffer=object())
    compositor.submit_iosurface(object(), width=100, height=50, pixel_buffer=object())
    compositor._on_display_link()
    assert compositor.sample_brightness_for_config({"initial_brightness": 0.7}) == pytest.approx(0.7)

    diagnostics = compositor.diagnostics_snapshot()

    assert diagnostics["capture_frames"] == 3
    assert diagnostics["capture_fps"] > 0.0
    assert diagnostics["display_link_ticks"] == 3
    assert diagnostics["display_link_fps"] > 0.0
    assert diagnostics["presented_frames"] == 2
    assert diagnostics["presented_fps"] > 0.0
    assert diagnostics["duplicate_frames"] == 1
    assert diagnostics["skipped_frames"] == 1
    assert diagnostics["brightness_samples"] == 1
    assert diagnostics["avg_capture_frame_interval_ms"] > 0.0
    assert diagnostics["avg_compositor_tick_ms"] > 0.0
    assert diagnostics["avg_warp_to_drawable_ms"] > 0.0
    assert diagnostics["avg_brightness_sample_ms"] > 0.0
    assert diagnostics["drawable_copy_frames"] == 2
    assert diagnostics["drawable_copy_pixels"] == 10_000
    assert diagnostics["mip_generation_frames"] == 2
    assert diagnostics["mip_generation_source_pixels"] == 10_000
    assert diagnostics["warp_dispatch_pixels"] == 462


def test_fullscreen_compositor_keeps_residency_diagnostics_when_pipeline_snapshot_fails(monkeypatch):
    import spoke.fullscreen_compositor as fullscreen_compositor

    from spoke.fullscreen_compositor import FullScreenCompositor

    class Pipeline:
        def diagnostics_snapshot(self):
            raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(fullscreen_compositor.logger, "debug", lambda *args, **kwargs: None)

    compositor = FullScreenCompositor.__new__(FullScreenCompositor)
    compositor._lock = threading.Lock()
    compositor._presented_count = 2
    compositor._pipeline = Pipeline()

    diagnostics = compositor.diagnostics_snapshot()

    assert diagnostics["presented_frames"] == 2
    assert diagnostics["capture_frames"] == 0
    assert "mip_generation_frames" not in diagnostics
