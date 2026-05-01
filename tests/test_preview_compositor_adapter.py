"""Fail-first witnesses for the Glassjaw preview compositor adapter."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


class _FakeScreen:
    def __init__(self):
        self._frame = _make_rect(0.0, 0.0, 1440.0, 900.0)

    def frame(self):
        return self._frame

    def backingScaleFactor(self):
        return 2.0


class _FakeWindow:
    def __init__(self):
        self._frame = _make_rect(100.0, 60.0, 1040.0, 520.0)
        self.alpha = None
        self.ordered_front = False

    def frame(self):
        return self._frame

    def setFrame_display_animate_(self, frame, _display, _animate):
        self._frame = frame

    def windowNumber(self):
        return 701

    def setAlphaValue_(self, alpha):
        self.alpha = alpha

    def orderFrontRegardless(self):
        self.ordered_front = True


class _FakeLayer:
    def __init__(self, frame):
        self._frame = frame
        self._background_color = None
        self._opacity = None
        self.animations = []

    def setFrame_(self, frame):
        if isinstance(frame, tuple):
            origin, size = frame
            self._frame = _make_rect(origin[0], origin[1], size[0], size[1])
        else:
            self._frame = frame

    def frame(self):
        return self._frame

    def setBackgroundColor_(self, color):
        self._background_color = color

    def setOpacity_(self, opacity):
        self._opacity = opacity

    def setContents_(self, contents):
        self.contents = contents

    def setCompositingFilter_(self, filter_name):
        self.compositing_filter = filter_name

    def addAnimation_forKey_(self, animation, key):
        self.animations.append((key, animation))


class _FakeView:
    def __init__(self, frame):
        self._frame = frame
        self._layer = _FakeLayer(frame)

    def frame(self):
        return self._frame

    def setFrame_(self, frame):
        self._frame = frame
        self._layer.setFrame_(frame)

    def layer(self):
        return self._layer


class _FakeTextContainer:
    def setContainerSize_(self, size):
        self.container_size = size


class _FakeLayoutManager:
    def __init__(self, height):
        self.height = height

    def ensureLayoutForTextContainer_(self, container):
        self.container = container

    def usedRectForTextContainer_(self, container):
        return _make_rect(0.0, 0.0, 0.0, self.height)


class _FakeTextView:
    def __init__(self, frame, text=""):
        self._frame = frame
        self._text = text
        self._container = _FakeTextContainer()
        self._layout = _FakeLayoutManager(64.0)
        self.scrolled_range = None

    def frame(self):
        return self._frame

    def setFrame_(self, frame):
        self._frame = frame

    def string(self):
        return self._text

    def setString_(self, text):
        self._text = text

    def textContainer(self):
        return self._container

    def layoutManager(self):
        return self._layout

    def textStorage(self):
        return None

    def scrollRangeToVisible_(self, visible_range):
        self.scrolled_range = visible_range


class _FakeClipView:
    def __init__(self):
        self.origin = SimpleNamespace(x=0.0, y=0.0)

    def bounds(self):
        return SimpleNamespace(origin=self.origin)

    def scrollToPoint_(self, point):
        self.origin = SimpleNamespace(x=point[0], y=point[1])


class _FakeScrollView:
    def __init__(self, frame, text_view):
        self._frame = frame
        self._text_view = text_view
        self._clip_view = _FakeClipView()
        self.hidden = False

    def frame(self):
        return self._frame

    def setFrame_(self, frame):
        self._frame = frame

    def contentView(self):
        return self._clip_view

    def reflectScrolledClipView_(self, clip_view):
        self.reflected_clip_view = clip_view

    def setHidden_(self, hidden):
        self.hidden = hidden


class _FakeClient:
    def __init__(self, identity):
        self.identity = identity
        self.published = []
        self.release_calls = 0

    def publish(self, snapshot):
        self.published.append(snapshot)
        return True

    def release(self):
        self.release_calls += 1


class _FakeHost:
    display_id = "display-42"

    def __init__(self):
        self.registered = []
        self.clients = {}

    def register_client(self, identity, *, window, content_view):
        self.registered.append((identity, window, content_view))
        client = self.clients.get(identity.client_id)
        if client is None:
            client = _FakeClient(identity)
            self.clients[identity.client_id] = client
        return client


class _FakeRegistry:
    def __init__(self, host):
        self.host = host
        self.host_requests = []

    def host_for_screen(self, screen):
        self.host_requests.append(screen)
        return self.host


def _import_overlay_and_compositor(mock_pyobjc):
    sys.modules.pop("spoke.overlay", None)
    sys.modules.pop("spoke.fullscreen_compositor", None)
    overlay_module = importlib.import_module("spoke.overlay")
    compositor_module = importlib.import_module("spoke.fullscreen_compositor")
    overlay_module._start_overlay_fill_worker = lambda work: work()
    return overlay_module, compositor_module


def _make_overlay(overlay_module, monkeypatch):
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)
    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(_make_rect(220.0, 220.0, 600.0, 80.0))
    overlay._text_view = _FakeTextView(_make_rect(0.0, 0.0, 576.0, 64.0))
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, 576.0, 64.0),
        overlay._text_view,
    )
    overlay._fill_layer = _FakeLayer(_make_rect(0.0, 0.0, 1040.0, 520.0))
    overlay._brightness = 0.37
    overlay._brightness_target = 0.37
    return overlay


def test_preview_adapter_publishes_preview_transcription_snapshot_without_starting_second_host(
    mock_pyobjc, monkeypatch
):
    overlay_module, compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)
    host = _FakeHost()
    registry = _FakeRegistry(host)

    def _forbidden_start_overlay_compositor(*_args, **_kwargs):
        raise AssertionError("preview adapter must use the injected registry")

    monkeypatch.setattr(
        compositor_module,
        "start_overlay_compositor",
        _forbidden_start_overlay_compositor,
    )

    overlay.set_compositor_registry(registry)
    assert overlay._publish_preview_compositor_snapshot(visible=True) is True

    assert registry.host_requests == [overlay._screen]
    identity, window, content_view = host.registered[-1]
    assert identity.client_id == "preview.transcription"
    assert identity.role == "preview"
    assert identity.display_id == host.display_id
    assert window is overlay._window
    assert content_view is overlay._content_view

    snapshot = host.clients["preview.transcription"].published[-1]
    assert isinstance(snapshot, compositor_module.OverlayRenderSnapshot)
    assert snapshot.identity is identity
    assert snapshot.visible is True
    assert snapshot.generation == 1
    assert snapshot.excluded_window_ids == (701,)
    assert snapshot.geometry.center_x == pytest.approx(1240.0)
    assert snapshot.geometry.center_y == pytest.approx(1160.0)
    assert snapshot.geometry.content_width_points == pytest.approx(
        (600.0 + overlay_module._PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII * 16.0) * 2.0
    )
    assert snapshot.geometry.content_height_points == pytest.approx(
        (80.0 + overlay_module._PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII * 16.0) * 2.0
    )
    assert snapshot.material.initial_brightness == pytest.approx(0.37)


def test_preview_warp_tuning_updates_visible_shared_snapshot(
    mock_pyobjc, monkeypatch
):
    overlay_module, _compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)
    host = _FakeHost()
    registry = _FakeRegistry(host)

    overlay.set_compositor_registry(registry)
    overlay._visible = True
    assert overlay._publish_preview_compositor_snapshot(visible=True) is True

    overlay.update_preview_warp_tuning(
        x_squeeze=3.25,
        y_squeeze=1.2,
        ring_amplitude_points=13.5,
        bleed_zone_frac=0.62,
        exterior_mix_width_points=33.0,
    )

    snapshot = host.clients["preview.transcription"].published[-1]
    assert snapshot.generation == 2
    assert snapshot.material.x_squeeze == pytest.approx(3.25)
    assert snapshot.material.y_squeeze == pytest.approx(1.2)
    assert snapshot.material.ring_amplitude_points == pytest.approx(13.5)
    assert snapshot.material.bleed_zone_frac == pytest.approx(0.62)
    assert snapshot.material.exterior_mix_width_points == pytest.approx(33.0)


def test_preview_warp_defaults_match_live_tuner_baseline(mock_pyobjc, monkeypatch):
    overlay_module, _compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)

    tuning = overlay.preview_warp_tuning_snapshot()

    assert tuning["core_magnification"] == pytest.approx(2.5)
    assert tuning["x_squeeze"] == pytest.approx(3.203601371951)
    assert tuning["y_squeeze"] == pytest.approx(1.814143483232)
    assert tuning["inflation_x_radii"] == pytest.approx(1.606088033537)
    assert tuning["inflation_y_radii"] == pytest.approx(2.297589557927)
    assert tuning["bleed_zone_frac"] == pytest.approx(0.702946360518)
    assert tuning["exterior_mix_width_points"] == pytest.approx(26.980754573171)
    assert tuning["ring_amplitude_points"] == pytest.approx(35.369188262195)


def test_preview_fill_sdf_body_matches_preview_rect_without_growing_warp(
    mock_pyobjc, monkeypatch
):
    overlay_module, _compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)
    overlay._ridge_scale = 2.0
    observed_sdf = []
    observed_alpha = []

    def _fake_sdf(total_w, total_h, rect_w, rect_h, radius, scale):
        observed_sdf.append((total_w, total_h, rect_w, rect_h, radius, scale))
        return "sdf"

    def _fake_alpha(sdf, *, width, interior_floor):
        observed_alpha.append((sdf, width, interior_floor))
        return "alpha"

    monkeypatch.setattr(overlay_module, "_overlay_rounded_rect_sdf", _fake_sdf)
    monkeypatch.setattr(overlay_module, "_glow_fill_alpha", _fake_alpha)
    monkeypatch.setattr(
        overlay_module,
        "_fill_field_to_image",
        lambda _alpha, _r, _g, _b: ("image", b"payload"),
    )

    overlay._apply_ridge_masks(600.0, 80.0)
    geometry = overlay._preview_compositor_geometry_snapshot()

    assert observed_sdf[-1][2] == pytest.approx(600.0)
    assert observed_sdf[-1][3] == pytest.approx(80.0)
    assert overlay._fill_layer.frame().size.width == pytest.approx(
        600.0 + 2 * overlay_module._OUTER_FEATHER
    )
    assert geometry.content_width_points == pytest.approx(
        (600.0 + overlay_module._PREVIEW_OPTICAL_SHELL_INFLATION_X_RADII * 16.0) * 2.0
    )
    assert geometry.content_height_points == pytest.approx(
        (80.0 + overlay_module._PREVIEW_OPTICAL_SHELL_INFLATION_Y_RADII * 16.0) * 2.0
    )


def test_preview_adapter_hide_releases_only_preview_client_and_leaves_assistant_state(
    mock_pyobjc, monkeypatch
):
    overlay_module, compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)
    host = _FakeHost()
    registry = _FakeRegistry(host)
    assistant_identity = compositor_module.OverlayClientIdentity(
        client_id="assistant.command",
        display_id=host.display_id,
        role="assistant",
    )
    assistant_client = host.register_client(
        assistant_identity,
        window=object(),
        content_view=object(),
    )
    assistant_client.publish("assistant-state")

    overlay.set_compositor_registry(registry)
    assert overlay._publish_preview_compositor_snapshot(visible=True) is True
    assert overlay._publish_preview_compositor_snapshot(visible=False) is True
    overlay._release_preview_compositor_client()

    preview_client = host.clients["preview.transcription"]
    assert preview_client.published[-1].visible is False
    assert preview_client.release_calls == 1
    assert assistant_client.published == ["assistant-state"]
    assert assistant_client.release_calls == 0


def test_tray_mode_stays_overlay_local_and_does_not_register_preview_client(
    mock_pyobjc, monkeypatch
):
    overlay_module, _compositor_module = _import_overlay_and_compositor(mock_pyobjc)
    overlay = _make_overlay(overlay_module, monkeypatch)
    host = _FakeHost()
    registry = _FakeRegistry(host)

    overlay.set_compositor_registry(registry)
    overlay.show_tray("edit this before sending", owner="user")

    assert "preview.transcription" not in host.clients
    assert all(
        identity.role != "tray"
        for identity, _window, _content_view in host.registered
    )
    assert overlay._typewriter_displayed == "edit this before sending"
    assert overlay._visible is True
