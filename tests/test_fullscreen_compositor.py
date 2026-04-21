"""Tests for shared fullscreen compositor ownership across overlays."""

import importlib
import sys
from types import SimpleNamespace


def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


class _FakeHostCompositor:
    instances = []

    def __init__(self, screen):
        self.screen = screen
        self.started_with = []
        self.updated_with = []
        self.excluded_window_ids = []
        self.stop_calls = 0
        _FakeHostCompositor.instances.append(self)

    def start(self, shell_config):
        self.started_with.append(shell_config)
        return True

    def update_shell_configs(self, shell_configs):
        self.updated_with.append(shell_configs)

    def set_excluded_window_ids(self, window_ids):
        self.excluded_window_ids.append(list(window_ids))

    def reset_temporal_state(self):
        return None

    def stop(self):
        self.stop_calls += 1

    def sample_brightness_for_config(self, config):
        return 0.5


class _FakeWindow:
    def __init__(self, number, x=0.0, y=0.0, width=640.0, height=120.0):
        self._number = number
        self._frame = _make_rect(x, y, width, height)

    def windowNumber(self):
        return self._number

    def frame(self):
        return self._frame


class _FakeContentView:
    def __init__(self, width=640.0, height=120.0):
        self._frame = _make_rect(0.0, 0.0, width, height)

    def frame(self):
        return self._frame


class _FakeScreen:
    def __init__(self, width=1920.0, height=1080.0, scale=2.0):
        self._frame = _make_rect(0.0, 0.0, width, height)
        self._scale = scale

    def frame(self):
        return self._frame

    def backingScaleFactor(self):
        return self._scale


class _FakeCaptureWindow:
    def __init__(self, number):
        self._number = number

    def windowID(self):
        return self._number


class _FakeContentFilterFactory:
    @classmethod
    def alloc(cls):
        return cls()

    def initWithDisplay_excludingWindows_(self, display, excluded_windows):
        return SimpleNamespace(display=display, excluded_windows=list(excluded_windows))


class _FakeStream:
    def __init__(self):
        self.updated_filters = []

    def updateContentFilter_completionHandler_(self, content_filter, completion):
        self.updated_filters.append(content_filter)
        completion(None)


class _FakeStreamConfiguration:
    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def setWidth_(self, width):
        self.width = width

    def setHeight_(self, height):
        self.height = height

    def setQueueDepth_(self, depth):
        self.depth = depth

    def setShowsCursor_(self, shows_cursor):
        self.shows_cursor = shows_cursor

    def setPixelFormat_(self, pixel_format):
        self.pixel_format = pixel_format

    def setContentScale_(self, scale):
        self.scale = scale


class _FakeSCStream:
    @classmethod
    def alloc(cls):
        return cls()

    def initWithFilter_configuration_delegate_(self, content_filter, configuration, delegate):
        self.content_filter = content_filter
        self.configuration = configuration
        self.delegate = delegate
        self.stream_output = None
        self.output_type = None
        self.handler_queue = None
        self.started = False
        return self

    def addStreamOutput_type_sampleHandlerQueue_error_(self, stream_output, output_type, handler_queue, error):
        self.stream_output = stream_output
        self.output_type = output_type
        self.handler_queue = handler_queue
        return True

    def startCaptureWithCompletionHandler_(self, completion):
        self.started = True
        completion(None)


class _FakeCompositorOutput:
    @classmethod
    def alloc(cls):
        return cls()

    def initWithRenderer_(self, renderer):
        self.renderer = renderer
        return self


class TestSharedFullscreenCompositor:
    def test_same_screen_overlays_share_one_fullscreen_host(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        monkeypatch.setattr(mod, "FullScreenCompositor", _FakeHostCompositor)
        monkeypatch.setattr(mod, "_shared_overlay_hosts", {}, raising=False)
        _FakeHostCompositor.instances.clear()

        screen = _FakeScreen()
        session_a = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(11),
            content_view=_FakeContentView(640.0, 120.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 120.0},
        )
        session_b = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(12, y=180.0),
            content_view=_FakeContentView(640.0, 96.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 96.0},
        )

        assert session_a is not None
        assert session_b is not None
        assert len(_FakeHostCompositor.instances) == 1
        host = _FakeHostCompositor.instances[0]
        assert len(host.started_with) == 1
        assert host.updated_with, "second overlay should update the shared host, not create a new compositor"
        assert len(host.updated_with[-1]) == 2
        assert host.excluded_window_ids[-1] == [11, 12]

    def test_shared_host_only_stops_after_last_overlay_releases(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        monkeypatch.setattr(mod, "FullScreenCompositor", _FakeHostCompositor)
        monkeypatch.setattr(mod, "_shared_overlay_hosts", {}, raising=False)
        _FakeHostCompositor.instances.clear()

        screen = _FakeScreen()
        session_a = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(21),
            content_view=_FakeContentView(640.0, 120.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 120.0},
        )
        session_b = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(22, y=180.0),
            content_view=_FakeContentView(640.0, 96.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 96.0},
        )
        host = _FakeHostCompositor.instances[0]

        session_a.stop()
        assert host.stop_calls == 0, "shared host must survive while another overlay still owns it"
        assert host.excluded_window_ids[-1] == [22]

        session_b.stop()
        assert host.stop_calls == 1

    def test_fullscreen_compositor_refreshes_live_capture_filter_when_exclusions_change(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        compositor = object.__new__(mod.FullScreenCompositor)
        compositor._extra_excluded_ids = set()
        compositor._window = None
        compositor._stream = _FakeStream()
        compositor._capture_content = SimpleNamespace(
            windows=lambda: [_FakeCaptureWindow(11), _FakeCaptureWindow(12), _FakeCaptureWindow(99)]
        )
        compositor._capture_display = object()

        monkeypatch.setattr(
            mod,
            "_load_screencapturekit_bridge",
            lambda: {"SCContentFilter": _FakeContentFilterFactory},
            raising=False,
        )
        monkeypatch.setattr(compositor, "_fetch_shareable_content", lambda bridge: compositor._capture_content)
        monkeypatch.setattr(compositor, "_match_display", lambda content: compositor._capture_display)

        compositor.set_excluded_window_ids([11, 12])

        assert compositor._stream.updated_filters, "live capture filter should refresh when exclusion IDs change"
        excluded_ids = sorted(int(window.windowID()) for window in compositor._stream.updated_filters[-1].excluded_windows)
        assert excluded_ids == [11, 12]

    def test_shared_host_snapshot_preserves_both_clients_across_mixed_updates(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        monkeypatch.setattr(mod, "FullScreenCompositor", _FakeHostCompositor)
        monkeypatch.setattr(mod, "_shared_overlay_hosts", {}, raising=False)
        _FakeHostCompositor.instances.clear()

        screen = _FakeScreen()
        session_a = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(31),
            content_view=_FakeContentView(640.0, 120.0),
            shell_config={
                "content_width_points": 640.0,
                "content_height_points": 120.0,
                "center_x": 320.0,
                "center_y": 240.0,
            },
        )
        session_b = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(32, y=180.0),
            content_view=_FakeContentView(640.0, 96.0),
            shell_config={
                "content_width_points": 640.0,
                "content_height_points": 96.0,
                "center_x": 320.0,
                "center_y": 420.0,
            },
        )
        session_a.update_shell_config(
            {
                "content_width_points": 680.0,
                "content_height_points": 120.0,
                "center_x": 340.0,
                "center_y": 240.0,
            }
        )
        session_b.update_shell_config_key("center_y", 444.0)

        host = mod._shared_overlay_hosts[mod._screen_registry_key(screen)]
        snapshot = host.debug_snapshot()

        assert snapshot["client_count"] == 2
        assert [entry["client_id"] for entry in snapshot["clients"]] == ["overlay:31", "overlay:32"]
        assert snapshot["clients"][0]["content_width_points"] == 680.0
        assert snapshot["clients"][1]["center_y"] == 444.0

    def test_overlay_session_reports_live_shared_client_count(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        monkeypatch.setattr(mod, "FullScreenCompositor", _FakeHostCompositor)
        monkeypatch.setattr(mod, "_shared_overlay_hosts", {}, raising=False)
        _FakeHostCompositor.instances.clear()

        screen = _FakeScreen()
        session_a = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(41),
            content_view=_FakeContentView(640.0, 120.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 120.0},
        )
        session_b = mod.start_overlay_compositor(
            screen=screen,
            window=_FakeWindow(42, y=180.0),
            content_view=_FakeContentView(640.0, 96.0),
            shell_config={"content_width_points": 640.0, "content_height_points": 96.0},
        )

        assert session_a.active_client_count == 2
        assert session_b.active_client_count == 2

        session_b.stop()

        assert session_a.active_client_count == 1
        assert session_b.active_client_count == 0

    def test_fullscreen_compositor_retains_stream_renderer_proxy(self, mock_pyobjc, monkeypatch):
        sys.modules.pop("spoke.fullscreen_compositor", None)
        sys.modules.pop("spoke.backdrop_stream", None)
        mod = importlib.import_module("spoke.fullscreen_compositor")
        backdrop_mod = importlib.import_module("spoke.backdrop_stream")

        compositor = object.__new__(mod.FullScreenCompositor)
        compositor._screen = _FakeScreen()
        compositor._pipeline = object()
        compositor._window = _FakeWindow(77)
        compositor._stream = None
        compositor._stream_output = None
        compositor._stream_handler_queue = None
        compositor._capture_content = None
        compositor._capture_display = None
        compositor._extra_excluded_ids = set()
        compositor._excluded_windows = lambda content: []

        fake_display = SimpleNamespace(frame=lambda: _make_rect(0.0, 0.0, 1728.0, 1117.0))
        fake_content = SimpleNamespace(windows=lambda: [])
        fake_stream = _FakeSCStream.alloc().initWithFilter_configuration_delegate_(None, None, None)

        monkeypatch.setattr(compositor, "_fetch_shareable_content", lambda bridge: fake_content)
        monkeypatch.setattr(compositor, "_match_display", lambda content: fake_display)
        monkeypatch.setattr(backdrop_mod, "_build_stream_output_class", lambda: None, raising=False)
        monkeypatch.setattr(backdrop_mod, "_ScreenCaptureKitStreamOutput", _FakeCompositorOutput, raising=False)
        monkeypatch.setattr(mod, "_load_screencapturekit_bridge", lambda: {
            "SCContentFilter": _FakeContentFilterFactory,
            "SCStreamConfiguration": _FakeStreamConfiguration,
            "SCStream": type("BridgeStream", (), {
                "alloc": classmethod(lambda cls: fake_stream),
            }),
            "SCStreamOutputTypeScreen": 1,
        }, raising=False)
        monkeypatch.setattr(mod, "_make_stream_handler_queue", lambda name: f"queue:{name}", raising=False)

        compositor._start_capture()

        assert compositor._stream_output is fake_stream.stream_output
        assert hasattr(compositor, "_stream_renderer_proxy"), (
            "FullScreenCompositor must retain the renderer proxy it hands to "
            "ScreenCaptureKit so sample-buffer callbacks keep a live Python target."
        )
        assert compositor._stream_renderer_proxy is fake_stream.stream_output.renderer
