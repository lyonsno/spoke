from types import SimpleNamespace


def test_start_capture_retains_renderer_proxy_until_stop(monkeypatch):
    from spoke import backdrop_stream
    from spoke.fullscreen_compositor import FullScreenCompositor

    class FakeShareableContent:
        @staticmethod
        def getShareableContentWithCompletionHandler_(handler):
            handler(fake_content)

    class FakeContentFilter:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithDisplay_excludingWindows_(self, display, excluded):
            self.display = display
            self.excluded = list(excluded)
            return self

    class FakeStreamConfiguration:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setWidth_(self, value):
            self.width = value

        def setHeight_(self, value):
            self.height = value

        def setQueueDepth_(self, value):
            self.queue_depth = value

        def setShowsCursor_(self, value):
            self.shows_cursor = value

        def setPixelFormat_(self, value):
            self.pixel_format = value

        def setContentScale_(self, value):
            self.content_scale = value

    class FakeStream:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithFilter_configuration_delegate_(self, content_filter, config, delegate):
            self.content_filter = content_filter
            self.config = config
            self.delegate = delegate
            return self

        def addStreamOutput_type_sampleHandlerQueue_error_(self, stream_output, output_type, queue, error):
            self.stream_output = stream_output
            self.output_type = output_type
            self.queue = queue
            return True

        def startCaptureWithCompletionHandler_(self, handler):
            handler(None)

        def stopCaptureWithCompletionHandler_(self, handler):
            self.stopped = True
            handler(None)

    class FakeStreamOutput:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithRenderer_(self, renderer):
            self.renderer = renderer
            return self

    fake_display = SimpleNamespace(
        frame=lambda: SimpleNamespace(size=SimpleNamespace(width=1728.0, height=1117.0))
    )
    fake_content = SimpleNamespace(
        windows=lambda: [],
        displays=lambda: [fake_display],
    )

    monkeypatch.setattr(
        backdrop_stream,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCShareableContent": FakeShareableContent,
            "SCContentFilter": FakeContentFilter,
            "SCStreamConfiguration": FakeStreamConfiguration,
            "SCStream": FakeStream,
            "SCStreamOutputTypeScreen": 0,
        },
    )
    monkeypatch.setattr(backdrop_stream, "_make_stream_handler_queue", lambda _label: object())
    monkeypatch.setattr(backdrop_stream, "_build_stream_output_class", lambda: None)
    monkeypatch.setattr(backdrop_stream, "_ScreenCaptureKitStreamOutput", FakeStreamOutput)

    compositor = object.__new__(FullScreenCompositor)
    compositor._screen = SimpleNamespace(
        backingScaleFactor=lambda: 2.0,
        frame=lambda: SimpleNamespace(size=SimpleNamespace(width=1728.0, height=1117.0)),
    )
    compositor._stream = None
    compositor._stream_output = None
    compositor._stream_renderer_proxy = None
    compositor._stream_handler_queue = None
    compositor._window = None
    compositor._extra_excluded_ids = set()
    compositor._match_display = lambda content: fake_display
    compositor._excluded_windows = lambda content: []

    compositor._start_capture()

    assert compositor._stream is not None
    assert compositor._stream_output is not None
    assert compositor._stream_renderer_proxy is not None
    assert compositor._stream_output.renderer is compositor._stream_renderer_proxy

    compositor._stop_capture()

    assert compositor._stream is None
    assert compositor._stream_output is None
    assert compositor._stream_renderer_proxy is None
