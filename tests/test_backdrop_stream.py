"""Tests for the ScreenCaptureKit backdrop renderer seam."""

import importlib
import sys
import types
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


def test_publish_live_sample_buffer_invokes_callback():
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    callback = MagicMock()
    renderer._sample_buffer_callback = callback

    renderer._publish_live_sample_buffer("sample-buffer")

    callback.assert_called_once_with("sample-buffer")


def test_consume_sample_buffer_direct_path_skips_image_conversion(monkeypatch):
    mod = _import_module()
    monkeypatch.setitem(sys.modules, "Quartz", types.ModuleType("Quartz"))
    monkeypatch.setattr(
        mod,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCStreamOutputTypeScreen": 7,
        },
    )

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 0.0
    renderer._sample_buffer_callback = MagicMock()
    renderer._publish_live_sample_buffer = MagicMock()
    renderer._publish_live_image = MagicMock()
    renderer._context = MagicMock()

    renderer._consume_sample_buffer("live-sample", 7)

    renderer._publish_live_sample_buffer.assert_called_once_with("live-sample")
    renderer._publish_live_image.assert_not_called()
    renderer._context.assert_not_called()


def test_consume_sample_buffer_blurred_sample_path_skips_image_conversion(monkeypatch):
    mod = _import_module()
    monkeypatch.setattr(
        mod,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCStreamOutputTypeScreen": 7,
        },
    )

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 5.4
    renderer._sample_buffer_callback = MagicMock()
    renderer._frame_callback = None
    renderer._blurred_sample_buffer = MagicMock(return_value="blurred-sample")
    renderer._publish_live_sample_buffer = MagicMock()
    renderer._publish_live_image = MagicMock()
    renderer._context = MagicMock()

    renderer._consume_sample_buffer("live-sample", 7)

    renderer._blurred_sample_buffer.assert_called_once_with("live-sample")
    renderer._publish_live_sample_buffer.assert_called_once_with("blurred-sample")
    renderer._publish_live_image.assert_not_called()
    renderer._context.assert_not_called()


def test_consume_sample_buffer_optical_shell_path_skips_image_conversion(monkeypatch):
    mod = _import_module()
    monkeypatch.setattr(
        mod,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCStreamOutputTypeScreen": 7,
        },
    )

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 0.0
    renderer._sample_buffer_callback = MagicMock()
    renderer._frame_callback = None
    renderer._optical_shell_config = {"enabled": True}
    renderer._optical_shell_sample_buffer = MagicMock(return_value="shell-sample")
    renderer._publish_live_sample_buffer = MagicMock()
    renderer._publish_live_image = MagicMock()
    renderer._context = MagicMock()

    renderer._consume_sample_buffer("live-sample", 7)

    renderer._optical_shell_sample_buffer.assert_called_once_with("live-sample")
    renderer._publish_live_sample_buffer.assert_called_once_with("shell-sample")
    renderer._publish_live_image.assert_not_called()
    renderer._context.assert_not_called()


def test_consume_sample_buffer_blurred_sample_failure_falls_back_to_image_path(monkeypatch):
    mod = _import_module()
    fake_quartz = types.ModuleType("Quartz")

    class FakeCIImage:
        @staticmethod
        def imageWithCVPixelBuffer_(pixel_buffer):
            return SimpleNamespace(
                extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0),
            )

    class FakeCIFilter:
        @staticmethod
        def filterWithName_(name):
            return None

    fake_quartz.CIImage = FakeCIImage
    fake_quartz.CIFilter = FakeCIFilter
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)
    monkeypatch.setattr(
        mod,
        "_load_screencapturekit_bridge",
        lambda: {
            "SCStreamOutputTypeScreen": 7,
            "CMSampleBufferGetImageBuffer": lambda sample_buffer: "pixel-buffer",
        },
    )

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 5.4
    renderer._sample_buffer_callback = MagicMock()
    renderer._frame_callback = None
    renderer._blurred_sample_buffer = MagicMock(return_value=None)
    renderer._publish_live_sample_buffer = MagicMock()
    renderer._publish_live_image = MagicMock()
    context = MagicMock()
    context.createCGImage_fromRect_.return_value = "fallback-image"
    renderer._context = MagicMock(return_value=context)

    renderer._consume_sample_buffer("live-sample", 7)

    renderer._publish_live_sample_buffer.assert_not_called()
    renderer._publish_live_image.assert_called_once_with("fallback-image")


def test_mark_sample_buffer_for_immediate_display_sets_attachment():
    mod = _import_module()
    attachment = {}
    bridge = {
        "CMSampleBufferGetSampleAttachmentsArray": lambda sample_buffer, create_if_necessary: [attachment],
        "kCMSampleAttachmentKey_DisplayImmediately": "display-immediately",
    }

    mod._mark_sample_buffer_for_immediate_display("sample-buffer", bridge)

    assert attachment["display-immediately"] is True


def test_metal_blur_downsample_factor_reads_env(monkeypatch):
    monkeypatch.setenv("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "0.75")
    mod = _import_module()

    assert mod._METAL_BLUR_DOWNSAMPLE == pytest.approx(0.75)


def test_metal_blur_pipeline_renders_into_downsampled_pixel_buffer(monkeypatch):
    monkeypatch.setenv("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "0.75")
    mod = _import_module()

    fake_quartz = types.ModuleType("Quartz")

    class FakeImage:
        def __init__(self, width, height):
            self._extent = _make_rect(0.0, 0.0, width, height)

        def extent(self):
            return self._extent

        def imageByApplyingTransform_(self, transform):
            return FakeImage(
                self._extent.size.width * transform.scale_x,
                self._extent.size.height * transform.scale_y,
            )

        def imageByCroppingToRect_(self, rect):
            return FakeImage(rect.size.width, rect.size.height)

    class FakeCIImage:
        @staticmethod
        def imageWithCVPixelBuffer_(pixel_buffer):
            return FakeImage(680.0, 160.0)

    class FakeBlurFilter:
        def __init__(self):
            self._values = {}

        def setDefaults(self):
            return None

        def setValue_forKey_(self, value, key):
            self._values[key] = value

        def valueForKey_(self, key):
            return self._values["inputImage"]

    class FakeCIFilter:
        @staticmethod
        def filterWithName_(name):
            assert name == "CIGaussianBlur"
            return FakeBlurFilter()

    fake_quartz.CIImage = FakeCIImage
    fake_quartz.CIFilter = FakeCIFilter
    fake_quartz.CGAffineTransformMakeScale = lambda sx, sy: SimpleNamespace(scale_x=sx, scale_y=sy)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    pipeline = mod._MetalBlurPipeline.__new__(mod._MetalBlurPipeline)
    pipeline._context = MagicMock()
    pipeline._create_pixel_buffer = MagicMock(return_value="pixel-buffer-out")
    pipeline._create_format_description = MagicMock(return_value="format-desc")
    pipeline._create_sample_buffer = MagicMock(return_value="blurred-sample")

    bridge = {
        "CMSampleBufferGetImageBuffer": lambda sample_buffer: "pixel-buffer-in",
        "CMSampleBufferGetPresentationTimeStamp": lambda sample_buffer: (1, 60, 1, 0),
        "CMSampleBufferGetDuration": lambda sample_buffer: (1, 60, 1, 0),
    }

    sample = pipeline.blurred_sample_buffer(
        "live-sample",
        blur_radius_points=5.4,
        bridge=bridge,
    )

    assert sample == "blurred-sample"
    pipeline._create_pixel_buffer.assert_called_once_with(510, 120)
    render_extent = pipeline._context.render_toCVPixelBuffer_bounds_colorSpace_.call_args[0][2]
    assert render_extent.size.width == pytest.approx(510.0)
    assert render_extent.size.height == pytest.approx(120.0)


def test_optical_shell_pipeline_uses_warp_kernel(monkeypatch):
    monkeypatch.setenv("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "1.0")
    mod = _import_module()

    fake_quartz = types.ModuleType("Quartz")

    class FakeImage:
        def __init__(self, width, height):
            self._extent = _make_rect(0.0, 0.0, width, height)

        def extent(self):
            return self._extent

        def imageByApplyingTransform_(self, transform):
            return self

        def imageByClampingToExtent(self):
            return self

        def imageByCroppingToRect_(self, rect):
            return self

        @staticmethod
        def imageWithCVPixelBuffer_(pixel_buffer):
            return FakeImage(680.0, 160.0)

        @staticmethod
        def imageWithCGImage_(image):
            return FakeImage(680.0, 160.0)

    fake_quartz.CIImage = FakeImage
    fake_quartz.CIFilter = types.SimpleNamespace(filterWithName_=lambda name: None)
    fake_quartz.CGAffineTransformIdentity = object()
    fake_quartz.CGAffineTransformScale = lambda transform, sx, sy: ("scale", sx, sy)
    fake_quartz.CGAffineTransformTranslate = lambda transform, tx, ty: ("translate", tx, ty)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    pipeline = mod._MetalBlurPipeline.__new__(mod._MetalBlurPipeline)
    pipeline._context = MagicMock()
    pipeline._create_pixel_buffer = MagicMock(return_value="pixel-buffer-out")
    pipeline._create_format_description = MagicMock(return_value="format-desc")
    pipeline._create_sample_buffer = MagicMock(return_value="shell-sample")
    kernel = MagicMock()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.return_value = FakeImage(680.0, 160.0)
    monkeypatch.setattr(mod, "_shell_warp_kernel", lambda: kernel)

    bridge = {
        "CMSampleBufferGetImageBuffer": lambda sample_buffer: "pixel-buffer-in",
        "CMSampleBufferGetPresentationTimeStamp": lambda sample_buffer: (1, 60, 1, 0),
        "CMSampleBufferGetDuration": lambda sample_buffer: (1, 60, 1, 0),
    }

    sample = pipeline.optical_shell_sample_buffer(
        "live-sample",
        shell_config={
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "core_magnification": 2.5,
            "band_width_points": 12.0,
            "tail_width_points": 10.0,
            "ring_amplitude_points": 72.0,
            "tail_amplitude_points": 18.0,
            "cleanup_blur_radius_points": 0.75,
        },
        cleanup_blur_radius_points=0.75,
        bridge=bridge,
    )

    assert sample == "shell-sample"
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.assert_called_once()
    args = kernel.applyWithExtent_roiCallback_inputImage_arguments_.call_args.args[3]
    assert args == pytest.approx(
        [680.0, 160.0, 600.0, 80.0, 16.0, 2.5, 12.0, 10.0, 72.0, 18.0]
    )


def test_optical_shell_debug_visualization_uses_grid_then_warp_kernel(monkeypatch):
    monkeypatch.setenv("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "1.0")
    mod = _import_module()

    fake_quartz = types.ModuleType("Quartz")

    class FakeImage:
        def __init__(self, width, height):
            self._extent = _make_rect(0.0, 0.0, width, height)

        def extent(self):
            return self._extent

        def imageByApplyingTransform_(self, transform):
            return self

        def imageByClampingToExtent(self):
            return self

        def imageByCroppingToRect_(self, rect):
            return self

        @staticmethod
        def imageWithCVPixelBuffer_(pixel_buffer):
            return FakeImage(680.0, 160.0)

        @staticmethod
        def imageWithCGImage_(image):
            return FakeImage(680.0, 160.0)

    fake_quartz.CIImage = FakeImage
    fake_quartz.CIFilter = types.SimpleNamespace(filterWithName_=lambda name: None)
    fake_quartz.CGAffineTransformIdentity = object()
    fake_quartz.CGAffineTransformScale = lambda transform, sx, sy: ("scale", sx, sy)
    fake_quartz.CGAffineTransformTranslate = lambda transform, tx, ty: ("translate", tx, ty)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    pipeline = mod._MetalBlurPipeline.__new__(mod._MetalBlurPipeline)
    pipeline._context = MagicMock()
    pipeline._create_pixel_buffer = MagicMock(return_value="pixel-buffer-out")
    pipeline._create_format_description = MagicMock(return_value="format-desc")
    pipeline._create_sample_buffer = MagicMock(return_value="shell-sample")
    kernel = MagicMock()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.return_value = FakeImage(680.0, 160.0)
    helper = MagicMock(return_value=FakeImage(680.0, 160.0))
    monkeypatch.setattr(mod, "_shell_warp_kernel", lambda: kernel)
    monkeypatch.setattr(mod, "_debug_shell_grid_ci_image", helper, raising=False)

    bridge = {
        "CMSampleBufferGetImageBuffer": lambda sample_buffer: "pixel-buffer-in",
        "CMSampleBufferGetPresentationTimeStamp": lambda sample_buffer: (1, 60, 1, 0),
        "CMSampleBufferGetDuration": lambda sample_buffer: (1, 60, 1, 0),
    }

    sample = pipeline.optical_shell_sample_buffer(
        "live-sample",
        shell_config={
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "core_magnification": 2.5,
            "band_width_points": 12.0,
            "tail_width_points": 10.0,
            "ring_amplitude_points": 72.0,
            "tail_amplitude_points": 18.0,
            "debug_visualize": True,
            "cleanup_blur_radius_points": 0.0,
        },
        cleanup_blur_radius_points=0.0,
        bridge=bridge,
    )

    assert sample == "shell-sample"
    helper.assert_called_once()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.assert_called_once()


def test_optical_shell_debug_visualization_skips_cleanup_blur(monkeypatch):
    monkeypatch.setenv("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "1.0")
    mod = _import_module()

    fake_quartz = types.ModuleType("Quartz")

    class FakeImage:
        def __init__(self, width, height):
            self._extent = _make_rect(0.0, 0.0, width, height)

        def extent(self):
            return self._extent

        def imageByApplyingTransform_(self, transform):
            return self

        def imageByClampingToExtent(self):
            return self

        def imageByCroppingToRect_(self, rect):
            return self

        @staticmethod
        def imageWithCVPixelBuffer_(pixel_buffer):
            return FakeImage(680.0, 160.0)

    blur_filter = MagicMock()
    blur_filter.valueForKey_.return_value = FakeImage(680.0, 160.0)
    fake_quartz.CIImage = FakeImage
    fake_quartz.CIFilter = types.SimpleNamespace(filterWithName_=MagicMock(return_value=blur_filter))
    fake_quartz.CGAffineTransformIdentity = object()
    fake_quartz.CGAffineTransformScale = lambda transform, sx, sy: ("scale", sx, sy)
    fake_quartz.CGAffineTransformTranslate = lambda transform, tx, ty: ("translate", tx, ty)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    pipeline = mod._MetalBlurPipeline.__new__(mod._MetalBlurPipeline)
    pipeline._context = MagicMock()
    pipeline._create_pixel_buffer = MagicMock(return_value="pixel-buffer-out")
    pipeline._create_format_description = MagicMock(return_value="format-desc")
    pipeline._create_sample_buffer = MagicMock(return_value="shell-sample")
    kernel = MagicMock()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.return_value = FakeImage(680.0, 160.0)
    helper = MagicMock(return_value=FakeImage(680.0, 160.0))
    monkeypatch.setattr(mod, "_shell_warp_kernel", lambda: kernel)
    monkeypatch.setattr(mod, "_debug_shell_grid_ci_image", helper, raising=False)

    bridge = {
        "CMSampleBufferGetImageBuffer": lambda sample_buffer: "pixel-buffer-in",
        "CMSampleBufferGetPresentationTimeStamp": lambda sample_buffer: (1, 60, 1, 0),
        "CMSampleBufferGetDuration": lambda sample_buffer: (1, 60, 1, 0),
    }

    sample = pipeline.optical_shell_sample_buffer(
        "live-sample",
        shell_config={
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "ring_amplitude_points": 72.0,
            "tail_amplitude_points": 18.0,
            "debug_visualize": True,
        },
        cleanup_blur_radius_points=0.75,
        bridge=bridge,
    )

    assert sample == "shell-sample"
    helper.assert_called_once()
    fake_quartz.CIFilter.filterWithName_.assert_not_called()


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


def test_capture_blurred_image_seeds_direct_debug_grid_when_visualize_enabled(monkeypatch):
    mod = _import_module()
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
    renderer._sample_buffer_callback = object()
    renderer._blur_radius_points = 0.0
    renderer._optical_shell_config = {"enabled": True, "debug_visualize": True}
    renderer._current_display = None
    renderer._current_display_frame = None
    renderer._current_content = None
    renderer._window_number = None
    renderer._lock = mod.threading.Lock()
    renderer._ci_context = MagicMock()
    renderer._stream_handler_queue = None
    renderer._metal_blur_pipeline_instance = None
    renderer._request_stream_start = MagicMock()
    renderer._update_stream = MagicMock()
    renderer._context = MagicMock(return_value=renderer._ci_context)
    monkeypatch.setattr(
        mod,
        "_debug_shell_grid_ci_image",
        MagicMock(return_value=SimpleNamespace(extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0))),
    )
    renderer._ci_context.createCGImage_fromRect_.return_value = "grid-image"

    image = renderer.capture_blurred_image(
        window_number=99,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=0.75,
    )

    assert image == "grid-image"


def test_capture_blurred_image_debug_visualize_warps_seeded_grid(monkeypatch):
    mod = _import_module()
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
    renderer._sample_buffer_callback = object()
    renderer._blur_radius_points = 0.0
    renderer._optical_shell_config = {
        "enabled": True,
        "debug_visualize": True,
        "content_width_points": 600.0,
        "content_height_points": 80.0,
        "corner_radius_points": 16.0,
        "core_magnification": 2.5,
        "band_width_points": 12.0,
        "tail_width_points": 10.0,
        "ring_amplitude_points": 72.0,
        "tail_amplitude_points": 18.0,
    }
    renderer._current_display = None
    renderer._current_display_frame = None
    renderer._current_content = None
    renderer._window_number = None
    renderer._lock = mod.threading.Lock()
    renderer._ci_context = MagicMock()
    renderer._stream_handler_queue = None
    renderer._metal_blur_pipeline_instance = None
    renderer._request_stream_start = MagicMock()
    renderer._update_stream = MagicMock()
    renderer._context = MagicMock(return_value=renderer._ci_context)

    class FakeImage:
        def __init__(self, width, height):
            self._extent = _make_rect(0.0, 0.0, width, height)

        def extent(self):
            return self._extent

    helper = MagicMock(return_value=FakeImage(680.0, 160.0))
    kernel = MagicMock()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.return_value = FakeImage(680.0, 160.0)
    monkeypatch.setattr(mod, "_debug_shell_grid_ci_image", helper)
    monkeypatch.setattr(mod, "_shell_warp_kernel", lambda: kernel)
    renderer._ci_context.createCGImage_fromRect_.return_value = "warped-grid-image"

    image = renderer.capture_blurred_image(
        window_number=99,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=0.75,
    )

    assert image == "warped-grid-image"
    helper.assert_called_once()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.assert_called_once()


def test_capture_blurred_image_debug_visualize_skips_stream_start(monkeypatch):
    mod = _import_module()
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
    renderer._sample_buffer_callback = object()
    renderer._blur_radius_points = 0.0
    renderer._optical_shell_config = {"enabled": True, "debug_visualize": True}
    renderer._current_display = None
    renderer._current_display_frame = None
    renderer._current_content = None
    renderer._window_number = None
    renderer._lock = mod.threading.Lock()
    renderer._ci_context = MagicMock()
    renderer._stream_handler_queue = None
    renderer._metal_blur_pipeline_instance = None
    renderer._request_stream_start = MagicMock()
    renderer._update_stream = MagicMock()
    renderer._context = MagicMock(return_value=renderer._ci_context)
    monkeypatch.setattr(
        mod,
        "_debug_shell_grid_ci_image",
        MagicMock(return_value=SimpleNamespace(extent=lambda: _make_rect(0.0, 0.0, 680.0, 160.0))),
    )
    renderer._ci_context.createCGImage_fromRect_.return_value = "grid-image"

    image = renderer.capture_blurred_image(
        window_number=99,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=0.75,
    )

    assert image == "grid-image"
    renderer._request_stream_start.assert_not_called()
    renderer._update_stream.assert_not_called()
