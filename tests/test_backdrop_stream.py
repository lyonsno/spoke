"""Tests for the ScreenCaptureKit backdrop renderer seam."""

import importlib
import math
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
    assert config.calls["queue_depth"] == 3
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
    renderer._has_live_content = False
    callback = MagicMock()
    renderer._frame_callback = callback

    renderer._publish_live_image("fresh-frame")

    assert renderer._latest_image == "fresh-frame"
    assert renderer._has_live_content is True
    callback.assert_called_once_with("fresh-frame")


def test_publish_live_sample_buffer_invokes_callback():
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._has_live_content = False
    callback = MagicMock()
    renderer._sample_buffer_callback = callback

    renderer._publish_live_sample_buffer("sample-buffer")

    assert renderer._has_live_content is True
    callback.assert_called_once_with("sample-buffer")


def test_publish_live_image_dispatches_callback_via_main_thread_helper(monkeypatch):
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._lock = mod.threading.Lock()
    renderer._latest_image = None
    renderer._has_live_content = False
    callback = MagicMock()
    renderer._frame_callback = callback
    dispatch = MagicMock(side_effect=lambda fn, arg: fn(arg))
    monkeypatch.setattr(mod, "_call_on_main_thread", dispatch)

    renderer._publish_live_image("fresh-frame")

    dispatch.assert_called_once()
    callback.assert_called_once_with("fresh-frame")


def test_publish_live_sample_buffer_dispatches_callback_via_main_thread_helper(monkeypatch):
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._has_live_content = False
    callback = MagicMock()
    renderer._sample_buffer_callback = callback
    dispatch = MagicMock(side_effect=lambda fn, arg: fn(arg))
    monkeypatch.setattr(mod, "_call_on_main_thread", dispatch)

    renderer._publish_live_sample_buffer("sample-buffer")

    dispatch.assert_called_once()
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


def test_consume_sample_buffer_optical_shell_direct_ciimage_path(monkeypatch):
    """When optical shell is active, consume extracts a CIImage from the
    SCK sample buffer, applies the warp, renders to CGImage, and publishes."""
    mod = _import_module()

    fake_ci_image = MagicMock()
    fake_ci_image.extent.return_value = MagicMock(size=MagicMock(width=100, height=50))
    fake_ci_image.imageByClampingToExtent.return_value = fake_ci_image
    fake_ci_image.imageByCroppingToRect_.return_value = fake_ci_image
    fake_quartz = types.ModuleType("Quartz")
    fake_quartz.CIImage = MagicMock()
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    bridge = {
        "SCStreamOutputTypeScreen": 7,
        "CMSampleBufferGetImageBuffer": MagicMock(return_value="pixel-buf"),
        "_CIImage_from_sample_buffer": MagicMock(return_value=fake_ci_image),
    }
    monkeypatch.setattr(mod, "_load_screencapturekit_bridge", lambda: bridge)
    monkeypatch.setattr(mod, "_apply_optical_shell_warp_ci_image", MagicMock(return_value=fake_ci_image))

    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 0.2
    renderer._sample_buffer_callback = MagicMock()
    renderer._frame_callback = MagicMock()
    renderer._optical_shell_config = {"enabled": True}
    renderer._publish_live_sample_buffer = MagicMock()
    renderer._publish_live_image = MagicMock()
    fake_context = MagicMock()
    fake_context.createCGImage_fromRect_ = MagicMock(return_value="cg-image")
    renderer._ci_context = fake_context
    renderer._lock = __import__("threading").Lock()
    renderer._latest_image = None
    renderer._publish_image_count = 0
    renderer._screen = MagicMock()
    renderer._screen.backingScaleFactor.return_value = 2.0

    renderer._consume_sample_buffer("live-sample", 7)

    # Should use _CIImage_from_sample_buffer, not Metal pipeline
    bridge["_CIImage_from_sample_buffer"].assert_called_once_with("live-sample")
    # Should publish via live image path
    renderer._publish_live_image.assert_called_once()
    renderer._publish_live_sample_buffer.assert_not_called()


def test_capture_blurred_image_clears_stale_cached_frame_before_stream_update(monkeypatch):
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 5.4
    renderer._stream = object()
    renderer._sample_buffer_callback = None
    renderer._latest_image = "stale-frame"
    renderer._has_live_content = True
    renderer._lock = mod.threading.Lock()
    renderer._update_stream = MagicMock(
        side_effect=lambda *args, **kwargs: renderer._clear_live_content()
    )
    renderer._request_stream_start = MagicMock()
    fallback = MagicMock()
    fallback.capture_blurred_image.return_value = "fallback-frame"
    renderer._fallback_renderer = MagicMock(return_value=fallback)

    image = renderer.capture_blurred_image(
        window_number=17,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=5.4,
    )

    assert image == "fallback-frame"
    assert renderer._latest_image is None
    assert renderer._has_live_content is False
    renderer._update_stream.assert_called_once()
    renderer._request_stream_start.assert_not_called()


def test_capture_blurred_image_keeps_fallback_until_direct_sample_path_is_live(monkeypatch):
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._blur_radius_points = 0.0
    renderer._stream = None
    renderer._sample_buffer_callback = MagicMock()
    renderer._latest_image = None
    renderer._has_live_content = False
    renderer._lock = mod.threading.Lock()
    renderer.supports_sample_buffer_presentation = MagicMock(return_value=True)
    renderer._request_stream_start = MagicMock()
    fallback = MagicMock()
    fallback.capture_blurred_image.return_value = "fallback-frame"
    renderer._fallback_renderer = MagicMock(return_value=fallback)

    image = renderer.capture_blurred_image(
        window_number=17,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=0.0,
    )

    assert image == "fallback-frame"
    renderer._request_stream_start.assert_called_once()


def test_stop_live_stream_stops_capture_and_clears_cached_state():
    mod = _import_module()
    renderer = mod._ScreenCaptureKitBackdropRenderer.__new__(mod._ScreenCaptureKitBackdropRenderer)
    renderer._lock = mod.threading.Lock()
    stream = MagicMock()
    renderer._stream = stream
    renderer._stream_output = object()
    renderer._stream_started = True
    renderer._startup_requested = True
    renderer._pending_signature = ("pending",)
    renderer._applied_signature = ("applied",)
    renderer._latest_image = "stale-frame"
    renderer._has_live_content = True

    renderer.stop_live_stream()

    stream.stopCaptureWithCompletionHandler_.assert_called_once()
    assert renderer._stream is None
    assert renderer._stream_output is None
    assert renderer._stream_started is False
    assert renderer._startup_requested is False
    assert renderer._pending_signature is None
    assert renderer._applied_signature is None
    assert renderer._latest_image is None
    assert renderer._has_live_content is False


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
        [
            680.0,
            160.0,
            600.0,
            80.0,
            mod._optical_shell_effective_corner_radius(16.0, 12.0),
            2.5,
            12.0,
            10.0,
            72.0,
            18.0,
        ]
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


def test_optical_shell_inside_envelope_is_edge_localized():
    mod = _import_module()

    center = mod._optical_shell_inside_envelope(40.0, 15.874016)
    near_edge = mod._optical_shell_inside_envelope(2.0, 15.874016)
    outside = mod._optical_shell_inside_envelope(0.0, 15.874016)

    assert center < 0.08
    assert near_edge > 0.9
    assert outside == 1.0


def test_optical_shell_center_envelope_prefers_center_and_rounds_shoulders():
    mod = _import_module()

    center = mod._optical_shell_center_envelope(
        offset_x=0.0,
        offset_y=0.0,
        content_width=600.0,
        content_height=80.0,
        band_width=11.338583,
    )
    axis_shoulder = mod._optical_shell_center_envelope(
        offset_x=180.0,
        offset_y=0.0,
        content_width=600.0,
        content_height=80.0,
        band_width=11.338583,
    )
    diagonal_shoulder = mod._optical_shell_center_envelope(
        offset_x=180.0,
        offset_y=22.0,
        content_width=600.0,
        content_height=80.0,
        band_width=11.338583,
    )
    near_rim = mod._optical_shell_center_envelope(
        offset_x=285.0,
        offset_y=0.0,
        content_width=600.0,
        content_height=80.0,
        band_width=11.338583,
    )

    assert center > 0.97
    assert 0.22 < axis_shoulder < 0.9
    assert diagonal_shoulder < 0.14
    assert axis_shoulder > diagonal_shoulder + 0.18
    assert near_rim < 0.05


def test_optical_shell_interior_flow_bridges_center_and_rim_regions():
    mod = _import_module()

    flow = mod._optical_shell_interior_flow(0.42, 0.28)

    assert flow > 0.72
    assert flow > 0.42


def test_optical_shell_core_displacement_envelope_quiets_center_and_peaks_in_shoulders():
    mod = _import_module()

    center = mod._optical_shell_core_displacement_envelope(0.98, 0.22)
    shoulder = mod._optical_shell_core_displacement_envelope(0.42, 0.64)
    quiet_rim = mod._optical_shell_core_displacement_envelope(0.04, 0.08)

    assert center < 0.08
    assert shoulder > 0.4
    assert shoulder > center + 0.35
    assert quiet_rim < 0.16


def test_apply_optical_shell_warp_inflates_corner_radius_for_smoother_field(monkeypatch):
    mod = _import_module()

    class FakeImage:
        pass

    image = FakeImage()
    extent = _make_rect(0.0, 0.0, 680.0, 160.0)
    kernel = MagicMock()
    kernel.applyWithExtent_roiCallback_inputImage_arguments_.return_value = image
    monkeypatch.setattr(mod, "_shell_warp_kernel", lambda: kernel)

    mod._apply_optical_shell_warp_ci_image(
        image,
        extent,
        {
            "content_width_points": 600.0,
            "content_height_points": 80.0,
            "corner_radius_points": 16.0,
            "core_magnification": 2.8,
            "band_width_points": 11.338583,
            "tail_width_points": 7.086614,
            "ring_amplitude_points": 132.0,
            "tail_amplitude_points": 4.0,
        },
    )

    args = kernel.applyWithExtent_roiCallback_inputImage_arguments_.call_args[0][3]
    assert args[4] == pytest.approx(
        mod._optical_shell_effective_corner_radius(16.0, 11.338583)
    )


def test_optical_shell_gradient_epsilon_scales_with_band_width_for_corner_smoothing():
    mod = _import_module()

    narrow = mod._optical_shell_gradient_epsilon(2.0)
    wide = mod._optical_shell_gradient_epsilon(11.338583)

    assert narrow == 1.0
    assert wide > 2.0


def test_optical_shell_corner_relief_preserves_flats_but_softens_corners():
    mod = _import_module()

    flat = mod._optical_shell_corner_relief(
        offset_x=285.0,
        offset_y=0.0,
        content_width=600.0,
        content_height=80.0,
        corner_radius=16.0,
        band_width=11.338583,
    )
    corner = mod._optical_shell_corner_relief(
        offset_x=270.0,
        offset_y=28.0,
        content_width=600.0,
        content_height=80.0,
        corner_radius=16.0,
        band_width=11.338583,
    )

    assert flat > 0.95
    assert corner < 0.82
    assert flat > corner + 0.12


def test_debug_shell_grid_profile_prioritizes_field_contours_over_coordinate_hints():
    mod = _import_module()

    profile = mod._debug_shell_grid_profile({"debug_grid_spacing_points": 14.0})

    assert profile["spacing"] == 14.0
    assert profile["field_major_step"] == 0.125
    assert profile["field_minor_step"] == 0.0625
    assert profile["field_contour_halfwidth"] == 0.012
    assert profile["field_minor_contour_halfwidth"] == 0.006
    assert profile["field_color"] == (45, 45, 45, 210)
    assert profile["field_minor_color"] == (70, 70, 70, 110)
    assert profile["longitudinal_hint_color"] == (65, 65, 65, 0)
    assert profile["radial_hint_color"] == (80, 80, 80, 0)
    assert profile["ring_color"] == (90, 90, 90, 0)
    assert profile["center_marker_shape"] == "circle"
    assert profile["center_marker_width_points"] == 12.0
    assert profile["center_marker_height_points"] == 12.0


def test_optical_shell_capsule_coordinate_fields_follow_pill_geometry():
    mod = _import_module()

    longitudinal, radial = mod._optical_shell_capsule_coordinate_fields(241, 101, 241.0, 101.0)

    center_y = 50
    center_x = 120
    body_x = 180
    cap_x = 230
    shoulder_y = 20

    assert longitudinal.shape == (101, 241)
    assert radial.shape == (101, 241)
    assert 0.0 <= longitudinal[center_y, center_x] < 0.01
    assert 0.0 <= radial[center_y, center_x] < 0.01
    assert 0.45 < longitudinal[center_y, body_x] < 0.55
    assert 0.98 < longitudinal[center_y, cap_x] <= 1.0
    assert longitudinal[center_y, body_x] < longitudinal[shoulder_y, cap_x] < longitudinal[center_y, cap_x]
    assert radial[center_y, center_x] < radial[shoulder_y, body_x]


def test_debug_shell_grid_ci_image_renders_without_unbound_field_masks(monkeypatch):
    fake_foundation = types.ModuleType("Foundation")

    class FakeNSData:
        @staticmethod
        def dataWithBytes_length_(payload, length):
            return payload[:length]

    fake_foundation.NSData = FakeNSData
    monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)

    fake_quartz = types.ModuleType("Quartz")

    class FakeCIImage:
        @staticmethod
        def imageWithCGImage_(image):
            return ("ci-image", image)

    fake_quartz.CIImage = FakeCIImage
    fake_quartz.CGColorSpaceCreateDeviceRGB = lambda: "colorspace"
    fake_quartz.CGDataProviderCreateWithCFData = lambda payload: ("provider", payload)
    fake_quartz.CGImageCreate = lambda *args: "cg-image"
    fake_quartz.kCGImageAlphaPremultipliedLast = 1
    fake_quartz.kCGRenderingIntentDefault = 0
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    mod = _import_module()

    image = mod._debug_shell_grid_ci_image(
        _make_rect(0.0, 0.0, 680.0, 160.0),
        {
            "content_width_points": 680.0,
            "content_height_points": 160.0,
            "corner_radius_points": 36.0,
            "core_magnification": 2.2,
            "ring_amplitude_points": 96.0,
        },
    )

    assert image == ("ci-image", "cg-image")


def test_optical_shell_depth_remap_is_monotone_and_center_weighted():
    mod = _import_module()

    rim = mod._optical_shell_depth_remap(0.0, 0.95)
    quarter = mod._optical_shell_depth_remap(0.25, 0.95)
    midpoint = mod._optical_shell_depth_remap(0.5, 0.95)
    near_center = mod._optical_shell_depth_remap(0.85, 0.95)
    center = mod._optical_shell_depth_remap(1.0, 0.95)

    assert rim == 0.0
    assert 0.28 < quarter < midpoint < near_center < center <= 1.0
    assert 0.58 < midpoint < 0.66
    assert near_center > 0.94


def test_optical_shell_source_depth_points_scales_from_boundary_inward():
    mod = _import_module()

    rim = mod._optical_shell_source_depth_points(0.0, 50.0)
    midpoint = mod._optical_shell_source_depth_points(0.5, 50.0)
    center = mod._optical_shell_source_depth_points(1.0, 50.0)

    assert rim == 0.0
    assert 24.0 < midpoint < 26.0
    assert center == 50.0


def test_optical_shell_capsule_spine_half_length_uses_half_height_radius():
    mod = _import_module()

    assert mod._optical_shell_capsule_spine_half_length(240.0, 100.0) == 70.0
    assert mod._optical_shell_capsule_spine_half_length(100.0, 100.0) == 1.0


def test_optical_shell_capsule_axis_decomposition_blends_into_endcaps():
    mod = _import_module()

    spine_half = 70.0
    capsule_radius = 50.0

    body_spine, body_radial = mod._optical_shell_capsule_axis_decomposition(60.0, spine_half, capsule_radius)
    seam_spine, seam_radial = mod._optical_shell_capsule_axis_decomposition(68.0, spine_half, capsule_radius)
    cap_shoulder_spine, cap_shoulder_radial = mod._optical_shell_capsule_axis_decomposition(75.0, spine_half, capsule_radius)
    cap_spine, cap_radial = mod._optical_shell_capsule_axis_decomposition(120.0, spine_half, capsule_radius)

    assert 59.95 < body_spine <= 60.0
    assert 0.0 <= body_radial < 0.05
    assert 67.0 < seam_spine < 68.0
    assert 0.0 < seam_radial < 1.0
    assert seam_spine + seam_radial == pytest.approx(68.0)
    assert 69.5 < cap_shoulder_spine <= spine_half
    assert 5.0 - 1e-6 < cap_shoulder_radial < 5.5
    assert cap_shoulder_spine + cap_shoulder_radial == pytest.approx(75.0)
    assert cap_spine == pytest.approx(spine_half)
    assert cap_radial == pytest.approx(50.0)


def test_optical_shell_capsule_longitudinal01_tracks_cap_angle():
    mod = _import_module()

    body = mod._optical_shell_capsule_longitudinal01(60.0, 0.0, 240.0, 100.0)
    cap_axis = mod._optical_shell_capsule_longitudinal01(110.0, 0.0, 240.0, 100.0)
    cap_shoulder = mod._optical_shell_capsule_longitudinal01(110.0, 35.0, 240.0, 100.0)

    assert 0.45 < body < 0.52
    assert 0.98 < cap_axis <= 1.0
    assert body < cap_shoulder < cap_axis
    assert 0.76 < cap_shoulder < 0.86


def test_optical_shell_pill_offset_sdf_tracks_nested_pill_offsets():
    mod = _import_module()

    center = mod._optical_shell_pill_offset_sdf(0.0, 0.0, 240.0, 100.0)
    body = mod._optical_shell_pill_offset_sdf(110.0, 0.0, 240.0, 100.0)
    side = mod._optical_shell_pill_offset_sdf(0.0, 40.0, 240.0, 100.0)
    cap = mod._optical_shell_pill_offset_sdf(70.0 + 40.0 / math.sqrt(2.0), 40.0 / math.sqrt(2.0), 240.0, 100.0)

    assert center == pytest.approx(-50.0)
    assert body == pytest.approx(-10.0)
    assert side == pytest.approx(-10.0)
    assert cap == pytest.approx(-10.0, abs=1e-3)


def test_optical_shell_pill_field01_tracks_scaled_capsule_family():
    mod = _import_module()

    field_x = mod._optical_shell_pill_field01(110.0, 0.0, 240.0, 100.0)
    field_y = mod._optical_shell_pill_field01(0.0, 40.0, 240.0, 100.0)
    field_diag = mod._optical_shell_pill_field01(
        70.0 + 40.0 / math.sqrt(2.0),
        40.0 / math.sqrt(2.0),
        240.0,
        100.0,
    )

    assert field_x == pytest.approx(0.8)
    assert field_y == pytest.approx(0.8)
    assert field_diag == pytest.approx(0.8, abs=1e-3)


def test_optical_shell_debug_field01_uses_remapped_scalar_not_raw_capsule_field():
    mod = _import_module()

    raw_mid = mod._optical_shell_pill_field01(60.0, 0.0, 240.0, 100.0)
    debug_mid = mod._optical_shell_debug_field01(60.0, 0.0, 240.0, 100.0, 0.95)
    raw_shoulder = mod._optical_shell_pill_field01(75.0, 20.0, 240.0, 100.0)
    debug_shoulder = mod._optical_shell_debug_field01(75.0, 20.0, 240.0, 100.0, 0.95)

    assert raw_mid == pytest.approx(0.0)
    assert debug_mid == pytest.approx(0.0)
    assert raw_shoulder > raw_mid
    assert 0.26 < debug_shoulder < raw_shoulder
    assert debug_mid < debug_shoulder


def test_optical_shell_debug_field01_keeps_body_flatter_before_rim_steepening():
    mod = _import_module()

    center = mod._optical_shell_debug_field01(0.0, 0.0, 240.0, 100.0, 0.95)
    mid_body = mod._optical_shell_debug_field01(60.0, 0.0, 240.0, 100.0, 0.95)
    shoulder = mod._optical_shell_debug_field01(75.0, 20.0, 240.0, 100.0, 0.95)
    inset_body = mod._optical_shell_debug_field01(110.0, 0.0, 240.0, 100.0, 0.95)
    near_rim = mod._optical_shell_debug_field01(119.0, 0.0, 240.0, 100.0, 0.95)

    assert center == pytest.approx(0.0)
    assert mid_body == pytest.approx(0.0)
    assert 0.26 < shoulder < 0.3
    assert 0.75 < inset_body < 0.79
    assert 0.97 < near_rim < 0.99
    assert center == mid_body < shoulder < inset_body < near_rim


def test_optical_shell_inside_depth01_tracks_rounded_rect_depth():
    mod = _import_module()

    center = mod._optical_shell_inside_depth01_from_sdf(-50.0, 240.0, 100.0)
    shoulder = mod._optical_shell_inside_depth01_from_sdf(-24.0, 240.0, 100.0)
    rim = mod._optical_shell_inside_depth01_from_sdf(-2.0, 240.0, 100.0)
    outside = mod._optical_shell_inside_depth01_from_sdf(4.0, 240.0, 100.0)

    assert 0.99 <= center <= 1.0
    assert center > shoulder > rim > 0.0
    assert outside == 0.0


def test_optical_shell_kernel_uses_single_depth_remap_curve():
    mod = _import_module()

    source = mod._SHELL_WARP_KERNEL_SOURCE

    assert "float capsuleRadius = max(halfRect.y, 1.0);" in source
    assert "float capsuleSdf = sdCapsule(p, spineHalf, capsuleRadius);" in source
    assert "depthRemap" in source
    assert "vec2(scaleX, scaleY)" in source


def test_optical_shell_kernel_avoids_global_center_depth_mix():
    mod = _import_module()

    source = mod._SHELL_WARP_KERNEL_SOURCE

    assert "float centerDepth = max(min(halfRect.x, halfRect.y), 1.0);" not in source
    assert "vec2 src = mix(boundary, c, source01);" not in source
    assert "float capsuleRadius = max(halfRect.y, 1.0);" in source


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
