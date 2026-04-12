"""Shared backdrop renderer helpers for snapshot and stream-backed overlays."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import warnings
from types import SimpleNamespace

try:
    import objc
    from Foundation import NSObject
except Exception:  # pragma: no cover - exercised via runtime import fallback
    objc = None

    class NSObject:  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

_SCK_FRAMEWORK_PATH = "/System/Library/Frameworks/ScreenCaptureKit.framework"
_SCK_DYLIB_PATH = f"{_SCK_FRAMEWORK_PATH}/ScreenCaptureKit"
_SCK_BRIDGESUPPORT_PATH = (
    f"{_SCK_FRAMEWORK_PATH}/Versions/A/Resources/BridgeSupport/ScreenCaptureKit.bridgesupport"
)
_METAL_FRAMEWORK_PATH = "/System/Library/Frameworks/Metal.framework"
_COREVIDEO_FRAMEWORK_PATH = "/System/Library/Frameworks/CoreVideo.framework"
_COREMEDIA_FRAMEWORK_PATH = "/System/Library/Frameworks/CoreMedia.framework"
_FRAME_INTERVAL_60_FPS = (1, 60, 0, 0)
_CV_PIXEL_FORMAT_BGRA = 1111970369
_CM_TIME_FLAGS_VALID = 1
_METAL_BLUR_DOWNSAMPLE = min(
    max(float(os.environ.get("SPOKE_BACKDROP_METAL_BLUR_DOWNSAMPLE", "1.0")), 0.25),
    1.0,
)

_BRIDGE_STATE: dict[str, object] | None = None
_BRIDGE_LOCK = threading.Lock()
_LIBDISPATCH = None
_LIBDISPATCH_LOCK = threading.Lock()


def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _screen_capture_kit_available() -> bool:
    return _load_screencapturekit_bridge() is not None


def _screen_display_id(screen) -> int | None:
    if screen is None or not hasattr(screen, "deviceDescription"):
        return None
    try:
        description = screen.deviceDescription()
    except Exception:
        return None
    if description is None:
        return None
    try:
        display_id = description["NSScreenNumber"]
    except Exception:
        display_id = description.get("NSScreenNumber") if hasattr(description, "get") else None
    return int(display_id) if display_id is not None else None


def _content_local_capture_rect(content_rect, capture_rect):
    return _make_rect(
        capture_rect.origin.x - content_rect.origin.x,
        capture_rect.origin.y - content_rect.origin.y,
        capture_rect.size.width,
        capture_rect.size.height,
    )


def _cgrect(rect):
    try:
        from Quartz import CGRectMake

        return CGRectMake(rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
    except Exception:
        return rect


def _configure_stream_geometry(config, *, content_rect, capture_rect, point_pixel_scale: float) -> None:
    local_rect = _content_local_capture_rect(content_rect, capture_rect)
    scale = max(point_pixel_scale, 1.0)
    pixel_width = max(1, int(round(capture_rect.size.width * scale)))
    pixel_height = max(1, int(round(capture_rect.size.height * scale)))

    config.setWidth_(pixel_width)
    config.setHeight_(pixel_height)
    if hasattr(config, "setQueueDepth_"):
        config.setQueueDepth_(1)
    if hasattr(config, "setShowsCursor_"):
        config.setShowsCursor_(False)
    if hasattr(config, "setScalesToFit_"):
        config.setScalesToFit_(False)
    if hasattr(config, "setContentScale_"):
        config.setContentScale_(scale)
    if hasattr(config, "setMinimumFrameInterval_"):
        config.setMinimumFrameInterval_(_FRAME_INTERVAL_60_FPS)
    if hasattr(config, "setSourceRect_"):
        config.setSourceRect_(_cgrect(local_rect))
    if hasattr(config, "setDestinationRect_"):
        config.setDestinationRect_(_cgrect(_make_rect(0.0, 0.0, capture_rect.size.width, capture_rect.size.height)))


def make_backdrop_renderer(screen, fallback_factory):
    if _screen_capture_kit_available():
        try:
            return _ScreenCaptureKitBackdropRenderer(screen, fallback_factory)
        except Exception:
            logger.debug("Falling back to Quartz backdrop renderer after ScreenCaptureKit init failure", exc_info=True)
    return fallback_factory()


def _libdispatch():
    global _LIBDISPATCH
    if _LIBDISPATCH is not None:
        return _LIBDISPATCH
    with _LIBDISPATCH_LOCK:
        if _LIBDISPATCH is not None:
            return _LIBDISPATCH
        lib = ctypes.CDLL("/usr/lib/system/libdispatch.dylib")
        lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        lib.dispatch_queue_create.restype = ctypes.c_void_p
        _LIBDISPATCH = lib
        return _LIBDISPATCH


def _make_stream_handler_queue(label: str):
    if objc is None:
        return None
    try:
        ptr = _libdispatch().dispatch_queue_create(label.encode("utf-8"), None)
        if not ptr:
            return None
        return objc.objc_object(c_void_p=ptr)
    except Exception:
        logger.debug("Failed to create dedicated ScreenCaptureKit sample handler queue", exc_info=True)
        return None


class _CMTimeStruct(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_longlong),
        ("timescale", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("epoch", ctypes.c_longlong),
    ]


class _CMSampleTimingInfoStruct(ctypes.Structure):
    _fields_ = [
        ("duration", _CMTimeStruct),
        ("presentationTimeStamp", _CMTimeStruct),
        ("decodeTimeStamp", _CMTimeStruct),
    ]


def _mark_sample_buffer_for_immediate_display(sample_buffer, bridge) -> None:
    attachments_getter = bridge.get("CMSampleBufferGetSampleAttachmentsArray")
    display_key = bridge.get("kCMSampleAttachmentKey_DisplayImmediately")
    if attachments_getter is None or display_key is None:
        return
    try:
        attachments = attachments_getter(sample_buffer, True)
    except Exception:
        logger.debug("Failed to fetch sample-buffer attachments for immediate display", exc_info=True)
        return
    if not attachments:
        return
    try:
        attachments[0][display_key] = True
    except Exception:
        logger.debug("Failed to mark sample buffer for immediate display", exc_info=True)


class _MetalBlurPipeline:
    def __init__(self):
        if objc is None:
            raise RuntimeError("PyObjC is unavailable")
        try:
            from Foundation import NSBundle, NSDictionary
            from Quartz import CIContext
        except Exception as exc:  # pragma: no cover - exercised by runtime-only path
            raise RuntimeError("Metal blur pipeline requires Foundation and Quartz") from exc

        metal_bundle = NSBundle.bundleWithPath_(_METAL_FRAMEWORK_PATH)
        objc.loadBundleFunctions(
            metal_bundle,
            globals(),
            [("MTLCreateSystemDefaultDevice", b"@")],
        )
        self._device = MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal device available")
        self._context = CIContext.contextWithMTLDevice_(self._device)
        if self._context is None:
            raise RuntimeError("Failed to create Metal-backed CIContext")

        self._cv = ctypes.CDLL(f"{_COREVIDEO_FRAMEWORK_PATH}/CoreVideo")
        self._cv.CVPixelBufferCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._cv.CVPixelBufferCreate.restype = ctypes.c_int32

        self._cm = ctypes.CDLL(f"{_COREMEDIA_FRAMEWORK_PATH}/CoreMedia")
        self._cm.CMVideoFormatDescriptionCreateForImageBuffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._cm.CMVideoFormatDescriptionCreateForImageBuffer.restype = ctypes.c_int32
        self._cm.CMSampleBufferCreateReadyWithImageBuffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_CMSampleTimingInfoStruct),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._cm.CMSampleBufferCreateReadyWithImageBuffer.restype = ctypes.c_int32

        self._pixel_buffer_attrs = NSDictionary.dictionaryWithDictionary_(
            {
                "IOSurfaceProperties": {},
                "MetalCompatibility": True,
            }
        )

    @staticmethod
    def _objc_ptr(value) -> ctypes.c_void_p:
        return ctypes.c_void_p(value.__c_void_p__().value)

    @staticmethod
    def _cm_time_struct(cm_time) -> _CMTimeStruct:
        value, timescale, flags, epoch = cm_time
        if int(timescale) == 0:
            return _CMTimeStruct(0, 0, 0, 0)
        return _CMTimeStruct(int(value), int(timescale), int(flags), int(epoch))

    def _create_pixel_buffer(self, width: int, height: int):
        pixel_buffer = ctypes.c_void_p()
        status = self._cv.CVPixelBufferCreate(
            None,
            max(1, int(width)),
            max(1, int(height)),
            _CV_PIXEL_FORMAT_BGRA,
            self._objc_ptr(self._pixel_buffer_attrs),
            ctypes.byref(pixel_buffer),
        )
        if status != 0 or not pixel_buffer.value:
            logger.debug("CVPixelBufferCreate failed for Metal blur output: %r", status)
            return None
        return objc.objc_object(c_void_p=pixel_buffer.value)

    def _create_format_description(self, pixel_buffer):
        format_desc = ctypes.c_void_p()
        status = self._cm.CMVideoFormatDescriptionCreateForImageBuffer(
            None,
            self._objc_ptr(pixel_buffer),
            ctypes.byref(format_desc),
        )
        if status != 0 or not format_desc.value:
            logger.debug("CMVideoFormatDescriptionCreateForImageBuffer failed: %r", status)
            return None
        return objc.objc_object(c_void_p=format_desc.value)

    def _create_sample_buffer(self, pixel_buffer, format_desc, *, source_sample, bridge):
        presentation = bridge["CMSampleBufferGetPresentationTimeStamp"](source_sample)
        duration = bridge["CMSampleBufferGetDuration"](source_sample)
        if int(duration[1]) == 0:
            duration = (1, 60, _CM_TIME_FLAGS_VALID, 0)
        if int(presentation[1]) == 0:
            presentation = duration
        timing = _CMSampleTimingInfoStruct(
            self._cm_time_struct(duration),
            self._cm_time_struct(presentation),
            self._cm_time_struct(presentation),
        )
        sample_buffer = ctypes.c_void_p()
        status = self._cm.CMSampleBufferCreateReadyWithImageBuffer(
            None,
            self._objc_ptr(pixel_buffer),
            self._objc_ptr(format_desc),
            ctypes.byref(timing),
            ctypes.byref(sample_buffer),
        )
        if status != 0 or not sample_buffer.value:
            logger.debug("CMSampleBufferCreateReadyWithImageBuffer failed: %r", status)
            return None
        wrapped_sample = objc.objc_object(c_void_p=sample_buffer.value)
        _mark_sample_buffer_for_immediate_display(wrapped_sample, bridge)
        return wrapped_sample

    def _render_ci_image_to_sample_buffer(self, output, extent, *, source_sample, bridge):
        width = max(1, int(round(extent.size.width)))
        height = max(1, int(round(extent.size.height)))
        pixel_buffer = self._create_pixel_buffer(width, height)
        if pixel_buffer is None:
            return None
        try:
            self._context.render_toCVPixelBuffer_bounds_colorSpace_(
                output,
                pixel_buffer,
                extent,
                None,
            )
        except Exception:
            logger.debug("Metal-backed CI render to CVPixelBuffer failed", exc_info=True)
            return None

        format_desc = self._create_format_description(pixel_buffer)
        if format_desc is None:
            return None
        return self._create_sample_buffer(
            pixel_buffer,
            format_desc,
            source_sample=source_sample,
            bridge=bridge,
        )

    @staticmethod
    def _centered_scale_transform(extent, scale_x: float, scale_y: float):
        from Quartz import (
            CGAffineTransformIdentity,
            CGAffineTransformScale,
            CGAffineTransformTranslate,
        )

        center_x = extent.origin.x + (extent.size.width / 2.0)
        center_y = extent.origin.y + (extent.size.height / 2.0)
        transform = CGAffineTransformTranslate(CGAffineTransformIdentity, -center_x, -center_y)
        transform = CGAffineTransformScale(transform, scale_x, scale_y)
        transform = CGAffineTransformTranslate(transform, center_x, center_y)
        return transform

    def blurred_sample_buffer(self, sample_buffer, *, blur_radius_points: float, bridge):
        pixel_buffer = bridge["CMSampleBufferGetImageBuffer"](sample_buffer)
        if pixel_buffer is None:
            return None
        try:
            from Quartz import CIImage, CIFilter, CGAffineTransformMakeScale
        except Exception:
            return None

        ci_image = CIImage.imageWithCVPixelBuffer_(pixel_buffer)
        if ci_image is None:
            return None
        extent = ci_image.extent() if hasattr(ci_image, "extent") else None
        if extent is None:
            return None

        working_image = ci_image
        working_extent = extent
        working_blur_radius = blur_radius_points
        if _METAL_BLUR_DOWNSAMPLE < 0.999:
            transform = CGAffineTransformMakeScale(_METAL_BLUR_DOWNSAMPLE, _METAL_BLUR_DOWNSAMPLE)
            candidate = ci_image.imageByApplyingTransform_(transform)
            if candidate is not None:
                working_image = candidate
                working_extent = candidate.extent() if hasattr(candidate, "extent") else extent
                working_blur_radius = blur_radius_points * _METAL_BLUR_DOWNSAMPLE

        output = working_image
        if working_blur_radius > 0.0:
            blur = CIFilter.filterWithName_("CIGaussianBlur")
            if blur is not None:
                blur.setDefaults()
                blur.setValue_forKey_(working_image, "inputImage")
                blur.setValue_forKey_(working_blur_radius, "inputRadius")
                candidate = blur.valueForKey_("outputImage")
                if candidate is not None:
                    output = candidate.imageByCroppingToRect_(working_extent)

        return self._render_ci_image_to_sample_buffer(
            output,
            working_extent,
            source_sample=sample_buffer,
            bridge=bridge,
        )

    def optical_shell_sample_buffer(
        self,
        sample_buffer,
        *,
        shell_config,
        cleanup_blur_radius_points: float,
        bridge,
    ):
        pixel_buffer = bridge["CMSampleBufferGetImageBuffer"](sample_buffer)
        if pixel_buffer is None:
            return None
        try:
            from Quartz import CIImage, CIFilter
        except Exception:
            return None
        ci_image = CIImage.imageWithCVPixelBuffer_(pixel_buffer)
        if ci_image is None:
            return None
        extent = ci_image.extent() if hasattr(ci_image, "extent") else None
        if extent is None:
            return None

        working_image = ci_image
        working_extent = extent
        cleanup_blur = max(float(cleanup_blur_radius_points), 0.0)
        if _METAL_BLUR_DOWNSAMPLE < 0.999:
            transform = self._centered_scale_transform(
                extent,
                _METAL_BLUR_DOWNSAMPLE,
                _METAL_BLUR_DOWNSAMPLE,
            )
            candidate = ci_image.imageByApplyingTransform_(transform)
            if candidate is not None:
                working_image = candidate
                working_extent = candidate.extent() if hasattr(candidate, "extent") else extent
                cleanup_blur *= _METAL_BLUR_DOWNSAMPLE

        clamped = (
            working_image.imageByClampingToExtent()
            if hasattr(working_image, "imageByClampingToExtent")
            else working_image
        )
        content_width = max(float(shell_config.get("content_width_points", working_extent.size.width)), 1.0)
        content_height = max(float(shell_config.get("content_height_points", working_extent.size.height)), 1.0)
        aspect_scale_x = min(max(content_height / content_width, 0.08), 1.0)
        normalized = clamped
        if abs(aspect_scale_x - 1.0) > 1e-6:
            transform = self._centered_scale_transform(working_extent, aspect_scale_x, 1.0)
            candidate = clamped.imageByApplyingTransform_(transform)
            if candidate is not None:
                normalized = candidate

        core_magnification = max(float(shell_config.get("core_magnification", 1.0)), 1.0)
        if core_magnification > 1.0:
            transform = self._centered_scale_transform(
                working_extent,
                core_magnification,
                core_magnification,
            )
            candidate = normalized.imageByApplyingTransform_(transform)
            if candidate is not None:
                normalized = candidate

        center_x = working_extent.origin.x + (working_extent.size.width / 2.0)
        center_y = working_extent.origin.y + (working_extent.size.height / 2.0)
        normalized_content_width = content_width * _METAL_BLUR_DOWNSAMPLE * aspect_scale_x
        normalized_content_height = content_height * _METAL_BLUR_DOWNSAMPLE
        shell_radius = max(
            2.0,
            min(normalized_content_width, normalized_content_height) * 0.5
            - 0.5 * float(shell_config.get("band_width_points", 12.0)),
        )
        band_width = max(
            1.0,
            float(shell_config.get("band_width_points", 12.0)) * _METAL_BLUR_DOWNSAMPLE,
        )
        tail_width = max(
            band_width,
            float(shell_config.get("tail_width_points", 9.0)) * _METAL_BLUR_DOWNSAMPLE,
        )

        output = normalized
        torus = CIFilter.filterWithName_("CITorusLensDistortion")
        if torus is not None:
            torus.setDefaults()
            torus.setValue_forKey_(output, "inputImage")
            torus.setValue_forKey_((center_x, center_y), "inputCenter")
            torus.setValue_forKey_(shell_radius, "inputRadius")
            torus.setValue_forKey_(band_width, "inputWidth")
            torus.setValue_forKey_(float(shell_config.get("ring_refraction", 2.0)), "inputRefraction")
            candidate = torus.valueForKey_("outputImage")
            if candidate is not None:
                output = candidate

        tail = CIFilter.filterWithName_("CITorusLensDistortion")
        if tail is not None:
            tail.setDefaults()
            tail.setValue_forKey_(output, "inputImage")
            tail.setValue_forKey_((center_x, center_y), "inputCenter")
            tail.setValue_forKey_(shell_radius + 0.25 * band_width, "inputRadius")
            tail.setValue_forKey_(tail_width, "inputWidth")
            tail.setValue_forKey_(float(shell_config.get("tail_refraction", 0.6)), "inputRefraction")
            candidate = tail.valueForKey_("outputImage")
            if candidate is not None:
                output = candidate

        if abs(aspect_scale_x - 1.0) > 1e-6:
            transform = self._centered_scale_transform(working_extent, 1.0 / aspect_scale_x, 1.0)
            candidate = output.imageByApplyingTransform_(transform)
            if candidate is not None:
                output = candidate

        if hasattr(output, "imageByCroppingToRect_"):
            output = output.imageByCroppingToRect_(working_extent)

        if cleanup_blur > 0.0:
            blur = CIFilter.filterWithName_("CIGaussianBlur")
            if blur is not None:
                blur.setDefaults()
                blur.setValue_forKey_(output, "inputImage")
                blur.setValue_forKey_(cleanup_blur, "inputRadius")
                candidate = blur.valueForKey_("outputImage")
                if candidate is not None and hasattr(candidate, "imageByCroppingToRect_"):
                    output = candidate.imageByCroppingToRect_(working_extent)

        return self._render_ci_image_to_sample_buffer(
            output,
            working_extent,
            source_sample=sample_buffer,
            bridge=bridge,
        )


def _load_screencapturekit_bridge() -> dict[str, object] | None:
    global _BRIDGE_STATE
    if _BRIDGE_STATE is not None:
        return _BRIDGE_STATE
    if objc is None:
        return None

    with _BRIDGE_LOCK:
        if _BRIDGE_STATE is not None:
            return _BRIDGE_STATE
        try:
            from Foundation import NSBundle
        except Exception:
            logger.debug("ScreenCaptureKit bridge unavailable: Foundation import failed", exc_info=True)
            return None

        try:
            xml = open(_SCK_BRIDGESUPPORT_PATH, "rb").read()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                objc.parseBridgeSupport(xml, globals(), "ScreenCaptureKit", _SCK_DYLIB_PATH)
            objc.loadBundle("ScreenCaptureKit", globals(), bundle_path=_SCK_FRAMEWORK_PATH)

            coremedia_bundle = NSBundle.bundleWithPath_(_COREMEDIA_FRAMEWORK_PATH)
            objc.loadBundleFunctions(
                coremedia_bundle,
                globals(),
                [
                    ("CMSampleBufferGetImageBuffer", b"^{__CVBuffer=}^{opaqueCMSampleBuffer=}"),
                    ("CMSampleBufferGetPresentationTimeStamp", b"{_CMTime=qiIq}^{opaqueCMSampleBuffer}"),
                    ("CMSampleBufferGetDuration", b"{_CMTime=qiIq}^{opaqueCMSampleBuffer}"),
                    ("CMSampleBufferGetSampleAttachmentsArray", b"^{__CFArray=}^{opaqueCMSampleBuffer=}B"),
                ],
            )

            _BRIDGE_STATE = {
                "SCShareableContent": objc.lookUpClass("SCShareableContent"),
                "SCContentFilter": objc.lookUpClass("SCContentFilter"),
                "SCStream": objc.lookUpClass("SCStream"),
                "SCStreamConfiguration": objc.lookUpClass("SCStreamConfiguration"),
                "SCStreamOutputTypeScreen": SCStreamOutputTypeScreen,
                "SCFrameStatusComplete": SCFrameStatusComplete,
                "SCStreamFrameInfoStatus": SCStreamFrameInfoStatus,
                "CMSampleBufferGetImageBuffer": CMSampleBufferGetImageBuffer,
                "CMSampleBufferGetPresentationTimeStamp": CMSampleBufferGetPresentationTimeStamp,
                "CMSampleBufferGetDuration": CMSampleBufferGetDuration,
                "CMSampleBufferGetSampleAttachmentsArray": CMSampleBufferGetSampleAttachmentsArray,
                "kCMSampleAttachmentKey_DisplayImmediately": kCMSampleAttachmentKey_DisplayImmediately,
            }
        except Exception:
            logger.debug("ScreenCaptureKit bridge load failed", exc_info=True)
            _BRIDGE_STATE = None
        return _BRIDGE_STATE


if objc is not None and hasattr(objc, "lookUpClass"):
    try:
        _ScreenCaptureKitStreamOutput = objc.lookUpClass("_ScreenCaptureKitStreamOutput")
    except Exception:
        class _ScreenCaptureKitStreamOutput(NSObject):
            def initWithRenderer_(self, renderer):
                self = objc.super(_ScreenCaptureKitStreamOutput, self).init()
                if self is None:
                    return None
                self._renderer = renderer
                return self

            def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
                self._renderer._consume_sample_buffer(sample_buffer, output_type)

        if hasattr(objc, "selector"):
            _ScreenCaptureKitStreamOutput.stream_didOutputSampleBuffer_ofType_ = objc.selector(
                _ScreenCaptureKitStreamOutput.stream_didOutputSampleBuffer_ofType_,
                selector=b"stream:didOutputSampleBuffer:ofType:",
                signature=b"v@:@@@",
            )
else:
    class _ScreenCaptureKitStreamOutput(NSObject):
        def initWithRenderer_(self, renderer):
            self._renderer = renderer
            return self

        def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
            self._renderer._consume_sample_buffer(sample_buffer, output_type)


class _ScreenCaptureKitBackdropRenderer:
    """Best-effort live backdrop renderer backed by ScreenCaptureKit."""

    def __init__(self, screen, fallback_factory):
        self._screen = screen
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._stream = None
        self._stream_output = None
        self._stream_started = False
        self._startup_requested = False
        self._pending_signature = None
        self._applied_signature = None
        self._latest_image = None
        self._frame_callback = None
        self._sample_buffer_callback = None
        self._blur_radius_points = 0.0
        self._optical_shell_config = None
        self._current_display = None
        self._current_display_frame = None
        self._current_content = None
        self._window_number = None
        self._lock = threading.Lock()
        self._ci_context = None
        self._stream_handler_queue = None
        self._metal_blur_pipeline_instance = None

    def _fallback_renderer(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        return self._fallback

    def _context(self):
        if self._ci_context is not None:
            return self._ci_context
        try:
            from Quartz import CIContext
        except Exception:
            return None
        try:
            self._ci_context = CIContext.contextWithOptions_(None)
        except Exception:
            logger.debug("Failed to create CIContext for ScreenCaptureKit backdrop", exc_info=True)
            self._ci_context = None
        return self._ci_context

    def set_frame_callback(self, callback) -> None:
        self._frame_callback = callback

    def set_sample_buffer_callback(self, callback) -> None:
        self._sample_buffer_callback = callback

    def set_live_blur_radius_points(self, blur_radius_points: float) -> None:
        self._blur_radius_points = max(float(blur_radius_points), 0.0)

    def set_live_optical_shell_config(self, config) -> None:
        if not config or config.get("enabled") is not True:
            self._optical_shell_config = None
            return
        self._optical_shell_config = dict(config)

    def _sample_handler_queue(self):
        if self._stream_handler_queue is None:
            self._stream_handler_queue = _make_stream_handler_queue("ai.spoke.backdrop-stream")
        return self._stream_handler_queue

    def _metal_blur_pipeline(self):
        instance = getattr(self, "_metal_blur_pipeline_instance", None)
        if instance is False:
            return None
        if instance is not None:
            return instance
        try:
            self._metal_blur_pipeline_instance = _MetalBlurPipeline()
        except Exception:
            logger.debug("Metal blur pipeline unavailable", exc_info=True)
            self._metal_blur_pipeline_instance = False
        return self._metal_blur_pipeline_instance or None

    def _blurred_sample_buffer(self, sample_buffer):
        pipeline = self._metal_blur_pipeline()
        if pipeline is None:
            return None
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return None
        return pipeline.blurred_sample_buffer(
            sample_buffer,
            blur_radius_points=self._blur_radius_points,
            bridge=bridge,
        )

    def _optical_shell_sample_buffer(self, sample_buffer):
        pipeline = self._metal_blur_pipeline()
        if pipeline is None or self._optical_shell_config is None:
            return None
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return None
        return pipeline.optical_shell_sample_buffer(
            sample_buffer,
            shell_config=self._optical_shell_config,
            cleanup_blur_radius_points=self._blur_radius_points,
            bridge=bridge,
        )

    def supports_sample_buffer_presentation(self, blur_radius_points: float | None = None) -> bool:
        if self._optical_shell_config is not None:
            return self._metal_blur_pipeline() is not None
        radius = self._blur_radius_points if blur_radius_points is None else max(blur_radius_points, 0.0)
        if radius <= 0.0:
            return True
        return self._metal_blur_pipeline() is not None

    def uses_direct_sample_buffers(self, blur_radius_points: float | None = None) -> bool:
        radius = self._blur_radius_points if blur_radius_points is None else max(blur_radius_points, 0.0)
        return self._sample_buffer_callback is not None and self.supports_sample_buffer_presentation(radius)

    def _dispatch_frame_callback(self, image) -> None:
        callback = self._frame_callback
        if callback is None:
            return
        try:
            callback(image)
        except Exception:
            logger.debug("ScreenCaptureKit frame callback failed", exc_info=True)

    def _publish_live_image(self, image) -> None:
        with self._lock:
            self._latest_image = image
        self._dispatch_frame_callback(image)

    def _publish_live_sample_buffer(self, sample_buffer) -> None:
        callback = self._sample_buffer_callback
        if callback is None:
            return
        try:
            callback(sample_buffer)
        except Exception:
            logger.debug("ScreenCaptureKit sample buffer callback failed", exc_info=True)

    def _signature_for(self, window_number, capture_rect, backing_scale):
        return (
            int(window_number) if window_number is not None else None,
            round(capture_rect.origin.x, 3),
            round(capture_rect.origin.y, 3),
            round(capture_rect.size.width, 3),
            round(capture_rect.size.height, 3),
            round(backing_scale, 3),
        )

    def _current_backing_scale(self) -> float:
        if self._screen is None or not hasattr(self._screen, "backingScaleFactor"):
            return 2.0
        try:
            return float(self._screen.backingScaleFactor())
        except Exception:
            return 2.0

    def _current_content_rect(self, content_filter):
        if content_filter is None:
            return self._current_display_frame
        try:
            rect = content_filter.contentRect()
            if rect is not None:
                return rect
        except Exception:
            pass
        return self._current_display_frame

    def _current_point_pixel_scale(self, content_filter) -> float:
        if content_filter is not None:
            try:
                scale = float(content_filter.pointPixelScale())
                if scale > 0.0:
                    return scale
            except Exception:
                pass
        return self._current_backing_scale()

    def _match_display(self, content):
        try:
            displays = list(content.displays())
        except Exception:
            return None
        if not displays:
            return None

        screen_display_id = _screen_display_id(self._screen)
        if screen_display_id is None:
            return displays[0]
        for display in displays:
            try:
                if int(display.displayID()) == screen_display_id:
                    return display
            except Exception:
                continue
        return displays[0]

    def _excluded_windows(self, content, window_number):
        if window_number is None or not hasattr(content, "windows"):
            return []
        try:
            windows = list(content.windows())
        except Exception:
            return []
        excluded = []
        for window in windows:
            try:
                if int(window.windowID()) == int(window_number):
                    excluded.append(window)
            except Exception:
                continue
        return excluded

    def _build_filter(self, content, display, window_number):
        bridge = _load_screencapturekit_bridge()
        SCContentFilter = bridge["SCContentFilter"]
        excluded_windows = self._excluded_windows(content, window_number)
        return SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, excluded_windows)

    def _build_configuration(self, content_filter, capture_rect):
        bridge = _load_screencapturekit_bridge()
        SCStreamConfiguration = bridge["SCStreamConfiguration"]
        config = SCStreamConfiguration.alloc().init()
        _configure_stream_geometry(
            config,
            content_rect=self._current_content_rect(content_filter),
            capture_rect=capture_rect,
            point_pixel_scale=self._current_point_pixel_scale(content_filter),
        )
        return config

    def _request_stream_start(self, *, window_number, capture_rect):
        if self._startup_requested:
            return
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return
        self._startup_requested = True
        SCShareableContent = bridge["SCShareableContent"]

        def got_content(content):
            self._startup_requested = False
            if content is None:
                return
            try:
                self._current_content = content
                self._current_display = self._match_display(content)
                if self._current_display is None:
                    return
                self._current_display_frame = self._current_display.frame()
                content_filter = self._build_filter(content, self._current_display, window_number)
                config = self._build_configuration(content_filter, capture_rect)
                SCStream = bridge["SCStream"]
                stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                    content_filter,
                    config,
                    None,
                )
                stream_output = _ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(self)
                success, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    stream_output,
                    bridge["SCStreamOutputTypeScreen"],
                    self._sample_handler_queue(),
                    None,
                )
                if not success:
                    logger.debug("ScreenCaptureKit addStreamOutput failed: %r", error)
                    return

                def started(error):
                    if error is not None:
                        logger.debug("ScreenCaptureKit startCapture failed: %r", error)
                        return
                    self._stream_started = True

                self._stream = stream
                self._stream_output = stream_output
                self._window_number = window_number
                self._applied_signature = self._signature_for(
                    window_number,
                    capture_rect,
                    self._current_backing_scale(),
                )
                stream.startCaptureWithCompletionHandler_(started)
            except Exception:
                logger.debug("ScreenCaptureKit stream startup failed", exc_info=True)
                self._stream = None
                self._stream_output = None
                self._stream_started = False

        SCShareableContent.getShareableContentWithCompletionHandler_(got_content)

    def _update_stream(self, *, window_number, capture_rect):
        if self._stream is None or self._current_display is None:
            return
        signature = self._signature_for(window_number, capture_rect, self._current_backing_scale())
        if signature == self._applied_signature or signature == self._pending_signature:
            return
        self._pending_signature = signature
        try:
            content_filter = self._build_filter(
                self._current_content if self._current_content is not None else SimpleNamespace(windows=lambda: []),
                self._current_display,
                window_number,
            )
            config = self._build_configuration(content_filter, capture_rect)
            self._window_number = window_number

            def updated_filter(error):
                if error is not None:
                    logger.debug("ScreenCaptureKit updateContentFilter failed: %r", error)
                    self._pending_signature = None
                    return

                def updated_config(error):
                    if error is not None:
                        logger.debug("ScreenCaptureKit updateConfiguration failed: %r", error)
                        self._pending_signature = None
                        return
                    self._applied_signature = signature
                    self._pending_signature = None

                self._stream.updateConfiguration_completionHandler_(config, updated_config)

            self._stream.updateContentFilter_completionHandler_(content_filter, updated_filter)
        except Exception:
            logger.debug("ScreenCaptureKit stream update failed", exc_info=True)
            self._pending_signature = None

    def _consume_sample_buffer(self, sample_buffer, output_type):
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return
        try:
            output_type_value = int(output_type)
        except Exception:
            output_type_value = output_type
        if output_type_value != bridge["SCStreamOutputTypeScreen"] or sample_buffer is None:
            return
        optical_shell_config = getattr(self, "_optical_shell_config", None)
        if self._sample_buffer_callback is not None:
            if optical_shell_config is not None:
                shell_sample_buffer = self._optical_shell_sample_buffer(sample_buffer)
                if shell_sample_buffer is not None:
                    self._publish_live_sample_buffer(shell_sample_buffer)
                    return
            elif self._blur_radius_points <= 0.0:
                self._publish_live_sample_buffer(sample_buffer)
                return
            else:
                blurred_sample_buffer = self._blurred_sample_buffer(sample_buffer)
                if blurred_sample_buffer is not None:
                    self._publish_live_sample_buffer(blurred_sample_buffer)
                    return
        if optical_shell_config is None and self._blur_radius_points <= 0.0 and self.uses_direct_sample_buffers():
            self._publish_live_sample_buffer(sample_buffer)
            return
        try:
            pixel_buffer = bridge["CMSampleBufferGetImageBuffer"](sample_buffer)
            if pixel_buffer is None:
                return
            from Quartz import CIImage, CIFilter

            ci_image = CIImage.imageWithCVPixelBuffer_(pixel_buffer)
            if ci_image is None:
                return
            extent = ci_image.extent() if hasattr(ci_image, "extent") else None
            if extent is None:
                return
            output = ci_image
            if self._blur_radius_points > 0.0:
                blur = CIFilter.filterWithName_("CIGaussianBlur")
                if blur is not None:
                    blur.setDefaults()
                    blur.setValue_forKey_(ci_image, "inputImage")
                    blur.setValue_forKey_(self._blur_radius_points, "inputRadius")
                    candidate = blur.valueForKey_("outputImage")
                    if candidate is not None:
                        output = candidate.imageByCroppingToRect_(extent)
            context = self._context()
            if context is None:
                return
            image = context.createCGImage_fromRect_(output, extent)
            if image is None:
                return
            self._publish_live_image(image)
        except Exception:
            logger.debug("ScreenCaptureKit sample processing failed", exc_info=True)

    def capture_blurred_image(self, *, window_number: int, capture_rect, blur_radius_points: float):
        self._blur_radius_points = max(blur_radius_points, 0.0)
        if self._stream is None:
            self._request_stream_start(window_number=window_number, capture_rect=capture_rect)
        else:
            self._update_stream(window_number=window_number, capture_rect=capture_rect)

        if self.uses_direct_sample_buffers(self._blur_radius_points):
            return None

        with self._lock:
            image = self._latest_image
        if image is not None:
            return image
        return self._fallback_renderer().capture_blurred_image(
            window_number=window_number,
            capture_rect=capture_rect,
            blur_radius_points=blur_radius_points,
        )
