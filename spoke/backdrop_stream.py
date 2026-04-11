"""Shared backdrop renderer helpers for snapshot and stream-backed overlays."""

from __future__ import annotations

import logging
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
_COREMEDIA_FRAMEWORK_PATH = "/System/Library/Frameworks/CoreMedia.framework"
_FRAME_INTERVAL_60_FPS = (1, 60, 0, 0)

_BRIDGE_STATE: dict[str, object] | None = None
_BRIDGE_LOCK = threading.Lock()


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
        config.setQueueDepth_(3)
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
        self._blur_radius_points = 0.0
        self._current_display = None
        self._current_display_frame = None
        self._current_content = None
        self._window_number = None
        self._lock = threading.Lock()
        self._ci_context = None

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
                    None,
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
            with self._lock:
                self._latest_image = image
        except Exception:
            logger.debug("ScreenCaptureKit sample processing failed", exc_info=True)

    def capture_blurred_image(self, *, window_number: int, capture_rect, blur_radius_points: float):
        self._blur_radius_points = max(blur_radius_points, 0.0)
        if self._stream is None:
            self._request_stream_start(window_number=window_number, capture_rect=capture_rect)
        else:
            self._update_stream(window_number=window_number, capture_rect=capture_rect)

        with self._lock:
            image = self._latest_image
        if image is not None:
            return image
        return self._fallback_renderer().capture_blurred_image(
            window_number=window_number,
            capture_rect=capture_rect,
            blur_radius_points=blur_radius_points,
        )
