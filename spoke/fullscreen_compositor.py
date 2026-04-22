"""Full-screen compositor for optical shell warp.

Captures the entire display via ScreenCaptureKit, applies the capsule
warp at the overlay's screen position, and presents the full frame
via a borderless full-screen CAMetalLayer window.  Every pixel on
screen is ours — no seam between warped and unwarped content because
there is no unwarped content visible.

The structural lag (1-2 frames at display refresh rate) is uniform
across the entire display, which reads as "slightly delayed" rather
than "torn overlay."
"""

from __future__ import annotations

import ctypes
import logging
import struct
import threading
import time

import objc

logger = logging.getLogger(__name__)


class FullScreenCompositor:
    """Full-display capture → Metal warp → full-screen presentation."""

    # Class-level registry of all active compositor window IDs.
    # Each compositor excludes all registered windows from its capture
    # to prevent feedback loops when multiple compositors run concurrently.
    _active_compositor_windows: set[int] = set()

    def __init__(self, screen):
        self._screen = screen
        self._lock = threading.Lock()
        self._running = False

        from spoke.metal_warp import get_metal_warp_pipeline
        self._pipeline = get_metal_warp_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Metal warp pipeline unavailable")

        # State
        self._latest_iosurface = None
        self._latest_pixel_buffer = None
        self._latest_width = 0
        self._latest_height = 0
        self._shell_config = None
        self._window = None
        self._metal_layer = None
        self._display_link = None
        self._stream = None
        self._stream_output = None
        self._stream_handler_queue = None

        # Diagnostics
        self._frame_count = 0
        self._presented_count = 0
        self._last_report_time = 0.0
        self._interval_frame_count = 0
        self._interval_presented = 0
        self._last_drawable_size = (0, 0)

        # Brightness sampling from the captured IOSurface
        self._sampled_brightness = 0.5
        self._brightness_sample_frame = 0
        _BRIGHTNESS_SAMPLE_INTERVAL_FRAMES = 15  # every ~250ms at 60fps

    def start(self, shell_config: dict) -> bool:
        """Create the full-screen window and start capture + render loop."""
        if self._running:
            return True
        try:
            self._shell_config = dict(shell_config) if shell_config else None
            # Reset temporal accumulation so the first frame doesn't
            # blend with stale content from a previous compositor session.
            if self._pipeline is not None:
                self._pipeline.reset_temporal_state()
            self._create_fullscreen_window()
            self._start_capture()
            self._start_display_link()
            self._running = True
            logger.info("FullScreenCompositor: started")
            return True
        except Exception:
            logger.info("FullScreenCompositor: failed to start", exc_info=True)
            self.stop()
            return False

    def stop(self) -> None:
        """Tear down everything."""
        self._running = False
        self._stop_display_link()
        self._stop_capture()
        self._destroy_fullscreen_window()
        with self._lock:
            self._latest_iosurface = None
            self._latest_pixel_buffer = None
        logger.info(
            "FullScreenCompositor: stopped (%d presented / %d ticks)",
            self._presented_count, self._frame_count,
        )

    def update_shell_config(self, config: dict) -> None:
        """Update the warp parameters (capsule position, size, etc.)."""
        with self._lock:
            self._shell_config = dict(config) if config else None

    def update_shell_config_key(self, key: str, value) -> None:
        """Update a single key in the shell config without replacing."""
        with self._lock:
            if self._shell_config is not None:
                self._shell_config[key] = value

    @property
    def sampled_brightness(self) -> float:
        """Average brightness of the capsule region."""
        return self._sampled_brightness

    def refresh_brightness(self) -> None:
        """Re-sample capsule region brightness.  Call from main thread."""
        with self._lock:
            config = self._shell_config
            w = self._latest_width
            h = self._latest_height
        if config is None or w <= 0 or h <= 0:
            return
        self._sample_iosurface_brightness(None, w, h, config)

    def _sample_iosurface_brightness(self, iosurface, w, h, config):
        """Sample average luminance of the capsule region.

        Uses CGWindowListCreateImage on a small rect centered on the
        capsule, excluding the compositor's own window.  This captures
        the raw desktop content behind the overlay — the same content
        the warp is transforming.
        """
        try:
            from Quartz import (
                CGWindowListCreateImage,
                kCGWindowListOptionOnScreenBelowWindow,
                CGImageGetWidth, CGImageGetHeight,
                CGImageGetDataProvider, CGDataProviderCopyData,
            )

            # Capsule center in points (config has pixel coords, convert back)
            scale = self._screen.backingScaleFactor() if hasattr(self._screen, 'backingScaleFactor') else 2.0
            cx_pts = config.get("center_x", w * 0.5) / scale
            cy_pts = config.get("center_y", h * 0.5) / scale
            rw_pts = config.get("content_width_points", w) / scale * 0.5
            rh_pts = config.get("content_height_points", h) / scale * 0.5

            # Screen origin (Quartz coords: top-left origin)
            screen_frame = self._screen.frame()
            sox = screen_frame.origin.x
            soy = screen_frame.origin.y
            sh = screen_frame.size.height

            # Convert Cocoa Y (bottom-up) to Quartz Y (top-down)
            # cy_pts is already in Metal coords (top-down from _start_fullscreen_compositor)
            capture_x = sox + cx_pts - rw_pts
            capture_y = soy + cy_pts - rh_pts
            capture_w = rw_pts * 2
            capture_h = rh_pts * 2

            if capture_w <= 0 or capture_h <= 0:
                return

            # Capture below the compositor window (excludes spoke overlay + compositor)
            wid = int(self._window.windowNumber()) if self._window is not None else 0
            rect = ((capture_x, capture_y), (capture_w, capture_h))
            image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenBelowWindow,
                wid,
                0,
            )
            if image is None:
                return

            iw = CGImageGetWidth(image)
            ih = CGImageGetHeight(image)
            data = CGDataProviderCopyData(CGImageGetDataProvider(image))
            if data is None or len(data) == 0 or iw * ih == 0:
                return

            bpp = len(data) // (iw * ih)
            total = 0.0
            count = 0
            step = max(5, 1)
            for sy in range(0, ih, step):
                for sx in range(0, iw, step):
                    offset = (sy * iw + sx) * bpp
                    if offset + 3 <= len(data):
                        r, g, b = data[offset], data[offset + 1], data[offset + 2]
                        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        total += lum
                        count += 1
            if count > 0:
                self._sampled_brightness = total / count
        except Exception:
            pass  # non-critical — keep previous brightness

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def window(self):
        """The full-screen NSWindow, for excluding from capture and layering UI."""
        return self._window

    # ------------------------------------------------------------------
    # Full-screen window
    # ------------------------------------------------------------------

    def _create_fullscreen_window(self):
        from AppKit import (
            NSWindow, NSScreen, NSColor,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorStationary,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSBackingStoreBuffered,
            NSView,
        )

        screen = self._screen
        frame = screen.frame()
        scale = screen.backingScaleFactor() if hasattr(screen, "backingScaleFactor") else 2.0

        # Borderless, transparent, full-screen, just below the command overlay
        # so the overlay UI (text, fill) renders on top of the warped screen.
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False,
        )
        # Command overlay is at level 26 (_OVERLAY_WINDOW_LEVEL + 1),
        # glow is at 25.  Compositor sits at 24 — above normal windows
        # but below both, so brightness sampling with
        # kCGWindowListOptionOnScreenBelowWindow from the glow window
        # captures the warped content (what the user sees through the fill).
        self._window.setLevel_(24)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Content view with CAMetalLayer
        content = NSView.alloc().initWithFrame_(((0, 0), (frame.size.width, frame.size.height)))
        content.setWantsLayer_(True)

        CAMetalLayer = objc.lookUpClass("CAMetalLayer")
        self._metal_layer = CAMetalLayer.alloc().init()
        self._metal_layer.setDevice_(self._pipeline.device)
        self._metal_layer.setPixelFormat_(80)  # BGRA8Unorm
        self._metal_layer.setFramebufferOnly_(False)
        self._metal_layer.setOpaque_(False)
        self._metal_layer.setContentsScale_(scale)
        pixel_w = int(frame.size.width * scale)
        pixel_h = int(frame.size.height * scale)
        self._metal_layer.setDrawableSize_((pixel_w, pixel_h))
        if hasattr(self._metal_layer, "setPresentsWithTransaction_"):
            self._metal_layer.setPresentsWithTransaction_(False)
        if hasattr(self._metal_layer, "setMaximumDrawableCount_"):
            self._metal_layer.setMaximumDrawableCount_(3)
        self._metal_layer.setFrame_(((0, 0), (frame.size.width, frame.size.height)))

        content.layer().addSublayer_(self._metal_layer)
        self._window.setContentView_(content)
        self._window.orderFrontRegardless()
        # Register in class-level set so other compositors exclude us
        try:
            wid = int(self._window.windowNumber())
            FullScreenCompositor._active_compositor_windows.add(wid)
        except Exception:
            pass

        logger.info(
            "FullScreenCompositor: window %dx%d (%.0fx%.0f px, scale=%.1f)",
            int(frame.size.width), int(frame.size.height), pixel_w, pixel_h, scale,
        )

    def _destroy_fullscreen_window(self):
        if self._window is not None:
            try:
                wid = int(self._window.windowNumber())
                FullScreenCompositor._active_compositor_windows.discard(wid)
            except Exception:
                pass
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
            self._window = None
        self._metal_layer = None

    # ------------------------------------------------------------------
    # SCK full-display capture
    # ------------------------------------------------------------------

    def _start_capture(self):
        from spoke.backdrop_stream import _load_screencapturekit_bridge, _make_stream_handler_queue

        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            raise RuntimeError("ScreenCaptureKit bridge unavailable")

        SCShareableContent = bridge["SCShareableContent"]

        # Synchronous-ish: use a threading event
        result = {"content": None, "error": None}
        event = threading.Event()

        def got_content(content, *args):
            result["content"] = content
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(got_content)
        event.wait(timeout=5.0)

        content = result["content"]
        if content is None:
            raise RuntimeError("Failed to get shareable content")

        # Find our display
        display = self._match_display(content)
        if display is None:
            raise RuntimeError("No matching display found")

        # Build filter: full display, exclude our compositor window
        SCContentFilter = bridge["SCContentFilter"]
        excluded = self._excluded_windows(content)
        logger.info(
            "FullScreenCompositor: excluding %d windows (IDs: %s, total in snapshot: %d)",
            len(excluded),
            [int(w.windowID()) for w in excluded] if excluded else [],
            len(list(content.windows())) if hasattr(content, "windows") else 0,
        )
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, excluded)

        # Configure: full display, no sourceRect
        SCStreamConfiguration = bridge["SCStreamConfiguration"]
        config = SCStreamConfiguration.alloc().init()
        scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
        display_frame = display.frame()
        pixel_w = int(display_frame.size.width * scale)
        pixel_h = int(display_frame.size.height * scale)
        config.setWidth_(pixel_w)
        config.setHeight_(pixel_h)
        config.setQueueDepth_(8)
        config.setShowsCursor_(False)  # real cursor is visible; baked-in cursor ghosts
        config.setPixelFormat_(1111970369)  # kCVPixelFormatType_32BGRA
        # Don't set minimumFrameInterval — let SCK deliver at max rate
        if hasattr(config, "setContentScale_"):
            config.setContentScale_(scale)

        # Create stream
        SCStream = bridge["SCStream"]
        stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, None,
        )

        # Build output handler
        from spoke.backdrop_stream import _build_stream_output_class
        _build_stream_output_class()
        from spoke.backdrop_stream import _ScreenCaptureKitStreamOutput

        # We need a minimal renderer interface for the stream output
        renderer_proxy = _CompositorRendererProxy(self, bridge)
        stream_output = _ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(renderer_proxy)

        self._stream_handler_queue = _make_stream_handler_queue("ai.spoke.fullscreen-compositor")

        result = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            stream_output,
            bridge["SCStreamOutputTypeScreen"],
            self._stream_handler_queue,
            None,
        )
        if isinstance(result, tuple):
            success = result[0]
        else:
            success = bool(result)
        if not success:
            raise RuntimeError("addStreamOutput failed")

        started_event = threading.Event()
        started_result = {"error": None}

        def on_started(*args):
            if args:
                started_result["error"] = args[0]
            started_event.set()

        stream.startCaptureWithCompletionHandler_(on_started)
        started_event.wait(timeout=5.0)

        if started_result["error"] is not None:
            raise RuntimeError(f"startCapture failed: {started_result['error']}")

        self._stream = stream
        self._stream_output = stream_output
        logger.info("FullScreenCompositor: SCK capture started (%dx%d)", pixel_w, pixel_h)

    def _stop_capture(self):
        stream = self._stream
        self._stream = None
        self._stream_output = None
        if stream is not None:
            try:
                stream.stopCaptureWithCompletionHandler_(lambda *args: None)
            except Exception:
                pass

    def _match_display(self, content):
        from spoke.backdrop_stream import _screen_display_id
        displays = list(content.displays()) if hasattr(content, "displays") else []
        if not displays:
            return None
        screen_id = _screen_display_id(self._screen)
        if screen_id is None:
            return displays[0]
        for d in displays:
            try:
                if int(d.displayID()) == screen_id:
                    return d
            except Exception:
                continue
        return displays[0]

    def set_excluded_window_ids(self, window_ids: list[int]) -> None:
        """Additional window IDs to exclude from capture (e.g. the overlay window)."""
        self._extra_excluded_ids = set(int(x) for x in window_ids)

    def _excluded_windows(self, content):
        """Exclude all compositor windows + extra windows from capture."""
        exclude_ids = set(getattr(self, "_extra_excluded_ids", set()))
        # Exclude all active compositor windows (prevents feedback loops)
        exclude_ids.update(FullScreenCompositor._active_compositor_windows)
        if self._window is not None:
            try:
                exclude_ids.add(int(self._window.windowNumber()))
            except Exception:
                pass
        if not exclude_ids:
            return []
        excluded = []
        for w in (list(content.windows()) if hasattr(content, "windows") else []):
            try:
                if int(w.windowID()) in exclude_ids:
                    excluded.append(w)
            except Exception:
                continue
        return excluded

    def submit_iosurface(self, iosurface, *, width: int, height: int, pixel_buffer=None):
        """Called from SCK handler queue — must never block.

        If pixel_buffer is provided, we hold a reference to it to prevent
        the SCK buffer pool from recycling the IOSurface before the
        display link thread renders it.
        """
        with self._lock:
            self._latest_iosurface = iosurface
            self._latest_pixel_buffer = pixel_buffer  # prevent recycling
            self._latest_width = width
            self._latest_height = height

    # ------------------------------------------------------------------
    # CVDisplayLink render loop
    # ------------------------------------------------------------------

    def _start_display_link(self):
        from spoke.metal_warp import _create_display_link, _start_display_link
        self._display_link = _create_display_link(self._on_display_link)
        if self._display_link is None:
            raise RuntimeError("Failed to create CVDisplayLink")
        _start_display_link(self._display_link)
        self._last_report_time = time.monotonic()

    def _stop_display_link(self):
        dl = self._display_link
        self._display_link = None
        if dl is not None:
            from spoke.metal_warp import _stop_display_link
            try:
                _stop_display_link(dl)
            except Exception:
                pass

    def _on_display_link(self):
        """Called from CVDisplayLink thread at display refresh rate."""
        if not self._running:
            return

        with self._lock:
            iosurface = self._latest_iosurface
            w = self._latest_width
            h = self._latest_height
            config = self._shell_config

        if iosurface is None or w <= 0 or h <= 0 or config is None:
            return

        self._frame_count += 1
        self._interval_frame_count += 1

        now = time.monotonic()
        elapsed = now - self._last_report_time
        if elapsed >= 5.0:
            fps = self._interval_frame_count / elapsed if elapsed > 0 else 0
            pfps = self._interval_presented / elapsed if elapsed > 0 else 0
            logger.info(
                "Compositor: %d ticks (%.1f fps), %d presented (%.1f fps)",
                self._interval_frame_count, fps,
                self._interval_presented, pfps,
            )
            self._last_report_time = now
            self._interval_frame_count = 0
            self._interval_presented = 0

        if self._metal_layer is None:
            return

        try:
            # Validate config before acquiring a drawable — an empty or
            # missing config would default to a full-screen warp that
            # flashes the entire display.
            warp_config = dict(config)
            if "content_width_points" not in warp_config or "center_x" not in warp_config:
                return

            if self._last_drawable_size != (w, h):
                self._metal_layer.setDrawableSize_((w, h))
                self._last_drawable_size = (w, h)
            drawable = self._metal_layer.nextDrawable()
            if drawable is None:
                return

            if self._pipeline.warp_to_drawable(
                iosurface, drawable, width=w, height=h, shell_config=warp_config,
            ):
                self._presented_count += 1
                self._interval_presented += 1

            elif self._frame_count <= 5:
                logger.info("Compositor tick[%d]: warp_to_drawable returned False", self._frame_count)
        except Exception:
            if self._frame_count <= 10:
                logger.info("Compositor tick[%d] failed", self._frame_count, exc_info=True)


class _CompositorRendererProxy:
    """Minimal proxy so _ScreenCaptureKitStreamOutput can deliver frames."""

    def __init__(self, compositor: FullScreenCompositor, bridge: dict):
        self._compositor = compositor
        self._bridge = bridge
        self._diag_n = 0

    def _consume_sample_buffer(self, sample_buffer, output_type):
        """Extract IOSurface from SCK sample buffer and submit to compositor."""
        bridge = self._bridge
        self._diag_n += 1

        if sample_buffer is None:
            return
        # Unwrap PyObjCPointer
        if hasattr(sample_buffer, "pointerAsInteger"):
            try:
                ptr = sample_buffer.pointerAsInteger
                sample_buffer = objc.objc_object(c_void_p=ptr)
            except Exception:
                return

        cv_lib = bridge.get("_cv_lib")
        if cv_lib is None:
            return

        # Try native bridge first, then direct bridge, then ctypes
        pb = None
        native_get = bridge.get("_native_GetImageBuffer")
        direct_get = bridge.get("_cm_direct_GetImageBuffer")
        if native_get is not None:
            try:
                pb = native_get(sample_buffer)
            except Exception:
                pb = None
        if pb is None and direct_get is not None:
            try:
                pb = direct_get(sample_buffer)
            except Exception:
                pb = None
        if pb is None:
            cm_lib = bridge.get("_cm_lib")
            if cm_lib is not None:
                raw_sb = objc.pyobjc_id(sample_buffer)
                pb_ptr = cm_lib.CMSampleBufferGetImageBuffer(raw_sb)
                if pb_ptr:
                    pb = objc.objc_object(c_void_p=pb_ptr)

        if pb is None:
            if self._diag_n <= 5:
                logger.info("Compositor frame[%d]: no pixel buffer", self._diag_n)
            return

        raw_pb = objc.pyobjc_id(pb)
        ios = cv_lib.CVPixelBufferGetIOSurface(raw_pb)
        if not ios:
            if self._diag_n <= 5:
                logger.info("Compositor frame[%d]: no IOSurface", self._diag_n)
            return

        ios_obj = objc.objc_object(c_void_p=ios)
        cv_lib.CVPixelBufferGetWidth.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetWidth.restype = ctypes.c_size_t
        cv_lib.CVPixelBufferGetHeight.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetHeight.restype = ctypes.c_size_t
        w = int(cv_lib.CVPixelBufferGetWidth(raw_pb))
        h = int(cv_lib.CVPixelBufferGetHeight(raw_pb))

        if self._diag_n <= 8:
            # Log pixel format to verify BGRA delivery
            pf = 0
            try:
                cv_lib.CVPixelBufferGetPixelFormatType.argtypes = [ctypes.c_void_p]
                cv_lib.CVPixelBufferGetPixelFormatType.restype = ctypes.c_uint32
                pf = cv_lib.CVPixelBufferGetPixelFormatType(raw_pb)
            except Exception:
                pass
            pf_str = pf.to_bytes(4, 'big').decode('ascii', errors='replace') if pf else '?'
            logger.info("Compositor frame[%d]: %dx%d IOSurface ptr=%s pb=%s fmt=%s(%d)", self._diag_n, w, h, hex(ios), hex(objc.pyobjc_id(pb)), pf_str, pf)

        if w > 0 and h > 0:
            self._compositor.submit_iosurface(ios_obj, width=w, height=h, pixel_buffer=pb)
