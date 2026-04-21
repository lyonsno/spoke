"""Full-screen compositor for optical shell warp.

Captures the entire display via ScreenCaptureKit, applies the capsule
warp at the overlay's screen position, and presents the full frame
via a borderless full-screen CAMetalLayer window. Every pixel on
screen is ours, so there is no seam between warped and unwarped content.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import threading
import time

import objc

logger = logging.getLogger(__name__)

_shared_overlay_hosts = {}


def _summarize_shell_config(config: dict) -> dict:
    keys = (
        "overlay_kind",
        "center_x",
        "center_y",
        "content_width_points",
        "content_height_points",
        "corner_radius_points",
        "min_brightness",
    )
    summary = {}
    for key in keys:
        value = config.get(key)
        if isinstance(value, (int, float)):
            summary[key] = round(float(value), 2)
        elif value is not None:
            summary[key] = value
    return summary


def _load_screencapturekit_bridge():
    from spoke.backdrop_stream import _load_screencapturekit_bridge as _load_bridge

    return _load_bridge()


def _make_stream_handler_queue(label: str):
    from spoke.backdrop_stream import _make_stream_handler_queue as _make_queue

    return _make_queue(label)


def _screen_registry_key(screen):
    try:
        from spoke.backdrop_stream import _screen_display_id

        display_id = _screen_display_id(screen)
        if display_id is not None:
            return ("display", int(display_id))
    except Exception:
        pass
    return ("object", id(screen))


def _scaled_shell_config_for_overlay(*, screen, window, content_view, shell_config: dict | None):
    """Project overlay-local shell config into fullscreen compositor pixel space."""
    if shell_config is None or screen is None or window is None or content_view is None:
        return None
    config = dict(shell_config)
    scale = screen.backingScaleFactor() if hasattr(screen, "backingScaleFactor") else 2.0
    screen_frame = screen.frame()
    win_frame = window.frame()
    content_frame = content_view.frame()

    capsule_cx = win_frame.origin.x + content_frame.origin.x + content_frame.size.width / 2
    capsule_cy_cocoa = win_frame.origin.y + content_frame.origin.y + content_frame.size.height / 2
    capsule_cy_metal = screen_frame.size.height - capsule_cy_cocoa
    config["center_x"] = capsule_cx * scale
    config["center_y"] = capsule_cy_metal * scale

    for key in (
        "content_width_points",
        "content_height_points",
        "corner_radius_points",
        "band_width_points",
        "tail_width_points",
    ):
        if key in config:
            config[key] = float(config[key]) * scale
    return config


def start_overlay_compositor(*, screen, window, content_view, shell_config: dict | None):
    """Register an overlay surface with the shared fullscreen compositor host."""
    config = _scaled_shell_config_for_overlay(
        screen=screen,
        window=window,
        content_view=content_view,
        shell_config=shell_config,
    )
    if config is None:
        return None
    registry_key = _screen_registry_key(screen)
    host = _shared_overlay_hosts.get(registry_key)
    if host is None:
        host = _SharedOverlayCompositorHost(screen=screen, registry_key=registry_key)
        _shared_overlay_hosts[registry_key] = host
    try:
        client_id = f"overlay:{int(window.windowNumber())}"
    except Exception:
        client_id = f"overlay:{id(window)}"
    session = host.register_client(client_id, config)
    if session is None and not host.has_clients:
        _shared_overlay_hosts.pop(registry_key, None)
    return session


class _OverlayCompositorSession:
    """Per-overlay handle backed by a shared fullscreen compositor host."""

    def __init__(self, host, client_id: str):
        self._host = host
        self._client_id = client_id
        self._sampled_brightness = 0.5
        self._stopped = False

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._host.remove_client(self._client_id)

    def update_shell_config(self, config: dict) -> None:
        if self._stopped:
            return
        self._host.update_client_config(self._client_id, config)

    def update_shell_config_key(self, key: str, value) -> None:
        if self._stopped:
            return
        self._host.update_client_config_key(self._client_id, key, value)

    def refresh_brightness(self) -> None:
        if self._stopped:
            return
        self._sampled_brightness = self._host.sample_brightness(self._client_id)

    @property
    def sampled_brightness(self) -> float:
        return self._sampled_brightness

    @property
    def active_client_count(self) -> int:
        if self._stopped:
            return 0
        return self._host.active_client_count


class _SharedOverlayCompositorHost:
    """One fullscreen compositor window per screen, shared by many overlays."""

    def __init__(self, *, screen, registry_key):
        self._screen = screen
        self._registry_key = registry_key
        self._compositor = FullScreenCompositor(screen)
        self._clients: dict[str, dict] = {}
        self._client_window_ids: dict[str, int] = {}
        self._running = False

    @property
    def has_clients(self) -> bool:
        return bool(self._clients)

    @property
    def active_client_count(self) -> int:
        return len(self._clients)

    def register_client(self, client_id: str, config: dict):
        self._clients[client_id] = dict(config)
        try:
            self._client_window_ids[client_id] = int(client_id.rsplit(":", 1)[1])
        except Exception:
            self._client_window_ids.pop(client_id, None)
        self._refresh_excluded_window_ids()
        if not self._running:
            if not self._compositor.start(self._ordered_shell_configs()):
                self._clients.pop(client_id, None)
                self._client_window_ids.pop(client_id, None)
                self._refresh_excluded_window_ids()
                return None
            self._running = True
        else:
            self._compositor.reset_temporal_state()
            self._compositor.update_shell_configs(self._ordered_shell_configs())
        self._log_state("register_client", client_id)
        return _OverlayCompositorSession(self, client_id)

    def remove_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)
        self._client_window_ids.pop(client_id, None)
        self._refresh_excluded_window_ids()
        if not self._running:
            return
        if self._clients:
            self._compositor.reset_temporal_state()
            self._compositor.update_shell_configs(self._ordered_shell_configs())
            self._log_state("remove_client", client_id)
            return
        self._compositor.stop()
        self._running = False
        _shared_overlay_hosts.pop(self._registry_key, None)
        self._log_state("remove_client", client_id)

    def update_client_config(self, client_id: str, config: dict) -> None:
        if client_id not in self._clients:
            return
        self._clients[client_id] = dict(config)
        if self._running:
            self._compositor.update_shell_configs(self._ordered_shell_configs())
        self._log_state("update_client_config", client_id)

    def update_client_config_key(self, client_id: str, key: str, value) -> None:
        config = self._clients.get(client_id)
        if config is None:
            return
        config[key] = value
        if self._running:
            self._compositor.update_shell_configs(self._ordered_shell_configs())
        self._log_state(f"update_client_config_key:{key}", client_id)

    def sample_brightness(self, client_id: str) -> float:
        config = self._clients.get(client_id)
        if config is None:
            return 0.5
        return self._compositor.sample_brightness_for_config(config)

    def debug_snapshot(self) -> dict:
        return {
            "registry_key": self._registry_key,
            "running": self._running,
            "client_count": len(self._clients),
            "clients": [
                {
                    "client_id": client_id,
                    **_summarize_shell_config(config),
                }
                for client_id, config in self._clients.items()
            ],
        }

    def _ordered_shell_configs(self) -> list[dict]:
        def _overlay_priority(item: tuple[str, dict]) -> tuple[int, int]:
            _, config = item
            kind = config.get("overlay_kind")
            if kind == "preview":
                return (0, 0)
            if kind == "assistant":
                return (1, 0)
            return (0, 1)

        ordered_items = sorted(self._clients.items(), key=_overlay_priority)
        return [dict(config) for _, config in ordered_items]

    def _refresh_excluded_window_ids(self) -> None:
        self._compositor.set_excluded_window_ids(sorted(self._client_window_ids.values()))

    def _log_state(self, action: str, client_id: str | None = None) -> None:
        snapshot = self.debug_snapshot()
        logger.info(
            "SharedOverlayHost[%s]: %s client=%s clients=%d snapshot=%s",
            self._registry_key,
            action,
            client_id,
            snapshot["client_count"],
            snapshot["clients"],
        )


class FullScreenCompositor:
    """Full-display capture -> Metal warp -> full-screen presentation."""

    _active_compositor_windows: set[int] = set()

    def __init__(self, screen):
        self._screen = screen
        self._lock = threading.Lock()
        self._running = False

        from spoke.metal_warp import get_metal_warp_pipeline

        self._pipeline = get_metal_warp_pipeline()
        if self._pipeline is None:
            raise RuntimeError("Metal warp pipeline unavailable")

        self._latest_iosurface = None
        self._latest_pixel_buffer = None
        self._latest_width = 0
        self._latest_height = 0
        self._shell_configs = []
        self._window = None
        self._metal_layer = None
        self._display_link = None
        self._stream = None
        self._stream_output = None
        self._stream_renderer_proxy = None
        self._stream_handler_queue = None
        self._capture_content = None
        self._capture_display = None
        self._extra_excluded_ids: set[int] = set()

        self._frame_count = 0
        self._presented_count = 0
        self._last_report_time = 0.0
        self._interval_frame_count = 0
        self._interval_presented = 0
        self._last_drawable_size = (0, 0)

        self._sampled_brightness = 0.5

    def start(self, shell_config: dict | list[dict]) -> bool:
        if self._running:
            return True
        try:
            self._shell_configs = self._normalize_shell_configs(shell_config)
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
        self._running = False
        self._stop_display_link()
        self._stop_capture()
        self._destroy_fullscreen_window()
        with self._lock:
            self._latest_iosurface = None
            self._latest_pixel_buffer = None
        logger.info(
            "FullScreenCompositor: stopped (%d presented / %d ticks)",
            self._presented_count,
            self._frame_count,
        )

    def update_shell_configs(self, configs: list[dict]) -> None:
        with self._lock:
            self._shell_configs = self._normalize_shell_configs(configs)

    def update_shell_config(self, config: dict) -> None:
        self.update_shell_configs([config] if config else [])

    def update_shell_config_key(self, key: str, value) -> None:
        with self._lock:
            if self._shell_configs:
                self._shell_configs[0][key] = value

    def reset_temporal_state(self) -> None:
        self._pipeline.reset_temporal_state()

    @property
    def sampled_brightness(self) -> float:
        return self._sampled_brightness

    def refresh_brightness(self) -> None:
        with self._lock:
            config = self._shell_configs[0] if self._shell_configs else None
            w = self._latest_width
            h = self._latest_height
        self._sampled_brightness = self.sample_brightness_for_config(config, w=w, h=h)

    def sample_brightness_for_config(self, config: dict | None, *, w: int | None = None, h: int | None = None) -> float:
        with self._lock:
            width = self._latest_width if w is None else w
            height = self._latest_height if h is None else h
        if config is None or width <= 0 or height <= 0:
            return self._sampled_brightness
        brightness = self._sample_iosurface_brightness(None, width, height, config)
        if brightness is not None:
            self._sampled_brightness = brightness
        return self._sampled_brightness

    def _sample_iosurface_brightness(self, iosurface, w, h, config):
        try:
            from Quartz import (
                CGDataProviderCopyData,
                CGImageGetDataProvider,
                CGImageGetHeight,
                CGImageGetWidth,
                CGWindowListCreateImage,
                kCGWindowListOptionOnScreenBelowWindow,
            )

            scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
            cx_pts = config.get("center_x", w * 0.5) / scale
            cy_pts = config.get("center_y", h * 0.5) / scale
            rw_pts = config.get("content_width_points", w) / scale * 0.5
            rh_pts = config.get("content_height_points", h) / scale * 0.5

            screen_frame = self._screen.frame()
            capture_x = screen_frame.origin.x + cx_pts - rw_pts
            capture_y = screen_frame.origin.y + cy_pts - rh_pts
            capture_w = rw_pts * 2
            capture_h = rh_pts * 2
            if capture_w <= 0 or capture_h <= 0:
                return None

            window_id = int(self._window.windowNumber()) if self._window is not None else 0
            rect = ((capture_x, capture_y), (capture_w, capture_h))
            image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenBelowWindow,
                window_id,
                0,
            )
            if image is None:
                return None

            iw = CGImageGetWidth(image)
            ih = CGImageGetHeight(image)
            data = CGDataProviderCopyData(CGImageGetDataProvider(image))
            if data is None or len(data) == 0 or iw * ih == 0:
                return None

            bpp = len(data) // (iw * ih)
            total = 0.0
            count = 0
            step = max(5, 1)
            for sy in range(0, ih, step):
                for sx in range(0, iw, step):
                    offset = (sy * iw + sx) * bpp
                    if offset + 3 <= len(data):
                        r, g, b = data[offset], data[offset + 1], data[offset + 2]
                        total += (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        count += 1
            if count > 0:
                return total / count
        except Exception:
            pass
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def window(self):
        return self._window

    def _create_fullscreen_window(self):
        from AppKit import (
            NSBackingStoreBuffered,
            NSColor,
            NSView,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
        )

        frame = self._screen.frame()
        scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            0,
            NSBackingStoreBuffered,
            False,
        )
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

        content = NSView.alloc().initWithFrame_(((0, 0), (frame.size.width, frame.size.height)))
        content.setWantsLayer_(True)

        CAMetalLayer = objc.lookUpClass("CAMetalLayer")
        self._metal_layer = CAMetalLayer.alloc().init()
        self._metal_layer.setDevice_(self._pipeline.device)
        self._metal_layer.setPixelFormat_(80)
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
        try:
            FullScreenCompositor._active_compositor_windows.add(int(self._window.windowNumber()))
        except Exception:
            pass

        logger.info(
            "FullScreenCompositor: window %dx%d (%.0fx%.0f px, scale=%.1f)",
            int(frame.size.width),
            int(frame.size.height),
            pixel_w,
            pixel_h,
            scale,
        )

    def _destroy_fullscreen_window(self):
        if self._window is not None:
            try:
                FullScreenCompositor._active_compositor_windows.discard(int(self._window.windowNumber()))
            except Exception:
                pass
            try:
                self._window.orderOut_(None)
            except Exception:
                pass
            self._window = None
        self._metal_layer = None

    def _fetch_shareable_content(self, bridge):
        shareable_content = bridge["SCShareableContent"]
        result = {"content": None}
        event = threading.Event()

        def got_content(content, *args):
            result["content"] = content
            event.set()

        shareable_content.getShareableContentWithCompletionHandler_(got_content)
        event.wait(timeout=5.0)
        return result["content"]

    def _start_capture(self):
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            raise RuntimeError("ScreenCaptureKit bridge unavailable")

        content = self._fetch_shareable_content(bridge)
        if content is None:
            raise RuntimeError("Failed to get shareable content")

        display = self._match_display(content)
        if display is None:
            raise RuntimeError("No matching display found")
        self._capture_content = content
        self._capture_display = display

        SCContentFilter = bridge["SCContentFilter"]
        excluded = self._excluded_windows(content)
        logger.info(
            "FullScreenCompositor: excluding %d windows (IDs: %s, total in snapshot: %d)",
            len(excluded),
            [int(w.windowID()) for w in excluded] if excluded else [],
            len(list(content.windows())) if hasattr(content, "windows") else 0,
        )
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, excluded)

        SCStreamConfiguration = bridge["SCStreamConfiguration"]
        config = SCStreamConfiguration.alloc().init()
        scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
        display_frame = display.frame()
        pixel_w = int(display_frame.size.width * scale)
        pixel_h = int(display_frame.size.height * scale)
        config.setWidth_(pixel_w)
        config.setHeight_(pixel_h)
        config.setQueueDepth_(8)
        config.setShowsCursor_(False)
        config.setPixelFormat_(1111970369)
        if hasattr(config, "setContentScale_"):
            config.setContentScale_(scale)

        SCStream = bridge["SCStream"]
        stream = SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, config, None)

        from spoke.backdrop_stream import _ScreenCaptureKitStreamOutput, _build_stream_output_class

        _build_stream_output_class()
        renderer_proxy = _CompositorRendererProxy(self, bridge)
        stream_output = _ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(renderer_proxy)

        self._stream_handler_queue = _make_stream_handler_queue("ai.spoke.fullscreen-compositor")
        result = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            stream_output,
            bridge["SCStreamOutputTypeScreen"],
            self._stream_handler_queue,
            None,
        )
        success = result[0] if isinstance(result, tuple) else bool(result)
        if not success:
            raise RuntimeError("addStreamOutput failed")

        self._stream = stream
        self._stream_output = stream_output
        self._stream_renderer_proxy = renderer_proxy

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

        logger.info("FullScreenCompositor: SCK capture started (%dx%d)", pixel_w, pixel_h)

    def _stop_capture(self):
        stream = self._stream
        self._stream = None
        self._stream_output = None
        self._stream_renderer_proxy = None
        self._capture_content = None
        self._capture_display = None
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
        for display in displays:
            try:
                if int(display.displayID()) == screen_id:
                    return display
            except Exception:
                continue
        return displays[0]

    def set_excluded_window_ids(self, window_ids: list[int]) -> None:
        self._extra_excluded_ids = set(int(x) for x in window_ids)
        self._refresh_live_capture_filter()

    def _refresh_live_capture_filter(self) -> None:
        stream = self._stream
        if stream is None:
            return
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return
        content = self._fetch_shareable_content(bridge)
        if content is None:
            return
        display = self._match_display(content)
        if display is None:
            return
        self._capture_content = content
        self._capture_display = display
        excluded = self._excluded_windows(content)
        SCContentFilter = bridge["SCContentFilter"]
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, excluded)
        try:
            stream.updateContentFilter_completionHandler_(content_filter, lambda *args: None)
        except Exception:
            logger.debug("FullScreenCompositor: updateContentFilter failed", exc_info=True)

    def _excluded_windows(self, content):
        exclude_ids = set(getattr(self, "_extra_excluded_ids", set()))
        exclude_ids.update(FullScreenCompositor._active_compositor_windows)
        if self._window is not None:
            try:
                exclude_ids.add(int(self._window.windowNumber()))
            except Exception:
                pass
        if not exclude_ids:
            return []
        excluded = []
        for window in (list(content.windows()) if hasattr(content, "windows") else []):
            try:
                if int(window.windowID()) in exclude_ids:
                    excluded.append(window)
            except Exception:
                continue
        return excluded

    def submit_iosurface(self, iosurface, *, width: int, height: int, pixel_buffer=None):
        with self._lock:
            self._latest_iosurface = iosurface
            self._latest_pixel_buffer = pixel_buffer
            self._latest_width = width
            self._latest_height = height

    def _start_display_link(self):
        from spoke.metal_warp import _create_display_link, _start_display_link

        self._display_link = _create_display_link(self._on_display_link)
        if self._display_link is None:
            raise RuntimeError("Failed to create CVDisplayLink")
        _start_display_link(self._display_link)
        self._last_report_time = time.monotonic()

    def _stop_display_link(self):
        display_link = self._display_link
        self._display_link = None
        if display_link is not None:
            from spoke.metal_warp import _stop_display_link

            try:
                _stop_display_link(display_link)
            except Exception:
                pass

    def _on_display_link(self):
        if not self._running:
            return

        with self._lock:
            iosurface = self._latest_iosurface
            w = self._latest_width
            h = self._latest_height
            configs = [dict(config) for config in self._shell_configs]

        if iosurface is None or w <= 0 or h <= 0 or not configs:
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
                self._interval_frame_count,
                fps,
                self._interval_presented,
                pfps,
            )
            self._last_report_time = now
            self._interval_frame_count = 0
            self._interval_presented = 0

        if self._metal_layer is None:
            return

        try:
            if not any("content_width_points" in config and "center_x" in config for config in configs):
                return

            if self._last_drawable_size != (w, h):
                self._metal_layer.setDrawableSize_((w, h))
                self._last_drawable_size = (w, h)
            drawable = self._metal_layer.nextDrawable()
            if drawable is None:
                return

            if self._pipeline.warp_to_drawable(
                iosurface,
                drawable,
                width=w,
                height=h,
                shell_config=configs,
            ):
                self._presented_count += 1
                self._interval_presented += 1
            elif self._frame_count <= 5:
                logger.info("Compositor tick[%d]: warp_to_drawable returned False", self._frame_count)
        except Exception:
            if self._frame_count <= 10:
                logger.info("Compositor tick[%d] failed", self._frame_count, exc_info=True)

    @staticmethod
    def _normalize_shell_configs(config_or_configs) -> list[dict]:
        if not config_or_configs:
            return []
        if isinstance(config_or_configs, dict):
            return [dict(config_or_configs)]
        return [dict(config) for config in config_or_configs if config]


class _CompositorRendererProxy:
    """Minimal proxy so _ScreenCaptureKitStreamOutput can deliver frames."""

    def __init__(self, compositor: FullScreenCompositor, bridge: dict):
        self._compositor = compositor
        self._bridge = bridge
        self._diag_n = 0

    def _consume_sample_buffer(self, sample_buffer, output_type):
        bridge = self._bridge
        self._diag_n += 1

        if sample_buffer is None:
            return
        if hasattr(sample_buffer, "pointerAsInteger"):
            try:
                sample_buffer = objc.objc_object(c_void_p=sample_buffer.pointerAsInteger)
            except Exception:
                return

        cv_lib = bridge.get("_cv_lib")
        if cv_lib is None:
            return

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
            pf = 0
            try:
                cv_lib.CVPixelBufferGetPixelFormatType.argtypes = [ctypes.c_void_p]
                cv_lib.CVPixelBufferGetPixelFormatType.restype = ctypes.c_uint32
                pf = cv_lib.CVPixelBufferGetPixelFormatType(raw_pb)
            except Exception:
                pass
            pf_str = pf.to_bytes(4, "big").decode("ascii", errors="replace") if pf else "?"
            logger.info(
                "Compositor frame[%d]: %dx%d IOSurface ptr=%s pb=%s fmt=%s(%d)",
                self._diag_n,
                w,
                h,
                hex(ios),
                hex(objc.pyobjc_id(pb)),
                pf_str,
                pf,
            )

        if w > 0 and h > 0:
            self._compositor.submit_iosurface(ios_obj, width=w, height=h, pixel_buffer=pb)
