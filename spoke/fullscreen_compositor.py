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
import os
import struct
import threading
import time
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import objc

logger = logging.getLogger(__name__)

if hasattr(objc, "ObjCPointerWarning"):
    warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

_shared_overlay_hosts: dict[tuple[str, int], "_SharedOverlayHost"] = {}
_SCK_TARGET_FPS = max(1, int(float(os.environ.get("SPOKE_FULLSCREEN_COMPOSITOR_FPS", "30"))))
_SCK_FRAME_INTERVAL = (1, _SCK_TARGET_FPS, 0, 0)


@dataclass(frozen=True)
class OverlayClientIdentity:
    client_id: str
    display_id: int | str
    role: Literal["assistant", "preview", "tray", "recovery"]


@dataclass(frozen=True)
class OpticalShellGeometrySnapshot:
    center_x: float
    center_y: float
    content_width_points: float
    content_height_points: float
    corner_radius_points: float
    band_width_points: float
    tail_width_points: float


@dataclass(frozen=True)
class OpticalShellMaterialSnapshot:
    initial_brightness: float
    min_brightness: float = 0.0
    core_magnification: float = 1.0
    ring_amplitude_points: float = 0.0
    tail_amplitude_points: float = 0.0
    mip_blur_strength: float = 1.0
    warp_mode: float = 0.0
    scar_amount: float = 0.0
    scar_seam_length_frac: float = 0.70
    scar_seam_thickness_frac: float = 0.15
    scar_seam_focus_frac: float = 0.34
    scar_vertical_grip: float = 0.20
    scar_horizontal_grip: float = 0.07
    scar_axis_rotation: float = 0.0
    scar_mirrored_lip: float = 0.0
    bleed_zone_frac: float | None = None
    exterior_mix_width_points: float | None = None
    x_squeeze: float | None = None
    y_squeeze: float | None = None
    cleanup_blur_radius_points: float = 0.0
    debug_visualize: bool = False
    debug_grid_spacing_points: float = 18.0


@dataclass(frozen=True)
class OverlayRenderSnapshot:
    identity: OverlayClientIdentity
    generation: int
    visible: bool
    geometry: OpticalShellGeometrySnapshot
    material: OpticalShellMaterialSnapshot
    excluded_window_ids: tuple[int, ...] = ()
    z_index: int = 0
    payload: dict | None = None
    optical_field: Mapping[str, Any] | None = None


def _load_screencapturekit_bridge():
    from spoke.backdrop_stream import _load_screencapturekit_bridge as load_bridge

    return load_bridge()


def _make_stream_handler_queue(name: str):
    from spoke.backdrop_stream import _make_stream_handler_queue as make_queue

    return make_queue(name)


def _build_stream_output_class():
    from spoke.backdrop_stream import _build_stream_output_class as build_stream_output_class

    return build_stream_output_class()


def _screen_registry_key(screen) -> tuple[str, int]:
    try:
        from spoke.backdrop_stream import _screen_display_id

        display_id = _screen_display_id(screen)
        if display_id is not None:
            return ("display", int(display_id))
    except Exception:
        pass
    return ("object", id(screen))


def _normalize_shell_configs(shell_configs) -> list[dict]:
    if shell_configs is None:
        return []
    if isinstance(shell_configs, dict):
        return [dict(shell_configs)]
    configs = []
    for config in shell_configs:
        if config:
            configs.append(dict(config))
    return configs


def _wants_continuous_present(shell_configs: list[dict]) -> bool:
    return any(bool(config.get("continuous_present")) for config in shell_configs)


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _agent_shell_card_text_payload(config: dict) -> dict[str, str]:
    payload = config.get("text")
    if not isinstance(payload, dict):
        return {}
    return {
        "primary": _string(payload.get("primary")).strip(),
        "secondary": _string(payload.get("secondary")).strip(),
        "latest_response": _string(payload.get("latest_response")).strip(),
    }


def _agent_shell_card_text_overlay_specs(
    shell_configs: list[dict] | tuple[dict, ...],
    *,
    screen_width_points: float,
    screen_height_points: float,
    scale: float,
) -> list[dict]:
    specs: list[dict] = []
    for config in shell_configs:
        if not isinstance(config, dict):
            continue
        if config.get("role") not in {"agent_card", "selected_thread"}:
            continue
        text = _agent_shell_card_text_payload(config)
        primary = text.get("primary", "")
        secondary = text.get("secondary", "")
        if not primary and not secondary:
            continue
        try:
            width = max(float(config.get("content_width_points", 0.0)), 1.0)
            height = max(float(config.get("content_height_points", 0.0)), 1.0)
            center_x = float(config.get("center_x", 0.0))
            center_y_top = float(config.get("center_y", 0.0))
        except (TypeError, ValueError):
            continue
        left = max(0.0, min(screen_width_points, center_x - width * 0.5))
        top = max(0.0, min(screen_height_points, center_y_top - height * 0.5))
        inset = max(8.0, min(16.0, width * 0.05))
        specs.append(
            {
                "client_id": _string(config.get("client_id")),
                "text": "\n".join(part for part in (primary, secondary) if part),
                "font_size": 15.0 if config.get("role") == "selected_thread" else 13.0,
                "frame": {
                    "x": round(left + inset, 3),
                    "y": round(top + inset, 3),
                    "width": round(max(1.0, width - inset * 2.0), 3),
                    "height": round(max(1.0, height - inset * 2.0), 3),
                },
            }
        )
    return specs


def _initial_brightness_from_shell_config(config: dict | None, fallback: float) -> float:
    if config is None:
        return fallback
    try:
        value = float(config.get("initial_brightness", fallback))
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, value))


def _configure_stream_frame_interval(config) -> None:
    if hasattr(config, "setMinimumFrameInterval_"):
        config.setMinimumFrameInterval_(_SCK_FRAME_INTERVAL)


def _average_ms(total_ms: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total_ms / count


def _clamp_sample_bounds(start: float, extent: float, limit: int) -> tuple[int, int] | None:
    lo = max(int(start), 0)
    hi = min(int(start + extent), int(limit))
    if hi <= lo:
        return None
    return lo, hi


def _sample_pixel_buffer_brightness(
    pixel_buffer,
    width: int,
    height: int,
    config: dict,
    screen,
) -> float | None:
    """Sample shell-region luminance from the live SCK pixel buffer."""
    if pixel_buffer is None or width <= 0 or height <= 0:
        return None
    bridge = _load_screencapturekit_bridge()
    if bridge is None:
        return None
    cv_lib = bridge.get("_cv_lib")
    if cv_lib is None:
        return None

    raw_pb = objc.pyobjc_id(pixel_buffer)
    readonly = 1
    try:
        cv_lib.CVPixelBufferLockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cv_lib.CVPixelBufferLockBaseAddress.restype = ctypes.c_int
        cv_lib.CVPixelBufferUnlockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        cv_lib.CVPixelBufferUnlockBaseAddress.restype = ctypes.c_int
        cv_lib.CVPixelBufferGetBaseAddress.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetBaseAddress.restype = ctypes.c_void_p
        cv_lib.CVPixelBufferGetBytesPerRow.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetBytesPerRow.restype = ctypes.c_size_t
        cv_lib.CVPixelBufferGetWidth.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetWidth.restype = ctypes.c_size_t
        cv_lib.CVPixelBufferGetHeight.argtypes = [ctypes.c_void_p]
        cv_lib.CVPixelBufferGetHeight.restype = ctypes.c_size_t
    except Exception:
        return None

    try:
        if cv_lib.CVPixelBufferLockBaseAddress(raw_pb, readonly) != 0:
            return None
    except Exception:
        return None

    try:
        base = cv_lib.CVPixelBufferGetBaseAddress(raw_pb)
        bytes_per_row = int(cv_lib.CVPixelBufferGetBytesPerRow(raw_pb))
        buffer_width = int(cv_lib.CVPixelBufferGetWidth(raw_pb) or width)
        buffer_height = int(cv_lib.CVPixelBufferGetHeight(raw_pb) or height)
        if not base or bytes_per_row <= 0 or buffer_width <= 0 or buffer_height <= 0:
            return None

        cx = float(config.get("center_x", buffer_width * 0.5))
        cy = float(config.get("center_y", buffer_height * 0.5))
        rw = float(config.get("content_width_points", buffer_width)) * 0.5
        rh = float(config.get("content_height_points", buffer_height)) * 0.5
        x_bounds = _clamp_sample_bounds(cx - rw, rw * 2.0, buffer_width)
        y_bounds = _clamp_sample_bounds(cy - rh, rh * 2.0, buffer_height)
        if x_bounds is None or y_bounds is None:
            return None

        x0, x1 = x_bounds
        y0, y1 = y_bounds
        data = (ctypes.c_ubyte * (bytes_per_row * buffer_height)).from_address(base)
        total = 0.0
        count = 0
        for sy in range(y0, y1, 5):
            row = sy * bytes_per_row
            for sx in range(x0, x1, 5):
                offset = row + (sx * 4)
                if offset + 2 < len(data):
                    b = data[offset]
                    g = data[offset + 1]
                    r = data[offset + 2]
                    total += (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                    count += 1
        if count <= 0:
            return None
        return total / count
    except Exception:
        return None
    finally:
        try:
            cv_lib.CVPixelBufferUnlockBaseAddress(raw_pb, readonly)
        except Exception:
            pass


def _fps_from_intervals(count: int, total_interval_ms: float) -> float:
    if count <= 1 or total_interval_ms <= 0:
        return 0.0
    return ((count - 1) * 1000.0) / total_interval_ms


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
        self._latest_frame_generation = 0
        self._shell_configs: list[dict] = []
        self._config_generation = 0
        self._rendered_frame_generation = 0
        self._rendered_config_generation = 0
        self._window = None
        self._metal_layer = None
        self._card_text_container_layer = None
        self._card_text_layers: dict[str, Any] = {}
        self._display_link = None
        self._stream = None
        self._stream_output = None
        self._stream_renderer_proxy = None
        self._stream_handler_queue = None
        self._capture_thread = None
        self._capture_start_cancelled = False
        self._extra_excluded_ids = set()
        self._capture_content = None
        self._capture_display = None

        # Diagnostics
        self._frame_count = 0
        self._presented_count = 0
        self._last_report_time = 0.0
        self._interval_frame_count = 0
        self._interval_presented = 0
        self._last_drawable_size = (0, 0)
        self._capture_frame_count = 0
        self._display_link_ticks = 0
        self._duplicate_frames = 0
        self._skipped_frames = 0
        self._brightness_samples = 0
        self._windowserver_brightness_samples = 0
        self._warp_to_drawable_calls = 0
        self._total_capture_frame_interval_ms = 0.0
        self._total_display_link_interval_ms = 0.0
        self._total_presented_interval_ms = 0.0
        self._total_brightness_sample_interval_ms = 0.0
        self._total_compositor_tick_ms = 0.0
        self._total_presented_frame_ms = 0.0
        self._total_warp_to_drawable_ms = 0.0
        self._total_brightness_sample_ms = 0.0
        self._total_windowserver_brightness_sample_ms = 0.0
        self._last_capture_frame_at = None
        self._last_display_link_at = None
        self._last_presented_at = None
        self._last_brightness_sample_at = None

        # Brightness sampling from the captured IOSurface
        self._sampled_brightness = 0.5
        self._brightness_sample_frame = 0
        _BRIGHTNESS_SAMPLE_INTERVAL_FRAMES = 15  # every ~250ms at 60fps

    def start(self, shell_config: dict) -> bool:
        """Create the full-screen window and start capture + render loop."""
        if self._running:
            return True
        try:
            self._shell_configs = _normalize_shell_configs(shell_config)
            if self._shell_configs:
                self._sampled_brightness = _initial_brightness_from_shell_config(
                    self._shell_configs[0],
                    self._sampled_brightness,
                )
            # Reset temporal accumulation so the first frame doesn't
            # blend with stale content from a previous compositor session.
            if self._pipeline is not None:
                self._pipeline.reset_temporal_state()
            self._create_fullscreen_window()
            self._sync_agent_shell_card_text_layers(self._shell_configs)
            self._start_display_link()
            self._running = True
            self._start_capture_async()
            logger.info("FullScreenCompositor: started")
            return True
        except Exception:
            logger.info("FullScreenCompositor: failed to start", exc_info=True)
            self.stop()
            return False

    def _start_capture_async(self) -> None:
        self._capture_start_cancelled = False
        self._capture_thread = threading.Thread(
            target=self._run_capture_start,
            name="SpokeFullScreenCompositorCapture",
            daemon=True,
        )
        self._capture_thread.start()

    def _run_capture_start(self) -> None:
        try:
            self._start_capture()
            if getattr(self, "_capture_start_cancelled", False):
                self._stop_capture()
        except Exception:
            if not getattr(self, "_capture_start_cancelled", False):
                logger.info("FullScreenCompositor: capture failed to start", exc_info=True)
                self._schedule_stop_after_capture_failure()

    def _schedule_stop_after_capture_failure(self) -> None:
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(self.stop)
        except Exception:
            self.stop()

    def stop(self) -> None:
        """Tear down everything."""
        self._capture_start_cancelled = True
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
        self.update_shell_configs([config] if config else [])

    def update_shell_configs(self, shell_configs) -> None:
        """Replace the active shell-config set."""
        normalized = _normalize_shell_configs(shell_configs)
        with self._lock:
            if normalized == self._shell_configs:
                return
            self._shell_configs = normalized
            self._config_generation += 1
        self._sync_agent_shell_card_text_layers(normalized)

    def update_shell_config_key(self, key: str, value) -> None:
        """Update a single key in the shell config without replacing."""
        with self._lock:
            if self._shell_configs:
                if self._shell_configs[0].get(key) == value:
                    return
                self._shell_configs[0][key] = value
                self._config_generation += 1

    def _sync_agent_shell_card_text_layers(self, shell_configs: list[dict]) -> None:
        container = getattr(self, "_card_text_container_layer", None)
        if container is None:
            return
        try:
            screen_frame = self._screen.frame()
            screen_width = float(screen_frame.size.width)
            screen_height = float(screen_frame.size.height)
            scale = (
                float(self._screen.backingScaleFactor())
                if hasattr(self._screen, "backingScaleFactor")
                else 2.0
            )
        except Exception:
            return
        specs = _agent_shell_card_text_overlay_specs(
            shell_configs,
            screen_width_points=screen_width,
            screen_height_points=screen_height,
            scale=scale,
        )
        if specs:
            logger.info("Agent Shell smoke labels synced: %d", len(specs))
        active_ids = {spec["client_id"] for spec in specs}
        layers = getattr(self, "_card_text_layers", {})
        for client_id, layer in list(layers.items()):
            if client_id in active_ids:
                continue
            try:
                layer.removeFromSuperlayer()
            except Exception:
                pass
            layers.pop(client_id, None)
        for spec in specs:
            client_id = spec["client_id"]
            layer = layers.get(client_id)
            if layer is None:
                try:
                    from Quartz import CATextLayer, CGColorCreateSRGB

                    layer = CATextLayer.alloc().init()
                    layer.setWrapped_(True)
                    layer.setAlignmentMode_("left")
                    layer.setTruncationMode_("end")
                    layer.setContentsScale_(scale)
                    if hasattr(layer, "setZPosition_"):
                        layer.setZPosition_(10000.0)
                    layer.setForegroundColor_(CGColorCreateSRGB(0.05, 0.06, 0.07, 0.96))
                    if hasattr(layer, "setShadowOpacity_"):
                        layer.setShadowOpacity_(0.35)
                    if hasattr(layer, "setShadowRadius_"):
                        layer.setShadowRadius_(1.5)
                    if hasattr(layer, "setShadowOffset_"):
                        layer.setShadowOffset_((0.0, -1.0))
                    container.addSublayer_(layer)
                except Exception:
                    logger.debug("Failed to create Agent Shell smoke text layer", exc_info=True)
                    continue
                layers[client_id] = layer
            frame = spec["frame"]
            try:
                layer.setFrame_(
                    (
                        (frame["x"], frame["y"]),
                        (frame["width"], frame["height"]),
                    )
                )
                layer.setString_(spec["text"])
                layer.setFontSize_(spec["font_size"])
                if hasattr(layer, "setHidden_"):
                    layer.setHidden_(False)
            except Exception:
                logger.debug("Failed to update Agent Shell smoke text layer", exc_info=True)
        self._card_text_layers = layers

    @property
    def sampled_brightness(self) -> float:
        """Average brightness of the capsule region."""
        return self._sampled_brightness

    @property
    def presented_count(self) -> int:
        """Number of frames the compositor has successfully presented."""
        return self._presented_count

    def _ensure_diagnostics_fields(self) -> None:
        defaults = {
            "_capture_frame_count": 0,
            "_display_link_ticks": 0,
            "_duplicate_frames": 0,
            "_skipped_frames": 0,
            "_brightness_samples": 0,
            "_windowserver_brightness_samples": 0,
            "_warp_to_drawable_calls": 0,
            "_total_capture_frame_interval_ms": 0.0,
            "_total_display_link_interval_ms": 0.0,
            "_total_presented_interval_ms": 0.0,
            "_total_brightness_sample_interval_ms": 0.0,
            "_total_compositor_tick_ms": 0.0,
            "_total_presented_frame_ms": 0.0,
            "_total_warp_to_drawable_ms": 0.0,
            "_total_brightness_sample_ms": 0.0,
            "_total_windowserver_brightness_sample_ms": 0.0,
            "_last_capture_frame_at": None,
            "_last_display_link_at": None,
            "_last_presented_at": None,
            "_last_brightness_sample_at": None,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def diagnostics_snapshot(self) -> dict[str, float | int]:
        """Return compositor residency counters and cheap timing proxies."""
        self._ensure_diagnostics_fields()
        with self._lock:
            capture_frames = int(self._capture_frame_count)
            display_ticks = int(self._display_link_ticks)
            presented_frames = int(self._presented_count)
            brightness_samples = int(self._brightness_samples)
            windowserver_brightness_samples = int(self._windowserver_brightness_samples)
            warp_calls = int(self._warp_to_drawable_calls)
            capture_interval_ms = float(self._total_capture_frame_interval_ms)
            display_interval_ms = float(self._total_display_link_interval_ms)
            presented_interval_ms = float(self._total_presented_interval_ms)
            brightness_interval_ms = float(self._total_brightness_sample_interval_ms)
            total_compositor_tick_ms = float(self._total_compositor_tick_ms)
            total_presented_frame_ms = float(self._total_presented_frame_ms)
            total_warp_to_drawable_ms = float(self._total_warp_to_drawable_ms)
            total_brightness_sample_ms = float(self._total_brightness_sample_ms)
            total_windowserver_brightness_sample_ms = float(
                self._total_windowserver_brightness_sample_ms
            )
            duplicate_frames = int(self._duplicate_frames)
            skipped_frames = int(self._skipped_frames)
        diagnostics = {
            "capture_frames": capture_frames,
            "capture_fps": _fps_from_intervals(capture_frames, capture_interval_ms),
            "display_link_ticks": display_ticks,
            "display_link_fps": _fps_from_intervals(display_ticks, display_interval_ms),
            "presented_frames": presented_frames,
            "presented_fps": _fps_from_intervals(presented_frames, presented_interval_ms),
            "skipped_frames": skipped_frames,
            "duplicate_frames": duplicate_frames,
            "brightness_samples": brightness_samples,
            "windowserver_brightness_samples": windowserver_brightness_samples,
            "avg_capture_frame_interval_ms": _average_ms(
                capture_interval_ms,
                max(capture_frames - 1, 0),
            ),
            "avg_display_link_interval_ms": _average_ms(
                display_interval_ms,
                max(display_ticks - 1, 0),
            ),
            "avg_presented_interval_ms": _average_ms(
                presented_interval_ms,
                max(presented_frames - 1, 0),
            ),
            "avg_brightness_sample_interval_ms": _average_ms(
                brightness_interval_ms,
                max(brightness_samples - 1, 0),
            ),
            "avg_compositor_tick_ms": _average_ms(
                total_compositor_tick_ms,
                display_ticks,
            ),
            "avg_presented_frame_ms": _average_ms(
                total_presented_frame_ms,
                presented_frames,
            ),
            "avg_warp_to_drawable_ms": _average_ms(
                total_warp_to_drawable_ms,
                warp_calls,
            ),
            "avg_brightness_sample_ms": _average_ms(
                total_brightness_sample_ms,
                brightness_samples,
            ),
            "avg_windowserver_brightness_sample_ms": _average_ms(
                total_windowserver_brightness_sample_ms,
                windowserver_brightness_samples,
            ),
        }
        pipeline = getattr(self, "_pipeline", None)
        pipeline_diagnostics = getattr(pipeline, "diagnostics_snapshot", None)
        if callable(pipeline_diagnostics):
            try:
                diagnostics.update(dict(pipeline_diagnostics()))
            except Exception:
                logger.debug("Metal pipeline diagnostics snapshot failed", exc_info=True)
        return diagnostics

    def refresh_brightness(self) -> None:
        """Re-sample capsule region brightness.  Call from main thread."""
        with self._lock:
            configs = list(self._shell_configs)
            iosurface = self._latest_iosurface
            pixel_buffer = self._latest_pixel_buffer
            w = self._latest_width
            h = self._latest_height
        if not configs or w <= 0 or h <= 0:
            return
        self._sample_brightness_with_diagnostics(
            iosurface,
            w,
            h,
            configs[0],
            pixel_buffer,
        )

    def sample_brightness_for_config(self, config: dict) -> float:
        """Sample brightness for a specific shell config."""
        with self._lock:
            iosurface = self._latest_iosurface
            pixel_buffer = self._latest_pixel_buffer
            w = self._latest_width
            h = self._latest_height
        if config is None or w <= 0 or h <= 0:
            return self._sampled_brightness
        self._sample_brightness_with_diagnostics(
            iosurface,
            w,
            h,
            config,
            pixel_buffer,
        )
        return self._sampled_brightness

    def _sample_brightness_with_diagnostics(
        self,
        iosurface,
        w,
        h,
        config,
        pixel_buffer=None,
    ) -> None:
        self._ensure_diagnostics_fields()
        sample_start = time.monotonic()
        source = None
        try:
            source = self._sample_iosurface_brightness(iosurface, w, h, config, pixel_buffer)
        finally:
            sample_end = time.monotonic()
            elapsed_ms = max((sample_end - sample_start) * 1000.0, 0.0)
            with self._lock:
                self._brightness_samples += 1
                self._total_brightness_sample_ms += elapsed_ms
                if source == "windowserver":
                    self._windowserver_brightness_samples += 1
                    self._total_windowserver_brightness_sample_ms += elapsed_ms
                if self._last_brightness_sample_at is not None:
                    self._total_brightness_sample_interval_ms += max(
                        (sample_end - self._last_brightness_sample_at) * 1000.0,
                        0.0,
                    )
                self._last_brightness_sample_at = sample_end

    def _sample_iosurface_brightness(self, iosurface, w, h, config, pixel_buffer=None):
        """Sample average luminance of the capsule region.

        Prefer the live SCK pixel buffer already feeding the Metal warp.
        Fall back to a WindowServer rect capture only when the frame buffer
        cannot be read.
        """
        sampled = _sample_pixel_buffer_brightness(pixel_buffer, w, h, config, self._screen)
        if sampled is not None:
            self._sampled_brightness = sampled
            return "pixel_buffer"

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
            return "windowserver"
        except Exception:
            pass  # non-critical — keep previous brightness
        return None

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
        try:
            from Quartz import CALayer

            self._card_text_container_layer = CALayer.alloc().init()
            self._card_text_container_layer.setFrame_(
                ((0, 0), (frame.size.width, frame.size.height))
            )
            if hasattr(self._card_text_container_layer, "setGeometryFlipped_"):
                self._card_text_container_layer.setGeometryFlipped_(True)
            if hasattr(self._card_text_container_layer, "setZPosition_"):
                self._card_text_container_layer.setZPosition_(10000.0)
            content.layer().addSublayer_(self._card_text_container_layer)
        except Exception:
            logger.debug("Agent Shell smoke text layer unavailable", exc_info=True)
            self._card_text_container_layer = None
            self._card_text_layers = {}
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
        self._card_text_container_layer = None
        self._card_text_layers = {}

    # ------------------------------------------------------------------
    # SCK full-display capture
    # ------------------------------------------------------------------

    def _start_capture(self):
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            raise RuntimeError("ScreenCaptureKit bridge unavailable")

        content = self._fetch_shareable_content(bridge)
        if content is None:
            raise RuntimeError("Failed to get shareable content")
        self._capture_content = content

        # Find our display
        display = self._match_display(content)
        if display is None:
            raise RuntimeError("No matching display found")
        self._capture_display = display

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
        _configure_stream_frame_interval(config)
        if hasattr(config, "setContentScale_"):
            config.setContentScale_(scale)

        # Create stream
        SCStream = bridge["SCStream"]
        stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, None,
        )

        # Build output handler
        _build_stream_output_class()
        from spoke.backdrop_stream import _ScreenCaptureKitStreamOutput

        # We need a minimal renderer interface for the stream output
        renderer_proxy = _CompositorRendererProxy(self, bridge)
        stream_output = _ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(renderer_proxy)
        self._stream_renderer_proxy = renderer_proxy

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
        self._stream_renderer_proxy = None
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
        stream = getattr(self, "_stream", None)
        if stream is not None:
            try:
                self._refresh_capture_filter()
            except Exception:
                logger.debug("Failed to refresh compositor exclusions", exc_info=True)

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

    def _fetch_shareable_content(self, bridge):
        SCShareableContent = bridge["SCShareableContent"]

        result = {"content": None}
        event = threading.Event()

        def got_content(content, *args):
            result["content"] = content
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(got_content)
        event.wait(timeout=5.0)
        return result["content"]

    def _refresh_capture_filter(self):
        stream = getattr(self, "_stream", None)
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
        content_filter = bridge["SCContentFilter"].alloc().initWithDisplay_excludingWindows_(
            display,
            self._excluded_windows(content),
        )
        if not hasattr(stream, "updateContentFilter_completionHandler_"):
            return
        finished = threading.Event()

        def on_updated(*args):
            finished.set()

        stream.updateContentFilter_completionHandler_(content_filter, on_updated)
        finished.wait(timeout=1.0)

    def submit_iosurface(self, iosurface, *, width: int, height: int, pixel_buffer=None):
        """Called from SCK handler queue — must never block.

        If pixel_buffer is provided, we hold a reference to it to prevent
        the SCK buffer pool from recycling the IOSurface before the
        display link thread renders it.
        """
        self._ensure_diagnostics_fields()
        now = time.monotonic()
        with self._lock:
            self._latest_iosurface = iosurface
            self._latest_pixel_buffer = pixel_buffer  # prevent recycling
            self._latest_width = width
            self._latest_height = height
            self._latest_frame_generation += 1
            self._capture_frame_count += 1
            if self._last_capture_frame_at is not None:
                self._total_capture_frame_interval_ms += max(
                    (now - self._last_capture_frame_at) * 1000.0,
                    0.0,
                )
            self._last_capture_frame_at = now

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

        self._ensure_diagnostics_fields()
        tick_start = time.monotonic()
        with self._lock:
            self._display_link_ticks += 1
            if self._last_display_link_at is not None:
                self._total_display_link_interval_ms += max(
                    (tick_start - self._last_display_link_at) * 1000.0,
                    0.0,
                )
            self._last_display_link_at = tick_start
            iosurface = self._latest_iosurface
            w = self._latest_width
            h = self._latest_height
            configs = list(self._shell_configs)
            frame_generation = self._latest_frame_generation
            config_generation = self._config_generation
            continuous_present = _wants_continuous_present(configs)

        try:
            if iosurface is None or w <= 0 or h <= 0 or not configs:
                return
            if (
                frame_generation == self._rendered_frame_generation
                and config_generation == self._rendered_config_generation
                and not continuous_present
            ):
                with self._lock:
                    self._duplicate_frames += 1
                return

            skipped_frames = max(frame_generation - self._rendered_frame_generation - 1, 0)
            if skipped_frames:
                with self._lock:
                    self._skipped_frames += skipped_frames

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

            # Validate config before acquiring a drawable — an empty or
            # missing config would default to a full-screen warp that
            # flashes the entire display.
            warp_configs = [dict(config) for config in configs if config]
            if not warp_configs:
                return
            if any("content_width_points" not in config or "center_x" not in config for config in warp_configs):
                return

            if self._last_drawable_size != (w, h):
                self._metal_layer.setDrawableSize_((w, h))
                self._last_drawable_size = (w, h)
            present_start = time.monotonic()
            drawable = self._metal_layer.nextDrawable()
            if drawable is None:
                return

            warp_start = time.monotonic()
            did_present = self._pipeline.warp_to_drawable(
                iosurface,
                drawable,
                width=w,
                height=h,
                shell_config=warp_configs if len(warp_configs) > 1 else warp_configs[0],
            )
            warp_end = time.monotonic()
            with self._lock:
                self._warp_to_drawable_calls += 1
                self._total_warp_to_drawable_ms += max(
                    (warp_end - warp_start) * 1000.0,
                    0.0,
                )

            if did_present:
                self._presented_count += 1
                self._interval_presented += 1
                present_end = time.monotonic()
                with self._lock:
                    self._total_presented_frame_ms += max(
                        (present_end - present_start) * 1000.0,
                        0.0,
                    )
                    if self._last_presented_at is not None:
                        self._total_presented_interval_ms += max(
                            (present_end - self._last_presented_at) * 1000.0,
                            0.0,
                        )
                    self._last_presented_at = present_end
                self._rendered_frame_generation = frame_generation
                self._rendered_config_generation = config_generation

            elif self._frame_count <= 5:
                logger.info("Compositor tick[%d]: warp_to_drawable returned False", self._frame_count)
        except Exception:
            if self._frame_count <= 10:
                logger.info("Compositor tick[%d] failed", self._frame_count, exc_info=True)
        finally:
            tick_end = time.monotonic()
            with self._lock:
                self._total_compositor_tick_ms += max(
                    (tick_end - tick_start) * 1000.0,
                    0.0,
                )


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


def _display_id_from_registry_key(registry_key: tuple[str, int]) -> int | str:
    kind, value = registry_key
    if kind == "display":
        return int(value)
    return f"{kind}:{value}"


def _snapshot_to_shell_config(snapshot: OverlayRenderSnapshot) -> dict:
    config = {
        "client_id": snapshot.identity.client_id,
        "role": snapshot.identity.role,
        "generation": snapshot.generation,
        "visible": snapshot.visible,
        "z_index": snapshot.z_index,
        "center_x": snapshot.geometry.center_x,
        "center_y": snapshot.geometry.center_y,
        "content_width_points": snapshot.geometry.content_width_points,
        "content_height_points": snapshot.geometry.content_height_points,
        "corner_radius_points": snapshot.geometry.corner_radius_points,
        "band_width_points": snapshot.geometry.band_width_points,
        "tail_width_points": snapshot.geometry.tail_width_points,
        "initial_brightness": snapshot.material.initial_brightness,
        "min_brightness": snapshot.material.min_brightness,
        "core_magnification": snapshot.material.core_magnification,
        "ring_amplitude_points": snapshot.material.ring_amplitude_points,
        "tail_amplitude_points": snapshot.material.tail_amplitude_points,
        "mip_blur_strength": snapshot.material.mip_blur_strength,
        "warp_mode": snapshot.material.warp_mode,
        "scar_amount": snapshot.material.scar_amount,
        "scar_seam_length_frac": snapshot.material.scar_seam_length_frac,
        "scar_seam_thickness_frac": snapshot.material.scar_seam_thickness_frac,
        "scar_seam_focus_frac": snapshot.material.scar_seam_focus_frac,
        "scar_vertical_grip": snapshot.material.scar_vertical_grip,
        "scar_horizontal_grip": snapshot.material.scar_horizontal_grip,
        "scar_axis_rotation": snapshot.material.scar_axis_rotation,
        "scar_mirrored_lip": snapshot.material.scar_mirrored_lip,
        "cleanup_blur_radius_points": snapshot.material.cleanup_blur_radius_points,
        "debug_visualize": snapshot.material.debug_visualize,
        "debug_grid_spacing_points": snapshot.material.debug_grid_spacing_points,
    }
    for key in (
        "bleed_zone_frac",
        "exterior_mix_width_points",
        "x_squeeze",
        "y_squeeze",
    ):
        value = getattr(snapshot.material, key)
        if value is not None:
            config[key] = value
    if snapshot.excluded_window_ids:
        config["excluded_window_ids"] = tuple(snapshot.excluded_window_ids)
    if isinstance(snapshot.payload, dict):
        config.update(snapshot.payload)
    if snapshot.optical_field is not None:
        config["optical_field"] = dict(snapshot.optical_field)
    return config


def _agent_shell_card_surface_configs(
    source_config: dict,
    *,
    previous: Mapping[str, dict] | None = None,
) -> dict[str, dict]:
    optical_fields = source_config.get("agent_shell_card_optical_fields")
    if not isinstance(optical_fields, dict):
        return {}
    requests = optical_fields.get("requests")
    if not isinstance(requests, list):
        return {}
    previous = previous or {}
    surfaces: dict[str, dict] = {}
    for request in requests:
        if not isinstance(request, dict):
            continue
        child_config = request.get("compiled_shell_config")
        if not isinstance(child_config, dict):
            continue
        caller_id = request.get("caller_id")
        client_id = (
            caller_id
            if isinstance(caller_id, str) and caller_id
            else child_config.get("client_id")
        )
        if not isinstance(client_id, str) or not client_id:
            continue
        surface = dict(child_config)
        surface["client_id"] = client_id
        surface["surface_attachment"] = "sibling"
        surface["movable"] = True
        try:
            source_left = float(source_config.get("center_x", 0.0)) - (
                float(source_config.get("content_width_points", 0.0)) * 0.5
            )
            source_bottom = float(source_config.get("center_y", 0.0)) - (
                float(source_config.get("content_height_points", 0.0)) * 0.5
            )
            surface["center_x"] = source_left + float(surface.get("center_x", 0.0))
            surface["center_y"] = source_bottom + float(surface.get("center_y", 0.0))
        except (TypeError, ValueError):
            pass
        existing = previous.get(client_id)
        if isinstance(existing, dict):
            for key in ("center_x", "center_y"):
                if key in existing:
                    surface[key] = existing[key]
        text = request.get("text")
        if isinstance(text, dict):
            surface["text"] = {
                "primary": _string(text.get("primary")),
                "secondary": _string(text.get("secondary")),
                "latest_response": _string(text.get("latest_response")),
            }
        surface["source_client_id"] = source_config.get("client_id", "")
        surface.setdefault("visible", True)
        surfaces[client_id] = surface
    return surfaces


def _snapshot_to_visible_shell_configs(snapshot: OverlayRenderSnapshot) -> list[dict]:
    return [_snapshot_to_shell_config(snapshot)]


def _snapshot_from_shell_config(
    identity: OverlayClientIdentity,
    shell_config: dict,
    *,
    generation: int,
    excluded_window_ids: tuple[int, ...] = (),
) -> OverlayRenderSnapshot:
    config = dict(shell_config)

    def _optional_float(key: str) -> float | None:
        if key not in config or config[key] is None:
            return None
        return float(config[key])

    geometry = OpticalShellGeometrySnapshot(
        center_x=float(config.get("center_x", 0.0)),
        center_y=float(config.get("center_y", 0.0)),
        content_width_points=float(config.get("content_width_points", 0.0)),
        content_height_points=float(config.get("content_height_points", 0.0)),
        corner_radius_points=float(config.get("corner_radius_points", 0.0)),
        band_width_points=float(config.get("band_width_points", 0.0)),
        tail_width_points=float(config.get("tail_width_points", 0.0)),
    )
    material = OpticalShellMaterialSnapshot(
        initial_brightness=float(config.get("initial_brightness", 0.5)),
        min_brightness=float(config.get("min_brightness", 0.0)),
        core_magnification=float(config.get("core_magnification", 1.0)),
        ring_amplitude_points=float(config.get("ring_amplitude_points", 0.0)),
        tail_amplitude_points=float(config.get("tail_amplitude_points", 0.0)),
        mip_blur_strength=float(config.get("mip_blur_strength", 1.0)),
        warp_mode=float(config.get("warp_mode", 0.0)),
        scar_amount=float(config.get("scar_amount", 0.0)),
        scar_seam_length_frac=float(config.get("scar_seam_length_frac", 0.70)),
        scar_seam_thickness_frac=float(config.get("scar_seam_thickness_frac", 0.15)),
        scar_seam_focus_frac=float(config.get("scar_seam_focus_frac", 0.34)),
        scar_vertical_grip=float(config.get("scar_vertical_grip", 0.20)),
        scar_horizontal_grip=float(config.get("scar_horizontal_grip", 0.07)),
        scar_axis_rotation=float(config.get("scar_axis_rotation", 0.0)),
        scar_mirrored_lip=float(config.get("scar_mirrored_lip", 0.0)),
        bleed_zone_frac=_optional_float("bleed_zone_frac"),
        exterior_mix_width_points=_optional_float("exterior_mix_width_points"),
        x_squeeze=_optional_float("x_squeeze"),
        y_squeeze=_optional_float("y_squeeze"),
        cleanup_blur_radius_points=float(config.get("cleanup_blur_radius_points", 0.0)),
        debug_visualize=bool(config.get("debug_visualize", False)),
        debug_grid_spacing_points=float(config.get("debug_grid_spacing_points", 18.0)),
    )
    payload = {
        key: config[key]
        for key in (
            "agent_thread_cards",
            "agent_thread_hud",
            "agent_shell_primitives",
            "agent_shell_card_renderer",
            "agent_shell_card_optical_fields",
            "surface_kind",
        )
        if key in config
    }
    optical_field = None
    if isinstance(config.get("optical_field"), dict):
        optical_field = MappingProxyType(dict(config["optical_field"]))
    return OverlayRenderSnapshot(
        identity=identity,
        generation=generation,
        visible=bool(config.get("visible", True)),
        geometry=geometry,
        material=material,
        excluded_window_ids=tuple(int(v) for v in config.get("excluded_window_ids", excluded_window_ids)),
        z_index=int(config.get("z_index", 0)),
        payload=payload,
        optical_field=optical_field,
    )


class OverlayCompositorRegistry:
    def host_for_screen(self, screen) -> "OverlayCompositorHost":
        registry_key = _screen_registry_key(screen)
        host = _shared_overlay_hosts.get(registry_key)
        if host is None:
            host = OverlayCompositorHost(registry_key, screen)
            _shared_overlay_hosts[registry_key] = host
        return host

    def release_empty_hosts(self) -> None:
        for registry_key, host in list(_shared_overlay_hosts.items()):
            if not host.client_count:
                _shared_overlay_hosts.pop(registry_key, None)


class OverlayCompositorHost:
    def __init__(self, registry_key, screen):
        self._registry_key = registry_key
        self._screen = screen
        self._display_id = _display_id_from_registry_key(registry_key)
        self._compositor = FullScreenCompositor(screen)
        self._clients: dict[str, dict] = {}
        self._agent_shell_card_surfaces: dict[str, dict] = {}
        self._started = False

    @property
    def display_id(self) -> int | str:
        return self._display_id

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def register_client(self, identity: OverlayClientIdentity, *, window, content_view) -> "OverlayCompositorClient":
        entry = self._clients.get(identity.client_id)
        if entry is None:
            entry = {
                "identity": identity,
                "window": window,
                "content_view": content_view,
                "snapshot": None,
                "generation": 0,
                "client": None,
            }
            self._clients[identity.client_id] = entry
        else:
            entry["identity"] = identity
            entry["window"] = window
            entry["content_view"] = content_view
        client = entry.get("client")
        if client is None or getattr(client, "_host", None) is not self:
            client = OverlayCompositorClient(self, identity)
            entry["client"] = client
        else:
            client.identity = identity
            client._client_id = identity.client_id
        return client

    def unregister_client(self, client_id: str) -> None:
        self.release_client(client_id)

    def publish(self, snapshot: OverlayRenderSnapshot) -> bool:
        entry = self._clients.get(snapshot.identity.client_id)
        if entry is None:
            return False
        entry["identity"] = snapshot.identity
        entry["snapshot"] = snapshot
        entry["generation"] = max(int(entry.get("generation", 0)), snapshot.generation)
        if not self._sync_host(start_if_needed=True):
            entry["snapshot"] = None
            if not any(client.get("snapshot") is not None for client in self._clients.values()):
                _shared_overlay_hosts.pop(self._registry_key, None)
            return False
        return True

    def add_client(self, client_id: str, window, content_view, shell_config: dict) -> bool:
        identity = OverlayClientIdentity(client_id=client_id, display_id=self.display_id, role="assistant")
        client = self.register_client(identity, window=window, content_view=content_view)
        return client.update_shell_config(shell_config)

    def update_client_config(self, client_id: str, shell_config: dict) -> bool:
        entry = self._clients.get(client_id)
        if entry is None:
            return False
        generation = int(entry.get("generation", 0)) + 1
        window_ids = self._window_ids_for_entry(entry)
        snapshot = _snapshot_from_shell_config(
            entry["identity"],
            shell_config,
            generation=generation,
            excluded_window_ids=tuple(window_ids),
        )
        return self.publish(snapshot)

    def update_client_configs(self, client_configs: dict[str, dict]) -> bool:
        """Publish several client configs with one compositor-visible state swap."""
        if not client_configs:
            return True
        updates: list[tuple[str, dict, OverlayRenderSnapshot]] = []
        previous: dict[str, tuple[OverlayRenderSnapshot | None, int]] = {}
        for client_id, shell_config in client_configs.items():
            entry = self._clients.get(client_id)
            if entry is None:
                return False
            generation = int(entry.get("generation", 0)) + 1
            window_ids = self._window_ids_for_entry(entry)
            snapshot = _snapshot_from_shell_config(
                entry["identity"],
                shell_config,
                generation=generation,
                excluded_window_ids=tuple(window_ids),
            )
            previous[client_id] = (entry.get("snapshot"), int(entry.get("generation", 0)))
            updates.append((client_id, shell_config, snapshot))
        for client_id, _shell_config, snapshot in updates:
            entry = self._clients[client_id]
            entry["identity"] = snapshot.identity
            entry["snapshot"] = snapshot
            entry["generation"] = max(int(entry.get("generation", 0)), snapshot.generation)
        if not self._sync_host(start_if_needed=True):
            for client_id, (snapshot, generation) in previous.items():
                entry = self._clients.get(client_id)
                if entry is not None:
                    entry["snapshot"] = snapshot
                    entry["generation"] = generation
            if not any(client.get("snapshot") is not None for client in self._clients.values()):
                _shared_overlay_hosts.pop(self._registry_key, None)
            return False
        return True

    def update_client_config_key(self, client_id: str, key: str, value) -> bool:
        entry = self._clients.get(client_id)
        if entry is None:
            return False
        snapshot = entry.get("snapshot")
        if snapshot is None:
            return False
        config = _snapshot_to_shell_config(snapshot)
        config[key] = value
        return self.update_client_config(client_id, config)

    def release_client(self, client_id: str) -> None:
        self._clients.pop(client_id, None)
        self._agent_shell_card_surfaces = {
            surface_id: config
            for surface_id, config in self._agent_shell_card_surfaces.items()
            if config.get("source_client_id") != client_id
        }
        if self._clients:
            self._sync_host()
            return
        if self._started:
            try:
                self._compositor.stop()
            except Exception:
                logger.debug("Failed to stop shared overlay compositor host", exc_info=True)
        self._started = False
        _shared_overlay_hosts.pop(self._registry_key, None)

    def render_snapshots(self) -> tuple[OverlayRenderSnapshot, ...]:
        snapshots = [entry["snapshot"] for entry in self._clients.values() if entry.get("snapshot") is not None]
        return tuple(sorted(snapshots, key=lambda snapshot: (snapshot.z_index, snapshot.identity.client_id)))

    def sample_brightness(self, client_id: str) -> float:
        entry = self._clients.get(client_id)
        if entry is None or entry.get("snapshot") is None:
            return 0.5
        config = _snapshot_to_shell_config(entry["snapshot"])
        sampler = getattr(self._compositor, "sample_brightness_for_config", None)
        if callable(sampler):
            return float(sampler(config))
        return float(getattr(self._compositor, "sampled_brightness", 0.5))

    def refresh_brightness(self, client_id: str) -> None:
        entry = self._clients.get(client_id)
        if entry is None or entry.get("snapshot") is None:
            return
        config = _snapshot_to_shell_config(entry["snapshot"])
        sampler = getattr(self._compositor, "sample_brightness_for_config", None)
        if callable(sampler):
            sampler(config)
            return
        refresher = getattr(self._compositor, "refresh_brightness", None)
        if callable(refresher):
            refresher()

    def sampled_brightness_for_client(self, client_id: str) -> float:
        return self.sample_brightness(client_id)

    def refresh_brightness_for_client(self, client_id: str) -> None:
        self.refresh_brightness(client_id)

    @property
    def presented_count(self) -> int:
        return int(getattr(self._compositor, "presented_count", 0))

    def diagnostics_snapshot(self) -> dict[str, float | int]:
        diagnostics = getattr(self._compositor, "diagnostics_snapshot", None)
        if callable(diagnostics):
            return dict(diagnostics())
        return {
            "capture_frames": 0,
            "capture_fps": 0.0,
            "display_link_ticks": 0,
            "display_link_fps": 0.0,
            "presented_frames": self.presented_count,
            "presented_fps": 0.0,
            "skipped_frames": 0,
            "duplicate_frames": 0,
            "brightness_samples": 0,
            "windowserver_brightness_samples": 0,
            "avg_capture_frame_interval_ms": 0.0,
            "avg_display_link_interval_ms": 0.0,
            "avg_presented_interval_ms": 0.0,
            "avg_brightness_sample_interval_ms": 0.0,
            "avg_compositor_tick_ms": 0.0,
            "avg_presented_frame_ms": 0.0,
            "avg_warp_to_drawable_ms": 0.0,
            "avg_brightness_sample_ms": 0.0,
            "avg_windowserver_brightness_sample_ms": 0.0,
        }

    def debug_snapshot(self) -> dict:
        clients = []
        for snapshot in self.render_snapshots():
            client_snapshot = {"client_id": snapshot.identity.client_id}
            client_snapshot.update(_snapshot_to_shell_config(snapshot))
            clients.append(client_snapshot)
        return {
            "client_count": len(clients),
            "clients": clients,
            "diagnostics": self.diagnostics_snapshot(),
        }

    def _window_ids_for_entry(self, entry: dict) -> list[int]:
        window_ids = []
        window = entry.get("window")
        try:
            window_ids.append(int(window.windowNumber()))
        except Exception:
            pass
        return window_ids

    def _sync_host(self, start_if_needed: bool = False) -> bool:
        overlay_window_ids = []
        for entry in self._clients.values():
            overlay_window_ids.extend(self._window_ids_for_entry(entry))
        snapshots = self.render_snapshots()
        for snapshot in snapshots:
            config = _snapshot_to_shell_config(snapshot)
            if "agent_shell_card_optical_fields" in config:
                self._agent_shell_card_surfaces = _agent_shell_card_surface_configs(
                    config,
                    previous=self._agent_shell_card_surfaces,
                )
        parent_shell_configs = [
            config
            for snapshot in snapshots
            if snapshot.visible
            for config in _snapshot_to_visible_shell_configs(snapshot)
        ]
        card_shell_configs = sorted(
            self._agent_shell_card_surfaces.values(),
            key=lambda config: (
                int(config.get("z_index", 0)),
                str(config.get("client_id", "")),
            ),
        )
        shell_configs = [*parent_shell_configs, *card_shell_configs]
        set_excluded = getattr(self._compositor, "set_excluded_window_ids", None)
        if callable(set_excluded):
            set_excluded(overlay_window_ids)
        if start_if_needed and not self._started:
            if not shell_configs:
                return False
            started = self._compositor.start(shell_configs[0])
            if not started:
                return False
            self._started = True
        update_all = getattr(self._compositor, "update_shell_configs", None)
        if callable(update_all):
            update_all(shell_configs)
        elif shell_configs and hasattr(self._compositor, "update_shell_config"):
            self._compositor.update_shell_config(shell_configs[0])
        return True


_SharedOverlayHost = OverlayCompositorHost


class OverlayCompositorClient:
    def __init__(self, host: OverlayCompositorHost, identity: OverlayClientIdentity | str):
        self._host = host
        if isinstance(identity, OverlayClientIdentity):
            self.identity = identity
        else:
            display_id = getattr(host, "display_id", "unknown")
            self.identity = OverlayClientIdentity(
                client_id=str(identity),
                display_id=display_id,
                role="assistant",
            )
        self._client_id = self.identity.client_id

    def publish(self, snapshot: OverlayRenderSnapshot) -> bool:
        if self._host is None:
            return False
        return self._host.publish(snapshot)

    def update_shell_config(self, shell_config: dict) -> bool:
        if self._host is None:
            return False
        return self._host.update_client_config(self._client_id, shell_config)

    def update_shell_config_key(self, key: str, value) -> bool:
        if self._host is None:
            return False
        return self._host.update_client_config_key(self._client_id, key, value)

    @property
    def sampled_brightness(self) -> float:
        return self.sample_brightness()

    def sample_brightness(self) -> float:
        if self._host is None:
            return 0.5
        sampler = getattr(self._host, "sample_brightness", None)
        if callable(sampler):
            return sampler(self._client_id)
        legacy_sampler = getattr(self._host, "sampled_brightness_for_client", None)
        if callable(legacy_sampler):
            return legacy_sampler(self._client_id)
        return 0.5

    def refresh_brightness(self) -> None:
        if self._host is None:
            return
        refresher = getattr(self._host, "refresh_brightness", None)
        if callable(refresher):
            refresher(self._client_id)
            return
        legacy_refresher = getattr(self._host, "refresh_brightness_for_client", None)
        if callable(legacy_refresher):
            legacy_refresher(self._client_id)

    @property
    def presented_count(self) -> int:
        if self._host is None:
            return 0
        count = getattr(self._host, "presented_count", 0)
        if callable(count):
            count = count()
        try:
            return int(count)
        except (TypeError, ValueError):
            return 0

    def diagnostics_snapshot(self) -> dict[str, float | int]:
        if self._host is None:
            return {}
        diagnostics = getattr(self._host, "diagnostics_snapshot", None)
        if callable(diagnostics):
            return dict(diagnostics())
        return {}

    def release(self) -> None:
        self.stop()

    def stop(self) -> None:
        if self._host is None:
            return
        host = self._host
        self._host = None
        host.release_client(self._client_id)


_OverlayCompositorSession = OverlayCompositorClient


def start_overlay_compositor(
    *,
    screen,
    window,
    content_view,
    shell_config,
    client_id: str = "assistant.command",
    role: str = "assistant",
    registry: OverlayCompositorRegistry | None = None,
):
    registry = registry or OverlayCompositorRegistry()
    host = registry.host_for_screen(screen)
    identity = OverlayClientIdentity(
        client_id=client_id,
        display_id=host.display_id,
        role=role,
    )
    session = host.register_client(identity, window=window, content_view=content_view)
    if not session.update_shell_config(shell_config):
        return None
    return session
