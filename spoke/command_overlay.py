"""Command response overlay.

A semi-transparent overlay for displaying streamed command responses. Visually
kin to the transcription overlay (same ethereal transparency, same floating
treatment) but differentiated by color and rhythm. The input overlay breathes
with voice amplitude. The output overlay pulses with a slow ease-in/ease-out
rhythm — mechanical but gentle, distinct from the organic input. Color is a
slow full-spectrum hue rotation (~6s cycle with velocity undulation), so the
overlay always appears to be roughly one color but you can never quite pin
down which one.
"""

from __future__ import annotations

import logging
import math
import numbers
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSScreen,
    NSScrollView,
    NSTextView,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSMakeRect, NSObject, NSRunLoop, NSTimer
from Quartz import CALayer, CAShapeLayer, CGPathCreateWithRoundedRect

from .backdrop_stream import (
    _apply_optical_shell_warp_ci_image,
    _debug_shell_grid_ci_image,
    make_backdrop_renderer,
)
from .overlay import _OVERLAY_WINDOW_LEVEL

logger = logging.getLogger(__name__)
_BACKDROP_DISPLAY_LAYER_CLASS = None

def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in {"0", "false", "False", "no", "off"}


_OVERLAY_WIDTH = _env("SPOKE_COMMAND_OVERLAY_WIDTH", 600.0)
_OVERLAY_HEIGHT = _env("SPOKE_COMMAND_OVERLAY_HEIGHT", 80.0)
_COMMAND_OVERLAY_WINDOW_LEVEL = _OVERLAY_WINDOW_LEVEL + 1
_OVERLAY_BOTTOM_MARGIN = _env("SPOKE_COMMAND_OVERLAY_BOTTOM_MARGIN", 300.0)
_OVERLAY_TOP_MARGIN = _env("SPOKE_COMMAND_OVERLAY_TOP_MARGIN", 140.0)
_OVERLAY_CORNER_RADIUS = _env("SPOKE_COMMAND_OVERLAY_CORNER_RADIUS", 16.0)
_FONT_SIZE = 15.5
_FADE_IN_S = 0.7
_ENTRANCE_POP_SCALE = 1.015  # ~1mm overshoot on a 600px overlay
_ENTRANCE_POP_S = 0.15
_FADE_OUT_S = 0.5  # fast dismiss fade (750ms total with 250ms hold)
_FADE_STEPS = 15
_DISMISS_DURATION_S = 0.2
_DISMISS_GROW_S = 0.06
_DISMISS_SHRINK_S = _DISMISS_DURATION_S - _DISMISS_GROW_S
_DISMISS_ANIM_FPS = 60.0
_DISMISS_GROW_SCALE = 1.018
_DISMISS_END_SCALE = 0.94

def _max_overlay_height(screen_height: float) -> float:
    return max(_OVERLAY_HEIGHT, screen_height - _OVERLAY_BOTTOM_MARGIN - _OVERLAY_TOP_MARGIN)

# Assistant glow: full spectrum rotation with velocity undulation
_COLOR_CYCLE_PERIOD = _env("SPOKE_COMMAND_COLOR_PERIOD", 6.0)  # seconds per full hue rotation
_COLOR_VELOCITY_PERIOD = 7.0  # velocity oscillation period (out of phase with everything)
_COLOR_VELOCITY_MIN = 0.3  # slowest speed multiplier (dwells)
_COLOR_VELOCITY_MAX = 1.7  # fastest speed multiplier (transitions)
_GLOW_COLOR = (0.6, 0.4, 0.9)  # initial color for setup (violet)
_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_TEXT_ALPHA_MIN", 1.0)
_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_TEXT_ALPHA_MAX", 1.0)
_ASSISTANT_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MIN", 1.0)
_ASSISTANT_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MAX", 1.0)
_BG_ALPHA = _env("SPOKE_COMMAND_BG_ALPHA", 0.715)
_PULSE_PERIOD = _env("SPOKE_COMMAND_PULSE_PERIOD", 2.0)  # base period (seconds)
_PULSE_PERIOD_USER = _PULSE_PERIOD * 1.5  # user text: 50% slower
_PULSE_PERIOD_ASST = 5.0  # assistant text: slow deep breath
_PULSE_PHASE_OFFSET_USER = 0.3  # user starts 30% ahead in phase
_PULSE_HZ = 30.0  # timer frequency for pulse animation

_OUTER_FEATHER = 220.0  # match preview overlay — room for the stretched-exp tails
_OPTICAL_SHELL_FEATHER = 140.0  # ~2 inches — the glow tail is faint but the eye catches
                                # any hard clip, so the window needs room for the full decay
_INNER_GLOW_DEPTH = 30.0
_OUTER_GLOW_PEAK_TARGET = 0.35
_BRIGHTNESS_CHASE = 0.08
_POINTS_PER_CM = 72.0 / 2.54
_COMMAND_BACKDROP_OVERSCAN_CM = _env("SPOKE_COMMAND_BACKDROP_OVERSCAN_CM", 1.5)
_COMMAND_BACKDROP_BLUR_RADIUS = _env("SPOKE_COMMAND_BACKDROP_BLUR_RADIUS", 9.0)
_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER", 3.0
)
_COMMAND_BACKDROP_PULSE_TIERS = max(
    1, int(round(_env("SPOKE_COMMAND_BACKDROP_PULSE_TIERS", 4.0)))
)
_COMMAND_BACKDROP_PULSE_ATTACK = _env("SPOKE_COMMAND_BACKDROP_PULSE_ATTACK", 0.38)
_COMMAND_BACKDROP_PULSE_RELEASE = _env("SPOKE_COMMAND_BACKDROP_PULSE_RELEASE", 0.12)
_COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER", 0.74
)
_COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER", 1.18
)
_COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER", 0.8
)
_COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER", 1.55
)
_COMMAND_BACKDROP_PULSE_OPACITY_MIN = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_OPACITY_MIN", 0.86
)
_COMMAND_BACKDROP_PULSE_OPACITY_MAX = _env(
    "SPOKE_COMMAND_BACKDROP_PULSE_OPACITY_MAX", 1.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION", 1.55
)
_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM", 4.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM", 3.0
)
_COMMAND_BACKDROP_OPTICAL_SHELL_RING_REFRACTION_LEGACY = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_REFRACTION", 2.6
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION_LEGACY = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION", 0.75
)
_COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS",
    (_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM / 10.0)
    * _POINTS_PER_CM
    * _COMMAND_BACKDROP_OPTICAL_SHELL_RING_REFRACTION_LEGACY,
)
_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS",
    (_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM / 10.0)
    * _POINTS_PER_CM
    * _COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION_LEGACY,
)
_COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS", 0.75
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK", 0.08
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK", 0.18
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT", 0.14
)
_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT", 0.38
)
_COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE", 0.16
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE = _env_bool(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", False
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER", 0.18
)
_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS = _env(
    "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS", 18.0
)
_COMMAND_BACKDROP_REFRESH_S = _env("SPOKE_COMMAND_BACKDROP_REFRESH_S", 1.0 / 30.0)
_RUN_LOOP_COMMON_MODE = "NSRunLoopCommonModes"
_EVENT_TRACKING_RUN_LOOP_MODE = "NSEventTrackingRunLoopMode"

# Adaptive compositing for command output.
_USER_TEXT_COLOR_DARK = (0.18, 0.19, 0.22)
_USER_TEXT_COLOR_LIGHT = (0.10, 0.12, 0.16)
_RESPONSE_TEXT_LIGHT_BG_TARGET = (0.07, 0.08, 0.11)
_THINKING_CUTOUT_DARK = (0.05, 0.05, 0.06)
_THINKING_CUTOUT_LIGHT = (0.80, 0.80, 0.78)


def _backdrop_display_layer_class():
    global _BACKDROP_DISPLAY_LAYER_CLASS
    if _BACKDROP_DISPLAY_LAYER_CLASS is not None:
        return _BACKDROP_DISPLAY_LAYER_CLASS
    if not hasattr(objc, "loadBundle") or not hasattr(objc, "lookUpClass"):
        _BACKDROP_DISPLAY_LAYER_CLASS = False
        return None
    try:
        objc.loadBundle(
            "AVFoundation",
            globals(),
            bundle_path="/System/Library/Frameworks/AVFoundation.framework",
        )
        _BACKDROP_DISPLAY_LAYER_CLASS = objc.lookUpClass("AVSampleBufferDisplayLayer")
    except Exception:
        logger.debug("AVSampleBufferDisplayLayer unavailable for command backdrop", exc_info=True)
        _BACKDROP_DISPLAY_LAYER_CLASS = False
    return _BACKDROP_DISPLAY_LAYER_CLASS or None


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _lerp_color(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return tuple(_lerp(s, e, t) for s, e in zip(start, end))


def _background_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    from .overlay import _BG_COLOR_DARK as _PREVIEW_BG_COLOR_DARK, _BG_COLOR_LIGHT as _PREVIEW_BG_COLOR_LIGHT

    return _lerp_color(_PREVIEW_BG_COLOR_DARK, _PREVIEW_BG_COLOR_LIGHT, _clamp01(brightness))


# Compositor graphic mode: fill matches background tone so text
# punch-through reveals the contrasting warped content underneath.
# Dark on dark, light on light — the overlay is a surface, not a glow.
_COMPOSITOR_FILL_DARK = (0.50, 0.51, 0.54)   # light fill on dark backgrounds — faint, translucent
_COMPOSITOR_FILL_LIGHT = (0.04, 0.04, 0.05)   # dark fill on light backgrounds — vivid, near-black


def _compositor_fill_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    # Steep sigmoid: commits to light or dark fill quickly, doesn't
    # linger in a bland mid-tone.  The transition is centered at 0.45
    # (biased slightly toward dark-fill) and covers ~0.15 of the
    # brightness range, so most backgrounds get a decisive choice.
    t = _clamp01(brightness)
    # Remap to steep sigmoid: 6x gain centered at 0.45
    t = _clamp01((t - 0.45) * 6.0 + 0.5)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep for clean edges
    return _lerp_color(_COMPOSITOR_FILL_DARK, _COMPOSITOR_FILL_LIGHT, t)


def _user_text_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(_USER_TEXT_COLOR_DARK, _USER_TEXT_COLOR_LIGHT, _clamp01(brightness))


def _response_color_for_brightness(
    color: tuple[float, float, float],
    brightness: float,
) -> tuple[float, float, float]:
    # Keep the hue-rotating identity, but bias toward a dark endpoint on
    # bright screens so the response remains readable.
    t = _clamp01(brightness) ** 1.15
    return _lerp_color(color, _RESPONSE_TEXT_LIGHT_BG_TARGET, t)


def _thinking_cutout_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(_THINKING_CUTOUT_DARK, _THINKING_CUTOUT_LIGHT, _clamp01(brightness))


def _assistant_text_alpha_for_breath(breath: float) -> float:
    return _lerp(_ASSISTANT_TEXT_ALPHA_MIN, _ASSISTANT_TEXT_ALPHA_MAX, _clamp01(breath))


def _quantize_unit_interval(value: float, steps: int) -> float:
    clamped = _clamp01(value)
    if steps <= 1:
        return clamped
    return round(clamped * (steps - 1)) / float(steps - 1)


def _advance_attack_release(current: float, target: float, *, attack: float, release: float) -> float:
    rate = attack if target > current else release
    return current + (_clamp01(target) - current) * _clamp01(rate)


def _command_backdrop_pulse_style(
    base_blur_radius_points: float,
    base_mask_width_multiplier: float,
    blur_drive: float,
) -> tuple[float, float, float]:
    drive = _quantize_unit_interval(blur_drive, _COMMAND_BACKDROP_PULSE_TIERS)
    blur_radius_points = max(
        0.0,
        base_blur_radius_points
        * _lerp(
            _COMMAND_BACKDROP_PULSE_BLUR_MIN_MULTIPLIER,
            _COMMAND_BACKDROP_PULSE_BLUR_MAX_MULTIPLIER,
            drive,
        ),
    )
    mask_width_multiplier = max(
        0.0,
        base_mask_width_multiplier
        * _lerp(
            _COMMAND_BACKDROP_PULSE_MASK_MIN_MULTIPLIER,
            _COMMAND_BACKDROP_PULSE_MASK_MAX_MULTIPLIER,
            drive,
        ),
    )
    # The sampled backdrop is the replacement background inside the masked region.
    # If we fade it down during the airy phase, the live crisp desktop leaks through
    # the transparent window and the blur reads like a translucent wash instead.
    backdrop_opacity = 1.0
    return blur_radius_points, mask_width_multiplier, backdrop_opacity


def _command_backdrop_blur_target_for_presence(presence: float) -> float:
    return 1.0 - _clamp01(presence)


def _command_optical_shell_config(
    content_width_points: float | None = None,
    content_height_points: float | None = None,
) -> dict[str, float | bool] | None:
    if not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
        return None
    width_points = (
        _OVERLAY_WIDTH if content_width_points is None else max(float(content_width_points), 1.0)
    )
    height_points = (
        _OVERLAY_HEIGHT
        if content_height_points is None
        else max(float(content_height_points), 1.0)
    )
    return {
        "enabled": True,
        "content_width_points": width_points,
        "content_height_points": height_points,
        "corner_radius_points": _OVERLAY_HEIGHT / 4.0,
        "core_magnification": _COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION,
        "band_width_points": _cm_to_points(_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM / 10.0),
        "tail_width_points": _cm_to_points(_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM / 10.0),
        "ring_amplitude_points": _COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS,
        "tail_amplitude_points": _COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS,
        "debug_visualize": _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE,
        "debug_grid_spacing_points": _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS,
        "cleanup_blur_radius_points": _COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS,
    }


def _fill_compositing_filter_for_brightness(brightness: float) -> str | None:
    from .overlay import _fill_compositing_filter_for_brightness as _preview_fill_compositing_filter_for_brightness

    return _preview_fill_compositing_filter_for_brightness(brightness)


def _ease_in(progress: float) -> float:
    clamped = _clamp01(progress)
    return clamped * clamped


def _cm_to_points(cm: float) -> float:
    return max(cm, 0.0) * _POINTS_PER_CM


def _command_backdrop_capture_overscan_points() -> float:
    # The blur neighborhood only needs enough extra image to support a
    # graceful fade beyond the bubble perimeter; it should stay much tighter
    # than the full SDF feather/glow margin.
    return _cm_to_points(_COMMAND_BACKDROP_OVERSCAN_CM)


def _command_backdrop_capture_overscan_pixels(backing_scale: float) -> float:
    return _command_backdrop_capture_overscan_points() * max(backing_scale, 0.0)


def _backdrop_capture_rect(screen_frame, window_frame, content_frame, overscan_points: float):
    overscan = max(overscan_points, 0.0)
    x = window_frame.origin.x + content_frame.origin.x - overscan
    width = content_frame.size.width + 2 * overscan
    height = content_frame.size.height + 2 * overscan
    cocoa_y = window_frame.origin.y + content_frame.origin.y - overscan
    screen_origin_y = getattr(screen_frame.origin, "y", 0.0)
    screen_height = getattr(screen_frame.size, "height", 0.0)
    y = screen_origin_y + screen_height - (cocoa_y - screen_origin_y) - height
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _backdrop_capture_pixel_size(capture_rect, backing_scale: float) -> tuple[float, float]:
    scale = max(backing_scale, 0.0)
    return (
        capture_rect.size.width * scale,
        capture_rect.size.height * scale,
    )


def _pin_timer_to_active_run_loop_modes(timer) -> None:
    if timer is None:
        return
    try:
        run_loop = NSRunLoop.currentRunLoop()
    except Exception:
        return
    for mode in (_RUN_LOOP_COMMON_MODE, _EVENT_TRACKING_RUN_LOOP_MODE):
        try:
            run_loop.addTimer_forMode_(timer, mode)
        except Exception:
            logger.debug("Failed to add command backdrop timer to run loop mode %s", mode, exc_info=True)


def _backdrop_mask_alpha(signed_distance, width: float):
    import numpy as np

    outside = np.exp(-np.sqrt(np.maximum(signed_distance, 0.0) / max(width, 1e-6)))
    return np.where(signed_distance <= 0.0, 1.0, outside).astype(np.float32)


def _command_backdrop_mask_falloff_width(scale: float) -> float:
    return max(scale, 1e-6) * max(_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER, 0.0)


def _boundary_outline_ci_image(extent, shell_config):
    """Faint pill-shaped outline at the sdf=0 boundary of the optical shell."""
    import numpy as np
    from Foundation import NSData
    from Quartz import (
        CIImage,
        CGColorSpaceCreateDeviceRGB,
        CGDataProviderCreateWithCFData,
        CGImageCreate,
        kCGImageAlphaPremultipliedLast,
        kCGRenderingIntentDefault,
    )

    width = max(1, int(round(extent.size.width)))
    height = max(1, int(round(extent.size.height)))
    content_w = float(shell_config.get("content_width_points", width))
    content_h = float(shell_config.get("content_height_points", height))
    capsule_radius = max(content_h * 0.5, 1.0)
    spine_half = max(content_w * 0.5 - capsule_radius, 0.0)

    cx = width * 0.5
    cy = height * 0.5
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5 - cx
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5 - cy

    # Capsule SDF matching the kernel.
    spine_dist = np.maximum(np.abs(xs) - spine_half, 0.0)
    capsule_sdf = np.hypot(spine_dist, ys) - capsule_radius

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    # Only draw boundary on the right half for comparison.
    right_half = xs >= 0.0
    boundary = (np.abs(capsule_sdf) < 1.5) & right_half
    rgba[boundary] = (255, 255, 255, 60)

    payload = NSData.dataWithBytes_length_(rgba.tobytes(), int(rgba.nbytes))
    provider = CGDataProviderCreateWithCFData(payload)
    image = CGImageCreate(
        width, height, 8, 32, width * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast,
        provider, None, False, kCGRenderingIntentDefault,
    )
    ci_image = CIImage.imageWithCGImage_(image)
    return ci_image.imageByCroppingToRect_(extent) if hasattr(ci_image, "imageByCroppingToRect_") else ci_image


class _QuartzBackdropRenderer:
    """Best-effort snapshot renderer for the assistant backdrop prototype."""

    def __init__(self) -> None:
        self._ci_context = None
        self._cached_center_mask_ci = None  # cached CIImage mask
        self._cached_center_mask_key = None  # (mw, mh, content_w, content_h) cache key

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
            logger.debug("Failed to create CIContext for command backdrop", exc_info=True)
            self._ci_context = None
        return self._ci_context

    def capture_blurred_image(self, *, window_number: int, capture_rect, blur_radius_points: float):
        overscan_points = _command_backdrop_capture_overscan_points()
        content_width_points = max(capture_rect.size.width - 2 * overscan_points, 1.0)
        content_height_points = max(capture_rect.size.height - 2 * overscan_points, 1.0)
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            context = self._context()
            if context is not None and hasattr(context, "createCGImage_fromRect_"):
                shell_config = _command_optical_shell_config(
                    content_width_points,
                    content_height_points,
                )
                if shell_config is not None:
                    extent = NSMakeRect(0.0, 0.0, capture_rect.size.width, capture_rect.size.height)
                    output = _debug_shell_grid_ci_image(extent, shell_config)
                    if output is not None:
                        output = _apply_optical_shell_warp_ci_image(output, extent, shell_config)
                        try:
                            image = context.createCGImage_fromRect_(output, extent)
                        except Exception:
                            logger.debug("Failed to seed debug shell grid image in Quartz backdrop renderer", exc_info=True)
                            image = None
                        if image is not None:
                            return image
        from spoke.backdrop_stream import _quartz_timer

        try:
            from Quartz import (
                CGWindowListCreateImage,
                kCGWindowListOptionOnScreenBelowWindow,
            )
        except Exception:
            return None

        _quartz_timer.begin("total")
        _quartz_timer.begin("capture")
        rect = (
            (capture_rect.origin.x, capture_rect.origin.y),
            (capture_rect.size.width, capture_rect.size.height),
        )
        try:
            image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenBelowWindow,
                window_number,
                0,
            )
        except Exception:
            logger.debug("Backdrop snapshot capture failed", exc_info=True)
            return None
        _quartz_timer.end("capture")
        if image is None:
            return image

        try:
            from Quartz import CIImage
        except Exception:
            return image

        try:
            context = self._context()
            if context is None:
                return image
            _quartz_timer.begin("ci_convert")
            ci_image = CIImage.imageWithCGImage_(image)
            extent = ci_image.extent() if hasattr(ci_image, "extent") else None
            _quartz_timer.end("ci_convert")
            if extent is None:
                return image
            output = ci_image

            # Apply optical shell warp to the real backdrop capture.
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                shell_config = _command_optical_shell_config(
                    content_width_points,
                    content_height_points,
                )
                if shell_config is not None:
                    _quartz_timer.begin("warp")
                    warped = _apply_optical_shell_warp_ci_image(output, extent, shell_config)
                    if warped is not None:
                        output = warped
                        if hasattr(output, "imageByCroppingToRect_"):
                            output = output.imageByCroppingToRect_(extent)
                    _quartz_timer.end("warp")


                    # Faint boundary outline over the warped backdrop.
                    if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                        outline = _boundary_outline_ci_image(extent, shell_config)
                        if outline is not None:
                            try:
                                from Quartz import CIFilter
                                comp = CIFilter.filterWithName_("CISourceOverCompositing")
                                if comp is not None:
                                    comp.setDefaults()
                                    comp.setValue_forKey_(outline, "inputImage")
                                    comp.setValue_forKey_(output, "inputBackgroundImage")
                                    composited = comp.valueForKey_("outputImage")
                                    if composited is not None:
                                        output = composited
                                        if hasattr(output, "imageByCroppingToRect_"):
                                            output = output.imageByCroppingToRect_(extent)
                            except Exception:
                                pass

            if not hasattr(context, "createCGImage_fromRect_"):
                return image
            _quartz_timer.begin("render")
            result = context.createCGImage_fromRect_(output, extent)
            _quartz_timer.end("render")
            _quartz_timer.end("total")
            _quartz_timer.frame_done()
            return result or image
        except Exception:
            logger.debug("Backdrop warp pass failed; using raw snapshot", exc_info=True)
            return image


def _dismiss_animation_state(elapsed_s: float) -> tuple[str, float, float, bool]:
    elapsed = max(elapsed_s, 0.0)
    if elapsed < _DISMISS_GROW_S:
        eased = _ease_in(elapsed / _DISMISS_GROW_S)
        scale = _lerp(1.0, _DISMISS_GROW_SCALE, eased)
        return "grow", scale, 1.0, False

    shrink_elapsed = elapsed - _DISMISS_GROW_S
    eased = _ease_in(shrink_elapsed / _DISMISS_SHRINK_S)
    scale = _lerp(_DISMISS_GROW_SCALE, _DISMISS_END_SCALE, eased)
    alpha = 1.0 - eased
    done = elapsed >= (_DISMISS_DURATION_S - 1e-9)
    return "shrink", scale, alpha, done


class CommandOverlay(NSObject):
    """Overlay for displaying streamed command responses."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(CommandOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._wrapper_view: NSView | None = None
        self._text_view: NSTextView | None = None
        self._scroll_view: NSScrollView | None = None
        self._content_view: NSView | None = None
        self._visible = False
        self._streaming = False
        self._response_text = ""
        self._utterance_text = ""
        self._utterance_label = None  # pinned header for user text

        # Fade state
        self._fade_timer: NSTimer | None = None
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 0
        self._cancel_timer_anim: NSTimer | None = None
        self._cancel_elapsed = 0.0
        self._cancel_phase = ""

        # Pulse state
        self._pulse_timer: NSTimer | None = None
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0
        self._tool_mode = False
        # Linger timer
        self._linger_timer: NSTimer | None = None

        # TTS amplitude state — drives window opacity during speech playback
        self._tts_amplitude = 0.0
        self._tts_active = False
        self._tts_blend = 0.0  # 0.0 = pure pulse, 1.0 = pure TTS

        # Thinking timer state
        self._thinking_timer: NSTimer | None = None
        self._thinking_seconds = 0.0

        # Cancel spring state — 0.0 = idle, 1.0 = fully wound
        self._cancel_spring = 0.0
        self._cancel_spring_target = 0.0  # 1.0 while winding, 0.0 while unwinding
        self._cancel_spring_fired = False  # True once threshold crossed
        self._on_cancel_spring_threshold: Callable[[], None] | None = None
        self._thinking_label = None  # NSTextField for the counter
        self._thinking_glow_layer = None  # CALayer for the glow behind the number
        self._thinking_inverted = False  # False = glowing number, True = cutout
        self._narrator_label = None  # NSTextField for narrator summary

        # Adaptive compositing defaults dark until we sample the screen.
        self._brightness = 0.0
        self._brightness_target = 0.0
        self._backdrop_base_blur_radius_points = _COMMAND_BACKDROP_BLUR_RADIUS
        self._backdrop_blur_radius_points = _COMMAND_BACKDROP_BLUR_RADIUS
        self._backdrop_base_mask_width_multiplier = _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER
        self._backdrop_mask_width_multiplier = _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER
        self._backdrop_blur_drive = 0.0
        self._backdrop_renderer = make_backdrop_renderer(
            self._screen,
            lambda: _QuartzBackdropRenderer(),
        )
        self._backdrop_layer = None
        self._backdrop_capture_overscan_points = _command_backdrop_capture_overscan_points()
        self._backdrop_capture_rect = None
        self._backdrop_capture_pixel_size = None
        self._backdrop_timer: NSTimer | None = None
        self._fullscreen_compositor = None

        return self

    def setup(self) -> None:
        """Create the command overlay window."""
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width

        f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        y = _OVERLAY_BOTTOM_MARGIN - f
        win_w = _OVERLAY_WIDTH + 2 * f
        win_h = _OVERLAY_HEIGHT + 2 * f
        frame = NSMakeRect(x, y, win_w, win_h)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(_COMMAND_OVERLAY_WINDOW_LEVEL)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        wrapper_frame = NSMakeRect(0, 0, win_w, win_h)
        wrapper = NSView.alloc().initWithFrame_(wrapper_frame)
        wrapper.setWantsLayer_(True)
        wrapper.layer().setAnchorPoint_((0.5, 0.5))
        wrapper.layer().setPosition_((win_w / 2, win_h / 2))
        self._wrapper_view = wrapper

        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setMasksToBounds_(False)
        content.layer().setBackgroundColor_(None)

        # Distance-field ridge + fill — same system as the preview overlay
        from .overlay import (
            _GLOW_COLOR as _OVERLAY_GLOW_COLOR,
            _overlay_layer_colors,
        )
        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        _, middle_rgb, _ = _overlay_layer_colors(_OVERLAY_GLOW_COLOR)

        self._ridge_scale = self._screen.backingScaleFactor() if hasattr(self._screen, 'backingScaleFactor') else 2.0

        backdrop_layer_cls = self._choose_backdrop_layer_class()
        self._backdrop_layer = backdrop_layer_cls.alloc().init()
        self._backdrop_layer_is_sample_buffer_display = (
            backdrop_layer_cls is not CALayer and hasattr(self._backdrop_layer, "enqueueSampleBuffer_")
        )
        # Configure CAMetalLayer if we got one
        self._backdrop_is_metal_layer = hasattr(self._backdrop_layer, "setDevice_")
        if self._backdrop_is_metal_layer:
            try:
                from spoke.metal_warp import get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    self._backdrop_layer.setDevice_(pipeline.device)
                    self._backdrop_layer.setPixelFormat_(80)  # BGRA8Unorm
                    self._backdrop_layer.setFramebufferOnly_(False)
                    self._backdrop_layer.setOpaque_(False)
                    scale = self._ridge_scale if hasattr(self, "_ridge_scale") else 2.0
                    self._backdrop_layer.setContentsScale_(scale)
            except Exception:
                logger.debug("Failed to configure CAMetalLayer", exc_info=True)
        if hasattr(self._backdrop_layer, "setContentsGravity_"):
            self._backdrop_layer.setContentsGravity_("resize")
        elif hasattr(self._backdrop_layer, "setVideoGravity_"):
            self._backdrop_layer.setVideoGravity_("resize")
        self._backdrop_layer.setOpacity_(1.0)

        # Fill layer — colored SDF image with baked alpha, same as preview overlay
        self._fill_layer = CALayer.alloc().init()
        self._fill_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._fill_layer.setContentsGravity_("resize")

        # Boost layer — white, sits below the fill.  When punch-through
        # is active on dark backgrounds, this layer brightens the warped
        # content visible through the text-shaped holes in the fill mask.
        # Uses the inverse of the punch-through mask (opaque where text is).
        self._boost_layer = CALayer.alloc().init()
        self._boost_layer.setFrame_(((0, 0), (win_w, win_h)))
        from Quartz import CGColorCreateSRGB
        self._boost_layer.setBackgroundColor_(CGColorCreateSRGB(1.0, 1.0, 1.0, 1.0))
        self._boost_layer.setOpacity_(0.0)
        self._boost_layer.setHidden_(True)

        self._apply_ridge_masks(w, h)
        wrapper.layer().insertSublayer_below_(self._backdrop_layer, self._fill_layer)
        wrapper.layer().insertSublayer_below_(self._boost_layer, self._fill_layer)
        wrapper.layer().insertSublayer_below_(self._fill_layer, content.layer())
        self._install_backdrop_frame_callback()
        self._install_backdrop_sample_buffer_callback()
        # Pass CAMetalLayer to SCK renderer and start display-link render loop
        self._metal_display_link_renderer = None
        if getattr(self, "_backdrop_is_metal_layer", False):
            renderer = getattr(self, "_backdrop_renderer", None)
            if renderer is not None and hasattr(renderer, "set_metal_backdrop_layer"):
                renderer.set_metal_backdrop_layer(self._backdrop_layer)
            try:
                from spoke.metal_warp import MetalDisplayLinkRenderer, get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    dl_renderer = MetalDisplayLinkRenderer(pipeline, self._backdrop_layer)
                    if dl_renderer.start():
                        self._metal_display_link_renderer = dl_renderer
                        if renderer is not None and hasattr(renderer, "set_display_link_renderer"):
                            renderer.set_display_link_renderer(dl_renderer)
                        logger.info("Command overlay: display-link Metal renderer active")
                    else:
                        logger.info("Command overlay: display-link start failed, falling back to CIWarpKernel")
            except Exception:
                logger.debug("Command overlay: MetalDisplayLinkRenderer unavailable", exc_info=True)

        # Cancel spring tint layer — sits above fill, masked to the same SDF shape
        self._spring_tint_layer = CALayer.alloc().init()
        self._spring_tint_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._spring_tint_layer.setOpacity_(0.0)
        wrapper.layer().insertSublayer_above_(self._spring_tint_layer, self._fill_layer)

        wrapper.addSubview_(content)
        self._content_view = content

        # Scroll view with text view for response text
        scroll_frame = NSMakeRect(24, 16, _OVERLAY_WIDTH - 48, _OVERLAY_HEIGHT - 32)
        self._scroll_view = NSScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setBorderType_(0)
        self._scroll_view.setAutoresizingMask_(18)
        # Kill clip view background — same fix as preview overlay
        clip_view = self._scroll_view.contentView()
        if clip_view and hasattr(clip_view, 'setDrawsBackground_'):
            clip_view.setDrawsBackground_(False)
        if clip_view and hasattr(clip_view, 'setWantsLayer_'):
            clip_view.setWantsLayer_(True)
            clip_view.layer().setBackgroundColor_(None)

        text_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 48, _OVERLAY_HEIGHT - 32)
        self._text_view = NSTextView.alloc().initWithFrame_(text_frame)
        self._text_view.setEditable_(False)
        self._text_view.setSelectable_(False)
        self._text_view.setDrawsBackground_(False)
        self._text_view.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, _TEXT_ALPHA_MIN)
        )
        self._text_view.setFont_(NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0))
        self._text_view.setString_("")
        self._text_view.textContainer().setWidthTracksTextView_(True)
        self._text_view.setHorizontallyResizable_(False)
        self._text_view.setVerticallyResizable_(True)

        self._scroll_view.setDocumentView_(self._text_view)
        content.addSubview_(self._scroll_view)

        # Thinking timer label — top-right corner of the content area
        from AppKit import NSTextField, NSTextAlignmentRight
        timer_w, timer_h = 60.0, 24.0
        timer_x = _OVERLAY_WIDTH - timer_w - 12
        timer_y = _OVERLAY_HEIGHT - timer_h - 6
        self._thinking_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(timer_x, timer_y, timer_w, timer_h)
        )
        self._thinking_label.setEditable_(False)
        self._thinking_label.setSelectable_(False)
        self._thinking_label.setBezeled_(False)
        self._thinking_label.setDrawsBackground_(False)
        self._thinking_label.setAlignment_(NSTextAlignmentRight)
        self._thinking_label.setFont_(
            NSFont.monospacedDigitSystemFontOfSize_weight_(13.0, 0.2)
        )
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.0
            )
        )
        self._thinking_label.setStringValue_("")
        self._thinking_label.setHidden_(True)
        content.addSubview_(self._thinking_label)

        # Narrator summary label — below the thinking timer, left-aligned
        # Matches the user utterance text style (same font, adaptive color)
        from AppKit import NSTextAlignmentLeft, NSLineBreakByTruncatingTail
        narrator_h = 22.0
        narrator_x = 14.0
        narrator_y = timer_y - narrator_h - 2
        narrator_w = _OVERLAY_WIDTH - 28
        self._narrator_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(narrator_x, narrator_y, narrator_w, narrator_h)
        )
        self._narrator_label.setEditable_(False)
        self._narrator_label.setSelectable_(False)
        self._narrator_label.setBezeled_(False)
        self._narrator_label.setDrawsBackground_(False)
        self._narrator_label.setAlignment_(NSTextAlignmentLeft)
        self._narrator_label.setLineBreakMode_(NSLineBreakByTruncatingTail)
        self._narrator_label.setFont_(
            NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0)
        )
        # Initial color set; will be updated by _apply_narrator_theme()
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        self._narrator_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4)
        )
        self._narrator_label.setStringValue_("")
        self._narrator_label.setHidden_(True)
        content.addSubview_(self._narrator_label)

        self._window.setContentView_(wrapper)
        self._window.setAlphaValue_(0.0)
        self._apply_surface_theme()
        self._set_overlay_scale(1.0)
        self._update_backdrop_capture_geometry()

    def _choose_backdrop_layer_class(self):
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            try:
                from spoke.metal_warp import get_metal_warp_pipeline
                pipeline = get_metal_warp_pipeline()
                if pipeline is not None:
                    import objc
                    CAMetalLayer = objc.lookUpClass("CAMetalLayer")
                    logger.info("Command overlay: using CAMetalLayer for optical shell")
                    return CAMetalLayer
            except Exception:
                logger.debug("CAMetalLayer unavailable", exc_info=True)
            return CALayer
        renderer = getattr(self, "_backdrop_renderer", None)
        blur_radius_points = getattr(self, "_backdrop_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS)
        if renderer is not None and hasattr(renderer, "supports_sample_buffer_presentation"):
            try:
                if renderer.supports_sample_buffer_presentation(blur_radius_points):
                    display_layer_class = _backdrop_display_layer_class()
                    if display_layer_class is not None:
                        return display_layer_class
            except Exception:
                logger.debug("Command backdrop renderer sample-buffer capability check failed", exc_info=True)
        return CALayer

        logger.info("Command overlay created")

    def _backdrop_layer_uses_sample_buffers(self) -> bool:
        return bool(getattr(self, "_backdrop_layer_is_sample_buffer_display", False))

    def _reset_backdrop_layer(self) -> None:
        layer = getattr(self, "_backdrop_layer", None)
        if layer is None:
            return
        if self._backdrop_layer_uses_sample_buffers() and hasattr(layer, "flushAndRemoveImage"):
            try:
                layer.flushAndRemoveImage()
            except Exception:
                logger.debug("Failed to flush command backdrop display layer", exc_info=True)
        if hasattr(layer, "setContents_"):
            layer.setContents_(None)
        if hasattr(layer, "setMask_"):
            layer.setMask_(None)

    def _current_optical_shell_config(self) -> dict[str, float | bool] | None:
        content = getattr(self, "_content_view", None)
        if content is None or not hasattr(content, "frame"):
            return _command_optical_shell_config()
        try:
            frame = content.frame()
            width = frame.size.width
            height = frame.size.height
        except Exception:
            return _command_optical_shell_config()
        if not isinstance(width, numbers.Real) or not isinstance(height, numbers.Real):
            return _command_optical_shell_config()
        return _command_optical_shell_config(width, height)

    def _apply_backdrop_pulse_style(self, breath: float) -> None:
        layer = getattr(self, "_backdrop_layer", None)
        if layer is None:
            return
        blur_target = _command_backdrop_blur_target_for_presence(breath)
        self._backdrop_blur_drive = _advance_attack_release(
            getattr(self, "_backdrop_blur_drive", blur_target),
            blur_target,
            attack=_COMMAND_BACKDROP_PULSE_ATTACK,
            release=_COMMAND_BACKDROP_PULSE_RELEASE,
        )
        base_blur_radius_points = getattr(
            self, "_backdrop_base_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS
        )
        base_mask_width_multiplier = getattr(
            self,
            "_backdrop_base_mask_width_multiplier",
            _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER,
        )
        (
            blur_radius_points,
            mask_width_multiplier,
            backdrop_opacity,
        ) = _command_backdrop_pulse_style(
            base_blur_radius_points,
            base_mask_width_multiplier,
            self._backdrop_blur_drive,
        )
        if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            mask_width_multiplier = min(
                mask_width_multiplier,
                _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_MASK_WIDTH_MULTIPLIER,
            )
        renderer = getattr(self, "_backdrop_renderer", None)
        shell_config = self._current_optical_shell_config()
        effective_blur_radius_points = blur_radius_points
        if shell_config is not None:
            effective_blur_radius_points = float(
                shell_config["cleanup_blur_radius_points"]
            )
        self._backdrop_blur_radius_points = effective_blur_radius_points
        if renderer is not None and hasattr(renderer, "set_live_blur_radius_points"):
            try:
                renderer.set_live_blur_radius_points(effective_blur_radius_points)
            except Exception:
                logger.debug("Failed to push live command backdrop blur radius", exc_info=True)
        if renderer is not None and hasattr(renderer, "set_live_optical_shell_config"):
            try:
                renderer.set_live_optical_shell_config(shell_config)
            except Exception:
                logger.debug("Failed to push command optical-shell config", exc_info=True)
        last_mask_width_multiplier = getattr(
            self,
            "_backdrop_mask_width_multiplier",
            base_mask_width_multiplier,
        )
        self._backdrop_mask_width_multiplier = mask_width_multiplier
        if abs(mask_width_multiplier - last_mask_width_multiplier) > 1e-6:
            capture_rect = getattr(self, "_backdrop_capture_rect", None)
            if capture_rect is not None:
                try:
                    mask_width = float(capture_rect.size.width)
                    mask_height = float(capture_rect.size.height)
                except (TypeError, ValueError):
                    mask_width = None
                    mask_height = None
                if mask_width is not None and mask_height is not None:
                    self._update_backdrop_mask(mask_width, mask_height)
        layer.setOpacity_(backdrop_opacity)

    # ── public interface ────────────────────────────────────

    def show(self, *, preserve_thinking_timer: bool = False) -> None:
        """Fade the overlay in and start or resume the thinking timer."""
        if self._window is None:
            return
        self._cancel_all_timers()
        self._visible = True
        self._streaming = True
        self._response_text = ""
        self._utterance_text = ""
        # Reset TTS state so stale blend doesn't affect new responses
        self._tts_active = False
        self._tts_blend = 0.0
        self._tts_amplitude = 0.0
        self._text_view.setString_("")
        if hasattr(self._text_view, "textStorage"):
            try:
                from AppKit import NSMutableAttributedString

                self._text_view.textStorage().setAttributedString_(
                    NSMutableAttributedString.alloc().initWithString_("")
                )
            except Exception:
                pass
        self._window.setAlphaValue_(0.0)
        self._set_overlay_scale(1.0)

        # Reset geometry
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN - f,
                       _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
            True, False
        )
        self._content_view.setFrame_(
            NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        )
        self._scroll_view.setFrame_(
            NSMakeRect(48, 16, _OVERLAY_WIDTH - 96, _OVERLAY_HEIGHT - 32)
        )
        self._apply_ridge_masks(_OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        self._fill_image_brightness = self._brightness
        self._apply_surface_theme()
        self._update_backdrop_capture_geometry()
        self._apply_backdrop_pulse_style(1.0)
        self._reset_backdrop_layer()

        self._window.orderFrontRegardless()
        self._refresh_backdrop_snapshot()

        # Entrance pop — start slightly oversized, ease back to 1.0.
        # Runs concurrently with the fade-in for a subtle "I just arrived" feel.
        self._set_overlay_scale(_ENTRANCE_POP_SCALE)
        self._pop_step = 0
        self._pop_steps = max(1, int(_ENTRANCE_POP_S * _DISMISS_ANIM_FPS))
        self._pop_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_entrancePopStep:", None, True
        )

        # Fade in
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 1
        interval = _FADE_IN_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

        # Start pulse animation
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0
        self._tool_mode = False
        self._pulse_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _PULSE_HZ, self, "pulseStep:", None, True
        )

        # Start or resume the thinking timer.
        self._start_thinking_timer(reset=not preserve_thinking_timer)

        # Full-screen compositor: captures entire display, warps capsule region,
        # presents full frame via Metal.  No seam between warped and unwarped.
        # Must start BEFORE the backdrop refresh timer — if compositor succeeds,
        # the old backdrop path is disabled entirely.
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            self._start_fullscreen_compositor()
        # Only start old backdrop path if compositor isn't running
        if getattr(self, "_fullscreen_compositor", None) is None:
            self._start_backdrop_refresh_timer()

    def set_brightness(self, brightness: float, immediate: bool = False) -> None:
        """Set screen brightness (0.0 dark – 1.0 bright) for adaptive compositing."""
        self._brightness_target = _clamp01(brightness)
        if immediate:
            self._brightness = self._brightness_target
            self._apply_surface_theme()

    def _entrancePopStep_(self, timer) -> None:
        """Ease the entrance pop back to scale 1.0."""
        self._pop_step += 1
        t = self._pop_step / self._pop_steps
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        scale = _ENTRANCE_POP_SCALE * (1.0 - eased) + 1.0 * eased
        self._set_overlay_scale(scale)
        if self._pop_step >= self._pop_steps:
            timer.invalidate()
            self._pop_timer = None
            self._set_overlay_scale(1.0)

    def cancel_dismiss(self) -> None:
        """Dismiss with a fast pop-then-shrink animation."""
        if self._window is None:
            return
        self._cancel_all_timers()
        self._streaming = False
        self._visible = True  # keep visible for the animation

        self._cancel_elapsed = 0.0
        self._cancel_phase = "grow"
        self._window.setAlphaValue_(1.0)
        self._set_overlay_scale(1.0)
        self._cancel_timer_anim = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_cancelAnimStep:", None, True
        )

    def _cancelAnimStep_(self, timer) -> None:
        """Animate the dismiss sequence: grow for 60ms, then shrink and fade."""
        self._cancel_elapsed += 1.0 / _DISMISS_ANIM_FPS
        phase, scale, alpha, done = _dismiss_animation_state(self._cancel_elapsed)
        self._cancel_phase = phase
        self._set_overlay_scale(scale)
        self._window.setAlphaValue_(alpha)

        if done:
            self._cancel_dismiss_animation()
            self._window.setAlphaValue_(0.0)
            self._set_overlay_scale(1.0)
            self._reset_backdrop_layer()
            if self._backdrop_renderer is not None and hasattr(self._backdrop_renderer, "stop_live_stream"):
                self._backdrop_renderer.stop_live_stream()
            self._stop_fullscreen_compositor()
            self._window.orderOut_(None)
            self._visible = False
            self._cancel_pulse()

    def hide(self) -> None:
        """Fade out — pulse continues during fade for visual continuity."""
        if self._window is None:
            return
        self._visible = False
        self._streaming = False
        self._cancel_linger()
        # Don't cancel pulse here — let it continue during fade-out.
        # It will be cancelled when the fade completes (window ordered out).
        self._start_fade_out()

    def set_cancel_spring(self, target: float) -> None:
        """Set the cancel spring target (0.0 = idle, 1.0 = winding up).

        The spring animates toward the target in the pulse tick, so
        calling this once starts the wind-up or snap-back.
        """
        self._cancel_spring_target = max(0.0, min(1.0, target))
        if target > 0.0:
            self._cancel_spring_fired = False  # reset for new wind-up

    def set_utterance(self, text: str) -> None:
        """Show the user's utterance in the text view at reduced opacity."""
        self._utterance_text = text
        if self._text_view is None or not self._visible:
            return
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        attr_str = NSMutableAttributedString.alloc().initWithString_(text)
        attr_str.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4),
            (0, len(text)),
        )
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.15)
        )
        glow.setShadowOffset_((0, 0))
        glow.setShadowBlurRadius_(3.0)
        attr_str.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(text))
        )
        self._text_view.textStorage().setAttributedString_(attr_str)
        self._update_layout()

    def append_token(self, token: str) -> None:
        """Append a streamed response token."""
        if self._text_view is None or not self._visible:
            return
        first_token = len(self._response_text) == 0
        self._response_text += token

        if first_token and self._utterance_text:
            # Add separator between utterance and response
            from AppKit import (
                NSMutableAttributedString,
                NSForegroundColorAttributeName,
            )
            sep = NSMutableAttributedString.alloc().initWithString_("\n\n")
            sep.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0),
                (0, 2),
            )
            self._text_view.textStorage().appendAttributedString_(sep)

        frag = self._make_response_fragment(token)
        self._text_view.textStorage().appendAttributedString_(frag)

        self._update_layout()

    def set_response_text(self, text: str) -> None:
        """Replace the visible assistant response with final canonical text.

        Rebuilds the full attributed string (utterance + response) in one shot
        so that _update_layout is called exactly once.  Calling set_utterance
        then append_token would call _update_layout twice: first with only the
        utterance (shrinking the window back to minimum height), then again with
        the full response — producing a visible size flicker.
        """
        self._response_text = ""
        if self._text_view is None or not self._visible:
            self._response_text = text
            return

        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )

        combined = NSMutableAttributedString.alloc().initWithString_("")

        if self._utterance_text:
            user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
            utt = NSMutableAttributedString.alloc().initWithString_(self._utterance_text)
            utt.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4),
                (0, len(self._utterance_text)),
            )
            glow = NSShadow.alloc().init()
            glow.setShadowColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.15)
            )
            glow.setShadowOffset_((0, 0))
            glow.setShadowBlurRadius_(3.0)
            utt.addAttribute_value_range_(
                NSShadowAttributeName, glow, (0, len(self._utterance_text))
            )
            combined.appendAttributedString_(utt)

            if text:
                sep = NSMutableAttributedString.alloc().initWithString_("\n\n")
                sep.addAttribute_value_range_(
                    NSForegroundColorAttributeName,
                    NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0),
                    (0, 2),
                )
                combined.appendAttributedString_(sep)

        self._text_view.textStorage().setAttributedString_(combined)

        if text:
            self._response_text = ""
            self.append_token(text)
        else:
            self._update_layout()

    def _current_hue_rgb(self):
        """Get the current hue rotation color as (r, g, b)."""
        hue = getattr(self, '_color_phase', 0.0)
        s, v = 0.75, 0.95
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            return (c + m, x + m, m)
        elif h6 < 2:
            return (x + m, c + m, m)
        elif h6 < 3:
            return (m, c + m, x + m)
        elif h6 < 4:
            return (m, x + m, c + m)
        elif h6 < 5:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)

    def _make_response_fragment(self, token: str):
        """Create an attributed string fragment for a response token.

        Text color matches the current hue rotation. Glow matches too.
        """
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        r, g, b = self._current_hue_rgb()
        r, g, b = _response_color_for_brightness((r, g, b), self._brightness)
        frag = NSMutableAttributedString.alloc().initWithString_(token)
        response_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            r, g, b, _TEXT_ALPHA_MAX
        )
        frag.addAttribute_value_range_(
            NSForegroundColorAttributeName, response_color, (0, len(token))
        )
        frag.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0),
            (0, len(token)),
        )
        # Text glow in the current hue color
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 0.6)
        )
        glow.setShadowOffset_((0, 0))
        glow.setShadowBlurRadius_(3.0)
        frag.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(token))
        )
        return frag

    def set_tool_active(self, active: bool) -> None:
        """Show or hide the tool execution indicator."""
        self._tool_mode = active
        if active and self._visible:
            if self._thinking_label is not None:
                self._thinking_label.setHidden_(False)
                self._thinking_label.setStringValue_("tool…")
            if self._thinking_timer is None:
                self._start_thinking_timer()

    def finish(self) -> None:
        """Called when the response stream is complete.

        The overlay stays visible indefinitely — the user dismisses it
        by holding spacebar (which triggers cancel_dismiss via __main__).
        No linger timer. The user is in control.
        """
        self._streaming = False
        self._stop_thinking_timer()
        # Pulse keeps running — the overlay is alive until dismissed

    # ── TTS amplitude ─────────────────────────────────────────

    _TTS_RISE = 0.70      # fast but not instant — avoids harsh pop-in
    _TTS_DECAY = 0.82     # gentle falloff — no clipping between words
    _TTS_MULTIPLIER = 25.0
    _TTS_ALPHA_MIN = _ASSISTANT_TEXT_ALPHA_MIN  # keep assistant text firmly readable during playback
    _TTS_ALPHA_MAX = 1.0   # full brightness on voice peaks

    def update_tts_amplitude(self, rms: float) -> None:
        """Update smoothed TTS amplitude from audio RMS.

        The pulse step reads _tts_amplitude when _tts_active is True to
        drive the response text alpha — no window alpha manipulation here,
        so there's no fighting between pulse and TTS updates.
        Must be called on the main thread.
        """
        if rms > self._tts_amplitude:
            self._tts_amplitude += (rms - self._tts_amplitude) * self._TTS_RISE
        else:
            self._tts_amplitude *= self._TTS_DECAY

    def tts_start(self) -> None:
        """Prepare overlay for TTS-driven amplitude."""
        self._tts_active = True
        self._tts_amplitude = 0.0

    def tts_stop(self) -> None:
        """Return to pulse-driven alpha after TTS ends."""
        self._tts_active = False
        self._tts_amplitude = 0.0

    # ── animation ───────────────────────────────────────────

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation.

        During fade-out, the utterance text fades at double rate so it
        disappears before the assistant response.
        """
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            eased = progress * progress
            alpha = eased
        else:
            eased = progress * progress
            alpha = self._fade_from * (1.0 - eased)

            # Fade utterance at double rate
            if self._text_view is not None and self._utterance_text:
                from AppKit import NSForegroundColorAttributeName
                utt_progress = min(progress * 2.0, 1.0)
                utt_alpha = 0.35 * (1.0 - utt_progress * utt_progress)
                utt_len = min(len(self._utterance_text), self._text_view.textStorage().length() if hasattr(self._text_view.textStorage(), 'length') else 0)
                if utt_len > 0:
                    try:
                        self._text_view.textStorage().addAttribute_value_range_(
                            NSForegroundColorAttributeName,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, utt_alpha),
                            (0, utt_len),
                        )
                    except Exception:
                        pass

        self._window.setAlphaValue_(alpha)

        if self._fade_step >= _FADE_STEPS:
            self._cancel_fade()
            if self._fade_direction == -1:
                self._window.setAlphaValue_(0.0)
                self._reset_backdrop_layer()
                if self._backdrop_renderer is not None and hasattr(self._backdrop_renderer, "stop_live_stream"):
                    self._backdrop_renderer.stop_live_stream()
                self._stop_fullscreen_compositor()
                self._window.orderOut_(None)
                self._cancel_pulse()  # now kill the pulse
            else:
                self._window.setAlphaValue_(1.0)

    def pulseStep_(self, timer) -> None:
        """Dual-phase pulse: user and assistant text breathe independently.

        Assistant text: faster period (0.8x base), double-smoothstep for
        extra-aggressive easing, violet-amber color oscillation.
        User text: slower period (1.5x base), single smoothstep, blue
        color shift, phase-offset by 0.3 and diverging naturally.
        """
        try:
            self._pulseStepInner()
        except Exception:
            logger.exception("pulseStep_ crashed")

    def _pulseStepInner(self):
        if self._text_view is None:
            return
        dt = 1.0 / _PULSE_HZ

        # When compositor is active, sample capsule-region brightness
        # directly — much more accurate than the glow's 4-patch screen
        # average.  Refresh every ~500ms (15 pulse ticks).
        compositor = getattr(self, "_fullscreen_compositor", None)
        if compositor is not None:
            _b_tick = getattr(self, '_brightness_sample_tick', 0)
            if _b_tick % 15 == 0:
                compositor.refresh_brightness()
            self._brightness_sample_tick = _b_tick + 1
            self._brightness_target = compositor.sampled_brightness
        target = getattr(self, "_brightness_target", 0.0)
        current = getattr(self, "_brightness", 0.0)
        if abs(target - current) > 0.001:
            self._brightness = current + (target - current) * _BRIGHTNESS_CHASE
        t = getattr(self, "_brightness", 0.0)
        self._apply_surface_theme()

        # Advance phases
        self._pulse_phase_asst += dt / _PULSE_PERIOD_ASST
        if self._pulse_phase_asst > 1.0:
            self._pulse_phase_asst -= 1.0
        self._pulse_phase_user += dt / _PULSE_PERIOD_USER
        if self._pulse_phase_user > 1.0:
            self._pulse_phase_user -= 1.0

        # Assistant: one slow breath, 5 seconds. Asymmetric:
        # ease-out toward bright (lingers near peak), ease-in back to dim
        # (dips quickly). Raw sine → squared so it spends more time high.
        raw_breath = 0.5 * (1.0 + math.cos(2.0 * math.pi * self._pulse_phase_asst))
        breath = raw_breath * raw_breath  # squared: lingers near 1.0, dips briefly
        self._apply_backdrop_pulse_style(breath)
        pulse_alpha_a = _assistant_text_alpha_for_breath(breath)

        # Smooth cross-fade between pulse and TTS-driven alpha over ~500ms.
        # _tts_blend ramps toward 1.0 when TTS is active, toward 0.0 when not.
        # At 30Hz pulse rate, 0.06 per tick ≈ 0.55s full ramp.
        _BLEND_RATE = 0.06
        if self._tts_active:
            self._tts_blend = min(self._tts_blend + _BLEND_RATE, 1.0)
        else:
            self._tts_blend = max(self._tts_blend - _BLEND_RATE, 0.0)

        if self._tts_blend > 0.0:
            tts_scaled = min(self._tts_amplitude * self._TTS_MULTIPLIER, 1.0)
            tts_alpha = self._TTS_ALPHA_MIN + tts_scaled * (self._TTS_ALPHA_MAX - self._TTS_ALPHA_MIN)
            alpha_a = pulse_alpha_a * (1.0 - self._tts_blend) + tts_alpha * self._tts_blend
        else:
            alpha_a = pulse_alpha_a

        # When text punch-through is active, force full text alpha
        # so the destinationOut compositing cleanly erases the fill.
        if getattr(self, "_text_punchthrough", False):
            alpha_a = 1.0

        # User: raw sine → single smoothstep (same aggressiveness as before)
        raw_u = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._pulse_phase_user))
        pulse_u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
        utt_alpha = 0.73 + 0.24 * pulse_u
        if getattr(self, "_text_punchthrough", False):
            utt_alpha = 1.0

        # Full spectrum hue rotation with velocity undulation
        # The speed varies sinusoidally so it dwells in some colors and
        # glides through others — always moving, never truly paused.
        self._color_velocity_phase += dt / _COLOR_VELOCITY_PERIOD
        if self._color_velocity_phase > 1.0:
            self._color_velocity_phase -= 1.0
        vel_raw = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._color_velocity_phase))
        vel = _COLOR_VELOCITY_MIN + vel_raw * (_COLOR_VELOCITY_MAX - _COLOR_VELOCITY_MIN)
        self._color_phase += (dt / _COLOR_CYCLE_PERIOD) * vel
        if self._color_phase > 1.0:
            self._color_phase -= 1.0
        hue = self._color_phase

        # ── Cancel spring animation ──
        # The spring chases _cancel_spring_target with asymmetric speed:
        # winding up is deliberate (~600ms to full), easing out is smooth
        # (~300ms with deceleration).
        _SPRING_THRESHOLD = 0.83  # ~500ms into the 600ms wind-up
        spring = self._cancel_spring
        target = self._cancel_spring_target
        if spring < target:
            # Wind up — deliberate
            spring = min(target, spring + dt / 0.6)
            # Fire cancel at threshold crossing — don't wait for release
            if spring >= _SPRING_THRESHOLD and not self._cancel_spring_fired:
                self._cancel_spring_fired = True
                self._cancel_spring_target = 0.0  # start ease-out
                target = 0.0
                cb = self._on_cancel_spring_threshold
                if cb is not None:
                    cb()
        if spring > target:
            # Ease out — smooth deceleration (quadratic decay)
            rate = dt / 0.3 * (1.0 + spring)  # faster when higher
            spring = max(target, spring - rate)
        self._cancel_spring = spring

        if spring > 0.001:
            # Blend hue toward warm amber (~0.08, orange-gold).
            # Use shortest arc around the hue wheel.
            amber_hue = 0.08
            if hue > 0.5 + amber_hue:
                target_hue = 1.0 + amber_hue  # wrap forward
            else:
                target_hue = amber_hue
            hue = hue + (target_hue - hue) * spring
            if hue >= 1.0:
                hue -= 1.0

        # Log color phase every ~1s (every 30th tick)
        if not hasattr(self, '_color_log_counter'):
            self._color_log_counter = 0
        self._color_log_counter += 1
        if self._color_log_counter % 30 == 0:
            logger.info("Color phase: %.3f hue, vel_phase=%.3f, spring=%.3f", hue, self._color_velocity_phase, spring)

        # Saturation and value — spring winds up saturation toward vivid red
        base_s, base_v = 0.35, 0.55  # saturated + darker — readable against opaque fill
        s = base_s + (0.75 - base_s) * spring  # 0.35 → 0.75 at full wind
        v = base_v + (0.65 - base_v) * spring  # 0.55 → 0.65 at full wind
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            r, g, b = c + m, x + m, m
        elif h6 < 2:
            r, g, b = x + m, c + m, m
        elif h6 < 3:
            r, g, b = m, c + m, x + m
        elif h6 < 4:
            r, g, b = m, x + m, c + m
        elif h6 < 5:
            r, g, b = x + m, m, c + m
        else:
            r, g, b = c + m, m, x + m
        response_r, response_g, response_b = _response_color_for_brightness((r, g, b), t)

        # Alternate color: offset hue by ~0.4, slightly out of phase so
        # when the main color is bright the alt is dark and vice versa.
        alt_hue = (hue + 0.4) % 1.0
        alt_breath = 1.0 - breath  # out of phase
        alt_v = base_v + (0.65 - base_v) * spring
        alt_v = alt_v * (0.6 + 0.4 * alt_breath)  # modulate darker
        alt_c = alt_v * s
        alt_x = alt_c * (1.0 - abs((alt_hue * 6.0) % 2.0 - 1.0))
        alt_m = alt_v - alt_c
        alt_h6 = alt_hue * 6.0
        if alt_h6 < 1:
            alt_r, alt_g, alt_b = alt_c + alt_m, alt_x + alt_m, alt_m
        elif alt_h6 < 2:
            alt_r, alt_g, alt_b = alt_x + alt_m, alt_c + alt_m, alt_m
        elif alt_h6 < 3:
            alt_r, alt_g, alt_b = alt_m, alt_c + alt_m, alt_x + alt_m
        elif alt_h6 < 4:
            alt_r, alt_g, alt_b = alt_m, alt_x + alt_m, alt_c + alt_m
        elif alt_h6 < 5:
            alt_r, alt_g, alt_b = alt_x + alt_m, alt_m, alt_c + alt_m
        else:
            alt_r, alt_g, alt_b = alt_c + alt_m, alt_m, alt_x + alt_m
        alt_r, alt_g, alt_b = _response_color_for_brightness((alt_r, alt_g, alt_b), t)

        # Update text colors per-range
        if self._text_view is not None:
            from AppKit import NSForegroundColorAttributeName as _FG_pulse
            ts = self._text_view.textStorage()
            total_len = ts.length() if hasattr(ts, 'length') else 0
            if total_len > 0 and self._utterance_text:
                utt_len = min(len(self._utterance_text), total_len)
                # User text: fixed dark gray (or white for punch-through).
                try:
                    if getattr(self, "_text_punchthrough", False):
                        ur, ug, ub = 1.0, 1.0, 1.0
                    else:
                        user_base = _user_text_color_for_brightness(t)
                        ur = user_base[0]
                        ug = user_base[1]
                        ub = user_base[2]
                    ts.addAttribute_value_range_(
                        _FG_pulse,
                        NSColor.colorWithSRGBRed_green_blue_alpha_(ur, ug, ub, utt_alpha),
                        (0, utt_len),
                    )
                except Exception:
                    pass
                # Response text: two visual layers per character.
                #
                # Foreground (anchor): dark version of the dual-hue color
                # rotation — value crushed to ~0.15-0.20, full opacity, no
                # blur, light font weight. Always legible.
                #
                # Shadow (atmosphere): bright dual-hue color, luminance-driven
                # blur. Lighter chars glow soft, darker chars stay crisp.
                # Renders behind the dark anchor, giving depth without
                # sacrificing readability.
                resp_start = utt_len + 2
                resp_len = total_len - resp_start
                _punchthrough = getattr(self, "_text_punchthrough", False)
                if resp_start < total_len and resp_len > 0:
                    from AppKit import (
                        NSShadowAttributeName as _SH_pulse,
                        NSShadow,
                        NSFontAttributeName as _FN_pulse,
                        NSFont,
                    )
                    light_font = NSFont.systemFontOfSize_weight_(
                        _FONT_SIZE, -0.2  # medium-light weight — thicker punch-through
                    )
                    try:
                        if _punchthrough:
                            # Punch-through mode: uniform white at full alpha.
                            # Text glyphs are the mask — alpha 1.0 erases
                            # the fill via CISourceOutCompositing.
                            ts.addAttribute_value_range_(
                                _FG_pulse,
                                NSColor.colorWithSRGBRed_green_blue_alpha_(
                                    1.0, 1.0, 1.0, 1.0
                                ),
                                (resp_start, resp_len),
                            )
                        else:
                            for ci in range(resp_len):
                                # 0.0 at edges, 1.0 at center
                                frac = ci / max(resp_len - 1, 1)
                                center_weight = 1.0 - abs(frac * 2.0 - 1.0)
                                # Bright color (for shadow/glow layer)
                                cr = _lerp(response_r, alt_r, center_weight)
                                cg = _lerp(response_g, alt_g, center_weight)
                                cb = _lerp(response_b, alt_b, center_weight)
                                # Dark anchor: same hue, value crushed
                                dr = cr * 0.22
                                dg = cg * 0.22
                                db = cb * 0.22
                                # Foreground = dark anchor
                                ts.addAttribute_value_range_(
                                    _FG_pulse,
                                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                                        dr, dg, db, 1.0
                                    ),
                                    (resp_start + ci, 1),
                                )
                                # Light font weight for the anchor
                                ts.addAttribute_value_range_(
                                    _FN_pulse, light_font,
                                    (resp_start + ci, 1),
                                )
                                # Shadow = bright glow, blur driven by luminance
                                lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
                                blur_radius = 5.0 + lum * 14.0
                                shadow = NSShadow.alloc().init()
                                shadow.setShadowColor_(
                                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                                        cr, cg, cb, 0.7 + 0.3 * lum
                                    )
                                )
                                shadow.setShadowOffset_((0, 0))
                                shadow.setShadowBlurRadius_(blur_radius)
                                ts.addAttribute_value_range_(
                                    _SH_pulse, shadow, (resp_start + ci, 1),
                                )
                    except Exception:
                        pass
            elif total_len > 0:
                self._text_view.setTextColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        response_r, response_g, response_b, alpha_a
                    )
                )

        # Pulse the glow with assistant phase oscillating color
        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)
        glow_opacity = 0.5 + 0.3 * breath
        # Keep the assistant surface in the same materially present opacity
        # band as the preview overlay on bright backgrounds. The preview gets
        # its presence from a much more assertive fill ramp than the command
        # overlay previously used, so mirror that shape here while preserving
        # the pulse-driven rhythm.
        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            fill_drive = _lerp(breath, breath * breath, t)
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                fill_min = _lerp(
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_DARK,
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MIN_LIGHT,
                    t,
                )
                fill_max = _lerp(
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_DARK,
                    _COMMAND_BACKDROP_OPTICAL_SHELL_FILL_MAX_LIGHT,
                    t,
                )
                if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                    fill_min = 0.0
                    fill_max = 0.0
                elif getattr(self, "_fullscreen_compositor", None) is not None:
                    # Graphic mode: dark-on-dark, light-on-light.
                    # Steep sigmoid so mid-tones commit to one side
                    # rather than lingering in a bland middle.
                    fill_drive = 0.5
                    ct = _clamp01((t - 0.45) * 6.0 + 0.5)
                    ct = ct * ct * (3.0 - 2.0 * ct)
                    fill_min = _lerp(0.72, 0.96, ct)
                    fill_max = _lerp(0.82, 0.99, ct)
            else:
                fill_min = _lerp(0.30, 0.84, t)
                fill_max = _lerp(0.92, 0.99, t)
            new_opacity = min(_lerp(fill_min, fill_max, fill_drive), 0.99)
            if abs(new_opacity - getattr(self, '_last_fill_opacity', -1.0)) > 0.005:
                self._fill_layer.setOpacity_(new_opacity)
                self._last_fill_opacity = new_opacity
        # Boost layer: on light backgrounds (dark fill), brighten the
        # warped content behind text holes so the cut-out text reads
        # light against the dark fill.  White layer masked to glyphs.
        boost_layer = getattr(self, "_boost_layer", None)
        if boost_layer is not None and getattr(self, "_text_punchthrough", False):
            _bt = _clamp01((t - 0.45) * 6.0 + 0.5)
            _bt = _bt * _bt * (3.0 - 2.0 * _bt)
            # Light bg (dark fill) → boost; dark bg (light fill) → no boost
            boost_opacity = _lerp(0.0, 0.5, _bt)
            if boost_opacity > 0.01:
                boost_layer.setHidden_(False)
                boost_layer.setOpacity_(boost_opacity)
            else:
                boost_layer.setHidden_(True)
                boost_layer.setOpacity_(0.0)
        elif boost_layer is not None:
            boost_layer.setHidden_(True)
        # Update text punch-through mask (if active)
        if getattr(self, "_text_punchthrough", False):
            self._update_punchthrough_mask()
        # Cancel spring: warm amber tint over the overlay shape.
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            if spring > 0.01:
                from Quartz import CGColorCreateSRGB
                # Warm golden-amber tint — visible, thermal, not alarming
                cg_color = CGColorCreateSRGB(0.55, 0.38, 0.05, 1.0)
                self._spring_tint_layer.setBackgroundColor_(cg_color)
                spring_scale = (
                    _COMMAND_BACKDROP_OPTICAL_SHELL_SPRING_OPACITY_SCALE
                    if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
                    else 0.5
                )
                if _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_REVEAL:
                    spring_scale = 0.0
                self._spring_tint_layer.setOpacity_(spring_scale * spring)
            else:
                self._spring_tint_layer.setOpacity_(0.0)

    def lingerDone_(self, timer) -> None:
        """Linger period over — fade out."""
        self._linger_timer = None
        if self._visible and not self._streaming:
            self.hide()

    # ── fade helpers ────────────────────────────────────────

    def _start_fade_out(self) -> None:
        self._cancel_fade()
        self._fade_step = 0
        self._fade_from = self._window.alphaValue() if self._window else 1.0
        self._fade_direction = -1
        interval = _FADE_OUT_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            self._fade_timer.invalidate()
            self._fade_timer = None

    def _cancel_pulse(self) -> None:
        if self._pulse_timer is not None:
            self._pulse_timer.invalidate()
            self._pulse_timer = None

    def _cancel_linger(self) -> None:
        if self._linger_timer is not None:
            self._linger_timer.invalidate()
            self._linger_timer = None

    def _cancel_backdrop_refresh(self) -> None:
        if self._backdrop_timer is not None:
            self._backdrop_timer.invalidate()
            self._backdrop_timer = None

    def _cancel_dismiss_animation(self) -> None:
        if self._cancel_timer_anim is not None:
            self._cancel_timer_anim.invalidate()
            self._cancel_timer_anim = None

    def _cancel_entrance_pop(self) -> None:
        if getattr(self, "_pop_timer", None) is not None:
            self._pop_timer.invalidate()
            self._pop_timer = None

    def _cancel_all_timers(self) -> None:
        self._cancel_dismiss_animation()
        self._cancel_entrance_pop()
        self._cancel_fade()
        self._cancel_pulse()
        self._cancel_linger()
        self._cancel_backdrop_refresh()
        self._stop_thinking_timer()

    def _set_overlay_scale(self, scale: float) -> None:
        if self._wrapper_view is None:
            return
        layer = self._wrapper_view.layer() if hasattr(self._wrapper_view, "layer") else None
        if layer is None:
            return
        try:
            # Re-assert center anchor so scale is always symmetric, even
            # after window geometry changes that may shift the layer bounds.
            bounds = layer.bounds()
            layer.setAnchorPoint_((0.5, 0.5))
            layer.setPosition_((
                bounds.origin.x + bounds.size.width / 2,
                bounds.origin.y + bounds.size.height / 2,
            ))
            layer.setValue_forKeyPath_(scale, "transform.scale")
        except Exception:
            logger.exception("Failed to update command overlay scale")

    # ── thinking timer ──────────────────────────────────────

    def _start_thinking_timer(self, *, reset: bool = True) -> None:
        """Start or resume the thinking counter."""
        if reset:
            self._thinking_seconds = 0.0
            self._thinking_inverted = False
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(False)
            if self._tool_mode:
                self._thinking_label.setStringValue_("tool…")
            else:
                self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")
            self._apply_thinking_label_theme()
        logger.info("Thinking timer %s", "started" if reset else "resumed")
        self._thinking_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "thinkingTick:", None, True
        )

    def invert_thinking_timer(self) -> None:
        """Switch from glowing number to negative-space cutout mode.

        Called when the first content token arrives — the visual flip
        signals that the model has transitioned from thinking to speaking.
        """
        self._thinking_inverted = True
        self._apply_thinking_label_theme()

    def set_narrator_summary(self, summary: str) -> None:
        """Update the narrator summary line displayed during thinking."""
        if self._narrator_label is not None:
            self._narrator_label.setStringValue_(summary)
            self._narrator_label.setHidden_(False)
            self._apply_narrator_theme()

    def _apply_narrator_theme(self) -> None:
        """Match narrator label color to user utterance style."""
        if self._narrator_label is None or self._narrator_label.isHidden():
            return
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        self._narrator_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4)
        )

    def _hide_narrator(self) -> None:
        """Hide the narrator summary label."""
        if self._narrator_label is not None:
            self._narrator_label.setHidden_(True)
            self._narrator_label.setStringValue_("")

    def _stop_thinking_timer(self) -> None:
        """Stop and fade the thinking counter."""
        if self._thinking_timer is not None:
            self._thinking_timer.invalidate()
            self._thinking_timer = None
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(True)
        self._hide_narrator()

    def thinkingTick_(self, timer) -> None:
        """Update the thinking counter every 100ms."""
        self._thinking_seconds += 0.1
        if self._thinking_label is not None and not self._thinking_label.isHidden():
            if self._tool_mode: self._thinking_label.setStringValue_("tool…")
            else: self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")

    # ── ridge masks ────────────────────────────────────────

    def _apply_ridge_masks(self, width: float, height: float) -> None:
        """Compute SDF and apply ridge mask + build fill image.

        The SDF and fill-alpha field are cached by geometry (width, height).
        When only brightness changes (same dimensions), we skip the expensive
        numpy SDF computation and just rebuild the colored image.
        """
        from .overlay import (
            _RIDGE_FALLOFF, _RIDGE_POWER, _OVERLAY_CORNER_RADIUS,
            _overlay_rounded_rect_sdf, _ridge_alpha, _interior_fill_alpha,
            _fill_field_to_image, _BG_COLOR_DARK,
        )
        f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
        scale = getattr(self, '_ridge_scale', 2.0)
        total_w = width + 2 * f
        total_h = height + 2 * f

        try:
            from .overlay import _glow_fill_alpha
            import numpy as np

            _has_compositor = getattr(self, "_fullscreen_compositor", None) is not None

            # ── Stage 1: geometry cache (SDF + derived fields) ──
            # Pure function of overlay dimensions — never changes with
            # brightness.  Only rebuilt on resize.
            geom_key = (width, height, scale, _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED)
            if getattr(self, '_sdf_geom_key', None) != geom_key:
                if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                    pw, ph = max(int(total_w), 1), max(int(total_h), 1)
                    xs = np.arange(pw, dtype=np.float32)[None, :] + 0.5 - pw * 0.5
                    ys = np.arange(ph, dtype=np.float32)[:, None] + 0.5 - ph * 0.5
                    half_w = width * 0.5
                    half_h = height * 0.5
                    corner_r = _OVERLAY_HEIGHT / 4.0
                    capsule_radius = max(min(corner_r, half_h), 1.0)
                    spine_half_x = max(half_w - capsule_radius, 0.0)
                    spine_half_y = max(half_h - capsule_radius, 0.0)
                    spine_dist_x = np.maximum(np.abs(xs) - spine_half_x, 0.0)
                    spine_dist_y = np.maximum(np.abs(ys) - spine_half_y, 0.0)
                    sdf = (np.hypot(spine_dist_x, spine_dist_y) - capsule_radius).astype(np.float32)

                    # Pre-compute geometry-derived fields (brightness-independent)
                    inside_d = np.maximum(-sdf, 0.0)
                    edge_ridge = np.exp(-((inside_d / (1.5 * scale)) ** 2.0)).astype(np.float32)
                    # Raw interior curve (no floor) — `_glow_fill_alpha` with
                    # floor=0 gives the pure exp(-sqrt) shape we'll rescale.
                    raw_interior = np.exp(-np.sqrt(
                        np.abs(sdf) / max(2.0 * scale, 1e-6)
                    )).astype(np.float32)
                    inside_mask = (sdf <= 0.0)

                    feather_px = _OPTICAL_SHELL_FEATHER * scale
                    ext_d = np.maximum(sdf, 0.0)
                    sigma = feather_px / 3.0
                    exterior = np.exp(-0.5 * (ext_d / sigma) ** 2.0).astype(np.float32)
                    ext_t = np.clip(ext_d / feather_px, 0.0, 1.0)
                    ext_exterior = (exterior * (0.13 * np.exp(-((ext_t / 0.30) ** 1.5)) + 0.007)).astype(np.float32)

                    self._sdf_raw_interior = raw_interior
                    self._sdf_edge_ridge = edge_ridge
                    self._sdf_inside_mask = inside_mask
                    self._sdf_ext_exterior = ext_exterior
                else:
                    sdf = _overlay_rounded_rect_sdf(
                        total_w, total_h, width, height,
                        _OVERLAY_CORNER_RADIUS, scale,
                    )
                    # Non-compositor path: fill alpha is brightness-independent
                    self._sdf_fallback_alpha = _glow_fill_alpha(sdf, width=2.5 * scale)

                self._sdf_geom_key = geom_key
                # Force appearance rebuild after geometry change
                self._sdf_appearance_b = -1.0

            # ── Stage 2: appearance (brightness-dependent fill alpha) ──
            # Cheap scalar ops on cached geometry arrays.  Only rebuilds
            # when brightness bucket changes (0.02 steps).
            _b = getattr(self, '_brightness', 0.0)
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _has_compositor:
                _b_rounded = round(_b * 50) / 50  # 0.02 steps
                if getattr(self, '_sdf_appearance_b', -1.0) != _b_rounded:
                    # Interior floor: on light backgrounds the dark fill
                    # needs to be fully opaque so punch-through text has
                    # strong contrast.  On dark backgrounds a softer floor
                    # lets the frosted quality show.  The value here IS
                    # the alpha written into the image — layer opacity
                    # multiplies on top of it.
                    _ifloor = _lerp(0.60, 0.85, _clamp01(_b))
                    # Rescale raw interior curve to [_ifloor, 1.0] + edge ridge
                    interior = (_ifloor + (1.0 - _ifloor) * self._sdf_raw_interior)
                    interior = np.clip(
                        interior + self._sdf_edge_ridge * 0.50,
                        0.0, 1.0,
                    ).astype(np.float32)
                    fill_alpha = np.where(self._sdf_inside_mask, interior, self._sdf_ext_exterior)
                    self._cached_fill_alpha = fill_alpha
                    self._sdf_appearance_b = _b_rounded
                fill_alpha = self._cached_fill_alpha
            elif _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                fill_alpha = self._sdf_fallback_alpha
            else:
                fill_alpha = getattr(self, '_sdf_fallback_alpha', None)
                if fill_alpha is None:
                    return

            if _has_compositor:
                bg_r, bg_g, bg_b = _compositor_fill_color_for_brightness(_b)
            else:
                bg_r, bg_g, bg_b = _background_color_for_brightness(_b)
            fill_image, self._fill_payload = _fill_field_to_image(
                fill_alpha,
                int(bg_r * 255), int(bg_g * 255), int(bg_b * 255),
            )
        except (ImportError, Exception):
            return

        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setContents_(fill_image)
            self._fill_layer.setFrame_(((0, 0), (total_w, total_h)))
            if hasattr(self._fill_layer, "setCompositingFilter_"):
                if getattr(self, "_fullscreen_compositor", None) is not None:
                    # Graphic mode: normal alpha blending (no additive glow)
                    self._fill_layer.setCompositingFilter_(None)
                else:
                    self._fill_layer.setCompositingFilter_(
                        _fill_compositing_filter_for_brightness(getattr(self, "_brightness", 0.0))
                    )
        # Keep the spring tint layer's mask in sync with the fill shape
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            self._spring_tint_layer.setFrame_(((0, 0), (total_w, total_h)))
            # Use the fill image as a mask so the tint only shows within the overlay
            mask = CALayer.alloc().init()
            mask.setFrame_(((0, 0), (total_w, total_h)))
            mask.setContents_(fill_image)
            mask.setContentsGravity_("resize")
            self._spring_tint_layer.setMask_(mask)

    def _update_backdrop_capture_geometry(self):
        if self._window is None or self._content_view is None or self._screen is None:
            return None
        try:
            screen_frame = self._screen.frame()
            win_frame = self._window.frame()
            content_frame = self._content_view.frame()
        except Exception:
            return None

        capture_rect = _backdrop_capture_rect(
            screen_frame,
            win_frame,
            content_frame,
            getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points()),
        )
        pixel_size = _backdrop_capture_pixel_size(
            capture_rect,
            getattr(self, "_ridge_scale", 1.0),
        )
        self._backdrop_capture_rect = capture_rect
        self._backdrop_capture_pixel_size = pixel_size
        return capture_rect, pixel_size

    def _update_backdrop_mask(self, width: float, height: float):
        if self._backdrop_layer is None:
            return
        try:
            from .overlay import (
                _fill_field_to_image,
                _overlay_rounded_rect_sdf,
            )
        except Exception:
            return

        overscan = getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points())
        scale = getattr(self, "_ridge_scale", 2.0)
        mask_width_multiplier = getattr(
            self,
            "_backdrop_mask_width_multiplier",
            _COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER,
        )
        signature = (
            round(float(width), 3),
            round(float(height), 3),
            round(float(overscan), 3),
            round(float(scale), 3),
            round(float(mask_width_multiplier), 6),
            bool(_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED),
        )
        cached_signature = getattr(self, "_backdrop_mask_signature", None)
        cached_mask = getattr(self, "_backdrop_mask_layer", None)
        if signature == cached_signature and cached_mask is not None:
            self._backdrop_layer.setMask_(cached_mask)
            return
        inner_width = max(width - 2 * overscan, 1.0)
        inner_height = max(height - 2 * overscan, 1.0)
        try:
            if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
                # Use capsule SDF for the mask to match the warp shape.
                import numpy as np
                pw, ph = max(int(width), 1), max(int(height), 1)
                xs = np.arange(pw, dtype=np.float32)[None, :] + 0.5 - pw * 0.5
                ys = np.arange(ph, dtype=np.float32)[:, None] + 0.5 - ph * 0.5
                inner_half_w = inner_width * 0.5
                inner_half_h = inner_height * 0.5
                corner_r = _OVERLAY_HEIGHT / 4.0
                capsule_radius = max(min(corner_r, inner_half_h), 1.0)
                spine_half_x = max(inner_half_w - capsule_radius, 0.0)
                spine_half_y = max(inner_half_h - capsule_radius, 0.0)
                spine_dist_x = np.maximum(np.abs(xs) - spine_half_x, 0.0)
                spine_dist_y = np.maximum(np.abs(ys) - spine_half_y, 0.0)
                sdf = (np.hypot(spine_dist_x, spine_dist_y) - capsule_radius).astype(np.float32)
            else:
                sdf = _overlay_rounded_rect_sdf(
                    width,
                    height,
                    inner_width,
                    inner_height,
                    _OVERLAY_CORNER_RADIUS,
                    scale,
                )
            alpha = _backdrop_mask_alpha(
                sdf,
                width=max(
                    scale,
                    _command_backdrop_mask_falloff_width(scale)
                    * (
                        mask_width_multiplier
                        / max(_COMMAND_BACKDROP_MASK_WIDTH_MULTIPLIER, 1e-6)
                    ),
                ),
            )
            mask_image, self._backdrop_mask_payload = _fill_field_to_image(
                alpha,
                255,
                255,
                255,
            )
        except Exception:
            return

        mask = CALayer.alloc().init()
        mask.setFrame_(((0, 0), (width, height)))
        mask.setContents_(mask_image)
        mask.setContentsGravity_("resize")
        self._backdrop_mask_signature = signature
        self._backdrop_mask_layer = mask
        self._backdrop_layer.setMask_(mask)

    def _install_backdrop_frame_callback(self):
        renderer = getattr(self, "_backdrop_renderer", None)
        if renderer is None or not hasattr(renderer, "set_frame_callback"):
            return
        # CAMetalLayer: Metal drawable path handles presentation directly.
        # Frame callback (setContents_) would cause "undefined behavior"
        # warnings on CAMetalLayer.
        if getattr(self, "_backdrop_is_metal_layer", False):
            renderer.set_frame_callback(None)
            return
        # When optical shell is active, SCK frames route through the CGImage
        # path (not sample buffers), so we always need the frame callback.
        if not _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED:
            if self._backdrop_layer_uses_sample_buffers() or (
                _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE
            ):
                renderer.set_frame_callback(None)
                return

        def apply_live_frame(image) -> None:
            if self._backdrop_layer is None:
                return
            self._backdrop_layer.setContents_(image)

        renderer.set_frame_callback(apply_live_frame)

    def _install_backdrop_sample_buffer_callback(self):
        renderer = getattr(self, "_backdrop_renderer", None)
        layer = getattr(self, "_backdrop_layer", None)
        if (
            renderer is None
            or layer is None
            or not hasattr(renderer, "set_sample_buffer_callback")
            or not hasattr(layer, "enqueueSampleBuffer_")
        ):
            return

        def apply_live_sample_buffer(sample_buffer) -> None:
            if self._backdrop_layer is None:
                return
            try:
                self._backdrop_layer.enqueueSampleBuffer_(sample_buffer)
            except Exception:
                logger.debug("Failed to enqueue command backdrop sample buffer", exc_info=True)

        renderer.set_sample_buffer_callback(apply_live_sample_buffer)

    def _start_fullscreen_compositor(self):
        """Start the full-screen compositor for zero-seam optical shell."""
        self._stop_fullscreen_compositor()
        try:
            from spoke.fullscreen_compositor import FullScreenCompositor
            shell_config = self._current_optical_shell_config()
            if shell_config is None:
                return
            # Add capsule center position in pixel coordinates
            scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
            screen_frame = self._screen.frame()
            win_frame = self._window.frame()
            content_frame = self._content_view.frame()
            # Capsule center in screen points
            capsule_cx = win_frame.origin.x + content_frame.origin.x + content_frame.size.width / 2
            # Cocoa Y is bottom-up; Metal texture Y is top-down
            capsule_cy_cocoa = win_frame.origin.y + content_frame.origin.y + content_frame.size.height / 2
            capsule_cy_metal = screen_frame.size.height - capsule_cy_cocoa
            shell_config["center_x"] = capsule_cx * scale
            shell_config["center_y"] = capsule_cy_metal * scale
            # Scale content dimensions to pixel space
            for k in ("content_width_points", "content_height_points",
                      "corner_radius_points", "band_width_points",
                      "tail_width_points"):
                if k in shell_config:
                    shell_config[k] = float(shell_config[k]) * scale
            compositor = FullScreenCompositor(self._screen)
            # Exclude both the compositor's own window and the spoke overlay
            try:
                overlay_wid = int(self._window.windowNumber())
                compositor.set_excluded_window_ids([overlay_wid])
            except Exception:
                pass
            if compositor.start(shell_config):
                self._fullscreen_compositor = compositor
                # Cancel the old backdrop refresh timer — compositor replaces it.
                # Don't call stop_live_stream here (can deadlock if SCK callback
                # is in progress).  The old stream will be stopped when the
                # overlay hides.
                self._cancel_backdrop_refresh()
                # Hide the old backdrop layer — compositor renders the warp now.
                # The stale content in this layer would show on top of live output.
                if self._backdrop_layer is not None:
                    self._backdrop_layer.setHidden_(True)
                # Stop the old per-overlay Metal display link renderer —
                # it shares the same MetalWarpPipeline singleton and
                # would race with the compositor's display link on
                # shared mutable state (accum textures, params buffer).
                old_dl = getattr(self, "_metal_display_link_renderer", None)
                if old_dl is not None:
                    try:
                        old_dl.stop()
                    except Exception:
                        pass
                    self._metal_display_link_renderer = None
                self._enable_text_punchthrough(True)
                # Force fill image rebuild with compositor-specific
                # colors and alpha profile (invalidate SDF + brightness
                # caches so _apply_surface_theme triggers a full rebuild).
                self._sdf_appearance_b = -1.0
                self._fill_image_brightness = -1.0
                self._apply_surface_theme()
                logger.info("Command overlay: full-screen compositor started")
            else:
                logger.info("Command overlay: full-screen compositor failed to start")
        except Exception:
            logger.info("Command overlay: full-screen compositor unavailable", exc_info=True)

    def _enable_text_punchthrough(self, enabled: bool) -> None:
        """Toggle text punch-through mode.

        When enabled, the fill layer gets a mask that is opaque
        everywhere except where text glyphs are.  The mask is
        regenerated each pulse tick via ``_update_punchthrough_mask``.
        Text-shaped holes in the fill reveal the warped compositor
        content underneath.
        """
        self._text_punchthrough = enabled
        fill = getattr(self, "_fill_layer", None)
        content = getattr(self, "_content_view", None)
        if fill is None:
            return
        if enabled:
            # Hide the real text — the mask IS the text now
            if content is not None:
                content.setHidden_(True)
        else:
            fill.setMask_(None)
            self._punchthrough_mask_layer = None
            boost = getattr(self, "_boost_layer", None)
            if boost is not None:
                boost.setHidden_(True)
                boost.setOpacity_(0.0)
                boost.setMask_(None)
                self._boost_mask_layer = None
            if content is not None:
                content.setHidden_(False)

    def _update_punchthrough_mask(self) -> None:
        """Render text into an inverted mask for the fill layer.

        Called each pulse tick when punch-through is active.  Draws a
        white (opaque) rect, then stamps the current attributed text
        with kCGBlendModeDestinationOut to create transparent holes
        where text glyphs are — revealing the warped compositor
        content underneath.

        Uses NSAttributedString.drawInRect_ via NSGraphicsContext
        (no CoreText dependency).
        """
        fill = getattr(self, "_fill_layer", None)
        content = getattr(self, "_content_view", None)
        if fill is None or content is None:
            return
        ts = self._text_view.textStorage()
        if ts is None or (hasattr(ts, 'length') and ts.length() == 0):
            fill.setMask_(None)
            return
        try:
            from Quartz import (
                CGBitmapContextCreate,
                CGBitmapContextCreateImage,
                CGColorSpaceCreateDeviceRGB,
                kCGImageAlphaPremultipliedLast,
                kCGBlendModeDestinationOut,
                CGRectMake,
                CGContextSetRGBFillColor,
                CGContextFillRect,
                CGContextSetBlendMode,
                CGContextSaveGState,
                CGContextRestoreGState,
                CGContextTranslateCTM,
            )
            from AppKit import NSGraphicsContext
            from Foundation import NSMakeRect

            fill_frame = fill.frame()
            fw = int(fill_frame[1][0])
            fh = int(fill_frame[1][1])
            if fw <= 0 or fh <= 0:
                return

            # Content offset within the fill layer
            content_frame = content.frame()
            cx = content_frame[0][0]
            cy = content_frame[0][1]

            # Scroll offset and text frame
            scroll_origin = self._scroll_view.contentView().bounds().origin
            text_frame = self._text_view.frame()

            cs = CGColorSpaceCreateDeviceRGB()
            ctx = CGBitmapContextCreate(
                None, fw, fh, 8, fw * 4,
                cs, kCGImageAlphaPremultipliedLast,
            )
            if ctx is None:
                return

            # Fill entire mask white (opaque) — fill shows through here
            CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0)
            CGContextFillRect(ctx, CGRectMake(0, 0, fw, fh))

            # Use NSGraphicsContext (flipped to match NSTextView) to
            # draw the attributed string with destinationOut blending.
            nsctx = NSGraphicsContext.graphicsContextWithCGContext_flipped_(ctx, True)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.setCurrentContext_(nsctx)

            CGContextSaveGState(ctx)
            CGContextSetBlendMode(ctx, kCGBlendModeDestinationOut)

            # In a flipped context, (0,0) is top-left.  The fill layer
            # covers the full window including feather; text sits at
            # (feather + scroll_inset_x, feather + scroll_inset_y).
            # CG origin is bottom-left, but we flipped the NSGraphicsContext
            # so we need to flip Y for the CTM.
            from Quartz import CGContextScaleCTM
            CGContextTranslateCTM(ctx, 0, fh)
            CGContextScaleCTM(ctx, 1.0, -1.0)

            # Now (0,0) is top-left in the flipped sense.
            # Content view is at (cx, cy) in wrapper coords (bottom-up).
            # In top-down: content top = fh - cy - content_h
            content_h = content_frame[1][1]
            text_x = cx + 24.0
            text_y = (fh - cy - content_h) + 16.0 - scroll_origin.y

            text_w = text_frame.size.width
            text_h = text_frame.size.height
            ts.drawInRect_(NSMakeRect(text_x, text_y, text_w, text_h))

            CGContextRestoreGState(ctx)
            NSGraphicsContext.restoreGraphicsState()

            mask_image = CGBitmapContextCreateImage(ctx)
            if mask_image is None:
                return

            # Update or create mask layer for fill (inverted: holes where text is)
            mask_layer = getattr(self, "_punchthrough_mask_layer", None)
            if mask_layer is None:
                mask_layer = CALayer.alloc().init()
                mask_layer.setContentsGravity_("resize")
                self._punchthrough_mask_layer = mask_layer
            mask_layer.setFrame_(((0, 0), (fw, fh)))
            mask_layer.setContents_(mask_image)
            if fill.mask() is not mask_layer:
                fill.setMask_(mask_layer)

            # Boost mask: opaque where text is (inverse of fill mask).
            # Only generate when boost layer is visible.
            boost_layer = getattr(self, "_boost_layer", None)
            if boost_layer is not None and not boost_layer.isHidden():
                boost_ctx = CGBitmapContextCreate(
                    None, fw, fh, 8, fw * 4,
                    cs, kCGImageAlphaPremultipliedLast,
                )
                if boost_ctx is not None:
                    # Clear background (transparent) — only glyphs will have alpha
                    boost_nsctx = NSGraphicsContext.graphicsContextWithCGContext_flipped_(boost_ctx, True)
                    NSGraphicsContext.saveGraphicsState()
                    NSGraphicsContext.setCurrentContext_(boost_nsctx)
                    CGContextSaveGState(boost_ctx)
                    CGContextTranslateCTM(boost_ctx, 0, fh)
                    CGContextScaleCTM(boost_ctx, 1.0, -1.0)
                    ts.drawInRect_(NSMakeRect(text_x, text_y, text_w, text_h))
                    CGContextRestoreGState(boost_ctx)
                    NSGraphicsContext.restoreGraphicsState()

                    boost_mask_image = CGBitmapContextCreateImage(boost_ctx)
                    if boost_mask_image is not None:
                        boost_mask = getattr(self, "_boost_mask_layer", None)
                        if boost_mask is None:
                            boost_mask = CALayer.alloc().init()
                            boost_mask.setContentsGravity_("resize")
                            self._boost_mask_layer = boost_mask
                        boost_mask.setFrame_(((0, 0), (fw, fh)))
                        boost_mask.setContents_(boost_mask_image)
                        if boost_layer.mask() is not boost_mask:
                            boost_layer.setMask_(boost_mask)

        except Exception:
            logger.debug("Failed to update punch-through mask", exc_info=True)

    def _stop_fullscreen_compositor(self):
        compositor = getattr(self, "_fullscreen_compositor", None)
        self._fullscreen_compositor = None
        self._enable_text_punchthrough(False)
        # Unhide the old backdrop layer in case the old path resumes
        backdrop = getattr(self, "_backdrop_layer", None)
        if backdrop is not None:
            try:
                backdrop.setHidden_(False)
            except Exception:
                pass
        if compositor is not None:
            try:
                compositor.stop()
            except Exception:
                logger.debug("Failed to stop full-screen compositor", exc_info=True)
        # Invalidate fill caches so the non-compositor path rebuilds
        # with its own colors/alpha profile on next render tick.
        self._sdf_cache_key = None
        self._fill_image_brightness = -1.0
        self._apply_surface_theme()

    def _start_backdrop_refresh_timer(self):
        self._cancel_backdrop_refresh()
        # Full-screen compositor handles all rendering — no need for the
        # old per-overlay backdrop capture/warp/present path.
        if getattr(self, "_fullscreen_compositor", None) is not None:
            return
        if self._backdrop_renderer is None or self._backdrop_layer is None:
            return
        if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED and _COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE:
            return
        self._backdrop_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _COMMAND_BACKDROP_REFRESH_S,
            self,
            "backdropRefreshTick:",
            None,
            True,
        )
        _pin_timer_to_active_run_loop_modes(self._backdrop_timer)

    def backdropRefreshTick_(self, timer) -> None:
        if not self._visible:
            self._cancel_backdrop_refresh()
            return
        self._refresh_backdrop_snapshot()

    def _refresh_backdrop_snapshot(self):
        # Full-screen compositor handles all rendering
        if getattr(self, "_fullscreen_compositor", None) is not None:
            return None
        if (
            self._backdrop_renderer is None
            or self._backdrop_layer is None
            or self._window is None
            or self._content_view is None
        ):
            return None
        geometry = self._update_backdrop_capture_geometry()
        if geometry is None:
            return None
        capture_rect, _ = geometry
        try:
            window_number = self._window.windowNumber()
        except Exception:
            return None
        blur_radius_points = getattr(self, "_backdrop_blur_radius_points", _COMMAND_BACKDROP_BLUR_RADIUS)
        shell_config = self._current_optical_shell_config()
        if shell_config is not None and hasattr(self._backdrop_renderer, "set_live_optical_shell_config"):
            try:
                self._backdrop_renderer.set_live_optical_shell_config(shell_config)
            except Exception:
                logger.debug("Failed to refresh live command optical-shell config", exc_info=True)
        image = self._backdrop_renderer.capture_blurred_image(
            window_number=window_number,
            capture_rect=capture_rect,
            blur_radius_points=blur_radius_points,
        )
        direct_sample_path = False
        sample_buffer_query = getattr(self._backdrop_renderer, "uses_direct_sample_buffers", None)
        if callable(sample_buffer_query):
            try:
                direct_sample_path = sample_buffer_query(blur_radius_points) is True
            except Exception:
                direct_sample_path = False
        if image is None and not direct_sample_path:
            return None

        overscan = getattr(self, "_backdrop_capture_overscan_points", _command_backdrop_capture_overscan_points())
        content_frame = self._content_view.frame()
        local_x = content_frame.origin.x - overscan
        local_y = content_frame.origin.y - overscan
        local_w = capture_rect.size.width
        local_h = capture_rect.size.height
        self._backdrop_layer.setFrame_(((local_x, local_y), (local_w, local_h)))
        if image is not None and hasattr(self._backdrop_layer, "setContents_"):
            self._backdrop_layer.setContents_(image)
        self._update_backdrop_mask(local_w, local_h)
        return image

    # ── layout ──────────────────────────────────────────────

    def _update_layout(self) -> None:
        """Resize window and scroll to bottom after text change."""
        try:
            layout = self._text_view.layoutManager()
            container = self._text_view.textContainer()
            if layout and container:
                layout.ensureLayoutForTextContainer_(container)
                text_rect = layout.usedRectForTextContainer_(container)
                text_height = text_rect.size.height
            else:
                text_height = _OVERLAY_HEIGHT - 32

            max_height = _max_overlay_height(self._screen.frame().size.height)
            new_height = min(max(_OVERLAY_HEIGHT, text_height + 24), max_height)

            f = _OPTICAL_SHELL_FEATHER if _COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED else _OUTER_FEATHER
            win_frame = self._window.frame()
            new_win_h = new_height + 2 * f
            if abs(win_frame.size.height - new_win_h) > 4:
                win_frame.size.height = new_win_h
                self._window.setFrame_display_animate_(win_frame, True, False)
                self._content_view.setFrame_(
                    NSMakeRect(f, f, _OVERLAY_WIDTH, new_height)
                )
                self._scroll_view.setFrame_(
                    NSMakeRect(48, 8, _OVERLAY_WIDTH - 96, new_height - 16)
                )
                self._apply_ridge_masks(_OVERLAY_WIDTH, new_height)
                self._update_backdrop_capture_geometry()
                shell_config = self._current_optical_shell_config()
                if shell_config is not None and hasattr(self._backdrop_renderer, "set_live_optical_shell_config"):
                    self._backdrop_renderer.set_live_optical_shell_config(shell_config)
                compositor = getattr(self, "_fullscreen_compositor", None)
                if compositor is not None and shell_config is not None:
                    scale = self._screen.backingScaleFactor() if hasattr(self._screen, "backingScaleFactor") else 2.0
                    screen_frame = self._screen.frame()
                    win_frame = self._window.frame()
                    content_frame = self._content_view.frame()
                    capsule_cx = win_frame.origin.x + content_frame.origin.x + content_frame.size.width / 2
                    capsule_cy_cocoa = win_frame.origin.y + content_frame.origin.y + content_frame.size.height / 2
                    shell_config["center_x"] = capsule_cx * scale
                    shell_config["center_y"] = (screen_frame.size.height - capsule_cy_cocoa) * scale
                    for k in ("content_width_points", "content_height_points",
                              "corner_radius_points", "band_width_points",
                              "tail_width_points"):
                        if k in shell_config:
                            shell_config[k] = float(shell_config[k]) * scale
                    compositor.update_shell_config(shell_config)
                if self._visible:
                    self._refresh_backdrop_snapshot()

            end = (self._text_view.string().length()
                   if hasattr(self._text_view.string(), 'length')
                   else len(self._response_text))
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            logger.debug("Command overlay layout update failed", exc_info=True)

    def _apply_surface_theme(self) -> None:
        if self._content_view is None:
            return
        # Keep the content view transparent so the same SDF fill/material carries
        # the assistant window chrome as the preview overlay.
        self._content_view.layer().setBackgroundColor_(None)
        if hasattr(self, "_fill_layer") and self._fill_layer is not None and hasattr(
            self._fill_layer, "setCompositingFilter_"
        ):
            if getattr(self, "_fullscreen_compositor", None) is not None:
                self._fill_layer.setCompositingFilter_(None)
            else:
                self._fill_layer.setCompositingFilter_(
                    _fill_compositing_filter_for_brightness(self._brightness)
            )
        last_t = getattr(self, "_fill_image_brightness", -1.0)
        # Tighter threshold when compositor is active — the graphic
        # fill color tracks brightness more visibly than the glow.
        threshold = 0.01 if getattr(self, "_fullscreen_compositor", None) is not None else 0.03
        if abs(self._brightness - last_t) > threshold:
            self._fill_image_brightness = self._brightness
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)
        self._apply_thinking_label_theme()
        self._apply_narrator_theme()

    def _apply_thinking_label_theme(self) -> None:
        if self._thinking_label is None or self._thinking_label.isHidden():
            return
        if self._thinking_inverted:
            cut_r, cut_g, cut_b = _thinking_cutout_color_for_brightness(self._brightness)
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(cut_r, cut_g, cut_b, 0.7)
            )
            return
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.7
            )
        )
