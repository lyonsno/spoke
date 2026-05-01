"""Screen-border glow overlay that pulses with voice amplitude.

A borderless, transparent, click-through NSWindow that draws a soft glow
around the screen edges. Intensity follows the RMS amplitude of the
microphone input, with fast rise and slow decay for a breathing effect.
"""

from __future__ import annotations
import colorsys
import logging
import math
import os
import time

import objc
from AppKit import (
    NSColor,
    NSScreen,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSData, NSObject, NSTimer
from Quartz import (
    CABasicAnimation,
    CALayer,
    CAMediaTimingFunction,
)

from .optical_shell_metrics import OpticalShellMetrics

logger = logging.getLogger(__name__)


def _scale_color_saturation(
    color: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    """Scale an RGB color's saturation while keeping its hue and value stable."""
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    return colorsys.hsv_to_rgb(hue, min(max(saturation * factor, 0.0), 1.0), value)

# Glow appearance
_GLOW_COLOR = (0.38, 0.52, 1.0)  # saturated cornflower — SC2 Protoss energy field
_GLOW_COLOR_DARK = _scale_color_saturation(
    (0.50, 0.59, 0.84), 0.40
)  # much closer to keyboard white on darker backgrounds
_GLOW_COLOR_LIGHT = _scale_color_saturation(
    (0.34, 0.50, 1.0), 0.50
)  # lighter backgrounds still read blue, but with half the prior saturation
_GLOW_CAP_COLOR = (1.0, 0.45, 0.15)  # angry sunset for cap countdown
_GLOW_WIDTH = 10.0  # thinner source — less intrusion into screen
_GLOW_SHADOW_RADIUS = 60.0  # broader bloom so a dimmer peak still reads as glow
_GLOW_MAX_OPACITY = 1.0  # bright scenes can drive the glow all the way to full strength
_GLOW_BASE_OPACITY = 0.1449  # 140% of the calmer baseline so the border keeps dancing at rest
_GLOW_PEAK_TARGET = 0.90
_GLOW_BASE_OPACITY_DARK = 0.375
_GLOW_BASE_OPACITY_LIGHT = 0.4116
_GLOW_PEAK_TARGET_DARK = 0.90
_GLOW_PEAK_TARGET_LIGHT = _GLOW_MAX_OPACITY
_EDGE_INNER_SATURATION_SCALE = 0.70
_EDGE_OUTER_SATURATION_SCALE = 1.80
# MacBook Pro 14"/16" (2021+) has asymmetric display corners.
# We use slightly tighter radii than the physical bezel so the glow
# source stays close to the corners — the bezel hides the overshoot.
# Keyed by (native_pixel_width, native_pixel_height).
_DISPLAY_CORNER_RADII: dict[tuple[int, int], tuple[float, float]] = {
    # 16" MacBook Pro (2021+): 3456×2234 native
    (3456, 2234): (20.0, 8.0),
    # 14" MacBook Pro (2021+): 3024×1964 native
    (3024, 1964): (10.0, 6.0),  # same baseline as 16"; we tune visually from here
}
_DISPLAY_NOTCH_PROFILE: dict[tuple[int, int], dict[str, float]] = {
    # Exact top-edge notch row profiles extracted from Apple's official
    # MacBook Pro M4 bezel PSD resources after flattening the visible opening.
    # We use M4 as the public-source baseline for current 14"/16" MacBook Pro
    # panels because it is closer to the local M2 Pro hardware than the newer
    # M5 package, and the big-box 16" machine is itself M4.
    # Values are the center-gap widths, in native pixels, for each row from
    # the top edge of the opening down through the 64 px notch height.
    (3024, 1964): {
        "profile_widths": (
            386, 380, 376, 376, 374, 372, 372, 372, 372, 372, 372, 370, 370,
            370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370,
            370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 370,
            370, 370, 370, 370, 370, 370, 370, 370, 370, 370, 368, 368, 366,
            366, 364, 364, 362, 360, 358, 356, 354, 352, 348, 344, 336,
        ),
        "bottom_radius": 0.0,
        "shoulder_smoothing": 3.0,
    },
    (3456, 2234): {
        "profile_widths": (
            386, 380, 378, 376, 374, 372, 372, 372, 372, 372, 372, 372, 372,
            372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372,
            372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372, 372,
            372, 372, 372, 372, 370, 370, 370, 370, 370, 370, 368, 368, 368,
            366, 366, 364, 362, 360, 360, 356, 354, 352, 348, 344, 336,
        ),
        "bottom_radius": 0.0,
        "shoulder_smoothing": 3.0,
    },
}
_DISPLAY_CORNER_FINISH_RADII: dict[tuple[int, int], tuple[float, float]] = {
    # Physical-ish bezel read for the outer-corner finish field. We keep the
    # live outline slightly tighter, then borrow these broader radii only in a
    # bounded corner zone so the visible roll feels softer without moving the
    # main perimeter contract.
    (3456, 2234): (20.0, 8.0),
    (3024, 1964): (18.0, 10.0),
}
_CORNER_RADIUS_TOP_DEFAULT = 10.0
_CORNER_RADIUS_BOTTOM_DEFAULT = 6.0
_CORNER_FINISH_RADIUS_TOP_DEFAULT = 18.0
_CORNER_FINISH_RADIUS_BOTTOM_DEFAULT = 10.0

_GLOW_MULTIPLIER = float(os.environ.get("SPOKE_GLOW_MULTIPLIER", "21.4"))
_GLOW_TEST_RMS = os.environ.get("SPOKE_GLOW_TEST_RMS")
_DIM_SCREEN = os.environ.get("SPOKE_DIM_SCREEN", "1") == "1"
_DIM_OPACITY_DARK = 0.42  # dim on dark backgrounds
_DIM_OPACITY_LIGHT = 0.636  # pumped 50%
_DIM_SDF_BLOOM_MULTIPLIER = 2.25

def _dim_target_for_brightness(brightness: float) -> float:
    # Spike to 0.80 at mid-gray
    if brightness <= 0.5:
        t = brightness / 0.5
        return _DIM_OPACITY_DARK + t * (0.80 - _DIM_OPACITY_DARK)
    else:
        t = (brightness - 0.5) / 0.5
        return 0.80 + t * (_DIM_OPACITY_LIGHT - 0.80)

# Amplitude smoothing: rise fast, decay slow
_RISE_FACTOR = 0.99  # 3x faster (was 0.90)
_DECAY_FACTOR = 0.16 # 3x faster (was 0.50)

# Fade timing (all 3x faster)
_FADE_IN_S = 0.026
_FADE_OUT_S = 0.066
_GLOW_SHOW_FADE_S = 0.066
_GLOW_HIDE_FADE_S = 0.2
_GLOW_SHOW_TIMING = "easeIn"
_DIM_SHOW_FADE_S = 0.36
_DIM_HIDE_FADE_S = 0.8
_WINDOW_TEARDOWN_CUSHION_S = 0.016


_BRIGHTNESS_SAMPLE_INTERVAL = 1.0  # seconds between recurring samples
_BRIGHTNESS_PATCH_SIZE = 50  # pixels per patch side
_DISTANCE_FIELD_SCALE_DEFAULT = 2.0
_NOTCH_BOTTOM_RADIUS = 8.0
_NOTCH_SHOULDER_SMOOTHING = 9.5
_LIGHT_BACKGROUND_EDGE_START = 0.55
_LIGHT_BACKGROUND_EDGE_BOOST = 0.664
_VIGNETTE_OPACITY_SCALE = 4.575  # back to original
_NOTCH_TOP_CORNER_GLOW_X_BAND = 14.0
_NOTCH_TOP_CORNER_GLOW_Y_BAND = 36.0
_NOTCH_TOP_CORNER_GLOW_ATTENUATION = 0.95
_NOTCH_TOP_CORNER_HELPER_X_BAND = 28.0
_NOTCH_TOP_CORNER_HELPER_Y_BAND = 52.0
_NOTCH_TOP_CORNER_HELPER_ATTENUATION = 1.0
_NOTCH_TOP_CORNER_HELPER_FALLOFF_SCALE = 1.75
_SCREEN_CORNER_HELPER_X_BAND = 34.0
_SCREEN_CORNER_HELPER_Y_BAND = 34.0
_SCREEN_CORNER_HELPER_ATTENUATION = 0.8
_SCREEN_CORNER_HELPER_CORNER_SMOOTHING = 40.0
_SCREEN_CORNER_HELPER_FALLOFF_SCALE = 1.4


def _sample_screen_brightness(screen) -> float:
    """Sample average brightness from 4 small patches (one per quadrant).

    Each patch is 50x50 pixels, inset 25% from each screen edge to avoid
    our own glow/vignette. Returns 0.0 (black) to 1.0 (white).
    ~4x faster than a fullscreen capture on retina displays.
    """
    try:
        from Quartz import (
            CGWindowListCreateImage,
            kCGWindowListOptionOnScreenBelowWindow,
            kCGNullWindowID,
            CGImageGetWidth,
            CGImageGetHeight,
            CGImageGetDataProvider,
            CGDataProviderCopyData,
        )
        frame = screen.frame()
        fw, fh = frame.size.width, frame.size.height
        ox, oy = frame.origin.x, frame.origin.y
        ps = _BRIGHTNESS_PATCH_SIZE

        # 4 patches at 25%/75% of each axis
        patch_centers = [
            (0.25, 0.25),  # bottom-left quadrant
            (0.75, 0.25),  # bottom-right
            (0.25, 0.75),  # top-left
            (0.75, 0.75),  # top-right
        ]

        total = 0.0
        samples = 0

        for cx_frac, cy_frac in patch_centers:
            px = ox + fw * cx_frac - ps / 2
            py = oy + fh * cy_frac - ps / 2
            rect = ((px, py), (ps, ps))

            image = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenBelowWindow,
                kCGNullWindowID,
                0,
            )
            if image is None:
                continue

            w = CGImageGetWidth(image)
            h = CGImageGetHeight(image)
            data = CGDataProviderCopyData(CGImageGetDataProvider(image))
            if data is None or len(data) == 0 or w * h == 0:
                continue

            bpp = len(data) // (w * h)
            # Sample every 5th pixel for speed
            step = max(5, 1)
            for sy in range(0, h, step):
                for sx in range(0, w, step):
                    offset = (sy * w + sx) * bpp
                    if offset + 3 <= len(data):
                        r, g, b = data[offset], data[offset + 1], data[offset + 2]
                        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                        total += lum
                        samples += 1

        return total / samples if samples > 0 else 0.5
    except Exception:
        logger.debug("Screen brightness sampling failed", exc_info=True)
        return 0.5


def _compress_screen_glow_peak(opacity: float) -> float:
    """Keep quiet glow intact while capping only the top end."""
    return min(opacity, _GLOW_PEAK_TARGET)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _lerp_color(start: tuple[float, float, float], end: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return tuple(_lerp(s, e, t) for s, e in zip(start, end))


def _glow_style_for_brightness(brightness: float) -> tuple[tuple[float, float, float], float, float]:
    """Derive glow color and intensity from sampled background brightness."""
    t = min(max(brightness, 0.0), 1.0)
    color = _lerp_color(_GLOW_COLOR_DARK, _GLOW_COLOR_LIGHT, t)
    base_opacity = _lerp(_GLOW_BASE_OPACITY_DARK, _GLOW_BASE_OPACITY_LIGHT, t)
    peak_target = _lerp(_GLOW_PEAK_TARGET_DARK, _GLOW_PEAK_TARGET_LIGHT, t)
    return color, base_opacity, peak_target


def _update_noise_floor(noise_floor: float, rms: float) -> tuple[float, float]:
    """Advance adaptive ambient floor and return (floor, signal_above_floor)."""
    if rms < noise_floor or noise_floor == 0.0:
        noise_floor += (rms - noise_floor) * 0.05
    else:
        noise_floor += (rms - noise_floor) * 0.002
    signal = max(rms - noise_floor, 0.0)
    return noise_floor, signal


def _edge_mix_for_brightness(brightness: float) -> tuple[float, float]:
    """Keep dark scenes purely additive, then fade in the vignette only on light backgrounds."""
    t = min(max(brightness, 0.0), 1.0)
    if t <= _LIGHT_BACKGROUND_EDGE_START:
        return 1.0, 0.0

    edge_t = (t - _LIGHT_BACKGROUND_EDGE_START) / (1.0 - _LIGHT_BACKGROUND_EDGE_START)
    additive_mix = 1.0 - edge_t
    subtractive_mix = edge_t * (1.0 + _LIGHT_BACKGROUND_EDGE_BOOST * edge_t)
    return additive_mix, subtractive_mix


def _edge_band_colors(
    base_color: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Derive tighter and wider edge bands from the current base glow color."""
    inner = _scale_color_saturation(base_color, _EDGE_INNER_SATURATION_SCALE)
    middle = base_color
    outer = _scale_color_saturation(base_color, _EDGE_OUTER_SATURATION_SCALE)
    return inner, middle, outer


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _screen_rect(screen, name: str) -> tuple[float, float, float, float] | None:
    accessor = getattr(screen, name, None)
    if accessor is None:
        return None
    try:
        rect = accessor() if callable(accessor) else accessor
    except Exception:
        return None
    if rect is None:
        return None
    try:
        return (
            _safe_float(rect.origin.x, 0.0),
            _safe_float(rect.origin.y, 0.0),
            _safe_float(rect.size.width, 0.0),
            _safe_float(rect.size.height, 0.0),
        )
    except Exception:
        return None


def _screen_backing_scale(screen) -> float:
    accessor = getattr(screen, "backingScaleFactor", None)
    if accessor is None:
        return _DISTANCE_FIELD_SCALE_DEFAULT
    try:
        value = accessor() if callable(accessor) else accessor
    except Exception:
        return _DISTANCE_FIELD_SCALE_DEFAULT
    scale = _safe_float(value, _DISTANCE_FIELD_SCALE_DEFAULT)
    return scale if scale > 0.0 else _DISTANCE_FIELD_SCALE_DEFAULT


def _display_shape_geometry(screen, width_pt: float, height_pt: float, scale: float) -> dict:
    """Resolve the live display silhouette used by the distance-field renderer."""
    pixel_width = max(int(round(width_pt * scale)), 1)
    pixel_height = max(int(round(height_pt * scale)), 1)
    top_r, bot_r = _DISPLAY_CORNER_RADII.get(
        (pixel_width, pixel_height),
        (_CORNER_RADIUS_TOP_DEFAULT, _CORNER_RADIUS_BOTTOM_DEFAULT),
    )
    geometry = {
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "top_radius": top_r * scale,
        "bottom_radius": bot_r * scale,
        "notch": None,
    }

    left_area = _screen_rect(screen, "auxiliaryTopLeftArea")
    right_area = _screen_rect(screen, "auxiliaryTopRightArea")
    if left_area is None or right_area is None:
        return geometry

    left_max_x = left_area[0] + left_area[2]
    right_min_x = right_area[0]
    notch_width = max(right_min_x - left_max_x, 0.0)
    notch_base_y = min(left_area[1], right_area[1])
    notch_height = max(height_pt - notch_base_y, 0.0)
    if notch_width <= 0.0 or notch_height <= 0.0:
        return geometry
    notch_profile = _DISPLAY_NOTCH_PROFILE.get((pixel_width, pixel_height), {})
    notch_bottom_r = notch_profile.get("bottom_radius", _NOTCH_BOTTOM_RADIUS)
    notch_shoulder = notch_profile.get("shoulder_smoothing", _NOTCH_SHOULDER_SMOOTHING)
    helper_bottom_r = _NOTCH_BOTTOM_RADIUS
    helper_shoulder = _NOTCH_SHOULDER_SMOOTHING

    geometry["notch"] = {
        "x": left_max_x * scale,
        "y": notch_base_y * scale,
        "width": notch_width * scale,
        "height": notch_height * scale,
        "bottom_radius": min(notch_height * scale * 0.45, notch_bottom_r * scale),
        "helper_bottom_radius": min(
            notch_height * scale * 0.45,
            helper_bottom_r * scale,
        ),
        "helper_shoulder_smoothing": helper_shoulder * scale,
        "shoulder_smoothing": notch_shoulder * scale,
        "body_inset": max(notch_profile.get("body_inset", 0.0), 0.0) * scale,
        "top_cap_height": max(notch_profile.get("top_cap_height", 0.0), 0.0) * scale,
        "top_cap_bottom_radius": max(notch_profile.get("top_cap_bottom_radius", 0.0), 0.0) * scale,
        "profile_widths": notch_profile.get("profile_widths"),
    }
    return geometry


def _asymmetric_rounded_rect_sdf(
    x,
    y,
    width: float,
    height: float,
    top_radius: float,
    bottom_radius: float,
    corner_smoothing: float = 24.0,
) :
    import numpy as np

    half_width = width * 0.5
    half_height = height * 0.5
    radii = np.where(y >= 0.0, top_radius, bottom_radius).astype(np.float32, copy=False)
    qx = np.abs(x) - (half_width - radii)
    qy = np.abs(y) - (half_height - radii)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))

    if corner_smoothing <= 0.0:
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside - radii

    # Smooth the inner corner ridge where the negative distances meet.
    seam = np.maximum(corner_smoothing - np.abs(qx - qy), 0.0)
    smoothed_max = np.maximum(qx, qy) + (seam * seam) * (0.25 / corner_smoothing)
    inside = np.minimum(smoothed_max, 0.0)
    return outside + inside - radii


def _notch_signed_distance_field(centered_x, centered_y, notch: dict):
    import numpy as np

    width = notch["width"]
    height = notch["height"]
    profile_widths = notch.get("profile_widths")
    if profile_widths:
        half_widths = _notch_profile_half_widths(centered_y, notch)
        horizontal_signed = np.abs(centered_x) - half_widths
        vertical_signed = np.maximum(centered_y - (height * 0.5), (-height * 0.5) - centered_y)
        return np.maximum(horizontal_signed, vertical_signed)

    body_inset = max(notch.get("body_inset", 0.0), 0.0)
    top_cap_height = min(max(notch.get("top_cap_height", 0.0), 0.0), height)
    top_cap_bottom_radius = max(notch.get("top_cap_bottom_radius", 0.0), 0.0)

    notch_signed = _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y,
        width,
        height,
        0.0,
        notch["bottom_radius"],
        corner_smoothing=0.0,
    )
    if body_inset <= 0.0 or top_cap_height <= 0.0 or top_cap_height >= height:
        return notch_signed

    body_width = max(width - body_inset * 2.0, 1.0)
    body_height = max(height - top_cap_height, 1.0)
    top_cap_center_y = height * 0.5 - top_cap_height * 0.5
    body_center_y = -top_cap_height * 0.5

    top_cap_signed = _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y - top_cap_center_y,
        width,
        top_cap_height,
        0.0,
        top_cap_bottom_radius,
        corner_smoothing=0.0,
    )
    body_signed = _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y - body_center_y,
        body_width,
        body_height,
        0.0,
        notch["bottom_radius"],
        corner_smoothing=0.0,
    )
    return np.minimum(top_cap_signed, body_signed)


def _resampled_notch_profile_widths(notch: dict):
    import numpy as np

    profile_widths = notch.get("profile_widths")
    if not profile_widths:
        return None

    height = notch["height"]
    profile = np.asarray(profile_widths, dtype=np.float32)
    target_height = max(int(round(height)), 1)
    if profile.size != target_height:
        src_rows = np.arange(profile.size, dtype=np.float32)
        dst_rows = np.linspace(0.0, profile.size - 1.0, num=target_height, dtype=np.float32)
        profile = np.interp(dst_rows, src_rows, profile).astype(np.float32)
    return profile


def _notch_profile_half_widths(centered_y, notch: dict):
    import numpy as np

    profile = _resampled_notch_profile_widths(notch)
    if profile is None:
        return np.full_like(centered_y, notch["width"] * 0.5, dtype=np.float32)

    target_height = profile.size
    height = notch["height"]
    row_from_top = (height * 0.5) - centered_y - 0.5
    sample_rows = np.clip(row_from_top, 0.0, float(target_height - 1))
    row_indices = np.rint(sample_rows).astype(np.int32, copy=False)
    return (profile[row_indices] * 0.5).astype(np.float32, copy=False)


def _notch_bottom_half_width(notch: dict) -> float:
    profile_widths = notch.get("profile_widths")
    if profile_widths:
        return float(profile_widths[-1]) * 0.5
    return float(notch["width"]) * 0.5


def _notch_shoulder_distance_field(centered_x, centered_y, notch: dict):
    helper_notch = dict(notch)
    helper_notch.pop("profile_widths", None)
    helper_notch["bottom_radius"] = max(
        helper_notch.get("helper_bottom_radius", 0.0),
        helper_notch.get("bottom_radius", 0.0),
    )
    return _notch_signed_distance_field(centered_x, centered_y, helper_notch)


def _legacy_notch_corner_signed_distance_field(geometry: dict):
    """Replay the pre-Apple continuous notch field for corner finishing only."""
    import numpy as np

    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_x = x - (width * 0.5)
    centered_y = (height - y) - (height * 0.5)

    signed = _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y,
        width,
        height,
        geometry["top_radius"],
        geometry["bottom_radius"],
    )

    notch = geometry.get("notch")
    if notch is None:
        return signed.astype(np.float32, copy=False)

    helper_notch = dict(notch)
    helper_notch.pop("profile_widths", None)
    helper_notch["bottom_radius"] = max(
        helper_notch.get("helper_bottom_radius", 0.0),
        helper_notch.get("bottom_radius", 0.0),
    )
    helper_notch["shoulder_smoothing"] = max(
        helper_notch.get("helper_shoulder_smoothing", 0.0),
        helper_notch.get("shoulder_smoothing", 0.0),
    )

    notch_center_x = notch["x"] + notch["width"] * 0.5 - width * 0.5
    notch_center_y = notch["y"] + notch["height"] * 0.5 - height * 0.5
    notch_signed = _notch_signed_distance_field(
        centered_x - notch_center_x,
        centered_y - notch_center_y,
        helper_notch,
    )
    outline = np.maximum(signed, -notch_signed)
    shoulder_smoothing = max(helper_notch.get("shoulder_smoothing", 0.0), 0.0)
    if shoulder_smoothing <= 0.0:
        return outline.astype(np.float32, copy=False)

    seam = np.maximum(shoulder_smoothing - np.abs(signed + notch_signed), 0.0)
    softened = outline + (seam * seam) * (0.25 / shoulder_smoothing)
    return softened.astype(np.float32, copy=False)


def _screen_corner_glow_lift_mask(
    geometry: dict,
    *,
    x_band: float | None = None,
    y_band: float | None = None,
    attenuation: float | None = None,
):
    import numpy as np

    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_y = (height - y) - (height * 0.5)

    x_band = (
        _SCREEN_CORNER_HELPER_X_BAND if x_band is None else float(x_band)
    )
    y_band = (
        _SCREEN_CORNER_HELPER_Y_BAND if y_band is None else float(y_band)
    )
    attenuation = (
        _SCREEN_CORNER_HELPER_ATTENUATION
        if attenuation is None
        else float(attenuation)
    )

    left = np.maximum(x_band - x, 0.0) / x_band
    right = np.maximum(x_band - (width - x), 0.0) / x_band
    top = np.maximum(y_band - ((height * 0.5) - centered_y), 0.0) / y_band
    bottom = np.maximum(y_band - ((height * 0.5) + centered_y), 0.0) / y_band
    horizontal = np.maximum(left, right)
    vertical = np.maximum(top, bottom)
    horizontal = horizontal * horizontal * (3.0 - (2.0 * horizontal))
    vertical = vertical * vertical * (3.0 - (2.0 * vertical))
    return np.clip(horizontal * vertical * attenuation, 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _soft_display_corner_signed_distance_field(geometry: dict):
    import numpy as np

    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_x = x - (width * 0.5)
    centered_y = (height - y) - (height * 0.5)
    pixel_key = (width, height)
    base_top, base_bottom = _DISPLAY_CORNER_RADII.get(
        pixel_key,
        (_CORNER_RADIUS_TOP_DEFAULT, _CORNER_RADIUS_BOTTOM_DEFAULT),
    )
    finish_top, finish_bottom = _DISPLAY_CORNER_FINISH_RADII.get(
        pixel_key,
        (_CORNER_FINISH_RADIUS_TOP_DEFAULT, _CORNER_FINISH_RADIUS_BOTTOM_DEFAULT),
    )
    top_scale = geometry["top_radius"] / max(base_top, 1e-6)
    bottom_scale = geometry["bottom_radius"] / max(base_bottom, 1e-6)
    finish_top_radius = max(geometry["top_radius"], finish_top * top_scale)
    finish_bottom_radius = max(geometry["bottom_radius"], finish_bottom * bottom_scale)
    return _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y,
        width,
        height,
        finish_top_radius,
        finish_bottom_radius,
        corner_smoothing=_SCREEN_CORNER_HELPER_CORNER_SMOOTHING,
    ).astype(np.float32, copy=False)

def _display_signed_distance_field(geometry: dict, *, soften_notch_shoulders: bool = False):
    """Signed distance field for the live display outline, including the notch when available."""
    import numpy as np

    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_x = x - (width * 0.5)
    centered_y = (height - y) - (height * 0.5)

    signed = _asymmetric_rounded_rect_sdf(
        centered_x,
        centered_y,
        width,
        height,
        geometry["top_radius"],
        geometry["bottom_radius"],
    )

    notch = geometry.get("notch")
    if notch is None:
        return signed.astype(np.float32, copy=False)

    notch_center_x = notch["x"] + notch["width"] * 0.5 - width * 0.5
    notch_center_y = notch["y"] + notch["height"] * 0.5 - height * 0.5
    notch_local_x = centered_x - notch_center_x
    notch_local_y = centered_y - notch_center_y
    notch_signed = _notch_signed_distance_field(
        notch_local_x,
        notch_local_y,
        notch,
    )
    outline = np.maximum(signed, -notch_signed)
    shoulder_smoothing = max(notch.get("shoulder_smoothing", 0.0), 0.0)
    if shoulder_smoothing <= 0.0 or not soften_notch_shoulders:
        return outline.astype(np.float32, copy=False)

    shoulder_notch_signed = _notch_shoulder_distance_field(
        notch_local_x, notch_local_y, notch
    )
    helper_outline = np.maximum(signed, -shoulder_notch_signed)
    shoulder_anchor = _notch_bottom_half_width(notch)
    helper_shoulder_smoothing = max(
        notch.get("helper_shoulder_smoothing", shoulder_smoothing),
        shoulder_smoothing,
    )
    shoulder_band = helper_shoulder_smoothing * 4.0
    shoulder_proximity = np.maximum(
        shoulder_band - np.abs(np.abs(notch_local_x) - shoulder_anchor),
        0.0,
    ) / shoulder_band
    bottom_edge_y = -notch["height"] * 0.5
    shoulder_vertical = (
        (notch_local_y <= notch["height"] * 0.5)
        & (notch_local_y >= bottom_edge_y - shoulder_band)
    )
    inside_margin = 2.0
    shoulder_zone = (
        (shoulder_proximity > 0.0)
        & (shoulder_vertical > 0.0)
        & ((-outline) > inside_margin)
    )
    softened = outline.copy()
    softened[shoulder_zone] = np.minimum(
        helper_outline[shoulder_zone],
        -inside_margin,
    )
    return softened.astype(np.float32, copy=False)


def _distance_field_opacity(distance: float, falloff: float, power: float) -> float:
    normalized = max(distance, 0.0) / max(falloff, 1e-6)
    return math.exp(-(normalized ** power))


def _distance_field_alpha(signed_distance, falloff: float, power: float):
    import numpy as np

    distance = np.clip(-signed_distance, 0.0, None)
    alpha = np.exp(-np.power(distance / max(falloff, 1e-6), power, dtype=np.float32))
    return np.where(signed_distance < 0.0, alpha, 0.0).astype(np.float32, copy=False)


def _notch_top_corner_glow_mask(
    geometry: dict,
    *,
    x_band: float | None = None,
    y_band: float | None = None,
    attenuation: float | None = None,
):
    import numpy as np

    notch = geometry.get("notch")
    if notch is None:
        return None

    width = geometry["pixel_width"]
    height = geometry["pixel_height"]
    x = np.arange(width, dtype=np.float32)[None, :] + 0.5
    y = np.arange(height, dtype=np.float32)[:, None] + 0.5
    centered_x = x - (width * 0.5)
    centered_y = (height - y) - (height * 0.5)

    notch_center_x = notch["x"] + notch["width"] * 0.5 - width * 0.5
    notch_center_y = notch["y"] + notch["height"] * 0.5 - height * 0.5
    notch_local_x = centered_x - notch_center_x
    notch_local_y = centered_y - notch_center_y

    top_half_width = (
        float(notch["profile_widths"][0]) * 0.5
        if notch.get("profile_widths")
        else notch["width"] * 0.5
    )
    x_band = _NOTCH_TOP_CORNER_GLOW_X_BAND if x_band is None else float(x_band)
    y_band = _NOTCH_TOP_CORNER_GLOW_Y_BAND if y_band is None else float(y_band)
    attenuation = (
        _NOTCH_TOP_CORNER_GLOW_ATTENUATION
        if attenuation is None
        else float(attenuation)
    )
    x_proximity = np.maximum(
        x_band - np.abs(np.abs(notch_local_x) - top_half_width),
        0.0,
    ) / x_band
    y_proximity = np.maximum(
        y_band - ((notch["height"] * 0.5) - notch_local_y),
        0.0,
    ) / y_band
    x_proximity = x_proximity * x_proximity * (3.0 - (2.0 * x_proximity))
    y_proximity = y_proximity * y_proximity * (3.0 - (2.0 * y_proximity))
    attenuation = 1.0 - (x_proximity * y_proximity * attenuation)
    return np.clip(attenuation, 1e-4, 1.0).astype(np.float32, copy=False)


def _alpha_field_to_image(alpha):
    """Convert a float alpha field into a CGImage suitable for a CALayer mask."""
    import numpy as np

    from Quartz import (
        CGColorSpaceCreateDeviceRGB,
        CGDataProviderCreateWithCFData,
        CGImageCreate,
        kCGImageAlphaPremultipliedLast,
        kCGRenderingIntentDefault,
    )

    mask_alpha = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba = np.empty(mask_alpha.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = mask_alpha
    payload = NSData.dataWithBytes_length_(rgba.tobytes(), int(rgba.nbytes))
    provider = CGDataProviderCreateWithCFData(payload)
    image = CGImageCreate(
        alpha.shape[1],
        alpha.shape[0],
        8,
        32,
        alpha.shape[1] * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast,
        provider,
        None,
        False,
        kCGRenderingIntentDefault,
    )
    return image, payload


def _distance_field_alphas_for_specs(
    geometry: dict,
    specs: list[dict],
    *,
    signed_distance=None,
) -> list[dict]:
    import numpy as np

    if signed_distance is None:
        signed_distance = _display_signed_distance_field(
            geometry, soften_notch_shoulders=True
        )
    notch_top_corner_mask = _notch_top_corner_glow_mask(geometry)
    notch_corner_lift = None
    helper_signed_distance = None
    screen_corner_lift = None
    screen_corner_helper_signed_distance = None
    if notch_top_corner_mask is not None and any(
        spec.get("fill_role") is not None for spec in specs
    ):
        helper_corner_mask = _notch_top_corner_glow_mask(
            geometry,
            x_band=_NOTCH_TOP_CORNER_HELPER_X_BAND,
            y_band=_NOTCH_TOP_CORNER_HELPER_Y_BAND,
            attenuation=_NOTCH_TOP_CORNER_HELPER_ATTENUATION,
        )
        helper_signed_distance = _legacy_notch_corner_signed_distance_field(geometry)
        notch_corner_lift = (1.0 - helper_corner_mask).astype(
            np.float32, copy=False
        )
    screen_corner_lift = _screen_corner_glow_lift_mask(geometry)
    screen_corner_helper_signed_distance = _soft_display_corner_signed_distance_field(
        geometry
    )

    alphas = []
    for spec in specs:
        alpha = _distance_field_alpha(
            signed_distance,
            spec["falloff"] * geometry["scale"],
            spec["power"],
        )
        if spec.get("fill_role") is not None and notch_top_corner_mask is not None:
            alpha = (alpha * notch_top_corner_mask).astype(np.float32, copy=False)
            if helper_signed_distance is not None and notch_corner_lift is not None:
                helper_alpha = _distance_field_alpha(
                    helper_signed_distance,
                    spec["falloff"]
                    * geometry["scale"]
                    * _NOTCH_TOP_CORNER_HELPER_FALLOFF_SCALE,
                    spec["power"],
                )
                alpha = (
                    (alpha * (1.0 - notch_corner_lift))
                    + (helper_alpha * notch_corner_lift)
                ).astype(np.float32, copy=False)
        if (
            screen_corner_helper_signed_distance is not None
            and screen_corner_lift is not None
        ):
            screen_corner_helper_alpha = _distance_field_alpha(
                screen_corner_helper_signed_distance,
                spec["falloff"]
                * geometry["scale"]
                * _SCREEN_CORNER_HELPER_FALLOFF_SCALE,
                spec["power"],
            )
            alpha = (
                (alpha * (1.0 - screen_corner_lift))
                + (screen_corner_helper_alpha * screen_corner_lift)
            ).astype(np.float32, copy=False)
        alphas.append({"alpha": alpha, "spec": spec})
    return alphas


def _distance_field_masks_for_specs(geometry: dict, specs: list[dict]) -> list[dict]:
    masks = []
    for entry in _distance_field_alphas_for_specs(geometry, specs):
        image, payload = _alpha_field_to_image(entry["alpha"])
        masks.append({"image": image, "payload": payload, "spec": entry["spec"]})
    return masks


def _continuous_glow_pass_specs():
    """Procedural additive passes driven from one shared distance field."""
    return [
        {
            "name": "core",
            "path_kind": "distance_field",
            "falloff": 3.2,
            "power": 2.7,
            "fill_role": "inner",
            "fill_alpha": 0.28,
        },
        {
            "name": "tight_bloom",
            "path_kind": "distance_field",
            "falloff": 7.2,
            "power": 3.2,
            "fill_role": "middle",
            "fill_alpha": 0.18,
        },
        {
            "name": "wide_bloom",
            "path_kind": "distance_field",
            "falloff": 15.0,
            "power": 3.7,
            "fill_role": "outer",
            "fill_alpha": 0.12,
        },
    ]


def _continuous_dimmer_pass_specs():
    """Masked hold-space dimmer, tuned as a broad low-intensity veil."""
    soft_bloom = next(
        spec for spec in _continuous_glow_pass_specs() if spec["name"] == "wide_bloom"
    )
    return [
        {
            "name": "hold_dimmer",
            "path_kind": "distance_field",
            "falloff": soft_bloom["falloff"] * 16.0,
            "power": 1.15,
            "alpha": min(soft_bloom["fill_alpha"] * _DIM_SDF_BLOOM_MULTIPLIER, 0.95),
        }
    ]


def _continuous_vignette_pass_specs():
    """Procedural subtractive passes driven from the same distance field."""
    return [
        {
            "name": "core",
            "path_kind": "distance_field",
            "falloff": 2.5,
            "power": 2.4,       # relaxed from 3.5 — softer edge
            "alpha": 0.65,      # eased from 0.88
            "color_scale": 0.08,
        },
        {
            "name": "mid",
            "path_kind": "distance_field",
            "falloff": 6.0,
            "power": 2.6,       # relaxed from 3.5
            "alpha": 0.35,      # eased from 0.52
            "color_scale": 0.10,
        },
        {
            "name": "tail",
            "path_kind": "distance_field",
            "falloff": 12.0,
            "power": 3.0,       # relaxed from 3.8
            "alpha": 0.28,      # eased from 0.45
            "color_scale": 0.12,
        },
    ]

def _glow_role_colors(base_color: tuple[float, float, float]) -> dict[str, NSColor]:
    """Build additive glow colors keyed by intensity role."""
    inner_rgb, middle_rgb, outer_rgb = _edge_band_colors(base_color)
    return {
        "inner": NSColor.colorWithSRGBRed_green_blue_alpha_(
            inner_rgb[0], inner_rgb[1], inner_rgb[2], 1.0
        ),
        "middle": NSColor.colorWithSRGBRed_green_blue_alpha_(
            middle_rgb[0], middle_rgb[1], middle_rgb[2], 1.0
        ),
        "outer": NSColor.colorWithSRGBRed_green_blue_alpha_(
            outer_rgb[0], outer_rgb[1], outer_rgb[2], 1.0
        ),
    }


def _vignette_pass_color(base_color: tuple[float, float, float], spec: dict) -> NSColor:
    """Build a tinted subtractive vignette color for a single pass."""
    r, g, b = base_color
    scale = spec["color_scale"]
    return NSColor.colorWithSRGBRed_green_blue_alpha_(
        r * scale,
        g * scale,
        b * scale,
        spec["alpha"],
    )


class GlowOverlay(NSObject):
    """Manages a screen-border glow window driven by audio amplitude."""

    def initWithScreen_(
        self,
        screen: NSScreen | None = None,
        metrics: OpticalShellMetrics | None = None,
    ):
        self = objc.super(GlowOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._metrics = metrics
        self._window: NSWindow | None = None
        self._glow_layer: CALayer | None = None
        self._smoothed_amplitude = 0.0
        self._visible = False
        self._fade_in_until = 0.0
        self._update_count = 0
        self._noise_floor = 0.0  # adaptive ambient noise level
        self._cap_factor = 1.0  # 1.0 = no cap, ramps down toward 0.25 near recording limit
        self._hide_timer = None
        self._hide_generation = 0
        self._glow_color = _GLOW_COLOR
        self._glow_base_opacity = _GLOW_BASE_OPACITY
        self._glow_peak_target = _GLOW_PEAK_TARGET
        self._brightness_timer = None
        self._brightness = 0.5
        return self

    def _cancel_pending_hide(self) -> None:
        if self._hide_timer is not None:
            self._hide_timer.invalidate()
            self._hide_timer = None

    def setup(self) -> None:
        """Create the overlay window and glow layer."""
        frame = self._screen.frame()

        # Borderless, transparent, non-activating window
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, 2, False  # NSWindowStyleMaskBorderless, NSBackingStoreBuffered
        )
        self._window.setLevel_(25)  # NSStatusWindowLevel + 1
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Content view must be layer-backed for Core Animation
        content = self._window.contentView()
        content.setWantsLayer_(True)

        w, h = frame.size.width, frame.size.height
        mask_scale = _screen_backing_scale(self._screen)
        geometry = _display_shape_geometry(self._screen, w, h, mask_scale)
        geometry["scale"] = mask_scale

        glow_colors = _glow_role_colors(_GLOW_COLOR)
        self._mask_payloads = []

        # ── Optional screen dim: SDF-shaped dark backdrop for glow contrast ──
        self._dim_layer = None
        self._dim_pass_layers = []
        if _DIM_SCREEN:
            self._dim_layer = CALayer.alloc().init()
            self._dim_layer.setFrame_(((0, 0), (w, h)))
            self._dim_layer.setOpacity_(0.0)
            dim_pass_layers = []
            for entry in _distance_field_masks_for_specs(geometry, _continuous_dimmer_pass_specs()):
                spec = entry["spec"]
                layer = CALayer.alloc().init()
                layer.setFrame_(((0, 0), (w, h)))
                mask_layer = CALayer.alloc().init()
                mask_layer.setFrame_(((0, 0), (w, h)))
                mask_layer.setContents_(entry["image"])
                mask_layer.setContentsScale_(mask_scale)
                layer.setMask_(mask_layer)
                layer.setBackgroundColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0, 0, 0, spec["alpha"]
                    ).CGColor()
                )
                self._dim_layer.addSublayer_(layer)
                self._mask_payloads.append(entry["payload"])
                dim_pass_layers.append({"layer": layer, "spec": spec})
            self._dim_pass_layers = dim_pass_layers
            content.layer().addSublayer_(self._dim_layer)

        # ── Container layer: holds shadow + masked fill ──────────
        # We control opacity on this layer to drive the whole effect.
        self._glow_layer = CALayer.alloc().init()
        self._glow_layer.setFrame_(((0, 0), (w, h)))
        self._glow_layer.setOpacity_(0.0)

        # Procedural additive passes: one continuous field, different falloff curves.
        glow_pass_layers = []
        for entry in _distance_field_masks_for_specs(geometry, _continuous_glow_pass_specs()):
            spec = entry["spec"]
            layer = CALayer.alloc().init()
            layer.setFrame_(((0, 0), (w, h)))
            mask_layer = CALayer.alloc().init()
            mask_layer.setFrame_(((0, 0), (w, h)))
            mask_layer.setContents_(entry["image"])
            mask_layer.setContentsScale_(mask_scale)
            layer.setMask_(mask_layer)
            self._glow_layer.addSublayer_(layer)
            self._mask_payloads.append(entry["payload"])
            glow_pass_layers.append({"layer": layer, "spec": spec})

        self._glow_pass_layers = glow_pass_layers
        self._shadow_shape = glow_pass_layers[-1]["layer"] if glow_pass_layers else None

        # ── Subtractive vignette: darkened colored edges for light backgrounds ──
        # Same distance-field geometry as the additive glow, with tinted falloff.
        self._vignette_layer = CALayer.alloc().init()
        self._vignette_layer.setFrame_(((0, 0), (w, h)))
        self._vignette_layer.setOpacity_(0.0)

        vignette_pass_layers = []
        for entry in _distance_field_masks_for_specs(geometry, _continuous_vignette_pass_specs()):
            spec = entry["spec"]
            layer = CALayer.alloc().init()
            layer.setFrame_(((0, 0), (w, h)))
            mask_layer = CALayer.alloc().init()
            mask_layer.setFrame_(((0, 0), (w, h)))
            mask_layer.setContents_(entry["image"])
            mask_layer.setContentsScale_(mask_scale)
            layer.setMask_(mask_layer)
            self._vignette_layer.addSublayer_(layer)
            self._mask_payloads.append(entry["payload"])
            vignette_pass_layers.append({"layer": layer, "spec": spec})

        self._vignette_pass_layers = vignette_pass_layers
        self._apply_glow_color(_GLOW_COLOR)
        content.layer().addSublayer_(self._glow_layer)
        content.layer().addSublayer_(self._vignette_layer)
        logger.info("Glow overlay created (%.0fx%.0f, border=%.0f, shadow=%.0f)",
                     w, h, _GLOW_WIDTH, _GLOW_SHADOW_RADIUS)

    def _apply_glow_color(self, base_color: tuple[float, float, float]) -> None:
        """Push the current glow color through the procedural glow/vignette passes."""
        glow_colors = _glow_role_colors(base_color)
        if hasattr(self, "_glow_pass_layers"):
            for entry in self._glow_pass_layers:
                layer = entry["layer"]
                spec = entry["spec"]
                fill_color = glow_colors[spec["fill_role"]]
                layer.setBackgroundColor_(
                    fill_color.colorWithAlphaComponent_(spec["fill_alpha"]).CGColor()
                )

        if hasattr(self, "_vignette_pass_layers"):
            for entry in self._vignette_pass_layers:
                layer = entry["layer"]
                spec = entry["spec"]
                layer.setBackgroundColor_(_vignette_pass_color(base_color, spec).CGColor())

    def show(self) -> None:
        """Fade the glow window in to base opacity."""
        if self._window is None:
            return
        self._cancel_pending_hide()
        self._hide_generation += 1
        self._visible = True
        self._smoothed_amplitude = 0.0
        self._update_count = 0
        self._noise_floor = 0.0
        self._cap_factor = 1.0
        sample_start = time.perf_counter()
        brightness = _sample_screen_brightness(self._screen)
        metrics = getattr(self, "_metrics", None)
        if metrics is not None:
            sample_end = time.perf_counter()
            metrics.record_brightness_sample(
                elapsed_ms=(sample_end - sample_start) * 1000.0,
                now=sample_end,
            )
        self._glow_color, self._glow_base_opacity, self._glow_peak_target = _glow_style_for_brightness(brightness)
        self._apply_glow_color(self._glow_color)
        self._brightness = brightness
        # Cross-fade: additive glow fades out, subtractive vignette fades in
        # as brightness increases. Fully additive at black, fully subtractive
        # at white, blended in between.
        self._additive_mix, self._subtractive_mix = _edge_mix_for_brightness(brightness)
        self._fade_in_until = time.monotonic() + 0.2  # let fade-in finish undisturbed
        self._window.orderFrontRegardless()

        pres = self._glow_layer.presentationLayer()
        current_opacity = pres.opacity() if pres is not None else self._glow_layer.opacity()
        self._glow_layer.removeAllAnimations()
        anim = CABasicAnimation.animationWithKeyPath_("opacity")
        anim.setFromValue_(current_opacity)
        anim.setToValue_(self._glow_base_opacity)
        anim.setDuration_(_GLOW_SHOW_FADE_S)
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_(_GLOW_SHOW_TIMING)
        )
        self._glow_layer.setOpacity_(self._glow_base_opacity)
        self._glow_layer.addAnimation_forKey_(anim, "fadeIn")

        # Fade in screen dim — adaptive opacity based on screen brightness.
        # Sample once per recording, not per-frame.
        if self._dim_layer is not None:
            dim_target = _dim_target_for_brightness(brightness)
            logger.info("Screen brightness=%.2f → dim opacity=%.2f", brightness, dim_target)

            pres = self._dim_layer.presentationLayer()
            current_opacity = pres.opacity() if pres is not None else self._dim_layer.opacity()
            self._dim_layer.removeAllAnimations()
            self._dim_layer.setOpacity_(dim_target)
            dim_anim = CABasicAnimation.animationWithKeyPath_("opacity")
            dim_anim.setFromValue_(current_opacity)
            dim_anim.setToValue_(dim_target)
            dim_anim.setDuration_(_DIM_SHOW_FADE_S)
            dim_anim.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_("easeIn")
            )
            self._dim_layer.addAnimation_forKey_(dim_anim, "dimIn")

        # Start recurring brightness sampling
        old_timer = getattr(self, "_brightness_timer", None)
        if old_timer is not None:
            old_timer.invalidate()
        from Foundation import NSTimer
        self._brightness_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _BRIGHTNESS_SAMPLE_INTERVAL, self, "brightnessResample:", None, True
        )

        logger.info("Glow show")

    def brightnessResample_(self, timer) -> None:
        """Recurring timer: re-sample screen brightness and adapt glow/dim."""
        if not self._visible:
            return
        sample_start = time.perf_counter()
        new_brightness = _sample_screen_brightness(self._screen)
        metrics = getattr(self, "_metrics", None)
        if metrics is not None:
            sample_end = time.perf_counter()
            metrics.record_brightness_sample(
                elapsed_ms=(sample_end - sample_start) * 1000.0,
                now=sample_end,
            )
        current_brightness = getattr(self, "_brightness", 0.0)
        if abs(new_brightness - current_brightness) < 0.02:
            return  # no meaningful change
        self._brightness = new_brightness
        new_color, new_base, new_peak = _glow_style_for_brightness(new_brightness)
        self._glow_color = new_color
        self._glow_base_opacity = new_base
        self._glow_peak_target = new_peak
        self._apply_glow_color(new_color)

        # Update cross-fade mix
        self._additive_mix, self._subtractive_mix = _edge_mix_for_brightness(new_brightness)

        # Smoothly adjust dim opacity
        if self._dim_layer is not None:
            dim_target = _dim_target_for_brightness(new_brightness)
            from Quartz import CABasicAnimation, CAMediaTimingFunction
            pres = self._dim_layer.presentationLayer()
            current = pres.opacity() if pres is not None else self._dim_layer.opacity()
            self._dim_layer.removeAllAnimations()
            self._dim_layer.setOpacity_(dim_target)
            anim = CABasicAnimation.animationWithKeyPath_("opacity")
            anim.setFromValue_(current)
            anim.setToValue_(dim_target)
            anim.setDuration_(0.5)
            anim.setTimingFunction_(CAMediaTimingFunction.functionWithName_("easeInEaseOut"))
            self._dim_layer.addAnimation_forKey_(anim, "dimAdapt")

    def hide(self) -> None:
        """Fade the glow window out smoothly."""
        if self._window is None:
            return
        self._cancel_pending_hide()
        bt = getattr(self, "_brightness_timer", None)
        if bt is not None:
            bt.invalidate()
            self._brightness_timer = None
        self._visible = False
        self._hide_generation += 1
        hide_generation = self._hide_generation
        logger.info("Glow hide (received %d amplitude updates)", self._update_count)

        try:
            pres = self._glow_layer.presentationLayer()
            current = pres.opacity() if pres is not None else self._glow_layer.opacity()
            anim = CABasicAnimation.animationWithKeyPath_("opacity")
            anim.setFromValue_(current)
            anim.setToValue_(0.0)
            anim.setDuration_(_GLOW_HIDE_FADE_S)
            anim.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_("easeIn")
            )
            self._glow_layer.setOpacity_(0.0)
            self._glow_layer.addAnimation_forKey_(anim, "fadeOut")

            # Fade out subtractive vignette in sync
            if hasattr(self, "_vignette_layer") and self._vignette_layer is not None:
                vpres = self._vignette_layer.presentationLayer()
                vcurrent = vpres.opacity() if vpres is not None else self._vignette_layer.opacity()
                vanim = CABasicAnimation.animationWithKeyPath_("opacity")
                vanim.setFromValue_(vcurrent)
                vanim.setToValue_(0.0)
                vanim.setDuration_(_GLOW_HIDE_FADE_S)
                vanim.setTimingFunction_(
                    CAMediaTimingFunction.functionWithName_("easeIn")
                )
                self._vignette_layer.setOpacity_(0.0)
                self._vignette_layer.addAnimation_forKey_(vanim, "vignetteOut")

            # Fade out screen dim — slow and sneaky so the brightness
            # return is imperceptible while attention is on injected text
            if self._dim_layer is not None:
                pres = self._dim_layer.presentationLayer()
                current_opacity = pres.opacity() if pres is not None else self._dim_layer.opacity()
                self._dim_layer.removeAllAnimations()
                self._dim_layer.setOpacity_(0.0)
                dim_anim = CABasicAnimation.animationWithKeyPath_("opacity")
                dim_anim.setFromValue_(current_opacity)
                dim_anim.setToValue_(0.0)
                dim_anim.setDuration_(_DIM_HIDE_FADE_S)
                dim_anim.setTimingFunction_(
                    CAMediaTimingFunction.functionWithName_("easeIn")
                )
                self._dim_layer.addAnimation_forKey_(dim_anim, "dimOut")

            # Order out after animation completes
            hide_delay = _GLOW_HIDE_FADE_S + _WINDOW_TEARDOWN_CUSHION_S
            if self._dim_layer is not None:
                hide_delay = max(hide_delay, _DIM_HIDE_FADE_S + _WINDOW_TEARDOWN_CUSHION_S)
            self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                hide_delay, self, "hideWindowAfterFade:", hide_generation, False
            )
        except Exception:
            # If animation fails, just snap off
            logger.exception("Fade-out animation failed, snapping off")
            self._glow_layer.setOpacity_(0.0)
            self._window.orderOut_(None)

    def hideWindowAfterFade_(self, timer) -> None:
        """Remove window after fade-out animation completes."""
        timer_generation = timer.userInfo() if timer is not None else None
        if timer_generation != self._hide_generation:
            return
        self._hide_timer = None
        if not self._visible and self._window is not None:
            self._window.orderOut_(None)

    def update_amplitude(self, rms: float) -> None:
        """Update glow intensity from an RMS amplitude value (0.0–1.0).

        Must be called on the main thread.
        """
        if not self._visible or self._glow_layer is None:
            return

        # Don't override the fade-in animation
        if time.monotonic() < self._fade_in_until:
            return

        if _GLOW_TEST_RMS is not None:
            try:
                rms = min(max(float(_GLOW_TEST_RMS), 0.0), 1.0)
            except ValueError:
                pass

        self._update_count += 1

        # Adaptive noise floor — slowly tracks ambient noise level.
        # Rises slowly (adapts to fan noise), falls slowly (doesn't
        # drop to zero between words).
        self._noise_floor, signal = _update_noise_floor(self._noise_floor, rms)

        # Smooth: rise fast, decay slow
        if signal > self._smoothed_amplitude:
            self._smoothed_amplitude += (signal - self._smoothed_amplitude) * _RISE_FACTOR
        else:
            self._smoothed_amplitude *= _DECAY_FACTOR

        # Map smoothed amplitude to opacity range [base, max]
        # Fixed multiplier — ceiling is absolute, floor is adaptive
        amplitude_linear = min(self._smoothed_amplitude * _GLOW_MULTIPLIER, 1.0)
        # Perceptual correction: log curve so glow tracks perceived loudness.
        # All smoothing math above stays linear; this is the last step
        # before "rendering" — the display gamma, essentially.
        amplitude_opacity = math.log1p(amplitude_linear * 20.0) / math.log1p(20.0)
        opacity = self._glow_base_opacity + amplitude_opacity * (_GLOW_MAX_OPACITY - self._glow_base_opacity)
        opacity = min(opacity, self._glow_peak_target)

        # Apply recording-cap countdown: shift color from turquoise to amber
        # as the cap approaches — passive visual warning visible at any opacity.
        if self._cap_factor < 1.0:
            cap_floor = 0.25
            scale = cap_floor + (1.0 - cap_floor) * self._cap_factor
            opacity *= scale
            t = 1.0 - self._cap_factor  # 0→1 as cap approaches
            r = self._glow_color[0] + t * (_GLOW_CAP_COLOR[0] - self._glow_color[0])
            g = self._glow_color[1] + t * (_GLOW_CAP_COLOR[1] - self._glow_color[1])
            b = self._glow_color[2] + t * (_GLOW_CAP_COLOR[2] - self._glow_color[2])
            self._apply_glow_color((r, g, b))

        # Cross-fade additive glow and subtractive vignette
        additive_mix = getattr(self, "_additive_mix", 1.0)
        subtractive_mix = getattr(self, "_subtractive_mix", 0.0)
        self._glow_layer.setOpacity_(opacity * additive_mix)
        if hasattr(self, "_vignette_layer") and self._vignette_layer is not None:
            self._vignette_layer.setOpacity_(opacity * subtractive_mix * _VIGNETTE_OPACITY_SCALE)

        # Log first few updates and then periodically to verify pipeline
        if self._update_count <= 3 or self._update_count % 50 == 0:
            logger.info("Glow amplitude: rms=%.4f smoothed=%.4f opacity=%.3f add=%.2f sub=%.2f (update #%d)",
                        rms, self._smoothed_amplitude, opacity, additive_mix, subtractive_mix, self._update_count)
