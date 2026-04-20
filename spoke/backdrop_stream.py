"""Shared backdrop renderer helpers for snapshot and stream-backed overlays."""

from __future__ import annotations

import ctypes
import logging
import math
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
_OPTICAL_SHELL_CORNER_RADIUS_INFLATION = 0.95
_OPTICAL_SHELL_NORMAL_EPS_MULTIPLIER = 0.22

# ---------------------------------------------------------------------------
# Warp kernel tuning constants
# ---------------------------------------------------------------------------
# Each is substituted into the GLSL source at compile time.  Changing a
# constant here changes the kernel; the app must be relaunched to pick
# it up.  Constants that appear in more than one expression share a
# single Python name so they stay in sync.

# How far past the capsule boundary (as a fraction of capsuleRadius) the
# warp bleeds before fading to identity via smoothstep.
# Larger → more squoot visible outside the pill.
_WARP_BLEED_ZONE_FRAC = 2.0

# Floor of field01 at the deepest interior.  Sets the minimum scale
# factor.  Lower → more compression at center.  Below ~0.5 the scale
# can invert (sourceField01 / field01 > 1).
_WARP_CENTER_FLOOR = 0.80

# Exponent applied to rawField before mixing with the center floor.
# Controls how the field distributes between rim and center.
# Lower → field drops faster toward center (wider "deep" zone).
# Higher → field stays near 1.0 longer, drops only near the center.
_WARP_FIELD_EXPONENT = 0.35

# depthRemap base exponent: how aggressively the center gets evacuated.
# curveBoost scales this down; _FLOOR is the hard minimum.
_WARP_REMAP_BASE_EXP_SCALE = 0.98
_WARP_REMAP_BASE_EXP_FLOOR = 0.02

# depthRemap rim exponent: how aggressive the remap is near the rim.
# Lower → remap kicks in closer to the boundary, pushing content that's
# only slightly off the midline harder toward the edge.
_WARP_REMAP_RIM_EXP = 0.1

# curveBoost derivation from coreMagnification and ringAmplitudePoints.
# curveBoost = min(CAP, mag_term + ring_term)
_WARP_CURVEBOOST_CAP = 0.95
_WARP_CURVEBOOST_MAG_SCALE = 0.35       # (coreMag - 1) * this
_WARP_CURVEBOOST_RING_DIVISOR = 240.0   # ringAmplitude / this
_WARP_CURVEBOOST_RING_CAP = 0.55        # ring term capped here

# Spine proximity boost: pixels near the horizontal center of the capsule
# body need more aggressive scaling to reach the rim and squoot, because
# they have more capsule to cross.  This multiplier scales the warp
# strength based on how far the pixel is from the nearest endcap.
# 0.0 = no boost (uniform warp everywhere).
# Higher = more violence at x-center relative to tips.
_WARP_SPINE_PROXIMITY_BOOST = 1.5

# Anisotropic scale ratio: how much harder x compresses relative to y.
# 1.0 = uniform (isotropic).  Values > 1 make x compress faster, pulling
# content toward the endcaps while preserving vertical extent so it
# curves around the pill instead of collapsing to a V on the midline.
_WARP_X_SQUEEZE = 2.5

# Same idea for y: how much harder y compresses relative to the base scale.
# 1.0 = same as base scale.  Values > 1 pull content toward top/bottom
# of the pill more aggressively.  Keep milder than x-squeeze.
_WARP_Y_SQUEEZE = 1.5

# Exterior magnification: a gentle inward pull outside the capsule that
# creates a lens/magnification effect around the boundary.  The pull
# decays exponentially with distance from the capsule surface.
# Strength is fraction of capsuleRadius; higher = stronger lens.
_WARP_EXTERIOR_MAG_STRENGTH = 3.0   # cranked for debugging
_WARP_EXTERIOR_MAG_DECAY = 0.5     # very slow falloff so it's visible far out

_SHELL_WARP_KERNEL = None

def _build_shell_warp_kernel_source() -> str:
    return """
float sdCapsule(vec2 p, float spineHalf, float radius) {
    p.x = abs(p.x);
    float spine_dist = max(p.x - spineHalf, 0.0);
    return length(vec2(spine_dist, p.y)) - radius;
}

vec2 capsuleGradient(vec2 p, float spineHalf) {
    float clampedX = clamp(p.x, -spineHalf, spineHalf);
    vec2 toP = p - vec2(clampedX, 0.0);
    float len = length(toP);
    return len > 1e-6 ? toP / len : vec2(0.0, 1.0);
}

float depthRemap(float inside01, float curveBoost) {
    float x = clamp(inside01, 0.0, 1.0);
    float baseExp = max(1.0 - curveBoost * %(remap_base_scale)s, %(remap_base_floor)s);
    float rimExp = mix(%(remap_rim_exp)s, 1.0, 1.0 - curveBoost);
    float exponent = mix(baseExp, rimExp, x * x);
    return pow(x, exponent);
}

kernel vec2 opticalShellWarp(
    float width,
    float height,
    float rectWidth,
    float rectHeight,
    float cornerRadius,
    float coreMagnification,
    float bandWidth,
    float tailWidth,
    float ringAmplitudePoints,
    float tailAmplitudePoints
) {
    vec2 d = destCoord();
    vec2 c = vec2(width * 0.5, height * 0.5);
    vec2 p = d - c;
    vec2 halfRect = vec2(rectWidth * 0.5, rectHeight * 0.5);
    float capsuleRadius = max(halfRect.y, 1.0);
    float spineHalf = max(halfRect.x - capsuleRadius, 0.0);

    float capsuleSdf = sdCapsule(p, spineHalf, capsuleRadius);

    float bleedZone = capsuleRadius * %(bleed_frac)s;
    if (capsuleSdf > bleedZone) return d;

    float curveBoost = min(
        %(cb_cap)s,
        max(0.0, (coreMagnification - 1.0) * %(cb_mag_scale)s)
            + min(ringAmplitudePoints / %(cb_ring_div)s, %(cb_ring_cap)s)
    );

    // Spine proximity: how far is this pixel from the nearest endcap?
    // 0 at the tips, 1 at the horizontal center of the body.
    float distFromTip = max(spineHalf - abs(p.x), 0.0);
    float spineProximity = spineHalf > 0.0 ? distFromTip / spineHalf : 0.0;

    float rawField = clamp(1.0 + capsuleSdf / capsuleRadius, 0.0, 1.0);
    // Lower the center floor for pixels near x-center so the scale
    // gets more aggressive where content has further to travel.
    float localFloor = %(center_floor)s
        / (1.0 + spineProximity * %(spine_boost)s);
    float field01 = mix(localFloor, 1.0, pow(rawField, %(field_exp)s));
    float sourceField01 = 1.0 - depthRemap(1.0 - field01, curveBoost);
    float scale = sourceField01 / field01;

    // Anisotropic scale: compress x harder than y so content reaches
    // the endcaps and curves around instead of collapsing to a V.
    float scaleX = pow(max(scale, 0.0), %(x_squeeze)s);
    float scaleY = pow(max(scale, 0.0), %(y_squeeze)s);
    vec2 warped = c + p * vec2(scaleX, scaleY);

    // Exterior: interior warp fades to magnified exterior.
    // magRampIn uses a short ramp (capsuleRadius * 0.15) to avoid
    // a discontinuity at sdf=0 while reaching full strength quickly.
    float exteriorT = max(capsuleSdf, 0.0);
    float magRampIn = smoothstep(0.0, capsuleRadius * 0.15, exteriorT);
    float magDecay = exp(-exteriorT / capsuleRadius * %(ext_mag_decay)s);
    vec2 n = capsuleGradient(p, spineHalf);
    float mag = %(ext_mag_strength)s * capsuleRadius * magRampIn * magDecay;
    vec2 magSrc = d - n * mag;
    magSrc = clamp(magSrc, vec2(0.0, 0.0), vec2(width, height));
    // Blend: interior warp → magnified exterior over the bleed zone.
    float warpFade = smoothstep(0.0, bleedZone, exteriorT);
    return mix(warped, magSrc, warpFade);
}
""" % {
        "bleed_frac": _WARP_BLEED_ZONE_FRAC,
        "center_floor": _WARP_CENTER_FLOOR,
        "field_exp": _WARP_FIELD_EXPONENT,
        "remap_base_scale": _WARP_REMAP_BASE_EXP_SCALE,
        "remap_base_floor": _WARP_REMAP_BASE_EXP_FLOOR,
        "remap_rim_exp": _WARP_REMAP_RIM_EXP,
        "cb_cap": _WARP_CURVEBOOST_CAP,
        "cb_mag_scale": _WARP_CURVEBOOST_MAG_SCALE,
        "cb_ring_div": _WARP_CURVEBOOST_RING_DIVISOR,
        "cb_ring_cap": _WARP_CURVEBOOST_RING_CAP,
        "spine_boost": _WARP_SPINE_PROXIMITY_BOOST,
        "x_squeeze": _WARP_X_SQUEEZE,
        "y_squeeze": _WARP_Y_SQUEEZE,
        "ext_mag_strength": _WARP_EXTERIOR_MAG_STRENGTH,
        "ext_mag_decay": _WARP_EXTERIOR_MAG_DECAY,
    }

_SHELL_WARP_KERNEL_SOURCE = _build_shell_warp_kernel_source()

_BRIDGE_STATE: dict[str, object] | None = None
_BRIDGE_LOCK = threading.Lock()
_LIBDISPATCH = None
_LIBDISPATCH_LOCK = threading.Lock()


def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )


def _rounded_rect_sdf(field_width: int, field_height: int, rect_width: float, rect_height: float, corner_radius: float):
    import numpy as np

    pw, ph = max(int(field_width), 1), max(int(field_height), 1)
    rw, rh = float(rect_width), float(rect_height)
    x = np.arange(pw, dtype=np.float32)[None, :] + 0.5
    y = np.arange(ph, dtype=np.float32)[:, None] + 0.5
    cx = x - pw * 0.5
    cy = y - ph * 0.5
    r = float(corner_radius)
    qx = np.abs(cx) - (rw * 0.5 - r)
    qy = np.abs(cy) - (rh * 0.5 - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return (outside + inside - r).astype(np.float32)


def _smoothstep01(value):
    import numpy as np

    clamped = np.clip(value, 0.0, 1.0)
    return (clamped * clamped * (3.0 - 2.0 * clamped)).astype(np.float32)


def _optical_shell_inside_envelope(distance_inside: float, band_width: float) -> float:
    depth = max(float(distance_inside), 0.0)
    falloff = max(float(band_width) * 1.35, 1.0)
    return math.exp(-((depth / falloff) ** 2.0))


def _smoothstep_scalar(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    t = min(max((float(value) - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _optical_shell_center_envelope(
    *,
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
    band_width: float,
) -> float:
    half_width = max(float(content_width) * 0.5 - float(band_width) * 0.2, 1.0)
    half_height = max(float(content_height) * 0.5 - float(band_width) * 0.2, 1.0)
    radius = math.hypot(abs(float(offset_x)) / half_width, abs(float(offset_y)) / half_height)
    return 1.0 - _smoothstep_scalar(0.08, 1.02, radius)


def _optical_shell_interior_flow(center_envelope: float, inside_envelope: float) -> float:
    center = min(max(float(center_envelope), 0.0), 1.0)
    inside = min(max(float(inside_envelope) * 1.35, 0.0), 1.0)
    return min(1.0, center + inside - center * inside * 0.35)


def _optical_shell_core_displacement_envelope(center_envelope: float, inside_envelope: float) -> float:
    center = min(max(float(center_envelope), 0.0), 1.0)
    inside = min(max(float(inside_envelope), 0.0), 1.0)
    return min(inside * (1.0 - center) * 1.25, 1.0)


def _optical_shell_curve_boost(core_magnification: float, ring_amplitude_points: float) -> float:
    return min(
        0.95,
        max(0.0, (float(core_magnification) - 1.0) * 0.35)
        + min(max(float(ring_amplitude_points), 0.0) / 240.0, 0.55),
    )


def _optical_shell_depth_remap(inside01: float, curve_boost: float) -> float:
    inside = min(max(float(inside01), 0.0), 1.0)
    boost = min(max(float(curve_boost), 0.0), 0.95)
    return min(max(inside + boost * inside * inside * (1.0 - inside), 0.0), 1.0)


def _optical_shell_source_depth_points(source01: float, center_depth: float) -> float:
    return min(max(float(source01), 0.0), 1.0) * max(float(center_depth), 0.0)


def _optical_shell_capsule_spine_half_length(content_width: float, content_height: float) -> float:
    return max(float(content_width) * 0.5 - float(content_height) * 0.5, 1.0)


def _optical_shell_capsule_axis_decomposition(
    offset_x: float,
    spine_half: float,
    capsule_radius: float,
) -> tuple[float, float]:
    px = float(offset_x)
    spine_half = max(float(spine_half), 1.0)
    abs_px = abs(px)
    join_sharpness = 0.75
    spine_abs = -math.log(
        math.exp(-join_sharpness * abs_px) + math.exp(-join_sharpness * spine_half)
    ) / join_sharpness
    spine_x = math.copysign(spine_abs, px)
    radial_x = px - spine_x
    return spine_x, radial_x


def _optical_shell_capsule_longitudinal01(
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
) -> float:
    spine_half = _optical_shell_capsule_spine_half_length(content_width, content_height)
    capsule_radius = max(float(content_height) * 0.5, 1.0)
    spine_x, radial_x = _optical_shell_capsule_axis_decomposition(offset_x, spine_half, capsule_radius)
    total_half = spine_half + capsule_radius
    body_longitudinal = min(max(abs(spine_x) / total_half, 0.0), 1.0)
    cap_angle = 1.0 - min(
        max(math.atan2(abs(float(offset_y)), max(abs(radial_x), 1e-4)) / (0.5 * math.pi), 0.0),
        1.0,
    )
    cap_longitudinal = min(max((spine_half + capsule_radius * cap_angle) / total_half, 0.0), 1.0)
    cap_blend = _smoothstep_scalar(
        max(spine_half - capsule_radius * 0.18, 0.0),
        spine_half + capsule_radius * 0.12,
        abs(float(offset_x)),
    )
    return body_longitudinal * (1.0 - cap_blend) + cap_longitudinal * cap_blend


def _optical_shell_capsule_coordinate_fields(
    field_width: int,
    field_height: int,
    content_width: float,
    content_height: float,
):
    import numpy as np

    width = max(int(field_width), 1)
    height = max(int(field_height), 1)
    content_width = max(float(content_width), 1.0)
    content_height = max(float(content_height), 1.0)
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5 - width * 0.5
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5 - height * 0.5
    spine_half = _optical_shell_capsule_spine_half_length(content_width, content_height)
    capsule_radius = max(content_height * 0.5, 1.0)
    abs_x = np.abs(xs)
    join_sharpness = 0.75
    spine_abs = -np.log(np.exp(-join_sharpness * abs_x) + np.exp(-join_sharpness * spine_half)) / join_sharpness
    spine_x = np.sign(xs) * spine_abs
    radial_x = xs - spine_x
    total_half = spine_half + capsule_radius
    body_longitudinal = np.clip(np.abs(spine_x) / total_half, 0.0, 1.0)
    cap_angle = 1.0 - np.clip(
        np.arctan2(np.abs(ys), np.maximum(np.abs(radial_x), 1e-4)) / (0.5 * math.pi),
        0.0,
        1.0,
    )
    cap_longitudinal = np.clip((spine_half + capsule_radius * cap_angle) / total_half, 0.0, 1.0)
    cap_blend_start = max(spine_half - capsule_radius * 0.18, 0.0)
    cap_blend_end = spine_half + capsule_radius * 0.12
    cap_blend = _smoothstep01((abs_x - cap_blend_start) / max(cap_blend_end - cap_blend_start, 1e-3))
    longitudinal = (body_longitudinal * (1.0 - cap_blend) + cap_longitudinal * cap_blend).astype(np.float32)
    radial = np.clip(np.hypot(radial_x, ys) / capsule_radius, 0.0, 1.0).astype(np.float32)
    return longitudinal, radial


def _optical_shell_center_bias_coordinate(coord01: float, curve_boost: float) -> float:
    coord = min(max(float(coord01), 0.0), 1.0)
    return 1.0 - _optical_shell_depth_remap(1.0 - coord, curve_boost)


def _optical_shell_pill_offset_sdf(
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
) -> float:
    half_width = max(float(content_width) * 0.5, 1.0)
    half_height = max(float(content_height) * 0.5, 1.0)
    corner_radius = half_height
    qx = abs(float(offset_x)) - half_width + corner_radius
    qy = abs(float(offset_y)) - half_height + corner_radius
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - corner_radius


def _optical_shell_pill_field01(
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
) -> float:
    capsule_radius = max(float(content_height) * 0.5, 1.0)
    sdf = _optical_shell_pill_offset_sdf(
        offset_x,
        offset_y,
        content_width,
        content_height,
    )
    return min(max(1.0 + sdf / capsule_radius, 0.0), 1.0)


def _optical_shell_debug_field01(
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
    curve_boost: float,
) -> float:
    field01 = _optical_shell_pill_field01(
        offset_x,
        offset_y,
        content_width,
        content_height,
    )
    return _optical_shell_center_bias_coordinate(field01, curve_boost)


def _optical_shell_local_center_depth(
    normal_x: float,
    normal_y: float,
    content_width: float,
    content_height: float,
) -> float:
    half_width = max(float(content_width) * 0.5, 1.0)
    half_height = max(float(content_height) * 0.5, 1.0)
    return max(
        min(
            half_width / max(abs(float(normal_x)), 1e-3),
            half_height / max(abs(float(normal_y)), 1e-3),
        ),
        1.0,
    )


def _optical_shell_inside_depth01_from_sdf(
    sdf: float,
    content_width: float,
    content_height: float,
) -> float:
    center_depth = max(min(float(content_width), float(content_height)) * 0.5, 1.0)
    return min(max(-float(sdf) / center_depth, 0.0), 1.0)


def _optical_shell_gradient_epsilon(band_width: float) -> float:
    return max(1.0, max(float(band_width), 0.0) * _OPTICAL_SHELL_NORMAL_EPS_MULTIPLIER)


def _optical_shell_effective_corner_radius(corner_radius: float, band_width: float) -> float:
    return max(float(corner_radius), 0.0) + max(float(band_width), 0.0) * _OPTICAL_SHELL_CORNER_RADIUS_INFLATION


def _optical_shell_corner_relief(
    *,
    offset_x: float,
    offset_y: float,
    content_width: float,
    content_height: float,
    corner_radius: float,
    band_width: float,
) -> float:
    half_width = float(content_width) * 0.5
    half_height = float(content_height) * 0.5
    inner_width = max(half_width - (float(corner_radius) + float(band_width) * 0.2), 1.0)
    inner_height = max(half_height - (float(corner_radius) + float(band_width) * 0.2), 1.0)
    norm_x = abs(float(offset_x)) / inner_width
    norm_y = abs(float(offset_y)) / inner_height
    cornerness = _smoothstep_scalar(0.38, 0.86, min(norm_x, norm_y))
    return 1.0 - 0.32 * cornerness


def _debug_shell_grid_profile(shell_config: dict) -> dict[str, float | bool]:
    spacing = max(float(shell_config.get("debug_grid_spacing_points", 18.0)), 6.0)
    return {
        "spacing": spacing,
        "field_major_step": 0.125,
        "field_minor_step": 0.0625,
        "field_contour_halfwidth": 0.012,
        "field_minor_contour_halfwidth": 0.006,
        "field_color": (45, 45, 45, 210),
        "field_minor_color": (70, 70, 70, 110),
        "longitudinal_hint_step": 0.125,
        "radial_hint_step": 0.125,
        "hint_contour_halfwidth": 0.004,
        "longitudinal_hint_color": (65, 65, 65, 0),
        "radial_hint_color": (80, 80, 80, 0),
        "ring_color": (90, 90, 90, 0),
        "ring_halfwidth": 0.75,
        "center_marker_shape": "circle",
        "center_marker_width_points": 12.0,
        "center_marker_height_points": 12.0,
        "center_marker_color": (0, 235, 90, 255),
    }


def _shell_warp_kernel():
    global _SHELL_WARP_KERNEL
    if _SHELL_WARP_KERNEL is not None:
        return _SHELL_WARP_KERNEL or None
    try:
        from Quartz import CIWarpKernel
    except Exception:
        _SHELL_WARP_KERNEL = False
        return None
    try:
        _SHELL_WARP_KERNEL = CIWarpKernel.alloc().initWithString_(_SHELL_WARP_KERNEL_SOURCE)
    except Exception:
        logger.debug("Failed to compile optical-shell warp kernel", exc_info=True)
        _SHELL_WARP_KERNEL = False
    return _SHELL_WARP_KERNEL or None


def _debug_shell_grid_ci_image(extent, shell_config):
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
    content_width = min(max(float(shell_config.get("content_width_points", width)), 1.0), float(width))
    content_height = min(max(float(shell_config.get("content_height_points", height)), 1.0), float(height))
    corner_radius = min(
        max(float(shell_config.get("corner_radius_points", 16.0)), 0.0),
        max(min(content_width, content_height) * 0.5 - 1.0, 0.0),
    )
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :] = np.array([210, 255, 240, 255], dtype=np.uint8)

    profile = _debug_shell_grid_profile(shell_config)
    center_x = width * 0.5
    center_y = height * 0.5
    xs = np.arange(width, dtype=np.float32)[None, :] + 0.5 - center_x
    ys = np.arange(height, dtype=np.float32)[:, None] + 0.5 - center_y
    def _contour_mask(field, step, halfwidth):
        normalized = field / max(float(step), 1e-4)
        distance = np.abs(normalized - np.rint(normalized)) * float(step)
        return distance < float(halfwidth)

    capsule_radius = max(content_height * 0.5, 1.0)
    spine_half = max(content_width * 0.5 - capsule_radius, 0.0)

    # Capsule SDF: distance to horizontal line segment minus radius.
    # Iso-contours are pills at every depth.
    spine_dist = np.maximum(np.abs(xs) - spine_half, 0.0)
    capsule_sdf = (np.hypot(spine_dist, ys) - capsule_radius).astype(np.float32)

    ring = np.abs(capsule_sdf) < float(profile["ring_halfwidth"])
    interior = capsule_sdf < 0.0
    curve_boost = _optical_shell_curve_boost(
        float(shell_config.get("core_magnification", 1.0)),
        float(shell_config.get("ring_amplitude_points", 12.0)),
    )
    raw_field01 = np.clip(1.0 + capsule_sdf / capsule_radius, 0.0, 1.0).astype(np.float32)
    field01 = np.clip(
        1.0 - np.clip((1.0 - raw_field01) + curve_boost * (1.0 - raw_field01) * raw_field01, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)
    major_field = _contour_mask(
        field01,
        float(profile["field_major_step"]),
        float(profile["field_contour_halfwidth"]),
    )
    minor_field = _contour_mask(
        field01,
        float(profile["field_minor_step"]),
        float(profile["field_minor_contour_halfwidth"]),
    ) & ~major_field
    rgba[interior & minor_field] = np.array(profile["field_minor_color"], dtype=np.uint8)
    rgba[interior & major_field] = np.array(profile["field_color"], dtype=np.uint8)
    rgba[ring] = np.array(profile["ring_color"], dtype=np.uint8)

    marker_width = float(profile["center_marker_width_points"]) * 0.5
    marker_height = float(profile["center_marker_height_points"]) * 0.5
    center_marker = (
        (xs / max(marker_width, 1.0)) ** 2
        + (ys / max(marker_height, 1.0)) ** 2
    ) <= 1.0
    rgba[center_marker] = np.array(profile["center_marker_color"], dtype=np.uint8)

    payload = NSData.dataWithBytes_length_(rgba.tobytes(), int(rgba.nbytes))
    provider = CGDataProviderCreateWithCFData(payload)
    image = CGImageCreate(
        width,
        height,
        8,
        32,
        width * 4,
        CGColorSpaceCreateDeviceRGB(),
        kCGImageAlphaPremultipliedLast,
        provider,
        None,
        False,
        kCGRenderingIntentDefault,
    )
    ci_image = CIImage.imageWithCGImage_(image)
    return ci_image.imageByCroppingToRect_(extent) if hasattr(ci_image, "imageByCroppingToRect_") else ci_image


def _apply_optical_shell_warp_ci_image(ci_image, extent, shell_config):
    if ci_image is None or extent is None:
        return ci_image
    warp_kernel = _shell_warp_kernel()
    if warp_kernel is None:
        return ci_image
    args = [
        float(extent.size.width),
        float(extent.size.height),
        float(shell_config.get("content_width_points", extent.size.width)),
        float(shell_config.get("content_height_points", extent.size.height)),
        _optical_shell_effective_corner_radius(
            float(shell_config.get("corner_radius_points", 16.0)),
            float(shell_config.get("band_width_points", 12.0)),
        ),
        float(shell_config.get("core_magnification", 1.0)),
        float(shell_config.get("band_width_points", 12.0)),
        float(shell_config.get("tail_width_points", 9.0)),
        float(shell_config.get("ring_amplitude_points", 12.0)),
        float(shell_config.get("tail_amplitude_points", 4.0)),
    ]
    try:
        candidate = warp_kernel.applyWithExtent_roiCallback_inputImage_arguments_(
            extent,
            lambda _index, rect: rect,
            ci_image,
            args,
        )
    except Exception:
        logger.debug("Optical-shell warp kernel application failed", exc_info=True)
        return ci_image
    return candidate if candidate is not None else ci_image


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
    sck_avail = _screen_capture_kit_available()
    sck_disabled = os.environ.get("SPOKE_BACKDROP_DISABLE_SCK", "").strip() not in ("", "0")
    logger.info("make_backdrop_renderer: SCK available=%s disabled=%s", sck_avail, sck_disabled)
    if sck_avail and not sck_disabled:
        try:
            renderer = _ScreenCaptureKitBackdropRenderer(screen, fallback_factory)
            logger.info("Backdrop renderer: ScreenCaptureKit (streaming)")
            return renderer
        except Exception:
            logger.info("Falling back to Quartz after SCK init failure", exc_info=True)
    logger.info("Backdrop renderer: Quartz snapshot fallback (SLOW)")
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
        output = clamped
        if shell_config.get("debug_visualize"):
            candidate = _debug_shell_grid_ci_image(working_extent, shell_config)
            if candidate is not None:
                output = candidate
            cleanup_blur = 0.0
        scaled_shell_config = dict(shell_config)
        scaled_shell_config["content_width_points"] = float(
            shell_config.get("content_width_points", working_extent.size.width)
        ) * _METAL_BLUR_DOWNSAMPLE
        scaled_shell_config["content_height_points"] = float(
            shell_config.get("content_height_points", working_extent.size.height)
        ) * _METAL_BLUR_DOWNSAMPLE
        scaled_shell_config["corner_radius_points"] = float(
            shell_config.get("corner_radius_points", 16.0)
        ) * _METAL_BLUR_DOWNSAMPLE
        scaled_shell_config["band_width_points"] = float(
            shell_config.get("band_width_points", 12.0)
        ) * _METAL_BLUR_DOWNSAMPLE
        scaled_shell_config["tail_width_points"] = float(
            shell_config.get("tail_width_points", 9.0)
        ) * _METAL_BLUR_DOWNSAMPLE
        output = _apply_optical_shell_warp_ci_image(output, working_extent, scaled_shell_config)

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

            # CoreMedia string constant — hardcode rather than loading via
            # objc.loadBundleVariables which doesn't reliably resolve string consts.
            _kCMSampleAttachmentKey_DisplayImmediately = "DisplayImmediately"

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
                "kCMSampleAttachmentKey_DisplayImmediately": _kCMSampleAttachmentKey_DisplayImmediately,
            }
        except Exception:
            logger.info("ScreenCaptureKit bridge load FAILED", exc_info=True)
            _BRIDGE_STATE = None
        if _BRIDGE_STATE is not None:
            logger.info("ScreenCaptureKit bridge loaded OK")
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
            # Apply optical shell warp via CIImage when Metal pipeline is unavailable.
            if optical_shell_config is not None:
                warped = _apply_optical_shell_warp_ci_image(output, extent, optical_shell_config)
                if warped is not None:
                    output = warped
                    if hasattr(output, "imageByCroppingToRect_"):
                        output = output.imageByCroppingToRect_(extent)
            elif self._blur_radius_points > 0.0:
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
        optical_shell_config = getattr(self, "_optical_shell_config", None)
        if optical_shell_config is not None and optical_shell_config.get("debug_visualize"):
            extent = _make_rect(0.0, 0.0, capture_rect.size.width, capture_rect.size.height)
            output = _debug_shell_grid_ci_image(extent, optical_shell_config)
            context = self._context()
            if output is not None and context is not None and hasattr(context, "createCGImage_fromRect_"):
                output = _apply_optical_shell_warp_ci_image(output, extent, optical_shell_config)
                try:
                    image = context.createCGImage_fromRect_(output, extent)
                except Exception:
                    logger.debug("Failed to seed debug shell grid image", exc_info=True)
                    image = None
                if image is not None:
                    with self._lock:
                        self._latest_image = image
                    return image

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
