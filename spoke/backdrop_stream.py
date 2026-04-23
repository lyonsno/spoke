"""Shared backdrop renderer helpers for snapshot and stream-backed overlays."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import threading
import time
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

# How far past the shell boundary (as a fraction of the corner radius) the
# warp bleeds before fading to identity via smoothstep.
# Larger → more squoot visible outside the pill.
_WARP_BLEED_ZONE_FRAC = 0.8

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

# Spine proximity boost: pixels near the horizontal center of the rounded shell
# body need more aggressive scaling to reach the rim and squoot, because
# they have more shell width to cross.  This multiplier scales the warp
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

# Exterior magnification: a gentle inward pull outside the rounded shell that
# creates a lens/magnification effect around the boundary.  The pull
# decays exponentially with distance from the capsule surface.
# Strength is a fraction of the corner radius; higher = stronger lens.
_WARP_EXTERIOR_MAG_STRENGTH = 0.6   # visible lens effect around boundary
_WARP_EXTERIOR_MAG_DECAY = 2.0     # fast falloff to keep it near the boundary

_SHELL_WARP_KERNEL = None


def _create_iosurface(width: int, height: int):
    """Create a reusable IOSurface for GPU-only rendering."""
    try:
        import objc as _objc
        _objc.loadBundle(
            "IOSurface",
            globals(),
            bundle_path="/System/Library/Frameworks/IOSurface.framework",
        )
        IOSurface = _objc.lookUpClass("IOSurface")
        props = {
            "IOSurfaceWidth": width,
            "IOSurfaceHeight": height,
            "IOSurfaceBytesPerElement": 4,
            "IOSurfacePixelFormat": 1111970369,  # kCVPixelFormatType_32BGRA
        }
        surface = IOSurface.alloc().initWithProperties_(props)
        return surface
    except Exception:
        logger.debug("Failed to create IOSurface", exc_info=True)
        return None

# ---------------------------------------------------------------------------
# Frame timing instrumentation
# ---------------------------------------------------------------------------
_FRAME_TIMING_INTERVAL_S = 5.0  # log summary every N seconds
_FRAME_TIMING_ENABLED = os.environ.get(
    "SPOKE_BACKDROP_FRAME_TIMING", ""
).strip() not in ("", "0")


class _FrameTimer:
    """Lightweight per-phase frame timer with periodic log summaries."""

    __slots__ = ("_label", "_phases", "_frame_totals", "_count", "_last_report", "_lock")

    def __init__(self, label: str):
        self._label = label
        self._phases: list[tuple[str, float]] = []  # (name, start_ns)
        self._frame_totals: dict[str, list[float]] = {}
        self._count = 0
        self._last_report = time.monotonic()
        self._lock = threading.Lock()

    def begin(self, phase: str) -> None:
        if not _FRAME_TIMING_ENABLED:
            return
        self._phases.append((phase, time.perf_counter_ns()))

    def end(self, phase: str) -> None:
        if not _FRAME_TIMING_ENABLED:
            return
        end_ns = time.perf_counter_ns()
        for i in range(len(self._phases) - 1, -1, -1):
            if self._phases[i][0] == phase:
                start_ns = self._phases.pop(i)[1]
                ms = (end_ns - start_ns) / 1_000_000
                with self._lock:
                    self._frame_totals.setdefault(phase, []).append(ms)
                return

    def frame_done(self) -> None:
        if not _FRAME_TIMING_ENABLED:
            return
        self._phases.clear()
        with self._lock:
            self._count += 1
            now = time.monotonic()
            elapsed = now - self._last_report
            if elapsed < _FRAME_TIMING_INTERVAL_S:
                return
            self._emit_and_reset(elapsed)

    def _emit_and_reset(self, elapsed: float) -> None:
        count = self._count
        totals = self._frame_totals
        self._frame_totals = {}
        self._count = 0
        self._last_report = time.monotonic()
        if count == 0:
            return
        fps = count / elapsed
        parts = [f"{self._label}: {count} frames in {elapsed:.1f}s ({fps:.1f} fps)"]
        for phase, times in totals.items():
            n = len(times)
            avg = sum(times) / n
            peak = max(times)
            parts.append(f"  {phase}: avg={avg:.2f}ms peak={peak:.2f}ms (n={n})")
        logger.info("\n".join(parts))


# Singleton timers for each pipeline path
_quartz_timer = _FrameTimer("quartz-shell")
_sck_timer = _FrameTimer("sck-shell")

def _build_shell_warp_kernel_source() -> str:
    # CIWarpKernel: returns vec2 coordinate. Geometry is parameterized by
    # ``cornerRadius`` so the optical shell can stay a rounded rectangle; a
    # true capsule is only the half-height special case. Blur is handled as a
    # separate pre-warp Gaussian pass — avoids CISampler buffer retention that
    # causes SCK stream stalls.
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

    float distFromTip = max(spineHalf - abs(p.x), 0.0);
    float spineProximity = spineHalf > 0.0 ? distFromTip / spineHalf : 0.0;

    float rawField = clamp(1.0 + capsuleSdf / capsuleRadius, 0.0, 1.0);
    float localFloor = %(center_floor)s
        / (1.0 + spineProximity * %(spine_boost)s);
    float field01 = mix(localFloor, 1.0, pow(rawField, %(field_exp)s));
    float sourceField01 = 1.0 - depthRemap(1.0 - field01, curveBoost);
    float scale = sourceField01 / field01;

    float scaleX = pow(max(scale, 0.0), %(x_squeeze)s);
    float scaleY = pow(max(scale, 0.0), %(y_squeeze)s);
    vec2 warped = c + p * vec2(scaleX, scaleY);

    float exteriorT = max(capsuleSdf, 0.0);
    float seamRamp = smoothstep(0.0, 2.0, exteriorT);
    float magDecay = exp(-exteriorT / capsuleRadius * %(ext_mag_decay)s);
    vec2 n = capsuleGradient(p, spineHalf);
    float tipDist = max(abs(p.x) - spineHalf, 0.0);
    float tipAtten = 1.0 - smoothstep(0.0, capsuleRadius * 0.8, tipDist);
    float mag = %(ext_mag_strength)s * capsuleRadius * seamRamp * magDecay * tipAtten;
    vec2 result = warped - n * mag;
    result = clamp(result, vec2(0.0, 0.0), vec2(width, height));
    return result;
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
        logger.debug("Failed to compile optical-shell general kernel", exc_info=True)
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


_geom_diag_count = 0


def _configure_stream_geometry(config, *, content_rect, capture_rect, point_pixel_scale: float) -> None:
    global _geom_diag_count
    local_rect = _content_local_capture_rect(content_rect, capture_rect)
    scale = max(point_pixel_scale, 1.0)
    pixel_width = max(1, int(round(capture_rect.size.width * scale)))
    pixel_height = max(1, int(round(capture_rect.size.height * scale)))
    _geom_diag_count += 1
    if _geom_diag_count <= 3:
        logger.info(
            "SCK geom: content=(%s,%s %sx%s) capture=(%s,%s %sx%s) local=(%s,%s %sx%s) scale=%s px=%sx%s",
            content_rect.origin.x, content_rect.origin.y,
            content_rect.size.width, content_rect.size.height,
            capture_rect.origin.x, capture_rect.origin.y,
            capture_rect.size.width, capture_rect.size.height,
            local_rect.origin.x, local_rect.origin.y,
            local_rect.size.width, local_rect.size.height,
            scale, pixel_width, pixel_height,
        )

    config.setWidth_(pixel_width)
    config.setHeight_(pixel_height)
    if hasattr(config, "setQueueDepth_"):
        config.setQueueDepth_(8)
    if hasattr(config, "setShowsCursor_"):
        config.setShowsCursor_(False)
    if hasattr(config, "setContentScale_"):
        config.setContentScale_(scale)
    if hasattr(config, "setMinimumFrameInterval_"):
        config.setMinimumFrameInterval_(_FRAME_INTERVAL_60_FPS)
    if hasattr(config, "setSourceRect_"):
        config.setSourceRect_(_cgrect(local_rect))


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


def _call_on_main_thread(callback, *args) -> None:
    if callback is None:
        return
    try:
        from Foundation import NSThread
    except Exception:
        callback(*args)
        return
    try:
        if NSThread.isMainThread():
            callback(*args)
            return
    except Exception:
        pass
    try:
        from PyObjCTools import AppHelper

        AppHelper.callAfter(callback, *args)
        return
    except Exception:
        logger.debug("Failed to schedule backdrop callback on main thread", exc_info=True)
    callback(*args)


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
        self._cached_gaussian_blur_filter = None

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

    def _gaussian_blur_filter(self, cifilter):
        cached = getattr(self, "_cached_gaussian_blur_filter", None)
        if cached is False:
            return None
        if cached is not None:
            return cached
        try:
            blur = cifilter.filterWithName_("CIGaussianBlur")
        except Exception:
            blur = None
        self._cached_gaussian_blur_filter = blur if blur is not None else False
        return blur

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
            blur = self._gaussian_blur_filter(CIFilter)
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

    _shell_diag_count = 0

    def optical_shell_sample_buffer(
        self,
        sample_buffer,
        *,
        shell_config,
        cleanup_blur_radius_points: float,
        bridge,
    ):
        try:
            return self._optical_shell_sample_buffer_inner(
                sample_buffer,
                shell_config=shell_config,
                cleanup_blur_radius_points=cleanup_blur_radius_points,
                bridge=bridge,
            )
        except Exception:
            _MetalBlurPipeline._shell_diag_count += 1
            if _MetalBlurPipeline._shell_diag_count <= 3:
                logger.info("SCK optical_shell_sample_buffer FAILED", exc_info=True)
            return None

    def _optical_shell_sample_buffer_inner(
        self,
        sample_buffer,
        *,
        shell_config,
        cleanup_blur_radius_points: float,
        bridge,
    ):
        _sck_timer.begin("total")
        try:
            from Quartz import CIImage, CIFilter
        except Exception:
            return None
        _sck_timer.begin("ci_setup")
        # Use _CIImage_from_sample_buffer if available — handles IOSurface
        # fallback for SCK sample buffers where CMSampleBufferGetImageBuffer
        # returns NULL through the ctypes bridge.
        ci_from_sb = bridge.get("_CIImage_from_sample_buffer")
        if ci_from_sb is not None:
            ci_image = ci_from_sb(sample_buffer)
        else:
            pixel_buffer = bridge["CMSampleBufferGetImageBuffer"](sample_buffer)
            if pixel_buffer is None:
                logger.info("SCK: CMSampleBufferGetImageBuffer returned None")
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
        _sck_timer.end("ci_setup")

        _sck_timer.begin("warp")
        output = _apply_optical_shell_warp_ci_image(output, working_extent, scaled_shell_config)
        _sck_timer.end("warp")

        if hasattr(output, "imageByCroppingToRect_"):
            output = output.imageByCroppingToRect_(working_extent)

        if cleanup_blur > 0.0:
            _sck_timer.begin("blur")
            blur = self._gaussian_blur_filter(CIFilter)
            if blur is not None:
                blur.setDefaults()
                blur.setValue_forKey_(output, "inputImage")
                blur.setValue_forKey_(cleanup_blur, "inputRadius")
                candidate = blur.valueForKey_("outputImage")
                if candidate is not None and hasattr(candidate, "imageByCroppingToRect_"):
                    output = candidate.imageByCroppingToRect_(working_extent)
            _sck_timer.end("blur")

        _sck_timer.begin("render")
        result = self._render_ci_image_to_sample_buffer(
            output,
            working_extent,
            source_sample=sample_buffer,
            bridge=bridge,
        )
        _sck_timer.end("render")
        _sck_timer.end("total")
        _sck_timer.frame_done()
        return result


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
            # CMSampleBufferGetPresentationTimeStamp / GetDuration accept
            # toll-free bridged ObjC objects (@) and return CMTime structs.
            # CMSampleBufferGetSampleAttachmentsArray likewise.
            objc.loadBundleFunctions(
                coremedia_bundle,
                globals(),
                [
                    ("CMSampleBufferGetPresentationTimeStamp", b"{_CMTime=qiIq}@"),
                    ("CMSampleBufferGetDuration", b"{_CMTime=qiIq}@"),
                    ("CMSampleBufferGetSampleAttachmentsArray", b"@@B"),
                ],
            )

            # Prefer pyobjc-framework-CoreMedia if installed — it has
            # correct type metadata for CMSampleBuffer functions.
            _has_native_coremedia = False
            try:
                import CoreMedia as _CoreMediaModule
                _native_GetImageBuffer = _CoreMediaModule.CMSampleBufferGetImageBuffer
                _native_GetAttachments = _CoreMediaModule.CMSampleBufferGetSampleAttachmentsArray
                _has_native_coremedia = True
                logger.info("SCK: using native pyobjc-framework-CoreMedia bindings")
            except (ImportError, AttributeError):
                logger.info("SCK: pyobjc-framework-CoreMedia not available, using manual bridge")

            # Manual bridge fallback
            _cm_direct_ns = {}
            objc.loadBundleFunctions(
                coremedia_bundle,
                _cm_direct_ns,
                [
                    ("CMSampleBufferGetImageBuffer", b"@@"),
                    ("CMSampleBufferGetSampleAttachmentsArray", b"@@B"),
                ],
            )
            _cm_direct_GetImageBuffer = _cm_direct_ns.get("CMSampleBufferGetImageBuffer")
            _cm_direct_GetAttachments = _cm_direct_ns.get("CMSampleBufferGetSampleAttachmentsArray")

            _cm_lib = ctypes.CDLL(f"{_COREMEDIA_FRAMEWORK_PATH}/CoreMedia")
            _cm_lib.CMSampleBufferGetImageBuffer.argtypes = [ctypes.c_void_p]
            _cm_lib.CMSampleBufferGetImageBuffer.restype = ctypes.c_void_p

            _cv_lib = ctypes.CDLL(f"{_COREVIDEO_FRAMEWORK_PATH}/CoreVideo")
            _cv_lib.CVPixelBufferGetIOSurface.argtypes = [ctypes.c_void_p]
            _cv_lib.CVPixelBufferGetIOSurface.restype = ctypes.c_void_p

            _iosurface_lib = ctypes.CDLL("/System/Library/Frameworks/IOSurface.framework/IOSurface")

            def _CMSampleBufferGetImageBuffer_via_ctypes(sample_buffer):
                # Use pyobjc_id for the raw ObjC pointer — more reliable
                # than __c_void_p__ for toll-free bridged CF types.
                raw_ptr = objc.pyobjc_id(sample_buffer)
                result_ptr = _cm_lib.CMSampleBufferGetImageBuffer(raw_ptr)
                if not result_ptr:
                    return None
                return objc.objc_object(c_void_p=result_ptr)

            # CFRetain/CFRelease for preventing premature release of
            # CVPixelBuffer returned by CMSampleBufferGetImageBuffer.
            _cf_lib = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            _cf_lib.CFRetain.argtypes = [ctypes.c_void_p]
            _cf_lib.CFRetain.restype = ctypes.c_void_p
            _cf_lib.CFRelease.argtypes = [ctypes.c_void_p]
            _cf_lib.CFRelease.restype = None

            # CFRetain/CFRelease for preventing premature release of
            # CVPixelBuffer returned by CMSampleBufferGetImageBuffer.
            _cf_lib = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            _cf_lib.CFRetain.argtypes = [ctypes.c_void_p]
            _cf_lib.CFRetain.restype = ctypes.c_void_p
            _cf_lib.CFRelease.argtypes = [ctypes.c_void_p]
            _cf_lib.CFRelease.restype = None

            _ci_from_sb_diag = [0]

            # Load CMSampleBufferGetFormatDescription for diagnostics
            _cm_lib.CMSampleBufferGetFormatDescription.argtypes = [ctypes.c_void_p]
            _cm_lib.CMSampleBufferGetFormatDescription.restype = ctypes.c_void_p
            _cm_lib.CMSampleBufferIsValid.argtypes = [ctypes.c_void_p]
            _cm_lib.CMSampleBufferIsValid.restype = ctypes.c_bool
            _cm_lib.CMSampleBufferDataIsReady.argtypes = [ctypes.c_void_p]
            _cm_lib.CMSampleBufferDataIsReady.restype = ctypes.c_bool

            def _CIImage_from_sample_buffer(sample_buffer):
                """Extract a CIImage from a CMSampleBuffer."""
                from Quartz import CIImage
                _ci_from_sb_diag[0] += 1
                n = _ci_from_sb_diag[0]

                if n <= 20:
                    raw = objc.pyobjc_id(sample_buffer)
                    is_valid = _cm_lib.CMSampleBufferIsValid(raw)
                    data_ready = _cm_lib.CMSampleBufferDataIsReady(raw)
                    fmt_desc = _cm_lib.CMSampleBufferGetFormatDescription(raw)
                    pb = _cm_lib.CMSampleBufferGetImageBuffer(raw)
                    # Check for IOSurface directly on the sample buffer
                    ios_ptr = None
                    if pb:
                        ios_ptr = _cv_lib.CVPixelBufferGetIOSurface(pb)
                    # Try getting the IOSurface from the format description
                    # or via CMSampleBufferGetDataBuffer
                    _cm_lib.CMSampleBufferGetDataBuffer.argtypes = [ctypes.c_void_p]
                    _cm_lib.CMSampleBufferGetDataBuffer.restype = ctypes.c_void_p
                    data_buf = _cm_lib.CMSampleBufferGetDataBuffer(raw)
                    logger.info(
                        "SCK ci_from_sb[%d]: valid=%s ready=%s fmt=%s pb=%s data_buf=%s",
                        n, is_valid, data_ready,
                        hex(fmt_desc) if fmt_desc else "None",
                        hex(pb) if pb else "NULL",
                        hex(data_buf) if data_buf else "NULL",
                    )
                    # If no pixel buffer, try creating CIImage directly
                    # from the sample buffer's IOSurface attachment
                    if not pb and hasattr(sample_buffer, "imageBuffer"):
                        try:
                            ib = sample_buffer.imageBuffer()
                            if ib is not None and n <= 5:
                                logger.info("SCK ci_from_sb[%d]: ObjC imageBuffer=%s", n, ib)
                        except Exception:
                            pass

                # Strategy 1: native pyobjc-framework-CoreMedia (correct
                # type metadata, no bridging issues).
                if _has_native_coremedia:
                    try:
                        pb = _native_GetImageBuffer(sample_buffer)
                        if pb is not None:
                            ci = CIImage.imageWithCVPixelBuffer_(pb)
                            if n <= 20:
                                logger.info("SCK ci_from_sb[%d]: native pb=%s ci=%s", n, pb, ci is not None)
                            return ci
                        elif n <= 20:
                            logger.info("SCK ci_from_sb[%d]: native returned None", n)
                    except Exception:
                        if n <= 20:
                            logger.info("SCK ci_from_sb[%d]: native raised", n, exc_info=True)

                # Strategy 2: manual bridge via loadBundleFunctions '@@'
                if _cm_direct_GetImageBuffer is not None:
                    try:
                        pb = _cm_direct_GetImageBuffer(sample_buffer)
                        if n <= 20:
                            logger.info("SCK ci_from_sb[%d]: direct pb=%s sb_type=%s", n, pb, type(sample_buffer).__name__)
                        if pb is not None:
                            ci = CIImage.imageWithCVPixelBuffer_(pb)
                            if ci is not None:
                                return ci
                    except Exception:
                        if n <= 20:
                            logger.info("SCK ci_from_sb[%d]: direct raised", n, exc_info=True)

                # Strategy 3: ctypes (least reliable)
                raw_ptr = objc.pyobjc_id(sample_buffer)
                pb_ptr = _cm_lib.CMSampleBufferGetImageBuffer(raw_ptr)
                if n <= 20:
                    logger.info("SCK ci_from_sb[%d]: ctypes pb=%s", n, hex(pb_ptr) if pb_ptr else "NULL")
                if not pb_ptr:
                    return None
                _cf_lib.CFRetain(pb_ptr)
                try:
                    return CIImage.imageWithCVPixelBuffer_(objc.objc_object(c_void_p=pb_ptr))
                finally:
                    _cf_lib.CFRelease(pb_ptr)

            CMSampleBufferGetImageBuffer = _CMSampleBufferGetImageBuffer_via_ctypes

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
                "_CIImage_from_sample_buffer": _CIImage_from_sample_buffer,
                "CMSampleBufferGetPresentationTimeStamp": CMSampleBufferGetPresentationTimeStamp,
                "CMSampleBufferGetDuration": CMSampleBufferGetDuration,
                "CMSampleBufferGetSampleAttachmentsArray": CMSampleBufferGetSampleAttachmentsArray,
                "kCMSampleAttachmentKey_DisplayImmediately": _kCMSampleAttachmentKey_DisplayImmediately,
                "_cv_lib": _cv_lib,
                "_cm_lib": _cm_lib,
                "_native_GetImageBuffer": _native_GetImageBuffer if _has_native_coremedia else None,
                "_cm_direct_GetImageBuffer": _cm_direct_GetImageBuffer,
            }
        except Exception:
            logger.info("ScreenCaptureKit bridge load FAILED", exc_info=True)
            _BRIDGE_STATE = None
        if _BRIDGE_STATE is not None:
            logger.info("ScreenCaptureKit bridge loaded OK")
        return _BRIDGE_STATE


_ScreenCaptureKitStreamOutput = None

def _build_stream_output_class():
    global _ScreenCaptureKitStreamOutput
    if _ScreenCaptureKitStreamOutput is not None:
        return
    if objc is None or not hasattr(objc, "lookUpClass"):
        class _Fallback(NSObject):
            def initWithRenderer_(self, renderer):
                self._renderer = renderer
                return self
            def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
                self._renderer._consume_sample_buffer(sample_buffer, output_type)
        _ScreenCaptureKitStreamOutput = _Fallback
        return

    try:
        _ScreenCaptureKitStreamOutput = objc.lookUpClass("_ScreenCaptureKitStreamOutput")
        return
    except Exception:
        pass

    # Register an informal protocol with the correct method signature.
    # This tells PyObjC the parameter types without importing the full
    # pyobjc-framework-ScreenCaptureKit (which pollutes global type
    # metadata and breaks other code paths).
    #
    # The correct signature: v@:@^{opaqueCMSampleBuffer=}q
    # - @ = SCStream (ObjC object)
    # - ^{opaqueCMSampleBuffer=} = CMSampleBufferRef (C struct pointer)
    # - q = NSInteger (SCStreamOutputType)
    try:
        objc.informal_protocol(
            "SCStreamOutput",
            [
                objc.selector(
                    None,
                    selector=b"stream:didOutputSampleBuffer:ofType:",
                    signature=b"v@:@^{opaqueCMSampleBuffer=}q",
                    isRequired=False,
                ),
            ],
        )
    except Exception:
        logger.debug("Failed to register SCStreamOutput informal protocol", exc_info=True)

    class _StreamOutput(NSObject):
        def initWithRenderer_(self, renderer):
            self = objc.super(_StreamOutput, self).init()
            if self is None:
                return None
            self._renderer = renderer
            self._frame_count = 0
            return self

        def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
            if self._frame_count == 0:
                logger.info(
                    "SCK: first sample buffer (output_type=%r, sb_type=%s)",
                    output_type, type(sample_buffer).__name__,
                )
            self._frame_count += 1
            pool = objc.autorelease_pool()
            pool.__enter__()
            try:
                self._renderer._consume_sample_buffer(sample_buffer, output_type)
            finally:
                pool.__exit__(None, None, None)

    _ScreenCaptureKitStreamOutput = _StreamOutput
# interferes with test mocking.


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
        self._has_live_content = False
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
        self._display_link_renderer = None
        self._cached_gaussian_blur_filter = None

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
            # Use Metal-backed CIContext for GPU-accelerated rendering.
            # Falls back to generic context if Metal is unavailable.
            from Foundation import NSBundle
            metal_bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/Metal.framework")
            objc.loadBundleFunctions(metal_bundle, globals(), [("MTLCreateSystemDefaultDevice", b"@")])
            device = MTLCreateSystemDefaultDevice()
            if device is not None:
                self._ci_context = CIContext.contextWithMTLDevice_(device)
                logger.info("SCK: CIContext backed by Metal device")
            else:
                self._ci_context = CIContext.contextWithOptions_(None)
        except Exception:
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

    def set_metal_backdrop_layer(self, layer) -> None:
        self._metal_backdrop_layer = layer

    def set_display_link_renderer(self, renderer) -> None:
        """Set the MetalDisplayLinkRenderer that drives drawable presentation."""
        self._display_link_renderer = renderer

    def set_live_optical_shell_config(self, config) -> None:
        if not config or config.get("enabled") is not True:
            self._optical_shell_config = None
            return
        self._optical_shell_config = dict(config)

    def _clear_live_content(self) -> None:
        with self._lock:
            self._latest_image = None
        self._has_live_content = False

    def stop_live_stream(self) -> None:
        dl_renderer = getattr(self, "_display_link_renderer", None)
        if dl_renderer is not None:
            dl_renderer.stop()
        stream = self._stream
        self._stream = None
        self._stream_output = None
        self._stream_started = False
        self._startup_requested = False
        self._pending_signature = None
        self._applied_signature = None
        self._clear_live_content()
        if stream is None:
            return
        if hasattr(stream, "stopCaptureWithCompletionHandler_"):
            try:
                stream.stopCaptureWithCompletionHandler_(lambda *args: None)
                return
            except Exception:
                logger.debug("Failed to stop ScreenCaptureKit capture stream", exc_info=True)
        try:
            stop = getattr(stream, "stopCapture", None)
            if callable(stop):
                stop()
        except Exception:
            logger.debug("Failed to stop ScreenCaptureKit capture stream", exc_info=True)

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

    def _gaussian_blur_filter(self, cifilter):
        cached = getattr(self, "_cached_gaussian_blur_filter", None)
        if cached is False:
            return None
        if cached is not None:
            return cached
        try:
            blur = cifilter.filterWithName_("CIGaussianBlur")
        except Exception:
            blur = None
        self._cached_gaussian_blur_filter = blur if blur is not None else False
        return blur

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

    _publish_image_count = 0
    _publish_image_last_report = 0.0
    _publish_image_interval_count = 0

    def _publish_live_image(self, image) -> None:
        self._publish_image_count += 1
        self._publish_image_interval_count += 1
        now = time.monotonic()
        elapsed = now - self._publish_image_last_report
        if elapsed >= 5.0:
            fps = self._publish_image_interval_count / elapsed if elapsed > 0 else 0
            logger.info("SCK publish: %d frames in %.1fs (%.1f fps)", self._publish_image_interval_count, elapsed, fps)
            self._publish_image_last_report = now
            self._publish_image_interval_count = 0
        with self._lock:
            self._latest_image = image
        self._has_live_content = True
        _call_on_main_thread(self._dispatch_frame_callback, image)

    def _publish_live_sample_buffer(self, sample_buffer) -> None:
        callback = self._sample_buffer_callback
        if callback is None:
            return
        self._has_live_content = True
        try:
            _call_on_main_thread(callback, sample_buffer)
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
        self._clear_live_content()
        bridge = _load_screencapturekit_bridge()
        if bridge is None:
            return
        self._startup_requested = True
        SCShareableContent = bridge["SCShareableContent"]

        def got_content(content):
            self._startup_requested = False
            if content is None:
                logger.info("SCK: getShareableContent returned None")
                return
            try:
                self._current_content = content
                self._current_display = self._match_display(content)
                if self._current_display is None:
                    logger.info("SCK: no matching display found")
                    return
                self._current_display_frame = self._current_display.frame()
                logger.info("SCK: display matched, frame=%r", self._current_display_frame)
                content_filter = self._build_filter(content, self._current_display, window_number)
                config = self._build_configuration(content_filter, capture_rect)
                SCStream = bridge["SCStream"]
                stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                    content_filter,
                    config,
                    None,
                )
                _build_stream_output_class()
                stream_output = _ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(self)
                handler_queue = self._sample_handler_queue()
                logger.info("SCK: handler queue=%r", handler_queue)
                try:
                    result = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                        stream_output,
                        bridge["SCStreamOutputTypeScreen"],
                        handler_queue,
                        None,
                    )
                except Exception:
                    logger.info("SCK: addStreamOutput raised", exc_info=True)
                    return
                # PyObjC may return (BOOL, NSError*) or just BOOL depending
                # on bridgesupport metadata availability.
                if isinstance(result, tuple):
                    success, error = result
                else:
                    success = bool(result)
                    error = None
                if not success:
                    logger.info("SCK: addStreamOutput failed: %r", error)
                    return
                logger.info("SCK: addStreamOutput succeeded")

                # With pyobjc-framework-ScreenCaptureKit installed, the
                # block correctly expects 1 arg (NSError*).  Without it,
                # bridgesupport says 0 args.  Accept either with *args.
                def started(*args):
                    error = args[0] if args else None
                    if error is not None:
                        logger.info("SCK: startCapture failed: %r", error)
                        return
                    logger.info("SCK: startCapture succeeded — stream is live")
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
                logger.info("SCK: stream startup failed", exc_info=True)
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
        self._clear_live_content()
        self._pending_signature = signature
        try:
            content_filter = self._build_filter(
                self._current_content if self._current_content is not None else SimpleNamespace(windows=lambda: []),
                self._current_display,
                window_number,
            )
            config = self._build_configuration(content_filter, capture_rect)
            self._window_number = window_number

            # PyObjC bridgesupport declares these completion blocks as
            # 0-arg.  Use *args to accept either 0 or 1 arguments.
            def updated_filter(*args):
                error = args[0] if args else None
                if error is not None:
                    logger.debug("ScreenCaptureKit updateContentFilter failed: %r", error)
                    self._pending_signature = None
                    return

                def updated_config(*args):
                    error = args[0] if args else None
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
        if sample_buffer is None:
            return
        # With the correct delegate signature, the sample buffer arrives
        # as a PyObjCPointer (raw C struct pointer) rather than an ObjC
        # object proxy.  CMSampleBuffer is toll-free bridged to NSObject,
        # so wrap it as an objc_object for downstream use.
        if hasattr(sample_buffer, "pointerAsInteger"):
            try:
                ptr = sample_buffer.pointerAsInteger
                sample_buffer = objc.objc_object(c_void_p=ptr)
            except Exception:
                return
        # output_type arrives as None when PyObjC bridges the NSInteger
        # SCStreamOutputType as an ObjC object (@) due to selector signature
        # mismatch.  Treat None as screen output — it's the only type we
        # subscribe to, and audio output would be a separate stream.
        if output_type is not None:
            try:
                output_type_value = int(output_type)
            except Exception:
                output_type_value = output_type
            if output_type_value != bridge["SCStreamOutputTypeScreen"]:
                return
        optical_shell_config = getattr(self, "_optical_shell_config", None)
        diag = getattr(self, "_consume_diag_n", 0)
        self._consume_diag_n = diag + 1
        if diag < 5:
            logger.info(
                "SCK consume[%d]: sb_callback=%s shell_config=%s blur=%.2f metal=%s",
                diag,
                self._sample_buffer_callback is not None,
                optical_shell_config is not None,
                self._blur_radius_points,
                self._metal_blur_pipeline() is not None,
            )
        if self._sample_buffer_callback is not None:
            if optical_shell_config is not None:
                # Direct CIImage path: extract image from SCK sample buffer,
                # apply warp, render to CGImage, publish.  Bypasses the Metal
                # sample-buffer pipeline entirely — simpler and avoids the
                # stale-IOSurface caching issue.
                ci_from_sb = bridge.get("_CIImage_from_sample_buffer")
                if ci_from_sb is not None:
                    try:
                        from Quartz import CIImage
                        ci_image = ci_from_sb(sample_buffer)
                        if ci_image is not None:
                            extent = ci_image.extent()
                            if extent is not None:
                                # SCK delivers frames at Retina pixel scale.
                                # The shell config dimensions are in points.
                                # Scale them to match the CIImage extent.
                                scale = self._current_backing_scale()
                                scaled_config = dict(optical_shell_config)
                                for k in ("content_width_points", "content_height_points",
                                          "corner_radius_points", "band_width_points",
                                          "tail_width_points"):
                                    if k in scaled_config:
                                        scaled_config[k] = float(scaled_config[k]) * scale
                                warped = _apply_optical_shell_warp_ci_image(
                                    ci_image.imageByClampingToExtent() if hasattr(ci_image, "imageByClampingToExtent") else ci_image,
                                    extent,
                                    scaled_config,
                                )
                                if warped is not None:
                                    output = warped.imageByCroppingToRect_(extent) if hasattr(warped, "imageByCroppingToRect_") else warped
                                else:
                                    output = ci_image
                                ctx = self._context()
                                if ctx is not None:
                                    cg = ctx.createCGImage_fromRect_(output, extent)
                                    if cg is not None:
                                        self._publish_live_image(cg)
                                        return
                    except Exception:
                        logger.debug("SCK: direct CIImage optical shell path failed", exc_info=True)
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
        # Display-link-driven Metal path — submit IOSurface to the renderer,
        # which presents via CVDisplayLink (never blocks the SCK queue).
        #
        # Extract the pixel buffer via _CIImage_from_sample_buffer's native
        # or direct bridge (not the ctypes path, which returns None for most
        # frames due to PyObjC bridging issues).  Then get the IOSurface
        # from the pixel buffer.
        dl_renderer = getattr(self, "_display_link_renderer", None)
        if dl_renderer is not None and optical_shell_config is not None:
            dl_diag = getattr(self, "_dl_submit_diag_n", 0)
            self._dl_submit_diag_n = dl_diag + 1
            try:
                cv_lib = bridge.get("_cv_lib")
                if cv_lib is None:
                    if dl_diag < 3:
                        logger.info("SCK dl-submit[%d]: cv_lib not available", dl_diag)
                else:
                    # Use the same extraction path as _CIImage_from_sample_buffer:
                    # try native pyobjc-framework-CoreMedia first, then direct bridge.
                    pb = None
                    ci_from_sb = bridge.get("_CIImage_from_sample_buffer")
                    if ci_from_sb is not None:
                        ci_image = ci_from_sb(sample_buffer)
                        if ci_image is not None:
                            # CIImage.imageWithCVPixelBuffer_ retains the
                            # pixel buffer — extract IOSurface from it.
                            extent = ci_image.extent() if hasattr(ci_image, "extent") else None
                            if extent is not None:
                                w = int(extent.size.width)
                                h = int(extent.size.height)
                                # Get IOSurface: CIImage may expose it via
                                # properties, or we can get it from the original
                                # sample buffer's pixel buffer.
                                ios_obj = None
                                # Try extracting pixel buffer via the native bridge
                                _native_get = bridge.get("_native_GetImageBuffer")
                                _direct_get = bridge.get("_cm_direct_GetImageBuffer")
                                if _native_get is not None:
                                    try:
                                        pb = _native_get(sample_buffer)
                                    except Exception:
                                        pb = None
                                if pb is None and _direct_get is not None:
                                    try:
                                        pb = _direct_get(sample_buffer)
                                    except Exception:
                                        pb = None
                                if pb is not None:
                                    raw_pb = objc.pyobjc_id(pb)
                                    ios = cv_lib.CVPixelBufferGetIOSurface(raw_pb)
                                    if ios:
                                        ios_obj = objc.objc_object(c_void_p=ios)
                                if ios_obj is None:
                                    # Fallback: ctypes path
                                    raw_sb = objc.pyobjc_id(sample_buffer)
                                    _cm_lib = bridge.get("_cm_lib")
                                    if _cm_lib is not None:
                                        pb_ptr = _cm_lib.CMSampleBufferGetImageBuffer(raw_sb)
                                        if pb_ptr:
                                            ios = cv_lib.CVPixelBufferGetIOSurface(pb_ptr)
                                            if ios:
                                                ios_obj = objc.objc_object(c_void_p=ios)
                                if ios_obj is not None and w > 0 and h > 0:
                                    if dl_diag < 3:
                                        logger.info("SCK dl-submit[%d]: IOSurface ok, %dx%d", dl_diag, w, h)
                                    scale = self._current_backing_scale()
                                    scaled_cfg = dict(optical_shell_config)
                                    for k in ("content_width_points", "content_height_points",
                                              "corner_radius_points", "band_width_points",
                                              "tail_width_points"):
                                        if k in scaled_cfg:
                                            scaled_cfg[k] = float(scaled_cfg[k]) * scale
                                    dl_renderer.set_shell_config(scaled_cfg)
                                    dl_renderer.submit_iosurface(ios_obj, width=w, height=h)
                                    self._has_live_content = True
                                    return
                                elif dl_diag < 5:
                                    logger.info("SCK dl-submit[%d]: no IOSurface from CIImage path (w=%d h=%d pb=%s)", dl_diag, w, h, pb is not None)
                            elif dl_diag < 5:
                                logger.info("SCK dl-submit[%d]: CIImage has no extent", dl_diag)
                        elif dl_diag < 5:
                            logger.info("SCK dl-submit[%d]: _CIImage_from_sample_buffer returned None", dl_diag)
                    elif dl_diag < 3:
                        logger.info("SCK dl-submit[%d]: no _CIImage_from_sample_buffer in bridge", dl_diag)
            except Exception:
                if dl_diag < 10:
                    logger.info("Metal display-link submit failed", exc_info=True)

        # CIWarpKernel fallback — used when Metal display-link is unavailable.
        _sck_timer.begin("total")
        try:
            from Quartz import CIImage, CIFilter

            ci_from_sb = bridge.get("_CIImage_from_sample_buffer")
            if ci_from_sb is not None:
                ci_image = ci_from_sb(sample_buffer)
            else:
                pixel_buffer = bridge["CMSampleBufferGetImageBuffer"](sample_buffer)
                if pixel_buffer is None:
                    return
                ci_image = CIImage.imageWithCVPixelBuffer_(pixel_buffer)
            if ci_image is None:
                return
            extent = ci_image.extent() if hasattr(ci_image, "extent") else None
            if extent is None:
                return

            diag = getattr(self, "_consume_diag_n", 0)
            if diag <= 7 and optical_shell_config is not None:
                logger.info(
                    "SCK cgimage fallback: extent=(%s,%s %sx%s) cfg_w=%s cfg_h=%s",
                    extent.origin.x, extent.origin.y,
                    extent.size.width, extent.size.height,
                    optical_shell_config.get("content_width_points"),
                    optical_shell_config.get("content_height_points"),
                )
            output = ci_image
            # Apply optical shell warp via CIImage when Metal pipeline is unavailable.
            if optical_shell_config is not None:
                _sck_timer.begin("pre_blur")
                pre_blur = self._gaussian_blur_filter(CIFilter)
                if pre_blur is not None:
                    pre_blur.setDefaults()
                    clamped = output.imageByClampingToExtent() if hasattr(output, "imageByClampingToExtent") else output
                    pre_blur.setValue_forKey_(clamped, "inputImage")
                    pre_blur.setValue_forKey_(1.5, "inputRadius")
                    blurred = pre_blur.valueForKey_("outputImage")
                    if blurred is not None:
                        output = blurred.imageByCroppingToRect_(extent) if hasattr(blurred, "imageByCroppingToRect_") else blurred
                _sck_timer.end("pre_blur")

                _sck_timer.begin("warp")
                scale = self._current_backing_scale()
                scaled_cfg = dict(optical_shell_config)
                for k in ("content_width_points", "content_height_points",
                          "corner_radius_points", "band_width_points",
                          "tail_width_points"):
                    if k in scaled_cfg:
                        scaled_cfg[k] = float(scaled_cfg[k]) * scale
                warped = _apply_optical_shell_warp_ci_image(output, extent, scaled_cfg)
                if warped is not None:
                    output = warped
                    if hasattr(output, "imageByCroppingToRect_"):
                        output = output.imageByCroppingToRect_(extent)
                _sck_timer.end("warp")
            elif self._blur_radius_points > 0.0:
                blur = self._gaussian_blur_filter(CIFilter)
                if blur is not None:
                    blur.setDefaults()
                    blur.setValue_forKey_(ci_image, "inputImage")
                    blur.setValue_forKey_(self._blur_radius_points, "inputRadius")
                    candidate = blur.valueForKey_("outputImage")
                    if candidate is not None:
                        output = candidate.imageByCroppingToRect_(extent)
            _sck_timer.begin("render")
            context = self._context()
            if context is None:
                return
            image = context.createCGImage_fromRect_(output, extent)
            _sck_timer.end("render")
            if image is None:
                return
            self._publish_live_image(image)
        except Exception:
            fb_diag = getattr(self, "_fallback_diag_n", 0)
            self._fallback_diag_n = fb_diag + 1
            if fb_diag < 5:
                logger.info("SCK CIWarpKernel fallback failed", exc_info=True)
        finally:
            _sck_timer.end("total")
            _sck_timer.frame_done()

    def capture_blurred_image(self, *, window_number: int, capture_rect, blur_radius_points: float):
        self._blur_radius_points = max(blur_radius_points, 0.0)
        self._capture_rect_for_crop = capture_rect
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
            if getattr(self, "_has_live_content", False):
                return None
            return self._fallback_renderer().capture_blurred_image(
                window_number=window_number,
                capture_rect=capture_rect,
                blur_radius_points=blur_radius_points,
            )

        with self._lock:
            image = self._latest_image
        if image is not None:
            return image
        return self._fallback_renderer().capture_blurred_image(
            window_number=window_number,
            capture_rect=capture_rect,
            blur_radius_points=blur_radius_points,
        )
