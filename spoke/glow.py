"""Screen-border glow overlay that pulses with voice amplitude.

A borderless, transparent, click-through NSWindow that draws a soft glow
around the screen edges. Intensity follows the RMS amplitude of the
microphone input, with fast rise and slow decay for a breathing effect.
"""

from __future__ import annotations
import colorsys
import ctypes
import logging
import math
import os
import struct
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
from .tintilla import (
    ADDITIVE_CURVE_MODE_EXPONENTIAL,
    ADDITIVE_CURVE_MODE_RATIONAL,
    SCREEN_GLOW_CORE_LAYER_ID,
    SCREEN_GLOW_TIGHT_BLOOM_LAYER_ID,
    SCREEN_GLOW_WIDE_BLOOM_LAYER_ID,
    SCREEN_DIMMER_LAYER_ID,
    SCREEN_VIGNETTE_CORE_LAYER_ID,
    SCREEN_VIGNETTE_MID_LAYER_ID,
    SCREEN_VIGNETTE_TAIL_LAYER_ID,
    WIDE_BLOOM_PROFILE_MIST,
    WIDE_BLOOM_PROFILE_QUEST,
    WIDE_BLOOM_PROFILE_TIGHT,
)

logger = logging.getLogger(__name__)

_GLOW_LAYER_IDS = [
    SCREEN_GLOW_CORE_LAYER_ID,
    SCREEN_GLOW_TIGHT_BLOOM_LAYER_ID,
    SCREEN_GLOW_WIDE_BLOOM_LAYER_ID,
]
_VIGNETTE_LAYER_IDS = [
    SCREEN_VIGNETTE_CORE_LAYER_ID,
    SCREEN_VIGNETTE_MID_LAYER_ID,
    SCREEN_VIGNETTE_TAIL_LAYER_ID,
]
_WIDE_BLOOM_PROFILE_SPECS = {
    WIDE_BLOOM_PROFILE_TIGHT: {"falloff": 18.0, "power": 3.4},
    WIDE_BLOOM_PROFILE_QUEST: {"falloff": 24.0, "power": 2.9},
    WIDE_BLOOM_PROFILE_MIST: {"falloff": 30.0, "power": 2.4},
}

_METAL_WIDE_BLOOM_ENABLED = os.environ.get("SPOKE_METAL_WIDE_BLOOM", "0") == "1"
_METAL_FRAME_INTERVAL_S = 1.0 / 60.0
_MECHA_VISOR_SINE_BOOST = float(os.environ.get("SPOKE_MECHA_VISOR_SINE_BOOST", "0.0"))
_MECHA_VISOR_SINE_HZ = float(os.environ.get("SPOKE_MECHA_VISOR_SINE_HZ", "0.25"))
_MECHA_VISOR_WIDE_BLOOM_FALLOFF_SCALE = float(
    os.environ.get("SPOKE_MECHA_VISOR_WIDE_BLOOM_FALLOFF_SCALE", "1.0")
)
_MECHA_VISOR_WIDE_BLOOM_ALPHA_SCALE = float(
    os.environ.get("SPOKE_MECHA_VISOR_WIDE_BLOOM_ALPHA_SCALE", "1.0")
)
_MECHA_VISOR_WIDE_BLOOM_POWER_SCALE = float(
    os.environ.get("SPOKE_MECHA_VISOR_WIDE_BLOOM_POWER_SCALE", "1.0")
)
_MECHA_VISOR_DISABLE_CORE_PASS = os.environ.get("SPOKE_MECHA_VISOR_DISABLE_CORE_PASS", "0") == "1"
_MTL_PIXEL_FORMAT_BGRA8_UNORM = 80
_MTL_LOAD_ACTION_CLEAR = 2
_MTL_STORE_ACTION_STORE = 1
_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP = 4
_MTL_BUFFER_LAYOUT = "<" + "f" * 28
_MTL_BUFFER_SIZE = struct.calcsize(_MTL_BUFFER_LAYOUT)
_METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float4 viewport_time_mix;
    float4 inner_params;
    float4 tight_params;
    float4 wide_params;
    float4 inner_color;
    float4 middle_color;
    float4 outer_color;
};

vertex VSOut vs_main(uint vid [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
    };
    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0),
    };

    VSOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

inline float pass_alpha(float signed_distance, float falloff, float power) {
    if (signed_distance >= 0.0) {
        return 0.0;
    }
    float distance = -signed_distance;
    return exp(-pow(distance / max(falloff, 1e-6), power));
}

inline float4 source_over(float4 dst, float3 rgb, float alpha) {
    float clamped = clamp(alpha, 0.0, 1.0);
    float4 src = float4(rgb * clamped, clamped);
    float one_minus_src = 1.0 - src.a;
    return float4(src.rgb + dst.rgb * one_minus_src, src.a + dst.a * one_minus_src);
}

fragment float4 fs_main(
    VSOut in [[stage_in]],
    constant Uniforms& u [[buffer(0)]],
    constant float* signed_field [[buffer(1)]]
) {
    uint width = max(uint(u.viewport_time_mix.x), 1u);
    uint height = max(uint(u.viewport_time_mix.y), 1u);
    uint x = min(uint(in.uv.x * float(width - 1u)), width - 1u);
    uint y = min(uint(in.uv.y * float(height - 1u)), height - 1u);
    float signed_distance = signed_field[y * width + x];

    float inner_alpha = pass_alpha(signed_distance, u.inner_params.x, u.inner_params.y) * u.inner_params.z;
    float tight_alpha = pass_alpha(signed_distance, u.tight_params.x, u.tight_params.y) * u.tight_params.z;
    float wide_alpha = pass_alpha(signed_distance, u.wide_params.x, u.wide_params.y) * u.wide_params.z;

    float noise = fract(sin(dot(float2(x, y), float2(12.9898, 78.233))) * 43758.5453);
    float wave = 0.5 + 0.5 * sin(u.viewport_time_mix.z * 3.1 + noise * 6.2831853 + signed_distance * 0.03);
    wide_alpha *= mix(0.72, 1.28, wave);

    float4 rgba = float4(0.0);
    rgba = source_over(rgba, u.inner_color.rgb, inner_alpha);
    rgba = source_over(rgba, u.middle_color.rgb, tight_alpha);
    rgba = source_over(rgba, u.outer_color.rgb, wide_alpha);
    rgba *= clamp(u.viewport_time_mix.w, 0.0, 1.0);
    return rgba;
}
"""

_metal_bundle_loaded = False
_metal_bundle_error = None


def _scale_color_saturation(
    color: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    """Scale an RGB color's saturation while keeping its hue and value stable."""
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    return colorsys.hsv_to_rgb(hue, min(max(saturation * factor, 0.0), 1.0), value)


def _load_metal_symbols():
    global _metal_bundle_loaded, _metal_bundle_error
    if _metal_bundle_loaded:
        return
    if _metal_bundle_error is not None:
        raise _metal_bundle_error
    try:
        metal_bundle = objc.loadBundle(
            "Metal",
            globals(),
            bundle_path="/System/Library/Frameworks/Metal.framework",
        )
        objc.loadBundle(
            "QuartzCore",
            globals(),
            bundle_path="/System/Library/Frameworks/QuartzCore.framework",
        )
        objc.loadBundleFunctions(
            metal_bundle,
            globals(),
            [("MTLCreateSystemDefaultDevice", b"@")],
        )
        _metal_bundle_loaded = True
    except Exception as exc:
        _metal_bundle_error = exc
        raise


def _metal_device_available() -> bool:
    try:
        _load_metal_symbols()
        return MTLCreateSystemDefaultDevice() is not None
    except Exception:
        return False


def _copy_bytes_to_metal_buffer(buffer, payload: bytes) -> None:
    ctypes.memmove(int(buffer.contents()), payload, len(payload))


def _build_metal_pipeline(device):
    _load_metal_symbols()
    pipeline_descriptor = objc.lookUpClass("MTLRenderPipelineDescriptor").alloc().init()
    library = device.newLibraryWithSource_options_error_(_METAL_SHADER_SOURCE, None, None)
    pipeline_descriptor.setVertexFunction_(library.newFunctionWithName_("vs_main"))
    pipeline_descriptor.setFragmentFunction_(library.newFunctionWithName_("fs_main"))
    attachment = pipeline_descriptor.colorAttachments().objectAtIndexedSubscript_(0)
    attachment.setPixelFormat_(_MTL_PIXEL_FORMAT_BGRA8_UNORM)
    return device.newRenderPipelineStateWithDescriptor_error_(pipeline_descriptor, None)


def _mecha_visor_signal_boost(signal: float, now: float | None = None) -> float:
    if _MECHA_VISOR_SINE_BOOST <= 0.0:
        return min(max(signal, 0.0), 1.0)

    now = time.monotonic() if now is None else now
    phase = 0.5 + 0.5 * math.sin(now * math.tau * _MECHA_VISOR_SINE_HZ)
    return min(max(signal + (_MECHA_VISOR_SINE_BOOST * phase), 0.0), 1.0)

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
_GLOW_BASE_OPACITY = 0.07245  # half the prior floor so the dark-language glow starts quieter at rest
_GLOW_PEAK_TARGET = 0.90
_GLOW_BASE_OPACITY_DARK = 0.1875
_GLOW_BASE_OPACITY_LIGHT = 0.2058
_GLOW_PEAK_TARGET_DARK = 0.90
_GLOW_PEAK_TARGET_LIGHT = 0.75
_EDGE_INNER_SATURATION_SCALE = 0.70
_EDGE_OUTER_SATURATION_SCALE = 1.80
# MacBook Pro 14"/16" (2021+) has asymmetric display corners.
# We use slightly tighter radii than the physical bezel so the glow
# source stays close to the corners — the bezel hides the overshoot.
# Keyed by (native_pixel_width, native_pixel_height).
_DISPLAY_CORNER_RADII: dict[tuple[int, int], tuple[float, float]] = {
    # 16" MacBook Pro (2021+): 3456×2234 native
    (3456, 2234): (10.0, 6.0),
    # 14" MacBook Pro (2021+): 3024×1964 native
    (3024, 1964): (10.0, 6.0),  # start from 16" values, tune visually
}
_CORNER_RADIUS_TOP_DEFAULT = 10.0
_CORNER_RADIUS_BOTTOM_DEFAULT = 6.0

_GLOW_MULTIPLIER = float(os.environ.get("SPOKE_GLOW_MULTIPLIER", "21.4"))
_DIM_SCREEN = os.environ.get("SPOKE_DIM_SCREEN", "1") == "1"
_DIM_OPACITY_DARK = 0.147  # 50% of the prior hold-time dim curve
_DIM_OPACITY_LIGHT = 0.2226

def _dim_target_for_brightness(brightness: float) -> float:
    # Keep the same curve shape, but at 50% of the prior hold-time opacity.
    if brightness <= 0.5:
        t = brightness / 0.5
        return _DIM_OPACITY_DARK + t * (0.28 - _DIM_OPACITY_DARK)
    else:
        t = (brightness - 0.5) / 0.5
        return 0.28 + t * (_DIM_OPACITY_LIGHT - 0.28)

# Amplitude smoothing: slower attack and slower release across the edge stack.
_RISE_FACTOR = 0.90
_DECAY_FACTOR = 0.40
_VIGNETTE_RISE_FACTOR = 0.9292893218813453
_VIGNETTE_DECAY_FACTOR = 0.282842712474619

# Fade timing: keep the same shape, but cut the visible attack/release speed in half.
_FADE_IN_S = 0.026
_FADE_OUT_S = 0.066
_GLOW_SHOW_FADE_S = 0.132
_GLOW_HIDE_FADE_S = 2.4
_GLOW_SHOW_TIMING = "easeIn"
_DIM_SHOW_FADE_S = 0.72
_DIM_HIDE_FADE_S = 1.6
_WINDOW_TEARDOWN_CUSHION_S = 0.016


_BRIGHTNESS_SAMPLE_INTERVAL = 1.0  # seconds between recurring samples
_BRIGHTNESS_PATCH_SIZE = 50  # pixels per patch side
_DISTANCE_FIELD_SCALE_DEFAULT = 2.0
_NOTCH_BOTTOM_RADIUS = 8.0
_NOTCH_SHOULDER_SMOOTHING = 9.5
_LIGHT_BACKGROUND_EDGE_START = 0.55
_LIGHT_BACKGROUND_EDGE_BOOST = 0.664
_VIGNETTE_OPACITY_SCALE = 0.78


def _sample_screen_brightness(screen, excluding_window_id=None) -> float:
    """Sample average brightness from 4 small patches (one per quadrant).

    Each patch is 50x50 pixels, inset 25% from each screen edge to avoid
    our own glow/vignette. Returns 0.0 (black) to 1.0 (white).
    ~4x faster than a fullscreen capture on retina displays.
    """
    try:
        from Quartz import (
            CGWindowListCreateImage,
            kCGWindowListOptionOnScreenBelowWindow,
            kCGWindowListOptionOnScreenOnly,
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

            option = (
                kCGWindowListOptionOnScreenBelowWindow
                if excluding_window_id not in (None, 0)
                else kCGWindowListOptionOnScreenOnly
            )
            window_id = excluding_window_id if excluding_window_id not in (None, 0) else kCGNullWindowID
            image = CGWindowListCreateImage(rect, option, window_id, 0)
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

    geometry["notch"] = {
        "x": left_max_x * scale,
        "y": notch_base_y * scale,
        "width": notch_width * scale,
        "height": notch_height * scale,
        "bottom_radius": min(notch_height * scale * 0.45, _NOTCH_BOTTOM_RADIUS * scale),
        "shoulder_smoothing": _NOTCH_SHOULDER_SMOOTHING * scale,
    }
    return geometry


def _asymmetric_rounded_rect_sdf(
    x,
    y,
    width: float,
    height: float,
    top_radius: float,
    bottom_radius: float,
) :
    import numpy as np

    half_width = width * 0.5
    half_height = height * 0.5
    radii = np.where(y >= 0.0, top_radius, bottom_radius).astype(np.float32, copy=False)
    qx = np.abs(x) - (half_width - radii)
    qy = np.abs(y) - (half_height - radii)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    
    # Smooth the inner corner ridge where the negative distances meet
    corner_smoothing = 24.0
    seam = np.maximum(corner_smoothing - np.abs(qx - qy), 0.0)
    smoothed_max = np.maximum(qx, qy) + (seam * seam) * (0.25 / corner_smoothing)
    
    inside = np.minimum(smoothed_max, 0.0)
    return outside + inside - radii


def _display_signed_distance_field(geometry: dict):
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
    notch_signed = _asymmetric_rounded_rect_sdf(
        centered_x - notch_center_x,
        centered_y - notch_center_y,
        notch["width"],
        notch["height"],
        0.0,
        notch["bottom_radius"],
    )
    shoulder_smoothing = max(notch.get("shoulder_smoothing", 0.0), 0.0)
    if shoulder_smoothing <= 0.0:
        return np.maximum(signed, -notch_signed).astype(np.float32, copy=False)

    seam = np.maximum(shoulder_smoothing - np.abs(signed + notch_signed), 0.0)
    softened = np.maximum(signed, -notch_signed) + (seam * seam) * (0.25 / shoulder_smoothing)
    return softened.astype(np.float32, copy=False)


def _distance_field_opacity(distance: float, falloff: float, power: float) -> float:
    return _distance_field_opacity_with_mode(
        distance,
        falloff,
        power,
        ADDITIVE_CURVE_MODE_EXPONENTIAL,
    )


def _distance_field_opacity_with_mode(
    distance: float,
    falloff: float,
    power: float,
    curve_mode: str,
) -> float:
    normalized = max(distance, 0.0) / max(falloff, 1e-6)
    if curve_mode == ADDITIVE_CURVE_MODE_RATIONAL:
        return 1.0 / (1.0 + normalized ** power)
    return math.exp(-(normalized ** power))


def _distance_field_alpha(
    signed_distance,
    falloff: float,
    power: float,
    curve_mode: str = ADDITIVE_CURVE_MODE_EXPONENTIAL,
    intensity_multiplier: float = 1.0,
):
    import numpy as np

    distance = np.clip(-signed_distance, 0.0, None)
    normalized = distance / max(falloff, 1e-6)
    if curve_mode == ADDITIVE_CURVE_MODE_RATIONAL:
        alpha = 1.0 / (1.0 + np.power(normalized, power, dtype=np.float32))
    else:
        alpha = np.exp(-np.power(normalized, power, dtype=np.float32))
    alpha = np.clip(alpha * float(intensity_multiplier), 0.0, 1.0)
    return np.where(signed_distance < 0.0, alpha, 0.0).astype(np.float32, copy=False)


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


def _distance_field_masks_for_specs(
    geometry: dict,
    specs: list[dict],
    *,
    curve_mode: str = ADDITIVE_CURVE_MODE_EXPONENTIAL,
    intensity_multiplier: float = 1.0,
    signed_distance=None,
) -> list[dict]:
    if signed_distance is None:
        signed_distance = _display_signed_distance_field(geometry)
    masks = []
    for spec in specs:
        alpha = _distance_field_alpha(
            signed_distance,
            spec["falloff"] * geometry["scale"],
            spec["power"],
            curve_mode=curve_mode,
            intensity_multiplier=intensity_multiplier,
        )
        image, payload = _alpha_field_to_image(alpha)
        masks.append({"image": image, "payload": payload, "spec": spec})
    return masks


def _scale_sdf_layer_alpha(alpha: float, intensity_multiplier: float) -> float:
    return min(alpha * intensity_multiplier, 1.0)


def _continuous_glow_pass_specs(
    wide_bloom_profile: str = WIDE_BLOOM_PROFILE_QUEST,
    intensity_multiplier: float = 1.0,
):
    """Procedural additive passes driven from one shared distance field."""
    wide_bloom_spec = _WIDE_BLOOM_PROFILE_SPECS.get(
        wide_bloom_profile,
        _WIDE_BLOOM_PROFILE_SPECS[WIDE_BLOOM_PROFILE_QUEST],
    )
    specs = [
        {
            "name": "core",
            "path_kind": "distance_field",
            "falloff": 5.0,
            "power": 2.1,
            "fill_role": "inner",
            "fill_alpha": _scale_sdf_layer_alpha(0.14, intensity_multiplier),
        },
        {
            "name": "tight_bloom",
            "path_kind": "distance_field",
            "falloff": 11.5,
            "power": 2.6,
            "fill_role": "middle",
            "fill_alpha": _scale_sdf_layer_alpha(0.18, intensity_multiplier),
        },
        {
            "name": "wide_bloom",
            "path_kind": "distance_field",
            "falloff": wide_bloom_spec["falloff"],
            "power": wide_bloom_spec["power"],
            "fill_role": "outer",
            "fill_alpha": _scale_sdf_layer_alpha(0.48, intensity_multiplier),
        },
    ]
    for spec in specs:
        if spec["name"] == "core" and _MECHA_VISOR_DISABLE_CORE_PASS:
            spec["fill_alpha"] = 0.0
        if spec["name"] != "wide_bloom":
            continue
        spec["falloff"] *= _MECHA_VISOR_WIDE_BLOOM_FALLOFF_SCALE
        spec["fill_alpha"] = min(spec["fill_alpha"] * _MECHA_VISOR_WIDE_BLOOM_ALPHA_SCALE, 1.0)
        spec["power"] *= _MECHA_VISOR_WIDE_BLOOM_POWER_SCALE
    return specs


def _continuous_vignette_pass_specs():
    """Procedural subtractive passes driven from the same distance field."""
    return [
        {
            "name": "core",
            "path_kind": "distance_field",
            "falloff": 21.0,
            "power": 0.95,
            "alpha": 1.0,
            "color_scale": 0.0009375,
            "floor_gain": 2.0,
            "peak_gain": 4.0,
        },
        {
            "name": "mid",
            "path_kind": "distance_field",
            "falloff": 42.0,
            "power": 1.05,
            "alpha": 1.0,
            "color_scale": 0.00375,
            "floor_gain": 1.0,
            "peak_gain": 0.7,
        },
        {
            "name": "tail",
            "path_kind": "distance_field",
            "falloff": 60.0,
            "power": 1.15,
            "alpha": 0.9,
            "color_scale": 0.015,
            "floor_gain": 0.75,
            "peak_gain": 0.5,
        },
    ]


def _vignette_pass_opacity(base_opacity: float, amplitude_opacity: float, spec: dict) -> float:
    """Scale each vignette stratum independently without giving up the shared RMS envelope."""
    floor_gain = float(spec.get("floor_gain", 1.0))
    peak_gain = float(spec.get("peak_gain", floor_gain))
    gain = floor_gain + (peak_gain - floor_gain) * min(max(amplitude_opacity, 0.0), 1.0)
    return min(max(base_opacity * gain, 0.0), 1.0)

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


class _MetalWideBloomRenderer:
    def __init__(self, parent_layer, geometry: dict, frame_size: tuple[float, float]):
        import numpy as np

        _load_metal_symbols()
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Metal default device unavailable")

        width_pt, height_pt = frame_size
        self._geometry = geometry
        self._device = device
        self._queue = device.newCommandQueue()
        self._pipeline = _build_metal_pipeline(device)
        self._field_buffer = device.newBufferWithLength_options_(
            int(geometry["pixel_width"] * geometry["pixel_height"] * 4),
            0,
        )
        signed_distance = _display_signed_distance_field(geometry).astype("float32", copy=False)
        _copy_bytes_to_metal_buffer(self._field_buffer, signed_distance.tobytes())
        self._uniform_buffer = device.newBufferWithLength_options_(_MTL_BUFFER_SIZE, 0)
        self._layer = objc.lookUpClass("CAMetalLayer").layer()
        self._layer.setDevice_(device)
        self._layer.setPixelFormat_(_MTL_PIXEL_FORMAT_BGRA8_UNORM)
        self._layer.setFramebufferOnly_(True)
        self._layer.setOpaque_(False)
        self._layer.setFrame_(((0, 0), (width_pt, height_pt)))
        self._layer.setContentsScale_(geometry["scale"])
        self._layer.setDrawableSize_((float(geometry["pixel_width"]), float(geometry["pixel_height"])))
        parent_layer.addSublayer_(self._layer)
        self._base_color = _GLOW_COLOR
        self._additive_mix = 1.0
        self._start_time = time.monotonic()

    def layer(self):
        return self._layer

    def set_base_color(self, base_color: tuple[float, float, float]) -> None:
        self._base_color = base_color

    def set_additive_mix(self, additive_mix: float) -> None:
        self._additive_mix = additive_mix

    def draw_frame(self, now: float | None = None) -> bool:
        drawable = self._layer.nextDrawable()
        if drawable is None:
            return False

        now = time.monotonic() if now is None else now
        t = max(now - self._start_time, 0.0)
        glow_rgbs = _glow_role_rgbs(self._base_color)
        core, tight, wide = _continuous_glow_pass_specs()
        uniforms = struct.pack(
            _MTL_BUFFER_LAYOUT,
            float(self._geometry["pixel_width"]),
            float(self._geometry["pixel_height"]),
            float(t),
            float(self._additive_mix),
            core["falloff"] * self._geometry["scale"],
            core["power"],
            core["fill_alpha"],
            0.0,
            tight["falloff"] * self._geometry["scale"],
            tight["power"],
            tight["fill_alpha"],
            0.0,
            wide["falloff"] * self._geometry["scale"],
            wide["power"],
            wide["fill_alpha"],
            0.0,
            glow_rgbs["inner"][0],
            glow_rgbs["inner"][1],
            glow_rgbs["inner"][2],
            0.0,
            glow_rgbs["middle"][0],
            glow_rgbs["middle"][1],
            glow_rgbs["middle"][2],
            0.0,
            glow_rgbs["outer"][0],
            glow_rgbs["outer"][1],
            glow_rgbs["outer"][2],
            0.0,
        )
        _copy_bytes_to_metal_buffer(self._uniform_buffer, uniforms)

        render_pass_descriptor = objc.lookUpClass("MTLRenderPassDescriptor").renderPassDescriptor()
        attachment = render_pass_descriptor.colorAttachments().objectAtIndexedSubscript_(0)
        attachment.setTexture_(drawable.texture())
        attachment.setLoadAction_(_MTL_LOAD_ACTION_CLEAR)
        attachment.setStoreAction_(_MTL_STORE_ACTION_STORE)
        attachment.setClearColor_((0.0, 0.0, 0.0, 0.0))

        command_buffer = self._queue.commandBuffer()
        encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)
        encoder.setRenderPipelineState_(self._pipeline)
        encoder.setFragmentBuffer_offset_atIndex_(self._uniform_buffer, 0, 0)
        encoder.setFragmentBuffer_offset_atIndex_(self._field_buffer, 0, 1)
        encoder.drawPrimitives_vertexStart_vertexCount_(_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP, 0, 4)
        encoder.endEncoding()
        command_buffer.presentDrawable_(drawable)
        command_buffer.commit()
        return True


def _maybe_create_metal_wide_bloom_renderer(parent_layer, geometry: dict, frame_size: tuple[float, float]):
    if not _METAL_WIDE_BLOOM_ENABLED:
        return None
    if not _metal_device_available():
        logger.warning("Mecha Visor smoke requested but Metal is unavailable; falling back")
        return None
    try:
        return _MetalWideBloomRenderer(parent_layer, geometry, frame_size)
    except Exception:
        logger.exception("Failed to initialize Mecha Visor Metal renderer; falling back")
        return None


class GlowOverlay(NSObject):
    """Manages a screen-border glow window driven by audio amplitude."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(GlowOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._glow_layer: CALayer | None = None
        self._smoothed_amplitude = 0.0
        self._vignette_smoothed_amplitude = 0.0
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
        self._visual_layer_state = None
        self._glow_geometry = None
        self._glow_signed_distance = None
        self._glow_mask_scale = 1.0
        self._active_additive_curve_mode = ADDITIVE_CURVE_MODE_EXPONENTIAL
        self._active_vignette_curve_mode = ADDITIVE_CURVE_MODE_EXPONENTIAL
        self._active_additive_mask_intensity = 1.0
        self._active_wide_bloom_profile = WIDE_BLOOM_PROFILE_QUEST
        self._additive_renderer = None
        self._cpu_additive_fallback_active = False
        self._metal_frame_timer = None
        return self

    def set_visual_layer_state(self, state) -> None:
        old_state = getattr(self, "_visual_layer_state", None)
        if old_state is state:
            return
        if old_state is not None and hasattr(old_state, "remove_listener"):
            old_state.remove_listener(self._on_visual_layer_state_change)
        self._visual_layer_state = state
        if state is not None and hasattr(state, "add_listener"):
            state.add_listener(self._on_visual_layer_state_change)
        self._apply_visual_layer_state()

    def _on_visual_layer_state_change(self, state) -> None:
        self._apply_visual_layer_state()

    def _apply_visual_layer_state(self) -> None:
        self._refresh_glow_masks_if_needed()
        state = getattr(self, "_visual_layer_state", None)
        self._set_additive_render_mode(getattr(self, "_cpu_additive_fallback_active", False))
        if hasattr(self, "_vignette_pass_layers"):
            for layer_id, entry in zip(_VIGNETTE_LAYER_IDS, self._vignette_pass_layers):
                entry["layer"].setHidden_(False if state is None else not state.is_visible(layer_id))
        if getattr(self, "_dim_layer", None) is not None:
            self._dim_layer.setHidden_(
                False if state is None else not state.is_visible(SCREEN_DIMMER_LAYER_ID)
            )
        self._apply_glow_color(self._glow_color)

    def _set_additive_render_mode(self, use_cpu_fallback: bool) -> None:
        state = getattr(self, "_visual_layer_state", None)
        if hasattr(self, "_glow_pass_layers"):
            for layer_id, entry in zip(_GLOW_LAYER_IDS, self._glow_pass_layers):
                entry["layer"].setHidden_(
                    (not use_cpu_fallback)
                    or (False if state is None else not state.is_visible(layer_id))
                )
        renderer = getattr(self, "_additive_renderer", None)
        if renderer is not None and hasattr(renderer, "layer"):
            metal_layer = renderer.layer()
            if metal_layer is not None:
                metal_layer.setHidden_(use_cpu_fallback)
        self._cpu_additive_fallback_active = bool(use_cpu_fallback)

    def _draw_additive_frame(self, now: float | None = None, *, start_timer: bool = False) -> bool:
        renderer = getattr(self, "_additive_renderer", None)
        if renderer is None:
            self._set_additive_render_mode(True)
            return False
        if renderer.draw_frame(now):
            self._set_additive_render_mode(False)
            if start_timer:
                self._start_metal_frame_timer()
            return True
        self._set_additive_render_mode(True)
        self._stop_metal_frame_timer()
        return False

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
        self._glow_geometry = geometry
        self._glow_signed_distance = _display_signed_distance_field(geometry)
        self._glow_mask_scale = mask_scale

        # ── Optional screen dim: subtle dark backdrop for glow contrast ──
        self._dim_layer = None
        if _DIM_SCREEN:
            self._dim_layer = CALayer.alloc().init()
            self._dim_layer.setFrame_(((0, 0), (w, h)))
            self._dim_layer.setBackgroundColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 1.0).CGColor()
            )
            self._dim_layer.setOpacity_(0.0)
            content.layer().addSublayer_(self._dim_layer)

        # ── Container layer: holds shadow + masked fill ──────────
        # We control opacity on this layer to drive the whole effect.
        self._glow_layer = CALayer.alloc().init()
        self._glow_layer.setFrame_(((0, 0), (w, h)))
        self._glow_layer.setOpacity_(0.0)
        self._additive_renderer = _maybe_create_metal_wide_bloom_renderer(
            self._glow_layer,
            geometry,
            (w, h),
        )

        # Procedural additive passes: one continuous field, different falloff curves.
        glow_pass_layers = []
        self._glow_mask_payloads = []
        self._vignette_mask_payloads = []
        self._mask_payloads = []
        curve_mode, intensity_multiplier, wide_bloom_profile = self._current_additive_tuning()
        self._active_additive_curve_mode = curve_mode
        self._active_additive_mask_intensity = intensity_multiplier
        self._active_wide_bloom_profile = wide_bloom_profile
        for entry in _distance_field_masks_for_specs(
            geometry,
            _continuous_glow_pass_specs(
                wide_bloom_profile=wide_bloom_profile,
                intensity_multiplier=intensity_multiplier,
            ),
            curve_mode=curve_mode,
            intensity_multiplier=intensity_multiplier,
            signed_distance=self._glow_signed_distance,
        ):
            spec = entry["spec"]
            layer = CALayer.alloc().init()
            layer.setFrame_(((0, 0), (w, h)))
            mask_layer = CALayer.alloc().init()
            mask_layer.setFrame_(((0, 0), (w, h)))
            mask_layer.setContents_(entry["image"])
            mask_layer.setContentsScale_(mask_scale)
            layer.setMask_(mask_layer)
            self._glow_layer.addSublayer_(layer)
            self._glow_mask_payloads.append(entry["payload"])
            glow_pass_layers.append({"layer": layer, "spec": spec})

        self._glow_pass_layers = glow_pass_layers
        self._shadow_shape = glow_pass_layers[-1]["layer"] if glow_pass_layers else None

        # ── Subtractive vignette: darkened colored edges for light backgrounds ──
        # Same distance-field geometry as the additive glow, with tinted falloff.
        self._vignette_layer = CALayer.alloc().init()
        self._vignette_layer.setFrame_(((0, 0), (w, h)))
        self._vignette_layer.setOpacity_(0.0)

        vignette_pass_layers = []
        for entry in _distance_field_masks_for_specs(
            geometry,
            _continuous_vignette_pass_specs(),
            curve_mode=curve_mode,
            signed_distance=self._glow_signed_distance,
        ):
            spec = entry["spec"]
            layer = CALayer.alloc().init()
            layer.setFrame_(((0, 0), (w, h)))
            mask_layer = CALayer.alloc().init()
            mask_layer.setFrame_(((0, 0), (w, h)))
            mask_layer.setContents_(entry["image"])
            mask_layer.setContentsScale_(mask_scale)
            layer.setMask_(mask_layer)
            self._vignette_layer.addSublayer_(layer)
            self._vignette_mask_payloads.append(entry["payload"])
            vignette_pass_layers.append({"layer": layer, "spec": spec})

        self._mask_payloads = self._glow_mask_payloads + self._vignette_mask_payloads
        self._vignette_pass_layers = vignette_pass_layers
        self._apply_glow_color(_GLOW_COLOR)
        self._set_additive_render_mode(self._additive_renderer is None)
        self._apply_visual_layer_state()
        content.layer().addSublayer_(self._glow_layer)
        content.layer().addSublayer_(self._vignette_layer)
        logger.info("Glow overlay created (%.0fx%.0f, border=%.0f, shadow=%.0f)",
                     w, h, _GLOW_WIDTH, _GLOW_SHADOW_RADIUS)

    def _current_additive_tuning(self) -> tuple[str, float, str]:
        state = getattr(self, "_visual_layer_state", None)
        if state is None:
            return (
                ADDITIVE_CURVE_MODE_EXPONENTIAL,
                1.0,
                WIDE_BLOOM_PROFILE_QUEST,
            )
        curve_mode = (
            state.additive_curve_mode()
            if hasattr(state, "additive_curve_mode")
            else ADDITIVE_CURVE_MODE_EXPONENTIAL
        )
        intensity = (
            state.additive_mask_intensity()
            if hasattr(state, "additive_mask_intensity")
            else 1.0
        )
        profile = (
            state.wide_bloom_profile()
            if hasattr(state, "wide_bloom_profile")
            else WIDE_BLOOM_PROFILE_QUEST
        )
        return (curve_mode, intensity, profile)

    def _refresh_glow_masks_if_needed(self) -> None:
        if not hasattr(self, "_glow_pass_layers") or getattr(self, "_glow_geometry", None) is None:
            return
        curve_mode, intensity_multiplier, wide_bloom_profile = self._current_additive_tuning()
        if (
            curve_mode == self._active_additive_curve_mode
            and curve_mode == self._active_vignette_curve_mode
            and intensity_multiplier == self._active_additive_mask_intensity
            and wide_bloom_profile == self._active_wide_bloom_profile
        ):
            return
        curve_changed = curve_mode != self._active_additive_curve_mode
        additive_changed = (
            curve_changed
            or intensity_multiplier != self._active_additive_mask_intensity
            or wide_bloom_profile != self._active_wide_bloom_profile
        )
        if additive_changed:
            self._active_additive_curve_mode = curve_mode
            self._active_additive_mask_intensity = intensity_multiplier
            self._active_wide_bloom_profile = wide_bloom_profile
            new_payloads = []
            masks = _distance_field_masks_for_specs(
                self._glow_geometry,
                _continuous_glow_pass_specs(
                    wide_bloom_profile=wide_bloom_profile,
                    intensity_multiplier=intensity_multiplier,
                ),
                curve_mode=curve_mode,
                intensity_multiplier=intensity_multiplier,
                signed_distance=self._glow_signed_distance,
            )
            for entry, layer_entry in zip(masks, self._glow_pass_layers):
                layer_entry["spec"] = entry["spec"]
                mask_layer = layer_entry["layer"].mask()
                new_payloads.append(entry["payload"])
                if mask_layer is None:
                    continue
                mask_layer.setContents_(entry["image"])
                mask_layer.setContentsScale_(self._glow_mask_scale)
            if new_payloads:
                self._glow_mask_payloads = new_payloads

        if curve_mode != self._active_vignette_curve_mode:
            self._active_vignette_curve_mode = curve_mode
            new_vignette_payloads = []
            vignette_masks = _distance_field_masks_for_specs(
                self._glow_geometry,
                _continuous_vignette_pass_specs(),
                curve_mode=curve_mode,
                signed_distance=self._glow_signed_distance,
            )
            for entry, layer_entry in zip(vignette_masks, self._vignette_pass_layers):
                layer_entry["spec"] = entry["spec"]
                mask_layer = layer_entry["layer"].mask()
                new_vignette_payloads.append(entry["payload"])
                if mask_layer is None:
                    continue
                mask_layer.setContents_(entry["image"])
                mask_layer.setContentsScale_(self._glow_mask_scale)
            if new_vignette_payloads:
                self._vignette_mask_payloads = new_vignette_payloads

        self._mask_payloads = self._glow_mask_payloads + getattr(self, "_vignette_mask_payloads", [])

    def _apply_glow_color(self, base_color: tuple[float, float, float]) -> None:
        """Push the current glow color through the procedural glow/vignette passes."""
        additive_renderer = getattr(self, "_additive_renderer", None)
        if additive_renderer is not None:
            additive_renderer.set_base_color(base_color)
            if self._visible:
                self._draw_additive_frame(time.monotonic())

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

    def _start_metal_frame_timer(self) -> None:
        if getattr(self, "_additive_renderer", None) is None:
            return
        if getattr(self, "_metal_frame_timer", None) is not None:
            return
        self._metal_frame_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _METAL_FRAME_INTERVAL_S,
            self,
            "metalFrameTick:",
            None,
            True,
        )

    def _stop_metal_frame_timer(self) -> None:
        if getattr(self, "_metal_frame_timer", None) is not None:
            self._metal_frame_timer.invalidate()
            self._metal_frame_timer = None

    def metalFrameTick_(self, timer) -> None:
        if not self._visible:
            return
        self._draw_additive_frame(time.monotonic())

    def show(self) -> None:
        """Fade the glow window in to base opacity."""
        if self._window is None:
            return
        self._cancel_pending_hide()
        self._hide_generation += 1
        self._visible = True
        self._smoothed_amplitude = 0.0
        self._vignette_smoothed_amplitude = 0.0
        self._update_count = 0
        self._noise_floor = 0.0
        self._cap_factor = 1.0
        brightness = _sample_screen_brightness(self._screen)
        self._glow_color, self._glow_base_opacity, self._glow_peak_target = _glow_style_for_brightness(brightness)
        self._apply_glow_color(self._glow_color)
        self._brightness = brightness
        # Cross-fade: additive glow fades out, subtractive vignette fades in
        # as brightness increases. Fully additive at black, fully subtractive
        # at white, blended in between.
        self._additive_mix, self._subtractive_mix = _edge_mix_for_brightness(brightness)
        if getattr(self, "_additive_renderer", None) is not None:
            self._additive_renderer.set_additive_mix(self._additive_mix)
            self._draw_additive_frame(time.monotonic(), start_timer=True)
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
        window_id = None
        if self._window is not None and hasattr(self._window, "windowNumber"):
            try:
                window_id = int(self._window.windowNumber())
            except Exception:
                window_id = None
        new_brightness = _sample_screen_brightness(self._screen, excluding_window_id=window_id)
        if abs(new_brightness - self._brightness) < 0.02:
            return  # no meaningful change
        self._brightness = new_brightness
        new_color, new_base, new_peak = _glow_style_for_brightness(new_brightness)
        self._glow_color = new_color
        self._glow_base_opacity = new_base
        self._glow_peak_target = new_peak
        self._apply_glow_color(new_color)

        # Update cross-fade mix
        self._additive_mix, self._subtractive_mix = _edge_mix_for_brightness(new_brightness)
        if getattr(self, "_additive_renderer", None) is not None:
            self._additive_renderer.set_additive_mix(self._additive_mix)
            self._draw_additive_frame(time.monotonic())

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
        self._stop_metal_frame_timer()
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

        self._update_count += 1

        # Adaptive noise floor — slowly tracks ambient noise level.
        # Rises slowly (adapts to fan noise), falls slowly (doesn't
        # drop to zero between words).
        if rms < self._noise_floor or self._noise_floor == 0.0:
            self._noise_floor += (rms - self._noise_floor) * 0.05  # fast adapt down
        else:
            self._noise_floor += (rms - self._noise_floor) * 0.002  # slow adapt up

        # Subtract floor — only signal above ambient triggers the glow
        signal = max(rms - self._noise_floor, 0.0)

        # Smooth the additive glow: rise fast, decay slow.
        if signal > self._smoothed_amplitude:
            self._smoothed_amplitude += (signal - self._smoothed_amplitude) * _RISE_FACTOR
        else:
            self._smoothed_amplitude *= _DECAY_FACTOR

        # The subtractive stack wants a snappier envelope than the glow.
        vignette_smoothed = getattr(self, "_vignette_smoothed_amplitude", 0.0)
        if signal > vignette_smoothed:
            self._vignette_smoothed_amplitude = vignette_smoothed + (
                signal - vignette_smoothed
            ) * _VIGNETTE_RISE_FACTOR
        else:
            self._vignette_smoothed_amplitude = vignette_smoothed * _VIGNETTE_DECAY_FACTOR

        # Map smoothed amplitude to opacity range [base, max]
        # Fixed multiplier — ceiling is absolute, floor is adaptive
        amplitude_linear = min(self._smoothed_amplitude * _GLOW_MULTIPLIER, 1.0)
        amplitude_linear = _mecha_visor_signal_boost(amplitude_linear)
        # Perceptual correction: log curve so glow tracks perceived loudness.
        # All smoothing math above stays linear; this is the last step
        # before "rendering" — the display gamma, essentially.
        amplitude_opacity = math.log1p(amplitude_linear * 20.0) / math.log1p(20.0)
        opacity = self._glow_base_opacity + amplitude_opacity * (_GLOW_MAX_OPACITY - self._glow_base_opacity)
        opacity = min(opacity, self._glow_peak_target)
        vignette_amplitude_linear = min(self._vignette_smoothed_amplitude * _GLOW_MULTIPLIER, 1.0)
        vignette_amplitude_opacity = math.log1p(vignette_amplitude_linear * 20.0) / math.log1p(20.0)
        vignette_opacity = self._glow_base_opacity + vignette_amplitude_opacity * (
            _GLOW_MAX_OPACITY - self._glow_base_opacity
        )
        vignette_opacity = min(vignette_opacity, self._glow_peak_target)

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
            base_vignette_opacity = vignette_opacity * subtractive_mix * _VIGNETTE_OPACITY_SCALE
            self._vignette_layer.setOpacity_(1.0 if base_vignette_opacity > 0.0 else 0.0)
            if hasattr(self, "_vignette_pass_layers"):
                for entry in self._vignette_pass_layers:
                    entry["layer"].setOpacity_(
                        _vignette_pass_opacity(
                            base_vignette_opacity,
                            vignette_amplitude_opacity,
                            entry["spec"],
                        )
                    )

        # Log first few updates and then periodically to verify pipeline
        if self._update_count <= 3 or self._update_count % 50 == 0:
            logger.info("Glow amplitude: rms=%.4f smoothed=%.4f opacity=%.3f add=%.2f sub=%.2f (update #%d)",
                        rms, self._smoothed_amplitude, opacity, additive_mix, subtractive_mix, self._update_count)
