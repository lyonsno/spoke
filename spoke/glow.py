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
_GLOW_BASE_OPACITY = 0.0966  # 140% of the calmer baseline so the border keeps dancing at rest
_GLOW_PEAK_TARGET = 0.1904
_GLOW_BASE_OPACITY_DARK = 0.0826
_GLOW_BASE_OPACITY_LIGHT = 0.2744
_GLOW_PEAK_TARGET_DARK = 0.168
_GLOW_PEAK_TARGET_LIGHT = _GLOW_MAX_OPACITY
_EDGE_INNER_SATURATION_SCALE = 0.70
_EDGE_OUTER_SATURATION_SCALE = 1.80
# MacBook Pro 14"/16" (2021+) has asymmetric display corners.
# We use slightly tighter radii than the physical bezel so the glow
# source stays close to the corners — the bezel hides the overshoot.
_CORNER_RADIUS_TOP = 10.0  # slightly tighter than physical ~18pt to fill corners
_CORNER_RADIUS_BOTTOM = 6.0  # slightly tighter than physical ~10pt

_GLOW_MULTIPLIER = float(os.environ.get("SPOKE_GLOW_MULTIPLIER", "21.4"))
_DIM_SCREEN = os.environ.get("SPOKE_DIM_SCREEN", "1") == "1"
_DIM_OPACITY_DARK = 0.28  # dim on dark backgrounds
_DIM_OPACITY_LIGHT = 0.424  # bright scenes move 20% closer to fully opaque
# Amplitude smoothing: rise fast, decay slow
_RISE_FACTOR = 0.95  # faster attack — vignette snaps to voice
_DECAY_FACTOR = 0.50  # very quick falloff between words

# Fade timing
_FADE_IN_S = 0.08
_FADE_OUT_S = 0.2
_GLOW_SHOW_FADE_S = 0.2
_GLOW_HIDE_FADE_S = 0.6
_GLOW_SHOW_TIMING = "easeIn"
_DIM_SHOW_FADE_S = 1.08
_DIM_HIDE_FADE_S = 2.4
_WINDOW_TEARDOWN_CUSHION_S = 0.05


_BRIGHTNESS_SAMPLE_INTERVAL = 1.0  # seconds between recurring samples
_BRIGHTNESS_PATCH_SIZE = 50  # pixels per patch side
_DISTANCE_FIELD_SCALE_DEFAULT = 2.0
_NOTCH_BOTTOM_RADIUS = 8.0
_NOTCH_SHOULDER_SMOOTHING = 9.5
_LIGHT_BACKGROUND_EDGE_BOOST = 0.664
_VIGNETTE_OPACITY_SCALE = 3.05  # back to original


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


def _edge_mix_for_brightness(brightness: float) -> tuple[float, float]:
    """Cross-fade additive glow into subtractive edge treatment with a slight bright-scene boost."""
    t = min(max(brightness, 0.0), 1.0)
    additive_mix = 1.0 - t
    subtractive_mix = t * (1.0 + _LIGHT_BACKGROUND_EDGE_BOOST * t)
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
    geometry = {
        "pixel_width": max(int(round(width_pt * scale)), 1),
        "pixel_height": max(int(round(height_pt * scale)), 1),
        "top_radius": _CORNER_RADIUS_TOP * scale,
        "bottom_radius": _CORNER_RADIUS_BOTTOM * scale,
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
    inside = np.minimum(np.maximum(qx, qy), 0.0)
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
    normalized = max(distance, 0.0) / max(falloff, 1e-6)
    return math.exp(-(normalized ** power))


def _distance_field_alpha(signed_distance, falloff: float, power: float):
    import numpy as np

    distance = np.clip(-signed_distance, 0.0, None)
    alpha = np.exp(-np.power(distance / max(falloff, 1e-6), power, dtype=np.float32))
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


def _distance_field_masks_for_specs(geometry: dict, specs: list[dict]) -> list[dict]:
    signed_distance = _display_signed_distance_field(geometry)
    masks = []
    for spec in specs:
        alpha = _distance_field_alpha(
            signed_distance,
            spec["falloff"] * geometry["scale"],
            spec["power"],
        )
        image, payload = _alpha_field_to_image(alpha)
        masks.append({"image": image, "payload": payload, "spec": spec})
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

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(GlowOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
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

        # Procedural additive passes: one continuous field, different falloff curves.
        glow_pass_layers = []
        self._mask_payloads = []
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
        brightness = _sample_screen_brightness(self._screen)
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
            dim_target = _DIM_OPACITY_DARK + brightness * (_DIM_OPACITY_LIGHT - _DIM_OPACITY_DARK)
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
        new_brightness = _sample_screen_brightness(self._screen)
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

        # Smoothly adjust dim opacity
        if self._dim_layer is not None:
            dim_target = _DIM_OPACITY_DARK + new_brightness * (_DIM_OPACITY_LIGHT - _DIM_OPACITY_DARK)
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
