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
    NSBezierPath,
    NSColor,
    NSScreen,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
)
from Foundation import NSObject, NSTimer
from Quartz import (
    CABasicAnimation,
    CAGradientLayer,
    CALayer,
    CAMediaTimingFunction,
    CAShapeLayer,
    CGPathCreateWithRoundedRect,
    kCAFillRuleEvenOdd,
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
_GLOW_BASE_OPACITY = 0.069  # clear presence in silence
_GLOW_PEAK_TARGET = 0.119
_GLOW_BASE_OPACITY_DARK = 0.059
_GLOW_BASE_OPACITY_LIGHT = 0.196
_GLOW_PEAK_TARGET_DARK = 0.105
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
_DIM_OPACITY_DARK = 0.20  # dim on dark backgrounds
_DIM_OPACITY_LIGHT = 0.52  # dim on light/white backgrounds
# Amplitude smoothing: rise fast, decay slow
_RISE_FACTOR = 0.90  # near-instant response to voice
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


def _sample_screen_brightness(screen) -> float:
    """Sample average brightness of the screen content below our window.

    Returns 0.0 (black) to 1.0 (white). Captures once per call — intended
    to be called at recording start, not per-frame.
    """
    try:
        from Quartz import (
            CGWindowListCreateImage,
            kCGWindowListOptionOnScreenBelowWindow,
            kCGNullWindowID,
            CGRectNull,
        )
        frame = screen.frame()
        rect = ((frame.origin.x, frame.origin.y),
                (frame.size.width, frame.size.height))

        # Capture everything below all windows at our level
        image = CGWindowListCreateImage(
            rect,
            kCGWindowListOptionOnScreenBelowWindow,
            kCGNullWindowID,
            0,  # kCGWindowImageDefault
        )
        if image is None:
            return 0.5  # fallback to mid-brightness

        from Quartz import CGImageGetWidth, CGImageGetHeight, CGImageGetDataProvider, CGDataProviderCopyData
        w = CGImageGetWidth(image)
        h = CGImageGetHeight(image)
        data = CGDataProviderCopyData(CGImageGetDataProvider(image))

        if data is None or len(data) == 0:
            return 0.5

        # Sample a grid of pixels rather than reading every pixel
        import struct
        total = 0.0
        samples = 0
        bytes_per_pixel = len(data) // (w * h) if w * h > 0 else 4
        step_x = max(w // 20, 1)
        step_y = max(h // 20, 1)

        for sy in range(0, h, step_y):
            for sx in range(0, w, step_x):
                offset = (sy * w + sx) * bytes_per_pixel
                if offset + 3 <= len(data):
                    # BGRA or RGBA — either way, first 3 bytes are color channels
                    r, g, b = data[offset], data[offset + 1], data[offset + 2]
                    # Perceived luminance
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


def _edge_band_colors(
    base_color: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Derive tighter and wider edge bands from the current base glow color."""
    inner = _scale_color_saturation(base_color, _EDGE_INNER_SATURATION_SCALE)
    middle = base_color
    outer = _scale_color_saturation(base_color, _EDGE_OUTER_SATURATION_SCALE)
    return inner, middle, outer


def _rounded_rect_path(x, y, w, h, top_radius, bottom_radius):
    """Create a CGPath rounded rect with different top and bottom corner radii."""
    from Quartz import (
        CGPathCreateMutable,
        CGPathMoveToPoint,
        CGPathAddArcToPoint,
        CGPathCloseSubpath,
    )
    # macOS coordinate system: y=0 is bottom
    path = CGPathCreateMutable()
    tr = top_radius
    br = bottom_radius

    # Start at bottom-left, just above the bottom-left corner
    CGPathMoveToPoint(path, None, x, y + br)
    # Bottom-left corner
    CGPathAddArcToPoint(path, None, x, y, x + br, y, br)
    # Bottom edge to bottom-right corner
    CGPathAddArcToPoint(path, None, x + w, y, x + w, y + br, br)
    # Right edge to top-right corner
    CGPathAddArcToPoint(path, None, x + w, y + h, x + w - tr, y + h, tr)
    # Top edge to top-left corner
    CGPathAddArcToPoint(path, None, x, y + h, x, y + h - tr, tr)
    CGPathCloseSubpath(path)
    return path


class GlowOverlay(NSObject):
    """Manages a screen-border glow window driven by audio amplitude."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(GlowOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._glow_layer: CAShapeLayer | None = None
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

        inner_rgb, middle_rgb, outer_rgb = _edge_band_colors(_GLOW_COLOR)
        inner_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            inner_rgb[0], inner_rgb[1], inner_rgb[2], 1.0
        )
        middle_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            middle_rgb[0], middle_rgb[1], middle_rgb[2], 1.0
        )
        outer_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            outer_rgb[0], outer_rgb[1], outer_rgb[2], 1.0
        )

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

        # ── Shadow-casting shape: thick border, full opacity ─────
        # This layer is a child that casts the soft bloom shadow.
        # Its own fill is hidden by the mask below — only its shadow is visible.
        shadow_shape = CAShapeLayer.alloc().init()

        outer = _rounded_rect_path(0, 0, w, h,
                                   _CORNER_RADIUS_TOP, _CORNER_RADIUS_BOTTOM)
        inner = _rounded_rect_path(_GLOW_WIDTH, _GLOW_WIDTH,
                                   w - 2 * _GLOW_WIDTH, h - 2 * _GLOW_WIDTH,
                                   max(_CORNER_RADIUS_TOP - _GLOW_WIDTH, 0),
                                   max(_CORNER_RADIUS_BOTTOM - _GLOW_WIDTH, 0))

        from Quartz import CGPathCreateMutableCopy, CGPathAddPath
        combined = CGPathCreateMutableCopy(outer)
        CGPathAddPath(combined, None, inner)

        shadow_shape.setPath_(combined)
        shadow_shape.setFillRule_(kCAFillRuleEvenOdd)
        shadow_shape.setFillColor_(inner_glow.CGColor())

        # Shadow bloom — the main visual
        shadow_shape.setShadowColor_(outer_glow.CGColor())
        shadow_shape.setShadowOffset_((0, 0))
        shadow_shape.setShadowRadius_(_GLOW_SHADOW_RADIUS)
        shadow_shape.setShadowOpacity_(1.0)

        self._glow_layer.addSublayer_(shadow_shape)
        self._shadow_shape = shadow_shape

        # Shape fill nearly transparent — just enough for CA to cast shadow
        shadow_shape.setFillColor_(
            inner_glow.colorWithAlphaComponent_(0.05).CGColor()
        )

        # Add 4 gradient layers for the visible feathered edge glow.
        # Exponential-style falloff: bright at edge, drops fast, long subtle tail.
        # Use NSColor objects (not CGColor) — PyObjC bridges these correctly
        # for CAGradientLayer, unlike raw CGColorRef pointers.
        edge_nscolor = inner_glow
        mid_nscolor = middle_glow.colorWithAlphaComponent_(0.25)
        faint_nscolor = outer_glow.colorWithAlphaComponent_(0.06)
        clear_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0)

        # CAGradientLayer wants CGColorRef — extract via id bridge
        colors = [
            edge_nscolor.CGColor(),
            mid_nscolor.CGColor(),
            faint_nscolor.CGColor(),
            clear_nscolor.CGColor(),
        ]
        locations = [0.0, 0.15, 0.4, 1.0]

        grad_depth = _GLOW_WIDTH * 4  # gradient extends 4x border width

        # Create a rounded-rect mask path so gradients follow screen curvature
        mask_path = _rounded_rect_path(0, 0, w, h,
                                       _CORNER_RADIUS_TOP, _CORNER_RADIUS_BOTTOM)

        edges = [
            # (origin, size, start_point, end_point)
            ((0, 0), (w, grad_depth), (0.5, 0.0), (0.5, 1.0)),          # bottom
            ((0, h - grad_depth), (w, grad_depth), (0.5, 1.0), (0.5, 0.0)),  # top
            ((0, 0), (grad_depth, h), (0.0, 0.5), (1.0, 0.5)),          # left
            ((w - grad_depth, 0), (grad_depth, h), (1.0, 0.5), (0.0, 0.5)),  # right
        ]
        for origin, size, start, end in edges:
            g = CAGradientLayer.alloc().init()
            g.setFrame_((origin, size))
            g.setColors_(colors)
            g.setLocations_(locations)
            g.setStartPoint_(start)
            g.setEndPoint_(end)

            # Mask to rounded screen shape so corners follow the bezel curve
            mask = CAShapeLayer.alloc().init()
            # Offset the mask path to account for the gradient layer's origin
            mask.setFrame_(((0, 0), (w, h)))
            mask.setPosition_((-origin[0] + w / 2, -origin[1] + h / 2))
            mask.setPath_(mask_path)
            g.setMask_(mask)

            self._glow_layer.addSublayer_(g)

        self._gradient_layers = [self._glow_layer.sublayers()[i] for i in range(1, 5)]

        # ── Subtractive vignette: darkened colored edges for light backgrounds ──
        # Same shape as the additive glow but uses dark-tinted colors.
        # Cross-faded with the additive glow based on background brightness.
        self._vignette_layer = CALayer.alloc().init()
        self._vignette_layer.setFrame_(((0, 0), (w, h)))
        self._vignette_layer.setOpacity_(0.0)

        # Vignette gradients: dark at edges, clear toward center
        vig_edge = NSColor.colorWithSRGBRed_green_blue_alpha_(0.02, 0.02, 0.04, 0.85)
        vig_mid = NSColor.colorWithSRGBRed_green_blue_alpha_(0.03, 0.03, 0.06, 0.45)
        vig_faint = NSColor.colorWithSRGBRed_green_blue_alpha_(0.04, 0.04, 0.08, 0.12)
        vig_clear = NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0)
        vig_colors = [
            vig_edge.CGColor(), vig_mid.CGColor(),
            vig_faint.CGColor(), vig_clear.CGColor(),
        ]

        for origin, size, start, end in edges:
            g = CAGradientLayer.alloc().init()
            g.setFrame_((origin, size))
            g.setColors_(vig_colors)
            g.setLocations_(locations)
            g.setStartPoint_(start)
            g.setEndPoint_(end)

            mask = CAShapeLayer.alloc().init()
            mask.setFrame_(((0, 0), (w, h)))
            mask.setPosition_((-origin[0] + w / 2, -origin[1] + h / 2))
            mask.setPath_(mask_path)
            g.setMask_(mask)

            self._vignette_layer.addSublayer_(g)

        self._vignette_gradient_layers = [self._vignette_layer.sublayers()[i] for i in range(4)]
        content.layer().addSublayer_(self._glow_layer)
        content.layer().addSublayer_(self._vignette_layer)
        logger.info("Glow overlay created (%.0fx%.0f, border=%.0f, shadow=%.0f)",
                     w, h, _GLOW_WIDTH, _GLOW_SHADOW_RADIUS)

    def _apply_glow_color(self, base_color: tuple[float, float, float]) -> None:
        """Push the current glow color through the bloom and gradient layers."""
        inner_rgb, middle_rgb, outer_rgb = _edge_band_colors(base_color)
        inner_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            inner_rgb[0], inner_rgb[1], inner_rgb[2], 1.0
        )
        middle_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            middle_rgb[0], middle_rgb[1], middle_rgb[2], 1.0
        )
        outer_glow = NSColor.colorWithSRGBRed_green_blue_alpha_(
            outer_rgb[0], outer_rgb[1], outer_rgb[2], 1.0
        )
        if hasattr(self, '_shadow_shape'):
            self._shadow_shape.setShadowColor_(outer_glow.CGColor())
            self._shadow_shape.setFillColor_(
                inner_glow.colorWithAlphaComponent_(0.05).CGColor()
            )
        if hasattr(self, '_gradient_layers'):
            edge = inner_glow
            mid = middle_glow.colorWithAlphaComponent_(0.25)
            faint = outer_glow.colorWithAlphaComponent_(0.06)
            clear = NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0)
            colors = [edge.CGColor(), mid.CGColor(), faint.CGColor(), clear.CGColor()]
            for gl in self._gradient_layers:
                gl.setColors_(colors)
        # Tint vignette with current hue — dark version of the glow color
        if hasattr(self, '_vignette_gradient_layers'):
            r, g, b = base_color
            # Dark tinted versions: multiply color by low factor for darkening
            vig_edge = NSColor.colorWithSRGBRed_green_blue_alpha_(r * 0.08, g * 0.08, b * 0.08, 0.85)
            vig_mid = NSColor.colorWithSRGBRed_green_blue_alpha_(r * 0.10, g * 0.10, b * 0.10, 0.45)
            vig_faint = NSColor.colorWithSRGBRed_green_blue_alpha_(r * 0.12, g * 0.12, b * 0.12, 0.12)
            vig_clear = NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0)
            vig_colors = [vig_edge.CGColor(), vig_mid.CGColor(), vig_faint.CGColor(), vig_clear.CGColor()]
            for gl in self._vignette_gradient_layers:
                gl.setColors_(vig_colors)

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
        self._additive_mix = 1.0 - brightness
        self._subtractive_mix = brightness
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

        logger.info("Glow show")

    def hide(self) -> None:
        """Fade the glow window out smoothly."""
        if self._window is None:
            return
        self._cancel_pending_hide()
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
            self._vignette_layer.setOpacity_(opacity * subtractive_mix * 2.5)

        # Log first few updates and then periodically to verify pipeline
        if self._update_count <= 3 or self._update_count % 50 == 0:
            logger.info("Glow amplitude: rms=%.4f smoothed=%.4f opacity=%.3f add=%.2f sub=%.2f (update #%d)",
                        rms, self._smoothed_amplitude, opacity, additive_mix, subtractive_mix, self._update_count)
