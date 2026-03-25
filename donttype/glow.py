"""Screen-border glow overlay that pulses with voice amplitude.

A borderless, transparent, click-through NSWindow that draws a soft glow
around the screen edges. Intensity follows the RMS amplitude of the
microphone input, with fast rise and slow decay for a breathing effect.
"""

from __future__ import annotations

import logging
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
from Foundation import NSObject
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

# Glow appearance
_GLOW_COLOR = (0.7, 0.92, 0.95)  # pale turquoise-white blue RGB
_GLOW_WIDTH = 10.0  # thinner source — less intrusion into screen
_GLOW_SHADOW_RADIUS = 30.0  # tighter bloom — stays near the edge
_GLOW_MAX_OPACITY = 1.0  # full brightness at peak to compensate for smaller size
_GLOW_BASE_OPACITY = 0.10  # clear presence in silence
# MacBook Pro 14"/16" (2021+) has asymmetric display corners.
# We use slightly tighter radii than the physical bezel so the glow
# source stays close to the corners — the bezel hides the overshoot.
_CORNER_RADIUS_TOP = 10.0  # slightly tighter than physical ~18pt to fill corners
_CORNER_RADIUS_BOTTOM = 6.0  # slightly tighter than physical ~10pt

# Amplitude smoothing: rise fast, decay slow
_RISE_FACTOR = 0.90  # near-instant response to voice
_DECAY_FACTOR = 0.50  # very quick falloff between words

# Fade timing
_FADE_IN_S = 0.08
_FADE_OUT_S = 0.2


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
        return self

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

        glow_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 1.0
        )

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
        shadow_shape.setFillColor_(glow_color.CGColor())

        # Shadow bloom — the main visual
        shadow_shape.setShadowColor_(glow_color.CGColor())
        shadow_shape.setShadowOffset_((0, 0))
        shadow_shape.setShadowRadius_(_GLOW_SHADOW_RADIUS)
        shadow_shape.setShadowOpacity_(1.0)

        self._glow_layer.addSublayer_(shadow_shape)

        # Shape fill nearly transparent — just enough for CA to cast shadow
        shadow_shape.setFillColor_(
            glow_color.colorWithAlphaComponent_(0.05).CGColor()
        )

        # Add 4 gradient layers for the visible feathered edge glow.
        # Exponential-style falloff: bright at edge, drops fast, long subtle tail.
        # Use NSColor objects (not CGColor) — PyObjC bridges these correctly
        # for CAGradientLayer, unlike raw CGColorRef pointers.
        edge_nscolor = glow_color
        mid_nscolor = glow_color.colorWithAlphaComponent_(0.25)
        faint_nscolor = glow_color.colorWithAlphaComponent_(0.06)
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

        content.layer().addSublayer_(self._glow_layer)
        logger.info("Glow overlay created (%.0fx%.0f, border=%.0f, shadow=%.0f)",
                     w, h, _GLOW_WIDTH, _GLOW_SHADOW_RADIUS)

    def show(self) -> None:
        """Fade the glow window in to base opacity."""
        if self._window is None:
            return
        self._visible = True
        self._smoothed_amplitude = 0.0
        self._update_count = 0
        self._noise_floor = 0.0
        self._fade_in_until = time.monotonic() + 0.2  # let fade-in finish undisturbed
        self._window.orderFrontRegardless()

        anim = CABasicAnimation.animationWithKeyPath_("opacity")
        anim.setFromValue_(0.0)
        anim.setToValue_(_GLOW_BASE_OPACITY)
        anim.setDuration_(0.2)
        anim.setTimingFunction_(
            CAMediaTimingFunction.functionWithName_("easeOut")
        )
        self._glow_layer.setOpacity_(_GLOW_BASE_OPACITY)
        self._glow_layer.addAnimation_forKey_(anim, "fadeIn")
        logger.info("Glow show")

    def hide(self) -> None:
        """Fade the glow window out smoothly."""
        if self._window is None:
            return
        self._visible = False
        logger.info("Glow hide (received %d amplitude updates)", self._update_count)

        try:
            current = self._glow_layer.opacity()
            anim = CABasicAnimation.animationWithKeyPath_("opacity")
            anim.setFromValue_(current)
            anim.setToValue_(0.0)
            anim.setDuration_(0.6)
            anim.setTimingFunction_(
                CAMediaTimingFunction.functionWithName_("easeIn")
            )
            self._glow_layer.setOpacity_(0.0)
            self._glow_layer.addAnimation_forKey_(anim, "fadeOut")

            # Order out after animation completes
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.65, self, "hideWindowAfterFade:", None, False
            )
        except Exception:
            # If animation fails, just snap off
            logger.exception("Fade-out animation failed, snapping off")
            self._glow_layer.setOpacity_(0.0)
            self._window.orderOut_(None)

    def hideWindowAfterFade_(self, timer) -> None:
        """Remove window after fade-out animation completes."""
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
        amplitude_opacity = self._smoothed_amplitude * 50.0
        opacity = _GLOW_BASE_OPACITY + min(amplitude_opacity, 1.0) * (_GLOW_MAX_OPACITY - _GLOW_BASE_OPACITY)

        self._glow_layer.setOpacity_(opacity)

        # Log first few updates and then periodically to verify pipeline
        if self._update_count <= 3 or self._update_count % 50 == 0:
            logger.info("Glow amplitude: rms=%.4f smoothed=%.4f opacity=%.3f (update #%d)",
                        rms, self._smoothed_amplitude, opacity, self._update_count)
