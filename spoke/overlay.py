"""Frosted transcription overlay.

A semi-transparent, system-font overlay at the bottom of the screen that shows
live transcription text during recording. Fades in when recording starts,
updates as preview transcriptions arrive (with typewriter effect), and fades
out after final injection. Text opacity breathes with voice amplitude.
"""

from __future__ import annotations

import logging
import os

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
from Foundation import NSMakeRect, NSObject, NSTimer
from Quartz import CAGradientLayer, CALayer, CAShapeLayer, CGPathCreateWithRoundedRect, CGAffineTransformIdentity

logger = logging.getLogger(__name__)

_OVERLAY_WIDTH = 600.0
_OVERLAY_HEIGHT = 80.0
_OVERLAY_BOTTOM_MARGIN = 120.0  # distance from bottom of screen
_OVERLAY_CORNER_RADIUS = 16.0
_OVERLAY_MAX_HEIGHT = 300.0  # max before text scrolls
_FONT_SIZE = 16.0
_FADE_IN_S = 0.75  # slow ease-in — overlay materializes gradually
_FADE_OUT_S = 0.35
_FADE_STEPS = 12  # number of steps for manual fade animation
_TYPEWRITER_INTERVAL = 0.03  # seconds between characters (~33 chars/sec)
def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default

_TEXT_ALPHA_MIN = _env("SPOKE_TEXT_ALPHA_MIN", 0.066)
_TEXT_ALPHA_MAX = _env("SPOKE_TEXT_ALPHA_MAX", 1.00)
_TEXT_AMP_SATURATION = _env("SPOKE_TEXT_AMP_SATURATION", 0.10)
_BG_ALPHA_MIN = _env("SPOKE_BG_ALPHA_MIN", 0.105)
_BG_ALPHA_MAX = _env("SPOKE_BG_ALPHA_MAX", 0.6)
_BG_AMP_SATURATION = _env("SPOKE_BG_AMP_SATURATION", 0.17)
_SMOOTH_RISE = _env("SPOKE_SMOOTH_RISE", 0.115)
_SMOOTH_DECAY = _env("SPOKE_SMOOTH_DECAY", 0.94)

# Inner glow — matches screen border glow, scaled to overlay size
_GLOW_COLOR = (0.7, 0.92, 0.95)  # same as screen glow
_INNER_GLOW_WIDTH = 3.0  # proportional to overlay vs screen size
_INNER_GLOW_DEPTH = 30.0  # gradient extends inward — diffuse
_OUTER_FEATHER = 40.0  # glow bleed past overlay edge (must contain shadow radius)


class TranscriptionOverlay(NSObject):
    """Manages a frosted overlay window for live transcription preview."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(TranscriptionOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._text_view: NSTextView | None = None
        self._scroll_view: NSScrollView | None = None
        self._visible = False
        self._fade_timer: NSTimer | None = None
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 0

        # Typewriter state
        self._typewriter_timer: NSTimer | None = None
        self._typewriter_target = ""  # full text we're typing toward
        self._typewriter_displayed = ""  # what's currently shown

        # Text breathing — separate heavy smoothing so text doesn't flicker
        self._text_amplitude = 0.0
        return self

    def setup(self) -> None:
        """Create the overlay window."""
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width

        # Window is oversized by _OUTER_FEATHER on each side for the feather bleed
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        y = _OVERLAY_BOTTOM_MARGIN - f
        win_w = _OVERLAY_WIDTH + 2 * f
        win_h = _OVERLAY_HEIGHT + 2 * f
        frame = NSMakeRect(x, y, win_w, win_h)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(25)  # above other windows
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Wrapper view — unclipped, holds both the dark box and the outer feather
        wrapper_frame = NSMakeRect(0, 0, win_w, win_h)
        wrapper = NSView.alloc().initWithFrame_(wrapper_frame)
        wrapper.setWantsLayer_(True)

        # Semi-transparent dark background with rounded corners (inset by feather)
        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        content.layer().setMasksToBounds_(True)
        content.layer().setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.1, 0.1, 0.12, _BG_ALPHA_MIN).CGColor()
        )

        # Glow color setup
        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(
            _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 1.0
        )
        clear_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0)

        # Inner glow — inward shadow via even-odd cutout on wrapper
        # Paths are in the layer's local coordinate space (origin 0,0).
        # The layer is sized to w+2*margin, h+2*margin and positioned so
        # the "hole" aligns with the content view.
        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        margin = _INNER_GLOW_DEPTH + 50
        from Quartz import CGPathCreateMutableCopy, CGPathAddPath, kCAFillRuleEvenOdd as kEO

        lw, lh = w + 2 * margin, h + 2 * margin  # layer size

        self._inner_shadow = CAShapeLayer.alloc().init()
        self._inner_shadow.setFrame_(((f - margin, f - margin), (lw, lh)))

        # All paths in layer-local coords: (0,0) is top-left of the layer
        outer = CGPathCreateWithRoundedRect(
            ((0, 0), (lw, lh)), 0, 0, None
        )
        inner = CGPathCreateWithRoundedRect(
            ((margin, margin), (w, h)),
            _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None
        )
        combined = CGPathCreateMutableCopy(outer)
        CGPathAddPath(combined, None, inner)

        self._inner_shadow.setPath_(combined)
        self._inner_shadow.setFillRule_(kEO)
        self._inner_shadow.setFillColor_(glow_nscolor.colorWithAlphaComponent_(0.15).CGColor())
        self._inner_shadow.setShadowColor_(glow_nscolor.CGColor())
        self._inner_shadow.setShadowOffset_((0, 0))
        self._inner_shadow.setShadowRadius_(2.4)  # very tight — sharp exponential feel
        self._inner_shadow.setShadowOpacity_(1.0)

        # Mask: only show inside the overlay's rounded rect
        inner_mask = CAShapeLayer.alloc().init()
        inner_mask.setFrame_(((0, 0), (lw, lh)))
        inner_mask.setPath_(CGPathCreateWithRoundedRect(
            ((margin, margin), (w, h)),
            _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None
        ))
        self._inner_shadow.setMask_(inner_mask)

        wrapper.layer().addSublayer_(self._inner_shadow)

        # Outer feather — two stacked shadows for a dramatic smeared glow
        # Layer 1: tight, bright — defines the edge
        self._outer_glow_tight = CALayer.alloc().init()
        self._outer_glow_tight.setFrame_(((f, f), (w, h)))
        self._outer_glow_tight.setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        self._outer_glow_tight.setBackgroundColor_(
            glow_nscolor.colorWithAlphaComponent_(0.01).CGColor()
        )
        self._outer_glow_tight.setShadowColor_(glow_nscolor.CGColor())
        self._outer_glow_tight.setShadowOffset_((0, 0))
        self._outer_glow_tight.setShadowRadius_(6.2)
        self._outer_glow_tight.setShadowOpacity_(0.3)
        wrapper.layer().insertSublayer_below_(self._outer_glow_tight, content.layer())

        # Layer 2: wide, diffuse — the smear
        self._outer_glow_wide = CALayer.alloc().init()
        self._outer_glow_wide.setFrame_(((f, f), (w, h)))
        self._outer_glow_wide.setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        self._outer_glow_wide.setBackgroundColor_(
            glow_nscolor.colorWithAlphaComponent_(0.01).CGColor()
        )
        self._outer_glow_wide.setShadowColor_(glow_nscolor.CGColor())
        self._outer_glow_wide.setShadowOffset_((0, 0))
        self._outer_glow_wide.setShadowRadius_(14.0)
        self._outer_glow_wide.setShadowOpacity_(0.5)
        wrapper.layer().insertSublayer_below_(self._outer_glow_wide, self._outer_glow_tight)

        wrapper.addSubview_(content)
        self._content_view = content

        # Scroll view with text view for scrollable transcription text
        scroll_frame = NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._scroll_view = NSScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setBorderType_(0)
        self._scroll_view.setAutoresizingMask_(18)

        text_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
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
        self._window.setContentView_(wrapper)

        self._window.setAlphaValue_(0.0)

        logger.info("Transcription overlay created")

    def show(self) -> None:
        """Fade the overlay in."""
        if self._window is None:
            return
        self._cancel_fade()
        self._cancel_typewriter()
        self._visible = True
        self._typewriter_target = ""
        self._typewriter_displayed = ""
        self._typewriter_hwm = 0  # furthest position typewriter has reached
        self._text_view.setString_("")
        self._window.setAlphaValue_(0.0)

        # Reset to default size (window includes feather margin)
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN - f,
                       _OVERLAY_WIDTH + 2 * f, _OVERLAY_HEIGHT + 2 * f),
            True, False
        )
        self._content_view.setFrame_(
            NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        )
        scroll_frame = NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._scroll_view.setFrame_(scroll_frame)
        self._reset_text_geometry(_OVERLAY_HEIGHT - 16, scroll_to_top=True)

        self._window.orderFrontRegardless()

        # Fade in using stepped timer
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_target = 1.0
        self._fade_direction = 1  # fading in
        interval = _FADE_IN_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )
        logger.info("Overlay show")

    def hide(self) -> None:
        """Fade the overlay out smoothly."""
        if self._window is None:
            return
        self._visible = False
        self._cancel_typewriter()
        self._start_fade_out()
        logger.info("Overlay hide")

    def _start_fade_out(self) -> None:
        """Animate fade-out using a repeating timer for smooth steps."""
        self._cancel_fade()
        self._fade_step = 0
        self._fade_from = self._window.alphaValue()
        self._fade_target = 0.0
        self._fade_direction = -1  # fading out
        interval = _FADE_OUT_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation."""
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            # Fade in: ease-in (slow start, confident finish)
            eased = progress * progress
            alpha = eased
        else:
            # Fade out: ease-in (slow start, fast end)
            eased = progress * progress
            alpha = self._fade_from * (1.0 - eased)

        self._window.setAlphaValue_(alpha)

        if self._fade_step >= _FADE_STEPS:
            self._cancel_fade()
            if self._fade_direction == -1:
                self._window.setAlphaValue_(0.0)
                self._window.orderOut_(None)
            else:
                self._window.setAlphaValue_(1.0)

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            self._fade_timer.invalidate()
            self._fade_timer = None

    # ── typewriter effect ────────────────────────────────────

    def set_text(self, text: str) -> None:
        """Update the target text — typewriter effect types toward it."""
        if self._text_view is None or not self._visible:
            return

        self._typewriter_target = text

        # If the new text doesn't start with what we've displayed,
        # the transcription revised earlier words.
        if not text.startswith(self._typewriter_displayed):
            # Find divergence point
            common = 0
            for i, (a, b) in enumerate(zip(self._typewriter_displayed, text)):
                if a == b:
                    common = i + 1
                else:
                    break

            # Allow small jitter (punctuation, capitalization) near the
            # typing frontier without triggering a full snap.
            _FUZZ = 3  # chars of slack behind the high-water mark
            if common < self._typewriter_hwm - _FUZZ:
                # Divergence is well behind the high-water mark — the user
                # already saw those characters typewrite in.  Snap the full
                # text instantly so we never re-animate already-seen content.
                self._cancel_typewriter()
                self._typewriter_displayed = text
                self._typewriter_hwm = len(text)
                self._text_view.setString_(text)
                self._update_layout()
                return
            else:
                self._typewriter_displayed = text[:common]
                self._text_view.setString_(self._typewriter_displayed)

        # Start typing if not already
        if self._typewriter_timer is None and len(self._typewriter_displayed) < len(self._typewriter_target):
            self._typewriter_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                _TYPEWRITER_INTERVAL, self, "typewriterStep:", None, True
            )

    def typewriterStep_(self, timer) -> None:
        """Append one character toward the target text."""
        if len(self._typewriter_displayed) < len(self._typewriter_target):
            self._typewriter_displayed = self._typewriter_target[:len(self._typewriter_displayed) + 1]
            self._typewriter_hwm = max(self._typewriter_hwm, len(self._typewriter_displayed))
            self._text_view.setString_(self._typewriter_displayed)
            self._update_layout()
        else:
            self._cancel_typewriter()

    def _cancel_typewriter(self) -> None:
        if self._typewriter_timer is not None:
            self._typewriter_timer.invalidate()
            self._typewriter_timer = None

    # ── amplitude-reactive text ──────────────────────────────

    def update_text_amplitude(self, amplitude: float) -> None:
        """Update text opacity based on voice amplitude (0.0–1.0).

        Must be called on the main thread. Uses heavy smoothing so the
        text breathes slowly rather than flickering at 62Hz.
        """
        if self._text_view is None or not self._visible:
            return

        # Heavy smoothing — rise slow, decay slow. Text should breathe,
        # not flicker. Rise 0.15, decay 0.92 gives ~1s response time.
        if amplitude > self._text_amplitude:
            self._text_amplitude += (amplitude - self._text_amplitude) * _SMOOTH_RISE
        else:
            self._text_amplitude *= _SMOOTH_DECAY

        scaled = min(self._text_amplitude / _TEXT_AMP_SATURATION, 1.0)

        text_alpha = _TEXT_ALPHA_MIN + scaled * (_TEXT_ALPHA_MAX - _TEXT_ALPHA_MIN)
        self._text_view.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, text_alpha)
        )

    def update_glow_amplitude(self, opacity: float, cap_factor: float = 1.0) -> None:
        """Update inner and outer glow opacity to match the screen glow.

        opacity should be the screen glow's current opacity (0.0–1.0).
        cap_factor scales the glow down during the recording cap countdown
        (1.0 = full, ramps toward 0.25 near the cap).
        """
        if not self._visible:
            return
        # Apply recording-cap countdown scaling
        if cap_factor < 1.0:
            cap_floor = 0.25
            scale = cap_floor + (1.0 - cap_floor) * cap_factor
            opacity *= scale
        if hasattr(self, '_inner_shadow'):
            self._inner_shadow.setShadowOpacity_(opacity)
        if hasattr(self, '_outer_glow_tight'):
            self._outer_glow_tight.setShadowOpacity_(opacity * 0.5)
        if hasattr(self, '_outer_glow_wide'):
            self._outer_glow_wide.setShadowOpacity_(opacity * 0.8)

    # ── layout helpers ───────────────────────────────────────

    def _reset_text_geometry(self, visible_height: float, scroll_to_top: bool = False) -> None:
        """Keep the document view and clip view in sync with the current overlay size."""
        if self._text_view is None or self._scroll_view is None:
            return

        doc_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 24, visible_height)
        self._text_view.setFrame_(doc_frame)

        container = self._text_view.textContainer()
        if container is not None and hasattr(container, "setContainerSize_"):
            container.setContainerSize_((_OVERLAY_WIDTH - 24, 1.0e7))

        clip_view = self._scroll_view.contentView() if hasattr(self._scroll_view, "contentView") else None
        if clip_view is not None and scroll_to_top:
            if hasattr(clip_view, "scrollToPoint_"):
                clip_view.scrollToPoint_((0, 0))
            elif hasattr(clip_view, "setBoundsOrigin_"):
                clip_view.setBoundsOrigin_((0, 0))
            if hasattr(self._scroll_view, "reflectScrolledClipView_"):
                self._scroll_view.reflectScrolledClipView_(clip_view)

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
                text_height = _OVERLAY_HEIGHT - 16

            new_height = min(max(_OVERLAY_HEIGHT, text_height + 24), _OVERLAY_MAX_HEIGHT)

            f = _OUTER_FEATHER
            win_frame = self._window.frame()
            new_win_h = new_height + 2 * f
            if abs(win_frame.size.height - new_win_h) > 4:
                win_frame.size.height = new_win_h
                self._window.setFrame_display_animate_(win_frame, True, False)
                self._content_view.setFrame_(
                    NSMakeRect(f, f, _OVERLAY_WIDTH, new_height)
                )
                self._scroll_view.setFrame_(NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, new_height - 16))
                # Rebuild inner shadow for new height
                if hasattr(self, '_inner_shadow'):
                    w, h = _OVERLAY_WIDTH, new_height
                    margin = _INNER_GLOW_DEPTH + 50
                    lw, lh = w + 2 * margin, h + 2 * margin
                    from Quartz import CGPathCreateMutableCopy, CGPathAddPath
                    self._inner_shadow.setFrame_(((f - margin, f - margin), (lw, lh)))
                    outer = CGPathCreateWithRoundedRect(
                        ((0, 0), (lw, lh)), 0, 0, None)
                    inner = CGPathCreateWithRoundedRect(
                        ((margin, margin), (w, h)),
                        _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None)
                    combined = CGPathCreateMutableCopy(outer)
                    CGPathAddPath(combined, None, inner)
                    self._inner_shadow.setPath_(combined)
                    # Update mask too
                    mask = self._inner_shadow.mask()
                    if mask:
                        mask.setFrame_(((0, 0), (lw, lh)))
                        mask.setPath_(CGPathCreateWithRoundedRect(
                            ((margin, margin), (w, h)),
                            _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None))
                # Resize outer feather shadows
                if hasattr(self, '_outer_glow_tight'):
                    self._outer_glow_tight.setFrame_(((f, f), (w, new_height)))
                if hasattr(self, '_outer_glow_wide'):
                    self._outer_glow_wide.setFrame_(((f, f), (w, new_height)))

            self._reset_text_geometry(max(new_height - 16, text_height))
            end = self._text_view.string().length() if hasattr(self._text_view.string(), 'length') else len(self._typewriter_displayed)
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            pass
