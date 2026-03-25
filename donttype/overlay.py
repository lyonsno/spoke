"""Frosted transcription overlay.

A semi-transparent, system-font overlay at the bottom of the screen that shows
live transcription text during recording. Fades in when recording starts,
updates as preview transcriptions arrive (with typewriter effect), and fades
out after final injection. Text opacity breathes with voice amplitude.
"""

from __future__ import annotations

import logging

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
_TYPEWRITER_INTERVAL = 0.02  # seconds between characters (~50 chars/sec)
_TEXT_ALPHA_MIN = 0.45  # text opacity floor (silence)
_TEXT_ALPHA_MAX = 1.00  # text opacity ceiling (loud speech)
_TEXT_AMP_SATURATION = 0.5  # amplitude at which text reaches full brightness


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

        # Center horizontally, fixed near bottom
        x = (sw - _OVERLAY_WIDTH) / 2
        y = _OVERLAY_BOTTOM_MARGIN
        frame = NSMakeRect(x, y, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(25)  # above other windows
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(True)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Semi-transparent dark background with rounded corners
        content_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        content.layer().setMasksToBounds_(True)
        content.layer().setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.1, 0.1, 0.12, 0.55).CGColor()
        )

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
        self._window.setContentView_(content)

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
        self._text_view.setString_("")
        self._window.setAlphaValue_(0.0)

        # Reset to default size
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width
        x = (sw - _OVERLAY_WIDTH) / 2
        self._window.setFrame_display_animate_(
            NSMakeRect(x, _OVERLAY_BOTTOM_MARGIN, _OVERLAY_WIDTH, _OVERLAY_HEIGHT),
            True, False
        )
        self._scroll_view.setFrame_(
            NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        )

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
        # the transcription revised earlier words — snap to common prefix
        if not text.startswith(self._typewriter_displayed):
            # Find common prefix
            common = 0
            for i, (a, b) in enumerate(zip(self._typewriter_displayed, text)):
                if a == b:
                    common = i + 1
                else:
                    break
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
            self._text_amplitude += (amplitude - self._text_amplitude) * 0.15
        else:
            self._text_amplitude *= 0.92

        scaled = min(self._text_amplitude / _TEXT_AMP_SATURATION, 1.0)
        alpha = _TEXT_ALPHA_MIN + scaled * (_TEXT_ALPHA_MAX - _TEXT_ALPHA_MIN)
        self._text_view.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, alpha)
        )

    # ── layout helpers ───────────────────────────────────────

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

            frame = self._window.frame()
            if abs(frame.size.height - new_height) > 4:
                frame.size.height = new_height
                self._window.setFrame_display_animate_(frame, True, False)
                self._scroll_view.setFrame_(
                    NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, new_height - 16)
                )

            end = self._text_view.string().length() if hasattr(self._text_view.string(), 'length') else len(self._typewriter_displayed)
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            pass
