"""Frosted transcription overlay.

A semi-transparent, system-font overlay at the bottom of the screen that shows
live transcription text during recording. Fades in when recording starts,
updates as preview transcriptions arrive, and fades out after final injection.
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
_FADE_IN_S = 0.15
_FADE_OUT_S = 0.35
_FADE_STEPS = 12  # number of steps for manual fade animation


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
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.1, 0.1, 0.12, 0.75).CGColor()
        )

        # Scroll view with text view for scrollable transcription text
        scroll_frame = NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._scroll_view = NSScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setHasVerticalScroller_(False)  # hidden but functional
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setBorderType_(0)  # no border
        self._scroll_view.setAutoresizingMask_(18)  # width + height

        text_frame = NSMakeRect(0, 0, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        self._text_view = NSTextView.alloc().initWithFrame_(text_frame)
        self._text_view.setEditable_(False)
        self._text_view.setSelectable_(False)
        self._text_view.setDrawsBackground_(False)
        self._text_view.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.9)
        )
        self._text_view.setFont_(NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0))
        self._text_view.setString_("")
        # Allow text to wrap at the scroll view width
        self._text_view.textContainer().setWidthTracksTextView_(True)
        self._text_view.setHorizontallyResizable_(False)
        self._text_view.setVerticallyResizable_(True)

        self._scroll_view.setDocumentView_(self._text_view)
        content.addSubview_(self._scroll_view)
        self._window.setContentView_(content)

        # Start fully transparent
        self._window.setAlphaValue_(0.0)

        logger.info("Transcription overlay created")

    def show(self) -> None:
        """Fade the overlay in."""
        if self._window is None:
            return
        self._cancel_fade()
        self._visible = True
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

        # Fade in using stepped timer (matches fade-out approach)
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_target = 1.0
        self._fade_direction = 1  # 1 = fading in
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
        self._start_fade_out()
        logger.info("Overlay hide")

    def _start_fade_out(self) -> None:
        """Animate fade-out using a repeating timer for smooth steps."""
        self._cancel_fade()
        self._fade_step = 0
        self._fade_from = self._window.alphaValue()
        self._fade_target = 0.0
        self._fade_direction = -1  # -1 = fading out
        interval = _FADE_OUT_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation."""
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            # Fade in: ease-out (fast start, slow end)
            eased = 1.0 - (1.0 - progress) * (1.0 - progress)
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

    def set_text(self, text: str) -> None:
        """Update the displayed transcription text, grow window, and scroll."""
        if self._text_view is None or not self._visible:
            return

        self._text_view.setString_(text)

        try:
            # Measure text height
            layout = self._text_view.layoutManager()
            container = self._text_view.textContainer()
            if layout and container:
                layout.ensureLayoutForTextContainer_(container)
                text_rect = layout.usedRectForTextContainer_(container)
                text_height = text_rect.size.height
            else:
                text_height = _OVERLAY_HEIGHT - 16

            # Grow window upward to fit text, capped at max height.
            # macOS origin is bottom-left, so increasing height grows upward.
            new_height = min(max(_OVERLAY_HEIGHT, text_height + 24), _OVERLAY_MAX_HEIGHT)

            frame = self._window.frame()
            if abs(frame.size.height - new_height) > 4:
                frame.size.height = new_height
                self._window.setFrame_display_animate_(frame, True, False)
                self._scroll_view.setFrame_(
                    NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, new_height - 16)
                )

            # Scroll to bottom so latest text is always visible
            end = self._text_view.string().length() if hasattr(self._text_view.string(), 'length') else len(text)
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            pass  # layout/resize is best-effort
