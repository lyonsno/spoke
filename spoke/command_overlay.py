"""Command response overlay.

A semi-transparent overlay for displaying streamed command responses. Visually
kin to the transcription overlay (same ethereal transparency, same floating
treatment) but differentiated by color and rhythm. The input overlay breathes
with voice amplitude. The output overlay pulses with a steady ease-in/ease-out
rhythm — mechanical but gentle, distinct from the organic input.
"""

from __future__ import annotations

import logging
import math
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
from Quartz import CALayer, CAShapeLayer, CGPathCreateWithRoundedRect

logger = logging.getLogger(__name__)

_OVERLAY_WIDTH = 600.0
_OVERLAY_HEIGHT = 80.0
_OVERLAY_BOTTOM_MARGIN = 300.0  # above the input overlay
_OVERLAY_CORNER_RADIUS = 16.0
_OVERLAY_MAX_HEIGHT = 400.0
_FONT_SIZE = 16.0
_FADE_IN_S = 0.5
_FADE_OUT_S = 0.5  # fast dismiss fade (750ms total with 250ms hold)
_FADE_STEPS = 15

def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default

# Assistant glow: full spectrum rotation with velocity undulation
_COLOR_CYCLE_PERIOD = _env("SPOKE_COMMAND_COLOR_PERIOD", 6.0)  # seconds per full hue rotation
_COLOR_VELOCITY_PERIOD = 7.0  # velocity oscillation period (out of phase with everything)
_COLOR_VELOCITY_MIN = 0.3  # slowest speed multiplier (dwells)
_COLOR_VELOCITY_MAX = 1.7  # fastest speed multiplier (transitions)
_GLOW_COLOR = (0.6, 0.4, 0.9)  # initial color for setup (violet)
_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_TEXT_ALPHA_MIN", 0.35)  # strong visible pulse
_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_TEXT_ALPHA_MAX", 1.0)
_BG_ALPHA = _env("SPOKE_COMMAND_BG_ALPHA", 0.35)
_PULSE_PERIOD = _env("SPOKE_COMMAND_PULSE_PERIOD", 2.0)  # base period (seconds)
_PULSE_PERIOD_USER = _PULSE_PERIOD * 1.5  # user text: 50% slower
_PULSE_PERIOD_ASST = 5.0  # assistant text: slow deep breath
_PULSE_PHASE_OFFSET_USER = 0.3  # user starts 30% ahead in phase
_PULSE_HZ = 30.0  # timer frequency for pulse animation

_OUTER_FEATHER = 40.0
_INNER_GLOW_DEPTH = 30.0
_OUTER_GLOW_PEAK_TARGET = 0.35


class CommandOverlay(NSObject):
    """Overlay for displaying streamed command responses."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(CommandOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._text_view: NSTextView | None = None
        self._scroll_view: NSScrollView | None = None
        self._content_view: NSView | None = None
        self._visible = False
        self._streaming = False
        self._response_text = ""
        self._utterance_text = ""
        self._utterance_label = None  # pinned header for user text

        # Fade state
        self._fade_timer: NSTimer | None = None
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 0

        # Pulse state
        self._pulse_timer: NSTimer | None = None
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0

        # Linger timer
        self._linger_timer: NSTimer | None = None

        # Thinking timer state
        self._thinking_timer: NSTimer | None = None
        self._thinking_seconds = 0.0
        self._thinking_label = None  # NSTextField for the counter
        self._thinking_glow_layer = None  # CALayer for the glow behind the number
        self._thinking_inverted = False  # False = glowing number, True = cutout

        return self

    def setup(self) -> None:
        """Create the command overlay window."""
        screen_frame = self._screen.frame()
        sw = screen_frame.size.width

        f = _OUTER_FEATHER
        x = (sw - _OVERLAY_WIDTH) / 2 - f
        y = _OVERLAY_BOTTOM_MARGIN - f
        win_w = _OVERLAY_WIDTH + 2 * f
        win_h = _OVERLAY_HEIGHT + 2 * f
        frame = NSMakeRect(x, y, win_w, win_h)

        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, 0, NSBackingStoreBuffered, False
        )
        self._window.setLevel_(25)
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(NSColor.clearColor())
        self._window.setIgnoresMouseEvents_(True)
        self._window.setHasShadow_(False)
        self._window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        wrapper_frame = NSMakeRect(0, 0, win_w, win_h)
        wrapper = NSView.alloc().initWithFrame_(wrapper_frame)
        wrapper.setWantsLayer_(True)

        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        content.layer().setMasksToBounds_(True)
        content.layer().setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.1, 0.1, 0.12, _BG_ALPHA).CGColor()
        )

        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(
            _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 1.0
        )

        # Inner glow
        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        margin = _INNER_GLOW_DEPTH + 50
        from Quartz import CGPathCreateMutableCopy, CGPathAddPath, kCAFillRuleEvenOdd as kEO

        lw, lh = w + 2 * margin, h + 2 * margin

        self._inner_shadow = CAShapeLayer.alloc().init()
        self._inner_shadow.setFrame_(((f - margin, f - margin), (lw, lh)))

        outer = CGPathCreateWithRoundedRect(((0, 0), (lw, lh)), 0, 0, None)
        inner = CGPathCreateWithRoundedRect(
            ((margin, margin), (w, h)),
            _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None
        )
        combined = CGPathCreateMutableCopy(outer)
        CGPathAddPath(combined, None, inner)

        self._inner_shadow.setPath_(combined)
        self._inner_shadow.setFillRule_(kEO)
        self._inner_shadow.setFillColor_(glow_nscolor.colorWithAlphaComponent_(0.12).CGColor())
        self._inner_shadow.setShadowColor_(glow_nscolor.CGColor())
        self._inner_shadow.setShadowOffset_((0, 0))
        self._inner_shadow.setShadowRadius_(2.4)
        self._inner_shadow.setShadowOpacity_(0.8)

        inner_mask = CAShapeLayer.alloc().init()
        inner_mask.setFrame_(((0, 0), (lw, lh)))
        inner_mask.setPath_(CGPathCreateWithRoundedRect(
            ((margin, margin), (w, h)),
            _OVERLAY_CORNER_RADIUS, _OVERLAY_CORNER_RADIUS, None
        ))
        self._inner_shadow.setMask_(inner_mask)
        wrapper.layer().addSublayer_(self._inner_shadow)

        # Outer glow layers
        self._outer_glow_tight = CALayer.alloc().init()
        self._outer_glow_tight.setFrame_(((f, f), (w, h)))
        self._outer_glow_tight.setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        self._outer_glow_tight.setBackgroundColor_(
            glow_nscolor.colorWithAlphaComponent_(0.01).CGColor()
        )
        self._outer_glow_tight.setShadowColor_(glow_nscolor.CGColor())
        self._outer_glow_tight.setShadowOffset_((0, 0))
        self._outer_glow_tight.setShadowRadius_(6.2)
        self._outer_glow_tight.setShadowOpacity_(0.2)
        wrapper.layer().insertSublayer_below_(self._outer_glow_tight, content.layer())

        self._outer_glow_wide = CALayer.alloc().init()
        self._outer_glow_wide.setFrame_(((f, f), (w, h)))
        self._outer_glow_wide.setCornerRadius_(_OVERLAY_CORNER_RADIUS)
        self._outer_glow_wide.setBackgroundColor_(
            glow_nscolor.colorWithAlphaComponent_(0.01).CGColor()
        )
        self._outer_glow_wide.setShadowColor_(glow_nscolor.CGColor())
        self._outer_glow_wide.setShadowOffset_((0, 0))
        self._outer_glow_wide.setShadowRadius_(14.0)
        self._outer_glow_wide.setShadowOpacity_(0.4)
        wrapper.layer().insertSublayer_below_(self._outer_glow_wide, self._outer_glow_tight)

        wrapper.addSubview_(content)
        self._content_view = content

        # Scroll view with text view for response text
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

        # Thinking timer label — top-right corner of the content area
        from AppKit import NSTextField, NSTextAlignmentRight
        timer_w, timer_h = 60.0, 24.0
        timer_x = _OVERLAY_WIDTH - timer_w - 12
        timer_y = _OVERLAY_HEIGHT - timer_h - 6
        self._thinking_label = NSTextField.alloc().initWithFrame_(
            NSMakeRect(timer_x, timer_y, timer_w, timer_h)
        )
        self._thinking_label.setEditable_(False)
        self._thinking_label.setSelectable_(False)
        self._thinking_label.setBezeled_(False)
        self._thinking_label.setDrawsBackground_(False)
        self._thinking_label.setAlignment_(NSTextAlignmentRight)
        self._thinking_label.setFont_(
            NSFont.monospacedDigitSystemFontOfSize_weight_(13.0, 0.2)
        )
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.0
            )
        )
        self._thinking_label.setStringValue_("")
        self._thinking_label.setHidden_(True)
        content.addSubview_(self._thinking_label)

        self._window.setContentView_(wrapper)
        self._window.setAlphaValue_(0.0)

        logger.info("Command overlay created")

    # ── public interface ────────────────────────────────────

    def show(self) -> None:
        """Fade the overlay in and start the pulse."""
        if self._window is None:
            return
        self._cancel_all_timers()
        self._visible = True
        self._streaming = True
        self._response_text = ""
        self._utterance_text = ""
        self._text_view.setString_("")
        self._window.setAlphaValue_(0.0)

        # Reset geometry
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
        self._scroll_view.setFrame_(
            NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, _OVERLAY_HEIGHT - 16)
        )

        self._window.orderFrontRegardless()

        # Fade in
        self._fade_step = 0
        self._fade_from = 0.0
        self._fade_direction = 1
        interval = _FADE_IN_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

        # Start pulse animation
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0
        self._pulse_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _PULSE_HZ, self, "pulseStep:", None, True
        )

        # Start thinking timer (glowing number mode)
        self._start_thinking_timer()

    def cancel_dismiss(self) -> None:
        """Pseudo-haptic dismiss: 250ms hold at bright → 500ms fade off.

        Total: 750ms from spacebar hold to gone. The user taps space,
        the overlay brightens briefly (confirmation), then snaps away.
        """
        if self._window is None:
            return
        self._cancel_all_timers()
        self._streaming = False
        self._visible = True  # keep visible for the animation

        self._cancel_step = 0
        self._cancel_phase = "hold"  # hold → fade
        # Brighten to full immediately in the current hue color
        self._window.setAlphaValue_(1.0)
        if self._text_view is not None:
            r, g, b = self._current_hue_rgb()
            self._text_view.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)
            )
        if hasattr(self, '_inner_shadow'):
            self._inner_shadow.setShadowOpacity_(1.0)
        self._cancel_timer_anim = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 30.0, self, "_cancelAnimStep:", None, True
        )

    def _cancelAnimStep_(self, timer) -> None:
        """Animate the dismiss sequence: hold → fade."""
        self._cancel_step += 1

        if self._cancel_phase == "hold":
            # Hold at full brightness for 250ms (~8 steps at 30fps)
            if self._cancel_step >= 8:
                self._cancel_step = 0
                self._cancel_phase = "fade"

        elif self._cancel_phase == "fade":
            # Fade to zero over 500ms (~15 steps at 30fps)
            progress = min(self._cancel_step / 15.0, 1.0)
            eased = progress * progress
            alpha = 1.0 - eased
            self._window.setAlphaValue_(alpha)
            if self._cancel_step >= 8:
                if self._cancel_timer_anim is not None:
                    self._cancel_timer_anim.invalidate()
                    self._cancel_timer_anim = None
                self._window.setAlphaValue_(0.0)
                self._window.orderOut_(None)
                self._visible = False
                self._cancel_pulse()

    def hide(self) -> None:
        """Fade out — pulse continues during fade for visual continuity."""
        if self._window is None:
            return
        self._visible = False
        self._streaming = False
        self._cancel_linger()
        # Don't cancel pulse here — let it continue during fade-out.
        # It will be cancelled when the fade completes (window ordered out).
        self._start_fade_out()

    def set_utterance(self, text: str) -> None:
        """Show the user's utterance in the text view at reduced opacity."""
        self._utterance_text = text
        if self._text_view is None or not self._visible:
            return
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
        )
        attr_str = NSMutableAttributedString.alloc().initWithString_(text)
        attr_str.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.4),
            (0, len(text)),
        )
        self._text_view.textStorage().setAttributedString_(attr_str)
        self._update_layout()
        glow.setShadowBlurRadius_(3.0)
        attr_str.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(text))
        )
        self._text_view.textStorage().setAttributedString_(attr_str)
        self._update_layout()

    def append_token(self, token: str) -> None:
        """Append a streamed response token."""
        if self._text_view is None or not self._visible:
            return
        first_token = len(self._response_text) == 0
        self._response_text += token

        if first_token and self._utterance_text:
            # Add separator between utterance and response
            from AppKit import (
                NSMutableAttributedString,
                NSForegroundColorAttributeName,
            )
            sep = NSMutableAttributedString.alloc().initWithString_("\n\n")
            sep.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0),
                (0, 2),
            )
            self._text_view.textStorage().appendAttributedString_(sep)

        frag = self._make_response_fragment(token)
        self._text_view.textStorage().appendAttributedString_(frag)

        self._update_layout()

    def _current_hue_rgb(self):
        """Get the current hue rotation color as (r, g, b)."""
        hue = getattr(self, '_color_phase', 0.0)
        s, v = 0.75, 0.95
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            return (c + m, x + m, m)
        elif h6 < 2:
            return (x + m, c + m, m)
        elif h6 < 3:
            return (m, c + m, x + m)
        elif h6 < 4:
            return (m, x + m, c + m)
        elif h6 < 5:
            return (x + m, m, c + m)
        else:
            return (c + m, m, x + m)

    def _make_response_fragment(self, token: str):
        """Create an attributed string fragment for a response token.

        Text color matches the current hue rotation. Glow matches too.
        """
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSFontAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        r, g, b = self._current_hue_rgb()
        frag = NSMutableAttributedString.alloc().initWithString_(token)
        response_color = NSColor.colorWithSRGBRed_green_blue_alpha_(
            r, g, b, _TEXT_ALPHA_MAX
        )
        frag.addAttribute_value_range_(
            NSForegroundColorAttributeName, response_color, (0, len(token))
        )
        frag.addAttribute_value_range_(
            NSFontAttributeName,
            NSFont.systemFontOfSize_weight_(_FONT_SIZE, 0.0),
            (0, len(token)),
        )
        # Text glow in the current hue color
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 0.6)
        )
        glow.setShadowOffset_((0, 0))
        glow.setShadowBlurRadius_(3.0)
        frag.addAttribute_value_range_(
            NSShadowAttributeName, glow, (0, len(token))
        )
        return frag

    def finish(self) -> None:
        """Called when the response stream is complete.

        The overlay stays visible indefinitely — the user dismisses it
        by holding spacebar (which triggers cancel_dismiss via __main__).
        No linger timer. The user is in control.
        """
        self._streaming = False
        self._stop_thinking_timer()
        # Pulse keeps running — the overlay is alive until dismissed

    # ── animation ───────────────────────────────────────────

    def fadeStep_(self, timer) -> None:
        """One step of the fade animation.

        During fade-out, the utterance text fades at double rate so it
        disappears before the assistant response.
        """
        self._fade_step += 1
        progress = self._fade_step / _FADE_STEPS

        if self._fade_direction == 1:
            eased = progress * progress
            alpha = eased
        else:
            eased = progress * progress
            alpha = self._fade_from * (1.0 - eased)

            # Fade utterance at double rate
            if self._text_view is not None and self._utterance_text:
                from AppKit import NSForegroundColorAttributeName
                utt_progress = min(progress * 2.0, 1.0)
                utt_alpha = 0.35 * (1.0 - utt_progress * utt_progress)
                utt_len = min(len(self._utterance_text), self._text_view.textStorage().length() if hasattr(self._text_view.textStorage(), 'length') else 0)
                if utt_len > 0:
                    try:
                        self._text_view.textStorage().addAttribute_value_range_(
                            NSForegroundColorAttributeName,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, utt_alpha),
                            (0, utt_len),
                        )
                    except Exception:
                        pass

        self._window.setAlphaValue_(alpha)

        if self._fade_step >= _FADE_STEPS:
            self._cancel_fade()
            if self._fade_direction == -1:
                self._window.setAlphaValue_(0.0)
                self._window.orderOut_(None)
                self._cancel_pulse()  # now kill the pulse
            else:
                self._window.setAlphaValue_(1.0)

    def pulseStep_(self, timer) -> None:
        logger.info("PULSE TICK hue=%.3f", getattr(self, '_color_phase', -1))
        """Dual-phase pulse: user and assistant text breathe independently.

        Assistant text: faster period (0.8x base), double-smoothstep for
        extra-aggressive easing, violet-amber color oscillation.
        User text: slower period (1.5x base), single smoothstep, blue
        color shift, phase-offset by 0.3 and diverging naturally.
        """
        if self._text_view is None:
            return
        dt = 1.0 / _PULSE_HZ

        # Advance phases
        self._pulse_phase_asst += dt / _PULSE_PERIOD_ASST
        if self._pulse_phase_asst > 1.0:
            self._pulse_phase_asst -= 1.0
        self._pulse_phase_user += dt / _PULSE_PERIOD_USER
        if self._pulse_phase_user > 1.0:
            self._pulse_phase_user -= 1.0

        # Assistant: one slow breath, 5 seconds. Asymmetric:
        # ease-out toward bright (lingers near peak), ease-in back to dim
        # (dips quickly). Raw sine → squared so it spends more time high.
        raw_breath = 0.5 * (1.0 + math.cos(2.0 * math.pi * self._pulse_phase_asst))
        breath = raw_breath * raw_breath  # squared: lingers near 1.0, dips briefly
        alpha_a = 0.40 + 0.25 * breath  # lower overall opacity, subtle pulse

        # User: raw sine → single smoothstep (same aggressiveness as before)
        raw_u = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._pulse_phase_user))
        pulse_u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
        utt_alpha = 0.2 + 0.25 * pulse_u

        # Full spectrum hue rotation with velocity undulation
        # The speed varies sinusoidally so it dwells in some colors and
        # glides through others — always moving, never truly paused.
        self._color_velocity_phase += dt / _COLOR_VELOCITY_PERIOD
        if self._color_velocity_phase > 1.0:
            self._color_velocity_phase -= 1.0
        vel_raw = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._color_velocity_phase))
        vel = _COLOR_VELOCITY_MIN + vel_raw * (_COLOR_VELOCITY_MAX - _COLOR_VELOCITY_MIN)
        self._color_phase += (dt / _COLOR_CYCLE_PERIOD) * vel
        if self._color_phase > 1.0:
            self._color_phase -= 1.0
        hue = self._color_phase
        # Log color phase every ~1s (every 30th tick)
        if not hasattr(self, '_color_log_counter'):
            self._color_log_counter = 0
        self._color_log_counter += 1
        if self._color_log_counter % 30 == 0:
            logger.info("Color phase: %.3f hue, vel_phase=%.3f", hue, self._color_velocity_phase)
        s, v = 0.75, 0.95  # vivid but not neon
        c = v * s
        x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
        m = v - c
        h6 = hue * 6.0
        if h6 < 1:
            r, g, b = c + m, x + m, m
        elif h6 < 2:
            r, g, b = x + m, c + m, m
        elif h6 < 3:
            r, g, b = m, c + m, x + m
        elif h6 < 4:
            r, g, b = m, x + m, c + m
        elif h6 < 5:
            r, g, b = x + m, m, c + m
        else:
            r, g, b = c + m, m, x + m

        # Update text colors per-range
        if self._text_view is not None:
            from AppKit import NSForegroundColorAttributeName as _FG_pulse
            ts = self._text_view.textStorage()
            total_len = ts.length() if hasattr(ts, 'length') else 0
            if total_len > 0 and self._utterance_text:
                utt_len = min(len(self._utterance_text), total_len)
                # User text: white with blue tint, breathing
                try:
                    ur = 0.85 + 0.15 * pulse_u
                    ug = 0.9 + 0.1 * pulse_u
                    ts.addAttribute_value_range_(
                        _FG_pulse,
                        NSColor.colorWithSRGBRed_green_blue_alpha_(ur, ug, 1.0, utt_alpha),
                        (0, utt_len),
                    )
                except Exception:
                    pass
                # Response text: current hue color
                resp_start = utt_len + 2
                if resp_start < total_len:
                    try:
                        ts.addAttribute_value_range_(
                            _FG_pulse,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha_a),
                            (resp_start, total_len - resp_start),
                        )
                    except Exception:
                        pass
            elif total_len > 0:
                self._text_view.setTextColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha_a)
                )

        # Pulse the glow with assistant phase oscillating color
        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)
        glow_opacity = 0.5 + 0.3 * pulse_a
        if hasattr(self, '_inner_shadow'):
            self._inner_shadow.setShadowColor_(glow_nscolor.CGColor())
            self._inner_shadow.setFillColor_(
                glow_nscolor.colorWithAlphaComponent_(0.12).CGColor()
            )
            self._inner_shadow.setShadowOpacity_(glow_opacity)
        if hasattr(self, '_outer_glow_tight'):
            self._outer_glow_tight.setShadowColor_(glow_nscolor.CGColor())
            self._outer_glow_tight.setShadowOpacity_(min(glow_opacity * 0.4, _OUTER_GLOW_PEAK_TARGET))
        if hasattr(self, '_outer_glow_wide'):
            self._outer_glow_wide.setShadowColor_(glow_nscolor.CGColor())
            self._outer_glow_wide.setShadowOpacity_(min(glow_opacity * 0.6, _OUTER_GLOW_PEAK_TARGET))

    def lingerDone_(self, timer) -> None:
        """Linger period over — fade out."""
        self._linger_timer = None
        if self._visible and not self._streaming:
            self.hide()

    # ── fade helpers ────────────────────────────────────────

    def _start_fade_out(self) -> None:
        self._cancel_fade()
        self._fade_step = 0
        self._fade_from = self._window.alphaValue() if self._window else 1.0
        self._fade_direction = -1
        interval = _FADE_OUT_S / _FADE_STEPS
        self._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval, self, "fadeStep:", None, True
        )

    def _cancel_fade(self) -> None:
        if self._fade_timer is not None:
            self._fade_timer.invalidate()
            self._fade_timer = None

    def _cancel_pulse(self) -> None:
        if self._pulse_timer is not None:
            self._pulse_timer.invalidate()
            self._pulse_timer = None

    def _cancel_linger(self) -> None:
        if self._linger_timer is not None:
            self._linger_timer.invalidate()
            self._linger_timer = None

    def _cancel_all_timers(self) -> None:
        self._cancel_fade()
        self._cancel_pulse()
        self._cancel_linger()
        self._stop_thinking_timer()

    # ── thinking timer ──────────────────────────────────────

    def _start_thinking_timer(self) -> None:
        """Start the thinking counter in glowing-number mode."""
        self._thinking_seconds = 0.0
        self._thinking_inverted = False
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(False)
            self._thinking_label.setStringValue_("0.0s")
            # Glowing number: violet text on transparent background
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.7
                )
            )
        logger.info("Thinking timer started")
        self._thinking_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "thinkingTick:", None, True
        )

    def invert_thinking_timer(self) -> None:
        """Switch from glowing number to negative-space cutout mode.

        Called when the first content token arrives — the visual flip
        signals that the model has transitioned from thinking to speaking.
        """
        self._thinking_inverted = True
        if self._thinking_label is not None:
            # Cutout mode: dark text that punches through the glow
            # Using very low alpha dark color to create the negative-space effect
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.05, 0.05, 0.06, 0.7)
            )

    def _stop_thinking_timer(self) -> None:
        """Stop and fade the thinking counter."""
        if self._thinking_timer is not None:
            self._thinking_timer.invalidate()
            self._thinking_timer = None
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(True)

    def thinkingTick_(self, timer) -> None:
        """Update the thinking counter every 100ms."""
        self._thinking_seconds += 0.1
        if self._thinking_label is not None and not self._thinking_label.isHidden():
            self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")

    # ── layout ──────────────────────────────────────────────

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
                self._scroll_view.setFrame_(
                    NSMakeRect(12, 8, _OVERLAY_WIDTH - 24, new_height - 16)
                )

            end = (self._text_view.string().length()
                   if hasattr(self._text_view.string(), 'length')
                   else len(self._response_text))
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            logger.debug("Command overlay layout update failed", exc_info=True)
