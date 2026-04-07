"""Command response overlay.

A semi-transparent overlay for displaying streamed command responses. Visually
kin to the transcription overlay (same ethereal transparency, same floating
treatment) but differentiated by color and rhythm. The input overlay breathes
with voice amplitude. The output overlay pulses with a slow ease-in/ease-out
rhythm — mechanical but gentle, distinct from the organic input. Color is a
slow full-spectrum hue rotation (~6s cycle with velocity undulation), so the
overlay always appears to be roughly one color but you can never quite pin
down which one.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable

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

def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


_OVERLAY_WIDTH = 600.0
_OVERLAY_HEIGHT = 80.0
_OVERLAY_BOTTOM_MARGIN = _env("SPOKE_COMMAND_OVERLAY_BOTTOM_MARGIN", 300.0)
_OVERLAY_TOP_MARGIN = _env("SPOKE_COMMAND_OVERLAY_TOP_MARGIN", 140.0)
_OVERLAY_CORNER_RADIUS = 16.0
_FONT_SIZE = 16.0
_FADE_IN_S = 0.7
_ENTRANCE_POP_SCALE = 1.015  # ~1mm overshoot on a 600px overlay
_ENTRANCE_POP_S = 0.15
_FADE_OUT_S = 0.5  # fast dismiss fade (750ms total with 250ms hold)
_FADE_STEPS = 15
_DISMISS_DURATION_S = 0.2
_DISMISS_GROW_S = 0.06
_DISMISS_SHRINK_S = _DISMISS_DURATION_S - _DISMISS_GROW_S
_DISMISS_ANIM_FPS = 60.0
_DISMISS_GROW_SCALE = 1.018
_DISMISS_END_SCALE = 0.94

def _max_overlay_height(screen_height: float) -> float:
    return max(_OVERLAY_HEIGHT, screen_height - _OVERLAY_BOTTOM_MARGIN - _OVERLAY_TOP_MARGIN)

# Assistant glow: full spectrum rotation with velocity undulation
_COLOR_CYCLE_PERIOD = _env("SPOKE_COMMAND_COLOR_PERIOD", 6.0)  # seconds per full hue rotation
_COLOR_VELOCITY_PERIOD = 7.0  # velocity oscillation period (out of phase with everything)
_COLOR_VELOCITY_MIN = 0.3  # slowest speed multiplier (dwells)
_COLOR_VELOCITY_MAX = 1.7  # fastest speed multiplier (transitions)
_GLOW_COLOR = (0.6, 0.4, 0.9)  # initial color for setup (violet)
_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_TEXT_ALPHA_MIN", 0.35)  # strong visible pulse
_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_TEXT_ALPHA_MAX", 1.0)
_ASSISTANT_TEXT_ALPHA_MIN = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MIN", 0.75)
_ASSISTANT_TEXT_ALPHA_MAX = _env("SPOKE_COMMAND_ASSISTANT_TEXT_ALPHA_MAX", 1.0)
_BG_ALPHA = _env("SPOKE_COMMAND_BG_ALPHA", 0.715)
_PULSE_PERIOD = _env("SPOKE_COMMAND_PULSE_PERIOD", 2.0)  # base period (seconds)
_PULSE_PERIOD_USER = _PULSE_PERIOD * 1.5  # user text: 50% slower
_PULSE_PERIOD_ASST = 5.0  # assistant text: slow deep breath
_PULSE_PHASE_OFFSET_USER = 0.3  # user starts 30% ahead in phase
_PULSE_HZ = 30.0  # timer frequency for pulse animation

_OUTER_FEATHER = 220.0  # match preview overlay — room for the stretched-exp tails
_INNER_GLOW_DEPTH = 30.0
_OUTER_GLOW_PEAK_TARGET = 0.35
_BRIGHTNESS_CHASE = 0.08

# Adaptive compositing for command output.
_USER_TEXT_COLOR_DARK = (0.92, 0.95, 1.0)
_USER_TEXT_COLOR_LIGHT = (0.10, 0.12, 0.16)
_RESPONSE_TEXT_LIGHT_BG_TARGET = (0.07, 0.08, 0.11)
_THINKING_CUTOUT_DARK = (0.05, 0.05, 0.06)
_THINKING_CUTOUT_LIGHT = (0.80, 0.80, 0.78)

# Radar-sweep spinner — tucked into the bottom-right corner
_SPINNER_PERIOD = 2.0  # seconds per full revolution
_SPINNER_RADIUS = 12.0  # points — smaller, corner detail
_SPINNER_MARGIN_RIGHT = 6.0  # snug against the corner radius
_SPINNER_MARGIN_BOTTOM = 6.0  # snug against the bottom edge


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _lerp_color(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return tuple(_lerp(s, e, t) for s, e in zip(start, end))


def _background_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    from .overlay import _BG_COLOR_DARK as _PREVIEW_BG_COLOR_DARK, _BG_COLOR_LIGHT as _PREVIEW_BG_COLOR_LIGHT

    return _lerp_color(_PREVIEW_BG_COLOR_DARK, _PREVIEW_BG_COLOR_LIGHT, _clamp01(brightness))


def _user_text_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(_USER_TEXT_COLOR_DARK, _USER_TEXT_COLOR_LIGHT, _clamp01(brightness))


def _response_color_for_brightness(
    color: tuple[float, float, float],
    brightness: float,
) -> tuple[float, float, float]:
    # Keep the hue-rotating identity, but bias toward a dark endpoint on
    # bright screens so the response remains readable.
    t = _clamp01(brightness) ** 1.15
    return _lerp_color(color, _RESPONSE_TEXT_LIGHT_BG_TARGET, t)


def _thinking_cutout_color_for_brightness(brightness: float) -> tuple[float, float, float]:
    return _lerp_color(_THINKING_CUTOUT_DARK, _THINKING_CUTOUT_LIGHT, _clamp01(brightness))


def _assistant_text_alpha_for_breath(breath: float) -> float:
    return _lerp(_ASSISTANT_TEXT_ALPHA_MIN, _ASSISTANT_TEXT_ALPHA_MAX, _clamp01(breath))


def _fill_compositing_filter_for_brightness(brightness: float) -> str | None:
    from .overlay import _fill_compositing_filter_for_brightness as _preview_fill_compositing_filter_for_brightness

    return _preview_fill_compositing_filter_for_brightness(brightness)


def _spinner_state(elapsed: float) -> tuple[float, float]:
    """Pure function: radar-sweep spinner state for a given elapsed time.

    Returns (angle_radians, fill_fraction) where:
    - angle_radians: how far the sweep hand has rotated (0 to 2*pi per revolution)
    - fill_fraction: how much of the circle is filled opaque (0.0 to 1.0)

    After a full revolution the sweep resets for the next cycle.
    """
    if elapsed <= 0.0:
        return (0.0, 0.0)
    phase = (elapsed % _SPINNER_PERIOD) / _SPINNER_PERIOD  # 0.0 to 1.0
    angle = phase * 2.0 * math.pi
    fill = phase  # linear fill — the opaque region trails the sweep hand
    return (angle, fill)


def _ring_cutout_mask(
    fill_alpha,
    spinner_cx: float,
    spinner_cy: float,
    radius: float,
    ring_width: float,
    highlight_angle: float,
    highlight_width: float,
    scale: float,
    strength: float = 0.85,
):
    """Stamp a ring-shaped cutout into fill_alpha.

    The ring is a thin annulus centered on (spinner_cx, spinner_cy).
    It has a directional highlight — the cutout is deepest at
    highlight_angle and fades around the ring. This creates a
    negative-space halo that rotates when called with varying angles.

    Modifies fill_alpha in-place.
    """
    import numpy as np

    ph, pw = fill_alpha.shape
    r_px = radius * scale
    rw_px = ring_width * scale

    x = np.arange(pw, dtype=np.float32)[None, :] + 0.5
    y = np.arange(ph, dtype=np.float32)[:, None] + 0.5
    dx = x - spinner_cx
    dy = y - spinner_cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Ring SDF: gaussian around the ring centerline
    ring_dist = np.abs(dist - r_px)
    ring_mask = np.exp(-(ring_dist / max(rw_px, 0.1)) ** 2)

    # Directional highlight: strongest at highlight_angle, fades around
    pixel_angle = np.arctan2(dx, dy)  # CW from top
    angle_diff = np.arctan2(np.sin(pixel_angle - highlight_angle),
                            np.cos(pixel_angle - highlight_angle))
    highlight = np.exp(-(angle_diff / highlight_width) ** 2)

    # Combine: ring shape * highlight * strength
    cutout = ring_mask * (0.05 + 0.95 * highlight) * strength
    fill_alpha *= (1.0 - cutout)


def _spinner_cutout_mask(
    fill_alpha,
    fill_width: float,
    fill_height: float,
    spinner_cx: float,
    spinner_cy: float,
    radius: float,
    sweep_fill: float,
    scale: float,
):
    """Multiply a hole into fill_alpha for the spinner's swept region.

    Modifies fill_alpha in-place. The swept region (behind the sweep hand)
    becomes transparent. The unswept region is untouched. A soft anti-aliased
    edge prevents jaggies.

    spinner_cx, spinner_cy: center of the spinner in fill-image coordinates
    (pixels at the given scale).

    Returns the modified fill_alpha (same array, modified in-place).
    """
    import numpy as np

    ph, pw = fill_alpha.shape
    r_px = radius * scale

    # Coordinate grids relative to spinner center
    x = np.arange(pw, dtype=np.float32)[None, :] + 0.5
    y = np.arange(ph, dtype=np.float32)[:, None] + 0.5
    dx = x - spinner_cx
    dy = y - spinner_cy
    dist = np.sqrt(dx * dx + dy * dy)

    # Circle region — only modify pixels inside the spinner circle
    edge_softness = 1.5 * scale
    circle_mask = np.clip((r_px - dist) / edge_softness, 0.0, 1.0)

    # Angle of each pixel (CW from top / 12 o'clock = 0)
    pixel_angle = np.arctan2(dx, dy)
    pixel_angle = pixel_angle % (2.0 * np.pi)

    # Swept region with soft angular edge at the sweep boundary.
    # Instead of a hard step, the cutout fades over ~6 degrees at the
    # sweep hand for a smooth feathered transition.
    fill_end = sweep_fill * 2.0 * np.pi
    edge_width = 0.10  # radians (~5.7 degrees) of angular feather
    # Smooth ramp: 1.0 deep in swept region, fades to 0.0 at sweep edge
    sweep_mask = np.clip((fill_end - pixel_angle) / edge_width, 0.0, 1.0)

    # The cutout: where swept AND inside circle, multiply fill_alpha toward 0
    cutout = sweep_mask * circle_mask
    fill_alpha *= (1.0 - cutout)

    return fill_alpha


def _ease_in(progress: float) -> float:
    clamped = _clamp01(progress)
    return clamped * clamped


def _dismiss_animation_state(elapsed_s: float) -> tuple[str, float, float, bool]:
    elapsed = max(elapsed_s, 0.0)
    if elapsed < _DISMISS_GROW_S:
        eased = _ease_in(elapsed / _DISMISS_GROW_S)
        scale = _lerp(1.0, _DISMISS_GROW_SCALE, eased)
        return "grow", scale, 1.0, False

    shrink_elapsed = elapsed - _DISMISS_GROW_S
    eased = _ease_in(shrink_elapsed / _DISMISS_SHRINK_S)
    scale = _lerp(_DISMISS_GROW_SCALE, _DISMISS_END_SCALE, eased)
    alpha = 1.0 - eased
    done = elapsed >= (_DISMISS_DURATION_S - 1e-9)
    return "shrink", scale, alpha, done


class CommandOverlay(NSObject):
    """Overlay for displaying streamed command responses."""

    def initWithScreen_(self, screen: NSScreen | None = None):
        self = objc.super(CommandOverlay, self).init()
        if self is None:
            return None

        self._screen = screen or NSScreen.mainScreen()
        self._window: NSWindow | None = None
        self._wrapper_view: NSView | None = None
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
        self._cancel_timer_anim: NSTimer | None = None
        self._cancel_elapsed = 0.0
        self._cancel_phase = ""

        # Pulse state
        self._pulse_timer: NSTimer | None = None
        self._pulse_phase_asst = 0.0

        self._pulse_phase_user = _PULSE_PHASE_OFFSET_USER
        self._color_phase = 0.75  # start at violet, not red
        self._color_velocity_phase = 0.0
        self._tool_mode = False
        # Linger timer
        self._linger_timer: NSTimer | None = None

        # TTS amplitude state — drives window opacity during speech playback
        self._tts_amplitude = 0.0
        self._tts_active = False
        self._tts_blend = 0.0  # 0.0 = pure pulse, 1.0 = pure TTS

        # Thinking timer state
        self._thinking_timer: NSTimer | None = None
        self._thinking_seconds = 0.0

        # Cancel spring state — 0.0 = idle, 1.0 = fully wound
        self._cancel_spring = 0.0
        self._cancel_spring_target = 0.0  # 1.0 while winding, 0.0 while unwinding
        self._cancel_spring_fired = False  # True once threshold crossed
        self._on_cancel_spring_threshold: Callable[[], None] | None = None
        self._thinking_label = None  # NSTextField for the counter
        self._thinking_glow_layer = None  # CALayer for the glow behind the number
        self._thinking_inverted = False  # False = glowing number, True = cutout
        self._spinner_metal = None  # SpinnerMetalLayer (lazy init)

        # Adaptive compositing defaults dark until we sample the screen.
        self._brightness = 0.0
        self._brightness_target = 0.0

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
        wrapper.layer().setAnchorPoint_((0.5, 0.5))
        wrapper.layer().setPosition_((win_w / 2, win_h / 2))
        self._wrapper_view = wrapper

        content_frame = NSMakeRect(f, f, _OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        content = NSView.alloc().initWithFrame_(content_frame)
        content.setWantsLayer_(True)
        content.layer().setMasksToBounds_(False)
        content.layer().setBackgroundColor_(None)

        # Distance-field ridge + fill — same system as the preview overlay
        from .overlay import (
            _GLOW_COLOR as _OVERLAY_GLOW_COLOR,
            _overlay_layer_colors,
        )
        w, h = _OVERLAY_WIDTH, _OVERLAY_HEIGHT
        _, middle_rgb, _ = _overlay_layer_colors(_OVERLAY_GLOW_COLOR)

        self._ridge_scale = self._screen.backingScaleFactor() if hasattr(self._screen, 'backingScaleFactor') else 2.0

        # Fill layer — colored SDF image with baked alpha, same as preview overlay
        self._fill_layer = CALayer.alloc().init()
        self._fill_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._fill_layer.setContentsGravity_("resize")

        self._apply_ridge_masks(w, h)
        wrapper.layer().insertSublayer_below_(self._fill_layer, content.layer())

        # Cancel spring tint layer — sits above fill, masked to the same SDF shape
        self._spring_tint_layer = CALayer.alloc().init()
        self._spring_tint_layer.setFrame_(((0, 0), (win_w, win_h)))
        self._spring_tint_layer.setOpacity_(0.0)
        wrapper.layer().insertSublayer_above_(self._spring_tint_layer, self._fill_layer)

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
        # Kill clip view background — same fix as preview overlay
        clip_view = self._scroll_view.contentView()
        if clip_view and hasattr(clip_view, 'setDrawsBackground_'):
            clip_view.setDrawsBackground_(False)
        if clip_view and hasattr(clip_view, 'setWantsLayer_'):
            clip_view.setWantsLayer_(True)
            clip_view.layer().setBackgroundColor_(None)

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
        self._apply_surface_theme()
        self._set_overlay_scale(1.0)

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
        # Reset TTS state so stale blend doesn't affect new responses
        self._tts_active = False
        self._tts_blend = 0.0
        self._tts_amplitude = 0.0
        self._text_view.setString_("")
        if hasattr(self._text_view, "textStorage"):
            try:
                from AppKit import NSMutableAttributedString

                self._text_view.textStorage().setAttributedString_(
                    NSMutableAttributedString.alloc().initWithString_("")
                )
            except Exception:
                pass
        self._window.setAlphaValue_(0.0)
        self._set_overlay_scale(1.0)

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
        self._apply_ridge_masks(_OVERLAY_WIDTH, _OVERLAY_HEIGHT)
        self._fill_image_brightness = self._brightness
        self._apply_surface_theme()

        self._window.orderFrontRegardless()

        # Entrance pop — start slightly oversized, ease back to 1.0.
        # Runs concurrently with the fade-in for a subtle "I just arrived" feel.
        self._set_overlay_scale(_ENTRANCE_POP_SCALE)
        self._pop_step = 0
        self._pop_steps = max(1, int(_ENTRANCE_POP_S * _DISMISS_ANIM_FPS))
        self._pop_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_entrancePopStep:", None, True
        )

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
        self._tool_mode = False
        self._pulse_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _PULSE_HZ, self, "pulseStep:", None, True
        )

        # Start thinking timer (glowing number mode)
        self._start_thinking_timer()

    def set_brightness(self, brightness: float, immediate: bool = False) -> None:
        """Set screen brightness (0.0 dark – 1.0 bright) for adaptive compositing."""
        self._brightness_target = _clamp01(brightness)
        if immediate:
            self._brightness = self._brightness_target
            self._apply_surface_theme()

    def _entrancePopStep_(self, timer) -> None:
        """Ease the entrance pop back to scale 1.0."""
        self._pop_step += 1
        t = self._pop_step / self._pop_steps
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        scale = _ENTRANCE_POP_SCALE * (1.0 - eased) + 1.0 * eased
        self._set_overlay_scale(scale)
        if self._pop_step >= self._pop_steps:
            timer.invalidate()
            self._pop_timer = None
            self._set_overlay_scale(1.0)

    def cancel_dismiss(self) -> None:
        """Dismiss with a fast pop-then-shrink animation."""
        if self._window is None:
            return
        self._cancel_all_timers()
        self._streaming = False
        self._visible = True  # keep visible for the animation

        self._cancel_elapsed = 0.0
        self._cancel_phase = "grow"
        self._window.setAlphaValue_(1.0)
        self._set_overlay_scale(1.0)
        self._cancel_timer_anim = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / _DISMISS_ANIM_FPS, self, "_cancelAnimStep:", None, True
        )

    def _cancelAnimStep_(self, timer) -> None:
        """Animate the dismiss sequence: grow for 60ms, then shrink and fade."""
        self._cancel_elapsed += 1.0 / _DISMISS_ANIM_FPS
        phase, scale, alpha, done = _dismiss_animation_state(self._cancel_elapsed)
        self._cancel_phase = phase
        self._set_overlay_scale(scale)
        self._window.setAlphaValue_(alpha)

        if done:
            self._cancel_dismiss_animation()
            self._window.setAlphaValue_(0.0)
            self._set_overlay_scale(1.0)
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

    def set_cancel_spring(self, target: float) -> None:
        """Set the cancel spring target (0.0 = idle, 1.0 = winding up).

        The spring animates toward the target in the pulse tick, so
        calling this once starts the wind-up or snap-back.
        """
        self._cancel_spring_target = max(0.0, min(1.0, target))
        if target > 0.0:
            self._cancel_spring_fired = False  # reset for new wind-up

    def set_utterance(self, text: str) -> None:
        """Show the user's utterance in the text view at reduced opacity."""
        self._utterance_text = text
        if self._text_view is None or not self._visible:
            return
        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )
        user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
        attr_str = NSMutableAttributedString.alloc().initWithString_(text)
        attr_str.addAttribute_value_range_(
            NSForegroundColorAttributeName,
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4),
            (0, len(text)),
        )
        glow = NSShadow.alloc().init()
        glow.setShadowColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.15)
        )
        glow.setShadowOffset_((0, 0))
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

    def set_response_text(self, text: str) -> None:
        """Replace the visible assistant response with final canonical text.

        Rebuilds the full attributed string (utterance + response) in one shot
        so that _update_layout is called exactly once.  Calling set_utterance
        then append_token would call _update_layout twice: first with only the
        utterance (shrinking the window back to minimum height), then again with
        the full response — producing a visible size flicker.
        """
        self._response_text = ""
        if self._text_view is None or not self._visible:
            self._response_text = text
            return

        from AppKit import (
            NSMutableAttributedString,
            NSForegroundColorAttributeName,
            NSShadowAttributeName,
            NSShadow,
        )

        combined = NSMutableAttributedString.alloc().initWithString_("")

        if self._utterance_text:
            user_r, user_g, user_b = _user_text_color_for_brightness(self._brightness)
            utt = NSMutableAttributedString.alloc().initWithString_(self._utterance_text)
            utt.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.4),
                (0, len(self._utterance_text)),
            )
            glow = NSShadow.alloc().init()
            glow.setShadowColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(user_r, user_g, user_b, 0.15)
            )
            glow.setShadowOffset_((0, 0))
            glow.setShadowBlurRadius_(3.0)
            utt.addAttribute_value_range_(
                NSShadowAttributeName, glow, (0, len(self._utterance_text))
            )
            combined.appendAttributedString_(utt)

            if text:
                sep = NSMutableAttributedString.alloc().initWithString_("\n\n")
                sep.addAttribute_value_range_(
                    NSForegroundColorAttributeName,
                    NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.0),
                    (0, 2),
                )
                combined.appendAttributedString_(sep)

        self._text_view.textStorage().setAttributedString_(combined)

        if text:
            self._response_text = ""
            self.append_token(text)
        else:
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
        r, g, b = _response_color_for_brightness((r, g, b), self._brightness)
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

    def set_tool_active(self, active: bool) -> None:
        """Show or hide the tool execution indicator."""
        self._tool_mode = active
        if active and self._visible:
            if self._thinking_label is not None:
                self._thinking_label.setHidden_(False)
                self._thinking_label.setStringValue_("tool…")
            if self._thinking_timer is None:
                self._start_thinking_timer()

    def finish(self) -> None:
        """Called when the response stream is complete.

        The overlay stays visible indefinitely — the user dismisses it
        by holding spacebar (which triggers cancel_dismiss via __main__).
        No linger timer. The user is in control.
        """
        self._streaming = False
        self._stop_thinking_timer()
        # Pulse keeps running — the overlay is alive until dismissed

    # ── TTS amplitude ─────────────────────────────────────────

    _TTS_RISE = 0.70      # fast but not instant — avoids harsh pop-in
    _TTS_DECAY = 0.82     # gentle falloff — no clipping between words
    _TTS_MULTIPLIER = 25.0
    _TTS_ALPHA_MIN = _ASSISTANT_TEXT_ALPHA_MIN  # keep assistant text firmly readable during playback
    _TTS_ALPHA_MAX = 1.0   # full brightness on voice peaks

    def update_tts_amplitude(self, rms: float) -> None:
        """Update smoothed TTS amplitude from audio RMS.

        The pulse step reads _tts_amplitude when _tts_active is True to
        drive the response text alpha — no window alpha manipulation here,
        so there's no fighting between pulse and TTS updates.
        Must be called on the main thread.
        """
        if rms > self._tts_amplitude:
            self._tts_amplitude += (rms - self._tts_amplitude) * self._TTS_RISE
        else:
            self._tts_amplitude *= self._TTS_DECAY

    def tts_start(self) -> None:
        """Prepare overlay for TTS-driven amplitude."""
        self._tts_active = True
        self._tts_amplitude = 0.0

    def tts_stop(self) -> None:
        """Return to pulse-driven alpha after TTS ends."""
        self._tts_active = False
        self._tts_amplitude = 0.0

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
        """Dual-phase pulse: user and assistant text breathe independently.

        Assistant text: faster period (0.8x base), double-smoothstep for
        extra-aggressive easing, violet-amber color oscillation.
        User text: slower period (1.5x base), single smoothstep, blue
        color shift, phase-offset by 0.3 and diverging naturally.
        """
        try:
            self._pulseStepInner()
        except Exception:
            logger.exception("pulseStep_ crashed")

    def _pulseStepInner(self):
        if self._text_view is None:
            return
        dt = 1.0 / _PULSE_HZ

        target = getattr(self, "_brightness_target", 0.0)
        current = getattr(self, "_brightness", 0.0)
        if abs(target - current) > 0.001:
            self._brightness = current + (target - current) * _BRIGHTNESS_CHASE
        t = getattr(self, "_brightness", 0.0)
        self._apply_surface_theme()

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
        pulse_alpha_a = _assistant_text_alpha_for_breath(breath)

        # Smooth cross-fade between pulse and TTS-driven alpha over ~500ms.
        # _tts_blend ramps toward 1.0 when TTS is active, toward 0.0 when not.
        # At 30Hz pulse rate, 0.06 per tick ≈ 0.55s full ramp.
        _BLEND_RATE = 0.06
        if self._tts_active:
            self._tts_blend = min(self._tts_blend + _BLEND_RATE, 1.0)
        else:
            self._tts_blend = max(self._tts_blend - _BLEND_RATE, 0.0)

        if self._tts_blend > 0.0:
            tts_scaled = min(self._tts_amplitude * self._TTS_MULTIPLIER, 1.0)
            tts_alpha = self._TTS_ALPHA_MIN + tts_scaled * (self._TTS_ALPHA_MAX - self._TTS_ALPHA_MIN)
            alpha_a = pulse_alpha_a * (1.0 - self._tts_blend) + tts_alpha * self._tts_blend
        else:
            alpha_a = pulse_alpha_a

        # User: raw sine → single smoothstep (same aggressiveness as before)
        raw_u = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._pulse_phase_user))
        pulse_u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
        utt_alpha = 0.325 + 0.125 * pulse_u

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

        # ── Cancel spring animation ──
        # The spring chases _cancel_spring_target with asymmetric speed:
        # winding up is deliberate (~600ms to full), easing out is smooth
        # (~300ms with deceleration).
        _SPRING_THRESHOLD = 0.83  # ~500ms into the 600ms wind-up
        spring = self._cancel_spring
        target = self._cancel_spring_target
        if spring < target:
            # Wind up — deliberate
            spring = min(target, spring + dt / 0.6)
            # Fire cancel at threshold crossing — don't wait for release
            if spring >= _SPRING_THRESHOLD and not self._cancel_spring_fired:
                self._cancel_spring_fired = True
                self._cancel_spring_target = 0.0  # start ease-out
                target = 0.0
                cb = self._on_cancel_spring_threshold
                if cb is not None:
                    cb()
        if spring > target:
            # Ease out — smooth deceleration (quadratic decay)
            rate = dt / 0.3 * (1.0 + spring)  # faster when higher
            spring = max(target, spring - rate)
        self._cancel_spring = spring

        if spring > 0.001:
            # Blend hue toward warm amber (~0.08, orange-gold).
            # Use shortest arc around the hue wheel.
            amber_hue = 0.08
            if hue > 0.5 + amber_hue:
                target_hue = 1.0 + amber_hue  # wrap forward
            else:
                target_hue = amber_hue
            hue = hue + (target_hue - hue) * spring
            if hue >= 1.0:
                hue -= 1.0

        # Log color phase every ~1s (every 30th tick)
        if not hasattr(self, '_color_log_counter'):
            self._color_log_counter = 0
        self._color_log_counter += 1
        if self._color_log_counter % 30 == 0:
            logger.info("Color phase: %.3f hue, vel_phase=%.3f, spring=%.3f", hue, self._color_velocity_phase, spring)

        # Saturation and value — spring winds up saturation toward vivid red
        base_s, base_v = 0.228, 0.81  # desaturated — legible, ambient, not neon
        s = base_s + (0.75 - base_s) * spring  # 0.228 → 0.75 at full wind
        v = base_v + (0.90 - base_v) * spring  # 0.81  → 0.90 at full wind
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
        response_r, response_g, response_b = _response_color_for_brightness((r, g, b), t)

        # Update text colors per-range
        if self._text_view is not None:
            from AppKit import NSForegroundColorAttributeName as _FG_pulse
            ts = self._text_view.textStorage()
            total_len = ts.length() if hasattr(ts, 'length') else 0
            if total_len > 0 and self._utterance_text:
                utt_len = min(len(self._utterance_text), total_len)
                # User text keeps the adaptive light/dark base, then breathes subtly.
                try:
                    user_base = _user_text_color_for_brightness(t)
                    ur = _lerp(user_base[0], 1.0, 0.08 * pulse_u)
                    ug = _lerp(user_base[1], 1.0, 0.06 * pulse_u)
                    ub = _lerp(user_base[2], 1.0, 0.04 * pulse_u)
                    ts.addAttribute_value_range_(
                        _FG_pulse,
                        NSColor.colorWithSRGBRed_green_blue_alpha_(ur, ug, ub, utt_alpha),
                        (0, utt_len),
                    )
                except Exception:
                    pass
                # Response text stays hue-rotating, but darkens on bright screens.
                resp_start = utt_len + 2
                if resp_start < total_len:
                    try:
                        ts.addAttribute_value_range_(
                            _FG_pulse,
                            NSColor.colorWithSRGBRed_green_blue_alpha_(
                                response_r, response_g, response_b, alpha_a
                            ),
                            (resp_start, total_len - resp_start),
                        )
                    except Exception:
                        pass
            elif total_len > 0:
                self._text_view.setTextColor_(
                    NSColor.colorWithSRGBRed_green_blue_alpha_(
                        response_r, response_g, response_b, alpha_a
                    )
                )

        # Pulse the glow with assistant phase oscillating color
        glow_nscolor = NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0)
        glow_opacity = 0.5 + 0.3 * breath
        # Drive the SDF fill layer with the pulse — the fill breathes
        # with the assistant's thinking/response animation.
        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setOpacity_(min(glow_opacity * 0.7, 0.85))
        # Cancel spring: warm amber tint over the overlay shape.
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            if spring > 0.01:
                from Quartz import CGColorCreateSRGB
                # Warm golden-amber tint — visible, thermal, not alarming
                cg_color = CGColorCreateSRGB(0.55, 0.38, 0.05, 1.0)
                self._spring_tint_layer.setBackgroundColor_(cg_color)
                self._spring_tint_layer.setOpacity_(0.5 * spring)
            else:
                self._spring_tint_layer.setOpacity_(0.0)

        # Spinner: advance time every pulse tick. Halos rotate at 30Hz
        # (just a transform, free). Fill image rebuilds at ~5Hz (every 6th
        # tick) to avoid CGImage allocation stutter — the halos carry the
        # visual motion, the cutout just needs to roughly track the sweep.
        if getattr(self, "_spinner_active", False):
            self._spinner_elapsed = getattr(self, "_spinner_elapsed", 0.0) + dt
            self._spinner_fill_counter = getattr(self, "_spinner_fill_counter", 0) + 1
            if self._spinner_fill_counter >= 3:  # ~10Hz — rings need smooth rotation
                self._spinner_fill_counter = 0
                self._update_spinner_fill()
            # Render Metal effect behind the cutout
            if self._spinner_metal is not None and self._spinner_metal.ready:
                angle, sweep_fill = _spinner_state(self._spinner_elapsed)
                metal_size = _SPINNER_RADIUS * 2.0 + 8.0  # diameter + margin
                self._spinner_metal.render(
                    time=self._spinner_elapsed,
                    sweep_angle=angle,
                    sweep_fill=sweep_fill,
                    radius=_SPINNER_RADIUS * getattr(self, "_ridge_scale", 2.0),
                    center=(metal_size / 2 * getattr(self, "_ridge_scale", 2.0),
                            metal_size / 2 * getattr(self, "_ridge_scale", 2.0)),
                    size=(metal_size * getattr(self, "_ridge_scale", 2.0),
                          metal_size * getattr(self, "_ridge_scale", 2.0)),
                    hue_color=(r, g, b),
                    brightness=self._brightness,
                )

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

    def _cancel_dismiss_animation(self) -> None:
        if self._cancel_timer_anim is not None:
            self._cancel_timer_anim.invalidate()
            self._cancel_timer_anim = None

    def _cancel_entrance_pop(self) -> None:
        if getattr(self, "_pop_timer", None) is not None:
            self._pop_timer.invalidate()
            self._pop_timer = None

    def _cancel_all_timers(self) -> None:
        self._cancel_dismiss_animation()
        self._cancel_entrance_pop()
        self._cancel_fade()
        self._cancel_pulse()
        self._cancel_linger()
        self._stop_thinking_timer()

    def _set_overlay_scale(self, scale: float) -> None:
        if self._wrapper_view is None:
            return
        layer = self._wrapper_view.layer() if hasattr(self._wrapper_view, "layer") else None
        if layer is None:
            return
        try:
            # Re-assert center anchor so scale is always symmetric, even
            # after window geometry changes that may shift the layer bounds.
            bounds = layer.bounds()
            layer.setAnchorPoint_((0.5, 0.5))
            layer.setPosition_((
                bounds.origin.x + bounds.size.width / 2,
                bounds.origin.y + bounds.size.height / 2,
            ))
            layer.setValue_forKeyPath_(scale, "transform.scale")
        except Exception:
            logger.exception("Failed to update command overlay scale")

    # ── thinking timer ──────────────────────────────────────

    def _start_thinking_timer(self) -> None:
        """Start the thinking counter in glowing-number mode with radar spinner."""
        self._thinking_seconds = 0.0
        self._thinking_inverted = False
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(False)
            if self._tool_mode: self._thinking_label.setStringValue_("tool…")
            else: self._thinking_label.setStringValue_("0.0s")
            # Glowing number: violet text on transparent background
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(
                    _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.7
                )
            )

        # Mark spinner as active — the fill image will include the cutout
        self._spinner_active = True
        self._spinner_elapsed = 0.0
        # Rebuild fill with spinner cutout at t=0
        self._update_spinner_fill()
        # Set up Metal effect layer (if available)
        self._setup_spinner_metal()

        logger.info("Thinking timer started (with spinner)")
        self._thinking_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "thinkingTick:", None, True
        )

    def _setup_spinner_metal(self) -> None:
        """Create or show the Metal effect layer behind the spinner cutout."""
        if self._wrapper_view is None:
            return

        if self._spinner_metal is None:
            try:
                from .spinner_metal import SpinnerMetalLayer
                self._spinner_metal = SpinnerMetalLayer()
            except Exception:
                logger.info("Failed to import SpinnerMetalLayer", exc_info=True)
                self._spinner_metal = None
                return

        if not self._spinner_metal.ready:
            # Set up the Metal layer covering the entire wrapper
            # (the effect is clipped by the fill's alpha anyway)
            f = _OUTER_FEATHER
            content_frame = self._content_view.frame() if self._content_view else None
            if content_frame is None:
                return
            # Position the Metal layer to cover the spinner region with margin
            diameter = _SPINNER_RADIUS * 2.0
            margin = 4.0  # extra pixels for glow
            metal_size = diameter + margin * 2
            # In wrapper coords (macOS: Y=0 at bottom)
            metal_x = f + content_frame.size.width - diameter - _SPINNER_MARGIN_RIGHT - margin
            metal_y = f + _SPINNER_MARGIN_BOTTOM - margin  # bottom-right
            frame = ((metal_x, metal_y), (metal_size, metal_size))
            scale = getattr(self, "_ridge_scale", 2.0)

            if not self._spinner_metal.setup(
                self._wrapper_view.layer(), frame, scale
            ):
                logger.info("Metal spinner unavailable, falling back to cutout-only")
                return
        else:
            self._spinner_metal.set_hidden(False)

    def _stamp_spinner_cutout(self, fill_alpha, width, height, f, scale):
        """Punch the spinner cutout and ring halos into a fill alpha array."""
        elapsed = getattr(self, "_spinner_elapsed", 0.0)
        _, sweep_fill = _spinner_state(elapsed)
        spinner_cx = (f + width - _SPINNER_RADIUS - _SPINNER_MARGIN_RIGHT) * scale
        ph = fill_alpha.shape[0]
        spinner_cy = ph - (f + _SPINNER_MARGIN_BOTTOM + _SPINNER_RADIUS) * scale

        # Main sweep cutout
        _spinner_cutout_mask(
            fill_alpha, width + 2 * f, height + 2 * f,
            spinner_cx, spinner_cy,
            _SPINNER_RADIUS, sweep_fill, scale,
        )

        # Ring 1: synced to sweep — same radius, highlight tracks sweep hand
        sweep_angle = (elapsed / _SPINNER_PERIOD) * 2.0 * math.pi
        _ring_cutout_mask(
            fill_alpha, spinner_cx, spinner_cy,
            radius=_SPINNER_RADIUS, ring_width=1.5,
            highlight_angle=sweep_angle, highlight_width=0.4,
            scale=scale, strength=0.9,
        )

        # Ring 2: inset, counter-rotating at 7/5x speed
        ring2_angle = -(elapsed / _SPINNER_PERIOD) * (7.0 / 5.0) * 2.0 * math.pi
        _ring_cutout_mask(
            fill_alpha, spinner_cx, spinner_cy,
            radius=_SPINNER_RADIUS * 0.6, ring_width=0.8,
            highlight_angle=ring2_angle, highlight_width=0.5,
            scale=scale, strength=0.8,
        )

    def _bake_fill_image(self, fill_alpha):
        """Convert fill alpha array to CGImage and update layers."""
        try:
            from .overlay import _fill_field_to_image

            bg_r, bg_g, bg_b = _background_color_for_brightness(
                getattr(self, '_brightness', 0.0)
            )
            fill_image, self._fill_payload = _fill_field_to_image(
                fill_alpha,
                int(bg_r * 255), int(bg_g * 255), int(bg_b * 255),
            )
        except (ImportError, Exception):
            return

        f = _OUTER_FEATHER
        width = getattr(self, "_base_fill_width", _OVERLAY_WIDTH)
        height = getattr(self, "_base_fill_height", _OVERLAY_HEIGHT)
        total_w = width + 2 * f
        total_h = height + 2 * f

        if hasattr(self, '_fill_layer') and self._fill_layer is not None:
            self._fill_layer.setContents_(fill_image)
            self._fill_layer.setFrame_(((0, 0), (total_w, total_h)))
            if hasattr(self._fill_layer, "setCompositingFilter_"):
                self._fill_layer.setCompositingFilter_(
                    _fill_compositing_filter_for_brightness(getattr(self, "_brightness", 0.0))
                )
        if hasattr(self, '_spring_tint_layer') and self._spring_tint_layer is not None:
            self._spring_tint_layer.setFrame_(((0, 0), (total_w, total_h)))
            mask = CALayer.alloc().init()
            mask.setFrame_(((0, 0), (total_w, total_h)))
            mask.setContents_(fill_image)
            mask.setContentsGravity_("resize")
            self._spring_tint_layer.setMask_(mask)

    def _update_spinner_fill(self) -> None:
        """Fast path: copy cached base fill, stamp cutout, bake image.

        Skips the SDF computation — only the cutout mask and CGImage
        rebuild happen per frame.
        """
        base = getattr(self, "_base_fill_alpha", None)
        if base is None:
            # No cached fill — fall back to full rebuild
            if self._content_view is None:
                return
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)
            return

        fill_alpha = base.copy()
        if getattr(self, "_spinner_active", False):
            f = _OUTER_FEATHER
            scale = getattr(self, '_ridge_scale', 2.0)
            width = self._base_fill_width
            height = self._base_fill_height
            self._stamp_spinner_cutout(fill_alpha, width, height, f, scale)

        self._bake_fill_image(fill_alpha)

    def invert_thinking_timer(self) -> None:
        """Switch from glowing number to negative-space cutout mode.

        Called when the first content token arrives — the visual flip
        signals that the model has transitioned from thinking to speaking.
        """
        self._thinking_inverted = True
        self._apply_thinking_label_theme()

    def _stop_thinking_timer(self) -> None:
        """Stop and fade the thinking counter."""
        if self._thinking_timer is not None:
            self._thinking_timer.invalidate()
            self._thinking_timer = None
        if self._thinking_label is not None:
            self._thinking_label.setHidden_(True)
        # Deactivate spinner and rebuild fill without the cutout
        self._spinner_active = False
        self._update_spinner_fill()
        # Hide the Metal effect layer
        if self._spinner_metal is not None and self._spinner_metal.ready:
            self._spinner_metal.set_hidden(True)

    def thinkingTick_(self, timer) -> None:
        """Update the thinking counter and spinner every 100ms."""
        self._thinking_seconds += 0.1
        if self._thinking_label is not None and not self._thinking_label.isHidden():
            if self._tool_mode: self._thinking_label.setStringValue_("tool…")
            else: self._thinking_label.setStringValue_(f"{self._thinking_seconds:.1f}s")
        # Spinner fill is now updated at 30Hz in the pulse timer, not here.

    # ── ridge masks ────────────────────────────────────────

    def _apply_ridge_masks(self, width: float, height: float) -> None:
        """Compute SDF and apply ridge mask + build fill image.

        Caches the base fill alpha so the spinner can stamp its cutout
        into a copy without recomputing the SDF every frame.
        """
        from .overlay import (
            _RIDGE_FALLOFF, _RIDGE_POWER, _OVERLAY_CORNER_RADIUS,
            _overlay_rounded_rect_sdf, _ridge_alpha, _interior_fill_alpha,
            _fill_field_to_image, _BG_COLOR_DARK,
        )
        f = _OUTER_FEATHER
        scale = getattr(self, '_ridge_scale', 2.0)
        total_w = width + 2 * f
        total_h = height + 2 * f

        try:
            from .overlay import _glow_fill_alpha

            sdf = _overlay_rounded_rect_sdf(
                total_w, total_h, width, height,
                _OVERLAY_CORNER_RADIUS, scale,
            )

            fill_alpha = _glow_fill_alpha(sdf, width=2.5 * scale)

            # Cache the base fill alpha for fast spinner updates
            self._base_fill_alpha = fill_alpha.copy()
            self._base_fill_width = width
            self._base_fill_height = height

            # Apply spinner cutout if active
            if getattr(self, "_spinner_active", False):
                self._stamp_spinner_cutout(fill_alpha, width, height, f, scale)

            self._bake_fill_image(fill_alpha)
        except (ImportError, Exception):
            return

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

            max_height = _max_overlay_height(self._screen.frame().size.height)
            new_height = min(max(_OVERLAY_HEIGHT, text_height + 24), max_height)

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
                self._apply_ridge_masks(_OVERLAY_WIDTH, new_height)

            end = (self._text_view.string().length()
                   if hasattr(self._text_view.string(), 'length')
                   else len(self._response_text))
            self._text_view.scrollRangeToVisible_((end, 0))
        except Exception:
            logger.debug("Command overlay layout update failed", exc_info=True)

    def _apply_surface_theme(self) -> None:
        if self._content_view is None:
            return
        # Keep the content view transparent so the same SDF fill/material carries
        # the assistant window chrome as the preview overlay.
        self._content_view.layer().setBackgroundColor_(None)
        if hasattr(self, "_fill_layer") and self._fill_layer is not None and hasattr(
            self._fill_layer, "setCompositingFilter_"
        ):
            self._fill_layer.setCompositingFilter_(
                _fill_compositing_filter_for_brightness(self._brightness)
            )
        last_t = getattr(self, "_fill_image_brightness", -1.0)
        if abs(self._brightness - last_t) > 0.03:
            self._fill_image_brightness = self._brightness
            content_frame = self._content_view.frame()
            self._apply_ridge_masks(content_frame.size.width, content_frame.size.height)
        self._apply_thinking_label_theme()

    def _apply_thinking_label_theme(self) -> None:
        if self._thinking_label is None or self._thinking_label.isHidden():
            return
        if self._thinking_inverted:
            cut_r, cut_g, cut_b = _thinking_cutout_color_for_brightness(self._brightness)
            self._thinking_label.setTextColor_(
                NSColor.colorWithSRGBRed_green_blue_alpha_(cut_r, cut_g, cut_b, 0.7)
            )
            return
        self._thinking_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                _GLOW_COLOR[0], _GLOW_COLOR[1], _GLOW_COLOR[2], 0.7
            )
        )
