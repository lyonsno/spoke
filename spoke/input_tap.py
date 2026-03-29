"""CGEventTap-based spacebar hold detection.

Installs a global event tap that distinguishes spacebar *taps* (forwarded
as normal space characters) from spacebar *holds* (triggers recording).

State machine:
    IDLE ──[space keyDown]──> WAITING  (suppress event, start timer)
    WAITING ──[space keyUp before timer]──> IDLE  (forward synth space)
    WAITING ──[timer fires]──> RECORDING  (call on_hold_start)
    RECORDING ──[space keyUp]──> IDLE  (call on_hold_end)
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable

import objc
from Foundation import NSObject, NSTimer

from Quartz import (
    CFMachPortCreateRunLoopSource,
    CFRunLoopAddSource,
    CFRunLoopGetMain,
    CGEventCreateKeyboardEvent,
    CGEventGetFlags,
    CGEventGetIntegerValueField,
    CGEventMaskBit,
    CGEventPost,
    CGEventTapCreate,
    CGEventTapEnable,
    kCFRunLoopCommonModes,
    kCGEventFlagMaskAlternate,
    kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl,
    kCGEventFlagMaskShift,
    kCGEventFlagsChanged,
    kCGEventKeyDown,
    kCGEventKeyUp,
    kCGEventTapOptionDefault,
    kCGHeadInsertEventTap,
    kCGHIDEventTap,
    kCGKeyboardEventKeycode,
    kCGSessionEventTap,
)
logger = logging.getLogger(__name__)

SPACEBAR_KEYCODE = 49
ENTER_KEYCODE = 36
# Modifiers that prevent recording when held during spacebar press.
# Shift is intentionally excluded — shift+space starts recording normally,
# and shift is detected at release to route to the tray.
_MODIFIER_MASK = (
    kCGEventFlagMaskCommand
    | kCGEventFlagMaskControl
    | kCGEventFlagMaskAlternate
)
_DEFAULT_HOLD_MS = 400
_SAFETY_TIMEOUT_S = 300.0  # 5 minutes — covers long dictations, only for truly stuck keyUp
_FORWARDING_TIMEOUT_S = 0.1  # auto-clear _forwarding if events never arrive


class _State(Enum):
    IDLE = auto()
    WAITING = auto()
    RECORDING = auto()


class SpacebarHoldDetector(NSObject):
    """Detects spacebar holds via a global CGEventTap.

    Parameters
    ----------
    on_hold_start : callable
        Called (on main thread) when spacebar has been held past the threshold.
    on_hold_end : callable
        Called (on main thread) when spacebar is released after a hold.
    hold_ms : int
        Milliseconds the spacebar must be held before triggering. Default 400.
    """

    def initWithHoldStart_holdEnd_holdMs_(
        self,
        on_hold_start: Callable[[], None],
        on_hold_end: Callable[[], None],
        hold_ms: int,
    ):
        self = objc.super(SpacebarHoldDetector, self).init()
        if self is None:
            return None
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._hold_s = hold_ms / 1000.0

        self._state = _State.IDLE
        self._shift_at_press = False
        self._shift_latched = False  # True if shift was seen during WAITING/RECORDING
        self._enter_held = False  # True if enter is currently held (for command fast path)
        self._hold_timer: NSTimer | None = None
        self._safety_timer: NSTimer | None = None
        self._forwarding = False
        self._forwarding_timer: NSTimer | None = None
        self._tap = None
        self._tap_source = None

        # Tray mode support — set by the delegate when tray is active.
        # When True, quick spacebar taps call on_hold_end instead of
        # forwarding a space character, and shift gestures route to
        # tray navigation callbacks.
        self.tray_active = False
        self._on_shift_tap: Callable[[], None] | None = None
        self._on_shift_tap_during_hold: Callable[[], None] | None = None
        self._on_enter_pressed: Callable[[], None] | None = None
        self._on_tray_delete: Callable[[], None] | None = None
        self._tray_shift_down = False
        self._tray_space_between = False
        self._shift_down_during_hold = False  # tracks shift press while spacebar held
        # Double-tap detection for delete gesture (shift held + double-tap spacebar)
        self._tray_last_shift_space_up: float = 0.0

        return self

    # ── public ──────────────────────────────────────────────

    def install(self) -> bool:
        """Install the global event tap. Returns False if permission denied."""
        event_mask = (
            CGEventMaskBit(kCGEventKeyDown)
            | CGEventMaskBit(kCGEventKeyUp)
            | CGEventMaskBit(kCGEventFlagsChanged)
        )

        # Store self on the module so the C callback can reach it.
        # CGEventTap refcon in PyObjC is unreliable, so we use a module global.
        global _active_detector  # noqa: PLW0603
        _active_detector = self

        self._tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            event_mask,
            _event_tap_callback,
            None,
        )
        if self._tap is None:
            logger.error(
                "CGEventTapCreate returned None — Accessibility permission required"
            )
            return False

        self._tap_source = CFMachPortCreateRunLoopSource(None, self._tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), self._tap_source, kCFRunLoopCommonModes)
        CGEventTapEnable(self._tap, True)
        logger.info("CGEventTap installed — spacebar hold detection active")
        return True

    def force_end(self) -> None:
        """Programmatically end a recording hold (e.g. recording cap reached)."""
        if self._state == _State.RECORDING:
            self._cancel_safety_timer()
            self._state = _State.IDLE
            self._on_hold_end(
                shift_held=False,
                enter_held=getattr(self, '_enter_held', False),
            )

    def uninstall(self) -> None:
        """Disable and remove the event tap."""
        if self._tap is not None:
            CGEventTapEnable(self._tap, False)
            self._tap = None
            self._tap_source = None
        self._cancel_hold_timer()
        self._cancel_safety_timer()
        self._cancel_forwarding_timer()
        self._forwarding = False
        self._enter_held = False
        self.tray_active = False
        self._state = _State.IDLE

        global _active_detector  # noqa: PLW0603
        _active_detector = None
        logger.info("CGEventTap uninstalled")

    # ── event handling (called from C callback, main thread) ─

    def handle_key_down(self, keycode: int, flags: int) -> bool:
        """Handle a keyDown event. Returns True to suppress, False to pass through."""
        if keycode != SPACEBAR_KEYCODE:
            return False

        # Pass through if any modifier is held (Cmd+Space, etc.)
        if flags & _MODIFIER_MASK:
            return False

        if self._state == _State.IDLE:
            self._state = _State.WAITING
            self._shift_at_press = bool(flags & kCGEventFlagMaskShift)
            self._shift_latched = self._shift_at_press
            self._start_hold_timer()
            return True  # suppress the space

        # Already WAITING or RECORDING — suppress key repeats
        return True

    def handle_key_up(self, keycode: int, flags: int = 0) -> bool:
        """Handle a keyUp event. Returns True to suppress, False to pass through.

        Parameters
        ----------
        keycode : int
            The key that was released.
        flags : int
            Modifier flags at the moment of release. Used to detect
            shift-release for command routing.
        """
        if keycode != SPACEBAR_KEYCODE:
            return False

        if self._state == _State.WAITING:
            # Released before hold threshold
            self._cancel_hold_timer()
            self._state = _State.IDLE
            shift_held = bool(flags & kCGEventFlagMaskShift) or self._shift_latched
            enter_held = getattr(self, '_enter_held', False)
            self._shift_latched = False
            if getattr(self, 'tray_active', False):
                # During tray, all spacebar taps route through on_hold_end
                # instead of forwarding a space character.
                # Double-tap detection: shift held + two spacebar taps within
                # 300ms = delete current tray entry.
                now = time.monotonic()
                last = getattr(self, '_tray_last_shift_space_up', 0.0)
                if shift_held and (now - last) < 0.3:
                    # Double-tap with shift held = delete
                    self._tray_last_shift_space_up = 0.0  # reset
                    on_delete = getattr(self, '_on_tray_delete', None)
                    if on_delete is not None:
                        on_delete()
                    return True
                if shift_held:
                    self._tray_last_shift_space_up = now
                else:
                    self._tray_last_shift_space_up = 0.0
                self._on_hold_end(shift_held=shift_held, enter_held=enter_held)
            elif shift_held:
                # Shift + quick tap = signal for tray recall (no space)
                self._on_hold_end(shift_held=True, enter_held=enter_held)
            else:
                # Normal quick tap = forward a space
                self._forward_space()
            return True

        if self._state == _State.RECORDING:
            self._cancel_safety_timer()
            self._state = _State.IDLE
            shift_held = bool(flags & kCGEventFlagMaskShift) or self._shift_latched
            enter_held = getattr(self, '_enter_held', False)
            self._shift_latched = False
            self._on_hold_end(shift_held=shift_held, enter_held=enter_held)
            return True

        return False

    # ── timers ──────────────────────────────────────────────

    def _start_hold_timer(self) -> None:
        self._hold_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self._hold_s, self, "holdTimerFired:", None, False
        )

    def holdTimerFired_(self, timer: NSTimer) -> None:
        """Called when spacebar has been held past the threshold."""
        self._hold_timer = None
        if self._state != _State.WAITING:
            return
        self._state = _State.RECORDING
        self._start_safety_timer()
        self._on_hold_start()

    def _cancel_hold_timer(self) -> None:
        if self._hold_timer is not None:
            self._hold_timer.invalidate()
            self._hold_timer = None

    def _start_safety_timer(self) -> None:
        """Auto-stop recording after 30s in case keyUp is missed."""
        self._safety_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _SAFETY_TIMEOUT_S, self, "safetyTimerFired:", None, False
        )

    def safetyTimerFired_(self, timer: NSTimer) -> None:
        """Emergency stop — spacebar keyUp was never received."""
        self._safety_timer = None
        if self._state == _State.RECORDING:
            logger.warning("Safety timeout — auto-stopping recording")
            self._state = _State.IDLE
            self._on_hold_end(
                shift_held=False,
                enter_held=getattr(self, '_enter_held', False),
            )

    def _cancel_safety_timer(self) -> None:
        if self._safety_timer is not None:
            self._safety_timer.invalidate()
            self._safety_timer = None

    # ── synthetic space forwarding ──────────────────────────

    def _forward_space(self) -> None:
        """Synthesize a space tap (keyDown + keyUp) and post it."""
        self._forwarding = True
        self._start_forwarding_timer()
        src = None  # use default source
        down = CGEventCreateKeyboardEvent(src, SPACEBAR_KEYCODE, True)
        up = CGEventCreateKeyboardEvent(src, SPACEBAR_KEYCODE, False)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        # _forwarding is cleared after both events pass through the tap,
        # or by the forwarding timer if they never arrive

    def _start_forwarding_timer(self) -> None:
        """Auto-clear _forwarding if synthetic events are lost."""
        self._cancel_forwarding_timer()
        self._forwarding_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _FORWARDING_TIMEOUT_S, self, "forwardingTimerFired:", None, False
        )

    def forwardingTimerFired_(self, timer: NSTimer) -> None:
        """Clear stuck _forwarding flag."""
        self._forwarding_timer = None
        if self._forwarding:
            logger.warning("Forwarding timeout — clearing stuck _forwarding flag")
            self._forwarding = False

    def _cancel_forwarding_timer(self) -> None:
        if self._forwarding_timer is not None:
            self._forwarding_timer.invalidate()
            self._forwarding_timer = None


# ── module-level callback ────────────────────────────────────

_active_detector: SpacebarHoldDetector | None = None


def _event_tap_callback(proxy, event_type, event, refcon):
    """Raw CGEventTap callback — must be a plain function, not a method."""
    det = _active_detector
    if det is None:
        return event

    # Let our own forwarded events pass through
    if det._forwarding:
        # Clear after keyUp so both down+up pass
        if event_type == kCGEventKeyUp:
            keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            if keycode == SPACEBAR_KEYCODE:
                det._forwarding = False
                det._cancel_forwarding_timer()
        return event

    keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)

    if event_type == kCGEventKeyDown:
        flags = CGEventGetFlags(event)
        # Track enter key state for command fast path
        if keycode == ENTER_KEYCODE:
            det._enter_held = True
            # If tray is active, Enter = send to assistant.
            # Only suppress Enter if the callback exists and is called.
            # Always let Enter through if tray is not active.
            if getattr(det, 'tray_active', False):
                on_enter = getattr(det, '_on_enter_pressed', None)
                if on_enter is not None:
                    on_enter()
                    return None  # suppress enter during tray
            # Enter passes through to the OS when tray is not active
        if keycode == SPACEBAR_KEYCODE:
            logger.info("keyDown space: flags=%#x shift=%s state=%s",
                        flags, bool(flags & kCGEventFlagMaskShift), det._state)
            # Mark space between shift down/up for tray shift-tap discrimination
            if getattr(det, 'tray_active', False) and getattr(det, '_tray_shift_down', False):
                det._tray_space_between = True
        if det.handle_key_down(keycode, flags):
            return None  # suppress
    elif event_type == kCGEventKeyUp:
        flags = CGEventGetFlags(event)
        # Track enter key release
        if keycode == ENTER_KEYCODE:
            det._enter_held = False
        if keycode == SPACEBAR_KEYCODE:
            logger.info("keyUp space: flags=%#x shift=%s state=%s",
                        flags, bool(flags & kCGEventFlagMaskShift), det._state)
        if det.handle_key_up(keycode, flags=flags):
            return None  # suppress
    elif event_type == kCGEventFlagsChanged:
        flags = CGEventGetFlags(event)
        shift_now = bool(flags & kCGEventFlagMaskShift)

        # Latch shift if it arrives while we're in WAITING or RECORDING
        if shift_now:
            if det._state in (_State.WAITING, _State.RECORDING):
                if not det._shift_latched:
                    logger.info("Shift latched during %s", det._state)
                det._shift_latched = True
                # Track shift press during spacebar hold for tray navigate-up
                if getattr(det, 'tray_active', False):
                    det._shift_down_during_hold = True
        elif not shift_now and det._state in (_State.WAITING, _State.RECORDING):
            # Shift released while spacebar is still held
            if getattr(det, 'tray_active', False) and getattr(det, '_shift_down_during_hold', False):
                det._shift_down_during_hold = False
                # Shift was tapped while spacebar held = navigate up (more recent).
                # Clear _shift_latched so the subsequent spacebar release doesn't
                # also see shift_held=True and fire navigate down.
                det._shift_latched = False
                logger.info("Shift tapped during spacebar hold — navigate up")
                on_tap = getattr(det, '_on_shift_tap_during_hold', None)
                if on_tap is not None:
                    on_tap()

        # Tray mode: track shift press/release for dismiss gesture (IDLE state)
        if getattr(det, 'tray_active', False) and det._state == _State.IDLE:
            if shift_now and not getattr(det, '_tray_shift_down', False):
                # Shift just pressed
                det._tray_shift_down = True
                det._tray_space_between = False
            elif not shift_now and getattr(det, '_tray_shift_down', False):
                # Shift just released
                det._tray_shift_down = False
                if not getattr(det, '_tray_space_between', False):
                    # No spacebar between shift down and up = shift tap (navigate down)
                    on_shift_tap = getattr(det, '_on_shift_tap', None)
                    if on_shift_tap is not None:
                        on_shift_tap()

    return event
