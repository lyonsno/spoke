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
# Modifiers that prevent recording when held during spacebar press.
# Shift is intentionally excluded — shift+space starts recording normally,
# and shift is detected at release to route the utterance as a command.
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
        self._hold_timer: NSTimer | None = None
        self._safety_timer: NSTimer | None = None
        self._forwarding = False
        self._forwarding_timer: NSTimer | None = None
        self._tap = None
        self._tap_source = None
        return self

    # ── public ──────────────────────────────────────────────

    def install(self) -> bool:
        """Install the global event tap. Returns False if permission denied."""
        event_mask = CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp)

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
            self._on_hold_end(shift_held=False)

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
            shift_held = bool(flags & kCGEventFlagMaskShift)
            if shift_held:
                # Shift + quick tap = signal for recall/dismiss (no space)
                self._on_hold_end(shift_held=True)
            else:
                # Normal quick tap = forward a space
                self._forward_space()
            return True

        if self._state == _State.RECORDING:
            self._cancel_safety_timer()
            self._state = _State.IDLE
            shift_held = bool(flags & kCGEventFlagMaskShift)
            self._on_hold_end(shift_held=shift_held)
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
            self._on_hold_end(shift_held=False)

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
        if keycode == SPACEBAR_KEYCODE:
            logger.info("keyDown space: flags=%#x shift=%s state=%s",
                        flags, bool(flags & kCGEventFlagMaskShift), det._state)
        if det.handle_key_down(keycode, flags):
            return None  # suppress
    elif event_type == kCGEventKeyUp:
        flags = CGEventGetFlags(event)
        if keycode == SPACEBAR_KEYCODE:
            logger.info("keyUp space: flags=%#x shift=%s state=%s",
                        flags, bool(flags & kCGEventFlagMaskShift), det._state)
        if det.handle_key_up(keycode, flags=flags):
            return None  # suppress

    return event
