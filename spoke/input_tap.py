"""CGEventTap-based spacebar hold detection.

Installs a global event tap that distinguishes spacebar *taps* (forwarded
as normal space characters) from spacebar *holds* (triggers recording).

State machine:
    IDLE ──[space keyDown]──> WAITING  (suppress event, start timer)
    WAITING ──[space keyUp before timer]──> IDLE  (forward synth space)
    WAITING ──[timer fires]──> RECORDING  (call on_hold_start)
    RECORDING ──[space keyUp]──> IDLE  (call on_hold_end)
    RECORDING ──[shift tap]──> LATCHED  (capture continues hands-free)
    LATCHED ──[space press+release]──> IDLE  (call on_hold_end)

Additional gestures (IDLE):
    Enter + route key (] or number row) ──> send chord (tray send)
    Space + Delete (overlay visible) ──> cancel generation

Enter is deloaded from recording routing. It no longer selects the
assistant destination during recording — route keys handle that now.
The send chord (Enter + route key) replaces the old enter-held pathway.
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
    CGEventGetTimestamp,
    CGEventMaskBit,
    CGEventPost,
    CGEventSourceKeyState,
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
    kCGEventTapDisabledByTimeout,
    kCGEventTapDisabledByUserInput,
    kCGEventTapOptionDefault,
    kCGHeadInsertEventTap,
    kCGHIDEventTap,
    kCGKeyboardEventKeycode,
    kCGEventSourceStateCombinedSessionState,
    kCGSessionEventTap,
)
logger = logging.getLogger(__name__)

SPACEBAR_KEYCODE = 49
RETURN_KEYCODE = 36
ENTER_KEYCODE = RETURN_KEYCODE
KEYPAD_ENTER_KEYCODE = 76
ENTER_KEYCODES = frozenset({ENTER_KEYCODE, KEYPAD_ENTER_KEYCODE})
DELETE_KEYCODE = 51
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
_ENTER_RELEASE_GRACE_S = 0.15  # small post-release grace so Enter can land a beat late
_DOUBLE_TAP_WINDOW_S = 0.3  # 300ms window for double-tap detection
_ENTER_STALE_GRACE_S = 0.75  # stale enter observations must not poison later space taps
_REPEAT_RELEASE_GRACE_S = 0.35  # after repeat has started, missed keyUp should recover quickly
_REPEAT_QUIET_RECOVERY_S = 0.75  # require a real quiet gap before trusting a false-up probe


def _current_enter_key_state() -> bool | None:
    """Return the real current Enter key state when Quartz exposes it.

    This lets a fresh space-rooted gesture correct stale `_enter_held` state if
    we ever missed a trailing Enter keyUp from an earlier consumed chord.
    """
    try:
        return any(
            bool(CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState, keycode))
            for keycode in ENTER_KEYCODES
        )
    except Exception:
        logger.debug("Could not query current Enter key state", exc_info=True)
        return None


def _current_space_key_state() -> bool | None:
    """Return the real current spacebar key state when Quartz exposes it."""
    try:
        return bool(
            CGEventSourceKeyState(
                kCGEventSourceStateCombinedSessionState,
                SPACEBAR_KEYCODE,
            )
        )
    except Exception:
        logger.debug("Could not query current Space key state", exc_info=True)
        return None


def _event_timestamp_ns(event) -> int | None:
    """Return the Quartz event timestamp when available."""
    try:
        return int(CGEventGetTimestamp(event))
    except Exception:
        logger.debug("Could not query event timestamp", exc_info=True)
        return None


class _State(Enum):
    IDLE = auto()
    WAITING = auto()
    RECORDING = auto()
    LATCHED = auto()


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
        self._enter_observed = False  # True if we actually saw an Enter keyDown since idle
        self._enter_latched = False  # True if enter joined a latched-exit chord
        self._enter_last_down_monotonic = 0.0
        self._on_cancel_spring_start: Callable[[], None] | None = None
        self._on_cancel_spring_release: Callable[[], None] | None = None
        self.cancel_spring_active = False  # set by app when spring is winding
        self._suppress_enter_keyup = False  # swallow trailing Enter keyUp after a consumed chord
        self._suppress_delete_keyup = False
        self._hold_timer: NSTimer | None = None
        self._safety_timer: NSTimer | None = None
        self._repeat_watchdog_timer: NSTimer | None = None
        self._forwarding = False
        self._forwarding_timer: NSTimer | None = None
        self._release_decision_timer: NSTimer | None = None
        self._tap = None
        self._tap_source = None
        self._awaiting_space_release = False
        self._latched_space_down = False
        self._latched_space_released = False
        self._pending_release_active = False
        self._pending_release_shift_held = False
        self._space_keydown_timestamp_ns: int | None = None
        self._last_space_keydown_monotonic = 0.0

        # Tray mode support — set by the delegate when tray is active.
        # When True, quick spacebar taps call on_hold_end instead of
        # forwarding a space character, and shift gestures route to
        # tray navigation callbacks.
        self.tray_active = False
        # Tracks whether the command overlay is visible — used by the
        # delegate for toggle logic but no longer drives event suppression.
        self.command_overlay_active = False
        # Tracks whether a host-side approval card is active on the command
        # surface. When True, Enter/Delete can route approval actions from IDLE
        # without repurposing space-rooted recording gestures.
        self.approval_active = False
        self._on_shift_tap: Callable[[], None] | None = None
        self._on_shift_tap_during_hold: Callable[[], None] | None = None
        self._on_shift_tap_idle: Callable[[], None] | None = None
        self._on_enter_pressed: Callable[[], None] | None = None
        self._on_tray_delete: Callable[[], None] | None = None
        self._on_approval_enter_pressed: Callable[[bool], None] | None = None
        self._on_approval_delete_pressed: Callable[[], None] | None = None
        self._tray_shift_down = False
        self._tray_space_between = False
        self._idle_shift_down = False
        self._idle_shift_interrupted = False
        self._shift_down_during_hold = False  # tracks shift press while spacebar held
        self._tray_gesture_consumed = False  # True if a tray gesture already fired this hold
        # Double-tap detection for delete gesture (shift held + double-tap spacebar)
        self._tray_last_shift_space_up: float = 0.0

        # Double-tap detection for Shift (toggle HUD)
        self._on_double_tap_shift: Callable[[], None] | None = None
        self._last_idle_shift_up: float = 0.0
        self._shift_single_tap_timer: NSTimer | None = None

        # Enter during WAITING = toggle command overlay (before hold threshold fires)
        self._on_enter_during_waiting: Callable[[], None] | None = None

        # Route key selection — set by delegate to enable route key interception
        # during RECORDING/LATCHED states.
        self._route_key_selector = None

        # Send chord — set by delegate. Called with keycode= when Enter + route
        # key fires from tray.
        self._on_send_chord: Callable[..., None] | None = None

        # Cancel generation — set by delegate. Called when Space + Delete fires
        # while command overlay is visible.
        self._on_cancel_generation: Callable[[], None] | None = None

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
        if self._state in (_State.RECORDING, _State.LATCHED):
            self._cancel_safety_timer()
            self._cancel_repeat_watchdog()
            self._state = _State.IDLE
            self._reset_route_keys()
            self._awaiting_space_release = True
            # enter_held is always False — enter is deloaded from recording
            # routing. Route keys handle destination selection now.
            self._on_hold_end(
                shift_held=False,
                enter_held=False,
            )

    def _finish_enter_release(self, shift_held: bool = False) -> None:
        """End an active space-rooted chord because Enter was released first.

        Enter is deloaded from recording routing — enter_held is always False.
        This method still ends the recording (enter release as a gesture exit)
        but does not route via the enter_held pathway.
        """
        source_state = self._state
        if source_state == _State.WAITING:
            self._cancel_hold_timer()
        elif source_state in (_State.RECORDING, _State.LATCHED):
            self._cancel_safety_timer()
            self._cancel_repeat_watchdog()
        if source_state == _State.LATCHED:
            self._latched_space_down = False
        self._state = _State.IDLE
        self._reset_route_keys()
        self._awaiting_space_release = True
        self._shift_latched = False
        self._enter_latched = False
        # enter_held=False: enter is deloaded from recording routing
        self._on_hold_end(shift_held=shift_held, enter_held=False)

    def uninstall(self) -> None:
        """Disable and remove the event tap."""
        if self._tap is not None:
            CGEventTapEnable(self._tap, False)
            self._tap = None
            self._tap_source = None
        self._cancel_hold_timer()
        self._cancel_safety_timer()
        self._cancel_forwarding_timer()
        self._cancel_release_decision_timer()
        self._cancel_shift_single_tap_timer()
        self._forwarding = False
        self._enter_held = False
        self._enter_observed = False
        self._enter_latched = False
        self._enter_last_down_monotonic = 0.0
        self._suppress_enter_keyup = False
        self._suppress_delete_keyup = False
        self._awaiting_space_release = False
        self._latched_space_down = False
        self._pending_release_active = False
        self._pending_release_shift_held = False
        self._space_keydown_timestamp_ns = None
        self.tray_active = False
        self.command_overlay_active = False
        self._idle_shift_down = False
        self._idle_shift_interrupted = False
        self._state = _State.IDLE

        global _active_detector  # noqa: PLW0603
        _active_detector = None
        logger.info("CGEventTap uninstalled")

    # ── event handling (called from C callback, main thread) ─

    def handle_key_down(self, keycode: int, flags: int) -> bool:
        """Handle a keyDown event. Returns True to suppress, False to pass through."""
        if keycode != SPACEBAR_KEYCODE:
            return False

        self._last_space_keydown_monotonic = time.monotonic()

        if getattr(self, "_awaiting_space_release", False):
            return True

        if getattr(self, "_pending_release_active", False):
            self._finish_pending_release(enter_held=False)

        # Pass through if any modifier is held (Cmd+Space, etc.)
        if flags & _MODIFIER_MASK:
            return False

        if self._state == _State.IDLE:
            self._state = _State.WAITING
            self._shift_at_press = bool(flags & kCGEventFlagMaskShift)
            self._shift_latched = self._shift_at_press
            self._tray_gesture_consumed = False
            self._shift_down_during_hold = False
            self._start_hold_timer()
            return True  # suppress the space

        if self._state == _State.LATCHED:
            self._shift_at_press = bool(flags & kCGEventFlagMaskShift)
            self._shift_latched = self._shift_at_press
            # Only count as a fresh press if the spacebar was fully released
            # since entering LATCHED.  Key-repeats from the original hold
            # arrive as keyDown events but should not arm the exit path.
            if getattr(self, '_latched_space_released', False):
                self._latched_space_down = True
            return True

        if self._state == _State.RECORDING:
            self._start_repeat_watchdog()

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

        if getattr(self, "_awaiting_space_release", False):
            self._awaiting_space_release = False
            self._cancel_repeat_watchdog()
            return True

        # Cancel spring capture: space released while spring is winding
        if getattr(self, 'cancel_spring_active', False):
            self._cancel_hold_timer()
            self._cancel_safety_timer()
            self._cancel_repeat_watchdog()
            self._suppress_enter_keyup = getattr(self, '_enter_held', False)
            self._state = _State.IDLE
            cb = getattr(self, '_on_cancel_spring_release', None)
            if cb is not None:
                cb()
            return True

        if self._state == _State.WAITING:
            # Released before hold threshold
            self._cancel_hold_timer()
            self._suppress_enter_keyup = getattr(self, '_enter_held', False)
            self._state = _State.IDLE
            # If a tray gesture already fired during this hold (e.g. shift-tap
            # navigate), swallow the spacebar release — don't insert or navigate.
            if getattr(self, '_tray_gesture_consumed', False):
                self._tray_gesture_consumed = False
                self._shift_latched = False
                self._enter_latched = False
                return True
            shift_held = bool(flags & kCGEventFlagMaskShift) or self._shift_latched
            # enter_held is deloaded from recording routing — always False.
            # Send chord (Enter + route key) handles enter-based sending now.
            self._shift_latched = False
            self._enter_latched = False
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
                self._on_hold_end(shift_held=shift_held, enter_held=False)
            elif shift_held:
                # Shift + quick tap = signal for tray recall (no space)
                self._on_hold_end(shift_held=True, enter_held=False)
            else:
                # Normal quick tap = forward a space
                self._forward_space()
            return True

        if self._state == _State.RECORDING:
            self._cancel_safety_timer()
            self._cancel_repeat_watchdog()
            self._suppress_enter_keyup = getattr(self, '_enter_held', False)
            self._state = _State.IDLE
            self._reset_route_keys()
            if getattr(self, '_tray_gesture_consumed', False):
                self._tray_gesture_consumed = False
                self._shift_latched = False
                self._enter_latched = False
                return True
            shift_held = bool(flags & kCGEventFlagMaskShift) or self._shift_latched
            # enter_held deloaded — route keys handle destination selection
            self._shift_latched = False
            self._enter_latched = False
            if getattr(self, 'tray_active', False):
                # Tray-intercepted holds are tray gestures, not recording-release
                # decisions. Keep them on the tray path even if shift came up
                # before spacebar.
                self._on_hold_end(shift_held=shift_held, enter_held=False)
                return True
            if shift_held:
                self._start_release_decision_timer(shift_held=True)
            else:
                self._on_hold_end(shift_held=False, enter_held=False)
            return True

        if self._state == _State.LATCHED:
            shift_held = bool(flags & kCGEventFlagMaskShift) or self._shift_latched
            # enter_held deloaded from recording routing
            self._shift_latched = False

            if getattr(self, '_latched_space_down', False):
                self._latched_space_down = False
                self._cancel_safety_timer()
                self._cancel_repeat_watchdog()
                self._suppress_enter_keyup = getattr(self, '_enter_held', False)
                self._state = _State.IDLE
                self._reset_route_keys()
                self._enter_latched = False
                # enter_held deloaded — route keys handle destination
                self._on_hold_end(shift_held=shift_held, enter_held=False)
                return True

            # Swallow the original release that let the user go hands-free,
            # and mark that the spacebar is now fully released so a fresh
            # press+release cycle can arm the exit path.
            self._latched_space_released = True
            return True

        return False

    def handle_route_key_down(self, keycode: int) -> bool:
        """Handle a route key keyDown. Returns True to suppress during recording.

        Route keys are only intercepted during RECORDING or LATCHED states.
        In IDLE or WAITING, the key passes through to the OS.
        """
        selector = getattr(self, "_route_key_selector", None)
        if selector is None:
            return False
        if self._state not in (_State.RECORDING, _State.LATCHED):
            return False
        selector.tap(keycode)
        return True

    def handle_send_chord(self, keycode: int) -> bool:
        """Handle a potential send chord (Enter + route key from tray).

        Returns True to suppress the key event if a send chord fires.
        The send chord fires when:
        - State is IDLE
        - Tray is active
        - Enter is currently held
        - The keycode is a valid route key

        The callback _on_send_chord is called with the keycode so the
        delegate can determine the destination.
        """
        if self._state != _State.IDLE:
            return False
        if not getattr(self, 'tray_active', False):
            return False
        if not getattr(self, '_enter_held', False):
            return False

        # Validate the keycode is a known route key
        from spoke.route_keys import ALL_ROUTE_KEYCODES
        if keycode not in ALL_ROUTE_KEYCODES:
            return False

        cb = getattr(self, '_on_send_chord', None)
        if cb is None:
            return False

        logger.info("Send chord detected: Enter + keycode=%d", keycode)
        self._suppress_enter_keyup = True
        self._enter_held = False
        self._enter_observed = False
        self._enter_last_down_monotonic = 0.0
        cb(keycode=keycode)
        return True

    def handle_cancel_gesture(self, keycode: int) -> bool:
        """Handle Space + Delete cancel gesture.

        Returns True to suppress the key event if cancel fires.
        Fires when:
        - Space is held (WAITING or RECORDING)
        - Command overlay is visible
        - The keycode is DELETE

        Cancels the current generation and returns to IDLE.
        """
        if keycode != DELETE_KEYCODE:
            return False
        if self._state not in (_State.WAITING, _State.RECORDING):
            return False
        if not getattr(self, 'command_overlay_active', False):
            return False

        cb = getattr(self, '_on_cancel_generation', None)
        if cb is None:
            return False

        logger.info("Cancel gesture detected: Space + Delete")
        self._cancel_hold_timer()
        if self._state == _State.RECORDING:
            self._cancel_safety_timer()
            self._cancel_repeat_watchdog()
        self._state = _State.IDLE
        self._awaiting_space_release = True
        self._suppress_delete_keyup = True
        cb()
        return True

    def _reset_route_keys(self) -> None:
        """Reset route key selection. Called at end of every recording."""
        selector = getattr(self, "_route_key_selector", None)
        if selector is not None:
            selector.reset()

    # ── timers ──────────────────────────────────────────────

    def _start_hold_timer(self) -> None:
        self._hold_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self._hold_s, self, "holdTimerFired:", None, False
        )

    def _promote_waiting_to_recording(self) -> None:
        """Transition WAITING -> RECORDING and invoke hold-start once."""
        self._cancel_hold_timer()
        self._state = _State.RECORDING
        self._start_safety_timer()
        self._on_hold_start()

    def _waiting_elapsed_meets_threshold(self, event_timestamp_ns: int | None) -> bool:
        start_ns = getattr(self, "_space_keydown_timestamp_ns", None)
        if start_ns is None or event_timestamp_ns is None:
            return False
        return (event_timestamp_ns - start_ns) >= int(self._hold_s * 1_000_000_000)

    def holdTimerFired_(self, timer: NSTimer) -> None:
        """Called when spacebar has been held past the threshold."""
        self._hold_timer = None
        if self._state != _State.WAITING:
            return
        self._promote_waiting_to_recording()

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
        if self._state in (_State.RECORDING, _State.LATCHED):
            logger.warning("Safety timeout — auto-stopping recording")
            self._state = _State.IDLE
            self._cancel_repeat_watchdog()
            self._awaiting_space_release = True
            # enter_held deloaded from recording routing
            self._on_hold_end(shift_held=False, enter_held=False)

    def _cancel_safety_timer(self) -> None:
        if self._safety_timer is not None:
            self._safety_timer.invalidate()
            self._safety_timer = None

    def _start_repeat_watchdog(self) -> None:
        self._cancel_repeat_watchdog()
        self._repeat_watchdog_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _REPEAT_RELEASE_GRACE_S, self, "repeatWatchdogFired:", None, False
        )

    def _cancel_repeat_watchdog(self) -> None:
        timer = getattr(self, "_repeat_watchdog_timer", None)
        if timer is not None:
            timer.invalidate()
            self._repeat_watchdog_timer = None

    def repeatWatchdogFired_(self, timer: NSTimer) -> None:
        """Recover a hold if repeat had started but the matching keyUp vanished."""
        self._repeat_watchdog_timer = None
        if self._state != _State.RECORDING:
            return
        quiet_for = time.monotonic() - getattr(self, "_last_space_keydown_monotonic", 0.0)
        if quiet_for < _REPEAT_QUIET_RECOVERY_S:
            self._start_repeat_watchdog()
            return
        is_space_down = _current_space_key_state()
        if is_space_down is not False:
            return
        self._recover_missed_space_release("repeat watchdog")

    def _recover_missed_space_release(self, reason: str) -> bool:
        """End an active hold after Quartz confirms Space is physically up."""
        if self._state not in (_State.RECORDING, _State.LATCHED):
            return False
        is_space_down = _current_space_key_state()
        if is_space_down is not False:
            return False

        logger.warning("%s recovered a missed spacebar keyUp", reason)
        source_state = self._state
        self._cancel_safety_timer()
        self._cancel_repeat_watchdog()
        self._state = _State.IDLE
        # Quartz confirmed the physical key is up. Requiring another release
        # would swallow the user's next fresh press behind a phantom keyUp.
        self._awaiting_space_release = False
        # enter_held deloaded from recording routing
        self._on_hold_end(
            shift_held=self._shift_latched,
            enter_held=False,
        )
        return True

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

    def _start_release_decision_timer(self, shift_held: bool) -> None:
        self._cancel_release_decision_timer()
        self._pending_release_active = True
        self._pending_release_shift_held = shift_held
        self._release_decision_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                _ENTER_RELEASE_GRACE_S,
                self,
                "releaseDecisionTimerFired:",
                None,
                False,
            )
        )

    def releaseDecisionTimerFired_(self, timer: NSTimer) -> None:
        self._release_decision_timer = None
        self._finish_pending_release(enter_held=False)

    def _cancel_release_decision_timer(self) -> None:
        if getattr(self, "_release_decision_timer", None) is not None:
            self._release_decision_timer.invalidate()
            self._release_decision_timer = None

    def _cancel_shift_single_tap_timer(self) -> None:
        if getattr(self, "_shift_single_tap_timer", None) is not None:
            self._shift_single_tap_timer.invalidate()
            self._shift_single_tap_timer = None

    def _shiftSingleTapFired_(self, timer: NSTimer) -> None:
        """Double-tap window expired — fire the deferred single-tap action."""
        self._shift_single_tap_timer = None
        on_shift_tap_idle = getattr(self, '_on_shift_tap_idle', None)
        if on_shift_tap_idle is not None:
            on_shift_tap_idle()

    def _finish_pending_release(self, enter_held: bool) -> None:
        if not getattr(self, "_pending_release_active", False):
            return
        shift_held = getattr(self, "_pending_release_shift_held", False)
        self._cancel_release_decision_timer()
        self._pending_release_active = False
        self._pending_release_shift_held = False
        # enter_held deloaded from recording routing — always pass False
        self._on_hold_end(shift_held=shift_held, enter_held=False)

    def _enter_observation_is_fresh(self) -> bool:
        last_down = getattr(self, "_enter_last_down_monotonic", 0.0) or 0.0
        if last_down <= 0.0:
            return False
        return (time.monotonic() - last_down) <= _ENTER_STALE_GRACE_S


# ── module-level callback ────────────────────────────────────

_active_detector: SpacebarHoldDetector | None = None


def _event_tap_callback(proxy, event_type, event, refcon):
    """Raw CGEventTap callback — must be a plain function, not a method."""
    det = _active_detector
    if det is None:
        return event

    if event_type == kCGEventTapDisabledByTimeout:
        if getattr(det, "_tap", None) is not None:
            CGEventTapEnable(det._tap, True)
            logger.warning("CGEventTap disabled by timeout — re-enabled")
        recover = getattr(det, "_recover_missed_space_release", None)
        if callable(recover):
            recover("event tap timeout")
        return event

    if event_type == kCGEventTapDisabledByUserInput:
        logger.warning("CGEventTap disabled by user input")
        return event

    # Let our own forwarded quick-tap events pass through only while idle.
    # A stale forwarding flag during an active hold must not leak physical
    # space repeats into the focused app while recording continues.
    forwarding_state = getattr(det, "_state", _State.IDLE)
    if not isinstance(forwarding_state, _State):
        forwarding_state = _State.IDLE
    if det._forwarding and forwarding_state == _State.IDLE:
        # Clear after keyUp so both down+up pass
        if event_type == kCGEventKeyUp:
            keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
            if keycode == SPACEBAR_KEYCODE:
                det._forwarding = False
                det._cancel_forwarding_timer()
        return event
    if det._forwarding:
        det._forwarding = False
        det._cancel_forwarding_timer()

    keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)

    if event_type == kCGEventKeyDown:
        flags = CGEventGetFlags(event)
        event_timestamp_ns = _event_timestamp_ns(event)
        if keycode == DELETE_KEYCODE and getattr(det, "approval_active", False) and det._state == _State.IDLE:
            det._suppress_delete_keyup = True
            cb = getattr(det, "_on_approval_delete_pressed", None)
            if cb is not None:
                cb()
            return None
        # Track enter key state for command fast path
        if keycode in ENTER_KEYCODES:
            if getattr(det, "approval_active", False) and det._state == _State.IDLE:
                det._suppress_enter_keyup = True
                det._enter_held = False
                det._enter_observed = False
                det._enter_last_down_monotonic = 0.0
                cb = getattr(det, "_on_approval_enter_pressed", None)
                if cb is not None:
                    cb(shift_held=bool(flags & kCGEventFlagMaskShift))
                return None
            det._enter_held = True
            det._enter_observed = True
            det._enter_last_down_monotonic = time.monotonic()
            if getattr(det, "_pending_release_active", False):
                det._finish_pending_release(enter_held=False)
                return None
            # Grace-window intercept: Enter during idle grace period cancels
            # the pending insert and toggles the overlay instead.
            grace_cb = getattr(det, '_on_enter_cancel_grace', None)
            if grace_cb is not None and det._state == _State.IDLE:
                det._enter_held = False
                logger.info("Enter during grace window — cancelling insert")
                grace_cb()
                return None
            if det._state == _State.WAITING:
                if not getattr(det, 'tray_active', False):
                    # Enter during WAITING (before hold threshold) = toggle
                    # assistant overlay. Cancel the hold timer and return to
                    # IDLE so no recording starts.
                    det._cancel_hold_timer()
                    det._state = _State.IDLE
                    det._enter_held = False
                    det._suppress_enter_keyup = True
                    cb = getattr(det, '_on_enter_during_waiting', None)
                    if cb is not None:
                        logger.info("Enter during WAITING — toggling command overlay")
                        cb()
                return None  # suppress enter during WAITING regardless
            if det._state == _State.RECORDING:
                # Fire cancel spring callback if the command overlay is active
                # (generation in progress and user is adding enter to the hold)
                if getattr(det, 'command_overlay_active', False):
                    cb = getattr(det, '_on_cancel_spring_start', None)
                    if cb is not None:
                        cb()
                return None  # suppress enter while space is held
            if det._state == _State.LATCHED and getattr(det, '_latched_space_down', False):
                det._enter_latched = True
                return None  # suppress enter while a latched exit chord is active
        if det._state == _State.IDLE and getattr(det, '_idle_shift_down', False):
            det._idle_shift_interrupted = True
        if keycode == SPACEBAR_KEYCODE:
            logger.debug("keyDown space: flags=%#x shift=%s state=%s",
                         flags, bool(flags & kCGEventFlagMaskShift), det._state)
            if det._state == _State.IDLE:
                actual_enter_held = _current_enter_key_state()
                if actual_enter_held is not None:
                    if actual_enter_held:
                        if (
                            getattr(det, '_enter_observed', False)
                            and det._enter_observation_is_fresh()
                        ):
                            det._enter_held = True
                        else:
                            # A bare Quartz "Enter is down" probe can stay
                            # wedged true across a missed keyUp or relaunch.
                            # Only trust it for a fresh space-rooted gesture
                            # if this process actually saw a recent Enter
                            # keyDown that could own the chord.
                            det._enter_held = False
                            det._enter_observed = False
                            det._enter_last_down_monotonic = 0.0
                    else:
                        det._enter_held = False
                        det._enter_observed = False
                        det._enter_last_down_monotonic = 0.0
                elif (
                    not getattr(det, '_enter_observed', False)
                    or not det._enter_observation_is_fresh()
                ):
                    # If Quartz cannot answer and we have not actually seen
                    # a fresh Enter keyDown since returning idle, do not let a
                    # stale command-chord state poison a fresh space press.
                    det._enter_held = False
                    det._enter_observed = False
                    det._enter_last_down_monotonic = 0.0
                det._space_keydown_timestamp_ns = event_timestamp_ns
            elif det._state == _State.WAITING and det._waiting_elapsed_meets_threshold(
                event_timestamp_ns
            ):
                logger.info("Delayed repeat crossed hold threshold — promoting to recording")
                det._promote_waiting_to_recording()
            # Mark space between shift down/up for tray shift-tap discrimination
            if getattr(det, 'tray_active', False) and getattr(det, '_tray_shift_down', False):
                det._tray_space_between = True
        # Send chord: Enter + route key from tray (IDLE)
        if det.handle_send_chord(keycode):
            return None  # suppress — send chord fired
        # Cancel gesture: Space + Delete while overlay visible
        if det.handle_cancel_gesture(keycode):
            return None  # suppress — cancel fired
        # Route key interception during RECORDING/LATCHED
        if det.handle_route_key_down(keycode):
            return None  # suppress route key
        if det.handle_key_down(keycode, flags):
            return None  # suppress
    elif event_type == kCGEventKeyUp:
        flags = CGEventGetFlags(event)
        if keycode == DELETE_KEYCODE and getattr(det, '_suppress_delete_keyup', False):
            det._suppress_delete_keyup = False
            return None
        # Track enter key release
        if keycode in ENTER_KEYCODES:
            # Cancel spring capture: either key releasing evaluates the spring
            if getattr(det, 'cancel_spring_active', False):
                det._enter_held = False
                det._enter_observed = False
                det._enter_last_down_monotonic = 0.0
                cb = getattr(det, '_on_cancel_spring_release', None)
                if cb is not None:
                    cb()
                return None
            if getattr(det, '_suppress_enter_keyup', False):
                det._suppress_enter_keyup = False
                det._enter_held = False
                det._enter_observed = False
                det._enter_last_down_monotonic = 0.0
                return None
            if det._state in (_State.WAITING, _State.RECORDING):
                shift_held = bool(flags & kCGEventFlagMaskShift) or getattr(
                    det, '_shift_latched', False
                )
                det._enter_held = False
                det._enter_observed = False
                det._enter_last_down_monotonic = 0.0
                det._finish_enter_release(shift_held=shift_held)
                return None
            if det._state == _State.LATCHED and getattr(det, '_latched_space_down', False):
                shift_held = bool(flags & kCGEventFlagMaskShift) or getattr(
                    det, '_shift_latched', False
                )
                det._enter_held = False
                det._enter_observed = False
                det._enter_last_down_monotonic = 0.0
                det._finish_enter_release(shift_held=shift_held)
                return None
            det._enter_held = False
            det._enter_observed = False
            det._enter_last_down_monotonic = 0.0
        if det._state == _State.IDLE and getattr(det, '_idle_shift_down', False):
            det._idle_shift_interrupted = True
        if keycode == SPACEBAR_KEYCODE:
            logger.debug("keyUp space: flags=%#x shift=%s state=%s",
                         flags, bool(flags & kCGEventFlagMaskShift), det._state)
            det._space_keydown_timestamp_ns = None
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
                det._tray_gesture_consumed = True
                logger.info("Shift tapped during spacebar hold — navigate up")
                on_tap = getattr(det, '_on_shift_tap_during_hold', None)
                if on_tap is not None:
                    on_tap()
            elif (
                det._state == _State.RECORDING
                and det._shift_latched
                and not getattr(det, 'tray_active', False)
            ):
                det._shift_latched = False
                det._latched_space_down = False
                det._latched_space_released = False
                det._state = _State.LATCHED
                logger.info("Shift tapped during active recording — latched recording")

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
        elif det._state == _State.IDLE:
            if shift_now and not getattr(det, '_idle_shift_down', False):
                det._idle_shift_down = True
                det._idle_shift_interrupted = False
            elif not shift_now and getattr(det, '_idle_shift_down', False):
                det._idle_shift_down = False
                if not getattr(det, '_idle_shift_interrupted', False):
                    # Double-tap Shift detection (not during tray)
                    now = time.monotonic()
                    last = getattr(det, '_last_idle_shift_up', 0.0)
                    if (now - last) < _DOUBLE_TAP_WINDOW_S:
                        det._last_idle_shift_up = 0.0  # reset
                        # Cancel the deferred single-tap from the first tap
                        det._cancel_shift_single_tap_timer()
                        cb = getattr(det, '_on_double_tap_shift', None)
                        if cb is not None:
                            logger.info("Double-tap Shift — toggling HUD")
                            cb()
                    else:
                        det._last_idle_shift_up = now
                        # Defer single-tap action until double-tap window expires.
                        # If a second tap arrives, the timer is cancelled.
                        det._cancel_shift_single_tap_timer()
                        det._shift_single_tap_timer = (
                            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                                _DOUBLE_TAP_WINDOW_S, det, "_shiftSingleTapFired:", None, False
                            )
                        )
                det._idle_shift_interrupted = False

    return event
