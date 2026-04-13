"""Tests for the CGEventTap spacebar hold detection state machine."""

from unittest.mock import MagicMock, call, patch


class TestSpacebarStateMachine:
    """Test the IDLE → WAITING → RECORDING state transitions."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        # Manual init to avoid objc.super issues in test
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._repeat_watchdog_timer = None
        det._last_space_keydown_monotonic = 0.0
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        return det, on_start, on_end

    def test_tap_spacebar_passes_through_on_quick_release(self, input_tap_module):
        """Quick tap: keyDown → keyUp before timer → should forward space."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # keyDown should suppress (returns True) and enter WAITING
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._state == mod._State.WAITING

        # keyUp before timer fires → back to IDLE, forward space
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE) is True
        assert det._state == mod._State.IDLE
        assert det._forwarding is True  # synth space posted

        on_start.assert_not_called()
        on_end.assert_not_called()

    def test_hold_spacebar_triggers_recording(self, input_tap_module):
        """Hold: keyDown → timer fires → should call on_hold_start."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # keyDown → WAITING
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        # Simulate timer firing
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING
        on_start.assert_called_once()

    def test_release_after_hold_triggers_end(self, input_tap_module):
        """Hold then release: should call on_hold_end."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)  # → RECORDING
        assert det._state == mod._State.RECORDING

        det.handle_key_up(mod.SPACEBAR_KEYCODE)
        assert det._state == mod._State.IDLE
        on_end.assert_called_once()

    def test_modifier_keys_pass_through(self, input_tap_module):
        """Cmd+Space, Ctrl+Space, Alt+Space should not be intercepted."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module
        Quartz = __import__("Quartz")

        for flag in [
            Quartz.kCGEventFlagMaskCommand,
            Quartz.kCGEventFlagMaskControl,
            Quartz.kCGEventFlagMaskAlternate,
        ]:
            result = det.handle_key_down(mod.SPACEBAR_KEYCODE, flag)
            assert result is False, f"Should pass through with modifier flag {flag:#x}"
            assert det._state == mod._State.IDLE

    def test_non_spacebar_keys_pass_through(self, input_tap_module):
        """Other keys should never be intercepted."""
        det, _, _ = self._make_detector(input_tap_module)

        for keycode in [0, 1, 13, 36, 48, 50, 51]:
            assert det.handle_key_down(keycode, 0) is False
            assert det.handle_key_up(keycode) is False

    def test_key_repeat_suppressed_while_waiting(self, input_tap_module):
        """Repeated keyDown events while WAITING should be suppressed."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)  # → WAITING
        # Repeated keyDown (key repeat)
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._state == mod._State.WAITING  # still waiting

    def test_key_repeat_suppressed_while_recording(self, input_tap_module):
        """Repeated keyDown events while RECORDING should be suppressed."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)  # → RECORDING
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._state == mod._State.RECORDING

    def test_repeat_watchdog_recovers_when_keyup_is_missed(self, input_tap_module):
        """If repeat began and the release never arrives, watchdog recovery should
        end the hold once Quartz reports space is now up."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, on_start, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)  # -> RECORDING
        assert det._state == mod._State.RECORDING

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)  # repeat while recording
        Quartz.CGEventSourceKeyState.return_value = False

        with patch.object(mod.time, "monotonic", return_value=11.0):
            det._last_space_keydown_monotonic = 10.0
            det.repeatWatchdogFired_(None)

        assert det._state == mod._State.IDLE
        assert det._awaiting_space_release is True
        on_start.assert_called_once()
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

    def test_repeat_watchdog_ignores_false_release_while_repeats_are_recent(
        self, input_tap_module
    ):
        """A transient false Quartz probe must not end a hold if repeats were
        still arriving recently."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, on_start, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)  # -> RECORDING
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)  # repeat while recording
        Quartz.CGEventSourceKeyState.return_value = False

        with patch.object(mod.time, "monotonic", return_value=10.2):
            det._last_space_keydown_monotonic = 10.0
            det.repeatWatchdogFired_(None)

        assert det._state == mod._State.RECORDING
        assert det._awaiting_space_release is False
        on_start.assert_called_once()
        on_end.assert_not_called()

    def test_safety_timer_stops_recording(self, input_tap_module):
        """Safety timeout should auto-stop recording."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)  # → RECORDING
        assert det._state == mod._State.RECORDING

        det.safetyTimerFired_(None)
        assert det._state == mod._State.IDLE
        on_end.assert_called_once()

    def test_safety_timer_noop_if_not_recording(self, input_tap_module):
        """Safety timer firing when not RECORDING should be a no-op."""
        det, _, on_end = self._make_detector(input_tap_module)

        # Fire safety timer while IDLE
        det.safetyTimerFired_(None)
        on_end.assert_not_called()

    def test_hold_timer_noop_if_not_waiting(self, input_tap_module):
        """Hold timer firing when not WAITING should be a no-op."""
        det, on_start, _ = self._make_detector(input_tap_module)

        # Fire hold timer while IDLE (e.g., after quick release cancelled it late)
        det.holdTimerFired_(None)
        on_start.assert_not_called()

    def test_keyup_while_idle_passes_through(self, input_tap_module):
        """keyUp for spacebar while IDLE should pass through."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE) is False


class TestEventTapCallback:
    """Test the module-level _event_tap_callback function."""

    def test_callback_returns_event_when_no_detector(self, input_tap_module):
        """With no active detector, events pass through."""
        mod = input_tap_module
        mod._active_detector = None
        event = MagicMock()
        result = mod._event_tap_callback(None, mod.kCGEventKeyDown, event, None)
        assert result is event

    def test_callback_suppresses_spacebar_keydown(self, input_tap_module):
        """With active detector, spacebar keyDown should be suppressed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = MagicMock()
        det._forwarding = False
        det.handle_key_down = MagicMock(return_value=True)
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0

        event = MagicMock()
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result is None  # suppressed

    def test_callback_passes_forwarded_events(self, input_tap_module):
        """When _forwarding is True, events pass through."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = MagicMock()
        det._forwarding = True
        mod._active_detector = det

        event = MagicMock()
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result is event  # passed through

    def test_forwarding_cleared_on_space_keyup(self, input_tap_module):
        """_forwarding flag should be cleared after forwarded space keyUp."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = MagicMock()
        det._forwarding = True
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE

        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)
        assert det._forwarding is False

    def test_timeout_disable_event_reenables_tap(self, input_tap_module):
        """Quartz timeout disable should immediately re-enable the event tap."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = MagicMock()
        det._forwarding = False
        det._tap = MagicMock()
        mod._active_detector = det

        event = MagicMock()
        result = mod._event_tap_callback(
            None,
            Quartz.kCGEventTapDisabledByTimeout,
            event,
            None,
        )

        Quartz.CGEventTapEnable.assert_called_once_with(det._tap, True)
        assert result is event


class TestForwardingRecovery:
    """Test that _forwarding recovers via timeout if synthetic events are lost."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        return det, on_start, on_end

    def test_forwarding_timer_clears_stuck_flag(self, input_tap_module):
        """If the forwarded keyUp is never seen, the forwarding timer should
        auto-clear _forwarding so hold detection keeps working."""
        mod = input_tap_module

        det, _, _ = self._make_detector(input_tap_module)

        # Quick tap → forward space → _forwarding = True
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.handle_key_up(mod.SPACEBAR_KEYCODE)
        assert det._forwarding is True

        # Simulate: forwarded keyUp never arrives, but timer fires
        det.forwardingTimerFired_(None)
        assert det._forwarding is False

    def test_forwarding_timer_noop_if_already_cleared(self, input_tap_module):
        """Timer firing after forwarding was already cleared should be a no-op."""
        mod = input_tap_module

        det, _, _ = self._make_detector(input_tap_module)
        det._forwarding = False

        # Should not raise or change state
        det.forwardingTimerFired_(None)
        assert det._forwarding is False

    def test_shift_space_starts_recording(self, input_tap_module):
        """Shift+Space should start recording (shift is only detected at release)."""
        mod = input_tap_module

        det, _, _ = self._make_detector(input_tap_module)
        shift_flag = 0x00020000  # kCGEventFlagMaskShift

        result = det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        assert result is True  # suppressed — recording starts
        assert det._state == mod._State.WAITING


class TestForceEnd:
    """Test programmatic hold termination (e.g. recording cap)."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        return det, on_start, on_end

    def test_force_end_during_recording(self, input_tap_module):
        """force_end while RECORDING should transition to IDLE and call on_hold_end."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        det.force_end()
        assert det._state == mod._State.IDLE
        assert det._awaiting_space_release is True
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

    def test_force_end_requires_release_before_new_hold(self, input_tap_module):
        """force_end should swallow repeated keyDowns until the physical release arrives."""
        mod = input_tap_module
        det, on_start, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det.force_end()

        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._state == mod._State.IDLE
        on_start.assert_called_once()
        on_end.assert_called_once()

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._awaiting_space_release is False

        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._state == mod._State.WAITING

    def test_force_end_while_idle_is_noop(self, input_tap_module):
        """force_end while IDLE should do nothing."""
        det, _, on_end = self._make_detector(input_tap_module)
        assert det._state == input_tap_module._State.IDLE

        det.force_end()
        assert det._state == input_tap_module._State.IDLE
        on_end.assert_not_called()

    def test_force_end_while_waiting_is_noop(self, input_tap_module):
        """force_end while WAITING should do nothing."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        det.force_end()
        assert det._state == mod._State.WAITING
        on_end.assert_not_called()


class TestUninstall:
    """Test event tap teardown and cleanup."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = MagicMock()
        det._tap_source = MagicMock()
        return det, on_start, on_end

    def test_uninstall_resets_state_to_idle(self, input_tap_module):
        """uninstall should leave the detector in IDLE state."""
        mod = input_tap_module
        det, _, _ = self._make_detector(input_tap_module)

        # Put into RECORDING state
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        det.uninstall()
        assert det._state == mod._State.IDLE

    def test_uninstall_nullifies_tap(self, input_tap_module):
        """uninstall should clear the tap and source references."""
        det, _, _ = self._make_detector(input_tap_module)
        assert det._tap is not None

        det.uninstall()
        assert det._tap is None
        assert det._tap_source is None

    def test_uninstall_clears_forwarding(self, input_tap_module):
        """uninstall should clear the _forwarding flag."""
        det, _, _ = self._make_detector(input_tap_module)
        det._forwarding = True

        det.uninstall()
        assert det._forwarding is False

    def test_uninstall_clears_global_detector(self, input_tap_module):
        """uninstall should clear the module-level _active_detector."""
        mod = input_tap_module
        det, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        det.uninstall()
        assert mod._active_detector is None


class TestShiftLateLatching:
    """Test shift detection via kCGEventFlagsChanged (late-latch path)."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._shift_latched = False
        return det, on_start, on_end

    def test_flags_changed_latches_shift_during_recording(self, input_tap_module):
        """Shift pressed after keyDown (during RECORDING) should latch."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end = self._make_detector(input_tap_module)
        mod._active_detector = det

        # Enter RECORDING
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING
        assert det._shift_latched is False

        # Simulate shift press via flagsChanged
        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        assert det._shift_latched is True

    def test_flags_changed_latches_shift_during_waiting(self, input_tap_module):
        """Shift pressed during WAITING should also latch."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        assert det._shift_latched is True

    def test_flags_changed_ignores_shift_during_idle(self, input_tap_module):
        """Shift pressed while IDLE should NOT latch."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _ = self._make_detector(input_tap_module)
        det._shift_latched = False
        mod._active_detector = det

        assert det._state == mod._State.IDLE

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        assert det._shift_latched is False

    def test_shift_tap_during_idle_fires_idle_callback_after_defer(self, input_tap_module):
        """Standalone shift tap while idle should trigger the idle shift callback
        after the double-tap window expires (deferred to avoid firing on double-tap)."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _ = self._make_detector(input_tap_module)
        det._on_shift_tap_idle = MagicMock()
        mod._active_detector = det

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        # Not fired immediately — deferred until double-tap window expires
        det._on_shift_tap_idle.assert_not_called()
        assert det._shift_single_tap_timer is not None

        # Simulate timer firing
        det._shiftSingleTapFired_(None)
        det._on_shift_tap_idle.assert_called_once_with()

    def test_shift_modified_typing_during_idle_does_not_fire_idle_callback(self, input_tap_module):
        """Typing another key between shift down/up should not count as an idle shift tap."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _ = self._make_detector(input_tap_module)
        det._on_shift_tap_idle = MagicMock()
        mod._active_detector = det
        event = MagicMock()

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        Quartz.CGEventGetIntegerValueField.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetIntegerValueField.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        det._on_shift_tap_idle.assert_not_called()


class TestLatchedRecording:
    """Hands-free latched recording after shift tap during an active hold."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._shift_latched = False
        det._shift_at_press = False
        det._enter_held = False
        det.tray_active = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._shift_down_during_hold = False
        det._tray_gesture_consumed = False
        det._tray_last_shift_space_up = 0.0
        return det, on_start, on_end

    def test_shift_tap_during_recording_enters_latched_state(self, input_tap_module):
        """Shift tap during active recording should switch to latched recording."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        assert det._state == mod._State.LATCHED
        assert det._shift_latched is False

    def test_spacebar_release_after_latching_does_not_end_recording(self, input_tap_module):
        """Once latched, releasing spacebar should keep recording alive."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        assert det._state == mod._State.LATCHED
        on_end.assert_not_called()

    def test_enter_during_latched_recording_passes_through(self, input_tap_module):
        """In LATCHED, bare Enter belongs to the foreground app, not Spoke."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result is event
        on_end.assert_not_called()
        assert det._state == mod._State.LATCHED
        assert det._enter_held is True

    def test_initial_spacebar_release_in_latched_is_swallowed(
        self, input_tap_module
    ):
        """The first spacebar release after entering LATCHED should be swallowed
        (the user is going hands-free), not trigger text insertion."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        # Key-repeat from the original hold, then release
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_not_called()
        assert det._state == mod._State.LATCHED

    def test_spacebar_tap_in_latched_inserts_at_cursor(
        self, input_tap_module
    ):
        """A fresh spacebar tap (after the initial release) in LATCHED should
        insert text at cursor."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        # Simulate the initial release (go hands-free)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_not_called()
        assert det._state == mod._State.LATCHED

        # Now a fresh press+release = insert at cursor
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_called_once_with(shift_held=False, enter_held=False)
        assert det._state == mod._State.IDLE

    def test_shift_space_release_from_latched_recording_routes_to_tray(
        self, input_tap_module
    ):
        """Shift+spacebar from latched recording should end into the tray."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        # Simulate the initial release (go hands-free)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_not_called()

        # Now shift+space = tray route
        shift_flag = mod.kCGEventFlagMaskShift
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag) is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag) is True

        on_end.assert_called_once_with(shift_held=True, enter_held=False)
        assert det._state == mod._State.IDLE

    def test_spacebar_tap_during_latched_recording_inserts_text(self, input_tap_module):
        """Quick spacebar tap in LATCHED mode should end recording and insert at cursor."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        # Simulate the initial release (go hands-free)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_not_called()

        # Fresh tap: key-down then key-up without shift
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_called_once_with(shift_held=False, enter_held=False)
        assert det._state == mod._State.IDLE

    def test_enter_tap_during_latched_exit_routes_as_command(self, input_tap_module):
        """A brief Enter tap during a latched-exit chord should be captured and send assistant."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED
        mod._active_detector = det
        event = MagicMock()

        # Simulate the initial release (go hands-free).
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_not_called()
        assert det._state == mod._State.LATCHED

        # Begin the exit chord with Space.
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        assert det._latched_space_down is True

        # Brief Enter tap before releasing Space.
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        assert result_down is None
        assert result_up is None

        on_end.assert_called_once_with(shift_held=False, enter_held=True)
        assert det._awaiting_space_release is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        assert det._state == mod._State.IDLE

    def test_enter_keyup_after_latched_space_release_stays_suppressed(
        self, input_tap_module
    ):
        """If a latched exit chord keeps Enter down through Space release, trailing keyUp stays swallowed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()
        det._state = mod._State.LATCHED

        # Simulate the initial release (go hands-free).
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_not_called()

        # Begin the exit chord with Space, then hold Enter.
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, 0) is True
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None

        # Space release should route as assistant send (enter_held=True).
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

        # Trailing Enter release must still be swallowed.
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)
        assert result_up is None


class TestTrayAwareness:
    """Test input tap behavior when tray_active is set."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        on_shift_tap = MagicMock()
        on_enter_pressed = MagicMock()
        on_tray_delete = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._on_shift_tap = on_shift_tap
        det._on_enter_pressed = on_enter_pressed
        det._on_tray_delete = on_tray_delete
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._release_decision_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._latched_space_down = False
        det._shift_latched = False
        det._shift_at_press = False
        det._shift_down_during_hold = False
        det._enter_held = False
        det._enter_latched = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det.tray_active = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._tray_last_shift_space_up = 0.0
        return det, on_start, on_end, on_shift_tap, on_enter_pressed, on_tray_delete

    def test_tray_spacebar_tap_calls_hold_end_not_forward(self, input_tap_module):
        """During tray, quick spacebar tap should call on_hold_end, not forward space."""
        mod = input_tap_module
        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_called_once_with(shift_held=False, enter_held=False)
        assert det._forwarding is False

    def test_tray_space_enter_space_release_routes_overlay_toggle(
        self, input_tap_module
    ):
        """With tray visible, space-first Enter should still toggle assistant overlay."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result_down is None
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_tray_space_enter_enter_release_routes_assistant_path(
        self, input_tap_module
    ):
        """With tray visible, Enter-first release should still route as assistant/send."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None

        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)
        assert result_up is None
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_tray_shift_hold_then_shift_release_stays_tray_native(
        self, input_tap_module
    ):
        """Shift held before spacebar in tray should not fall into LATCHED on shift release."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, on_start, on_end, _, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det
        event = MagicMock()

        shift_flag = mod.kCGEventFlagMaskShift
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag) is True
        det.holdTimerFired_(None)
        on_start.assert_called_once()
        assert det._state == mod._State.RECORDING

        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetIntegerValueField.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        assert det._state != mod._State.LATCHED

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_called_once_with(shift_held=True, enter_held=False)
        assert det._state == mod._State.IDLE

    def test_non_tray_quick_release_still_forwards_space(self, input_tap_module):
        """Regression: when NOT in tray, quick release should forward space."""
        mod = input_tap_module
        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det.tray_active = False

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        on_end.assert_not_called()
        assert det._forwarding is True

    def test_enter_keydown_sets_enter_held(self, input_tap_module):
        """Enter keyDown should set _enter_held flag."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert det._enter_held is True

    def test_keypad_enter_keydown_sets_enter_held(self, input_tap_module):
        """Keypad Enter keyDown should also set _enter_held."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.KEYPAD_ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert det._enter_held is True

    def test_enter_keyup_clears_enter_held(self, input_tap_module):
        """Enter keyUp should clear _enter_held flag."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        assert det._enter_held is False

    def test_space_keydown_refreshes_stale_enter_state_from_keyboard_state(
        self, input_tap_module
    ):
        """A missed Enter keyUp must not poison the next plain space gesture."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True  # stale from an earlier consumed chord
        mod._active_detector = det

        Quartz.CGEventSourceKeyState.side_effect = (
            lambda _state, keycode: keycode == mod.SPACEBAR_KEYCODE
        )
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None
        assert det._enter_held is False

        det.holdTimerFired_(None)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

    def test_space_keydown_clears_stale_enter_when_keyboard_probe_unavailable(
        self, input_tap_module
    ):
        """If Quartz cannot answer and no Enter keyDown was seen, stale Enter must clear."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True
        det._enter_observed = False
        mod._active_detector = det

        Quartz.CGEventSourceKeyState.side_effect = RuntimeError("boom")
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None
        assert det._enter_held is False

        det.holdTimerFired_(None)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

    def test_space_keydown_preserves_observed_enter_when_keyboard_probe_unavailable(
        self, input_tap_module
    ):
        """If we actually saw Enter keyDown, an unavailable probe must not erase the chord."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True
        det._enter_observed = True
        det._enter_last_down_monotonic = 10.0
        mod._active_detector = det

        Quartz.CGEventSourceKeyState.side_effect = RuntimeError("boom")
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        with patch.object(mod.time, "monotonic", return_value=10.2):
            result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None
        assert det._enter_held is True

        det.holdTimerFired_(None)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_space_keydown_clears_stale_observed_enter_when_probe_unavailable(
        self, input_tap_module
    ):
        """A long-stale Enter observation must not poison fresh space taps forever."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True
        det._enter_observed = True
        det._enter_last_down_monotonic = 10.0
        mod._active_detector = det

        Quartz.CGEventSourceKeyState.side_effect = RuntimeError("boom")
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        with patch.object(mod.time, "monotonic", return_value=11.0):
            result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result_down is None
        assert det._enter_held is False
        assert det._enter_observed is False

        det.holdTimerFired_(None)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=False)

    def test_space_keydown_preserves_real_enter_first_chord(
        self, input_tap_module
    ):
        """A real Enter-first chord should still route as an assistant gesture."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det

        Quartz.CGEventSourceKeyState.return_value = True
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None
        assert det._enter_held is True

        det.holdTimerFired_(None)
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_stale_repeat_keydown_promotes_waiting_to_recording_immediately(
        self, input_tap_module
    ):
        """A delayed repeat should use event timestamps instead of waiting another hold window."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, on_start, _, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.side_effect = [
            1_000_000_000,
            1_500_000_000,
        ]

        first = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert first is None
        assert det._state == mod._State.WAITING
        on_start.assert_not_called()

        second = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert second is None
        assert det._state == mod._State.RECORDING
        on_start.assert_called_once()

    def test_hold_timer_does_not_trust_space_key_state_probe_for_active_hold(
        self, input_tap_module
    ):
        """A live hold must still start even if Quartz reports Space as up."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, on_start, _, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.return_value = 1_000_000_000

        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result is None
        assert det._state == mod._State.WAITING

        Quartz.CGEventSourceKeyState.return_value = False
        det.holdTimerFired_(None)

        assert det._state == mod._State.RECORDING
        assert det._awaiting_space_release is False
        on_start.assert_called_once()

    def test_enter_during_tray_does_not_fire_callback(self, input_tap_module):
        """Tray visibility alone should not arm bare Enter as a Spoke command."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, on_enter, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        on_enter.assert_not_called()
        assert result is event
        assert det._enter_held is True

    def test_enter_outside_tray_no_callback(self, input_tap_module):
        """Enter pressed outside tray should not fire on_enter_pressed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, on_enter, _ = self._make_detector(input_tap_module)
        det.tray_active = False
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        on_enter.assert_not_called()
        assert det._enter_held is True

    def test_enter_release_during_recording_routes_as_command(self, input_tap_module):
        """If Enter releases first during recording, the chord should route as assistant."""
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)
        assert result_up is None
        on_end.assert_called_once_with(shift_held=False, enter_held=True)
        assert det._awaiting_space_release is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

    def test_enter_tap_before_recording_release_routes_assistant(self, input_tap_module):
        """A brief Enter tap during recording should win if Enter releases before Space."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        assert result_down is None, "Enter keyDown must be suppressed during recording"
        assert result_up is None, "Enter keyUp must be suppressed during recording"

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True

        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_keypad_enter_tap_before_recording_release_routes_assistant(
        self, input_tap_module
    ):
        """Keypad Enter should route the recording chord to assistant too."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)

        Quartz.CGEventGetIntegerValueField.return_value = mod.KEYPAD_ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        assert result_down is None
        assert result_up is None

        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0) is True
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_enter_during_waiting_toggles_overlay_not_assistant(self, input_tap_module):
        """Enter during WAITING (before hold threshold) should toggle overlay
        and return to IDLE, not route as enter_held to the assistant."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, on_enter_toggle, _, _ = self._make_detector(input_tap_module)
        det._on_enter_during_waiting = on_enter_toggle
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result_down is None, "Enter keyDown must be suppressed during waiting"
        assert det._state == mod._State.IDLE, "Should return to IDLE after toggle"
        on_enter_toggle.assert_called_once()
        on_end.assert_not_called()

    def test_enter_keyup_after_space_release_stays_suppressed(self, input_tap_module):
        """If Space releases first, the trailing Enter keyUp must still be swallowed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0

        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result_down is None

        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)
        on_end.assert_called_once_with(shift_held=False, enter_held=True)

        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)
        assert result_up is None

    def test_enter_suppressed_during_waiting(self, input_tap_module):
        """Enter pressed during WAITING (before timer fires) must be suppressed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        assert result_down is None, "Enter keyDown must be suppressed during waiting"
        assert result_up is None, "Enter keyUp must be suppressed during waiting"
        assert det._enter_held is False

    def test_shift_release_then_enter_within_grace_routes_as_command(
        self, input_tap_module
    ):
        """A slightly late Enter press after release should still win over shift."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=mod.kCGEventFlagMaskShift)

        on_end.assert_not_called()
        assert det._pending_release_active is True

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result is None
        on_end.assert_called_once_with(shift_held=True, enter_held=True)
        assert det._pending_release_active is False

    def test_recording_release_falls_back_when_enter_does_not_arrive(
        self, input_tap_module
    ):
        """Without a late Enter press, a shift-held release should settle back to shift-only."""
        mod = input_tap_module

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=mod.kCGEventFlagMaskShift)

        on_end.assert_not_called()
        assert det._pending_release_active is True

        det.releaseDecisionTimerFired_(None)

        on_end.assert_called_once_with(shift_held=True, enter_held=False)
        assert det._pending_release_active is False

    def test_shift_tap_during_tray_fires_callback(self, input_tap_module):
        """Shift down then up (no spacebar) during tray should fire on_shift_tap."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, on_shift_tap, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        Quartz.CGEventGetIntegerValueField.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        on_shift_tap.assert_called_once()

    def test_shift_space_not_shift_tap(self, input_tap_module):
        """Shift down, spacebar, shift up should NOT fire on_shift_tap."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, on_shift_tap, _, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det

        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        Quartz.CGEventGetIntegerValueField.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = mod.kCGEventFlagMaskShift
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetIntegerValueField.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

        on_shift_tap.assert_not_called()

    def test_double_tap_spacebar_with_shift_fires_delete(self, input_tap_module):
        """Shift held + double-tap spacebar during tray should fire on_tray_delete."""
        mod = input_tap_module
        det, _, on_end, _, _, on_delete = self._make_detector(input_tap_module)
        det.tray_active = True

        shift_flag = mod.kCGEventFlagMaskShift

        # First tap: shift+spacebar down then up (navigates up)
        det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag)
        on_end.assert_called_once()  # first tap = navigate up
        on_delete.assert_not_called()

        # Second tap: shift+spacebar down then up within 300ms (deletes)
        det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag)
        on_delete.assert_called_once()
        # on_hold_end should NOT be called a second time (delete takes priority)
        assert on_end.call_count == 1

    def test_slow_double_tap_does_not_delete(self, input_tap_module):
        """Two shift+spacebar taps > 300ms apart should both navigate, not delete."""
        import time as _time
        mod = input_tap_module
        det, _, on_end, _, _, on_delete = self._make_detector(input_tap_module)
        det.tray_active = True

        shift_flag = mod.kCGEventFlagMaskShift

        # First tap
        det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag)

        # Force the timestamp to be old
        det._tray_last_shift_space_up = _time.monotonic() - 0.5

        # Second tap (after window expired)
        det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag)

        on_delete.assert_not_called()
        assert on_end.call_count == 2  # both were navigate


class TestCommandOverlayFlags:
    """Test input tap behavior for command_overlay_active and related flags."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        on_dismiss = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._on_command_overlay_dismiss = on_dismiss
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._release_decision_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._latched_space_down = False
        det._latched_space_released = False
        det._shift_latched = False
        det._shift_at_press = False
        det._shift_down_during_hold = False
        det._enter_held = False
        det._enter_latched = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det.tray_active = False
        det.command_overlay_active = False
        det._command_overlay_just_dismissed = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._tray_last_shift_space_up = 0.0
        det._tray_gesture_consumed = False
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._on_shift_tap = None
        det._on_shift_tap_during_hold = None
        det._on_shift_tap_idle = None
        det._on_enter_pressed = None
        det._on_tray_delete = None
        return det, on_start, on_end, on_dismiss

    def test_enter_passes_through_when_overlay_not_active(self, input_tap_module):
        """Enter keyDown should pass through when command_overlay_active=False."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _ = self._make_detector(input_tap_module)
        det.command_overlay_active = False
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result is event  # passed through
        assert det._enter_held is True

    def test_spacebar_does_not_dismiss_overlay_on_keydown(self, input_tap_module):
        """Spacebar keyDown while command_overlay_active=True must NOT instant-
        dismiss.  Overlay dismiss is handled by the hold-and-release path
        (_on_hold_end with no audio), not by bare spacebar press."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, on_dismiss = self._make_detector(input_tap_module)
        det.command_overlay_active = True
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        on_dismiss.assert_not_called()
        assert det.command_overlay_active is True

    def test_enter_during_waiting_toggles_overlay(self, input_tap_module):
        """Enter during WAITING should toggle overlay and return to IDLE,
        not route as enter_held to the assistant."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, on_dismiss = self._make_detector(input_tap_module)
        on_toggle = MagicMock()
        det._on_enter_during_waiting = on_toggle
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        assert result is None
        assert det._state == mod._State.IDLE
        on_toggle.assert_called_once()
        on_end.assert_not_called()


class TestDoubleTapGestures:
    """Test double-tap Enter and double-tap Shift gestures."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        mod = input_tap_module
        on_start = MagicMock()
        on_end = MagicMock()
        on_double_enter = MagicMock()
        on_double_shift = MagicMock()
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._on_hold_start = on_start
        det._on_hold_end = on_end
        det._hold_s = hold_ms / 1000.0
        det._state = mod._State.IDLE
        det._hold_timer = None
        det._safety_timer = None
        det._forwarding = False
        det._forwarding_timer = None
        det._release_decision_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._latched_space_down = False
        det._latched_space_released = False
        det._shift_latched = False
        det._shift_at_press = False
        det._shift_down_during_hold = False
        det._enter_held = False
        det._enter_latched = False
        det._suppress_enter_keyup = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det.tray_active = False
        det.command_overlay_active = False
        det._command_overlay_just_dismissed = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._tray_last_shift_space_up = 0.0
        det._tray_gesture_consumed = False
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._on_shift_tap = None
        det._on_shift_tap_during_hold = None
        det._on_shift_tap_idle = None
        det._on_enter_pressed = None
        det._on_tray_delete = None
        det._on_command_overlay_dismiss = None
        det.cancel_spring_active = False
        det._on_cancel_spring_start = None
        det._on_cancel_spring_release = None
        det._on_enter_during_waiting = on_double_enter
        det._on_double_tap_shift = on_double_shift
        det._last_idle_shift_up = 0.0
        det._shift_single_tap_timer = None
        return det, on_start, on_end, on_double_enter, on_double_shift

    def _enter_tap(self, mod, event):
        """Simulate a bare Enter tap (keyDown + keyUp) via the event tap."""
        Quartz = __import__("Quartz")
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

    def _shift_tap(self, mod, event):
        """Simulate a bare Shift tap (flagsChanged down + flagsChanged up)."""
        Quartz = __import__("Quartz")
        shift_flag = Quartz.kCGEventFlagMaskShift
        # Shift down
        Quartz.CGEventGetIntegerValueField.return_value = 56  # shift keycode
        Quartz.CGEventGetFlags.return_value = shift_flag
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)
        # Shift up
        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventFlagsChanged, event, None)

    def _enter_down(self, mod, event):
        """Simulate Enter keyDown only."""
        Quartz = __import__("Quartz")
        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

    def _space_down(self, mod, event):
        """Simulate spacebar keyDown (enters WAITING)."""
        Quartz = __import__("Quartz")
        Quartz.CGEventGetIntegerValueField.return_value = mod.SPACEBAR_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

    def test_enter_during_waiting_fires_toggle(self, input_tap_module, monkeypatch):
        """Enter key-down during WAITING (before hold threshold) should fire toggle."""
        import time as _time
        mod = input_tap_module
        det, on_start, _, on_enter_toggle, _ = self._make_detector(mod)
        mod._active_detector = det
        event = MagicMock()

        # Hold spacebar to enter WAITING
        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._space_down(mod, event)
        assert det._state == mod._State.WAITING

        # Enter during WAITING — should toggle and return to IDLE
        self._enter_down(mod, event)
        on_enter_toggle.assert_called_once()
        assert det._state == mod._State.IDLE
        on_start.assert_not_called()  # recording never started

    def test_enter_idle_passes_through(self, input_tap_module, monkeypatch):
        """Enter in IDLE (no spacebar held) should pass through — no toggle."""
        mod = input_tap_module
        det, _, _, on_enter_toggle, _ = self._make_detector(mod)
        mod._active_detector = det
        event = MagicMock()

        self._enter_tap(mod, event)
        on_enter_toggle.assert_not_called()

    def test_double_tap_shift_fires_callback(self, input_tap_module, monkeypatch):
        """Two Shift taps within 300ms should fire _on_double_tap_shift."""
        import time as _time
        mod = input_tap_module
        det, _, _, _, on_double_shift = self._make_detector(mod)
        mod._active_detector = det
        event = MagicMock()

        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)
        on_double_shift.assert_not_called()

        now = 1000.2
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)
        on_double_shift.assert_called_once()

    def test_slow_double_tap_shift_does_not_fire(self, input_tap_module, monkeypatch):
        """Two Shift taps more than 300ms apart should NOT fire the callback."""
        import time as _time
        mod = input_tap_module
        det, _, _, _, on_double_shift = self._make_detector(mod)
        mod._active_detector = det
        event = MagicMock()

        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)

        now = 1000.5
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)
        on_double_shift.assert_not_called()

    def test_double_tap_shift_does_not_fire_during_tray(self, input_tap_module, monkeypatch):
        """Double-tap Shift while tray is active should NOT toggle HUD —
        Shift belongs to tray navigation."""
        import time as _time
        mod = input_tap_module
        det, _, _, _, on_double_shift = self._make_detector(mod)
        det.tray_active = True
        det._on_shift_tap = MagicMock()
        mod._active_detector = det
        event = MagicMock()

        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)

        now = 1000.2
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)
        on_double_shift.assert_not_called()

    def test_enter_passes_through_on_single_tap(self, input_tap_module, monkeypatch):
        """A single Enter tap should pass through to the OS, not be suppressed."""
        import time as _time
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det, _, _, on_double_enter, _ = self._make_detector(mod)
        mod._active_detector = det
        event = MagicMock()

        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        result_down = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        result_up = mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        # Enter should pass through (not suppressed)
        assert result_down is event
        assert result_up is event
        on_double_enter.assert_not_called()

    def test_double_tap_shift_does_not_fire_single_tap(self, input_tap_module, monkeypatch):
        """Double-tap Shift should NOT fire _on_shift_tap_idle (TTS toggle).
        The single-tap action must be deferred and cancelled when the second
        tap arrives within the double-tap window."""
        import time as _time
        mod = input_tap_module
        det, _, _, _, on_double_shift = self._make_detector(mod)
        on_single_shift = MagicMock()
        det._on_shift_tap_idle = on_single_shift
        mod._active_detector = det
        event = MagicMock()

        # First shift tap
        now = 1000.0
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)

        # Single-tap should NOT have fired yet (deferred)
        on_single_shift.assert_not_called()

        # Second shift tap within window
        now = 1000.2
        monkeypatch.setattr(_time, "monotonic", lambda: now)
        self._shift_tap(mod, event)

        # Double-tap should fire, single-tap should never fire
        on_double_shift.assert_called_once()
        on_single_shift.assert_not_called()
