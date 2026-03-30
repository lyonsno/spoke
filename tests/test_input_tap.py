"""Tests for the CGEventTap spacebar hold detection state machine."""

from unittest.mock import MagicMock, call


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

    def test_shift_tap_during_idle_fires_idle_callback(self, input_tap_module):
        """Standalone shift tap while idle should trigger the idle shift callback."""
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

    def test_enter_during_latched_recording_ends_with_command_route(self, input_tap_module):
        """Enter during latched recording should stop capture and route to assistant."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()

        result = mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        assert result is None
        on_end.assert_called_once_with(shift_held=False, enter_held=True)
        assert det._state == mod._State.IDLE

    def test_shift_space_release_from_latched_recording_routes_to_tray(
        self, input_tap_module
    ):
        """Shift+spacebar from latched recording should end into the tray."""
        mod = input_tap_module
        det, _, on_end = self._make_detector(input_tap_module)
        det._state = mod._State.LATCHED

        shift_flag = mod.kCGEventFlagMaskShift
        assert det.handle_key_down(mod.SPACEBAR_KEYCODE, shift_flag) is True
        assert det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=shift_flag) is True

        on_end.assert_called_once_with(shift_held=True, enter_held=False)
        assert det._state == mod._State.IDLE


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
        det._shift_latched = False
        det._shift_at_press = False
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
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        on_end.assert_called_once_with(shift_held=False, enter_held=False)
        assert det._forwarding is False

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

    def test_enter_during_tray_fires_callback(self, input_tap_module):
        """Enter pressed while tray is active should fire on_enter_pressed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, _, _, on_enter, _ = self._make_detector(input_tap_module)
        det.tray_active = True
        mod._active_detector = det

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        event = MagicMock()
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)

        on_enter.assert_called_once()

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

    def test_enter_held_passed_on_recording_release(self, input_tap_module):
        """Enter held during recording should pass enter_held=True to on_hold_end."""
        mod = input_tap_module
        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        det._enter_held = True

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        on_end.assert_called_once_with(shift_held=False, enter_held=True)

    def test_enter_tap_before_recording_release_stays_latched(self, input_tap_module):
        """A brief Enter tap during recording should still route the release as command."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det, _, on_end, _, _, _ = self._make_detector(input_tap_module)
        mod._active_detector = det
        event = MagicMock()

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)

        Quartz.CGEventGetIntegerValueField.return_value = mod.ENTER_KEYCODE
        Quartz.CGEventGetFlags.return_value = 0
        mod._event_tap_callback(None, Quartz.kCGEventKeyDown, event, None)
        mod._event_tap_callback(None, Quartz.kCGEventKeyUp, event, None)

        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        on_end.assert_called_once_with(shift_held=False, enter_held=True)

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
