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
        on_end.assert_called_once_with(shift_held=False)

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
