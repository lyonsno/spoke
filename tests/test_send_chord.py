"""Tests for send chord (Enter + ]), enter deloading, and cancel gesture.

Lane S of ButterfingerFinalFuckers packet: send chord + enter deloading.
"""

import time
from unittest.mock import MagicMock, patch


class TestSendChordDetection:
    """Enter + ] from IDLE (with tray active) = send chord."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        """Create a detector with standard test wiring."""
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
        det._repeat_watchdog_timer = None
        det._last_space_keydown_monotonic = 0.0
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._enter_held = False
        det._enter_observed = False
        det._enter_latched = False
        det._enter_last_down_monotonic = 0.0
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det._shift_latched = False
        det._shift_at_press = False
        det._latched_space_down = False
        det._latched_space_released = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det._space_keydown_timestamp_ns = None
        det.tray_active = False
        det.command_overlay_active = False
        det.approval_active = False
        det._tray_gesture_consumed = False
        det._shift_down_during_hold = False
        det._tray_last_shift_space_up = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._route_key_selector = None
        det._on_send_chord = None
        det._on_cancel_generation = None
        return det, on_start, on_end

    def test_send_chord_enter_plus_bracket_from_tray(self, input_tap_module):
        """Enter held + ] keyDown while tray active = send chord fires."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = True

        # Enter is held (tracked by event tap callback)
        det._enter_held = True
        det._enter_observed = True

        # ] keyDown while IDLE + tray active + Enter held = send chord
        from spoke.route_keys import BRACKET_RIGHT_KEYCODE
        result = det.handle_send_chord(BRACKET_RIGHT_KEYCODE)
        assert result is True, "Send chord should be detected and suppress the key"
        on_send.assert_called_once()

    def test_send_chord_does_not_fire_without_enter(self, input_tap_module):
        """Without Enter held, ] keyDown should not fire send chord."""
        det, _, _ = self._make_detector(input_tap_module)

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = True
        det._enter_held = False

        from spoke.route_keys import BRACKET_RIGHT_KEYCODE
        result = det.handle_send_chord(BRACKET_RIGHT_KEYCODE)
        assert result is False
        on_send.assert_not_called()

    def test_send_chord_does_not_fire_without_tray(self, input_tap_module):
        """Send chord only fires when tray is active."""
        det, _, _ = self._make_detector(input_tap_module)

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = False
        det._enter_held = True

        from spoke.route_keys import BRACKET_RIGHT_KEYCODE
        result = det.handle_send_chord(BRACKET_RIGHT_KEYCODE)
        assert result is False
        on_send.assert_not_called()

    def test_send_chord_with_non_bracket_route_key(self, input_tap_module):
        """Enter + number-row route key = send to that destination."""
        det, _, _ = self._make_detector(input_tap_module)

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = True
        det._enter_held = True

        # Use number key 6 (keycode 22) as a route key
        result = det.handle_send_chord(22)
        assert result is True
        on_send.assert_called_once()
        # The callback should receive the keycode for destination routing
        call_args = on_send.call_args
        assert call_args is not None
        assert call_args[1].get("keycode") == 22 or (
            len(call_args[0]) > 0 and call_args[0][0] == 22
        )

    def test_send_chord_only_in_idle_state(self, input_tap_module):
        """Send chord should only fire from IDLE state, not during recording."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = True
        det._enter_held = True
        det._state = mod._State.RECORDING

        from spoke.route_keys import BRACKET_RIGHT_KEYCODE
        result = det.handle_send_chord(BRACKET_RIGHT_KEYCODE)
        assert result is False
        on_send.assert_not_called()

    def test_send_chord_suppresses_enter_keyup(self, input_tap_module):
        """After send chord fires, the trailing Enter keyUp should be suppressed."""
        det, _, _ = self._make_detector(input_tap_module)

        on_send = MagicMock()
        det._on_send_chord = on_send
        det.tray_active = True
        det._enter_held = True

        from spoke.route_keys import BRACKET_RIGHT_KEYCODE
        det.handle_send_chord(BRACKET_RIGHT_KEYCODE)

        assert det._suppress_enter_keyup is True, (
            "After send chord, trailing Enter keyUp must be suppressed"
        )


class TestEnterDeloading:
    """Enter-held-at-release no longer routes to assistant during recording.

    The old enter_held pathway during space-rooted recording is removed. Enter's
    identity during recording is reduced to 'confirm/submit' — it does not
    select the assistant destination. Route keys handle that now.
    """

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
        det._repeat_watchdog_timer = None
        det._last_space_keydown_monotonic = 0.0
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._enter_held = False
        det._enter_observed = False
        det._enter_latched = False
        det._enter_last_down_monotonic = 0.0
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det._shift_latched = False
        det._shift_at_press = False
        det._latched_space_down = False
        det._latched_space_released = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det._space_keydown_timestamp_ns = None
        det.tray_active = False
        det.command_overlay_active = False
        det.approval_active = False
        det._tray_gesture_consumed = False
        det._shift_down_during_hold = False
        det._tray_last_shift_space_up = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._route_key_selector = None
        det._on_send_chord = None
        det._on_cancel_generation = None
        return det, on_start, on_end

    def test_recording_release_with_enter_held_does_not_route_to_assistant(
        self, input_tap_module
    ):
        """Space release after recording with enter held should NOT pass
        enter_held=True to on_hold_end. Enter is deloaded from recording
        routing; route keys handle destination selection now."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # Start recording
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        # Enter is held during recording
        det._enter_held = True

        # Release spacebar
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)
        assert det._state == mod._State.IDLE

        # on_hold_end should have been called with enter_held=False
        # (enter is deloaded from recording routing)
        on_end.assert_called()
        _, kwargs = on_end.call_args
        assert kwargs.get("enter_held", None) is False, (
            "enter_held must be False during recording release — enter is deloaded"
        )

    def test_waiting_release_with_enter_held_does_not_route_to_assistant(
        self, input_tap_module
    ):
        """Quick tap while enter held should NOT route to assistant either.
        Enter-held routing is removed from all recording states."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # Enter WAITING
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        # Enter is held
        det._enter_held = True

        # Release spacebar quickly
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        # Should not have called on_hold_end with enter_held=True
        if on_end.called:
            _, kwargs = on_end.call_args
            assert kwargs.get("enter_held", None) is not True, (
                "enter_held routing removed from WAITING release"
            )

    def test_finish_enter_release_removed_from_recording(self, input_tap_module):
        """_finish_enter_release should no longer end recording via enter_held
        routing. The method should either not exist or not pass enter_held=True."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # If _finish_enter_release still exists, it should not route via
        # enter_held anymore. If it's been removed entirely, that's fine.
        if hasattr(det, '_finish_enter_release'):
            det._state = mod._State.RECORDING
            det._finish_enter_release(shift_held=False)
            if on_end.called:
                _, kwargs = on_end.call_args
                assert kwargs.get("enter_held", None) is not True, (
                    "_finish_enter_release must not pass enter_held=True"
                )

    def test_latched_exit_with_enter_does_not_use_enter_held(
        self, input_tap_module
    ):
        """Latched recording exits should not use enter_held for routing.
        Enter during latched state is just a confirm, not a destination selector."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        # Get into LATCHED state
        det._state = mod._State.LATCHED
        det._latched_space_down = True
        det._latched_space_released = True
        det._enter_held = True
        det._enter_latched = True

        # Space down + release from latched
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.handle_key_up(mod.SPACEBAR_KEYCODE, flags=0)

        if on_end.called:
            _, kwargs = on_end.call_args
            assert kwargs.get("enter_held", None) is not True, (
                "Latched exit with enter must not use enter_held routing"
            )


class TestCancelGesture:
    """Space + Delete while command overlay visible = cancel generation."""

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
        det._repeat_watchdog_timer = None
        det._last_space_keydown_monotonic = 0.0
        det._forwarding = False
        det._forwarding_timer = None
        det._tap = None
        det._tap_source = None
        det._awaiting_space_release = False
        det._enter_held = False
        det._enter_observed = False
        det._enter_latched = False
        det._enter_last_down_monotonic = 0.0
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det._shift_latched = False
        det._shift_at_press = False
        det._latched_space_down = False
        det._latched_space_released = False
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det._space_keydown_timestamp_ns = None
        det.tray_active = False
        det.command_overlay_active = False
        det.approval_active = False
        det._tray_gesture_consumed = False
        det._shift_down_during_hold = False
        det._tray_last_shift_space_up = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._route_key_selector = None
        det._on_send_chord = None
        det._on_cancel_generation = None
        return det, on_start, on_end

    def test_space_delete_cancels_when_overlay_visible(self, input_tap_module):
        """Space held (WAITING/RECORDING) + Delete = cancel generation."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_cancel = MagicMock()
        det._on_cancel_generation = on_cancel
        det.command_overlay_active = True

        # Space down -> WAITING
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        # Delete pressed while space held + overlay visible
        result = det.handle_cancel_gesture(mod.DELETE_KEYCODE)
        assert result is True, "Cancel gesture should suppress the delete key"
        on_cancel.assert_called_once()

    def test_space_delete_does_not_cancel_without_overlay(self, input_tap_module):
        """Space + Delete without visible command overlay does nothing."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_cancel = MagicMock()
        det._on_cancel_generation = on_cancel
        det.command_overlay_active = False

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        result = det.handle_cancel_gesture(mod.DELETE_KEYCODE)
        assert result is False
        on_cancel.assert_not_called()

    def test_cancel_gesture_returns_to_idle(self, input_tap_module):
        """After cancel gesture, detector returns to IDLE."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_cancel = MagicMock()
        det._on_cancel_generation = on_cancel
        det.command_overlay_active = True

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.handle_cancel_gesture(mod.DELETE_KEYCODE)

        assert det._state == mod._State.IDLE, (
            "After cancel gesture, state must return to IDLE"
        )

    def test_cancel_gesture_only_during_space_hold(self, input_tap_module):
        """Cancel gesture requires space to be held (WAITING or RECORDING)."""
        det, _, _ = self._make_detector(input_tap_module)
        mod = input_tap_module

        on_cancel = MagicMock()
        det._on_cancel_generation = on_cancel
        det.command_overlay_active = True

        # IDLE state — space not held
        assert det._state == mod._State.IDLE
        result = det.handle_cancel_gesture(mod.DELETE_KEYCODE)
        assert result is False
        on_cancel.assert_not_called()


class TestFlickAnimation:
    """Overlay flick animation on send chord."""

    def test_overlay_has_flick_send_method(self, input_tap_module):
        """The overlay module should expose a flick_send method."""
        import importlib
        import sys
        sys.modules.pop("spoke.overlay", None)
        overlay_mod = importlib.import_module("spoke.overlay")
        # The TranscriptionOverlay class should have a flick_send method
        assert hasattr(overlay_mod.TranscriptionOverlay, "flick_send"), (
            "TranscriptionOverlay must have a flick_send() method for send chord animation"
        )
