"""Tests for route key destination selection during recording.

Tests cover:
- Route key binding data structure
- Tap-to-toggle selection state machine
- Single-active constraint (only one route key at a time)
- Reset on recording end
- CGEventTap interception of route keys during RECORDING/LATCHED states
- Route keys ignored during IDLE/WAITING states
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def route_keys_module(mock_pyobjc):
    """Import spoke.route_keys with mocked PyObjC."""
    sys.modules.pop("spoke.route_keys", None)
    mod = importlib.import_module("spoke.route_keys")
    yield mod
    sys.modules.pop("spoke.route_keys", None)


class TestRouteKeyBindings:
    """Test the route key binding data structure."""

    def test_default_bindings_include_assistant_bracket(self, route_keys_module):
        """The `]` key should be bound to 'assistant' by default."""
        bindings = route_keys_module.default_bindings()
        # keycode 30 is `]` on US keyboard
        assert route_keys_module.BRACKET_RIGHT_KEYCODE in bindings
        entry = bindings[route_keys_module.BRACKET_RIGHT_KEYCODE]
        assert entry["destination"] == "assistant"

    def test_default_bindings_include_number_row(self, route_keys_module):
        """Number row keys 6-0 and `-`, `=` should have bindings."""
        bindings = route_keys_module.default_bindings()
        for keycode in route_keys_module.NUMBER_ROW_KEYCODES:
            assert keycode in bindings, f"keycode {keycode} should be in default bindings"

    def test_binding_has_required_fields(self, route_keys_module):
        """Each binding entry must have destination, label, and flavor."""
        bindings = route_keys_module.default_bindings()
        for keycode, entry in bindings.items():
            assert "destination" in entry, f"keycode {keycode} missing 'destination'"
            assert "label" in entry, f"keycode {keycode} missing 'label'"
            assert "flavor" in entry, f"keycode {keycode} missing 'flavor'"
            assert entry["flavor"] in ("persistent", "contingent", "one-shot"), (
                f"keycode {keycode} has invalid flavor: {entry['flavor']}"
            )

    def test_all_route_keycodes_constant(self, route_keys_module):
        """ALL_ROUTE_KEYCODES should include bracket and number row."""
        all_codes = route_keys_module.ALL_ROUTE_KEYCODES
        assert route_keys_module.BRACKET_RIGHT_KEYCODE in all_codes
        for kc in route_keys_module.NUMBER_ROW_KEYCODES:
            assert kc in all_codes


class TestRouteKeySelector:
    """Test the route key selection state machine."""

    def test_initial_state_no_selection(self, route_keys_module):
        """Fresh selector has no active route key."""
        sel = route_keys_module.RouteKeySelector()
        assert sel.active_keycode is None
        assert sel.active_destination is None

    def test_tap_selects_route_key(self, route_keys_module):
        """Tapping a route key selects it."""
        sel = route_keys_module.RouteKeySelector()
        sel.tap(route_keys_module.BRACKET_RIGHT_KEYCODE)
        assert sel.active_keycode == route_keys_module.BRACKET_RIGHT_KEYCODE
        assert sel.active_destination == "assistant"

    def test_tap_same_key_deselects(self, route_keys_module):
        """Tapping the same route key again deselects it."""
        sel = route_keys_module.RouteKeySelector()
        sel.tap(route_keys_module.BRACKET_RIGHT_KEYCODE)
        assert sel.active_keycode == route_keys_module.BRACKET_RIGHT_KEYCODE
        sel.tap(route_keys_module.BRACKET_RIGHT_KEYCODE)
        assert sel.active_keycode is None
        assert sel.active_destination is None

    def test_tap_different_key_switches(self, route_keys_module):
        """Tapping a different route key deactivates the previous one."""
        sel = route_keys_module.RouteKeySelector()
        bracket = route_keys_module.BRACKET_RIGHT_KEYCODE
        number_key = route_keys_module.NUMBER_ROW_KEYCODES[0]

        sel.tap(bracket)
        assert sel.active_keycode == bracket

        sel.tap(number_key)
        assert sel.active_keycode == number_key
        assert sel.active_keycode != bracket

    def test_single_active_constraint(self, route_keys_module):
        """Only one route key can be active at a time."""
        sel = route_keys_module.RouteKeySelector()
        keys = [route_keys_module.BRACKET_RIGHT_KEYCODE] + list(
            route_keys_module.NUMBER_ROW_KEYCODES[:3]
        )
        for k in keys:
            sel.tap(k)
        # Only the last tapped should be active
        assert sel.active_keycode == keys[-1]

    def test_reset_clears_selection(self, route_keys_module):
        """reset() clears any active selection."""
        sel = route_keys_module.RouteKeySelector()
        sel.tap(route_keys_module.BRACKET_RIGHT_KEYCODE)
        assert sel.active_keycode is not None
        sel.reset()
        assert sel.active_keycode is None
        assert sel.active_destination is None

    def test_unknown_keycode_ignored(self, route_keys_module):
        """Tapping a keycode not in bindings does nothing."""
        sel = route_keys_module.RouteKeySelector()
        sel.tap(999)
        assert sel.active_keycode is None

    def test_custom_bindings(self, route_keys_module):
        """Selector accepts custom bindings override."""
        custom = {
            42: {"destination": "custom_dest", "label": "X", "flavor": "one-shot"},
        }
        sel = route_keys_module.RouteKeySelector(bindings=custom)
        sel.tap(42)
        assert sel.active_keycode == 42
        assert sel.active_destination == "custom_dest"

    def test_on_change_callback(self, route_keys_module):
        """on_change callback fires on selection and deselection."""
        changes = []
        sel = route_keys_module.RouteKeySelector(
            on_change=lambda keycode, dest: changes.append((keycode, dest))
        )
        bracket = route_keys_module.BRACKET_RIGHT_KEYCODE
        sel.tap(bracket)
        assert changes == [(bracket, "assistant")]
        sel.tap(bracket)  # deselect
        assert changes == [(bracket, "assistant"), (None, None)]


class TestRouteKeyInterception:
    """Test that route keys are intercepted during RECORDING/LATCHED and ignored otherwise."""

    def _make_detector(self, input_tap_module, hold_ms=400):
        """Create a detector with route key support wired."""
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
        det._shift_at_press = False
        det._shift_latched = False
        det._enter_held = False
        det._enter_observed = False
        det._enter_latched = False
        det._enter_last_down_monotonic = 0.0
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det._release_decision_timer = None
        det._pending_release_active = False
        det._pending_release_shift_held = False
        det._space_keydown_timestamp_ns = None
        det.tray_active = False
        det.command_overlay_active = False
        det.approval_active = False
        det._on_shift_tap = None
        det._on_shift_tap_during_hold = None
        det._on_shift_tap_idle = None
        det._on_enter_pressed = None
        det._on_tray_delete = None
        det._on_approval_enter_pressed = None
        det._on_approval_delete_pressed = None
        det._tray_shift_down = False
        det._tray_space_between = False
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._shift_down_during_hold = False
        det._tray_gesture_consumed = False
        det._tray_last_shift_space_up = 0.0
        det._on_double_tap_shift = None
        det._last_idle_shift_up = 0.0
        det._shift_single_tap_timer = None
        det._on_enter_during_waiting = None
        det._on_cancel_spring_start = None
        det._on_cancel_spring_release = None
        det.cancel_spring_active = False
        det._latched_space_down = False
        det._latched_space_released = False

        # Route key support
        det._route_key_selector = None
        return det, on_start, on_end

    def test_route_key_suppressed_during_recording(self, input_tap_module, route_keys_module):
        """Route key taps should be suppressed (not passed through) during RECORDING."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        selector = route_keys_module.RouteKeySelector()
        det._route_key_selector = selector

        # Get into RECORDING state
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        assert det._state == mod._State.RECORDING

        # Tap `]` — should be suppressed
        bracket_kc = route_keys_module.BRACKET_RIGHT_KEYCODE
        suppressed = det.handle_route_key_down(bracket_kc)
        assert suppressed is True
        assert selector.active_keycode == bracket_kc

    def test_route_key_suppressed_during_latched(self, input_tap_module, route_keys_module):
        """Route key taps should be suppressed during LATCHED state."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        selector = route_keys_module.RouteKeySelector()
        det._route_key_selector = selector

        # Get into LATCHED state
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det._state = mod._State.LATCHED

        bracket_kc = route_keys_module.BRACKET_RIGHT_KEYCODE
        suppressed = det.handle_route_key_down(bracket_kc)
        assert suppressed is True
        assert selector.active_keycode == bracket_kc

    def test_route_key_not_suppressed_during_idle(self, input_tap_module, route_keys_module):
        """Route key taps should pass through during IDLE."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        selector = route_keys_module.RouteKeySelector()
        det._route_key_selector = selector

        assert det._state == mod._State.IDLE
        bracket_kc = route_keys_module.BRACKET_RIGHT_KEYCODE
        suppressed = det.handle_route_key_down(bracket_kc)
        assert suppressed is False
        assert selector.active_keycode is None  # no selection when not recording

    def test_route_key_not_suppressed_during_waiting(self, input_tap_module, route_keys_module):
        """Route key taps should pass through during WAITING."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        selector = route_keys_module.RouteKeySelector()
        det._route_key_selector = selector

        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        assert det._state == mod._State.WAITING

        bracket_kc = route_keys_module.BRACKET_RIGHT_KEYCODE
        suppressed = det.handle_route_key_down(bracket_kc)
        assert suppressed is False

    def test_route_key_reset_on_hold_end(self, input_tap_module, route_keys_module):
        """Route key selection resets when recording ends (spacebar released)."""
        det, on_start, on_end = self._make_detector(input_tap_module)
        mod = input_tap_module

        selector = route_keys_module.RouteKeySelector()
        det._route_key_selector = selector

        # Record, select route key, then release
        det.handle_key_down(mod.SPACEBAR_KEYCODE, 0)
        det.holdTimerFired_(None)
        det.handle_route_key_down(route_keys_module.BRACKET_RIGHT_KEYCODE)
        assert selector.active_keycode is not None

        det.handle_key_up(mod.SPACEBAR_KEYCODE)
        # After hold end, selector should be reset
        assert selector.active_keycode is None


class TestGhostIndicators:
    """Test ghost indicator overlay rendering for route keys."""

    @pytest.fixture
    def overlay_module(self, mock_pyobjc):
        """Import spoke.overlay with mocked PyObjC."""
        for name in list(sys.modules):
            if name.startswith("spoke."):
                sys.modules.pop(name, None)
        # Pre-install a fake dedup module to avoid import issues
        fake_dedup = types.ModuleType("spoke.dedup")
        fake_dedup.ontology_term_spans = MagicMock(return_value=[])
        sys.modules["spoke.dedup"] = fake_dedup
        fake_glow = types.ModuleType("spoke.glow")
        fake_glow._alpha_field_to_image = MagicMock(return_value=(MagicMock(), MagicMock()))
        sys.modules["spoke.glow"] = fake_glow
        mod = importlib.import_module("spoke.overlay")
        yield mod
        for name in list(sys.modules):
            if name.startswith("spoke."):
                sys.modules.pop(name, None)

    def test_ghost_indicator_layer_exists(self, overlay_module, route_keys_module):
        """GhostIndicatorLayer should be instantiable with bindings."""
        bindings = route_keys_module.default_bindings()
        ghost = overlay_module.GhostIndicatorLayer(bindings=bindings)
        assert ghost is not None
        assert ghost._bindings == bindings

    def test_ghost_update_active_key(self, overlay_module, route_keys_module):
        """Updating the active key should change which ghost is sharpened."""
        bindings = route_keys_module.default_bindings()
        ghost = overlay_module.GhostIndicatorLayer(bindings=bindings)
        bracket = route_keys_module.BRACKET_RIGHT_KEYCODE

        ghost.set_active(bracket)
        assert ghost.active_keycode == bracket

        ghost.set_active(None)
        assert ghost.active_keycode is None

    def test_ghost_bracket_is_separate(self, overlay_module, route_keys_module):
        """The `]` key ghost should be marked as separate (pill to the right)."""
        bindings = route_keys_module.default_bindings()
        ghost = overlay_module.GhostIndicatorLayer(bindings=bindings)
        bracket = route_keys_module.BRACKET_RIGHT_KEYCODE
        assert ghost.is_bracket_key(bracket) is True
        # Number row keys are not bracket
        for kc in route_keys_module.NUMBER_ROW_KEYCODES:
            assert ghost.is_bracket_key(kc) is False
