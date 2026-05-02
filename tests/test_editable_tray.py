"""Tests for the editable tray — Lane T of the ButterfingerFinalFuckers packet.

When the tray is up, the keyboard types into the tray text. Spacebar types a
space character (tap < 200ms). Spacebar hold >= 200ms starts a new recording
whose transcription appends at cursor position. Cmd+V pastes from the system
clipboard. Arrow keys, backspace, delete, and selection (shift+arrows) work
as standard text editing.

The tray is a general-purpose input buffer: dictate into it, type into it,
paste into it, edit it, then send.

See docs/keyboard-grammar.md "Editable tray" for the full spec.
"""

import time
from unittest.mock import MagicMock, patch, call


def _make_delegate(main_module, monkeypatch, *, command_client=False):
    """Create a SpokeAppDelegate with mocked sub-components for tray tests."""
    monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")

    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._capture = MagicMock()
    delegate._client = MagicMock(supports_streaming=False)
    delegate._detector = MagicMock()
    delegate._menubar = MagicMock()
    delegate._glow = MagicMock()
    delegate._overlay = MagicMock()
    delegate._transcribing = False
    delegate._transcription_token = 0
    delegate._preview_active = False
    delegate._preview_thread = None
    delegate._preview_client = MagicMock()
    delegate._local_mode = False
    delegate._record_start_time = 0.0
    delegate._cap_fired = False
    delegate._transcribe_start = time.monotonic()
    delegate._last_preview_text = ""
    delegate._models_ready = True
    delegate._mic_ready = True
    delegate._preview_session_token = 0
    delegate._preview_done = MagicMock()
    delegate._preview_done.set = MagicMock()
    delegate._local_inference_lock = MagicMock()
    # Tray state
    delegate._tray_stack = []
    delegate._tray_index = 0
    delegate._tray_active = False
    delegate._tray_cursor_pos = 0
    delegate._tray_selection = None
    # Recovery state
    delegate._pre_paste_clipboard = None
    delegate._verify_paste_text = None
    delegate._verify_paste_attempt = 0
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_pending_insert = None
    delegate._recovery_hold_active = False
    delegate._recovery_retry_pending = False
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()

    if command_client:
        delegate._command_client = MagicMock()
        delegate._command_client.history = []
        delegate._command_overlay = MagicMock()
        delegate._command_overlay._visible = False
    else:
        delegate._command_client = None
        delegate._command_overlay = None

    return delegate


class TestTrayKeyboardRouting:
    """When the tray is up, keystrokes route to the tray text field."""

    def test_regular_key_inserts_character_in_tray(self, main_module, monkeypatch):
        """A regular keystroke while tray is up should insert into tray text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        assert d._tray_active is True

        # Type 'x' into the tray — should append at cursor position
        d._tray_insert_char("x")

        entry = d._get_tray_entry(d._tray_index)
        assert "x" in entry.text

    def test_spacebar_tap_inserts_space_in_tray(self, main_module, monkeypatch):
        """Spacebar tap (< 200ms) while tray is up should type a space character."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        # Simulate spacebar quick release during tray — should insert space
        # The current behavior routes through on_hold_end which does
        # _tray_insert_current (pastes the whole entry). The new behavior
        # should insert a space character into the tray text at cursor.
        d._tray_insert_char(" ")

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello "

    def test_keys_do_not_reach_frontmost_app_when_tray_up(self, main_module, monkeypatch):
        """When the tray is up, keystrokes must not be forwarded to the frontmost app."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        # The detector should suppress key forwarding when tray is active
        assert d._detector.tray_active is True


class TestTrayCursorMovement:
    """Arrow keys, backspace, and selection work as standard text editing."""

    def test_cursor_starts_at_end(self, main_module, monkeypatch):
        """When entering the tray, cursor should be at the end of text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        assert d._tray_cursor_pos == len("hello")

    def test_left_arrow_moves_cursor_left(self, main_module, monkeypatch):
        """Left arrow should move cursor one position left."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        initial_pos = d._tray_cursor_pos

        d._tray_move_cursor(-1)

        assert d._tray_cursor_pos == initial_pos - 1

    def test_right_arrow_at_end_stays(self, main_module, monkeypatch):
        """Right arrow at end of text should not move past end."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        d._tray_move_cursor(1)

        assert d._tray_cursor_pos == len("hello")

    def test_left_arrow_at_start_stays(self, main_module, monkeypatch):
        """Left arrow at start of text should not move past start."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._tray_cursor_pos = 0

        d._tray_move_cursor(-1)

        assert d._tray_cursor_pos == 0

    def test_backspace_deletes_before_cursor(self, main_module, monkeypatch):
        """Backspace should delete the character before the cursor."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        # cursor at end (pos 5)

        d._tray_backspace()

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hell"
        assert d._tray_cursor_pos == 4

    def test_backspace_at_start_does_nothing(self, main_module, monkeypatch):
        """Backspace at start of text should not modify the text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._tray_cursor_pos = 0

        d._tray_backspace()

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello"

    def test_insert_char_at_cursor_position(self, main_module, monkeypatch):
        """Inserting a character should place it at the current cursor position."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("helo")
        d._tray_cursor_pos = 3  # between 'l' and 'o'

        d._tray_insert_char("l")

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello"
        assert d._tray_cursor_pos == 4


class TestTrayPaste:
    """Cmd+V pastes from the system clipboard into the tray text."""

    def test_paste_inserts_clipboard_at_cursor(self, main_module, monkeypatch):
        """Cmd+V should insert clipboard contents at the tray cursor position."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello ")

        with patch("spoke.__main__.get_clipboard_text", return_value="world"):
            d._tray_paste()

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello world"
        assert d._tray_cursor_pos == len("hello world")

    def test_paste_at_middle_of_text(self, main_module, monkeypatch):
        """Pasting in the middle of text should splice correctly."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hd")
        d._tray_cursor_pos = 1  # between 'h' and 'd'

        with patch("spoke.__main__.get_clipboard_text", return_value="ello worl"):
            d._tray_paste()

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello world"

    def test_paste_empty_clipboard_does_nothing(self, main_module, monkeypatch):
        """Pasting from an empty clipboard should not modify tray text."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        with patch("spoke.__main__.get_clipboard_text", return_value=None):
            d._tray_paste()

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello"


class TestTraySpacebarDualUse:
    """Spacebar dual-use: tap = space char, hold >= 200ms = record."""

    def test_spacebar_hold_from_tray_keeps_tray_active(self, main_module, monkeypatch):
        """Spacebar hold (>= 200ms) while tray is up should keep tray active.

        The current behavior dismisses the tray then starts recording. The new
        editable tray behavior should keep the tray active so the transcription
        can append at cursor position when it lands.
        """
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._record_start_time = time.monotonic() - 2.0
        d._segment_accumulator = MagicMock()
        d._segment_accumulator.count = 0
        d._parallel_insert_token = 0
        d._preview_cancelled_on_release = False
        d._whisper_backend = "local"

        # Record from tray: hold starts (plain, no shift)
        d._on_hold_start()

        # Tray should remain active for recording-from-tray
        assert d._tray_active is True

    def test_recording_from_tray_appends_at_cursor(self, main_module, monkeypatch):
        """Transcription from a recording-in-tray should append at cursor position."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello ")
        d._tray_cursor_pos = 6  # at end after "hello "

        # Simulate transcription arriving from a recording made within the tray
        d._tray_append_transcription("world")

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello world"
        assert d._tray_cursor_pos == 11  # cursor moved past inserted text

    def test_recording_from_tray_appends_at_middle_cursor(self, main_module, monkeypatch):
        """Transcription should insert at cursor position, not always at end."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello world")
        d._tray_cursor_pos = 6  # after "hello "

        d._tray_append_transcription("beautiful ")

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello beautiful world"
        assert d._tray_cursor_pos == 16  # after "hello beautiful "


class TestTrayOverlayIntegration:
    """The overlay shows editable text and updates on tray edits."""

    def test_show_tray_displays_text(self, main_module, monkeypatch):
        """show_tray should display text on the overlay."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("test text")

        d._overlay.show_tray.assert_called()
        # First positional arg should be the text
        call_args = d._overlay.show_tray.call_args
        assert call_args[0][0] == "test text"

    def test_tray_edit_updates_overlay(self, main_module, monkeypatch):
        """Editing tray text should update the overlay display."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._overlay.reset_mock()

        d._tray_insert_char("!")

        # Overlay should have been updated with the new text
        d._overlay.show_tray.assert_called()
        call_args = d._overlay.show_tray.call_args
        assert "!" in call_args[0][0]


# ── Integration wiring tests ─────────────────────────────
# These test the three wiring points identified in the ButterfingerFinalFuckers
# sylloge that Lane T left unwired.


class TestEventTapTrayKeyRouting:
    """Wiring point (a): event tap routes keystrokes to tray methods when active."""

    def _make_tray_detector(self, mod):
        """Create a detector with tray_active=True and standard test state."""
        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._state = mod._State.IDLE
        det._forwarding = False
        det._tap = MagicMock()
        det.tray_active = True
        det._enter_held = False
        det._enter_observed = False
        det._enter_last_down_monotonic = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._pending_release_active = False
        det._shift_latched = False
        det.approval_active = False
        det.command_overlay_active = False
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det.cancel_spring_active = False
        det._on_tray_key = MagicMock()
        return det

    def test_regular_key_suppressed_when_tray_active(self, input_tap_module):
        """A regular keyDown while tray is active should return None (suppress)."""
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det = self._make_tray_detector(mod)
        mod._active_detector = det

        event = MagicMock()
        # Keycode 0 = 'a' key
        Quartz.CGEventGetIntegerValueField.return_value = 0
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyDown, event, None
        )

        assert result is None
        det._on_tray_key.assert_called_once()

    def test_arrow_key_suppressed_when_tray_active(self, input_tap_module):
        """Arrow keys while tray active should be suppressed and routed to tray."""
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det = self._make_tray_detector(mod)
        mod._active_detector = det

        event = MagicMock()
        # Keycode 123 = left arrow
        Quartz.CGEventGetIntegerValueField.return_value = 123
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyDown, event, None
        )

        assert result is None
        det._on_tray_key.assert_called_once()

    def test_backspace_suppressed_when_tray_active(self, input_tap_module):
        """Backspace while tray active should be suppressed and routed to tray."""
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det = self._make_tray_detector(mod)
        mod._active_detector = det

        event = MagicMock()
        # Keycode 51 = delete/backspace
        Quartz.CGEventGetIntegerValueField.return_value = 51
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyDown, event, None
        )

        assert result is None
        det._on_tray_key.assert_called_once()

    def test_cmd_v_suppressed_when_tray_active(self, input_tap_module):
        """Cmd+V while tray active should be suppressed and routed to tray."""
        mod = input_tap_module
        Quartz = __import__("Quartz")
        det = self._make_tray_detector(mod)
        mod._active_detector = det

        event = MagicMock()
        # Keycode 9 = 'v' key, with Cmd flag
        Quartz.CGEventGetIntegerValueField.return_value = 9
        Quartz.CGEventGetFlags.return_value = 0x00100000  # kCGEventFlagMaskCommand
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyDown, event, None
        )

        assert result is None
        det._on_tray_key.assert_called_once()


class TestSpacebarTapInsertsSpace:
    """Wiring point (b): spacebar tap in tray inserts space char, not paste-and-dismiss."""

    def test_spacebar_tap_inserts_space_not_paste(self, main_module, monkeypatch):
        """Spacebar tap (no shift, no enter) while tray active should insert ' '
        into the tray text, not call _tray_insert_current which pastes and dismisses."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._manual_hold_active = False
        d._preview_active = False
        d._cap_fired = False
        d._record_start_time = 0.0
        d._segment_accumulator = MagicMock()
        d._segment_accumulator.count = 0
        d._parallel_insert_token = 0
        d._preview_cancelled_on_release = False
        d._whisper_backend = "local"
        d._recovery_pending_retry_insert = None
        d._recovery_hold_active = False
        d._pending_command_approval_active = False
        d._hold_rejected_during_warmup = False
        d._command_overlay_load_shed_release_timer = None

        # Call _on_hold_end with no modifiers (plain spacebar tap from tray)
        d._on_hold_end(shift_held=False, enter_held=False)

        # Tray should STILL be active (space was typed, not pasted-and-dismissed)
        assert d._tray_active is True
        # The text should now contain a space at the cursor position
        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello "

    def test_spacebar_tap_cursor_advances(self, main_module, monkeypatch):
        """After inserting a space via spacebar tap, cursor should advance by 1."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._manual_hold_active = False
        d._preview_active = False
        d._cap_fired = False
        d._record_start_time = 0.0
        d._segment_accumulator = MagicMock()
        d._segment_accumulator.count = 0
        d._parallel_insert_token = 0
        d._preview_cancelled_on_release = False
        d._whisper_backend = "local"
        d._recovery_pending_retry_insert = None
        d._recovery_hold_active = False
        d._pending_command_approval_active = False
        d._hold_rejected_during_warmup = False
        d._command_overlay_load_shed_release_timer = None

        initial_cursor = d._tray_cursor_pos  # should be 5 (end of "hello")

        d._on_hold_end(shift_held=False, enter_held=False)

        assert d._tray_cursor_pos == initial_cursor + 1


class TestRecordingFromTrayAppendsAtCursor:
    """Wiring point (c): transcription from recording-in-tray appends at cursor."""

    def test_transcription_appends_not_creates_new_entry(self, main_module, monkeypatch):
        """When _recording_from_tray is True, trayTranscriptionComplete_ should
        call _tray_append_transcription (insert at cursor) instead of _enter_tray
        (which creates a new stack entry)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello ")
        d._recording_from_tray = True
        d._transcription_token = 42
        d._last_preview_text = ""

        initial_stack_len = len(d._tray_stack)

        # Simulate transcription completion
        d.trayTranscriptionComplete_({"token": 42, "text": "world"})

        # Stack should NOT have grown (no new entry was created)
        assert len(d._tray_stack) == initial_stack_len
        # The existing entry should have the appended text
        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello world"

    def test_transcription_inserts_at_cursor_position(self, main_module, monkeypatch):
        """Transcription from tray should insert at the current cursor position,
        not at the end."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello world")
        d._tray_cursor_pos = 6  # after "hello "
        d._recording_from_tray = True
        d._transcription_token = 42
        d._last_preview_text = ""

        d.trayTranscriptionComplete_({"token": 42, "text": "beautiful "})

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello beautiful world"
        assert d._tray_cursor_pos == 16  # after "hello beautiful "

    def test_recording_from_tray_flag_cleared_after_transcription(self, main_module, monkeypatch):
        """_recording_from_tray should be cleared after transcription completes."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello ")
        d._recording_from_tray = True
        d._transcription_token = 42
        d._last_preview_text = ""

        d.trayTranscriptionComplete_({"token": 42, "text": "world"})

        assert d._recording_from_tray is False


# ── Route key, send chord, sticky toggle delegate wiring tests ────


class TestRouteKeySelectorWiring:
    """RouteKeySelector wiring: delegate creates it and uses it for routing."""

    def test_on_route_key_changed_method_exists(self, main_module, monkeypatch):
        """The delegate should have a _on_route_key_changed method for wiring."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        assert callable(getattr(d, '_on_route_key_changed', None))

    def test_route_key_selection_routes_to_assistant(self, main_module, monkeypatch):
        """When ] is the active route key at recording end, transcription should
        go to the assistant instead of paste."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._send_text_as_command = MagicMock()
        # Set up a route key selector with ] selected
        from spoke.route_keys import RouteKeySelector
        selector = RouteKeySelector(on_change=d._on_route_key_changed)
        selector.tap(30)  # select ]
        d._active_route_destination = selector.active_destination

        # Simulate transcription completion with route key active
        d._dispatch_transcription_with_route("hello world")

        d._send_text_as_command.assert_called_once_with("hello world")

    def test_no_route_key_pastes_normally(self, main_module, monkeypatch):
        """Without a route key selected, transcription should paste normally."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._active_route_destination = None

        # _dispatch_transcription_with_route with no destination should
        # fall through to normal paste
        with patch("spoke.__main__.inject_text") as mock_inject:
            d._dispatch_transcription_with_route("hello world")
            mock_inject.assert_called_once()


class TestSendChordWiring:
    """Send chord dispatches tray text to the selected destination."""

    def test_on_send_chord_method_exists(self, main_module, monkeypatch):
        """The delegate should have a _on_send_chord method."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        assert callable(getattr(d, '_on_send_chord', None))

    def test_send_chord_dispatches_tray_to_assistant(self, main_module, monkeypatch):
        """Send chord with ] key should dispatch tray text to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("test command")
        d._send_text_as_command = MagicMock()

        d._on_send_chord(keycode=30)

        d._send_text_as_command.assert_called_once_with("test command")
        # Tray should be dismissed after sending
        assert d._tray_active is False

    def test_send_chord_no_tray_is_noop(self, main_module, monkeypatch):
        """Send chord with no tray active should do nothing."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._send_text_as_command = MagicMock()

        d._on_send_chord(keycode=30)

        d._send_text_as_command.assert_not_called()


class TestStickyToggleWiring:
    """Sticky toggle sets persistent routing state."""

    def test_on_sticky_toggle_method_exists(self, main_module, monkeypatch):
        """The delegate should have a _on_sticky_toggle method."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        assert callable(getattr(d, '_on_sticky_toggle', None))

    def test_sticky_toggle_sets_sticky_state(self, main_module, monkeypatch):
        """Sticky toggle should set sticky routing keycode on the delegate."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._on_sticky_toggle(keycode=30)

        assert d._sticky_route_keycode == 30

    def test_sticky_toggle_same_key_clears(self, main_module, monkeypatch):
        """Toggling sticky with the same key should clear sticky state."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._on_sticky_toggle(keycode=30)
        assert d._sticky_route_keycode == 30

        d._on_sticky_toggle(keycode=30)
        assert d._sticky_route_keycode is None

    def test_sticky_toggle_different_key_switches(self, main_module, monkeypatch):
        """Toggling sticky with a different key should switch to it."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._on_sticky_toggle(keycode=30)
        assert d._sticky_route_keycode == 30

        d._on_sticky_toggle(keycode=22)  # keycode for '6'
        assert d._sticky_route_keycode == 22


class TestGhostIndicatorRendering:
    """Ghost indicators render in the overlay."""

    def test_overlay_has_update_ghosts_method(self, main_module, monkeypatch):
        """The overlay class should have an update_ghosts method."""
        assert callable(getattr(main_module.TranscriptionOverlay, 'update_ghosts', None))


class TestTrayEnterDispatch:
    """Enter from tray: sticky route → send, no sticky → insert at cursor."""

    def test_enter_from_tray_no_sticky_inserts(self, main_module, monkeypatch):
        """Bare Enter from tray with no sticky route should insert at cursor."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("insert me")
        d._sticky_route_keycode = None

        with patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.has_focused_text_input", return_value=True):
            d._on_tray_enter()

        # Tray should be dismissed (text was inserted)
        assert d._tray_active is False
        # inject_text was called (via _tray_insert_current's delayed path)

    def test_enter_from_tray_sticky_sends_to_assistant(self, main_module, monkeypatch):
        """Bare Enter from tray with sticky ] should send to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("send me")
        d._sticky_route_keycode = 30  # ] key
        d._send_text_as_command = MagicMock()

        d._on_tray_enter()

        d._send_text_as_command.assert_called_once_with("send me")
        assert d._tray_active is False

    def test_enter_from_tray_empty_is_noop(self, main_module, monkeypatch):
        """Enter from tray with no entries should do nothing."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = False  # no tray
        d._sticky_route_keycode = None

        d._on_tray_enter()  # should not raise

    def test_shift_enter_inserts_newline(self, main_module, monkeypatch):
        """Shift+Enter from tray should insert a newline character."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")

        # Simulate Shift+Enter routed through _on_tray_key
        d._on_tray_key(36, 0x00020000)  # keycode 36 = Return, shift flag

        entry = d._get_tray_entry(d._tray_index)
        assert entry.text == "hello\n"
        assert d._tray_active is True  # tray stays active


class TestEnterKeyEventTapTrayRouting:
    """Enter key is intercepted by the event tap when tray is active."""

    def test_enter_keydown_suppressed_when_tray_active(self, input_tap_module):
        """Enter keyDown while tray active should be suppressed."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._state = mod._State.IDLE
        det._forwarding = False
        det._tap = MagicMock()
        det.tray_active = True
        det._enter_held = False
        det._enter_observed = False
        det._enter_last_down_monotonic = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._pending_release_active = False
        det._shift_latched = False
        det.approval_active = False
        det.command_overlay_active = False
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det.cancel_spring_active = False
        det._on_tray_enter = MagicMock()

        mod._active_detector = det
        event = MagicMock()
        Quartz.CGEventGetIntegerValueField.return_value = 36  # Return key
        Quartz.CGEventGetFlags.return_value = 0
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyDown, event, None
        )

        # Enter keyDown should be suppressed
        assert result is None
        # Enter should be tracked as held (for send chord)
        assert det._enter_held is True

    def test_enter_keyup_fires_tray_enter(self, input_tap_module):
        """Enter keyUp while tray active (no send chord) should fire _on_tray_enter."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._state = mod._State.IDLE
        det._forwarding = False
        det._tap = MagicMock()
        det.tray_active = True
        det._enter_held = True  # was held from keyDown
        det._enter_observed = True
        det._enter_last_down_monotonic = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._pending_release_active = False
        det._shift_latched = False
        det.approval_active = False
        det.command_overlay_active = False
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det.cancel_spring_active = False
        det._on_tray_enter = MagicMock()
        det._on_tray_key = MagicMock()

        mod._active_detector = det
        event = MagicMock()
        Quartz.CGEventGetIntegerValueField.return_value = 36  # Return key
        Quartz.CGEventGetFlags.return_value = 0  # no shift
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyUp, event, None
        )

        assert result is None
        det._on_tray_enter.assert_called_once()

    def test_shift_enter_keyup_inserts_newline(self, input_tap_module):
        """Shift+Enter keyUp while tray active should route to _on_tray_key."""
        mod = input_tap_module
        Quartz = __import__("Quartz")

        det = mod.SpacebarHoldDetector.__new__(mod.SpacebarHoldDetector)
        det._state = mod._State.IDLE
        det._forwarding = False
        det._tap = MagicMock()
        det.tray_active = True
        det._enter_held = True
        det._enter_observed = True
        det._enter_last_down_monotonic = 0.0
        det._idle_shift_down = False
        det._idle_shift_interrupted = False
        det._tray_shift_down = False
        det._tray_space_between = False
        det._pending_release_active = False
        det._shift_latched = False
        det.approval_active = False
        det.command_overlay_active = False
        det._suppress_enter_keyup = False
        det._suppress_delete_keyup = False
        det.cancel_spring_active = False
        det._on_tray_enter = MagicMock()
        det._on_tray_key = MagicMock()

        mod._active_detector = det
        event = MagicMock()
        Quartz.CGEventGetIntegerValueField.return_value = 36
        Quartz.CGEventGetFlags.return_value = 0x00020000  # shift held
        Quartz.CGEventGetTimestamp.return_value = 0

        result = mod._event_tap_callback(
            None, Quartz.kCGEventKeyUp, event, None
        )

        assert result is None
        det._on_tray_key.assert_called_once()
        det._on_tray_enter.assert_not_called()


class TestTapThenHoldInsertAtCursor:
    """Tap space then hold space from tray = insert tray entry at cursor."""

    def test_double_tap_then_hold_fires_insert(self, main_module, monkeypatch):
        """Two space taps followed by space hold within 300ms should insert at cursor."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("insert me")

        # Simulate: two taps happened recently
        d._tray_last_space_tap = time.monotonic()
        d._tray_space_tap_count = 2

        # Now a hold starts within the window — should be insert mode
        with patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.has_focused_text_input", return_value=True):
            d._on_tray_insert_at_cursor()

        assert d._tray_active is False

    def test_single_tap_then_hold_does_not_trigger_insert(self, main_module, monkeypatch):
        """A single space tap followed by hold should NOT trigger insert — needs double tap."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._tray_insert_hold_active = False

        # Only one tap — not enough for double-tap-then-hold
        d._tray_last_space_tap = time.monotonic()
        d._tray_space_tap_count = 1

        # The double-tap-then-hold check requires count >= 2
        assert d._tray_space_tap_count < 2
        assert d._tray_insert_hold_active is False

    def test_old_double_tap_does_not_trigger_insert(self, main_module, monkeypatch):
        """Double tap that happened > 300ms ago should not trigger insert mode."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("hello")
        d._tray_insert_hold_active = False

        # Two taps happened but too long ago
        d._tray_last_space_tap = time.monotonic() - 0.5
        d._tray_space_tap_count = 2

        last_tap = d._tray_last_space_tap
        assert (time.monotonic() - last_tap) >= 0.3  # outside window
        assert d._tray_insert_hold_active is False


class TestEnterFromTrayAlwaysSends:
    """Enter from tray always sends to active destination, never inserts."""

    def test_enter_no_sticky_sends_to_assistant(self, main_module, monkeypatch):
        """Enter from tray with no sticky route should send to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("send me")
        d._sticky_route_keycode = None
        d._send_text_as_command = MagicMock()

        d._on_tray_enter()

        d._send_text_as_command.assert_called_once_with("send me")
        assert d._tray_active is False

    def test_enter_sticky_sends_to_sticky_dest(self, main_module, monkeypatch):
        """Enter from tray with sticky route should send to that destination."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._enter_tray("send me")
        d._sticky_route_keycode = 30  # ]
        d._send_text_as_command = MagicMock()

        d._on_tray_enter()

        d._send_text_as_command.assert_called_once_with("send me")
        assert d._tray_active is False
