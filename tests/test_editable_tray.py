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
