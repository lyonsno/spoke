"""Tests for the tray — a speech-native stacked clipboard.

The tray is entered by releasing spacebar while shift is held during
recording. It holds recent transcriptions for review, insertion, sending
to the assistant, or later retrieval.

See docs/keyboard-grammar.md "The tray" for the full spec.
"""

import time
from unittest.mock import MagicMock, patch


def _make_delegate(main_module, monkeypatch, *, command_client=False):
    """Create a SpokeAppDelegate with mocked sub-components."""
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
    delegate._preview_session_token = 0
    delegate._preview_done = MagicMock()
    delegate._preview_done.set = MagicMock()
    delegate._local_inference_lock = MagicMock()
    # Tray state
    delegate._tray_stack = []
    delegate._tray_index = 0
    delegate._tray_active = False
    delegate._undoable_tray_insert = None
    delegate._tray_pending_inject = None
    # Recovery state (implementation detail of tray)
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


class TestTrayEntry:
    """Shift+release during recording enters the tray."""

    def test_shift_release_spawns_tray_transcription(self, main_module, monkeypatch):
        """Shift+release during recording should spawn tray transcription thread."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread") as MockThread:
            MockThread.return_value = MagicMock()
            d._on_hold_end(shift_held=True)

        # Tray transcription thread should be spawned
        MockThread.assert_called_once()
        # tray_active is deferred until transcription completes
        assert d._transcribing is True

    def test_shift_release_does_not_send_to_command(self, main_module, monkeypatch):
        """Tray must intercept — text should not go to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._send_text_as_command = MagicMock()
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread"):
            d._on_hold_end(shift_held=True)

        d._send_text_as_command.assert_not_called()

    def test_enter_held_sends_to_command(self, main_module, monkeypatch):
        """Enter held at release should send directly to assistant (fast path)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 2.0

        with patch.object(main_module.threading, "Thread") as MockThread:
            MockThread.return_value = MagicMock()
            d._on_hold_end(shift_held=False, enter_held=True)

        # Should route to command pathway, not tray
        assert d._tray_active is not True


class TestTrayStack:
    """Tray stack lifecycle — push, navigate, consume, delete."""

    def test_enter_tray_pushes_to_stack(self, main_module, monkeypatch):
        """Entering the tray should push text onto the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("hello world")

        assert d._tray_stack == ["hello world"]
        assert d._tray_index == 0
        assert d._tray_active is True

    def test_multiple_entries_stack(self, main_module, monkeypatch):
        """Multiple tray entries should stack with most recent on top."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("first")
        d._dismiss_tray()
        d._enter_tray("second")

        assert d._tray_stack == ["first", "second"]
        assert d._tray_index == 1  # viewing the most recent

    def test_navigate_down(self, main_module, monkeypatch):
        """Navigate down should show older entries."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "middle", "newest"]
        d._tray_index = 2
        d._tray_active = True

        d._tray_navigate_down()

        assert d._tray_index == 1

    def test_navigate_up(self, main_module, monkeypatch):
        """Navigate up should show more recent entries."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "middle", "newest"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_navigate_up()

        assert d._tray_index == 2

    def test_navigate_up_past_top_dismisses(self, main_module, monkeypatch):
        """Navigating up past the most recent entry should dismiss the tray."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "newest"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_navigate_up()

        assert d._tray_active is False

    def test_navigate_down_stops_at_bottom(self, main_module, monkeypatch):
        """Navigating down past the oldest entry should stop (no wrap)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["oldest", "newest"]
        d._tray_index = 0
        d._tray_active = True

        d._tray_navigate_down()

        assert d._tray_index == 0  # didn't move

    def test_delete_entry(self, main_module, monkeypatch):
        """Delete should remove the current entry from the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["keep", "delete me", "also keep"]
        d._tray_index = 1
        d._tray_active = True

        d._tray_delete_current()

        assert "delete me" not in d._tray_stack
        assert len(d._tray_stack) == 2

    def test_delete_last_entry_dismisses(self, main_module, monkeypatch):
        """Deleting the only entry should dismiss the tray."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["only one"]
        d._tray_index = 0
        d._tray_active = True

        d._tray_delete_current()

        assert d._tray_stack == []
        assert d._tray_active is False

    def test_insert_consumes_entry(self, main_module, monkeypatch):
        """Inserting text should remove it from the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["first", "insert this"]
        d._tray_index = 1
        d._tray_active = True
        d._recovery_text = "insert this"

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text"):
            d._tray_insert_current()

        assert "insert this" not in d._tray_stack

    def test_delayed_tray_insert_becomes_undoable(self, main_module, monkeypatch):
        """Completed tray insertion should become the next idle undo target."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_pending_inject = "insert this"

        with patch("spoke.__main__.inject_text") as mock_inject:
            d.trayInjectDelayed_(None)

        mock_inject.assert_called_once_with("insert this")
        assert d._undoable_tray_insert == "insert this"
        assert d._tray_pending_inject is None

    def test_dismiss_preserves_stack(self, main_module, monkeypatch):
        """Dismissing the tray should preserve the stack for re-entry."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["preserved"]
        d._tray_index = 0
        d._tray_active = True

        d._dismiss_tray()

        assert d._tray_active is False
        assert d._tray_stack == ["preserved"]

    def test_reentry_shows_top(self, main_module, monkeypatch):
        """Re-entering the tray should show the top of the stack."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["old", "recent"]
        d._tray_index = 0  # was viewing old item

        d._enter_tray("newest")

        assert d._tray_index == 2  # viewing the newest


class TestTrayGestures:
    """Gestures available while the tray is active."""

    def test_spacebar_hold_from_tray_starts_recording(self, main_module, monkeypatch):
        """Plain spacebar hold from tray should dismiss tray and start recording."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["old text"]
        d._tray_index = 0
        d._recovery_text = "old text"
        d._detector._shift_at_press = False  # plain spacebar, no shift
        d._detector._shift_latched = False

        d._on_hold_start()

        # Should dismiss tray and start recording
        d._capture.start.assert_called_once()
        assert d._tray_active is False

    def test_shift_spacebar_hold_from_tray_navigates(self, main_module, monkeypatch):
        """Shift+spacebar hold from tray should wait for release (navigation)."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["old text"]
        d._tray_index = 0
        d._recovery_text = "old text"
        d._detector._shift_at_press = True  # shift held

        d._on_hold_start()

        # Should NOT start recording — waits for release to navigate
        d._capture.start.assert_not_called()
        assert d._recovery_hold_active is True

    def test_enter_from_tray_sends_to_assistant(self, main_module, monkeypatch):
        """Enter key from tray should send current text to assistant."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_active = True
        d._tray_stack = ["send this"]
        d._tray_index = 0
        d._send_text_as_command = MagicMock()

        d._tray_send_current()

        d._send_text_as_command.assert_called_once_with("send this")
        assert "send this" not in d._tray_stack

    def test_short_shift_hold_recalls_into_tray(self, main_module, monkeypatch):
        """Short shift-hold should enter tray with last tray entry."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._tray_stack = ["previous text"]
        d._capture.stop.return_value = b"audio"
        d._record_start_time = time.monotonic() - 0.5  # < 800ms

        d._on_hold_end(shift_held=True)

        assert d._tray_active is True
        assert d._tray_index == 0  # viewing the existing entry

    def test_shift_empty_prefers_undo_of_last_tray_insert(self, main_module, monkeypatch):
        """Idle shift tap should undo the last tray insertion before plain tray recall."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)
        d._undoable_tray_insert = "restored text"
        d._capture.stop.return_value = b""

        with patch("spoke.__main__.undo_last_insert") as mock_undo:
            d._on_hold_end(shift_held=True)

        mock_undo.assert_called_once_with()
        assert d._undoable_tray_insert is None
        assert d._tray_active is True
        assert d._tray_stack == ["restored text"]
        assert d._tray_index == 0
        d._overlay.show_tray.assert_called_once_with("restored text")


class TestTrayRecoveryUnification:
    """Paste failure enters the tray automatically."""

    def test_paste_failure_enters_tray(self, main_module, monkeypatch):
        """OCR verification failure should enter tray."""
        d = _make_delegate(main_module, monkeypatch, command_client=True)

        d._enter_tray("failed paste text")

        assert d._tray_active is True
        assert "failed paste text" in d._tray_stack
