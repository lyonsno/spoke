"""Tests for paste-failure recovery overlay and delegate wiring.

Covers: overlay recovery mode, three-column button layout, clipboard toggle,
spacebar dismiss, and delegate-level recovery flow branching.
"""

import time
from unittest.mock import MagicMock, patch, call


def _make_delegate(main_module, monkeypatch):
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
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_previous_app = None
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()
    return delegate


class TestRecoveryFlowBranching:
    """_inject_result_text should branch on has_focused_text_input."""

    def test_normal_paste_when_text_field_focused(self, main_module, monkeypatch):
        """When a text field is focused, use normal inject_text path."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._inject_result_text("hello world", "Pasted!")

        mock_inject.assert_called_once()
        # Overlay should get the text and then be hidden
        d._overlay.set_text.assert_called_with("hello world")

    def test_recovery_mode_when_no_text_field(self, main_module, monkeypatch):
        """When no text field is focused, enter recovery mode."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.has_focused_text_input", return_value=False), \
             patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.set_pasteboard_only") as mock_pb:
            d._inject_result_text("hello world", "Pasted!")

        # Should NOT have called inject_text (no Cmd+V)
        mock_inject.assert_not_called()
        # Should have put text on pasteboard without pasting
        mock_pb.assert_called_once_with("hello world")
        # Should show recovery overlay
        d._overlay.show_recovery.assert_called_once()

    def test_recovery_sets_menubar_status(self, main_module, monkeypatch):
        """Recovery mode should update menubar to indicate no text field."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.has_focused_text_input", return_value=False), \
             patch("spoke.__main__.inject_text"), \
             patch("spoke.__main__.set_pasteboard_only"):
            d._inject_result_text("hello world", "Pasted!")

        # Menubar should show recovery status
        status_calls = [c[0][0] for c in d._menubar.set_status_text.call_args_list]
        assert any("text field" in s.lower() or "⌘v" in s.lower() or "clipboard" in s.lower()
                    for s in status_calls), f"Expected recovery status, got: {status_calls}"


class TestRecoveryDismiss:
    """Recovery overlay dismiss behavior."""

    def test_spacebar_hold_dismisses_recovery(self, main_module, monkeypatch):
        """Starting a new recording should dismiss recovery overlay."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "some text"
        d._overlay._recovery_mode = True

        with patch("spoke.__main__.restore_pasteboard"):
            d._on_hold_start()

        # Recovery state should be cleared
        assert d._recovery_text is None
        # Overlay should have dismiss_recovery called
        d._overlay.dismiss_recovery.assert_called_once()

    def test_dismiss_button_restores_clipboard(self, main_module, monkeypatch):
        """Dismiss callback should restore original clipboard and hide overlay."""
        d = _make_delegate(main_module, monkeypatch)
        saved_clipboard = [("public.utf8-plain-text", b"original text")]
        d._recovery_saved_clipboard = saved_clipboard
        d._recovery_text = "some transcription"

        with patch("spoke.__main__.restore_pasteboard") as mock_restore:
            d._on_recovery_dismiss()

        mock_restore.assert_called_once()
        d._overlay.dismiss_recovery.assert_called_once()
        d._menubar.set_status_text.assert_called_with("Ready — hold spacebar")


class TestRecoveryInsert:
    """Recovery Insert button behavior."""

    def test_insert_pastes_when_text_field_available(self, main_module, monkeypatch):
        """Insert should Cmd+V and dismiss when a text field is now focused."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"old")]
        d._recovery_previous_app = None

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.restore_pasteboard"):
            d._on_recovery_insert()

        mock_inject.assert_called_once()
        d._overlay.dismiss_recovery.assert_called_once()

    def test_insert_noop_when_no_text_field(self, main_module, monkeypatch):
        """Insert should be a no-op (with visual reject) when no text field."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"

        with patch("spoke.__main__.has_focused_text_input", return_value=False), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._on_recovery_insert()

        mock_inject.assert_not_called()
        # Should signal rejection on the overlay
        d._overlay.flash_insert_reject.assert_called_once()


class TestInsertClipboardRestore:
    """Verify Insert path restores original clipboard, not transcription."""

    def test_insert_restores_original_clipboard_before_paste(self, main_module, monkeypatch):
        """Insert should restore original clipboard before inject_text so the
        save/restore cycle inside inject_text preserves the original contents."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"original")]
        d._recovery_previous_app = None

        restore_calls = []
        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.restore_pasteboard") as mock_restore:
            # Track call order
            mock_restore.side_effect = lambda saved: restore_calls.append(("restore", saved))
            mock_inject.side_effect = lambda text, on_restored=None: restore_calls.append(("inject", text))
            d._on_recovery_insert()

        # restore_pasteboard must be called BEFORE inject_text
        assert len(restore_calls) == 2
        assert restore_calls[0][0] == "restore"
        assert restore_calls[1][0] == "inject"
        # The restored clipboard should be the original, not the transcription
        assert restore_calls[0][1] == [("public.utf8-plain-text", b"original")]


class TestRecoveryClipboardToggle:
    """Recovery Clipboard button toggle behavior."""

    def test_first_click_shows_old_clipboard_preview(self, main_module, monkeypatch):
        """First Clipboard click: transcription already on clipboard, show old contents preview."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"original")]
        d._recovery_clipboard_state = "idle"  # not yet toggled

        d._on_recovery_clipboard_toggle()

        # Should transition to transcription_on_clipboard state
        assert d._recovery_clipboard_state == "transcription_on_clipboard"
        # Should show preview of old clipboard contents
        d._overlay.set_clipboard_preview.assert_called_once()

    def test_second_click_restores_old_clipboard(self, main_module, monkeypatch):
        """Second click: restore old clipboard, show transcription preview."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"original")]
        d._recovery_clipboard_state = "transcription_on_clipboard"

        with patch("spoke.__main__.restore_pasteboard") as mock_restore:
            d._on_recovery_clipboard_toggle()

        mock_restore.assert_called_once()

    def test_toggle_cycles_back_and_forth(self, main_module, monkeypatch):
        """Multiple clicks should alternate clipboard state."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"original")]
        d._recovery_clipboard_state = "idle"

        with patch("spoke.__main__.set_pasteboard_only"), \
             patch("spoke.__main__.restore_pasteboard"):
            d._on_recovery_clipboard_toggle()
            assert d._recovery_clipboard_state == "transcription_on_clipboard"

            d._on_recovery_clipboard_toggle()
            assert d._recovery_clipboard_state == "original_on_clipboard"

            d._on_recovery_clipboard_toggle()
            assert d._recovery_clipboard_state == "transcription_on_clipboard"


class TestSetPasteboardOnly:
    """Test inject.set_pasteboard_only function."""

    def test_sets_pasteboard_without_cmd_v(self, inject_module):
        """Should set pasteboard text but NOT post keyboard events."""
        AppKit = __import__("AppKit")
        Quartz = __import__("Quartz")
        Quartz.CGEventPost.reset_mock()

        mock_pb = MagicMock()
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.set_pasteboard_only("recovery text")

        mock_pb.clearContents.assert_called_once()
        mock_pb.setString_forType_.assert_called_with(
            "recovery text", AppKit.NSPasteboardTypeString
        )
        # No keyboard events should be posted
        Quartz.CGEventPost.assert_not_called()

    def test_empty_text_is_noop(self, inject_module):
        """set_pasteboard_only('') should do nothing."""
        AppKit = __import__("AppKit")
        inject_module.set_pasteboard_only("")
        AppKit.NSPasteboard.generalPasteboard.assert_not_called()

    def test_no_restore_timer_scheduled(self, inject_module):
        """set_pasteboard_only should NOT schedule a pasteboard restore."""
        AppKit = __import__("AppKit")
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()

        mock_pb = MagicMock()
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.set_pasteboard_only("text")

        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_not_called()


class TestSavePasteboardPublic:
    """Test inject.save_pasteboard (public API for recovery)."""

    def test_save_pasteboard_returns_saved_items(self, inject_module):
        """save_pasteboard() should return the current pasteboard contents."""
        AppKit = __import__("AppKit")

        mock_item = MagicMock()
        mock_item.types.return_value = ["public.utf8-plain-text"]
        mock_item.dataForType_.return_value = b"clipboard text"

        mock_pb = MagicMock()
        mock_pb.pasteboardItems.return_value = [mock_item]
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        result = inject_module.save_pasteboard()
        assert result is not None
        assert len(result) == 1

    def test_save_pasteboard_returns_none_when_empty(self, inject_module):
        """save_pasteboard() should return None when pasteboard is empty."""
        AppKit = __import__("AppKit")

        mock_pb = MagicMock()
        mock_pb.pasteboardItems.return_value = []
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        result = inject_module.save_pasteboard()
        assert result is None


class TestRestorePasteboardPublic:
    """Test inject.restore_pasteboard (public API for recovery)."""

    def test_restore_pasteboard_clears_and_writes(self, inject_module):
        """restore_pasteboard() should clear and write saved items."""
        AppKit = __import__("AppKit")

        mock_pb = MagicMock()
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        saved = [("public.utf8-plain-text", b"original text")]
        inject_module.restore_pasteboard(saved)

        mock_pb.clearContents.assert_called_once()
        mock_pb.writeObjects_.assert_called_once()
