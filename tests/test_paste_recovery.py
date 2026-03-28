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
        # Overlay should be ordered out before the focus check
        d._overlay.order_out.assert_called()

    def test_always_attempts_paste(self, main_module, monkeypatch):
        """_inject_result_text should always attempt paste, regardless of focus."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.save_pasteboard", return_value=None):
            d._inject_result_text("hello world", "Pasted!")

        # Should always attempt paste
        mock_inject.assert_called_once()

    def test_schedules_ocr_verification(self, main_module, monkeypatch):
        """After paste, should schedule OCR verification timer."""
        Foundation = __import__("Foundation")
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.inject_text"), \
             patch("spoke.__main__.save_pasteboard", return_value=None):
            d._inject_result_text("hello world", "Pasted!")

        # Verification timer should be scheduled
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called()
        assert d._verify_paste_text == "hello world"

    def test_verify_result_enters_recovery_on_failure(self, main_module, monkeypatch):
        """verifyPasteResult_ should enter recovery when OCR doesn't find the text."""
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = "hello world"
        d._pre_paste_clipboard = [("public.utf8-plain-text", b"original")]

        with patch("spoke.__main__.set_pasteboard_only"), \
             patch("spoke.__main__.save_pasteboard", return_value=None):
            d.verifyPasteResult_({"found": False, "text": "hello world", "attempt": 1})

        d._overlay.show_recovery.assert_called_once()

    def test_verify_result_clears_state_on_success(self, main_module, monkeypatch):
        """verifyPasteResult_ should clear verify state when text is found."""
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = "hello world"

        d.verifyPasteResult_({"found": True, "text": "hello world", "attempt": 0})

        assert d._verify_paste_text is None


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

    def test_insert_dismisses_overlay_and_schedules_paste(self, main_module, monkeypatch):
        """Insert should dismiss overlay and schedule a delayed paste."""
        Foundation = __import__("Foundation")
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"
        d._recovery_saved_clipboard = [("public.utf8-plain-text", b"old")]
        d._recovery_previous_app = None

        d._on_recovery_insert()

        # Overlay should be dismissed immediately
        d._overlay.dismiss_recovery.assert_called_once()
        # Paste should be scheduled via NSTimer (not called immediately)
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called()
        # Pending insert should be stored
        assert d._recovery_pending_insert == ("transcribed text", [("public.utf8-plain-text", b"old")])

    def test_delayed_insert_reenters_recovery_when_no_text_field(self, main_module, monkeypatch):
        """doRecoveryInsert_ should re-enter recovery if target didn't refocus."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_pending_insert = ("transcribed text", None)

        with patch("spoke.__main__.has_focused_text_input", return_value=False), \
             patch("spoke.__main__.inject_text") as mock_inject, \
             patch.object(d, "_enter_recovery_mode") as mock_recovery:
            d.doRecoveryInsert_(None)

        mock_inject.assert_not_called()
        mock_recovery.assert_called_once_with("transcribed text")


class TestInsertClipboardRestore:
    """Verify Insert path restores original clipboard, not transcription."""

    def test_delayed_insert_restores_original_clipboard_before_paste(self, main_module, monkeypatch):
        """doRecoveryInsert_ should restore original clipboard before inject_text."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_pending_insert = (
            "transcribed text",
            [("public.utf8-plain-text", b"original")],
        )

        restore_calls = []
        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.restore_pasteboard") as mock_restore:
            mock_restore.side_effect = lambda saved: restore_calls.append(("restore", saved))
            mock_inject.side_effect = lambda text, on_restored=None: restore_calls.append(("inject", text))
            d.doRecoveryInsert_(None)

        # restore_pasteboard must be called BEFORE inject_text
        assert len(restore_calls) == 2
        assert restore_calls[0][0] == "restore"
        assert restore_calls[1][0] == "inject"
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


class TestSentinelClipboardPreservation:
    """Test _NOT_CAPTURED sentinel distinguishes 'never captured' from 'empty clipboard'."""

    def test_none_clipboard_preserved_as_none(self, main_module, monkeypatch):
        """When _pre_paste_clipboard is None (clipboard was empty), recovery
        should use None — not fall back to save_pasteboard()."""
        d = _make_delegate(main_module, monkeypatch)
        d._pre_paste_clipboard = None  # clipboard was empty at capture time

        with patch("spoke.__main__.save_pasteboard") as mock_save, \
             patch("spoke.__main__.set_pasteboard_only"):
            d._enter_recovery_mode("test text")

        # Should NOT have called save_pasteboard — we already have the saved state (None)
        mock_save.assert_not_called()
        assert d._recovery_saved_clipboard is None

    def test_not_captured_falls_back_to_save(self, main_module, monkeypatch):
        """When _pre_paste_clipboard is _NOT_CAPTURED, recovery should
        call save_pasteboard() as fallback."""
        d = _make_delegate(main_module, monkeypatch)
        d._pre_paste_clipboard = main_module._NOT_CAPTURED

        saved = [("public.utf8-plain-text", b"current clipboard")]
        with patch("spoke.__main__.save_pasteboard", return_value=saved) as mock_save, \
             patch("spoke.__main__.set_pasteboard_only"):
            d._enter_recovery_mode("test text")

        mock_save.assert_called_once()
        assert d._recovery_saved_clipboard == saved

    def test_concrete_clipboard_preserved(self, main_module, monkeypatch):
        """When _pre_paste_clipboard has concrete saved data, recovery uses it directly."""
        d = _make_delegate(main_module, monkeypatch)
        saved = [("public.utf8-plain-text", b"original")]
        d._pre_paste_clipboard = saved

        with patch("spoke.__main__.save_pasteboard") as mock_save, \
             patch("spoke.__main__.set_pasteboard_only"):
            d._enter_recovery_mode("test text")

        mock_save.assert_not_called()
        assert d._recovery_saved_clipboard is saved

    def test_sentinel_reset_after_recovery_entry(self, main_module, monkeypatch):
        """_pre_paste_clipboard should be reset to _NOT_CAPTURED after entering recovery."""
        d = _make_delegate(main_module, monkeypatch)
        d._pre_paste_clipboard = None

        with patch("spoke.__main__.set_pasteboard_only"):
            d._enter_recovery_mode("test text")

        assert d._pre_paste_clipboard is main_module._NOT_CAPTURED


class TestOCRVerifyRetry:
    """Test OCR verification retry path and stale-text guard."""

    def test_attempt_0_schedules_retry(self, main_module, monkeypatch):
        """First failed OCR check should schedule a retry, not enter recovery."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = "hello world"
        d._verify_paste_attempt = 0

        d.verifyPasteResult_({"found": False, "text": "hello world", "attempt": 0})

        # Should schedule retry timer, not enter recovery
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        # Timer delay should be 0.2s with correct selector
        call_args = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == 0.2
        assert call_args[0][2] == "verifyPaste:"
        # Attempt counter should be incremented
        assert d._verify_paste_attempt == 1
        # Should NOT have entered recovery
        d._overlay.show_recovery.assert_not_called()

    def test_stale_text_discarded(self, main_module, monkeypatch):
        """If verify text changed (new recording started), discard the result."""
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = "new text"  # changed since verification started

        d.verifyPasteResult_({"found": False, "text": "old text", "attempt": 1})

        # Should silently discard — no recovery, no retry
        d._overlay.show_recovery.assert_not_called()
        assert d._verify_paste_text == "new text"  # unchanged

    def test_stale_text_discarded_on_success(self, main_module, monkeypatch):
        """Even successful results for stale text should be discarded."""
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = "new text"

        d.verifyPasteResult_({"found": True, "text": "old text", "attempt": 0})

        # verify_paste_text should NOT be cleared — it belongs to a different paste
        assert d._verify_paste_text == "new text"

    def test_cleared_text_discarded(self, main_module, monkeypatch):
        """If verify text is None (cleared by new hold), discard the result."""
        d = _make_delegate(main_module, monkeypatch)
        d._verify_paste_text = None

        d.verifyPasteResult_({"found": False, "text": "some text", "attempt": 1})

        d._overlay.show_recovery.assert_not_called()


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


class TestTruncatePreview:
    """Test the _truncate_preview utility for clipboard preview display."""

    def test_none_returns_empty(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        assert mod._truncate_preview(None) == "(empty)"

    def test_empty_string_returns_empty(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        assert mod._truncate_preview("") == "(empty)"

    def test_short_text_unchanged(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        assert mod._truncate_preview("hello") == "hello"

    def test_newlines_replaced_with_spaces(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        assert mod._truncate_preview("line1\nline2\rline3") == "line1 line2 line3"

    def test_long_text_truncated_with_ellipsis(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        long_text = "a" * 100
        result = mod._truncate_preview(long_text)
        assert len(result) == 46  # 45 chars + ellipsis
        assert result.endswith("…")

    def test_exact_max_length_not_truncated(self, mock_pyobjc):
        import importlib, sys
        sys.modules.pop("spoke.overlay", None)
        mod = importlib.import_module("spoke.overlay")
        exact = "a" * 45
        assert mod._truncate_preview(exact) == exact
