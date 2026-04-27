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
    delegate._mic_ready = True
    delegate._preview_session_token = 0
    delegate._preview_done = MagicMock()
    delegate._preview_done.set = MagicMock()
    delegate._local_inference_lock = MagicMock()
    # Tray state
    delegate._tray_stack = []
    delegate._tray_index = 0
    delegate._tray_active = False
    # Recovery state
    delegate._recovery_saved_clipboard = None
    delegate._recovery_text = None
    delegate._recovery_clipboard_state = "idle"
    delegate._recovery_hold_active = False
    delegate._recovery_pending_retry_insert = None
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()
    return delegate


class TestRecoveryFlowBranching:
    """_inject_result_text should branch on has_focused_text_input."""

    def test_normal_paste_when_text_field_focused(self, main_module, monkeypatch):
        """When a text field is focused, stage the normal inject after preview fade."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._inject_result_text("hello world", "Pasted!")

        mock_inject.assert_not_called()
        assert d._result_pending_inject == ("hello world", "Pasted!")
        call_args = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == (
            d._INSERT_OVERLAY_FADE_OUT_S + d._POST_OVERLAY_REFOCUS_DELAY_S
        )
        assert call_args[0][2] == "resultInjectDelayed:"
        d._overlay.hide.assert_called_once_with(
            fade_duration=d._INSERT_OVERLAY_FADE_OUT_S
        )
        d._overlay.order_out.assert_not_called()

    def test_always_attempts_paste(self, main_module, monkeypatch):
        """Normal paste should wait briefly for focus to settle after order_out."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.inject_text") as mock_inject, \
             patch("spoke.__main__.save_pasteboard", return_value=None):
            d._inject_result_text("hello world", "Pasted!")

        mock_inject.assert_not_called()
        assert d._result_pending_inject == ("hello world", "Pasted!")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        assert call_args[0][0] == (
            d._INSERT_OVERLAY_FADE_OUT_S + d._POST_OVERLAY_REFOCUS_DELAY_S
        )
        assert call_args[0][2] == "resultInjectDelayed:"

    def test_delayed_insert_saves_to_tray_and_skips_ocr_verification(
        self, main_module, monkeypatch
    ):
        """Delayed normal paste should preserve text in tray without OCR verification."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)
        d._result_pending_inject = ("hello world", "Pasted!")

        with patch("spoke.__main__.inject_text") as mock_inject:
            d.resultInjectDelayed_(None)

        mock_inject.assert_called_once()
        assert Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args is None
        assert not hasattr(d, "_verify_paste_text")
        assert [entry.text for entry in d._tray_stack] == ["hello world"]
        assert d._tray_active is False

    def test_delayed_insert_does_not_collect_pre_paste_evidence(
        self, main_module, monkeypatch
    ):
        """Delayed insert should not start AX/snapshot paste verification work."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.threading.Thread") as mock_thread, \
             patch("spoke.paste_verify.capture_verification_snapshot", return_value="snapshot") as mock_capture, \
             patch("spoke.__main__.inject_text"):
            d._inject_result_text("hello world", "Pasted!")
            d.resultInjectDelayed_(None)

        mock_thread.assert_not_called()
        mock_capture.assert_not_called()
        assert not hasattr(d, "_verify_paste_preexisting_match")
        assert not hasattr(d, "_verify_paste_preexisting_snapshot")

    def test_normal_insert_fades_overlay_before_instant_order_out(
        self, main_module, monkeypatch
    ):
        """Normal insert should not yank the preview overlay offscreen before the fade."""
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.inject_text"):
            d._inject_result_text("hello world", "Pasted!")

        d._overlay.hide.assert_called_once_with(
            fade_duration=d._INSERT_OVERLAY_FADE_OUT_S
        )
        d._overlay.order_out.assert_not_called()

    def test_does_not_start_snapshot_capture_after_preview_fade(self, main_module, monkeypatch):
        """Normal insert should paste after fade without starting OCR snapshot capture."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)

        with patch("spoke.__main__.threading.Thread") as mock_thread, \
             patch("spoke.paste_verify.capture_verification_snapshot", return_value="snapshot") as mock_capture, \
            patch("spoke.__main__.inject_text") as mock_inject:
            d._inject_result_text("hello world", "Pasted!")

            assert mock_capture.call_count == 0
            mock_inject.assert_not_called()

            d.resultInjectDelayed_(None)

        mock_inject.assert_called_once()
        mock_thread.assert_not_called()
        assert mock_capture.call_count == 0
        assert not hasattr(d, "_verify_paste_preexisting_match")
        assert not hasattr(d, "_verify_paste_preexisting_snapshot")


class TestRecoveryDismiss:
    """Recovery overlay dismiss behavior."""

    def test_spacebar_hold_during_tray_starts_recording(self, main_module, monkeypatch):
        """Plain spacebar hold during tray should dismiss tray and start recording."""
        d = _make_delegate(main_module, monkeypatch)
        d._tray_active = True
        d._tray_stack = ["some text"]
        d._recovery_text = "some text"
        d._detector._shift_at_press = False  # plain spacebar, no shift
        d._detector._shift_latched = False

        d._on_hold_start()

        # Should dismiss tray and start recording
        d._capture.start.assert_called_once()
        assert d._tray_active is False

    def test_shift_space_during_tray_navigates_down(self, main_module, monkeypatch):
        """Shift+space during tray should navigate down (toward older entries)."""
        d = _make_delegate(main_module, monkeypatch)
        d._tray_active = True
        d._tray_stack = ["old", "newest"]
        d._tray_index = 1  # at the top (most recent)
        d._recovery_text = "newest"

        d._on_hold_end(shift_held=True)

        # Should navigate down to older entry
        assert d._tray_index == 0

    def test_shift_space_at_bottom_stays(self, main_module, monkeypatch):
        """Shift+space at bottom of tray should stay (no wrap)."""
        d = _make_delegate(main_module, monkeypatch)
        d._tray_active = True
        d._tray_stack = ["only"]
        d._tray_index = 0
        d._recovery_text = "only"

        d._on_hold_end(shift_held=True)

        # Should stay at bottom — no wrap, no dismiss
        assert d._tray_active is True
        assert d._tray_index == 0

    def test_space_release_during_tray_inserts(self, main_module, monkeypatch):
        """Spacebar release during tray should insert text at cursor."""
        d = _make_delegate(main_module, monkeypatch)
        d._tray_active = True
        d._tray_stack = ["insert this"]
        d._tray_index = 0
        d._recovery_text = "insert this"
        d._recovery_hold_active = True  # simulates _on_hold_start tray intercept

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text"):
            d._on_hold_end(shift_held=False)

        assert d._tray_active is False

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

    def test_insert_button_delegates_to_retry(self, main_module, monkeypatch):
        """Insert button should use the same logic as spacebar retry."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"

        with patch.object(d, "_recovery_retry_insert") as mock_retry:
            d._on_recovery_insert()

        mock_retry.assert_called_once()

    def test_retry_bounces_when_no_text_field(self, main_module, monkeypatch):
        """Retry should bounce overlay when no text field is focused."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"

        with patch("spoke.__main__.has_focused_text_input", return_value=False):
            d._recovery_retry_insert()

        d._overlay.bounce.assert_called_once()
        # Should NOT have dismissed
        d._overlay.dismiss_recovery.assert_not_called()

    def test_retry_pastes_and_clears_recovery_without_ocr_when_text_field(self, main_module, monkeypatch):
        """Retry should paste and finish without OCR verification when a text field is focused."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._recovery_retry_insert()
            mock_inject.assert_not_called()
            Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
            d.doRecoveryRetryInsert_(None)

        mock_inject.assert_called_once()
        assert not hasattr(d, "_recovery_retry_pending")
        assert not hasattr(d, "_verify_paste_text")
        assert d._recovery_text is None
        assert Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args is None

    def test_retry_does_not_start_snapshot_capture_before_delayed_insert(self, main_module, monkeypatch):
        """Retry should not start OCR snapshot capture before the delayed paste fires."""
        Foundation = __import__("Foundation")
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.reset_mock()
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_text = "transcribed text"

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.threading.Thread") as mock_thread, \
             patch("spoke.paste_verify.capture_verification_snapshot", return_value="snapshot") as mock_capture, \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._recovery_retry_insert()

            mock_thread.assert_not_called()
            assert mock_capture.call_count == 0
            mock_inject.assert_not_called()
            assert d._recovery_pending_retry_insert == "transcribed text"

            d.doRecoveryRetryInsert_(None)

        mock_inject.assert_called_once_with("transcribed text")
        assert mock_capture.call_count == 0
        assert not hasattr(d, "_verify_paste_preexisting_match")
        assert not hasattr(d, "_verify_paste_preexisting_snapshot")

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


class TestInsertPaste:
    """Verify Insert path calls inject_text with the transcription."""

    def test_delayed_insert_calls_inject_text(self, main_module, monkeypatch):
        """doRecoveryInsert_ should call inject_text with the transcription text."""
        d = _make_delegate(main_module, monkeypatch)
        d._recovery_pending_insert = (
            "transcribed text",
            [("public.utf8-plain-text", b"original")],
        )

        with patch("spoke.__main__.has_focused_text_input", return_value=True), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d.doRecoveryInsert_(None)

        mock_inject.assert_called_once()
        assert mock_inject.call_args[0][0] == "transcribed text"


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


class TestInsertPathNoOCR:
    """Insertion should not launch paste-verification OCR work."""

    def test_insert_path_does_not_launch_late_snapshot_capture(
        self, main_module, monkeypatch
    ):
        """Insert should not launch any late snapshot thread after OCR removal."""
        d = _make_delegate(main_module, monkeypatch)
        capture_started = []
        capture_release = []

        class _ControlledThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                capture_started.append(True)

            def finish(self):
                if self._target is not None:
                    self._target()

        threads = []

        def _make_thread(*args, **kwargs):
            thread = _ControlledThread(**kwargs)
            threads.append(thread)
            return thread

        with patch("spoke.__main__.threading.Thread", side_effect=_make_thread), \
            patch("spoke.paste_verify.capture_verification_snapshot", side_effect=lambda: capture_release.append(True) or "snapshot"), \
             patch("spoke.__main__.inject_text") as mock_inject:
            d._inject_result_text("hello world", "Pasted!")

            assert capture_started == []
            d.resultInjectDelayed_(None)

            assert capture_started == []
            mock_inject.assert_called_once()
            assert not hasattr(d, "_verify_paste_preexisting_match")
            assert not hasattr(d, "_verify_paste_preexisting_snapshot")

            assert threads == []

        assert capture_release == []
        assert not hasattr(d, "_verify_paste_preexisting_snapshot")


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
