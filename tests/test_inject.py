"""Tests for text injection via pasteboard + synthetic Cmd+V.

Requires PyObjC mocks since inject.py imports AppKit/Quartz at module level.
"""

from unittest.mock import MagicMock, call
import json
import time

import pytest


class TestInjectText:
    """Test the inject_text() function."""

    def test_empty_text_is_noop(self, inject_module):
        """inject_text('') should do nothing."""
        AppKit = __import__("AppKit")
        inject_module.inject_text("")
        AppKit.NSPasteboard.generalPasteboard.assert_not_called()

    def test_sets_pasteboard_and_posts_cmd_v(self, inject_module):
        """Should set pasteboard to text and synthesize Cmd+V."""
        AppKit = __import__("AppKit")
        Quartz = __import__("Quartz")

        mock_pb = MagicMock()
        mock_pb.stringForType_.return_value = "original clipboard"
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.inject_text("hello world")

        # Pasteboard should be cleared and set
        mock_pb.clearContents.assert_called()
        mock_pb.setString_forType_.assert_called_with(
            "hello world", AppKit.NSPasteboardTypeString
        )

        # Should have posted keyboard events (Cmd+V down + up)
        assert Quartz.CGEventPost.call_count == 2

    def test_saves_original_pasteboard(self, inject_module):
        """Should read all pasteboard items before overwriting."""
        AppKit = __import__("AppKit")

        mock_pb = MagicMock()
        mock_pb.pasteboardItems.return_value = []
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.inject_text("new text")

        # Should have read the pasteboard items for save/restore
        mock_pb.pasteboardItems.assert_called_once()

    def test_reports_separate_paste_phases(self, inject_module, monkeypatch, caplog):
        pb = MagicMock()
        pb.pasteboardItems.return_value = []
        __import__("AppKit").NSPasteboard.generalPasteboard.return_value = pb
        clock = iter([10.0, 10.125, 10.150, 10.160])
        monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
        with caplog.at_level("INFO"):
            inject_module.inject_text("private dictation")
        records = [r for r in caplog.records if r.msg == "Paste timing %s"]
        assert len(records) == 1
        timing = json.loads(records[0].args[0])
        assert timing["outcome"] == "events_posted_destination_unverified"
        assert timing["phase"] == "post_cmd_v"
        assert timing["save_ms"] == pytest.approx(125)
        assert timing["write_ms"] == pytest.approx(25)
        assert timing["post_ms"] == pytest.approx(10)
        assert timing["total_ms"] == pytest.approx(160)
        assert timing["pid"] > 0
        assert "private dictation" not in records[0].getMessage()

    @pytest.mark.parametrize("failure_phase", ["save", "write", "post_cmd_v"])
    def test_reports_failure_without_claiming_paste(self, inject_module, monkeypatch, caplog, failure_phase):
        pb = MagicMock()
        pb.pasteboardItems.return_value = []
        __import__("AppKit").NSPasteboard.generalPasteboard.return_value = pb
        failure = RuntimeError("native failure")
        if failure_phase == "save":
            pb.pasteboardItems.side_effect = failure
        elif failure_phase == "write":
            pb.setString_forType_.side_effect = failure
        else:
            monkeypatch.setattr(inject_module, "_post_cmd_v", MagicMock(side_effect=failure))
        with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="native failure"):
            inject_module.inject_text("private dictation")
        timing = json.loads(next(r.args[0] for r in caplog.records if r.msg == "Paste timing %s"))
        assert timing["outcome"] == "failed"
        assert timing["phase"] == failure_phase
        assert timing["failed_phase_ms"] >= 0
        assert not any(r.msg == "Injected %d chars" for r in caplog.records)


class TestPasteboardRestore:
    """Test the pasteboard restore timing."""

    def test_default_restore_delay_is_1s(self, inject_module):
        """Default pasteboard restore delay should be 1 second."""
        AppKit = __import__("AppKit")
        Foundation = __import__("Foundation")

        mock_pb = MagicMock()
        mock_pb.stringForType_.return_value = "original"
        mock_pb.pasteboardItems.return_value = []
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.inject_text("transcribed text")

        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()
        call_args = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        delay = call_args[0][0]
        assert delay == 1.0

    def test_configurable_restore_delay(self, inject_module, monkeypatch):
        """SPOKE_RESTORE_DELAY_MS should override the default."""
        AppKit = __import__("AppKit")
        Foundation = __import__("Foundation")

        monkeypatch.setenv("SPOKE_RESTORE_DELAY_MS", "2000")

        mock_pb = MagicMock()
        mock_pb.stringForType_.return_value = "original"
        mock_pb.pasteboardItems.return_value = []
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        inject_module.inject_text("transcribed text")

        call_args = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args
        delay = call_args[0][0]
        assert delay == 2.0

    def test_on_restored_callback(self, inject_module):
        """on_restored callback should be passed through to the timer."""
        AppKit = __import__("AppKit")
        Foundation = __import__("Foundation")

        mock_pb = MagicMock()
        mock_pb.pasteboardItems.return_value = []
        AppKit.NSPasteboard.generalPasteboard.return_value = mock_pb

        callback = MagicMock()
        inject_module.inject_text("text", on_restored=callback)

        # Timer was scheduled — the callback is wired through the restorer
        Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once()

    @pytest.mark.parametrize("new_copy", [True, False])
    def test_restore_only_owns_its_unchanged_clipboard(self, inject_module, monkeypatch, new_copy):
        pb = MagicMock()
        pb.pasteboardItems.return_value = []
        pb.changeCount.return_value = 10
        __import__("AppKit").NSPasteboard.generalPasteboard.return_value = pb
        restore = MagicMock()
        monkeypatch.setattr(inject_module, "_restore_pasteboard", restore)
        callback = MagicMock()
        inject_module.inject_text("dictation", on_restored=callback)
        if new_copy:
            inject_module.set_pasteboard_only("chosen history text")
            pb.changeCount.return_value = 12
        target = __import__("Foundation").NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args.args[1]
        target.fire_(None)
        assert restore.call_count == (0 if new_copy else 1)
        callback.assert_called_once_with()

    def test_skip_callback_reports_release_without_claiming_restore(self, inject_module):
        pb = MagicMock()
        pb.pasteboardItems.return_value = []
        pb.changeCount.return_value = 10
        __import__("AppKit").NSPasteboard.generalPasteboard.return_value = pb
        restored, skipped = MagicMock(), MagicMock()
        inject_module.inject_text("dictation", on_restored=restored, on_restore_skipped=skipped)
        pb.changeCount.return_value = 12
        target = __import__("Foundation").NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.call_args.args[1]
        target.fire_(None)
        restored.assert_not_called()
        skipped.assert_called_once_with()


def test_manual_copy_rejected_write_is_not_success(inject_module):
    pb = MagicMock()
    pb.setString_forType_.return_value = False
    __import__("AppKit").NSPasteboard.generalPasteboard.return_value = pb
    with pytest.raises(RuntimeError, match="clipboard"):
        inject_module.set_pasteboard_only("history transcript")


class TestPostCmdV:
    """Test the synthetic keystroke generation."""

    def test_creates_keydown_and_keyup(self, inject_module):
        """Should create both keyDown and keyUp events for 'v'."""
        Quartz = __import__("Quartz")
        Quartz.CGEventCreateKeyboardEvent.reset_mock()
        Quartz.CGEventPost.reset_mock()

        inject_module._post_cmd_v()

        # Two events created: keyDown (True) and keyUp (False)
        calls = Quartz.CGEventCreateKeyboardEvent.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == inject_module._V_KEYCODE  # keycode
        assert calls[0][0][2] is True   # keyDown
        assert calls[1][0][2] is False  # keyUp

        # Both events posted
        assert Quartz.CGEventPost.call_count == 2

    def test_sets_command_flag_on_keydown_clears_on_keyup(self, inject_module):
        """keyDown should have Cmd flag, keyUp should clear flags to prevent
        modifier state from sticking (which causes Cmd+Space / Spotlight)."""
        Quartz = __import__("Quartz")
        Quartz.CGEventSetFlags.reset_mock()

        inject_module._post_cmd_v()

        # CGEventSetFlags called twice (once per event)
        assert Quartz.CGEventSetFlags.call_count == 2
        calls = Quartz.CGEventSetFlags.call_args_list
        # keyDown gets Command flag
        assert calls[0][0][1] == Quartz.kCGEventFlagMaskCommand
        # keyUp gets flags cleared to 0
        assert calls[1][0][1] == 0
