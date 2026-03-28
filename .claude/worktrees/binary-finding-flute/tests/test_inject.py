"""Tests for text injection via pasteboard + synthetic Cmd+V.

Requires PyObjC mocks since inject.py imports AppKit/Quartz at module level.
"""

from unittest.mock import MagicMock, call


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
        """DICTATE_RESTORE_DELAY_MS should override the default."""
        AppKit = __import__("AppKit")
        Foundation = __import__("Foundation")

        monkeypatch.setenv("DICTATE_RESTORE_DELAY_MS", "2000")

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
