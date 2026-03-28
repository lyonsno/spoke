"""Tests for the menubar status item."""

from unittest.mock import MagicMock


class TestMenuBarIcon:
    """Test the NSStatusItem wrapper."""

    def test_setup_creates_status_item(self, menubar_module):
        """setup() should create an NSStatusItem with the idle icon."""
        AppKit = __import__("AppKit")

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        AppKit.NSStatusBar.systemStatusBar.assert_called_once()
        AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_.assert_called()

    def test_set_recording_changes_icon(self, menubar_module):
        """set_recording(True) should switch to the recording image."""
        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._idle_image = MagicMock(name="idle")
        icon._recording_image = MagicMock(name="recording")
        mock_button = MagicMock()
        icon._status_item = MagicMock()
        icon._status_item.button.return_value = mock_button

        icon.set_recording(True)
        mock_button.setImage_.assert_called_with(icon._recording_image)

        icon.set_recording(False)
        mock_button.setImage_.assert_called_with(icon._idle_image)

    def test_quit_callback(self, menubar_module):
        """Selecting Quit from menu should call the quit callback."""
        quit_fn = MagicMock()
        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = quit_fn

        icon.quitApp_(None)
        quit_fn.assert_called_once()

    def test_set_status_text(self, menubar_module):
        """set_status_text should update the menu label."""
        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        mock_label = MagicMock()
        icon._status_item_label = mock_label

        icon.set_status_text("Recording…")
        mock_label.setTitle_.assert_called_with("Recording…")
