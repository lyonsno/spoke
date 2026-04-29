"""Tests for the menubar status item."""

from pathlib import Path
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

    def test_refresh_menu_rebuilds_the_current_status_item_menu(self, menubar_module):
        """refresh_menu() should rebuild the dropdown in place."""
        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._status_item = MagicMock()
        icon._build_menu = MagicMock()

        icon.refresh_menu()

        icon._build_menu.assert_called_once_with()

    def test_refresh_menu_preserves_current_status_text(self, menubar_module):
        """refresh_menu() should not reset the live status label back to Idle."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "assistant": {
                    "selected": "qwen3p5-35B-A3B",
                    "models": [("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True)],
                }
            }
        )
        icon._status_item = status_item_menu_holder
        icon._idle_image = None
        icon._recording_image = None
        icon._status_text = "Requesting mic access…"

        icon.refresh_menu()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Requesting mic access…", None, "") for call in calls)

    def test_refresh_menu_preserves_status_set_via_set_status_text(self, menubar_module):
        """A live status update should survive a later menu rebuild."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = None
        icon._status_item = status_item_menu_holder
        icon._idle_image = None
        icon._recording_image = None
        icon._status_text = "Idle"
        icon._status_item_label = MagicMock()

        icon.set_status_text("Ready — hold spacebar")
        icon.refresh_menu()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Ready — hold spacebar", None, "") for call in calls)

    def test_build_menu_shows_source_discriminant(self, menubar_module):
        """setup() should show which checkout launched the running instance."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = None
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        expected_title = f"Source: {Path(menubar_module.__file__).resolve().parents[1].name}"
        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == (expected_title, None, "") for call in calls)

    def test_build_menu_shows_branch_discriminant(self, menubar_module, monkeypatch):
        """setup() should show the running git branch when available."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )
        monkeypatch.setattr(menubar_module, "_branch_menu_label", lambda repo_root=None: "Branch: smoke")

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = None
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Branch: smoke", None, "") for call in calls)

    def test_branch_menu_label_falls_back_to_detached_head(self, menubar_module, monkeypatch):
        """Detached checkouts should still expose a legible identifier."""
        outputs = iter(["", "abc1234"])

        monkeypatch.setattr(menubar_module, "_run_git", lambda repo_root, *args: next(outputs))

        assert menubar_module._branch_menu_label() == "Branch: detached@abc1234"

    def test_build_menu_shows_local_whisper_submenu(self, menubar_module):
        """Local-mode callback state should build a Local Whisper submenu."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "transcription": {
                    "selected": "mlx-community/whisper-large-v3-turbo",
                    "models": [
                        (
                            "mlx-community/whisper-large-v3-turbo",
                            "Whisper Large",
                            True,
                        )
                    ],
                },
                "preview": {
                    "selected": "mlx-community/whisper-medium.en-mlx-8bit",
                    "models": [
                        (
                            "mlx-community/whisper-medium.en-mlx-8bit",
                            "Whisper Medium",
                            True,
                        )
                    ],
                },
                "local_whisper": {
                    "title": "Local Whisper",
                    "items": [
                        ("decode_timeout", "Decode timeout guard (30s)", True, True),
                        (
                            "eager_eval",
                            "Stability mode (eager eval) [mlx-whisper update needed]",
                            False,
                            False,
                        ),
                    ],
                },
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Local Whisper", None, "") for call in calls)
        assert any(call.args == ("Decode timeout guard (30s)", "selectModel:", "") for call in calls)
        assert any(
            call.args
            == ("Stability mode (eager eval) [mlx-whisper update needed]", "selectModel:", "")
            for call in calls
        )

    def test_build_menu_shows_assistant_submenu(self, menubar_module):
        """Command-mode callback state should build an Assistant submenu."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "assistant": {
                    "selected": "qwen3p5-35B-A3B",
                    "models": [
                        ("qwen3p5-35B-A3B", "qwen3p5-35B-A3B", True),
                        ("qwen3-14b", "qwen3-14b", True),
                    ],
                }
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Assistant Model", None, "") for call in calls)
        assert any(call.args == ("qwen3p5-35B-A3B", "selectModel:", "") for call in calls)
        assert any(call.args == ("qwen3-14b", "selectModel:", "") for call in calls)

    def test_build_menu_shows_assistant_backend_submenu(self, menubar_module):
        """Command-mode callback state should build an Assistant Backend submenu."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "assistant_backend": {
                    "title": "Assistant Backend",
                    "items": [
                        ("local", "Local OMLX", False, True),
                        ("sidecar", "Sidecar OMLX", True, True),
                        ("configure", "Set Sidecar URL…", False, True),
                    ],
                }
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Assistant Backend", None, "") for call in calls)
        assert any(call.args == ("Local OMLX", "selectModel:", "") for call in calls)
        assert any(call.args == ("Sidecar OMLX", "selectModel:", "") for call in calls)
        assert any(call.args == ("Set Sidecar URL…", "selectModel:", "") for call in calls)

    def test_build_menu_shows_agent_shell_submenu(self, menubar_module):
        """Modal Agent Shell provider selection should be visible in the menubar."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "agent_shell": {
                    "title": "Agent Shell",
                    "items": [
                        ("off", "Off", False, True),
                        ("codex", "Codex", True, True),
                        ("claude-code", "Claude Code", False, False),
                    ],
                }
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        assert any(call.args == ("Agent Shell", None, "") for call in calls)
        assert any(call.args == ("Off", "selectModel:", "") for call in calls)
        assert any(call.args == ("Codex", "selectModel:", "") for call in calls)
        assert any(call.args == ("Claude Code", "selectModel:", "") for call in calls)

    def test_agent_shell_selection_refreshes_menu_checkmark(self, menubar_module, monkeypatch):
        """Selecting an Agent Shell provider should visibly update the menu in-place."""
        AppKit = __import__("AppKit")

        created_items = []

        class FakeMenuItem:
            def __init__(self, title, action, key):
                self.title = title
                self.action = action
                self.key = key
                self.target = None
                self.represented_object = None
                self.enabled = None
                self.state = 0
                self.submenu = None

            def setEnabled_(self, enabled):
                self.enabled = enabled

            def setTarget_(self, target):
                self.target = target

            def setRepresentedObject_(self, represented_object):
                self.represented_object = represented_object

            def setState_(self, state):
                self.state = state

            def setSubmenu_(self, submenu):
                self.submenu = submenu

        class FakeMenuItemAllocator:
            def initWithTitle_action_keyEquivalent_(self, title, action, key):
                item = FakeMenuItem(title, action, key)
                created_items.append(item)
                return item

        monkeypatch.setattr(AppKit.NSMenuItem, "alloc", MagicMock(return_value=FakeMenuItemAllocator()))
        monkeypatch.setattr(AppKit.NSMenuItem, "separatorItem", MagicMock(return_value=FakeMenuItem("—", None, "")))
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = MagicMock()

        selected = "off"

        def on_select(selection=None):
            nonlocal selected
            if selection is not None:
                role, value = selection
                assert role == "agent_shell"
                selected = value
                return None
            return {
                "agent_shell": {
                    "title": "Agent Shell",
                    "items": [
                        ("off", "Off", selected == "off", True),
                        ("codex", "Codex", selected == "codex", True),
                        ("claude-code", "Claude Code", selected == "claude-code", False),
                    ],
                }
            }

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(side_effect=on_select)
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()
        codex_item = next(item for item in created_items if item.title == "Codex")
        assert codex_item.state == 0

        sender = MagicMock()
        sender.representedObject.return_value = codex_item.represented_object
        icon.selectModel_(sender)

        rebuilt_codex_item = [
            item for item in created_items if item.title == "Codex"
        ][-1]
        rebuilt_off_item = [
            item for item in created_items if item.title == "Off"
        ][-1]
        assert rebuilt_codex_item.state == 1
        assert rebuilt_off_item.state == 0

    def test_build_menu_shows_launch_target_submenu(self, menubar_module):
        """Registry-backed launch targets should appear as flat items at the bottom of the menu."""
        AppKit = __import__("AppKit")

        status_item_menu_holder = MagicMock(name="status_item_holder")
        status_item_menu_holder.button.return_value = MagicMock()
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = (
            status_item_menu_holder
        )

        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            return_value={
                "launch_target": {
                    "title": "Launch Target",
                    "selected": "main",
                    "items": [
                        ("main", "Main", True),
                        ("smoke", "Smoke", True),
                    ],
                }
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        calls = AppKit.NSMenuItem.alloc.return_value.initWithTitle_action_keyEquivalent_.call_args_list
        # Launch targets are now flat items at the bottom, no submenu header
        assert any(call.args == ("Main", "selectModel:", "") for call in calls)
        assert any(call.args == ("Smoke", "selectModel:", "") for call in calls)

    def test_launch_target_items_have_dispatch_state_and_enabled_wiring(
        self, menubar_module, monkeypatch
    ):
        """Launch-target menu rows must carry enough state for a click to
        dispatch the selected target, visually mark the active target, and
        disable missing worktrees.
        """
        AppKit = __import__("AppKit")

        created_items = []

        class FakeMenuItem:
            def __init__(self, title, action, key):
                self.title = title
                self.action = action
                self.key = key
                self.target = None
                self.represented_object = None
                self.enabled = None
                self.state = 0
                self.submenu = None

            def setEnabled_(self, enabled):
                self.enabled = enabled

            def setTarget_(self, target):
                self.target = target

            def setRepresentedObject_(self, represented_object):
                self.represented_object = represented_object

            def setState_(self, state):
                self.state = state

            def setSubmenu_(self, submenu):
                self.submenu = submenu

        class FakeMenuItemAllocator:
            def initWithTitle_action_keyEquivalent_(self, title, action, key):
                item = FakeMenuItem(title, action, key)
                created_items.append(item)
                return item

        monkeypatch.setattr(AppKit.NSMenuItem, "alloc", MagicMock(return_value=FakeMenuItemAllocator()))
        monkeypatch.setattr(AppKit.NSMenuItem, "separatorItem", MagicMock(return_value=FakeMenuItem("—", None, "")))
        AppKit.NSStatusBar.systemStatusBar.return_value.statusItemWithLength_.return_value = MagicMock()

        selected = []
        icon = menubar_module.MenuBarIcon.__new__(menubar_module.MenuBarIcon)
        icon._on_quit = MagicMock()
        icon._on_select_model = MagicMock(
            side_effect=lambda selection=None: selected.append(selection)
            if selection is not None
            else {
                "launch_target": {
                    "title": "Launch Target",
                    "selected": "main",
                    "items": [
                        ("main", "Main", True),
                        ("smoke", "Smoke", False),
                    ],
                }
            }
        )
        icon._status_item = None
        icon._idle_image = None
        icon._recording_image = None

        icon.setup()

        main_item = next(item for item in created_items if item.title == "Main")
        smoke_item = next(item for item in created_items if item.title == "Smoke")
        assert main_item.action == "selectModel:"
        assert main_item.target is icon
        assert main_item.represented_object == ("launch_target", "main")
        assert main_item.enabled is True
        assert main_item.state == 1
        assert smoke_item.represented_object == ("launch_target", "smoke")
        assert smoke_item.enabled is False
        assert smoke_item.state == 0

        sender = MagicMock()
        sender.representedObject.return_value = smoke_item.represented_object
        icon.selectModel_(sender)

        assert selected == [("launch_target", "smoke")]
