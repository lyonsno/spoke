"""NSStatusItem menubar icon for Spoke.

Shows a mic icon that changes state between idle and recording.
Phase 2 will add amplitude-driven animation.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from typing import Callable

from AppKit import (
    NSImage,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSVariableStatusItemLength,
)
import objc
from Foundation import NSObject

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_LABEL = f"Source: {_REPO_ROOT.name}"


def _run_git(repo_root: Path, *args: str) -> str:
    """Return trimmed git stdout for the running checkout, or an empty string."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _branch_menu_label(repo_root: Path = _REPO_ROOT) -> str:
    """Resolve a human-legible branch label for the running checkout."""
    branch = _run_git(repo_root, "branch", "--show-current")
    if branch:
        return f"Branch: {branch}"

    short_head = _run_git(repo_root, "rev-parse", "--short", "HEAD")
    if short_head:
        return f"Branch: detached@{short_head}"

    return "Branch: unknown"


class MenuBarIcon(NSObject):
    """Manages the menubar status item.

    Parameters
    ----------
    on_quit : callable
        Called when the user selects Quit from the menu.
    on_toggle_model : callable, optional
        Called when the user toggles the ASR model.
    """

    def initWithQuitCallback_(self, on_quit: Callable[[], None]):
        return self.initWithQuitCallback_selectModelCallback_(on_quit, None)

    def initWithQuitCallback_toggleModelCallback_(
        self,
        on_quit: Callable[[], None],
        on_toggle_model: Callable[[], None] | None = None,
    ):
        """Backwards compat — wraps old toggle into new select."""
        return self.initWithQuitCallback_selectModelCallback_(on_quit, on_toggle_model)

    def initWithQuitCallback_selectModelCallback_(
        self,
        on_quit: Callable[[], None],
        on_select_model: Callable | None = None,
    ):
        self = objc.super(MenuBarIcon, self).init()
        if self is None:
            return None
        self._on_quit = on_quit
        self._on_select_model = on_select_model
        self._status_item = None
        self._status_text = "Idle"
        self._branch_label = _branch_menu_label()
        self._idle_image = None
        self._recording_image = None
        return self

    def setup(self) -> None:
        """Create the status item and menu."""
        self._branch_label = getattr(self, "_branch_label", _branch_menu_label())
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )

        # SF Symbols — available on macOS 11+
        self._idle_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "mic", "Spoke — idle"
        )
        if self._idle_image is not None:
            self._idle_image.setTemplate_(True)
            
        self._recording_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "mic.fill", "Spoke — recording"
        )
        if self._recording_image is not None:
            self._recording_image.setTemplate_(True)

        button = self._status_item.button()
        if button is not None:
            button.setImage_(self._idle_image)
            button.setToolTip_(f"Spoke — {self._branch_label}")

        self._build_menu()

    def set_recording(self, recording: bool) -> None:
        """Update the icon to reflect recording state."""
        button = self._status_item.button()
        if button is None:
            return
        button.setImage_(self._recording_image if recording else self._idle_image)

    def set_vad_state(self, is_speech: bool, is_recording: bool) -> None:
        """Update menubar visual based on recording and VAD state."""
        import logging
        logger = logging.getLogger(__name__)
        
        if self._status_item is None or self._status_item.button() is None:
            return

        button = self._status_item.button()
        
        if not is_recording:
            button.setImage_(self._idle_image)
            return

        from AppKit import NSColor, NSImage, NSSize, NSRect, NSCompositeSourceAtop, NSImageNameStatusAvailable, NSImageNameStatusPartiallyAvailable
        
        # If contentTintColor makes it disappear (maybe macOS version issue), 
        # let's just use different built-in SF symbols or images that have those colors.
        # But wait, we can just tint the image manually.
        
        base_img = self._recording_image
        if base_img is None:
            return
            
        color = NSColor.greenColor() if is_speech else NSColor.redColor()
        
        # Create a new tinted image
        tinted = base_img.copy()
        tinted.setTemplate_(False)
        tinted.lockFocus()
        color.set()
        from AppKit import NSRectFillUsingOperation
        rect = ((0, 0), tinted.size())
        NSRectFillUsingOperation(rect, NSCompositeSourceAtop)
        tinted.unlockFocus()
        
        button.setImage_(tinted)

    def set_status_text(self, text: str) -> None:
        """Update the status label in the dropdown menu."""
        self._status_text = text
        if hasattr(self, "_status_item_label") and self._status_item_label is not None:
            self._status_item_label.setTitle_(text)

    def refresh_menu(self) -> None:
        """Rebuild the dropdown menu in place."""
        if self._status_item is None:
            return
        self._build_menu()

    # ── private ─────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu = NSMenu.new()

        self._status_item_label = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            getattr(self, "_status_text", "Idle"), None, ""
        )
        self._status_item_label.setEnabled_(False)
        menu.addItem_(self._status_item_label)

        source_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            _SOURCE_LABEL, None, ""
        )
        source_item.setEnabled_(False)
        menu.addItem_(source_item)

        branch_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            getattr(self, "_branch_label", "Branch: unknown"), None, ""
        )
        branch_item.setEnabled_(False)
        menu.addItem_(branch_item)

        added_menu_section = False
        if getattr(self, '_on_select_model', None) is not None:
            model_state = self._on_select_model(None)
            if isinstance(model_state, dict):
                launch_target = model_state.get("launch_target")
                if launch_target:
                    menu.addItem_(
                        self._build_choice_submenu_item(
                            launch_target["title"],
                            "launch_target",
                            launch_target["selected"],
                            launch_target["items"],
                        )
                    )
                    added_menu_section = True
                assistant = model_state.get("assistant")
                transcription = model_state.get("transcription")
                preview = model_state.get("preview")
                if assistant:
                    menu.addItem_(
                        self._build_choice_submenu_item(
                            "Assistant",
                            "assistant",
                            assistant["selected"],
                            assistant["models"],
                        )
                    )
                    added_menu_section = True
                if transcription:
                    menu.addItem_(
                        self._build_choice_submenu_item(
                            "Transcription",
                            "transcription",
                            transcription["selected"],
                            transcription["models"],
                        )
                    )
                    added_menu_section = True
                if preview:
                    menu.addItem_(
                        self._build_choice_submenu_item(
                            "Preview",
                            "preview",
                            preview["selected"],
                            preview["models"],
                        )
                    )
                    added_menu_section = True
                local_whisper = model_state.get("local_whisper")
                if local_whisper:
                    menu.addItem_(
                        self._build_toggle_submenu_item(
                            local_whisper["title"],
                            "local_whisper",
                            local_whisper["items"],
                        )
                    )
                    added_menu_section = True
            elif model_state:
                for model_id, label, enabled in model_state:
                    item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                        label, "selectModel:", ""
                    )
                    item.setTarget_(self)
                    item.setRepresentedObject_(model_id)
                    item.setEnabled_(enabled)
                    menu.addItem_(item)
                    added_menu_section = True

        if added_menu_section:
            menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Spoke", "quitApp:", "q"
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        self._status_item.setMenu_(menu)

    def _build_choice_submenu_item(
        self,
        title: str,
        role: str,
        selected_value: str,
        items,
    ):
        submenu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            title, None, ""
        )
        submenu = NSMenu.new()
        for item_id, label, enabled in items:
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                label, "selectModel:", ""
            )
            item.setTarget_(self)
            item.setRepresentedObject_((role, item_id))
            item.setEnabled_(enabled)
            if item_id == selected_value:
                item.setState_(1)
            submenu.addItem_(item)
        submenu_item.setSubmenu_(submenu)
        return submenu_item

    def _build_toggle_submenu_item(self, title: str, role: str, items):
        submenu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            title, None, ""
        )
        submenu = NSMenu.new()
        for item_spec in items:
            if len(item_spec) == 3:
                key, label, selected = item_spec
                enabled = True
            else:
                key, label, selected, enabled = item_spec
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                label, "selectModel:", ""
            )
            item.setTarget_(self)
            item.setRepresentedObject_((role, key))
            item.setState_(1 if selected else 0)
            item.setEnabled_(enabled)
            submenu.addItem_(item)
        submenu_item.setSubmenu_(submenu)
        return submenu_item

    def selectModel_(self, sender) -> None:
        selection = sender.representedObject()
        if selection and getattr(self, '_on_select_model', None) is not None:
            self._on_select_model(selection)

    def cleanup(self) -> None:
        """Remove the status item from the menu bar."""
        if self._status_item is not None:
            NSStatusBar.systemStatusBar().removeStatusItem_(self._status_item)
            self._status_item = None

    def quitApp_(self, sender) -> None:
        self._on_quit()
