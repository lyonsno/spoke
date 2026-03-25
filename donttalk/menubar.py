"""NSStatusItem menubar icon for DontTalk.

Shows a mic icon that changes state between idle and recording.
Phase 2 will add amplitude-driven animation.
"""

from __future__ import annotations

import logging
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


class MenuBarIcon(NSObject):
    """Manages the menubar status item.

    Parameters
    ----------
    on_quit : callable
        Called when the user selects Quit from the menu.
    """

    def initWithQuitCallback_(self, on_quit: Callable[[], None]):
        self = objc.super(MenuBarIcon, self).init()
        if self is None:
            return None
        self._on_quit = on_quit
        self._status_item = None
        self._idle_image = None
        self._recording_image = None
        return self

    def setup(self) -> None:
        """Create the status item and menu."""
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )

        # SF Symbols — available on macOS 11+
        self._idle_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "mic", "DontTalk — idle"
        )
        self._recording_image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            "mic.fill", "DontTalk — recording"
        )

        button = self._status_item.button()
        if button is not None:
            button.setImage_(self._idle_image)

        self._build_menu()

    def set_recording(self, recording: bool) -> None:
        """Update the icon to reflect recording state."""
        button = self._status_item.button()
        if button is None:
            return
        button.setImage_(self._recording_image if recording else self._idle_image)

    def set_status_text(self, text: str) -> None:
        """Update the status label in the dropdown menu."""
        if hasattr(self, "_status_item_label") and self._status_item_label is not None:
            self._status_item_label.setTitle_(text)

    # ── private ─────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu = NSMenu.new()

        self._status_item_label = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Idle", None, ""
        )
        self._status_item_label.setEnabled_(False)
        menu.addItem_(self._status_item_label)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit DontTalk", "quitApp:", "q"
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)

        self._status_item.setMenu_(menu)

    def quitApp_(self, sender) -> None:
        self._on_quit()
