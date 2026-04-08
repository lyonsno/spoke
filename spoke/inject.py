"""Text injection via pasteboard + synthetic Cmd+V.

Saves the current pasteboard contents (all types), sets the transcribed text,
sends a synthetic Cmd+V keystroke, then restores the original pasteboard
after a configurable delay.
"""

from __future__ import annotations

import logging
import os

import objc as _objc
from AppKit import NSPasteboard, NSPasteboardTypeString
from Foundation import NSObject, NSTimer
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

logger = logging.getLogger(__name__)

_V_KEYCODE = 9

_DEFAULT_RESTORE_DELAY_S = 1.0


def _get_restore_delay() -> float:
    """Read restore delay from SPOKE_RESTORE_DELAY_MS, default 1000."""
    raw = os.environ.get("SPOKE_RESTORE_DELAY_MS")
    if raw is None:
        return _DEFAULT_RESTORE_DELAY_S
    try:
        ms = int(raw)
    except ValueError:
        logger.warning("SPOKE_RESTORE_DELAY_MS=%r is not an integer, using default", raw)
        return _DEFAULT_RESTORE_DELAY_S
    if ms < 0:
        logger.warning("SPOKE_RESTORE_DELAY_MS=%d is negative, using default", ms)
        return _DEFAULT_RESTORE_DELAY_S
    return ms / 1000.0


def save_pasteboard() -> list[tuple[str, bytes]] | None:
    """Save the current general pasteboard contents. Public API for recovery mode."""
    pb = NSPasteboard.generalPasteboard()
    return _save_pasteboard(pb)


def restore_pasteboard(saved: list[tuple[str, bytes]] | None) -> None:
    """Restore previously saved pasteboard contents. Public API for recovery mode."""
    pb = NSPasteboard.generalPasteboard()
    _restore_pasteboard(pb, saved)


def set_pasteboard_only(text: str) -> None:
    """Set pasteboard to *text* without pasting or scheduling a restore.

    Used for recovery mode: the text stays on the clipboard for the user
    to manually Cmd+V wherever they want.
    """
    if not text:
        return
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    logger.info("Pasteboard set (no paste) — %d chars for manual recovery", len(text))


def _save_pasteboard(pb: NSPasteboard) -> list[tuple[str, bytes]] | None:
    """Save all items/types from the pasteboard. Returns None if empty."""
    items = pb.pasteboardItems()
    if not items or len(items) == 0:
        return None

    saved: list[tuple[str, bytes]] = []
    for item in items:
        for ptype in item.types():
            data = item.dataForType_(ptype)
            if data is not None:
                saved.append((ptype, bytes(data)))
    return saved if saved else None


def _restore_pasteboard(pb: NSPasteboard, saved: list[tuple[str, bytes]] | None) -> None:
    """Restore previously saved pasteboard contents."""
    pb.clearContents()
    if not saved:
        return

    from AppKit import NSPasteboardItem

    item = NSPasteboardItem.alloc().init()
    for ptype, data in saved:
        from Foundation import NSData
        ns_data = NSData.dataWithBytes_length_(data, len(data))
        item.setData_forType_(ns_data, ptype)
    pb.writeObjects_([item])


def inject_text(text: str, on_restored: object = None) -> None:
    """Paste *text* at the current cursor position.

    1. Save current pasteboard (all types)
    2. Set pasteboard to *text*
    3. Synthesize Cmd+V
    4. Schedule pasteboard restore after a configurable delay

    Parameters
    ----------
    on_restored : callable, optional
        Called (on main thread) after the pasteboard has been restored.
    """
    if not text:
        return

    pb = NSPasteboard.generalPasteboard()

    # Save all pasteboard contents (not just strings)
    saved = _save_pasteboard(pb)

    # Set our text
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)

    # Synthesize Cmd+V
    _post_cmd_v()

    logger.info("Injected %d chars", len(text))

    restore_delay = _get_restore_delay()

    # Restore pasteboard after a delay (must run on main thread via NSTimer)
    def _do_restore(timer: NSTimer) -> None:
        _restore_pasteboard(pb, saved)
        logger.debug("Pasteboard restored")
        if on_restored is not None:
            on_restored()

    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        restore_delay,
        _PasteboardRestorer.alloc().initWithCallback_(_do_restore),
        "fire:",
        None,
        False,
    )


def inject_text_raw(text: str) -> None:
    """Paste *text* without saving/restoring the pasteboard.

    For use in rapid-fire contexts (e.g. hands-free dictation) where the
    caller manages clipboard lifecycle externally.
    """
    if not text:
        return
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    _post_cmd_v()
    logger.info("Injected (raw) %d chars", len(text))


def _post_cmd_v() -> None:
    """Post a synthetic Cmd+V keystroke."""
    src = None  # default event source

    down = CGEventCreateKeyboardEvent(src, _V_KEYCODE, True)
    CGEventSetFlags(down, kCGEventFlagMaskCommand)

    up = CGEventCreateKeyboardEvent(src, _V_KEYCODE, False)
    # Clear modifier flags on keyUp so Command doesn't "stick" in the
    # system modifier state — otherwise the next real keypress (e.g.
    # spacebar) is interpreted as Cmd+Space, opening Spotlight.
    CGEventSetFlags(up, 0)

    CGEventPost(kCGHIDEventTap, down)
    CGEventPost(kCGHIDEventTap, up)


# ── tiny helper to bridge NSTimer → Python callable ──────────


class _PasteboardRestorer(NSObject):
    """NSObject wrapper so NSTimer can call back into Python."""

    def initWithCallback_(self, callback):
        self = _objc.super(_PasteboardRestorer, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def fire_(self, timer):
        self._callback(timer)
