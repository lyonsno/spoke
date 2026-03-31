"""Source ref resolution for screen context.

Maps stable source refs (scene blocks, AX hints, clipboard, selection,
last response, literal text) to exact text for downstream consumption
(e.g., TTS via read_aloud).

See docs/screen-context-v1.md for the design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from spoke.scene_capture import SceneCaptureCache

logger = logging.getLogger(__name__)

_VALID_KINDS = frozenset({
    "last_response",
    "selection",
    "clipboard",
    "scene_block",
    "ax_hint",
    "literal",
})


class RefResolutionError(Exception):
    """Raised when a source ref cannot be resolved to text."""


@dataclass
class SourceRef:
    """A reference to a text source that can be resolved to exact text."""

    kind: Literal[
        "last_response",
        "selection",
        "clipboard",
        "scene_block",
        "ax_hint",
        "literal",
    ]
    value: str


def parse_ref(ref_string: str) -> SourceRef:
    """Parse a source ref string to a SourceRef.

    Accepts both prefixed and bare formats:
      - Prefixed: 'scene_block:scene-abc:block-1' (explicit kind)
      - Bare:     'scene-abc:block-1' (kind inferred from shape)

    Bare scene refs (matching 'scene-*:block-*') are inferred as
    scene_block. Bare AX refs (matching 'scene-*:focus' or similar
    hint patterns) are inferred as ax_hint. This is necessary because
    models consistently pass the ref values from capture_context
    directly without prepending the kind prefix.
    """
    colon = ref_string.find(":")
    if colon < 0:
        raise ValueError(f"Invalid source ref (no colon): {ref_string!r}")

    kind = ref_string[:colon]
    value = ref_string[colon + 1:]

    if kind in _VALID_KINDS:
        return SourceRef(kind=kind, value=value)

    # Kind not recognized — try to infer from the full string shape.
    # Scene block refs look like "scene-abc:block-N"
    if ref_string.startswith("scene-") and ":block-" in ref_string:
        return SourceRef(kind="scene_block", value=ref_string)

    # AX hint refs look like "scene-abc:focus" (or other hint suffixes)
    if ref_string.startswith("scene-") and ":" in ref_string:
        return SourceRef(kind="ax_hint", value=ref_string)

    raise ValueError(f"Unknown source ref kind: {kind!r}")


def resolve(
    ref: SourceRef,
    *,
    scene_cache: SceneCaptureCache | None = None,
    last_response: str | None = None,
) -> str:
    """Resolve a SourceRef to exact text.

    Raises RefResolutionError if the ref cannot be resolved or resolves
    to empty text.
    """
    text: str | None = None

    if ref.kind == "literal":
        text = ref.value

    elif ref.kind == "scene_block":
        if scene_cache is None:
            raise RefResolutionError("No scene cache available for scene_block ref")
        text = scene_cache.resolve_block(ref.value)
        if text is None:
            raise RefResolutionError(f"Scene block ref not found: {ref.value}")

    elif ref.kind == "ax_hint":
        if scene_cache is None:
            raise RefResolutionError("No scene cache available for ax_hint ref")
        text = scene_cache.resolve_ax_hint(ref.value)
        if text is None:
            raise RefResolutionError(f"AX hint ref not found: {ref.value}")

    elif ref.kind == "clipboard":
        text = _get_clipboard_text()

    elif ref.kind == "selection":
        text = _get_selection_text()

    elif ref.kind == "last_response":
        if not last_response:
            raise RefResolutionError("No last response available")
        text = last_response

    if not text:
        raise RefResolutionError(
            f"Ref resolved to empty text: {ref.kind}:{ref.value}"
        )

    return text


def _get_clipboard_text() -> str:
    """Get the current clipboard text. Returns empty string on failure."""
    try:
        from AppKit import NSPasteboard

        pb = NSPasteboard.generalPasteboard()
        text = pb.stringForType_("public.utf8-plain-text")
        return text or ""
    except Exception:
        logger.debug("Failed to read clipboard", exc_info=True)
        return ""


def _get_selection_text() -> str:
    """Get the currently selected text in the frontmost app.

    Uses Cmd+C to copy the selection to the clipboard, reads it, then
    restores the original clipboard content.

    Known limitations:
    - Only saves/restores plain text — rich content, images, and file
      references on the clipboard are lost during the round-trip.
    - Uses a fixed 100ms delay which may be too short for slow apps.
    - If selected text equals existing clipboard content, returns empty.
    The primary screen-context path (capture_context → scene_block ref)
    avoids these issues entirely. This is a secondary convenience path.

    Returns empty string on failure.
    """
    try:
        from AppKit import NSPasteboard

        pb = NSPasteboard.generalPasteboard()

        # Save current clipboard
        original = pb.stringForType_("public.utf8-plain-text")

        # Simulate Cmd+C
        _simulate_cmd_c()

        import time
        time.sleep(0.1)  # Give the app time to process

        # Read the new clipboard content
        text = pb.stringForType_("public.utf8-plain-text") or ""

        # Restore original clipboard
        if original is not None:
            pb.clearContents()
            pb.setString_forType_(original, "public.utf8-plain-text")

        # If clipboard didn't change, selection was probably empty
        if text == (original or ""):
            return ""

        return text
    except Exception:
        logger.debug("Failed to get selection text", exc_info=True)
        return ""


def _simulate_cmd_c():
    """Simulate a Cmd+C keypress via CGEvent."""
    try:
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            CGEventSetFlags,
            kCGEventFlagMaskCommand,
            kCGHIDEventTap,
        )

        # 'c' key = keycode 8
        event_down = CGEventCreateKeyboardEvent(None, 8, True)
        CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        event_up = CGEventCreateKeyboardEvent(None, 8, False)
        CGEventSetFlags(event_up, kCGEventFlagMaskCommand)

        CGEventPost(kCGHIDEventTap, event_down)
        CGEventPost(kCGHIDEventTap, event_up)
    except Exception:
        logger.debug("Failed to simulate Cmd+C", exc_info=True)
