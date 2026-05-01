"""Smoke test hook: intercept the command pathway and reposition the overlay.

This replaces the normal assistant command flow with the positioning pipeline.
For smoke testing only — in production this would be a route key mode.

Usage: monkey-patch SpokeApp._command_transcribe_worker with
       positioning_transcribe_worker.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


def positioning_transcribe_worker(app, wav_bytes: bytes, token: int) -> None:
    """Replacement for _command_transcribe_worker that repositions the overlay.

    Transcribes the audio, runs the positioning pipeline, and moves the overlay.
    """
    from AppKit import NSScreen
    from PIL import Image

    # Step 1: Transcribe
    text = app._transcribe_segments_and_tail(wav_bytes)
    if text is None:
        text = app._transcribe_full_buffer(wav_bytes)

    if not text or app._transcribe_token != token:
        logger.info("Positioning: no text or stale token")
        _finish_on_main(app, None)
        return

    logger.info("Positioning: utterance = %r", text)

    # Step 2: Capture screen using Spoke's existing capture infrastructure
    try:
        from ..scene_capture import _capture_screen, _save_image
        import tempfile
        import os

        result = _capture_screen()
        if result is None:
            raise RuntimeError("_capture_screen returned None")

        cg_image = result[0]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        _save_image(cg_image, tmp_path)
        screenshot = Image.open(tmp_path).copy()
        os.unlink(tmp_path)
    except Exception as e:
        logger.warning("Screen capture failed: %s — trying screencapture CLI", e)
        import subprocess
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        subprocess.run(["screencapture", "-x", tmp_path], check=True)
        screenshot = Image.open(tmp_path).copy()
        os.unlink(tmp_path)

    logger.info("Positioning: captured screen %s", screenshot.size)

    # Step 3: Run positioning pipeline
    from .reposition import reposition
    result = reposition(text, screenshot)

    if result is None:
        logger.info("Positioning: no viable position found")
        _finish_on_main(app, None)
        return

    logger.info(
        "Positioning: resolved %r → x=%.2f y=%.2f w=%.2f h=%.2f (%.1fs)",
        result.get("content_desc", "?"),
        result["x"], result["y"], result["width"], result["height"],
        result.get("elapsed_s", 0),
    )

    # Step 4: Move overlay on main thread
    _finish_on_main(app, result)


def _finish_on_main(app, result: dict | None) -> None:
    """Schedule overlay repositioning on the main thread."""
    from PyObjCTools import AppHelper

    def _do():
        app._transcribing = False
        if app._menubar is not None:
            app._menubar.set_recording(False)

        if result is None:
            if app._menubar is not None:
                app._menubar.set_status_text("No position found")
            if app._overlay is not None:
                app._overlay.hide()
            return

        screen = _get_main_screen_frame()
        if screen is None:
            return

        sw, sh = screen[1]

        # Convert fractional rect to screen pixels
        x = result["x"] * sw
        y = result["y"] * sh
        w = result["width"] * sw
        h = result["height"] * sh

        # macOS screen coordinates: origin is bottom-left
        # Our y is from top, so flip it
        mac_y = sh - y - h

        content_desc = result.get("content_desc", "positioned")
        elapsed = result.get("elapsed_s", 0)

        logger.info(
            "Positioning overlay to (%.0f, %.0f) %.0fx%.0f — %r in %.1fs",
            x, mac_y, w, h, content_desc, elapsed,
        )

        if app._overlay is not None:
            _move_overlay(app._overlay, x, mac_y, w, h)

        if app._menubar is not None:
            app._menubar.set_status_text(
                f"Positioned: avoiding {content_desc} ({elapsed:.1f}s)"
            )

    AppHelper.callAfter(_do)


def _get_main_screen_frame():
    """Get the main screen's frame as ((x, y), (w, h))."""
    from AppKit import NSScreen
    screen = NSScreen.mainScreen()
    if screen is None:
        return None
    frame = screen.frame()
    return frame


def _move_overlay(overlay, x: float, y: float, w: float, h: float) -> None:
    """Move and resize the overlay window."""
    from AppKit import NSMakeRect

    # Get the fringe width from the overlay
    f = getattr(overlay, "_fringe_width", 8.0)

    win_frame = NSMakeRect(x - f, y - f, w + 2 * f, h + 2 * f)
    overlay._window.setFrame_display_animate_(win_frame, True, True)

    # Update content view to match
    content_frame = NSMakeRect(f, f, w, h)
    overlay._content_view.setFrame_(content_frame)

    # Show if hidden
    if not overlay._window.isVisible():
        overlay._window.orderFront_(None)
        overlay._window.setAlphaValue_(1.0)


def install_positioning_hook(app) -> None:
    """Replace the command pathway with the positioning pipeline.

    Call this during app setup to make enter-held recordings
    trigger repositioning instead of assistant commands.
    """
    import types

    original = app._command_transcribe_worker

    def patched_worker(wav_bytes, token):
        positioning_transcribe_worker(app, wav_bytes, token)

    app._command_transcribe_worker = patched_worker
    app._original_command_transcribe_worker = original
    logger.info("Positioning smoke hook installed — enter-held → reposition")
