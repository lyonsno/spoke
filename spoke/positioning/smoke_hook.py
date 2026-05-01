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

    import sys
    print(f"[POSITIONING] Step 1: transcribing (token={token})", file=sys.stderr, flush=True)
    logger.info("Positioning step 1: transcribing (token=%d)", token)

    # Step 1: Transcribe
    text = app._transcribe_segments_and_tail(wav_bytes)
    if text is None:
        text = app._transcribe_full_buffer(wav_bytes)

    current_token = getattr(app, '_transcription_token', None)
    print(f"[POSITIONING] Transcribed: {text[:80] if text else 'None'!r} (token={token}, current={current_token})", file=sys.stderr, flush=True)
    logger.info("Positioning: transcribed %r (token=%d, current=%s)", text[:80] if text else None, token, current_token)

    if not text or current_token != token:
        logger.info("Positioning: no text or stale token (token=%d, current=%s)", token, current_token)
        print(f"[POSITIONING] BAIL: no text or stale token", file=sys.stderr, flush=True)
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
        # Clear command_overlay_active so the next enter tap during recording
        # routes as a command gesture instead of triggering the cancel spring.
        # The positioning pathway leaves the overlay visible but it is not an
        # active generation — the cancel spring should not intercept.
        det = getattr(app, '_detector', None)
        if det is not None:
            det.command_overlay_active = False
        # Hide the user preview overlay so it doesn't sit there stale
        if getattr(app, '_overlay', None) is not None:
            app._overlay.hide()

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

        target = getattr(app, '_command_overlay', None) or getattr(app, '_overlay', None)
        if target is not None:
            _move_overlay(target, x, mac_y, w, h)

        # Flash the debug grid showing which cells the model marked YES/NO
        content_map = result.get("content_map")
        if content_map is not None:
            _flash_debug_grid(sw, sh, content_map, content_desc)

        if app._menubar is not None:
            app._menubar.set_status_text(
                f"Positioned: avoiding {content_desc} ({elapsed:.1f}s)"
            )

    AppHelper.callAfter(_do)


_debug_grid_window = None  # prevent GC from collecting the window


def _flash_debug_grid(
    sw: float, sh: float,
    content_map: dict[str, bool],
    content_desc: str,
    duration: float = 3.0,
) -> None:
    """Flash a transparent 4×4 grid overlay on screen showing YES/NO cells.

    YES cells (contain content to avoid) are red, NO cells (empty) are green.
    Each cell shows its label and YES/NO. The content description is shown
    at the top. The overlay auto-dismisses after `duration` seconds.
    """
    from AppKit import (
        NSBackingStoreBuffered,
        NSBorderlessWindowMask,
        NSColor,
        NSMakeRect,
        NSWindow,
    )
    from Quartz import CALayer, CATextLayer

    rows, cols = 4, 4
    row_labels = "ABCD"
    cell_w = sw / cols
    cell_h = sh / rows

    # Create a full-screen transparent window
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, sw, sh),
        NSBorderlessWindowMask,
        NSBackingStoreBuffered,
        False,
    )
    win.setLevel_(2147483647)  # above everything
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setIgnoresMouseEvents_(True)
    win.setHasShadow_(False)

    content = win.contentView()
    content.setWantsLayer_(True)
    root = content.layer()

    for r_idx, row_letter in enumerate(row_labels):
        for c_idx in range(cols):
            cell_key = f"{row_letter}{c_idx + 1}"
            has_content = content_map.get(cell_key, True)

            # Cell background
            cell = CALayer.alloc().init()
            # macOS y: bottom-left origin. Row A is the top of the screen.
            mac_y = sh - (r_idx + 1) * cell_h
            cell.setFrame_(((c_idx * cell_w, mac_y), (cell_w, cell_h)))
            if has_content:
                cell.setBackgroundColor_(
                    NSColor.colorWithRed_green_blue_alpha_(1.0, 0.2, 0.2, 0.25).CGColor()
                )
            else:
                cell.setBackgroundColor_(
                    NSColor.colorWithRed_green_blue_alpha_(0.2, 1.0, 0.2, 0.25).CGColor()
                )
            cell.setBorderWidth_(1.0)
            cell.setBorderColor_(
                NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.4).CGColor()
            )
            root.addSublayer_(cell)

            # Cell label
            label = CATextLayer.alloc().init()
            label_h = 40
            label.setFrame_(((c_idx * cell_w, mac_y + cell_h / 2 - label_h / 2),
                             (cell_w, label_h)))
            label.setString_(f"{cell_key}: {'YES' if has_content else 'NO'}")
            label.setFontSize_(16)
            label.setForegroundColor_(
                NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.9).CGColor()
            )
            label.setAlignmentMode_("center")
            label.setContentsScale_(2.0)
            root.addSublayer_(label)

    # Title bar showing what content was detected
    title = CATextLayer.alloc().init()
    title.setFrame_(((0, sh - 30), (sw, 30)))
    title.setString_(f"Avoiding: {content_desc}")
    title.setFontSize_(14)
    title.setForegroundColor_(
        NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 0.5, 0.9).CGColor()
    )
    title.setBackgroundColor_(
        NSColor.colorWithRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.5).CGColor()
    )
    title.setAlignmentMode_("center")
    title.setContentsScale_(2.0)
    root.addSublayer_(title)

    win.setAlphaValue_(1.0)
    win.orderFront_(None)

    # Hold a strong reference so GC doesn't collect the window before
    # the delayed dismiss fires.
    global _debug_grid_window
    _debug_grid_window = win

    # Auto-dismiss after duration via a background thread that
    # schedules the orderOut on the main thread.
    def _dismiss():
        from PyObjCTools import AppHelper
        def _do_dismiss():
            global _debug_grid_window
            if _debug_grid_window is not None:
                _debug_grid_window.orderOut_(None)
                _debug_grid_window = None
        AppHelper.callAfter(_do_dismiss)
    threading.Timer(duration, _dismiss).start()


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

    # Suppress _update_layout so text-driven auto-sizing doesn't
    # override the externally computed frame.
    overlay._positioning_override = True

    # Get the fringe width from the overlay
    f = getattr(overlay, "_fringe_width", 8.0)

    win_frame = NSMakeRect(x - f, y - f, w + 2 * f, h + 2 * f)
    overlay._window.setFrame_display_animate_(win_frame, True, True)

    # Update content view to match
    content_frame = NSMakeRect(f, f, w, h)
    overlay._content_view.setFrame_(content_frame)

    # Resize SDF fill and boost layers to match the new content area
    fill = getattr(overlay, "_fill_layer", None)
    if fill is not None:
        fill.setFrame_(((0, 0), (w, h)))
    boost = getattr(overlay, "_boost_layer", None)
    if boost is not None:
        boost.setFrame_(((0, 0), (w, h)))
    spring_tint = getattr(overlay, "_spring_tint_layer", None)
    if spring_tint is not None:
        spring_tint.setFrame_(((0, 0), (w, h)))

    # Resize backdrop layer to cover the new content area
    backdrop = getattr(overlay, "_backdrop_layer", None)
    if backdrop is not None:
        overscan = getattr(overlay, "_backdrop_capture_overscan_points", 0)
        backdrop.setFrame_(((-overscan, -overscan),
                            (w + 2 * overscan, h + 2 * overscan)))

    # Resize scroll view and text container to fit the new content area
    scroll = getattr(overlay, "_scroll_view", None)
    if scroll is not None:
        scroll.setFrame_(NSMakeRect(12, 8, w - 24, h - 16))
    text_view = getattr(overlay, "_text_view", None)
    if text_view is not None:
        container = text_view.textContainer()
        if container is not None:
            container.setContainerSize_((w - 24, 1.0e7))
        # Reset text view frame to match scroll view bounds
        if scroll is not None:
            text_view.setFrame_(scroll.bounds())

    # Regenerate the SDF fill image at the new dimensions so the
    # capsule shape matches the repositioned frame.
    ridge_masks = getattr(overlay, "_apply_ridge_masks", None)
    if ridge_masks is not None:
        try:
            ridge_masks(w, h)
        except Exception:
            logger.warning("Failed to regenerate ridge masks at new size", exc_info=True)

    # Update backdrop capture geometry for the new window position
    # so the warp/blur samples from the correct screen region.
    update_backdrop = getattr(overlay, "_update_backdrop_capture_geometry", None)
    if update_backdrop is not None:
        try:
            update_backdrop()
        except Exception:
            logger.warning("Failed to update backdrop capture geometry", exc_info=True)

    # Don't force _refresh_backdrop_snapshot here — the Metal pipeline
    # may not be ready for the new dimensions yet. The backdrop will
    # refresh on its own during the next frame callback.

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
