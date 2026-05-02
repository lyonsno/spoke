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

from ..optical_field import (
    OpticalFieldBounds,
    OpticalFieldRequest,
)

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
    # SPOKE_POSITIONING_MODE: "grid" (default) or "twostep"
    import os as _os
    mode = _os.environ.get("SPOKE_POSITIONING_MODE", "grid")

    if mode == "bbox":
        from .reposition import reposition_bbox, update_bearing
        current_overlay = None
        prev_req = getattr(app, '_positioning_field_request', None)
        if prev_req is not None:
            screen = _get_main_screen_frame_for_worker()
            if screen is not None:
                sw, sh = screen
                b = prev_req.bounds
                current_overlay = {
                    "x": b.x / sw, "y": (sh - b.y - b.height) / sh,
                    "width": b.width / sw, "height": b.height / sh,
                }
        screen_dims = _get_main_screen_frame_for_worker()
        scr_w = int(screen_dims[0]) if screen_dims else 1920
        scr_h = int(screen_dims[1]) if screen_dims else 1080

        # Get previous bearing if available
        bearing = getattr(app, '_positioning_bearing', None)

        def _on_step_bbox(debug_lines):
            partial = {"utterance": text, "elapsed_s": 0, "x": 0, "y": 0, "width": 0, "height": 0,
                        "content_desc": "running..."}
            from PyObjCTools import AppHelper
            AppHelper.callAfter(lambda: _show_diagnostic_overlay(partial, debug_lines))

        try:
            result = reposition_bbox(text, screenshot, current_overlay,
                                     screen_w=scr_w, screen_h=scr_h,
                                     on_step=_on_step_bbox,
                                     bearing=bearing)
        except Exception as e:
            logger.exception("BBox positioning pipeline failed")
            import traceback
            _flash_error_on_main(f"Pipeline error: {e}\n{traceback.format_exc()[-500:]}")
            _finish_on_main(app, None)
            return
        pipeline_fn = reposition_bbox

        # Fire background bearing update — runs after overlay moves
        screenshot_b64 = result.pop("_screenshot_b64", None)
        if screenshot_b64:
            def _update_bearing_bg():
                try:
                    new_bearing = update_bearing(
                        screenshot_b64, text, result, bearing, scr_w, scr_h,
                    )
                    app._positioning_bearing = new_bearing
                    logger.info("Bearing updated: %s", new_bearing[:200])
                except Exception:
                    logger.warning("Background bearing update failed", exc_info=True)
            threading.Thread(target=_update_bearing_bg, daemon=True).start()
    elif mode == "twostep":
        from .reposition import reposition_twostep
        # Pass current overlay position if we have one from a previous run
        current_overlay = None
        prev_req = getattr(app, '_positioning_field_request', None)
        if prev_req is not None:
            screen = _get_main_screen_frame_for_worker()
            if screen is not None:
                sw, sh = screen
                b = prev_req.bounds
                current_overlay = {
                    "x": b.x / sw, "y": (sh - b.y - b.height) / sh,
                    "width": b.width / sw, "height": b.height / sh,
                }
        def _on_step(debug_lines):
            """Update diagnostic overlay incrementally as each step completes."""
            # Build a partial result dict for the diagnostic overlay
            partial = {"utterance": text, "elapsed_s": 0, "x": 0, "y": 0, "width": 0, "height": 0,
                        "content_desc": "running..."}
            from PyObjCTools import AppHelper
            AppHelper.callAfter(lambda: _show_diagnostic_overlay(partial, debug_lines))

        try:
            result = reposition_twostep(text, screenshot, current_overlay, on_step=_on_step)
        except Exception as e:
            logger.exception("Two-step positioning pipeline failed")
            import traceback
            _flash_error_on_main(f"Pipeline error: {e}\n{traceback.format_exc()[-500:]}")
            _finish_on_main(app, None)
            return
        pipeline_fn = reposition_twostep
    else:
        from .reposition import reposition
        try:
            result = reposition(text, screenshot)
        except Exception as e:
            logger.exception("Positioning pipeline failed")
            import traceback
            _flash_error_on_main(f"Pipeline error: {e}\n{traceback.format_exc()[-500:]}")
            _finish_on_main(app, None)
            return
        pipeline_fn = reposition

    if result is None:
        logger.info("Positioning: no viable position found")
        raw_debug = getattr(pipeline_fn, '_last_debug', None)
        debug_lines = [f"Utterance: {text!r}", "Result: None (no viable rectangle)"]
        if raw_debug:
            debug_lines.extend(raw_debug)
        _flash_error_on_main("\n".join(debug_lines))
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


_POSITIONING_CALLER_ID = "semantic_positioning"


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
            "Positioning overlay via optical field → (%.0f, %.0f) %.0fx%.0f — %r in %.1fs",
            x, mac_y, w, h, content_desc, elapsed,
        )

        # Emit an OpticalFieldRequest with the computed bounds.
        # The compositor owns the rendering — we don't touch NSWindow
        # frames, SDF layers, or backdrop geometry directly.
        request = OpticalFieldRequest(
            caller_id=_POSITIONING_CALLER_ID,
            bounds=OpticalFieldBounds(x=x, y=mac_y, width=w, height=h),
            role="assistant",
            state="materialize",
            visible=True,
        )

        # Push through the compositor's optical field backend if available,
        # otherwise fall back to direct overlay move for smoke testing.
        compositor = getattr(app, '_fullscreen_compositor', None)
        backend = getattr(compositor, '_optical_field_backend', None) if compositor else None
        if backend is not None:
            from ..optical_field import compile_placeholder_shell_config
            backend.upsert(request)
            shell_config = compile_placeholder_shell_config(request)
            # Find the active client to push the config to
            command_overlay = getattr(app, '_command_overlay', None)
            client_id = getattr(command_overlay, '_compositor_client_id', None) if command_overlay else None
            if client_id and hasattr(compositor, 'update_client_config'):
                compositor.update_client_config(client_id, shell_config)
                logger.info("Pushed optical field config to compositor client %s", client_id)
            else:
                # No compositor client — push as a shell config update
                compositor.update_shell_config(shell_config)
                logger.info("Pushed optical field config as shell config")
        else:
            # Fallback: direct overlay move (legacy smoke path)
            logger.info("No optical field backend — falling back to direct overlay move")
            command_overlay = getattr(app, '_command_overlay', None)
            preview_overlay = getattr(app, '_overlay', None)
            target = command_overlay or preview_overlay
            if target is not None:
                _move_overlay(target, x, mac_y, w, h)

        # Store the last request so position persists across show/hide
        app._positioning_field_request = request

        utterance = result.get("utterance", "")

        # Show a standalone smoke rectangle at the computed position
        # with user prompt + model response text to gauge text capacity
        smoke_text = f"{utterance}\n\n{content_desc}"
        _show_smoke_rect(x, mac_y, w, h, smoke_text)

        # Flash the debug grid showing which cells the model marked YES/NO
        content_map = result.get("content_map")
        # target_mode inverts colors: YES=green (go here) vs YES=red (content to avoid)
        positive_mode = content_desc.startswith("targeting:") or content_desc.startswith("finding:")
        if content_map is not None:
            _flash_debug_grid(sw, sh, content_map, content_desc,
                              utterance=utterance, target_mode=positive_mode,
                              elapsed_s=elapsed)

        # Persistent diagnostic overlay in upper right — survives grid dismiss
        from .reposition import reposition_bbox as _bbox_fn
        from .reposition import reposition_twostep as _twostep_fn
        from .reposition import reposition as _grid_fn
        debug_steps = (getattr(_bbox_fn, '_last_debug', None)
                       or getattr(_twostep_fn, '_last_debug', None)
                       or getattr(_grid_fn, '_last_debug', None))
        _show_diagnostic_overlay(result, debug_steps)

        if app._menubar is not None:
            app._menubar.set_status_text(
                f"Positioned: {content_desc} ({elapsed:.1f}s)"
            )

    AppHelper.callAfter(_do)


_debug_grid_window = None  # prevent GC from collecting the window
_debug_error_window = None  # prevent GC from collecting error window
_debug_diag_window = None  # persistent diagnostic overlay
_smoke_rect_window = None  # standalone positioned rectangle


def _show_smoke_rect(x: float, y: float, w: float, h: float, text: str) -> None:
    """Show a standalone semi-transparent rectangle at the computed position.

    Persists until the next positioning run replaces it. Filled with text
    (user prompt + model response) clipped to fit.
    """
    from AppKit import (
        NSBackingStoreBuffered,
        NSBorderlessWindowMask,
        NSColor,
        NSFont,
        NSMakeRect,
        NSWindow,
    )
    from Quartz import CATextLayer

    global _smoke_rect_window
    if _smoke_rect_window is not None:
        _smoke_rect_window.orderOut_(None)

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, w, h),
        NSBorderlessWindowMask,
        NSBackingStoreBuffered,
        False,
    )
    win.setLevel_(2147483647 - 1)  # just below debug overlays
    win.setOpaque_(False)
    win.setBackgroundColor_(
        NSColor.colorWithRed_green_blue_alpha_(0.05, 0.05, 0.12, 0.82)
    )
    win.setIgnoresMouseEvents_(True)
    win.setHasShadow_(False)

    content = win.contentView()
    content.setWantsLayer_(True)

    padding = 12
    label = CATextLayer.alloc().init()
    label.setFrame_(((padding, padding), (w - 2 * padding, h - 2 * padding)))
    label.setString_(text)
    label.setFont_("Menlo")
    label.setFontSize_(13)
    label.setForegroundColor_(
        NSColor.colorWithRed_green_blue_alpha_(0.9, 0.9, 0.9, 0.95).CGColor()
    )
    label.setWrapped_(True)
    label.setTruncationMode_("end")
    label.setContentsScale_(2.0)
    content.layer().addSublayer_(label)

    win.setAlphaValue_(1.0)
    win.orderFront_(None)
    _smoke_rect_window = win


def _flash_error_on_main(text: str, duration: float = 5.0) -> None:
    """Flash a debug error message on screen when positioning fails."""
    from PyObjCTools import AppHelper

    def _do():
        from AppKit import (
            NSBackingStoreBuffered,
            NSBorderlessWindowMask,
            NSColor,
            NSMakeRect,
            NSWindow,
        )
        from Quartz import CATextLayer

        global _debug_error_window
        if _debug_error_window is not None:
            _debug_error_window.orderOut_(None)

        # Get screen size
        from AppKit import NSScreen
        screen = NSScreen.mainScreen()
        if screen is None:
            return
        sf = screen.frame()
        sw, sh = sf.size.width, sf.size.height

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(50, sh - 350, sw - 100, 300),
            NSBorderlessWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        win.setLevel_(2147483647)
        win.setOpaque_(False)
        win.setBackgroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.75)
        )
        win.setIgnoresMouseEvents_(True)
        win.setHasShadow_(False)

        content = win.contentView()
        content.setWantsLayer_(True)

        label = CATextLayer.alloc().init()
        label.setFrame_(((10, 10), (sw - 120, 280)))
        label.setString_(text)
        label.setFontSize_(12)
        label.setForegroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(1.0, 0.4, 0.4, 1.0).CGColor()
        )
        label.setWrapped_(True)
        label.setContentsScale_(2.0)
        content.layer().addSublayer_(label)

        win.setAlphaValue_(1.0)
        win.orderFront_(None)
        _debug_error_window = win

        def _dismiss():
            def _do_dismiss():
                global _debug_error_window
                if _debug_error_window is not None:
                    _debug_error_window.orderOut_(None)
                    _debug_error_window = None
            AppHelper.callAfter(_do_dismiss)
        threading.Timer(duration, _dismiss).start()

    AppHelper.callAfter(_do)


def _show_diagnostic_overlay(result: dict, debug_steps: list[str] | None) -> None:
    """Show a persistent small diagnostic overlay in the upper right.

    Displays: utterance, pipeline steps, elapsed time. Persists until
    the next positioning run replaces it.
    """
    from PyObjCTools import AppHelper

    def _do():
        try:
            _do_inner()
        except Exception:
            logger.warning("Diagnostic overlay failed", exc_info=True)

    def _do_inner():
        from AppKit import (
            NSBackingStoreBuffered,
            NSBorderlessWindowMask,
            NSColor,
            NSMakeRect,
            NSScreen,
            NSWindow,
        )
        from Quartz import CATextLayer

        global _debug_diag_window
        if _debug_diag_window is not None:
            _debug_diag_window.orderOut_(None)

        screen = NSScreen.mainScreen()
        if screen is None:
            return
        sf = screen.frame()
        sw, sh = sf.size.width, sf.size.height

        # Build diagnostic text
        lines = []
        utterance = result.get("utterance", "")
        if utterance:
            lines.append(f'"{utterance}"')

        if debug_steps:
            for step in debug_steps:
                lines.append(f"  {step}")

        elapsed = result.get("elapsed_s", 0)
        content_desc = result.get("content_desc", "?")
        lines.append(f"→ {content_desc}")
        lines.append(f"  {elapsed:.1f}s total")

        rect = f"x={result['x']:.2f} y={result['y']:.2f} w={result['width']:.2f} h={result['height']:.2f}"
        lines.append(f"  {rect}")

        text = "\n".join(lines)

        # Top-right diagnostic panel — top third of screen
        win_w = 480
        win_h = int(sh / 3)
        win_x = sw - win_w - 8
        win_y = sh - win_h - 40  # below menu bar

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(win_x, win_y, win_w, win_h),
            NSBorderlessWindowMask,
            NSBackingStoreBuffered,
            False,
        )
        win.setLevel_(2147483647)
        win.setOpaque_(False)
        win.setBackgroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.65)
        )
        win.setIgnoresMouseEvents_(True)
        win.setHasShadow_(False)

        content = win.contentView()
        content.setWantsLayer_(True)

        label = CATextLayer.alloc().init()
        label.setFrame_(((8, 4), (win_w - 16, win_h - 8)))
        label.setString_(text)
        label.setFont_("Menlo")
        label.setFontSize_(11)
        label.setForegroundColor_(
            NSColor.colorWithRed_green_blue_alpha_(0.8, 1.0, 0.8, 0.9).CGColor()
        )
        label.setWrapped_(True)
        label.setContentsScale_(2.0)
        content.layer().addSublayer_(label)

        win.setAlphaValue_(1.0)
        win.orderFront_(None)
        _debug_diag_window = win

    AppHelper.callAfter(_do)


def _flash_debug_grid(
    sw: float, sh: float,
    content_map: dict[str, bool],
    content_desc: str,
    duration: float = 3.0,
    utterance: str = "",
    target_mode: bool = False,
    elapsed_s: float = 0.0,
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

    from .reposition import POS_ROWS, POS_COLS
    import string
    rows, cols = POS_ROWS, POS_COLS
    row_labels = string.ascii_uppercase[:rows]
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
            # In avoid mode: YES(content)=red, NO(empty)=green
            # In target mode: YES(occupy)=green, NO(skip)=red
            is_green = (not has_content) if not target_mode else has_content
            if is_green:
                cell.setBackgroundColor_(
                    NSColor.colorWithRed_green_blue_alpha_(0.2, 1.0, 0.2, 0.25).CGColor()
                )
            else:
                cell.setBackgroundColor_(
                    NSColor.colorWithRed_green_blue_alpha_(1.0, 0.2, 0.2, 0.25).CGColor()
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

    # Title bar showing utterance and resolved content description
    if content_desc.startswith("targeting:"):
        title_text = f"Targeting: {content_desc.removeprefix('targeting: ')}"
    elif content_desc.startswith("finding:"):
        title_text = f"Finding: {content_desc.removeprefix('finding: ')}"
    else:
        title_text = f"Avoiding: {content_desc}"
    if utterance:
        title_text = f"\"{utterance}\" → {title_text}"
    if elapsed_s > 0:
        title_text = f"{title_text}  ({elapsed_s:.1f}s)"
    title = CATextLayer.alloc().init()
    title.setFrame_(((0, sh - 30), (sw, 30)))
    title.setString_(title_text)
    title.setFontSize_(13)
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


def _get_main_screen_frame_for_worker() -> tuple[float, float] | None:
    """Get the main screen's (width, height) — safe to call from worker thread."""
    from AppKit import NSScreen
    screen = NSScreen.mainScreen()
    if screen is None:
        return None
    frame = screen.frame()
    return (frame.size.width, frame.size.height)


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
