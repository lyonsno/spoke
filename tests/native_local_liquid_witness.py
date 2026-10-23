"""Exercise the actual SCK/Metal compositor over a deterministic moving desktop.

Run from a Screen Recording-authorized host under a Greenroom lease. The
retained stills reveal boundary misalignment, not refresh-rate tear freedom.
"""

import argparse
import ctypes
import hashlib
import json
import math
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def validate_arm(arm, expected_size):
    if not arm.get("capture_sizes"):
        raise ValueError("No native capture frames")
    if any(size != list(expected_size) for size in arm["capture_sizes"]):
        raise ValueError("Effective capture dimensions differ from requested viewport")
    if arm.get("displayed_frames", 0) <= 0:
        raise ValueError("No native presentation observed")
    rows = arm.get("presentation_rows", [])
    if (not rows or any(row.get("outcome") == "observer_unavailable" for row in rows)
            or not any(row.get("outcome") == "displayed"
                       and math.isfinite(row.get("presented_host_seconds", 0))
                       and row.get("presented_host_seconds", 0) > 0
                       and not row.get("stale_epoch") for row in rows)):
        raise ValueError("Native presentation observer missing or failed")
    warps = arm.get("warp_rows", [])
    if not warps or any(
        row.get("capture_size") != list(expected_size)
        or row.get("drawable_size") != list(expected_size)
        or row.get("dispatch_count", 0) <= 0 or row.get("skip_reason") is not None
        or row.get("composited_to_drawable_texture") is not True for row in warps
    ):
        raise ValueError("House warp route missing, skipped or dimension-mismatched")


def finish_arm(compositor, arm):
    arm.update(compositor.diagnostics_snapshot())
    before, after = arm["pipeline_before"], compositor._pipeline.diagnostics_snapshot()
    arm["pipeline_after"] = after
    for count, pixels, average in (
        ("drawable_copy_frames", "drawable_copy_pixels", "avg_drawable_copy_mp"),
        ("mip_generation_frames", "mip_generation_source_pixels", "avg_mip_generation_source_mp"),
        ("warp_dispatches", "warp_dispatch_pixels", "avg_warp_dispatch_mp"),
    ):
        for key in (count, pixels):
            arm[key] = after[key] - before[key]
            if arm[key] < 0:
                raise ValueError("Pipeline counters changed identity during arm")
        arm[average] = arm[pixels] / max(arm[count], 1) / 1000000


def run_native(args, report, save):
    import objc
    import Quartz as Q
    from AppKit import (NSApplication, NSScreen, NSWindow, NSView, NSColor,
                        NSBezierPath, NSBitmapImageRep)
    from Foundation import NSObject, NSTimer
    from spoke.fullscreen_compositor import FullScreenCompositor
    from spoke.house_optical_primitive import compile_house_optical_shell_config
    native_callbacks = []
    native_capture = None
    stop_events = []
    if args.native_capture_library:
        native_capture = ctypes.CDLL(str(args.native_capture_library.resolve()))
        state_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
        frame_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_double)
        native_capture.spoke_capture_start.argtypes = [ctypes.c_char_p, frame_type, state_type]
        native_capture.spoke_capture_start.restype = None
        native_capture.spoke_capture_stop.argtypes = [state_type]
        native_capture.spoke_capture_stop.restype = None
        report["capture_producer"] = "typed Swift/SCK; Python SCK startup is bypassed, not verified"
        report["native_capture_library"] = str(args.native_capture_library.resolve())
        report["native_capture_library_sha256"] = hashlib.sha256(args.native_capture_library.read_bytes()).hexdigest()
    if not Q.CGPreflightScreenCaptureAccess():
        raise RuntimeError("Effective host lacks Screen Recording access")
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # Match Spoke's accessory application host.
    report["activation_policy"] = int(app.activationPolicy())
    screen = NSScreen.mainScreen()
    frame, scale = screen.frame(), float(screen.backingScaleFactor())
    width, height = 760.0, 500.0
    left = round((frame.size.width - width) / 2)
    top = round((frame.size.height - height) / 2)
    crop = (left + 80, top + 70, 600.0, 360.0)
    began = time.monotonic()
    report.update(phase="native-main-loop", bridge_version=objc.__version__,
                  scale=scale, capture_rect_points=crop, source_draws=[], arms=[],
                  began_monotonic_seconds=began)

    class MovingLines(NSView):
        def drawRect_(self, dirty):
            elapsed = time.monotonic() - began
            phase = (elapsed * args.speed) % 48.0
            report["source_draws"].append({"elapsed": elapsed, "phase_points": phase})
            NSColor.colorWithCalibratedWhite_alpha_(0.12, 1).setFill()
            NSBezierPath.fillRect_(self.bounds())
            for y in range(-48, int(height) + 48, 48):
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.8, 0.7, 1).setFill()
                NSBezierPath.fillRect_(((0, y + phase), (width, 12)))
                NSColor.whiteColor().setFill()
                NSBezierPath.fillRect_(((0, y + phase + 20), (width, 2)))
            for x in range(0, int(width), 80):
                NSColor.colorWithCalibratedWhite_alpha_(0.5, 1).setFill()
                NSBezierPath.fillRect_(((x, 0), (2, height)))

    source = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((frame.origin.x + left, frame.origin.y + frame.size.height - top - height),
         (width, height)), 0, 2, False)
    source.setReleasedWhenClosed_(False)
    source.setLevel_(1200)
    report["fixture_window_levels"] = {"source": 1200, "compositor": 1201}
    source.setIgnoresMouseEvents_(True)
    view = MovingLines.alloc().initWithFrame_(((0, 0), (width, height)))
    source.setContentView_(view)
    source.orderFrontRegardless()
    config = compile_house_optical_shell_config(360 * scale, 150 * scale)
    config.update(center_x=(left + width / 2) * scale,
                  center_y=(top + height / 2) * scale,
                  client_id="local-liquid-witness", temporal_blend=1.0,
                  mip_blur_strength=0.0)
    report["shell_config"] = config
    state = {"index": 0, "compositor": None, "original_warp": None, "arm": None, "started": 0,
             "observed": None, "still": 0, "next_still": 0}

    def snapshot(arm):
        # Both arms retain the same screen rectangle, including live pixels
        # beyond every localized edge. Never crop away the disputed boundary.
        image = Q.CGWindowListCreateImage(
            ((frame.origin.x + left, top), (width, height)),
            Q.kCGWindowListOptionOnScreenOnly, Q.kCGNullWindowID, 0)
        if image is None:
            raise RuntimeError("WindowServer witness image missing")
        png = NSBitmapImageRep.alloc().initWithCGImage_(image).representationUsingType_properties_(4, {})
        if png is None:
            raise RuntimeError("PNG encoding failed")
        name = f"{arm['mode']}-{state['still']:02d}.png"
        data = bytes(png)
        (args.out / name).write_bytes(data)
        arm["stills"].append({"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                              "elapsed": time.monotonic() - began})
        state["still"] += 1

    def start_arm():
        mode = ("full_display", "localized")[state["index"]]
        if state["index"] == 0:
            reference = {"mode": "source", "stills": []}
            snapshot(reference)
            report["source_reference"] = reference["stills"]
        compositor = FullScreenCompositor(screen, capture_rect=crop if mode == "localized" else None)
        original_create = compositor._create_fullscreen_window

        def create_window():
            original_create()
            compositor.window.setLevel_(1201)
            color_space = compositor._metal_layer.colorspace()
            report["presentation_color_space"] = {
                "assigned": color_space is not None,
                "name": str(Q.CGColorSpaceGetName(color_space)) if color_space is not None else None,
                "display_profile": str(screen.colorSpace()),
            }

        compositor._create_fullscreen_window = create_window
        if args.render_on_capture:
            compositor._start_display_link = lambda: setattr(compositor, "_last_report_time", time.monotonic())
            report["presentation_driver"] = "direct SCK callback; no CVDisplayLink wait; callback backpressure is possible"
        else:
            report["presentation_driver"] = "existing CVDisplayLink"
        if native_capture is not None:
            viewport = compositor._capture_viewport
            rect = viewport.source_rect if viewport is not None else ((0, 0), (frame.size.width, frame.size.height))
            expected_size = viewport.pixel_size if viewport is not None else (round(frame.size.width * scale), round(frame.size.height * scale))

            def start_capture(attempt):
                @frame_type
                def received(pointer, width_, height_, source_time):
                    try:
                        pixel_buffer = objc.objc_object(c_void_p=pointer)
                        if mode not in report.setdefault("capture_attachments", {}):
                            attachments = Q.CVBufferCopyAttachments(pixel_buffer, 0) or {}
                            report["capture_attachments"][mode] = {str(k): str(v) for k, v in attachments.items()}
                        surface = Q.CVPixelBufferGetIOSurface(pixel_buffer)
                        compositor.submit_iosurface(surface, width=width_, height=height_,
                            pixel_buffer=pixel_buffer, source_display_seconds=source_time or None)
                        if args.render_on_capture:
                            compositor._on_display_link()
                    except BaseException:
                        report.setdefault("native_callback_errors", []).append(traceback.format_exc())

                @state_type
                def started(error):
                    compositor._set_capture_attempt_state(attempt,
                        "failed" if error else "started", RuntimeError(error.decode()) if error else None)

                native_callbacks.extend([received, started])
                request = {"x": rect[0][0], "y": rect[0][1], "widthPoints": rect[1][0],
                    "heightPoints": rect[1][1], "width": expected_size[0], "height": expected_size[1],
                    "fps": args.fps, "excludedWindowID": int(compositor.window.windowNumber()),
                    "sourceWindowID": int(source.windowNumber())}
                report.setdefault("typed_capture_requests", []).append(request)
                native_capture.spoke_capture_start(json.dumps(request).encode(), received, started)

            def stop_capture():
                event = threading.Event()
                stop_events.append(event)

                @state_type
                def stopped(error):
                    if error:
                        report.setdefault("native_callback_errors", []).append(error.decode())
                    event.set()

                native_callbacks.append(stopped)
                native_capture.spoke_capture_stop(stopped)

            compositor._start_capture = start_capture
            compositor._stop_capture = stop_capture
        arm = {"mode": mode, "capture_sizes": [], "presentation_rows": [], "warp_rows": [], "stills": [],
               "states": [], "requested_fps": args.fps,
               "expected_size": list((round(crop[2] * scale), round(crop[3] * scale))
                   if mode == "localized" else
                   (round(frame.size.width * scale), round(frame.size.height * scale)))}
        report["arms"].append(arm)
        state.update(compositor=compositor, arm=arm, started=time.monotonic(), observed=None,
                     still=0, next_still=0)
        pipeline = compositor._pipeline
        arm["pipeline_before"] = pipeline.diagnostics_snapshot()
        original_warp = pipeline.warp_to_drawable
        state["original_warp"] = original_warp

        def warp(surface, drawable, **kwargs):
            texture = drawable.texture()
            size = [int(texture.width()), int(texture.height())] if texture is not None else None
            result = original_warp(surface, drawable, **kwargs)
            outcome = pipeline.warp_dispatches_by_client_snapshot().get("local-liquid-witness", {})
            arm["warp_rows"].append({"capture_size": [kwargs["width"], kwargs["height"]],
                                     "drawable_size": size, "returned": bool(result), **outcome})
            return result

        pipeline.warp_to_drawable = warp
        original_submit = compositor.submit_iosurface

        def submit(surface, *, width, height, **kwargs):
            arm["capture_sizes"].append([int(width), int(height)])
            return original_submit(surface, width=width, height=height, **kwargs)

        compositor.submit_iosurface = submit
        if not compositor.start(config, on_capture_state=lambda status, error=None:
                                arm["states"].append({"state": status, "error": str(error)})):
            raise RuntimeError("Compositor startup rejected")
        compositor._presentation_timing._emit = arm["presentation_rows"].append
        save()

    def finish():
        if state["compositor"] is not None:
            state["compositor"].stop()
            state["compositor"]._pipeline.warp_to_drawable = state["original_warp"]
        source.orderOut_(None)
        timer.invalidate()
        app.stop_(None)
        # Wake NSApplication.run after stop; no global key or event tap.
        from AppKit import NSEvent
        event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            15, (0, 0), 0, 0, 0, None, 0, 0, 0)
        app.postEvent_atStart_(event, False)

    def tick():
        try:
            view.setNeedsDisplay_(True)
            if state["compositor"] is None:
                if any(not event.is_set() for event in stop_events):
                    return
                if state["index"] == 2:
                    if report.get("native_callback_errors"):
                        raise RuntimeError("Typed capture callback failed")
                    report.update(phase="complete", passed=True)
                    finish()
                    return
                start_arm()
                return
            compositor, arm = state["compositor"], state["arm"]
            if compositor.capture_start_state == "failed":
                raise RuntimeError(compositor.capture_start_error)
            now = time.monotonic()
            if state["observed"] is None:
                if compositor.diagnostics_snapshot()["displayed_frames"] > 0:
                    state["observed"] = now
                elif now - state["started"] >= args.startup_seconds:
                    raise RuntimeError("Observation window ended before native presentation")
                return
            elapsed = now - state["observed"]
            if elapsed >= args.seconds:
                if "timing_only" not in arm:
                    timing = {"pipeline_before": arm["pipeline_before"]}
                    finish_arm(compositor, timing)
                    arm["timing_only"] = {**timing, "end_monotonic_seconds": now,
                                          "presentation_row_count": len(arm["presentation_rows"])}
                if state["still"] < args.stills:
                    if now >= state["next_still"]:
                        snapshot(arm)
                        state["next_still"] = now + 0.1
                    return
                finish_arm(compositor, arm)
                validate_arm(arm, arm["expected_size"])
                compositor.stop()
                compositor._pipeline.warp_to_drawable = state["original_warp"]
                state["compositor"] = None
                state["index"] += 1
                save()
        except BaseException:
            report["error"] = traceback.format_exc()
            frames = sys._current_frames()
            report["thread_stacks"] = {thread.name: traceback.format_stack(frames[thread.ident])
                for thread in threading.enumerate() if thread.ident in frames}
            if state["compositor"] is not None:
                current = state["compositor"]
                state["arm"].update(current.diagnostics_snapshot())
                state["arm"]["startup_objects"] = {
                    key: getattr(current, key, None) is not None for key in (
                        "_capture_content", "_capture_display", "_stream", "_stream_output",
                        "_stream_handler_queue", "_latest_pixel_buffer")}
            finish()
        finally:
            if report.get("error") or report.get("passed"):
                save()

    class Driver(NSObject):
        def tick_(self, unused):
            tick()

    driver = Driver.alloc().init()
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / args.fps, driver, "tick:", None, True)
    try:
        app.run()
    finally:
        timer.invalidate()
        if state["compositor"] is not None:
            state["compositor"].stop()
            state["compositor"]._pipeline.warp_to_drawable = state["original_warp"]
        source.orderOut_(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--startup-seconds", type=float, default=15)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--speed", type=float, default=240)
    parser.add_argument("--stills", type=int, default=6)
    parser.add_argument("--native-capture-library", type=Path)
    parser.add_argument("--render-on-capture", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "phase": "identity", "argv": sys.argv,
              "repo_root": str(args.repo_root.resolve()),
              "claim": "Native presentation and sparse moving-boundary frames; not tear-free scanout proof"}

    def save():
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if args.render_on_capture and not args.native_capture_library:
            raise ValueError("Direct capture rendering requires the explicit typed producer")
        root = args.repo_root.resolve()
        sys.path.insert(0, str(root))
        import spoke.fullscreen_compositor as module
        if Path(module.__file__).resolve() != root / "spoke/fullscreen_compositor.py":
            raise RuntimeError("Effective compositor differs from requested source")
        report["revision"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("spoke/fullscreen_compositor.py", "spoke/compositor_viewport.py", "spoke/metal_warp.py")}
        # The module reads cadence at import time; set it explicitly for this
        # isolated witness without changing the live app or launcher.
        module._SCK_TARGET_FPS = args.fps
        module._SCK_FRAME_INTERVAL = (1, args.fps, 0, 0)
        os.environ["SPOKE_COMPOSITOR_PRESENTATION_TIMING"] = "1"
        run_native(args, report, save)
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
