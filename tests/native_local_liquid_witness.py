"""Exercise the actual SCK/Metal compositor over a deterministic moving desktop.

Run from a Screen Recording-authorized host under a Greenroom lease. The
retained stills reveal boundary misalignment, not refresh-rate tear freedom.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
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


def run_native(args, report, save):
    import objc
    import Quartz as Q
    from AppKit import (NSApplication, NSScreen, NSWindow, NSView, NSColor,
                        NSBezierPath, NSBitmapImageRep)
    from Foundation import NSObject, NSTimer
    from spoke.fullscreen_compositor import FullScreenCompositor
    from spoke.house_optical_primitive import compile_house_optical_shell_config

    if not Q.CGPreflightScreenCaptureAccess():
        raise RuntimeError("Effective host lacks Screen Recording access")
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(2)
    screen = NSScreen.mainScreen()
    frame, scale = screen.frame(), float(screen.backingScaleFactor())
    width, height = 760.0, 500.0
    left = round((frame.size.width - width) / 2)
    top = round((frame.size.height - height) / 2)
    crop = (left + 80, top + 70, 600.0, 360.0)
    began = time.monotonic()
    report.update(phase="native-main-loop", bridge_version=objc.__version__,
                  scale=scale, capture_rect_points=crop, source_draws=[], arms=[])

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
    source.setLevel_(23)
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
    state = {"index": 0, "compositor": None, "arm": None, "started": 0,
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
        compositor = FullScreenCompositor(screen, capture_rect=crop if mode == "localized" else None)
        arm = {"mode": mode, "capture_sizes": [], "presentation_rows": [], "stills": [],
               "states": [], "requested_fps": args.fps,
               "expected_size": list((round(crop[2] * scale), round(crop[3] * scale))
                   if mode == "localized" else
                   (round(frame.size.width * scale), round(frame.size.height * scale)))}
        report["arms"].append(arm)
        state.update(compositor=compositor, arm=arm, started=time.monotonic(), observed=None,
                     still=0, next_still=0)
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
                if state["still"] < args.stills:
                    if now >= state["next_still"]:
                        snapshot(arm)
                        state["next_still"] = now + 0.1
                    return
                arm.update(compositor.diagnostics_snapshot())
                validate_arm(arm, arm["expected_size"])
                compositor.stop()
                state["compositor"] = None
                state["index"] += 1
                save()
                if state["index"] == 2:
                    report.update(phase="complete", passed=True)
                    finish()
        except BaseException:
            report["error"] = traceback.format_exc()
            if state["compositor"] is not None:
                state["arm"].update(state["compositor"].diagnostics_snapshot())
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "phase": "identity", "argv": sys.argv,
              "repo_root": str(args.repo_root.resolve()),
              "claim": "Native presentation and sparse moving-boundary frames; not tear-free scanout proof"}

    def save():
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
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
