"""Transparent native carrier inspection. Never activates or covers the desktop."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


SOURCES = ("house_window.py", "monitor_window.py", "recording_history_window.py",
           "diaulos_switcher_overlay.py", "optical_lifecycle.py")


def check_sources(root, expected):
    if set(expected) != set(SOURCES):
        raise ValueError("Incomplete source identity")
    actual = {name: hashlib.sha256((root / "spoke" / name).read_bytes()).hexdigest() for name in SOURCES}
    if actual != expected:
        raise ValueError("Source revision mismatch")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-sources", type=json.loads, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"phase": "source", "passed": False, "repo_root": str(root), "windows": {}}
    hosts = []
    try:
        report["sources"] = check_sources(root, args.expected_sources)
        sys.path.insert(0, str(root))
        import AppKit as AK
        import Foundation as FD
        import objc
        from spoke import house_window, monitor_window, recording_history_window, diaulos_switcher_overlay
        from spoke.house_optical_primitive import (
            house_dismiss_materialization_fill_state,
            house_dismiss_text_collapse_state,
        )
        for module in (house_window, monitor_window, recording_history_window, diaulos_switcher_overlay):
            assert Path(module.__file__).resolve().parent == root / "spoke", "Wrong import route"
        app = AK.NSApplication.sharedApplication()
        app.setActivationPolicy_(AK.NSApplicationActivationPolicyProhibited)
        app.finishLaunching()
        report["pyobjc"] = objc.__version__

        def settle(predicate):
            deadline = time.monotonic() + 40
            while not predicate():
                if time.monotonic() > deadline:
                    raise RuntimeError("Native witness observation expired")
                FD.NSRunLoop.currentRunLoop().runUntilDate_(FD.NSDate.dateWithTimeIntervalSinceNow_(0.02))

        delegate = SimpleNamespace(
            _audio_spool=SimpleNamespace(config=SimpleNamespace(root=out / "synthetic-recordings", enabled=True)),
            _history_model={}, _history_model_options=lambda: [], _history_live_busy=lambda: False,
            _overlay_compositor_registry=None,
        )
        history = recording_history_window.RecordingHistoryWindow.alloc().initWithDelegate_(delegate)
        teleporter = diaulos_switcher_overlay.DiaulosSwitcherOverlay.alloc().initWithDelegate_(delegate)
        hosts.extend([history, teleporter])
        hosts.extend(monitor_window.MonitorWindow.alloc().initWithMonitor_(name) for name in monitor_window.MONITORS)

        for host in hosts:
            report["phase"] = type(host).__name__ + "_setup"
            host.setup()
            window = host._panel
            assert not window.isVisible(), "Witness unexpectedly ordered a window"
            # AppKit constrains ordinary windows back onto the display when
            # ordered. WindowServer transparency, not offscreen coordinates,
            # isolates this witness; the internal native layers remain readable.
            window.setAlphaValue_(0.0)
            window.setIgnoresMouseEvents_(True)
            window.setLevel_(-1000)
            window.orderFront_(None)
            try:
                assert window.alphaValue() == 0 and not window.isKeyWindow(), "Native order changed isolation/focus"
                FD.NSRunLoop.currentRunLoop().runUntilDate_(FD.NSDate.dateWithTimeIntervalSinceNow_(0.1))
                if isinstance(host, monitor_window.MonitorWindow):
                    settle(lambda: not host._webview.isLoading())
                    assert str(host._webview.URL().absoluteString()) == monitor_window.MONITORS[host._name][1]
                    snapshots = []
                    host._webview.takeSnapshotWithConfiguration_completionHandler_(None, lambda image, error: snapshots.append((image, error)))
                    settle(lambda: bool(snapshots))
                    assert snapshots[0][0] is not None and snapshots[0][1] is None, "Native WebKit snapshot failed"
            except BaseException:
                window.orderOut_(None)
                raise
            assert window.contentView() is not None
            foreground = host._shell_foreground
            foreground.displayIfNeeded()
            size = foreground.bounds().size
            width, height = int(size.width), int(size.height)
            assert width > 0 and height > 0
            entry = report["windows"][host._shell_client_id] = {
                "size": [width, height], "window_alpha": window.alphaValue(), "frames": {},
                "background_alpha": window.backgroundColor().alphaComponent(),
            }
            # Real native CALayer masks, with the same plane values used by the
            # shared runner. Capture/scanout and live scrolling are not exercised.
            states = [(name, {"opacity": fraction, "height_frac": fraction},
                       {"alpha": 1.0, "width_frac": 1.0, "height_frac": fraction})
                      for name, fraction in (("rest", 1.0), ("reveal", 0.4), ("hidden", 0.0))]
            states.extend((name, house_dismiss_materialization_fill_state(progress),
                           house_dismiss_text_collapse_state(progress))
                          for name, progress in (("dismiss80", 0.8), ("dismiss60", 0.6)))
            for name, body, text in states:
                report["phase"] = host._shell_client_id + "_" + name
                host._set_shell_planes(body, text)
                bitmap = AK.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
                    None, width, height, 8, 4, True, False, AK.NSDeviceRGBColorSpace, width * 4, 32,
                )
                graphics = AK.NSGraphicsContext.graphicsContextWithBitmapImageRep_(bitmap)
                foreground.layer().renderInContext_(graphics.CGContext())
                png = bitmap.representationUsingType_properties_(AK.NSBitmapImageFileTypePNG, {})
                path = out / (host._shell_client_id + "-" + name + ".png")
                assert png and png.writeToFile_atomically_(str(path), True)
                raw = bytes(bitmap.bitmapData())
                occupied = sum(alpha != 0 for alpha in raw[3::4])
                colors = len(set(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4])))
                rows = [y for y in range(height) if any(raw[(y * width) * 4 + 3:((y + 1) * width) * 4:4])]
                row_span = rows[-1] - rows[0] + 1 if rows else 0
                entry["frames"][name] = {"path": str(path), "occupied_pixels": occupied, "colors": colors,
                                          "occupied_row_span": row_span, "body": body, "text": text}
                if name.startswith("dismiss") and host._shell_body_layer is None:
                    limit = height * min(body["height_frac"], text["height_frac"])
                    assert row_span <= limit + 2, "Native pixels extend outside closing body"
            frames = entry["frames"]
            assert frames["rest"]["occupied_pixels"] > 0, "Blank native carrier"
            assert frames["rest"]["colors"] > 128, "Native carrier has no painted controls"
            assert 0 < frames["reveal"]["occupied_pixels"] < frames["rest"]["occupied_pixels"], "Reveal did not clip native pixels"
            assert frames["hidden"]["occupied_pixels"] == 0, "Hidden native pixels leaked"
            assert entry["background_alpha"] == 0, "Unmasked window background can leak outside captured layer"
            assert window.alphaValue() == 0 and not window.isKeyWindow()
            window.orderOut_(None)
        report.update(passed=True, phase="complete", claim_limit="Installed AppKit native carrier pixels and reveal masks only; transparent input-ignoring non-key windows, no SCK, Metal presentation, desktop scanout or latency claim.")
    except BaseException:
        report["error"] = traceback.format_exc()
        return 1
    finally:
        for host in hosts:
            try:
                host.cleanup()
            except Exception:
                report.update(passed=False, cleanup_error=traceback.format_exc())
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
