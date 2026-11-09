"""Exercise production startup completion through the installed Cocoa bridge."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    report = {"passed": False, "phase": "source", "repo_root": str(root)}
    release = threading.Event()
    cap = None
    try:
        sources = [root / "spoke/__main__.py", root / "spoke/capture.py"]
        report["source_digest"] = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
        assert report["source_digest"] == args.expected_digest, "capture source digest mismatch"
        sys.path.insert(0, str(root))
        os.environ["SPOKE_VAD_ENABLED"] = "0"
        os.environ["SPOKE_MODEL_PREFERENCES_PATH"] = str(args.out.parent / "preferences.json")
        os.environ.pop("SPOKE_LAUNCH_TARGET_ID", None)
        report["phase"] = "bridge"
        import objc
        from Foundation import NSDate, NSRunLoop
        from spoke import __main__ as mod, capture
        assert [Path(m.__file__).resolve() for m in (mod, capture)] == sources
        report["bridge_version"] = objc.__version__
        report["effective_modules"] = [str(p) for p in sources]
        d = objc.super(mod.SpokeAppDelegate, mod.SpokeAppDelegate.alloc()).init()
        cap = d._capture = capture.AudioCapture()
        for name, value in {"_models_ready": True, "_mic_ready": True,
                "_manual_hold_active": False, "_transcribing": False, "_tray_active": False,
                "_recovery_text": None, "_command_overlay": None, "_tts_client": None,
                "_glow": None, "_overlay": None, "_diaulos_switcher": None,
                "_handsfree": None, "_whisper_backend": "local", "_preview_active": False}.items():
            setattr(d, name, value)
        d._menubar = MagicMock()
        d._detector = MagicMock(_enter_held=False, _shift_at_press=False, _shift_latched=False)
        preview = threading.Event()
        d._preview_loop = lambda token: preview.set()
        entered = threading.Event()
        caller = threading.get_ident()
        stream = MagicMock()

        def open_stream(**kwargs):
            entered.set()
            assert threading.get_ident() != caller, "native open ran on AppKit caller"
            assert release.wait(10), "fixture release was not delivered"
            return stream

        def run_until(predicate):
            deadline = time.monotonic() + 10
            while not predicate() and time.monotonic() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))
            assert predicate(), "native completion was not delivered"

        with patch.object(capture.sd, "InputStream", side_effect=open_stream):
            report["phase"] = "pending_open"
            d._on_hold_start()
            assert entered.wait(10)
            assert d._manual_capture_start_pending and not d._preview_active
            assert not any(c.args == (True,) for c in d._menubar.set_recording.call_args_list)
            release.set()
            report["phase"] = "native_completion"
            run_until(lambda: not d._manual_capture_start_pending)
            assert d._manual_hold_active and d._preview_active and preview.wait(10)
            assert d._menubar.set_recording.call_args.args == (True,)
            report["pending_and_completion_passed"] = True
            report["phase"] = "stale_completion"
            token = d._manual_capture_start_generation
            d._manual_capture_start_generation += 1
            d._manual_hold_active = False
            before = list(d._menubar.set_recording.call_args_list)
            delivered = threading.Event()

            def send_stale():
                d.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "manualCaptureStarted:", (token, cap._start_future), True)
                delivered.set()
            worker = threading.Thread(target=send_stale, daemon=True)
            worker.start()
            run_until(delivered.is_set)
            worker.join()
            assert d._menubar.set_recording.call_args_list == before
            report["stale_completion_ignored"] = True
        report.update(passed=True, phase="complete")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        release.set()
        try:
            if cap is not None:
                cap.stop()
                assert cap._teardown_complete.wait(10), "fixture teardown did not finish"
        except BaseException as exc:
            report.update(passed=False, cleanup_error=f"{type(exc).__name__}: {exc}")
        finally:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
