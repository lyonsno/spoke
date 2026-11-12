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
    pill = None
    try:
        sources = [root / p for p in ("spoke/__main__.py", "spoke/capture.py", "spoke/handsfree.py",
                                     "spoke/glow.py")]
        report["source_digest"] = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
        assert report["source_digest"] == args.expected_digest, "capture source digest mismatch"
        sys.path.insert(0, str(root))
        os.environ["SPOKE_VAD_ENABLED"] = "0"
        os.environ["SPOKE_MODEL_PREFERENCES_PATH"] = str(args.out.parent / "preferences.json")
        os.environ.pop("SPOKE_LAUNCH_TARGET_ID", None)
        report["phase"] = "bridge"
        import objc
        from Foundation import NSDate, NSRunLoop
        from spoke import __main__ as mod, capture, handsfree, glow
        assert [Path(m.__file__).resolve() for m in (mod, capture, handsfree, glow)] == sources
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
        d._glow = MagicMock()
        def show_glow():
            assert cap._start_future is not None, "graphics ran before capture was admitted"
            report["capture_admitted_before_graphics"] = True
        d._glow.show.side_effect = show_glow
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
            cap.stop()
            assert cap._teardown_complete.wait(10)
            report["phase"] = "handsfree_completion"
            hf = d._handsfree = handsfree.HandsFreeController(d)
            hf.on_state_change = d._on_handsfree_state_change
            with patch("spoke.inject.save_pasteboard", return_value=None):
                hf._start_dictating()
            run_until(lambda: hf._capture_start_future is None)
            assert hf.is_dictating and cap.is_recording
            assert d._menubar.set_recording.call_args.args == (True,)
            hf._stop_dictation_capture()
            report["handsfree_completion_passed"] = True
            report["phase"] = "persistent_glow_release"
            import numpy as np
            from spoke.audio_spool import AudioSpool, AudioSpoolConfig
            spool = d._audio_spool = AudioSpool(AudioSpoolConfig(root=args.out.parent / "fixture-audio"))
            d._handsfree = None
            d._local_mode = False
            d._transcription_token = d._parallel_insert_token = 0
            d._transcribe_worker = d._parallel_insert_worker = lambda *args, **kwargs: None
            broken = d._glow = glow.GlowOverlay.alloc().initWithScreen_(None)
            broken._window = MagicMock()
            broken._glow_layer = MagicMock()
            broken._glow_layer.presentationLayer.return_value = None
            broken._glow_layer.opacity.return_value = 0.0
            broken._glow_layer.setOpacity_.side_effect = RuntimeError("persistent native glow-layer failure")
            broken._dim_layer = None
            samples = np.linspace(-0.2, 0.2, 1024, dtype=np.float32).reshape(-1, 1)
            retained = []
            with patch.object(glow, "_sample_screen_brightness", return_value=0.5):
                for _ in range(2):
                    prior = {r["capture_id"] for r in spool.list_recordings()}
                    d._on_hold_start()
                    run_until(lambda: not d._manual_capture_start_pending)
                    assert cap.is_recording and d._manual_capture_owned
                    broken._fade_in_until = 0
                    d.amplitudeUpdate_(0.1)
                    cap._audio_callback(samples, len(samples), None, None)
                    expected_audio = cap.get_buffer()
                    assert len(expected_audio) > 44
                    d._on_hold_end()
                    assert not cap.is_recording and not d._manual_capture_owned
                    assert not d._manual_hold_active
                    assert cap._teardown_complete.wait(10)
                    added = [r for r in spool.list_recordings() if r["capture_id"] not in prior]
                    assert len(added) == 1
                    capture_id = added[0]["capture_id"]
                    actual_audio = spool.read_recording_audio(capture_id)
                    assert actual_audio == expected_audio
                    assert added[0]["preserve_audio"] is True
                    retained.append({"capture_id": capture_id, "bytes": len(actual_audio),
                                     "sha256": hashlib.sha256(actual_audio).hexdigest()})
            report["persistent_glow_release_and_retry_passed"] = True
            report["retained_fixture_audio"] = retained
            report["phase"] = "history_glow_cleanup"
            from contextlib import nullcontext
            from types import SimpleNamespace
            from spoke.inference_process import InferenceControl, InferenceCancelled
            client = SimpleNamespace(supports_cancellation=True, cancel=MagicMock(),
                                     cancellation_scope=nullcontext)
            control = d._inference_control = InferenceControl()
            d._client = client
            d._dictation_delivery_records = lambda: ()
            returned = False
            try:
                with control.context(None, client):
                    assert d._history_can_take_over()
                    returned = d._cancel_transcription_for_recovery()
            except InferenceCancelled:
                pass
            assert returned is True, "History cancellation did not reach replacement admission"
            client.cancel.assert_called_once_with()
            assert broken._window.orderOut_.called
            assert not d._transcribing
            report["history_glow_cleanup_passed"] = True
        report["phase"] = "pill_timer_bridge"
        from Foundation import NSTimer
        from spoke.overlay import TranscriptionOverlay
        pill = TranscriptionOverlay.alloc().initWithScreen_(None)
        pill._window = MagicMock()
        pill._show_started_at = time.perf_counter()
        pill._last_show_timing = {"show_generation": 1}
        pill._fade_direction = 1
        pill._fade_step = 0
        pill._fade_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.01, pill, "fadeStep:", None, True)
        run_until(lambda: "first_fade_callback_ms" in pill._last_show_timing)
        pill._cancel_fade()
        assert pill._last_show_timing["first_fade_callback_ms"] >= 0
        assert pill._window.setAlphaValue_.called
        report["installed_pill_timer_clock_passed"] = True
        report["pill_timer_clock"] = dict(pill._last_show_timing)
        report.update(passed=True, phase="complete")
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        release.set()
        try:
            if pill is not None:
                pill._cancel_fade()
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
