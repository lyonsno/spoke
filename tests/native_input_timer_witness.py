"""Replay installed Foundation selectors without a tap, window, or microphone."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--expected-digest', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'passed': False, 'phase': 'source', 'repo_root': str(args.repo_root.resolve())}
    try:
        source = args.repo_root.resolve() / 'spoke/input_tap.py'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        report['source_digest'] = digest
        if digest != args.expected_digest:
            raise ValueError('input source digest mismatch')
        sys.path.insert(0, str(args.repo_root.resolve()))
        report['phase'] = 'bridge'
        import objc
        from Foundation import NSDate, NSRunLoop, NSTimer
        from spoke import input_tap as mod
        if Path(mod.__file__).resolve() != source:
            raise ValueError('effective input module is not requested source')
        report['bridge_version'] = objc.__version__
        report['effective_module'] = str(Path(mod.__file__).resolve())
        starts, ends, shifts = [], [], []
        det = mod.SpacebarHoldDetector.alloc().initWithHoldStart_holdEnd_holdMs_(
            lambda: starts.append('start'), lambda **kw: ends.append(kw), 20)
        det._on_shift_tap_idle = lambda: shifts.append('shift')
        routes = [('_hold_timer', 'holdTimerFired:', mod._State.WAITING),
                  ('_safety_timer', 'safetyTimerFired:', mod._State.RECORDING),
                  ('_repeat_watchdog_timer', 'repeatWatchdogFired:', mod._State.IDLE),
                  ('_forwarding_timer', 'forwardingTimerFired:', mod._State.IDLE),
                  ('_release_decision_timer', 'releaseDecisionTimerFired:', mod._State.IDLE),
                  ('_shift_single_tap_timer', '_shiftSingleTapFired:', mod._State.IDLE)]
        report['phase'] = 'selectors'
        report['routes'] = []
        for slot, selector, state in routes:
            det._state = state
            det._forwarding = True
            det._pending_release_active = True
            det._pending_release_shift_held = False
            old = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                3600, det, selector, None, False)
            current = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                3600, det, selector, None, False)
            setattr(det, slot, current)
            before = (len(starts), len(ends), len(shifts))
            old.fire()
            assert getattr(det, slot) == current, slot
            assert det._state == state and det._forwarding and det._pending_release_active, slot
            assert before == (len(starts), len(ends), len(shifts)), slot
            current.fire()
            assert getattr(det, slot) is None, slot
            old.invalidate()
            current.invalidate()
            for other, _, _ in routes:
                timer = getattr(det, other, None)
                if timer is not None:
                    timer.invalidate()
                    setattr(det, other, None)
            report['routes'].append({'slot': slot, 'stale_ignored': True, 'current_dispatched': True})
        report['phase'] = 'scheduled_delivery'
        det._state = mod._State.WAITING
        before = len(starts)
        det._start_hold_timer()
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.08))
        assert len(starts) == before + 1 and det._state == mod._State.RECORDING
        det._cancel_safety_timer()
        report['scheduled_hold_started_once'] = True
        report['phase'] = 'native_key_adapter'
        from Quartz import CGEventCreateKeyboardEvent, CGEventSetFlags, CGEventGetFlags
        det._state = mod._State.IDLE
        det._forwarding = False
        det._awaiting_space_release = False
        det._monitor_context = lambda: 'fleet'
        spaces = []
        det._on_monitor_space = lambda: spaces.append('space')
        mod._active_detector = det
        down = CGEventCreateKeyboardEvent(None, mod.SPACEBAR_KEYCODE, True)
        up = CGEventCreateKeyboardEvent(None, mod.SPACEBAR_KEYCODE, False)
        CGEventSetFlags(down, 0)
        CGEventSetFlags(up, 0)
        report['key_fixture'] = {'down_flags': int(CGEventGetFlags(down)),
                                 'up_flags': int(CGEventGetFlags(up)),
                                 'enter_commands': mod.ENTER_COMMANDS_ENABLED}
        assert report['key_fixture']['down_flags'] == report['key_fixture']['up_flags'] == 0
        assert mod._event_tap_callback(None, mod.kCGEventKeyDown, down, None) is None, 'down not consumed'
        assert mod._event_tap_callback(None, mod.kCGEventKeyUp, up, None) is None, 'up not consumed'
        report['key_result'] = {'spaces': spaces, 'state': det._state.name}
        assert spaces == ['space'] and det._state == mod._State.IDLE, 'unexpected monitor action/state'
        assert det._hold_timer is None and not det._monitor_gestures().space_down
        mod._active_detector = None
        report['native_key_adapter_passed'] = True
        report.update(passed=True, phase='complete')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
