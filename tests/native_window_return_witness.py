"""Offscreen AppKit/WebKit conformance; never activates or posts global keys."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def check_sources(root, expected):
    actual = {name: hashlib.sha256((root / 'spoke' / name).read_bytes()).hexdigest()
              for name in expected}
    assert actual == expected, 'Source revision mismatch'
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--expected-sources', type=json.loads, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'phase': 'source', 'passed': False, 'repo_root': str(root), 'pages': {}}
    hosts = []
    try:
        report['sources'] = check_sources(root, args.expected_sources)
        sys.path.insert(0, str(root))
        from spoke import monitor_window as mod
        from spoke.window_focus import WindowFocus, _hi, _cf, _cfstr, _CF_TRUE
        assert Path(mod.__file__).resolve() == root / 'spoke/monitor_window.py'
        import AppKit as AK
        import Foundation as FD
        import objc
        report['pyobjc_version'] = objc.__version__
        app = AK.NSApplication.sharedApplication()
        app.setActivationPolicy_(AK.NSApplicationActivationPolicyAccessory)
        app.finishLaunching()
        report['phase'] = 'ax_capture'
        origin = WindowFocus.capture()
        assert origin is not None and origin._ax_window, 'No actual focused AX window captured'
        report['captured_pid'] = origin.app.processIdentifier()
        report['ax_window_retained'] = True
        report['phase'] = 'already_main_setter'
        attribute, value = _cfstr(b'AXMain'), ctypes.c_void_p()
        try:
            assert _hi.AXUIElementCopyAttributeValue(origin._ax_window, attribute, ctypes.byref(value)) == 0
            _cf.CFBooleanGetValue.restype = ctypes.c_bool
            _cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
            assert value.value and _cf.CFBooleanGetValue(value), 'Probe requires an already-main window'
            current = WindowFocus.capture()
            _cf.CFEqual.restype = ctypes.c_bool
            _cf.CFEqual.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            assert current and current.app.processIdentifier() == origin.app.processIdentifier()
            assert _cf.CFEqual(current._ax_window, origin._ax_window), 'Focus changed before idempotent probe'
            error = _hi.AXUIElementSetAttributeValue(origin._ax_window, attribute, _CF_TRUE)
            report['already_main_setter_error'] = error
            assert error == 0, 'Installed window-main setter rejected the CFBoolean'
        finally:
            if value.value:
                _cf.CFRelease(value)
            if attribute:
                _cf.CFRelease(attribute)

        def settle(predicate):
            deadline = time.monotonic() + 40
            while not predicate():
                if time.monotonic() > deadline:
                    raise RuntimeError('Witness observation expired, not a product timeout')
                FD.NSRunLoop.currentRunLoop().runUntilDate_(FD.NSDate.dateWithTimeIntervalSinceNow_(0.02))

        def js(host, code):
            answer = []
            host._webview.evaluateJavaScript_completionHandler_(code, lambda value, error: answer.append((value, error)))
            settle(lambda: bool(answer))
            value, error = answer[0]
            assert error is None, str(error)
            return value

        for name, (_, url) in mod.MONITORS.items():
            report['phase'] = name + '_load'
            host = mod.MonitorWindow.alloc().initWithMonitor_(name)
            hosts.append(host)
            host.setup()
            host._window.setFrameOrigin_((AK.NSScreen.mainScreen().frame().size.width + 200, 100))
            host._window.orderFront_(None)
            settle(lambda: not host._webview.isLoading() and host._webview.URL() is not None)
            effective = str(host._webview.URL().absoluteString())
            assert effective == url, 'Wrong effective monitor route'
            if name == 'fleet':
                settle(lambda: js(host, "typeof loading !== 'undefined' && !loading && !!document.querySelector('#rows > .row')"))
            else:
                settle(lambda: js(host, "!!document.querySelector('[data-filter=all]')"))
                js(host, "document.querySelector('[data-filter=all]').click();undefined")
            page = report['pages'][name] = {'requested_url': url, 'effective_url': effective}
            host._set_visible(False)
            js(host, "window.__arrivalRequests=[];const oldFetch=window.fetch;window.fetch=(...args)=>{window.__arrivalRequests.push({url:String(args[0]),at:performance.now()});return oldFetch(...args)};undefined")
            report['phase'] = name + '_resume'
            host._set_visible(True)
            requests = js(host, 'JSON.stringify(window.__arrivalRequests)')
            page['immediate_requests'] = json.loads(requests)
            paths = [item['url'] for item in page['immediate_requests']]
            assert any(path.startswith('/api/fleet' if name == 'fleet' else '/api/state') for path in paths)
            if name == 'greenroom':
                assert '/api/state?view=all' in paths
                assert '/api/smoke-requests' in paths
            images = []
            host._webview.takeSnapshotWithConfiguration_completionHandler_(None, lambda value, error: images.append((value, error)))
            settle(lambda: bool(images))
            image, error = images[0]
            assert image is not None and error is None
            bitmap = AK.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
            png = bitmap.representationUsingType_properties_(AK.NSBitmapImageFileTypePNG, {})
            assert png and png.writeToFile_atomically_(str(out / (name + '.png')), True)
            page['snapshot'] = str(out / (name + '.png'))
            report['phase'] = name + '_native_close'
            calls = []
            class Return:
                def restore(self):
                    calls.append('restore')
            host._focus_return = Return()
            was_key = bool(host._window.isKeyWindow())
            host._window.performClose_(None)
            assert not host._window.isVisible()
            assert calls == (['restore'] if was_key else []), 'Native close lost focus ownership'
            page['native_close_respected_ownership'] = True
            page['native_was_key'] = was_key
        report.update(phase='complete', passed=True,
                      claim_limit='Actual AX capture and idempotent already-main setter, installed AppKit close callbacks and immediate GETs on actual retained WebKit pages. No app activation/window raising or physical hotkey claim.')
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        for host in hosts:
            try:
                host.cleanup()
            except Exception:
                report.update(passed=False, phase='cleanup', cleanup_error=traceback.format_exc())
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
