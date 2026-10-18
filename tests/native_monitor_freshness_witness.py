"""Actual local monitor pages in offscreen retained WebKit hosts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--sources', type=json.loads, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'phase': 'source', 'passed': False, 'repo_root': str(root), 'pages': {}}
    hosts = []
    try:
        actual = {name: hashlib.sha256((root / 'spoke' / name).read_bytes()).hexdigest() for name in args.sources}
        report['sources'] = actual
        assert actual == args.sources, 'Source revision mismatch'
        sys.path.insert(0, str(root))
        from spoke import monitor_window as mod
        assert Path(mod.__file__).resolve() == root / 'spoke/monitor_window.py'
        import AppKit as AK
        import Foundation as FD
        import objc
        report['pyobjc_version'] = objc.__version__
        app = AK.NSApplication.sharedApplication()
        app.setActivationPolicy_(AK.NSApplicationActivationPolicyAccessory)
        app.finishLaunching()

        def settle(predicate):
            deadline = time.monotonic() + 60
            while not predicate():
                if time.monotonic() > deadline:
                    raise RuntimeError('Witness observation expired; not a product timeout')
                FD.NSRunLoop.currentRunLoop().runUntilDate_(FD.NSDate.dateWithTimeIntervalSinceNow_(0.02))

        def js(host, source):
            values = []
            host._webview.evaluateJavaScript_completionHandler_(source, lambda value, error: values.append((value, error)))
            settle(lambda: bool(values))
            value, error = values[0]
            assert error is None, str(error)
            return value

        def screenshot(host, name):
            images = []
            host._webview.takeSnapshotWithConfiguration_completionHandler_(None, lambda value, error: images.append((value, error)))
            settle(lambda: bool(images))
            image, error = images[0]
            assert image is not None and error is None, str(error)
            bitmap = AK.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
            png = bitmap.representationUsingType_properties_(AK.NSBitmapImageFileTypePNG, {})
            path = out / (name + '.png')
            assert png and png.writeToFile_atomically_(str(path), True)
            return str(path)

        for name, (_, url) in mod.MONITORS.items():
            report['phase'] = name + '_load'
            host = mod.MonitorWindow.alloc().initWithMonitor_(name)
            hosts.append(host)
            host.setup()
            host._window.setFrameOrigin_((AK.NSScreen.mainScreen().frame().size.width + 200, 100))
            host._window.orderFront_(None)
            settle(lambda: not host._webview.isLoading() and host._webview.URL() is not None)
            assert str(host._webview.URL().absoluteString()) == url, 'Wrong effective route'
            text = "document.getElementById('spoke-monitor-freshness')?.textContent||''"
            settle(lambda: js(host, "typeof last !== 'undefined' && !!last"))
            if name == 'greenroom':
                settle(lambda: js(host, "typeof lastSmoke !== 'undefined' && !!lastSmoke"))
            entry = report['pages'][name] = {'requested_url': url, 'effective_url': url,
                                            'baseline': js(host, text), 'baseline_png': screenshot(host, name + '-baseline')}
            assert 'Last observation' in entry['baseline'] or 'Observed' in entry['baseline'], 'Actual payload never consumed'
            # Hold page refresh initiation while measuring the already-retained reopen state.
            js(host, "window.__originalLoad=window.load;window.load=()=>{};window.__originalSmoke=window.loadSmoke;window.loadSmoke=()=>{};undefined")
            host._set_visible(False)
            host._set_visible(True)
            entry['reopen'] = js(host, text)
            assert 'Unverified' in entry['reopen'], 'Cached view looked current on reopen'
            entry['reopen_png'] = screenshot(host, name + '-reopen')
            report['phase'] = name + '_failure'
            code = ("fetch('/api/state?view=__spoke_invalid_view__',{headers:auth()})" if name == 'greenroom' else
                    "fetch('/api/fleet',{headers:{Authorization:'Bearer spoke-witness-invalid'}})")
            js(host, "window.__failureDone=false;" + code + ".then(r=>{window.__failureCode=r.status;window.__failureDone=true}).catch(e=>{window.__failureError=String(e);window.__failureDone=true});undefined")
            settle(lambda: js(host, 'window.__failureDone'))
            entry['failure_code'] = js(host, 'window.__failureCode')
            entry['failure'] = js(host, text)
            assert entry['failure_code'] == (400 if name == 'greenroom' else 401)
            assert 'Unverified' in entry['failure'] and 'HTTP' in entry['failure']
            entry['failure_png'] = screenshot(host, name + '-failure')
            if name == 'greenroom':
                report['phase'] = name + '_drag_deferral'
                js(host, "window.load=window.__originalLoad;draggedJob={job_id:'spoke-witness-held'};window.__dragDone=false;load().finally(()=>window.__dragDone=true);undefined")
                settle(lambda: js(host, 'window.__dragDone'))
                entry['drag_deferred'] = js(host, text)
                assert 'Queue: Unverified' in entry['drag_deferred'] and 'deferred during drag' in entry['drag_deferred']
                assert js(host, 'queueAvailable'), 'Diagnostic changed queue availability during drag'
                entry['drag_png'] = screenshot(host, name + '-drag-deferred')
                js(host, 'finishDrag();undefined')
                entry['drag_finished'] = js(host, text)
                assert 'Queue: Observed' in entry['drag_finished'], 'Completed repaint did not establish receipt'
            report['phase'] = name + '_render_failure'
            js(host, "window.render=()=>{throw Error('spoke diagnostic render failure')};window.load=window.__originalLoad;window.__renderProbeDone=false;load().catch(()=>{}).finally(()=>window.__renderProbeDone=true);undefined")
            settle(lambda: js(host, 'window.__renderProbeDone'))
            entry['render_failure'] = js(host, text)
            assert 'Unverified' in entry['render_failure'] and 'display update failed' in entry['render_failure']
            entry['render_failure_png'] = screenshot(host, name + '-render-failure')
        report.update(phase='complete', passed=True,
                      claim_limit='Installed WebKit actual page/render receipts, retained reopen invalidation, actual read-only HTTP400/401 and injected render failures. No live Spoke restart, global keys, backend restart or historical-cause claim.')
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
