"""Exercise native AppKit termination without windows or global input."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


_CHILD = r'''
import ast
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import sys
import threading
import time
import subprocess
from types import SimpleNamespace
from AppKit import NSApplication
from PyObjCTools import AppHelper

root, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
revision = sys.argv[4]
source = subprocess.check_output(["git", "show", f"{revision}:spoke/__main__.py"], cwd=root, text=True) if revision else (root / "spoke/__main__.py").read_text()
tree = ast.parse(source)
delegate_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpokeAppDelegate")
names = {"_record_history_delivery", "_drain_delivery_receipts", "_quit"}
methods = [n for n in delegate_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
cls = ast.ClassDef(name="Witness", bases=[], keywords=[], body=methods, decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
logger = logging.getLogger("native-shutdown-witness")
_DELIVERY_RECEIPT_LOCK = threading.Lock()
_record_runtime_phase = lambda *args, **kwargs: None
NSApp = NSApplication.sharedApplication()
NSApp.setActivationPolicy_(2)
exec(compile(module, str(root / "spoke/__main__.py"), "exec"))
delegate = Witness()
delegate._recording_history = None
delegate._detector = SimpleNamespace(uninstall=lambda: None)
delegate._handsfree = None
delegate._diaulos_switcher = None
delegate._menubar = None
delegate._close_clients = lambda: None
entered = threading.Event()

def write(capture_id, attempt_id, *, state, detail):
    entered.set()
    time.sleep(0.1)
    (output / state).write_text(f"{capture_id}:{attempt_id}")

delegate._audio_spool = SimpleNamespace(record_delivery=write)
history = {"capture_id": "capture", "attempt_id": "attempt"}
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
handlers = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in {"_handle_sigterm", "_cleanup_after_sigterm"}]
_LOCK_PATH = str(output / "absent-lock")
exec(compile(ast.fix_missing_locations(ast.Module(body=handlers, type_ignores=[])), "sigterm", "exec"))
if mode == "signal_during_submit":
    import signal
    signal.signal(signal.SIGTERM, _handle_sigterm)
    original_submit = ThreadPoolExecutor.submit
    interrupted = threading.Event()
    main_id = threading.get_ident()

    def diagnose_deadlock():
        interrupted.wait()
        time.sleep(0.1)
        frame = sys._current_frames().get(main_id)
        stack = []
        while frame is not None:
            stack.append(frame.f_code.co_name)
            frame = frame.f_back
        if "_drain_delivery_receipts" in stack and "_record_history_delivery" in stack:
            print(f"diagnosed signal re-entry deadlock: {stack}", flush=True)
            os._exit(93)

    threading.Thread(target=diagnose_deadlock, daemon=True).start()

    def submit_with_signal(self, *args, **kwargs):
        future = original_submit(self, *args, **kwargs)
        if not interrupted.is_set():
            interrupted.set()
            os.kill(os.getpid(), signal.SIGTERM)
        return future

    ThreadPoolExecutor.submit = submit_with_signal
delegate._record_history_delivery(history, "insert_requested")
delegate._record_history_delivery(history, "clipboard_restored")
assert entered.wait(5)
import objc
print(f"native AppKit route; PyObjC {objc.__version__}; revision={revision or 'working-tree'}; mode={mode}", flush=True)
if mode == "menu":
    delegate._quit()
else:
    if mode == "sigterm":
        _handle_sigterm(15, None)
    NSApp.run()
raise AssertionError("native termination unexpectedly returned")
'''


@pytest.mark.parametrize("mode", ["menu", "sigterm", "signal_during_submit"])
def test_native_termination_drains_accepted_delivery_receipts(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    revision = os.environ.get("SPOKE_SHUTDOWN_WITNESS_REVISION", "")
    report = {
        "route": "native AppKit termination, extracted production methods, isolated receipt writer",
        "python": sys.executable, "repo_root": str(root), "mode": mode,
        "revision": revision or None,
        "source_kind": "exact-revision" if revision else "working-tree",
        "failure_phase": "source_identity", "persisted": [],
    }
    def preserve():
        (tmp_path / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    preserve()
    try:
        if not revision:
            report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["failure_phase"] = "child_launch"
        preserve()
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, str(root), str(tmp_path), mode, revision],
            capture_output=True, text=True, check=False,
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        preserve()
        raise
    report.update(returncode=child.returncode, stdout=child.stdout, stderr=child.stderr,
                  failure_phase=None if child.returncode == 0 else "child_execution",
                  persisted=[name for name in ("insert_requested", "clipboard_restored") if (tmp_path / name).is_file()])
    preserve()
    assert child.returncode == 0, child.stdout + child.stderr
    assert (tmp_path / "insert_requested").is_file(), "native exit dropped accepted receipt"
    assert (tmp_path / "clipboard_restored").read_text() == "capture:attempt"


def test_native_witness_preserves_child_launch_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SPOKE_SHUTDOWN_WITNESS_REVISION", "6ccff3a05f4e3205fbc4cf9f9b960f5ae2510021")
    def fail(*args, **kwargs):
        raise OSError("child launch unavailable")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(OSError, match="child launch unavailable"):
        test_native_termination_drains_accepted_delivery_receipts(tmp_path, "menu")
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["failure_phase"] == "child_launch"
    assert report["persisted"] == []
    assert report["error"] == "OSError: child launch unavailable"


def test_native_witness_preserves_source_identity_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("SPOKE_SHUTDOWN_WITNESS_REVISION", raising=False)
    def fail(*args, **kwargs):
        raise OSError("source identity unavailable")

    monkeypatch.setattr(subprocess, "check_output", fail)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    with pytest.raises(OSError, match="source identity unavailable"):
        test_native_termination_drains_accepted_delivery_receipts(tmp_path, "menu")
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["failure_phase"] == "source_identity"
    assert report["revision"] is None
