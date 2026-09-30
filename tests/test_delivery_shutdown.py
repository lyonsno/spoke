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
delegate._record_history_delivery(history, "insert_requested")
delegate._record_history_delivery(history, "clipboard_restored")
assert entered.wait(5)
import objc
print(f"native AppKit route; PyObjC {objc.__version__}; revision={revision or 'working-tree'}; mode={mode}", flush=True)
if mode == "menu":
    delegate._quit()
else:
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handler = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_sigterm")
    _LOCK_PATH = str(output / "absent-lock")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[])), "sigterm", "exec"))
    _handle_sigterm(15, None)
raise AssertionError("native termination unexpectedly returned")
'''


@pytest.mark.parametrize("mode", ["menu", "sigterm"])
def test_native_termination_drains_accepted_delivery_receipts(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    revision = os.environ.get("SPOKE_SHUTDOWN_WITNESS_REVISION", "")
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(root), str(tmp_path), mode, revision],
        capture_output=True, text=True, check=False,
    )
    (tmp_path / "result.json").write_text(json.dumps({
        "route": "native AppKit termination, extracted production methods, isolated receipt writer",
        "python": sys.executable, "repo_root": str(root), "mode": mode,
        "revision": revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_kind": "exact-revision" if revision else "working-tree",
        "returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr,
        "persisted": [name for name in ("insert_requested", "clipboard_restored") if (tmp_path / name).is_file()],
    }, indent=2) + "\n")
    assert child.returncode == 0, child.stdout + child.stderr
    assert (tmp_path / "insert_requested").is_file(), "native exit dropped accepted receipt"
    assert (tmp_path / "clipboard_restored").read_text() == "capture:attempt"
