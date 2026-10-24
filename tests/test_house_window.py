"""Shared native House carrier, independent of the content it presents."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_mocked_main_scope_preserves_registered_native_house_class():
    result = subprocess.run([sys.executable, "-c", """
import importlib
import sys
from tests import conftest
original = importlib.import_module('spoke.house_window')
frameworks = conftest.mock_pyobjc.__wrapped__()
main = conftest.main_module.__wrapped__(next(frameworks))
try:
    next(main)
finally:
    main.close()
    next(frameworks, None)
assert sys.modules.get('spoke.house_window') is original, 'Native House module was discarded'
assert importlib.import_module('spoke.house_window').HouseWindowShell is original.HouseWindowShell
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_witness_rejects_missing_or_wrong_source(tmp_path):
    path = Path(__file__).with_name("native_popup_house_witness.py")
    assert path.exists(), "Native popup witness missing"
    spec = importlib.util.spec_from_file_location("popup_witness", path)
    witness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(witness)
    (tmp_path / "spoke").mkdir()
    for name in witness.SOURCES:
        (tmp_path / "spoke" / name).write_text("wrong revision")
    for expected in ({}, dict.fromkeys(witness.SOURCES, "old")):
        with pytest.raises(ValueError):
            witness.check_sources(tmp_path, expected)


def test_utility_windows_share_the_native_house_runner():
    assert importlib.util.find_spec("spoke.house_window"), "Shared native House runner missing"
    subprocess.run([sys.executable, "-c", """
from spoke.house_window import HouseWindowShell
from spoke.diaulos_switcher_overlay import DiaulosSwitcherOverlay
from spoke.monitor_window import MonitorWindow
from spoke.recording_history_window import RecordingHistoryWindow
for consumer in (DiaulosSwitcherOverlay, MonitorWindow, RecordingHistoryWindow):
    assert issubclass(consumer, HouseWindowShell)
    assert 'publishOpticalShell_' not in consumer.__dict__
    instance = consumer.alloc().init()
    for selector in ('publishOpticalShell:', 'opticalShellStep:', 'opticalShellDidPresent:'):
        assert instance.respondsToSelector_(selector)
"""], check=True, capture_output=True, text=True)


def test_borderless_teleporter_command_w_uses_group_close():
    subprocess.run([sys.executable, "-c", """
import AppKit as AK
import objc
from Foundation import NSObject
from spoke.diaulos_switcher_overlay import _SwitcherPanel
AK.NSApplication.sharedApplication()
class CloseOwner(NSObject):
    @objc.python_method
    def hide(self):
        self.closed = True
owner = CloseOwner.alloc().init()
owner.closed = False
panel = _SwitcherPanel.alloc().initWithContentRect_styleMask_backing_defer_(((0,0),(10,10)), 0, 2, False)
panel.setDelegate_(owner)
panel.performClose_(None)
assert owner.closed, 'Borderless panel bypassed group close'
assert not panel.isVisible()
"""], check=True, capture_output=True, text=True)
