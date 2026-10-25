"""Shared native House carrier, independent of the content it presents."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("contract", ["background", "ordering", "shadow", "body_mask"])
def test_native_window_dismissal_has_one_visual_owner(contract):
    result = subprocess.run([sys.executable, "-c", """
import sys
from types import SimpleNamespace
import AppKit as AK
from Quartz import CGPathGetBoundingBox, CALayer
from spoke.house_window import HouseWindowShell
from spoke.house_optical_primitive import (
    house_dismiss_materialization_fill_state,
    house_dismiss_text_collapse_state,
)

AK.NSApplication.sharedApplication()
window = AK.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
    ((0, 0), (300, 200)), 15, 2, False)
background = window.backgroundColor().CGColor()
host = HouseWindowShell.alloc().init()
host._init_window_shell(window, 'test.popup')
host._delegate = SimpleNamespace(_overlay_compositor_registry=object())
host.visible = True
try:
    contract = sys.argv[1]
    if contract == 'background':
        assert window.backgroundColor().alphaComponent() == 0, 'Unmasked native background remains opaque'
        assert host._shell_foreground.layer().backgroundColor() == background, 'Resting background was lost'
    elif contract == 'ordering':
        # Teleporter prepares entrance without the standard-window initializer.
        window.setAnimationBehavior_(AK.NSWindowAnimationBehaviorDefault)
        host._prepare_shell_entrance()
        assert window.animationBehavior() == AK.NSWindowAnimationBehaviorNone, 'AppKit can independently animate House ordering'
    elif contract == 'shadow':
        host._prepare_shell_entrance()
        host._shell_transition.rest()
        host._shell_waiting = False
        host._shell_registered = True
        host._shell_final = {}
        window.setHasShadow_(True)
        assert host._dismiss_window_shell(True)
        assert not window.hasShadow(), 'Resting rectangle shadow survives the closing body'
    else:
        height = host._shell_foreground.bounds().size.height
        for progress in (1.0, 0.8, 0.6, 0.4, 0.0):
            body = house_dismiss_materialization_fill_state(progress)
            text = house_dismiss_text_collapse_state(progress)
            host._set_shell_planes(body, text)
            mask = CGPathGetBoundingBox(host._shell_text_mask.path())
            expected = height * min(body['height_frac'], text['height_frac'])
            if expected > 0:
                assert abs(mask.size.height - expected) < 1e-6, ('Native window exceeds closing body', progress, mask, expected)
            else:
                assert mask.size.height == 0
            assert abs(host._shell_foreground.layer().opacity() - min(body['opacity'], text['alpha'])) < 1e-6
        # Teleporter owns separate planes: preserve its tuned text choreography.
        host._shell_body_layer = CALayer.layer()
        body = house_dismiss_materialization_fill_state(0.8)
        text = house_dismiss_text_collapse_state(0.8)
        host._set_shell_planes(body, text)
        assert abs(CGPathGetBoundingBox(host._shell_text_mask.path()).size.height - height * text['height_frac']) < 1e-6
        assert abs(host._shell_foreground.layer().opacity() - text['alpha']) < 1e-6
finally:
    host._stop_shell_ticks()
assert not window.isVisible(), 'Test unexpectedly exposed a native window'
""", contract], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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


def test_teleporter_hide_releases_native_shadow_during_house_tail():
    result = subprocess.run([sys.executable, "-c", """
from types import SimpleNamespace
import AppKit as AK
from spoke.diaulos_switcher_overlay import DiaulosSwitcherOverlay
AK.NSApplication.sharedApplication()
host = DiaulosSwitcherOverlay.alloc().initWithDelegate_(
    SimpleNamespace(_overlay_compositor_registry=object()))
host.setup()
try:
    host.visible = True
    host._prepare_shell_entrance()
    host._shell_transition.rest()
    host._shell_waiting = False
    host._shell_registered = True
    host._shell_final = {}
    host._panel.setHasShadow_(True)
    assert host.hide(restore_previous=False)
    assert not host._panel.hasShadow(), 'Teleporter dismissal retains native shadow'
    assert host._shell_transition.phase == 'dismiss'
    assert host._shell_tick_timer is not None
    assert not host.visible and host._panel.ignoresMouseEvents()
    assert not host._panel.isVisible(), 'Test unexpectedly exposed a native window'
finally:
    host.cleanup()
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
