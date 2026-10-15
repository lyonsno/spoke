import ctypes
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from spoke import window_focus as mod


def workspace(monkeypatch, app):
    monkeypatch.setattr(mod.AK, 'NSWorkspace', SimpleNamespace(
        sharedWorkspace=lambda: SimpleNamespace(frontmostApplication=lambda: app)))


def test_capture_retains_exact_window_and_restores_after_activation(monkeypatch):
    app = MagicMock()
    app.processIdentifier.return_value = 12345
    workspace(monkeypatch, app)
    hi, cf = MagicMock(), MagicMock()
    hi.AXUIElementCreateApplication.return_value = 101
    def copy(element, attribute, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 202
        return 0
    hi.AXUIElementCopyAttributeValue.side_effect = copy
    monkeypatch.setattr(mod, '_hi', hi)
    monkeypatch.setattr(mod, '_cf', cf)
    monkeypatch.setattr(mod, '_cfstr', lambda value: value)
    monkeypatch.setattr(mod, '_CF_TRUE', 303, raising=False)
    target = mod.WindowFocus.capture()
    try:
        assert target._ax_window == 202
        current = MagicMock()
        current.processIdentifier.return_value = os.getpid()
        workspace(monkeypatch, current)
        events = []
        app.activateWithOptions_.side_effect = lambda options: events.append('activate') or True
        hi.AXUIElementPerformAction.side_effect = lambda *args: events.append('raise') or 0
        assert target.restore() is True
        assert events == ['activate', 'raise']
        hi.AXUIElementPerformAction.assert_called_once_with(202, b'AXRaise')
        hi.AXUIElementSetAttributeValue.assert_called_once_with(202, b'AXMain', 303)
    finally:
        target.__del__()
    assert target._ax_window is None and target._ax_app is None
    assert sum(call.args == (202,) for call in cf.CFRelease.call_args_list) == 1


def test_return_does_not_steal_focus_after_external_switch(monkeypatch):
    current, prior = MagicMock(), MagicMock()
    current.processIdentifier.return_value = os.getpid() + 1
    workspace(monkeypatch, current)
    assert mod.WindowFocus(prior).restore() is False
    prior.activateWithOptions_.assert_not_called()


def test_capture_retains_another_spoke_window_as_the_immediate_origin(monkeypatch):
    current = MagicMock()
    current.processIdentifier.return_value = os.getpid()
    workspace(monkeypatch, current)
    native = MagicMock()
    monkeypatch.setattr(mod.AK, 'NSApp', lambda: native)
    target = mod.WindowFocus.capture()
    assert target._local_window is native.keyWindow.return_value
    assert target.restore() is True
    native.keyWindow.return_value.makeKeyAndOrderFront_.assert_called_once_with(None)


def test_other_spoke_window_keeps_its_focus(monkeypatch):
    native, closing, target = MagicMock(), MagicMock(), MagicMock()
    native.keyWindow.return_value = MagicMock()
    monkeypatch.setattr(mod.AK, 'NSApp', lambda: native)
    assert mod.restore_window_focus(SimpleNamespace(_focus_return=target), closing) is False
    target.restore.assert_not_called()


def test_preclose_owner_survives_automatic_key_transfer(monkeypatch):
    native, closing, target = MagicMock(), MagicMock(), MagicMock()
    native.keyWindow.return_value = MagicMock()
    monkeypatch.setattr(mod.AK, 'NSApp', lambda: native)
    owner = SimpleNamespace(_focus_return=target)
    mod.restore_window_focus(owner, closing, was_key=True)
    target.restore.assert_called_once_with()
    target.reset_mock()
    native.keyWindow.return_value = None
    assert mod.restore_window_focus(owner, closing, was_key=False) is False
    target.restore.assert_not_called()


def test_missing_ax_window_retains_application_fallback(monkeypatch):
    app = MagicMock()
    app.processIdentifier.return_value = 12345
    workspace(monkeypatch, app)
    monkeypatch.setattr(mod, '_hi', None)
    target = mod.WindowFocus.capture()
    current = MagicMock()
    current.processIdentifier.return_value = os.getpid()
    workspace(monkeypatch, current)
    assert target.restore() is True
    app.activateWithOptions_.assert_called_once_with(2)
