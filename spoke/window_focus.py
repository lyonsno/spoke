"""Remember a summon origin without taking focus back from a later app switch."""

import ctypes
import logging
import os

import AppKit as AK

from .focus_check import _cf, _cfstr, _hi

logger = logging.getLogger(__name__)
_CF_TRUE = ctypes.c_void_p.in_dll(_cf, 'kCFBooleanTrue').value if _cf is not None else None

if _hi is not None:
    _hi.AXUIElementCreateApplication.restype = ctypes.c_void_p
    _hi.AXUIElementCreateApplication.argtypes = [ctypes.c_int]
    _hi.AXUIElementPerformAction.restype = ctypes.c_int32
    _hi.AXUIElementPerformAction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _hi.AXUIElementSetAttributeValue.restype = ctypes.c_int32
    _hi.AXUIElementSetAttributeValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


class WindowFocus:
    def __init__(self, app, ax_app=None, ax_window=None, local_window=None):
        self.app = app
        self._ax_app, self._ax_window = ax_app, ax_window
        self._local_window = local_window

    def __del__(self):
        values = self._ax_window, self._ax_app
        self._ax_window = self._ax_app = None
        if _cf is not None:
            for value in values:
                if value:
                    _cf.CFRelease(value)

    @classmethod
    def capture(cls, app=None):
        app = app if app is not None else AK.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        if app.processIdentifier() == os.getpid():
            native_app = AK.NSApp()
            window = native_app.keyWindow() if native_app is not None else None
            return cls(app, local_window=window) if window is not None else None
        target = cls(app)
        if _hi is not None and _cf is not None:
            target._ax_app = _hi.AXUIElementCreateApplication(app.processIdentifier())
            attribute = _cfstr(b'AXFocusedWindow')
            if target._ax_app and attribute:
                try:
                    value = ctypes.c_void_p()
                    error = _hi.AXUIElementCopyAttributeValue(target._ax_app, attribute, ctypes.byref(value))
                    if error == 0:
                        target._ax_window = value.value
                finally:
                    _cf.CFRelease(attribute)
            elif attribute:
                _cf.CFRelease(attribute)
        return target

    def restore(self):
        current = AK.NSWorkspace.sharedWorkspace().frontmostApplication()
        if current is None or current.processIdentifier() != os.getpid():
            return False
        if self._local_window is not None:
            if not self._local_window.isVisible():
                return False
            self._local_window.makeKeyAndOrderFront_(None)
            return True
        if not self.app.activateWithOptions_(1 << 1):
            return False
        if self._ax_window:
            action, attribute = _cfstr(b'AXRaise'), _cfstr(b'AXMain')
            try:
                if action:
                    error = _hi.AXUIElementPerformAction(self._ax_window, action)
                    if error:
                        logger.debug('Prior window raise unavailable: AX error %s', error)
                if attribute:
                    error = _hi.AXUIElementSetAttributeValue(self._ax_window, attribute, _CF_TRUE)
                    if error:
                        logger.debug('Prior window main focus unavailable: AX error %s', error)
            finally:
                for value in (action, attribute):
                    if value:
                        _cf.CFRelease(value)
        return True


def restore_window_focus(owner, window, was_key=None):
    # AppKit may have cleared the closing window before windowWillClose arrives.
    # A different key Spoke window or external app owns any newer focus decision.
    if was_key is False:
        return False
    if was_key is None:
        native_app = AK.NSApp()
        if native_app is None:
            return False
        key = native_app.keyWindow()
        if key is not None and key != window:
            return False
    target = getattr(owner, '_focus_return', None)
    if target is None:
        app = getattr(owner, '_previous_app', None)
        target = WindowFocus(app) if app is not None else None
    return target.restore() if target is not None else False
