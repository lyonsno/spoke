"""Shared fixtures for donttype tests.

Stubs out PyObjC/Quartz/AppKit so tests run without macOS GUI runtime.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, Mock

import pytest


def _make_fake_quartz():
    """Create a fake Quartz module with all constants and functions used."""
    q = types.ModuleType("Quartz")
    # Constants — use actual macOS values where it matters for logic
    q.kCGEventKeyDown = 10
    q.kCGEventKeyUp = 11
    q.kCGEventFlagMaskCommand = 0x00100000
    q.kCGEventFlagMaskControl = 0x00040000
    q.kCGEventFlagMaskAlternate = 0x00080000
    q.kCGEventFlagMaskShift = 0x00020000
    q.kCGKeyboardEventKeycode = 9
    q.kCGSessionEventTap = 1
    q.kCGHeadInsertEventTap = 0
    q.kCGHIDEventTap = 0
    q.kCGEventTapOptionDefault = 0
    q.kCFRunLoopCommonModes = "kCFRunLoopCommonModes"
    q.kCGEventFlagMaskCommand = 0x00100000

    # Functions — all mocked
    q.CGEventTapCreate = MagicMock(return_value=MagicMock())  # non-None = success
    q.CGEventTapEnable = MagicMock()
    q.CGEventMaskBit = lambda x: 1 << x
    q.CGEventGetIntegerValueField = MagicMock(return_value=0)
    q.CGEventGetFlags = MagicMock(return_value=0)
    q.CGEventCreateKeyboardEvent = MagicMock(return_value=MagicMock())
    q.CGEventSetFlags = MagicMock()
    q.CGEventPost = MagicMock()
    q.CFMachPortCreateRunLoopSource = MagicMock(return_value=MagicMock())
    q.CFRunLoopAddSource = MagicMock()
    q.CFRunLoopGetMain = MagicMock(return_value=MagicMock())

    # Glow module (Core Animation / Quartz)
    q.CABasicAnimation = MagicMock()
    q.CAGradientLayer = MagicMock()
    q.CALayer = MagicMock()
    q.CAMediaTimingFunction = MagicMock()
    q.CAShapeLayer = MagicMock()
    q.kCAGravityCenter = "center"
    q.CGPathCreateWithRoundedRect = MagicMock(return_value=MagicMock())
    q.CGPathCreateMutable = MagicMock(return_value=MagicMock())
    q.CGPathCreateMutableCopy = MagicMock(return_value=MagicMock())
    q.CGPathMoveToPoint = MagicMock()
    q.CGPathAddArcToPoint = MagicMock()
    q.CGPathCloseSubpath = MagicMock()
    q.CGPathAddPath = MagicMock()
    q.CGAffineTransformIdentity = MagicMock()
    q.kCAFillRuleEvenOdd = "even-odd"
    return q


def _make_fake_foundation():
    """Create a fake Foundation module."""
    f = types.ModuleType("Foundation")
    f.NSObject = type("NSObject", (), {
        "alloc": classmethod(lambda cls: cls()),
        "init": lambda self: self,
    })
    f.NSMakeRect = MagicMock(return_value=MagicMock())
    f.NSNumber = MagicMock()
    f.NSTimer = MagicMock()
    f.NSData = MagicMock()
    f.NSRunLoop = MagicMock()
    f.NSDefaultRunLoopMode = "NSDefaultRunLoopMode"
    return f


def _make_fake_appkit():
    """Create a fake AppKit module."""
    a = types.ModuleType("AppKit")
    for name in [
        "NSApp",
        "NSAlert",
        "NSApplication",
        "NSBezierPath",
        "NSColor",
        "NSImage",
        "NSMenu",
        "NSMenuItem",
        "NSPasteboard",
        "NSPasteboardItem",
        "NSScreen",
        "NSScrollView",
        "NSStatusBar",
        "NSTextView",
        "NSView",
        "NSWindow",
    ]:
        setattr(a, name, MagicMock())
    a.NSApplicationActivationPolicyAccessory = 2
    a.NSVariableStatusItemLength = -1
    a.NSPasteboardTypeString = "public.utf8-plain-text"
    a.NSBackingStoreBuffered = 2
    a.NSFont = MagicMock()
    a.NSMutableParagraphStyle = MagicMock()
    a.NSTextField = MagicMock()
    a.NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0
    a.NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8
    a.NSWindowCollectionBehaviorStationary = 1 << 4
    return a


def _make_fake_objc():
    """Create a fake objc module."""
    o = types.ModuleType("objc")
    o.super = super  # Python super is fine for testing
    o.IBAction = lambda func: func
    return o


@pytest.fixture
def mock_pyobjc():
    """Install fake PyObjC modules and yield them. Restores originals on teardown."""
    fakes = {
        "objc": _make_fake_objc(),
        "Quartz": _make_fake_quartz(),
        "Foundation": _make_fake_foundation(),
        "AppKit": _make_fake_appkit(),
        "PyObjCTools": types.ModuleType("PyObjCTools"),
    }
    fakes["PyObjCTools"].AppHelper = MagicMock()

    # Also need to handle sub-frameworks that PyObjC sometimes imports
    fakes["Quartz.CoreGraphics"] = fakes["Quartz"]

    # Save originals
    saved = {}
    for name in fakes:
        saved[name] = sys.modules.get(name)

    # Install fakes
    sys.modules.update(fakes)

    yield fakes

    # Restore
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def input_tap_module(mock_pyobjc):
    """Import donttype.input_tap with mocked PyObjC."""
    # Remove cached module if present
    sys.modules.pop("donttype.input_tap", None)
    mod = importlib.import_module("donttype.input_tap")
    yield mod
    sys.modules.pop("donttype.input_tap", None)


@pytest.fixture
def inject_module(mock_pyobjc):
    """Import donttype.inject with mocked PyObjC."""
    sys.modules.pop("donttype.inject", None)
    mod = importlib.import_module("donttype.inject")
    yield mod
    sys.modules.pop("donttype.inject", None)


@pytest.fixture
def menubar_module(mock_pyobjc):
    """Import donttype.menubar with mocked PyObjC."""
    sys.modules.pop("donttype.menubar", None)
    mod = importlib.import_module("donttype.menubar")
    yield mod
    sys.modules.pop("donttype.menubar", None)


@pytest.fixture
def main_module(mock_pyobjc):
    """Import donttype.__main__ with mocked PyObjC and sub-modules."""
    # Clear all cached donttype modules so they re-import against fakes
    for name in list(sys.modules):
        if name.startswith("donttype."):
            sys.modules.pop(name, None)
    mod = importlib.import_module("donttype.__main__")
    yield mod
    for name in list(sys.modules):
        if name.startswith("donttype."):
            sys.modules.pop(name, None)
