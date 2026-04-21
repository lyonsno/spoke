"""Shared fixtures for spoke tests.

Stubs out PyObjC/Quartz/AppKit so tests run without macOS GUI runtime.

SAFETY: Tests must never touch the production lock file or heartbeat file.
The env vars below redirect spoke to temp paths so that running pytest
cannot kill a live spoke instance via the single-instance guard.
"""

import importlib
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, Mock

import pytest

# ── Test isolation: redirect runtime state files to temp paths ──
# These must be set before any spoke module is imported, because the
# paths are evaluated at import time as module-level constants.
_test_runtime_dir = tempfile.mkdtemp(prefix="spoke-test-")
os.environ["SPOKE_LOCK_PATH"] = os.path.join(_test_runtime_dir, ".spoke.lock")
os.environ["SPOKE_HEARTBEAT_PATH"] = os.path.join(_test_runtime_dir, ".spoke-heartbeat.json")


def _make_fake_quartz():
    """Create a fake Quartz module with all constants and functions used."""
    q = types.ModuleType("Quartz")
    # Constants — use actual macOS values where it matters for logic
    q.kCGEventFlagsChanged = 12
    q.kCGEventKeyDown = 10
    q.kCGEventKeyUp = 11
    q.kCGEventTapDisabledByTimeout = 0xFFFFFFFE
    q.kCGEventTapDisabledByUserInput = 0xFFFFFFFF
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
    q.CGEventGetTimestamp = MagicMock(return_value=0)
    q.CGEventSourceKeyState = MagicMock(return_value=False)
    q.CGEventCreateKeyboardEvent = MagicMock(return_value=MagicMock())
    q.CGEventSetFlags = MagicMock()
    q.CGEventPost = MagicMock()
    q.CFMachPortCreateRunLoopSource = MagicMock(return_value=MagicMock())
    q.CFRunLoopAddSource = MagicMock()
    q.CFRunLoopGetMain = MagicMock(return_value=MagicMock())
    q.CGRectNull = MagicMock(name="CGRectNull")
    q.CGRectInfinite = MagicMock(name="CGRectInfinite")
    q.CGWindowListCopyWindowInfo = MagicMock(return_value=[])
    q.CGWindowListCreateImage = MagicMock(return_value=None)
    q.CGImageGetWidth = MagicMock(return_value=0)
    q.CGImageGetHeight = MagicMock(return_value=0)
    q.kCGWindowImageBoundsIgnoreFraming = 1
    q.kCGWindowListExcludeDesktopElements = 1 << 4
    q.kCGWindowListOptionIncludingWindow = 1 << 0
    q.kCGWindowListOptionOnScreenOnly = 1 << 1
    q.kCGNullWindowID = 0

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
    q.kCGEventSourceStateCombinedSessionState = 0
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
        "NSMutableAttributedString",
        "NSScrollView",
        "NSStatusBar",
        "NSShadow",
        "NSTextView",
        "NSView",
        "NSWindow",
        "NSRunningApplication",
    ]:
        setattr(a, name, MagicMock())
    a.NSApplicationActivationPolicyAccessory = 2
    a.NSForegroundColorAttributeName = "NSForegroundColor"
    a.NSFontAttributeName = "NSFont"
    a.NSShadowAttributeName = "NSShadow"
    a.NSParagraphStyleAttributeName = "NSParagraphStyle"
    a.NSVariableStatusItemLength = -1
    a.NSPasteboardTypeString = "public.utf8-plain-text"
    a.NSBackingStoreBuffered = 2
    a.NSFont = MagicMock()
    a.NSMutableParagraphStyle = MagicMock()
    a.NSTextField = MagicMock()
    a.NSPanel = MagicMock()
    a.NSWorkspace = MagicMock()
    a.NSEvent = MagicMock()
    a.NSEvent.addLocalMonitorForEventsMatchingMask_handler_ = MagicMock(return_value=MagicMock())
    a.NSEvent.removeMonitor_ = MagicMock()
    a.NSWindowStyleMaskNonactivatingPanel = 128
    a.NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0
    a.NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8
    a.NSWindowCollectionBehaviorStationary = 1 << 4
    return a


def _make_fake_objc():
    """Create a fake objc module."""
    o = types.ModuleType("objc")
    o.super = super  # Python super is fine for testing
    o.IBAction = lambda func: func
    o.arch = "arm64"
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

    prefixes = ("objc", "Quartz", "Foundation", "AppKit", "PyObjCTools")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }

    for name in list(saved):
        sys.modules.pop(name, None)

    # Install fakes
    sys.modules.update(fakes)

    yield fakes

    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def input_tap_module(mock_pyobjc):
    """Import spoke.input_tap with mocked PyObjC."""
    # Remove cached module if present
    sys.modules.pop("spoke.input_tap", None)
    mod = importlib.import_module("spoke.input_tap")
    yield mod
    sys.modules.pop("spoke.input_tap", None)


@pytest.fixture
def inject_module(mock_pyobjc):
    """Import spoke.inject with mocked PyObjC."""
    sys.modules.pop("spoke.inject", None)
    mod = importlib.import_module("spoke.inject")
    yield mod
    sys.modules.pop("spoke.inject", None)


@pytest.fixture
def menubar_module(mock_pyobjc):
    """Import spoke.menubar with mocked PyObjC."""
    sys.modules.pop("spoke.menubar", None)
    mod = importlib.import_module("spoke.menubar")
    yield mod
    sys.modules.pop("spoke.menubar", None)


@pytest.fixture
def main_module(mock_pyobjc):
    """Import spoke.__main__ with mocked PyObjC and sub-modules."""
    # Clear all cached spoke modules so they re-import against fakes
    for name in list(sys.modules):
        if name.startswith("spoke."):
            sys.modules.pop(name, None)

    # Pre-install a fake focus_check module to avoid loading real ctypes/AX APIs
    fake_focus_check = types.ModuleType("spoke.focus_check")
    fake_focus_check.has_focused_text_input = MagicMock(return_value=True)
    fake_focus_check.focused_text_contains = MagicMock(return_value=None)
    fake_focus_check._get_focused_role = MagicMock(return_value="AXTextField")
    sys.modules["spoke.focus_check"] = fake_focus_check

    mod = importlib.import_module("spoke.__main__")
    yield mod
    for name in list(sys.modules):
        if name.startswith("spoke."):
            sys.modules.pop(name, None)
