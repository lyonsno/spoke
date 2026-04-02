"""Detect whether a text input element is focused via the Accessibility API.

Uses ctypes to call AXUIElementCopyAttributeValue directly, avoiding a
pyobjc-framework-ApplicationServices dependency. Requires Accessibility
permissions (already granted for the CGEventTap).

Fail-open: if the API call fails for any reason, has_focused_text_input()
returns True so the normal paste path is attempted rather than falsely
triggering recovery mode.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

logger = logging.getLogger(__name__)

# Roles where paste definitely won't work — these are non-text UI elements
# that happen to be focused. Everything else (including AXWindow, AXGroup,
# AXWebArea, etc.) is treated as potentially pasteable because many apps
# (terminals, Electron apps, etc.) don't report standard text roles.
_NON_TEXT_ROLES = frozenset({
    "AXButton",
    "AXMenu",
    "AXMenuItem",
    "AXMenuBar",
    "AXMenuBarItem",
    "AXToolbar",
    "AXImage",
    "AXStaticText",
    "AXProgressIndicator",
})

# ── ctypes setup (loaded once at module level) ──────────────────

_cf_path = ctypes.util.find_library("CoreFoundation")
_hi_path = "/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/HIServices.framework/Versions/A/HIServices"

_cf = ctypes.cdll.LoadLibrary(_cf_path) if _cf_path else None

try:
    _hi = ctypes.cdll.LoadLibrary(_hi_path)
except OSError:
    _hi = None

# AX error codes
_kAXErrorSuccess = 0

# CFString encoding
_kCFStringEncodingUTF8 = 0x08000100

# Configure signatures once at module level — argtypes are critical
# to avoid segfaults from incorrect argument marshalling.
if _hi is not None:
    _hi.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
    _hi.AXUIElementCreateSystemWide.argtypes = []

    _hi.AXUIElementCopyAttributeValue.restype = ctypes.c_int32
    _hi.AXUIElementCopyAttributeValue.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    ]

if _cf is not None:
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32
    ]

    _cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
    _cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    _cf.CFStringGetLength.restype = ctypes.c_long
    _cf.CFStringGetLength.argtypes = [ctypes.c_void_p]

    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32
    ]

    _cf.CFRelease.restype = None
    _cf.CFRelease.argtypes = [ctypes.c_void_p]


def _get_focused_role() -> str | None:
    """Return the AXRole of the system-wide focused UI element, or None.

    Returns None if no element is focused or the role cannot be determined.
    Raises on ctypes/API failures (caller catches).
    """
    if _cf is None or _hi is None:
        return None

    system_wide = _hi.AXUIElementCreateSystemWide()
    if not system_wide:
        return None

    try:
        attr_name = _cfstr(b"AXFocusedUIElement")
        if not attr_name:
            return None

        try:
            value = ctypes.c_void_p()
            err = _hi.AXUIElementCopyAttributeValue(
                system_wide, attr_name, ctypes.byref(value)
            )
            if err != _kAXErrorSuccess or not value.value:
                return None

            try:
                role_attr = _cfstr(b"AXRole")
                if not role_attr:
                    return None
                try:
                    role_value = ctypes.c_void_p()
                    err2 = _hi.AXUIElementCopyAttributeValue(
                        value, role_attr, ctypes.byref(role_value)
                    )
                    if err2 != _kAXErrorSuccess or not role_value.value:
                        return None
                    try:
                        return _cfstr_to_python(role_value)
                    finally:
                        _cf.CFRelease(role_value)
                finally:
                    _cf.CFRelease(role_attr)
            finally:
                _cf.CFRelease(value)
        finally:
            _cf.CFRelease(attr_name)
    finally:
        _cf.CFRelease(system_wide)


def has_focused_text_input() -> bool:
    """Return True if a text input element is currently focused.

    Fail-open: returns True on any error so the normal paste path
    is attempted rather than falsely entering recovery mode.
    """
    try:
        role = _get_focused_role()
    except Exception:
        logger.debug("Focus check failed — falling back to paste", exc_info=True)
        return True

    if role is None:
        logger.info("Focus check: no focused element — recovery mode")
        return False

    if role in _NON_TEXT_ROLES:
        logger.info("Focus check: role=%s — non-text element, recovery mode", role)
        return False

    logger.info("Focus check: role=%s — treating as pasteable", role)
    return True


def focused_text_contains(expected: str) -> bool | None:
    """Return whether the focused element's AXValue contains the expected text.

    Returns True on a positive AXValue match, False when a readable focused text
    value is present but does not contain the expected text, and None when the
    focused value is unavailable or cannot be trusted for verification.
    """
    expected_norm = " ".join((expected or "").split())
    if not expected_norm:
        return None

    try:
        value = _get_focused_value()
    except Exception:
        logger.debug("Focused value lookup failed — falling back to OCR", exc_info=True)
        return None

    if not value:
        return None

    value_norm = " ".join(value.split())
    return expected_norm in value_norm


# ── CFString helpers ────────────────────────────────────────────

def _cfstr(s: bytes) -> ctypes.c_void_p | None:
    """Create a CFStringRef from bytes. Caller must CFRelease."""
    if _cf is None:
        return None
    ref = _cf.CFStringCreateWithCString(None, s, _kCFStringEncodingUTF8)
    return ref if ref else None


def _cfstr_to_python(cf_string: ctypes.c_void_p) -> str | None:
    """Convert a CFStringRef to a Python string."""
    if _cf is None or not cf_string:
        return None
    ptr = _cf.CFStringGetCStringPtr(cf_string, _kCFStringEncodingUTF8)
    if ptr:
        return ptr.decode("utf-8")

    # Fallback: CFStringGetCString with buffer
    length = _cf.CFStringGetLength(cf_string)
    buf_size = length * 4 + 1  # UTF-8 worst case
    buf = ctypes.create_string_buffer(buf_size)
    if _cf.CFStringGetCString(cf_string, buf, buf_size, _kCFStringEncodingUTF8):
        return buf.value.decode("utf-8")
    return None


def _get_focused_value() -> str | None:
    """Return the AXValue string for the focused element, when available."""
    if _cf is None or _hi is None:
        return None

    role = _get_focused_role()
    if role is None or role in _NON_TEXT_ROLES:
        return None

    system_wide = _hi.AXUIElementCreateSystemWide()
    if not system_wide:
        return None

    try:
        focus_attr = _cfstr(b"AXFocusedUIElement")
        if not focus_attr:
            return None
        try:
            value = ctypes.c_void_p()
            err = _hi.AXUIElementCopyAttributeValue(
                system_wide, focus_attr, ctypes.byref(value)
            )
            if err != _kAXErrorSuccess or not value.value:
                return None
            try:
                value_attr = _cfstr(b"AXValue")
                if not value_attr:
                    return None
                try:
                    value_ref = ctypes.c_void_p()
                    err2 = _hi.AXUIElementCopyAttributeValue(
                        value, value_attr, ctypes.byref(value_ref)
                    )
                    if err2 != _kAXErrorSuccess or not value_ref.value:
                        return None
                    try:
                        return _cfstr_to_python(value_ref)
                    finally:
                        _cf.CFRelease(value_ref)
                finally:
                    _cf.CFRelease(value_attr)
            finally:
                _cf.CFRelease(value)
        finally:
            _cf.CFRelease(focus_attr)
    finally:
        _cf.CFRelease(system_wide)
