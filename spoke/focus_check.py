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

# Roles that accept pasted text.
# AXWebArea is included because browsers report it as the focused element even
# when a <textarea> or <input> is focused. This means pasting into a
# non-editable web page will still fail silently (same as current behavior),
# but the alternative — falsely triggering recovery in browsers — is worse.
_TEXT_INPUT_ROLES = frozenset({
    "AXTextField",
    "AXTextArea",
    "AXComboBox",
    "AXSearchField",
    "AXWebArea",
})

# ── ctypes setup (loaded once at module level) ──────────────────

_cf_path = ctypes.util.find_library("CoreFoundation")
_ax_path = ctypes.util.find_library("ApplicationServices")

_cf = ctypes.cdll.LoadLibrary(_cf_path) if _cf_path else None
_ax = ctypes.cdll.LoadLibrary(_ax_path) if _ax_path else None

# AX error codes
_kAXErrorSuccess = 0

# CFString encoding
_kCFStringEncodingUTF8 = 0x08000100

# Configure restype once at module level (not on every call)
if _ax is not None:
    _ax.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
    _ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int32

if _cf is not None:
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
    _cf.CFStringGetCString.restype = ctypes.c_bool


def _get_focused_role() -> str | None:
    """Return the AXRole of the system-wide focused UI element, or None.

    Returns None if no element is focused or the role cannot be determined.
    Raises on ctypes/API failures (caller catches).
    """
    if _cf is None or _ax is None:
        return None

    system_wide = _ax.AXUIElementCreateSystemWide()
    if not system_wide:
        return None

    try:
        attr_name = _cfstr("AXFocusedUIElement")
        if not attr_name:
            return None

        try:
            value = ctypes.c_void_p()
            err = _ax.AXUIElementCopyAttributeValue(
                system_wide, attr_name, ctypes.byref(value)
            )
            if err != _kAXErrorSuccess or not value.value:
                return None

            try:
                role_attr = _cfstr("AXRole")
                if not role_attr:
                    return None
                try:
                    role_value = ctypes.c_void_p()
                    err2 = _ax.AXUIElementCopyAttributeValue(
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
        return False

    return role in _TEXT_INPUT_ROLES


# ── CFString helpers ────────────────────────────────────────────

def _cfstr(s: str) -> ctypes.c_void_p | None:
    """Create a CFStringRef from a Python string. Caller must CFRelease."""
    if _cf is None:
        return None
    ref = _cf.CFStringCreateWithCString(
        None, s.encode("utf-8"), _kCFStringEncodingUTF8
    )
    return ref if ref else None


def _cfstr_to_python(cf_string: ctypes.c_void_p) -> str | None:
    """Convert a CFStringRef to a Python string."""
    if _cf is None or not cf_string.value:
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
