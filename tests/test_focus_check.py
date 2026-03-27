"""Tests for AX-based focused text input detection.

Mocks the ctypes layer since tests run without macOS GUI runtime.
"""

import ctypes
import importlib
import sys
import types
from unittest.mock import MagicMock, patch


def _import_focus_check():
    """Import spoke.focus_check, clearing cache first."""
    sys.modules.pop("spoke.focus_check", None)
    return importlib.import_module("spoke.focus_check")


class TestHasFocusedTextInput:
    """Test has_focused_text_input() detection logic."""

    def test_returns_true_when_text_field_focused(self):
        """AXTextField role → text input available."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXTextField"):
            assert mod.has_focused_text_input() is True

    def test_returns_true_when_text_area_focused(self):
        """AXTextArea role → text input available."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXTextArea"):
            assert mod.has_focused_text_input() is True

    def test_returns_true_when_web_area_focused(self):
        """AXWebArea (browser) → treated as text input."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXWebArea"):
            assert mod.has_focused_text_input() is True

    def test_returns_true_when_combo_box_focused(self):
        """AXComboBox → text input available."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXComboBox"):
            assert mod.has_focused_text_input() is True

    def test_returns_true_when_search_field_focused(self):
        """AXSearchField → text input available."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXSearchField"):
            assert mod.has_focused_text_input() is True

    def test_returns_false_when_no_focused_element(self):
        """No focused element → no text input."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value=None):
            assert mod.has_focused_text_input() is False

    def test_returns_false_when_button_focused(self):
        """AXButton → not a text input."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXButton"):
            assert mod.has_focused_text_input() is False

    def test_returns_false_when_menu_focused(self):
        """AXMenu → not a text input."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXMenu"):
            assert mod.has_focused_text_input() is False

    def test_fails_open_on_exception(self):
        """Any exception in _get_focused_role → return True (attempt paste)."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", side_effect=OSError("AX API failed")):
            assert mod.has_focused_text_input() is True

    def test_fails_open_on_runtime_error(self):
        """RuntimeError → return True (fail-open)."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", side_effect=RuntimeError("ctypes crash")):
            assert mod.has_focused_text_input() is True
