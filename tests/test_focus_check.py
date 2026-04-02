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

    def test_returns_true_when_window_focused(self):
        """AXWindow → treated as pasteable (terminals, Electron apps)."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXWindow"):
            assert mod.has_focused_text_input() is True

    def test_returns_true_when_group_focused(self):
        """AXGroup → treated as pasteable (custom app views)."""
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_role", return_value="AXGroup"):
            assert mod.has_focused_text_input() is True

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


class TestFocusedTextContains:
    def test_returns_true_when_expected_is_present(self):
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_value", return_value="prefix dictated text suffix"):
            assert mod.focused_text_contains("dictated text") is True

    def test_returns_false_when_value_present_without_expected_text(self):
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_value", return_value="other text entirely"):
            assert mod.focused_text_contains("dictated text") is False

    def test_returns_none_when_value_unavailable(self):
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_value", return_value=None):
            assert mod.focused_text_contains("dictated text") is None

    def test_returns_none_on_lookup_error(self):
        mod = _import_focus_check()
        with patch.object(mod, "_get_focused_value", side_effect=RuntimeError("AXValue failed")):
            assert mod.focused_text_contains("dictated text") is None
