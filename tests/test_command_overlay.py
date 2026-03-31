"""Tests for CommandOverlay state machines and timer logic.

Tests cover: thinking timer lifecycle, dismiss animation phases,
show/finish/hide state transitions, and timer cancellation.
All tests use mocked PyObjC — no GUI runtime required.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )

def _make_overlay(mock_pyobjc):
    """Create a CommandOverlay with mocked internals."""
    sys.modules.pop("spoke.command_overlay", None)
    mod = importlib.import_module("spoke.command_overlay")
    overlay = mod.CommandOverlay.__new__(mod.CommandOverlay)
    overlay._window = MagicMock()
    overlay._window.alphaValue.return_value = 1.0
    overlay._content_view = MagicMock()
    overlay._content_view.layer.return_value = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._text_view = MagicMock()
    overlay._text_view.textStorage.return_value = MagicMock()
    overlay._text_view.textStorage.return_value.length.return_value = 0
    overlay._thinking_label = MagicMock()
    overlay._thinking_label.isHidden.return_value = False
    overlay._screen = MagicMock()
    overlay._screen.frame.return_value = MagicMock(
        size=MagicMock(width=1920, height=1080)
    )
    overlay._visible = False
    overlay._streaming = False
    overlay._response_text = ""
    overlay._utterance_text = ""
    overlay._fade_timer = None
    overlay._pulse_timer = None
    overlay._linger_timer = None
    overlay._thinking_timer = None
    overlay._cancel_timer_anim = None
    overlay._thinking_seconds = 0.0
    overlay._thinking_inverted = False
    overlay._fade_step = 0
    overlay._fade_from = 0.0
    overlay._fade_direction = 0
    overlay._pulse_phase_asst = 0.0
    overlay._pulse_phase_user = 0.0
    overlay._color_phase = 0.0
    overlay._color_velocity_phase = 0.0
    overlay._tts_amplitude = 0.0
    overlay._tts_active = False
    overlay._tts_blend = 0.0
    overlay._tool_mode = False
    overlay._brightness = 0.0
    overlay._brightness_target = 0.0
    overlay._fill_layer = MagicMock()
    overlay._cancel_step = 0
    overlay._cancel_phase = ""
    return overlay, mod


class _FakeLayoutManager:
    def __init__(self, height):
        self.height = height

    def ensureLayoutForTextContainer_(self, container):
        self._container = container

    def usedRectForTextContainer_(self, container):
        return _make_rect(0.0, 0.0, 0.0, self.height)


class TestThinkingTimer:
    """Test the thinking timer state machine."""

    def test_start_sets_initial_state(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()

        assert overlay._thinking_seconds == 0.0
        assert overlay._thinking_inverted is False
        assert overlay._thinking_timer is not None
        overlay._thinking_label.setHidden_.assert_called_with(False)
        overlay._thinking_label.setStringValue_.assert_called_with("0.0s")

    def test_tick_increments_time(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()

        for _ in range(10):
            overlay.thinkingTick_(None)

        assert abs(overlay._thinking_seconds - 1.0) < 0.01
        overlay._thinking_label.setStringValue_.assert_called_with("1.0s")

    def test_invert_sets_flag(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()
        assert overlay._thinking_inverted is False

        overlay.invert_thinking_timer()
        assert overlay._thinking_inverted is True

    def test_stop_clears_timer_and_hides_label(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer()
        assert overlay._thinking_timer is not None

        overlay._stop_thinking_timer()
        assert overlay._thinking_timer is None
        overlay._thinking_label.setHidden_.assert_called_with(True)

    def test_stop_is_safe_when_no_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        assert overlay._thinking_timer is None
        overlay._stop_thinking_timer()  # should not raise


class TestDismissAnimation:
    """Test the cancel_dismiss two-phase animation state machine."""

    def test_cancel_dismiss_initializes_hold_phase(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True

        overlay.cancel_dismiss()

        assert overlay._cancel_step == 0
        assert overlay._cancel_phase == "hold"
        assert overlay._streaming is False
        overlay._window.setAlphaValue_.assert_called_with(1.0)

    def test_hold_phase_transitions_to_fade_after_8_steps(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        # Simulate 8 animation steps (hold phase)
        for _ in range(8):
            overlay._cancelAnimStep_(None)

        assert overlay._cancel_phase == "fade"
        assert overlay._cancel_step == 0  # reset for fade

    def test_fade_phase_completes_and_hides_window(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        # Advance through hold phase
        for _ in range(8):
            overlay._cancelAnimStep_(None)

        # Advance through fade phase until completion
        for _ in range(8):
            overlay._cancelAnimStep_(None)

        assert overlay._visible is False
        overlay._window.orderOut_.assert_called()

    def test_cancel_dismiss_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.cancel_dismiss()  # should not raise


class TestShowFinishHide:
    """Test overlay lifecycle state transitions."""

    def test_show_sets_visible_and_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()

        assert overlay._visible is True
        assert overlay._streaming is True
        assert overlay._response_text == ""

    def test_show_starts_thinking_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()

        assert overlay._thinking_timer is not None
        assert overlay._thinking_seconds == 0.0

    def test_finish_clears_streaming_and_stops_thinking(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()
        assert overlay._streaming is True

        overlay.finish()
        assert overlay._streaming is False
        assert overlay._thinking_timer is None

    def test_hide_clears_visible_and_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True

        overlay.hide()
        assert overlay._visible is False
        assert overlay._streaming is False

    def test_show_resets_text(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._response_text = "old response"
        overlay._utterance_text = "old utterance"

        overlay.show()
        assert overlay._response_text == ""
        assert overlay._utterance_text == ""

    def test_show_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.show()
        assert overlay._visible is False

    def test_set_response_text_replaces_existing_response(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = ""
        overlay._response_text = "Let me check."

        def _append(text):
            overlay._response_text += text

        overlay.append_token = MagicMock(side_effect=_append)

        overlay.set_response_text("Done.")

        assert overlay._response_text == "Done."
        overlay._text_view.setString_.assert_called_once_with("")
        overlay.append_token.assert_called_once_with("Done.")

    def test_hide_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.hide()  # should not raise


class TestTimerCancellation:
    """Test timer cleanup methods."""

    def test_cancel_all_timers_clears_everything(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        # Start all timers
        overlay.show()
        assert overlay._fade_timer is not None
        assert overlay._pulse_timer is not None
        assert overlay._thinking_timer is not None

        overlay._cancel_all_timers()
        assert overlay._fade_timer is None
        assert overlay._pulse_timer is None
        assert overlay._linger_timer is None
        assert overlay._thinking_timer is None

    def test_cancel_fade_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._fade_timer = None
        overlay._cancel_fade()  # should not raise

    def test_cancel_pulse_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._pulse_timer = None
        overlay._cancel_pulse()  # should not raise

    def test_cancel_linger_safe_when_none(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._linger_timer = None
        overlay._cancel_linger()  # should not raise


class TestLingerDone:
    """Test the linger completion callback."""

    def test_linger_done_hides_when_visible_and_not_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = False

        overlay.lingerDone_(None)
        # hide() sets _visible = False
        assert overlay._visible is False

    def test_linger_done_noop_when_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True

        overlay.lingerDone_(None)
        assert overlay._visible is True  # still visible — streaming

    def test_linger_done_noop_when_not_visible(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = False
        overlay._streaming = False

        overlay.lingerDone_(None)
        assert overlay._visible is False

class TestAdaptiveCompositing:
    """Test brightness-adaptive command overlay styling."""

    def test_set_brightness_immediate_snaps(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.set_brightness(0.8, immediate=True)

        assert overlay._brightness == pytest.approx(0.8)
        assert overlay._brightness_target == pytest.approx(0.8)

    def test_show_uses_light_background_after_brightness_snap(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay.set_brightness(1.0, immediate=True)

        mod.NSColor.colorWithSRGBRed_green_blue_alpha_.reset_mock()
        overlay.show()

        bg_call = None
        for call in mod.NSColor.colorWithSRGBRed_green_blue_alpha_.call_args_list:
            r, g, b, a = call[0]
            if a == pytest.approx(mod._BG_ALPHA) and min(r, g, b) > 0.85:
                bg_call = call[0]
                break
        assert bg_call is not None

    def test_response_color_darkens_for_bright_backgrounds(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._response_color_for_brightness((0.82, 0.66, 0.94), 0.0)
            light = mod._response_color_for_brightness((0.82, 0.66, 0.94), 1.0)

            assert sum(light) < sum(dark)
            assert light[2] == max(light)
            assert max(light) < 0.12
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_user_text_turns_dark_on_light_backgrounds(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._user_text_color_for_brightness(0.0)
            light = mod._user_text_color_for_brightness(1.0)

            assert min(dark) > 0.9
            assert max(light) < 0.17
        finally:
            sys.modules.pop("spoke.command_overlay", None)

class TestGeometryCaps:
    def test_update_layout_can_grow_assistant_overlay_near_notch(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(1000.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj

        overlay._update_layout()

        frame = overlay._window.setFrame_display_animate_.call_args[0][0]
        expected_height = 640.0
        assert frame.size.height == pytest.approx(expected_height + 2 * mod._OUTER_FEATHER)
        assert overlay._content_view.setFrame_.call_args[0][0].size.height == pytest.approx(expected_height)


class TestToolState:
    """Test the tool execution visual state machine."""

    def test_set_tool_active_shows_label(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._thinking_label.setHidden_.reset_mock()
        
        overlay.set_tool_active(True)
        
        assert overlay._tool_mode is True
        overlay._thinking_label.setHidden_.assert_called_with(False)
        overlay._thinking_label.setStringValue_.assert_called_with("tool…")

    def test_thinking_tick_shows_tool_in_tool_mode(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.set_tool_active(True)
        overlay._thinking_label.setStringValue_.reset_mock()
        
        overlay.thinkingTick_(None)
        
        overlay._thinking_label.setStringValue_.assert_called_with("tool…")

    def test_set_tool_active_false_preserves_mode_until_tick(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.set_tool_active(True)
        overlay.set_tool_active(False)
        
        assert overlay._tool_mode is False
        overlay.thinkingTick_(None)
        # Now it should show seconds again
        assert "s" in overlay._thinking_label.setStringValue_.call_args[0][0]
