"""Tests for CommandOverlay state machines and timer logic.

Tests cover: thinking timer lifecycle, dismiss animation phases,
show/finish/hide state transitions, and timer cancellation.
All tests use mocked PyObjC — no GUI runtime required.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _make_overlay(mock_pyobjc):
    """Create a CommandOverlay with mocked internals."""
    sys.modules.pop("spoke.command_overlay", None)
    mod = importlib.import_module("spoke.command_overlay")
    overlay = mod.CommandOverlay.__new__(mod.CommandOverlay)
    overlay._window = MagicMock()
    overlay._window.alphaValue.return_value = 1.0
    overlay._content_view = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._text_view = MagicMock()
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
    overlay._cancel_step = 0
    overlay._cancel_phase = ""
    return overlay, mod


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

    def test_fade_phase_uses_quadratic_alpha_curve_before_completion(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        for _ in range(8):
            overlay._cancelAnimStep_(None)

        overlay._cancelAnimStep_(None)
        overlay._cancelAnimStep_(None)
        alpha_calls = [call.args[0] for call in overlay._window.setAlphaValue_.call_args_list]

        assert alpha_calls[-2] == pytest.approx(1.0 - (1.0 / 15.0) ** 2)
        assert alpha_calls[-1] == pytest.approx(1.0 - (2.0 / 15.0) ** 2)

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
