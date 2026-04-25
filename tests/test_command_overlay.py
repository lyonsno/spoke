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
    mod._start_overlay_fill_worker = lambda work: work()
    overlay = mod.CommandOverlay.__new__(mod.CommandOverlay)
    overlay._window = MagicMock()
    overlay._window.alphaValue.return_value = 1.0
    overlay._window.frame.return_value = _make_rect(620.0, 260.0, 680.0, 160.0)
    overlay._wrapper_view = MagicMock()
    overlay._wrapper_view.layer.return_value = MagicMock()
    overlay._content_view = MagicMock()
    overlay._content_view.layer.return_value = MagicMock()
    overlay._content_view.frame.return_value = _make_rect(28.0, 28.0, 624.0, 104.0)
    overlay._scroll_view = MagicMock()
    overlay._scroll_view.frame.return_value = _make_rect(48.0, 16.0, 528.0, 72.0)
    overlay._text_view = MagicMock()
    overlay._text_view.textStorage.return_value = MagicMock()
    overlay._text_view.textStorage.return_value.length.return_value = 0
    overlay._thinking_label = MagicMock()
    overlay._thinking_label.isHidden.return_value = False
    overlay._narrator_label = MagicMock()
    overlay._screen = MagicMock()
    overlay._screen.frame.return_value = MagicMock(
        size=MagicMock(width=1920, height=1080)
    )
    overlay._screen.backingScaleFactor.return_value = 2.0
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
    overlay._brightness_timer = None
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_capture_rect = _make_rect(0.0, 0.0, 680.0, 160.0)
    overlay._backdrop_base_blur_radius_points = 5.4
    overlay._backdrop_blur_radius_points = 5.4
    overlay._backdrop_base_mask_width_multiplier = 9.0
    overlay._backdrop_mask_width_multiplier = 9.0
    overlay._backdrop_timer = None
    overlay._backdrop_layer = MagicMock()
    overlay._fill_layer = MagicMock()
    overlay._boost_layer = MagicMock()
    overlay._spring_tint_layer = MagicMock()
    overlay._fullscreen_compositor = None
    overlay._pop_timer = None
    overlay._cancel_step = 0
    overlay._cancel_phase = ""
    overlay._narrator_label = None
    overlay._narrator_typewriter_timer = None
    overlay._narrator_full_text = ""
    overlay._narrator_revealed = 0
    overlay._narrator_lines = []
    overlay._narrator_shimmer_timer = None
    overlay._narrator_shimmer_phase = 0.0
    overlay._narrator_shimmer_active = False
    overlay._narrator_suppressed = False
    overlay._collapsed_text = ""
    return overlay, mod


class _FakeAttributedString:
    def __init__(self, text):
        self.text = text

    def addAttribute_value_range_(self, *_args):
        return None

    def appendAttributedString_(self, other):
        self.text += other.text


def _install_fake_attributed_string(monkeypatch):
    class _Builder:
        def initWithString_(self, text):
            return _FakeAttributedString(text)

    class _Alloc:
        def alloc(self):
            return _Builder()

    monkeypatch.setattr(
        sys.modules["AppKit"],
        "NSMutableAttributedString",
        _Alloc(),
        raising=False,
    )


class _FakeLayoutManager:
    def __init__(self, height):
        self.height = height

    def ensureLayoutForTextContainer_(self, container):
        self._container = container

    def usedRectForTextContainer_(self, container):
        return _make_rect(0.0, 0.0, 0.0, self.height)


class _FakeTextContainer:
    def __init__(self):
        self.size = None

    def setContainerSize_(self, size):
        self.size = size


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
    """Test the pop-then-shrink dismiss animation state machine."""

    def test_cancel_dismiss_initializes_grow_phase(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True

        overlay.cancel_dismiss()

        assert overlay._cancel_elapsed == pytest.approx(0.0)
        assert overlay._cancel_phase == "grow"
        assert overlay._streaming is False
        overlay._window.setAlphaValue_.assert_called_with(1.0)

    def test_grow_phase_expands_overlay_for_first_60ms(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        phase, scale, alpha, done = mod._dismiss_animation_state(0.05)

        assert phase == "grow"
        assert scale > 1.0
        assert alpha == pytest.approx(1.0)
        assert done is False

    def test_shrink_phase_starts_after_60ms(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        phase, scale, alpha, done = mod._dismiss_animation_state(0.12)

        assert phase == "shrink"
        assert scale < mod._DISMISS_GROW_SCALE
        assert alpha < 1.0
        assert done is False

    def test_animation_completes_after_200ms(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay.cancel_dismiss()

        for _ in range(12):
            overlay._cancelAnimStep_(None)

        assert overlay._visible is False
        overlay._window.orderOut_.assert_called()
        overlay._wrapper_view.layer.return_value.setValue_forKeyPath_.assert_called_with(
            1.0, "transform.scale"
        )

    def test_cancel_dismiss_with_no_window_is_noop(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._window = None
        overlay.cancel_dismiss()  # should not raise

    def test_cancel_all_timers_clears_dismiss_timer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.cancel_dismiss()

        overlay._cancel_all_timers()

        assert overlay._cancel_timer_anim is None


class TestShowFinishHide:
    """Test overlay lifecycle state transitions."""

    def test_init_initializes_collapsed_thinking_state(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")

        overlay = mod.CommandOverlay.alloc().initWithScreen_(None)

        assert overlay._collapsed_text == ""

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

    def test_show_starts_recurring_brightness_sampling(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.show()

        assert overlay._brightness_timer is not None

    def test_show_defers_pulse_until_entrance_fade_finishes(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)

        overlay.show()

        assert overlay._fade_timer is not None
        assert overlay._pulse_timer is None

        for _ in range(mod._FADE_STEPS):
            overlay.fadeStep_(overlay._fade_timer)

        assert overlay._pulse_timer is not None

    def test_show_fade_in_is_fast_enough_to_feel_immediate(self, mock_pyobjc):
        _, mod = _make_overlay(mock_pyobjc)

        assert mod._FADE_IN_S == pytest.approx(0.4)

    def test_visual_start_waits_until_after_entrance_fade(self, mock_pyobjc):
        _, mod = _make_overlay(mock_pyobjc)

        assert mod._COMMAND_VISUAL_START_DELAY_S >= mod._FADE_IN_S + 0.1

    def test_show_can_resume_thinking_timer_without_resetting_elapsed_state(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._thinking_seconds = 4.2
        overlay._thinking_inverted = True

        overlay.show(preserve_thinking_timer=True)

        assert overlay._thinking_timer is not None
        assert overlay._thinking_seconds == 4.2
        assert overlay._thinking_inverted is True
        overlay._thinking_label.setStringValue_.assert_called_with("4.2s")

    def test_finish_clears_streaming_and_stops_thinking(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay.show()
        assert overlay._streaming is True

        overlay.finish()
        assert overlay._streaming is False
        assert overlay._thinking_timer is None


class TestWindowLayering:
    """Command overlay should stack independently from the preview overlay."""

    def test_setup_places_command_overlay_above_preview_overlay_level(self, mock_pyobjc):
        sys.modules.pop("spoke.overlay", None)
        sys.modules.pop("spoke.command_overlay", None)
        overlay_mod = importlib.import_module("spoke.overlay")
        command_mod = importlib.import_module("spoke.command_overlay")

        assert command_mod._COMMAND_OVERLAY_WINDOW_LEVEL == overlay_mod._OVERLAY_WINDOW_LEVEL + 1

    def test_setup_keeps_three_line_narrator_label_within_overlay_bounds(
        self, mock_pyobjc, monkeypatch
    ):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)

        thinking_builder = MagicMock()
        thinking_label = MagicMock()
        thinking_builder.initWithFrame_.return_value = thinking_label
        narrator_builder = MagicMock()
        narrator_label = MagicMock()
        narrator_builder.initWithFrame_.return_value = narrator_label
        text_field_cls = MagicMock()
        text_field_cls.alloc.side_effect = [thinking_builder, narrator_builder]
        monkeypatch.setattr(sys.modules["AppKit"], "NSTextField", text_field_cls)
        monkeypatch.setattr(sys.modules["AppKit"], "NSTextAlignmentRight", 2, raising=False)
        monkeypatch.setattr(sys.modules["AppKit"], "NSTextAlignmentLeft", 0, raising=False)
        monkeypatch.setattr(sys.modules["AppKit"], "NSLineBreakByWordWrapping", 0, raising=False)

        overlay = mod.CommandOverlay.alloc().initWithScreen_(None)
        overlay._screen = MagicMock()
        overlay._screen.frame.return_value = _make_rect(0.0, 0.0, 1920.0, 1080.0)
        overlay._screen.backingScaleFactor.return_value = 2.0

        overlay.setup()

        narrator_frame = narrator_builder.initWithFrame_.call_args[0][0]
        narrator_label.setMaximumNumberOfLines_.assert_called_once_with(3)
        assert narrator_frame.size.height == pytest.approx(45.0)
        assert narrator_frame.origin.y >= 0.0
        assert narrator_frame.origin.y + narrator_frame.size.height <= mod._OVERLAY_HEIGHT

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
        overlay._collapsed_text = "Thought for 2s"

        overlay.show()
        assert overlay._response_text == ""
        assert overlay._utterance_text == ""
        assert overlay._collapsed_text == ""

    def test_show_clears_attributed_text_storage_before_reuse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.show()

        overlay._text_view.textStorage().setAttributedString_.assert_called_once()

    def test_show_can_skip_thinking_timer_for_recalled_history(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._start_thinking_timer = MagicMock()

        overlay.show(start_thinking_timer=False)

        overlay._start_thinking_timer.assert_not_called()

    def test_show_with_initial_transcript_lays_out_before_ordering_front(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        events = []
        overlay._update_layout = MagicMock(side_effect=lambda: events.append("layout"))
        overlay._window.orderFrontRegardless.side_effect = lambda: events.append("front")

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert overlay._utterance_text == "User prompt"
        assert overlay._response_text == "Assistant response"
        assert events[:2] == ["layout", "front"]

    def test_show_rebuilds_default_fill_geometry_before_reuse(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._fill_image_brightness = 0.0
        overlay._apply_ridge_masks = MagicMock()

        overlay.show()

        overlay._apply_ridge_masks.assert_called_once_with(
            mod._OVERLAY_WIDTH,
            mod._OVERLAY_HEIGHT,
        )

    def test_optical_show_defers_heavy_backdrop_startup_until_after_first_paint(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._refresh_backdrop_snapshot = MagicMock()
        overlay._start_fullscreen_compositor = MagicMock()
        overlay._start_backdrop_refresh_timer = MagicMock()

        overlay.show()

        overlay._window.orderFrontRegardless.assert_called_once()
        overlay._refresh_backdrop_snapshot.assert_not_called()
        overlay._start_fullscreen_compositor.assert_not_called()
        overlay._start_backdrop_refresh_timer.assert_not_called()
        assert overlay._visual_start_timer is not None

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
        # New path uses setAttributedString_ to rebuild in one shot, not setString_("")
        overlay._text_view.textStorage().setAttributedString_.assert_called_once()
        overlay.append_token.assert_called_once_with("Done.")

    def test_set_response_text_with_utterance_calls_layout_once(self, mock_pyobjc):
        """set_response_text must not trigger an intermediate layout with only the
        utterance text — that shrinks the window before growing it, causing visible flicker."""
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "What is the capital of France?"
        overlay._response_text = "Paris."

        layout_calls = []
        overlay._update_layout = MagicMock(side_effect=lambda: layout_calls.append(1))

        overlay.set_response_text("Paris is the capital of France.")

        assert len(layout_calls) == 1, (
            f"set_response_text called _update_layout {len(layout_calls)} time(s); "
            "expected exactly 1 — intermediate calls shrink the window causing flicker"
        )

    def test_set_response_text_uses_extra_gap_between_utterance_and_collapsed_thinking(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "User prompt"
        overlay._collapsed_text = "Thought for 2s"
        overlay.append_token = MagicMock()
        _install_fake_attributed_string(monkeypatch)

        overlay.set_response_text("Done.")

        combined = (
            overlay._text_view.textStorage()
            .setAttributedString_.call_args[0][0]
            .text
        )
        assert "User prompt\n\nThought for 2s" in combined
        assert "User prompt\nThought for 2s" not in combined

    def test_append_token_keeps_breathing_room_after_collapsed_thinking(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "User prompt"
        overlay._collapsed_text = "Thought for 2s"
        overlay._response_text = ""
        overlay._update_layout = MagicMock()
        _install_fake_attributed_string(monkeypatch)
        overlay._make_tool_indicator_fragment = MagicMock(
            side_effect=lambda token: _FakeAttributedString(token)
        )
        overlay._make_response_fragment = MagicMock(
            side_effect=lambda token: _FakeAttributedString(token)
        )

        overlay.append_token("[calling list_directory…]")

        first_append = (
            overlay._text_view.textStorage()
            .appendAttributedString_.call_args_list[0][0][0]
            .text
        )
        assert first_append == "\n\n"

    def test_set_thinking_collapsed_starts_below_utterance(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "A wrapped user prompt that needs breathing room."
        overlay._update_layout = MagicMock()
        _install_fake_attributed_string(monkeypatch)

        overlay.set_thinking_collapsed("Thought for 4s")

        appended = (
            overlay._text_view.textStorage()
            .appendAttributedString_.call_args[0][0]
            .text
        )
        assert appended == "\n\nThought for 4s"

    def test_append_token_refreshes_punchthrough_mask_after_layout(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "User prompt"
        overlay._response_text = ""
        overlay._text_punchthrough = True
        overlay._update_layout = MagicMock()
        overlay._update_punchthrough_mask = MagicMock()
        _install_fake_attributed_string(monkeypatch)
        overlay._make_response_fragment = MagicMock(
            side_effect=lambda token: _FakeAttributedString(token)
        )

        overlay.append_token("Done.")

        overlay._update_layout.assert_called_once()
        overlay._update_punchthrough_mask.assert_called_once()

    def test_append_token_renders_tool_indicator_without_monkeypatch(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._response_text = "Already streaming"
        overlay._update_layout = MagicMock()
        _install_fake_attributed_string(monkeypatch)

        overlay.append_token("[calling search_file…]")

        appended = (
            overlay._text_view.textStorage()
            .appendAttributedString_.call_args[0][0]
            .text
        )
        assert appended == "[calling search_file…]"

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
        assert overlay._pulse_timer is None
        assert overlay._thinking_timer is not None
        assert overlay._brightness_timer is not None

        overlay._cancel_all_timers()
        assert overlay._fade_timer is None
        assert overlay._pulse_timer is None
        assert overlay._linger_timer is None
        assert overlay._thinking_timer is None
        assert overlay._brightness_timer is None

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

    def test_show_keeps_content_background_clear_after_brightness_snap(self, mock_pyobjc):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay.set_brightness(1.0, immediate=True)

        overlay._content_view.layer.return_value.setBackgroundColor_.reset_mock()
        overlay.show()

        overlay._content_view.layer.return_value.setBackgroundColor_.assert_called_with(None)

    def test_background_fill_tracks_preview_overlay_chrome(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        sys.modules.pop("spoke.overlay", None)
        command_mod = importlib.import_module("spoke.command_overlay")
        overlay_mod = importlib.import_module("spoke.overlay")
        try:
            assert command_mod._background_color_for_brightness(0.0) == pytest.approx(
                overlay_mod._BG_COLOR_DARK
            )
            assert command_mod._background_color_for_brightness(1.0) == pytest.approx(
                overlay_mod._BG_COLOR_LIGHT
            )
        finally:
            sys.modules.pop("spoke.command_overlay", None)
            sys.modules.pop("spoke.overlay", None)

    def test_dark_background_fill_uses_additive_experiment(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._fill_compositing_filter_for_brightness(0.0) == "plusL"
            assert mod._fill_compositing_filter_for_brightness(1.0) is None
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_apply_surface_theme_updates_fill_compositing_filter(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._content_view.frame.return_value = _make_rect(0.0, 0.0, 600.0, 80.0)

        overlay._brightness = 0.0
        overlay._apply_surface_theme()
        overlay._fill_layer.setCompositingFilter_.assert_called_with("plusL")

        overlay._fill_layer.setCompositingFilter_.reset_mock()
        overlay._brightness = 1.0
        overlay._fill_image_brightness = -1.0
        overlay._apply_surface_theme()
        overlay._fill_layer.setCompositingFilter_.assert_called_with(None)

    def test_apply_backdrop_pulse_style_pushes_optical_shell_config_when_enabled(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._backdrop_renderer.set_live_blur_radius_points = MagicMock()
        overlay._backdrop_renderer.set_live_optical_shell_config = MagicMock()
        overlay._update_backdrop_mask = MagicMock()

        overlay._apply_backdrop_pulse_style(1.0)

        config = overlay._backdrop_renderer.set_live_optical_shell_config.call_args[0][0]
        assert config["enabled"] is True
        assert config["content_width_points"] > mod._OVERLAY_WIDTH
        assert config["content_height_points"] > mod._OVERLAY_HEIGHT
        assert config["ring_amplitude_points"] > 0.0
        assert config["tail_amplitude_points"] > 0.0

    def test_backdrop_mask_generation_is_not_run_synchronously(
        self, mock_pyobjc, monkeypatch
    ):
        """Backdrop mask SDF/image work should sit behind the fill worker seam."""
        overlay, mod = _make_overlay(mock_pyobjc)
        queued = []

        import spoke.overlay as ov_mod

        def forbidden_sync_call(*_args):
            raise AssertionError("backdrop mask generation ran on the caller thread")

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", forbidden_sync_call)
        monkeypatch.setattr(ov_mod, "_fill_field_to_image", forbidden_sync_call)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: queued.append(work))

        overlay._update_backdrop_mask(680.0, 160.0)

        assert len(queued) == 1

    def test_optical_shell_peak_assistant_breath_keeps_fill_light_enough_to_show_backdrop(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._brightness = 1.0
        overlay._brightness_target = 1.0
        overlay._fill_image_brightness = 1.0
        overlay._pulse_phase_asst = 0.0
        overlay._pulse_phase_user = 0.0
        overlay._tts_active = False
        overlay._tts_blend = 0.0
        overlay._cancel_spring = 0.0
        overlay._cancel_spring_target = 0.0
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        fill_opacity = overlay._fill_layer.setOpacity_.call_args[0][0]
        assert fill_opacity <= 0.45, (
            "Optical-shell mode should keep the fill translucent enough for the warped backdrop to read."
        )

    def test_assistant_text_alpha_floor_and_ceiling_breathe_within_legible_band(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._assistant_text_alpha_for_breath(0.0) == pytest.approx(0.85)
            assert mod._assistant_text_alpha_for_breath(1.0) == pytest.approx(0.95)
            assert mod.CommandOverlay._TTS_ALPHA_MIN == pytest.approx(0.85)
            assert mod.CommandOverlay._TTS_ALPHA_MAX == pytest.approx(0.95)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_fill_layer_opacity_breathes_in_tighter_legible_band(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._fill_layer_opacity_for_breath(0.0) == pytest.approx(0.85)
            assert mod._fill_layer_opacity_for_breath(1.0) == pytest.approx(0.95)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_user_text_alpha_band_stays_tight_around_ninety_percent(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._user_text_alpha_for_breath(0.0) == pytest.approx(0.85)
            assert mod._user_text_alpha_for_breath(1.0) == pytest.approx(0.95)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_assistant_foreground_flips_from_dark_to_light_with_surface_brightness(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._assistant_foreground_color_for_brightness(0.0)
            light = mod._assistant_foreground_color_for_brightness(1.0)

            assert max(dark) < 0.2
            assert min(light) > 0.9
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_user_text_turns_bright_on_light_backgrounds(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark = mod._user_text_color_for_brightness(0.0)
            light = mod._user_text_color_for_brightness(1.0)

            assert max(dark) < 0.25
            assert min(light) > 0.9
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_text_contrast_mapping_snaps_aggressively_near_midpoint(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            below = mod._contrast_mix_for_brightness(0.45)
            above = mod._contrast_mix_for_brightness(0.55)

            assert below < 0.15
            assert above > 0.85
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_brightness_resample_updates_target_while_visible(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._screen = MagicMock()

        monkeypatch.setattr(mod, "_sample_screen_brightness_for_overlay", lambda _screen: 0.83)

        overlay.brightnessResample_(None)

        assert overlay._brightness_target == pytest.approx(0.83)

    def test_brightness_crossing_reaches_contrast_band_in_one_pulse(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._advance_command_brightness(0.0, 1.0) > 0.56
            assert mod._advance_command_brightness(1.0, 0.0) < 0.44
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_compositor_brightness_resamples_without_half_second_lag(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._text_view.textStorage.return_value.length.return_value = 0
        overlay._brightness = 0.0
        overlay._brightness_target = 0.0
        overlay._cancel_spring = 0.0
        overlay._cancel_spring_target = 0.0
        overlay._cancel_spring_fired = False
        overlay._on_cancel_spring_threshold = None
        overlay._text_punchthrough = False
        overlay._apply_surface_theme = MagicMock()
        overlay._apply_backdrop_pulse_style = MagicMock()
        compositor = MagicMock()
        compositor.sampled_brightness = 1.0
        overlay._fullscreen_compositor = compositor

        for _ in range(7):
            overlay._pulseStepInner()

        assert compositor.refresh_brightness.call_count >= 2

    def test_response_fragment_uses_blurry_colored_underlay_with_crisp_foreground(
        self, mock_pyobjc, monkeypatch
    ):
        sys.modules.pop("spoke.command_overlay", None)
        try:
            mod = importlib.import_module("spoke.command_overlay")

            class _FakeAttr:
                def __init__(self, text):
                    self.text = text
                    self.attrs = []

                def addAttribute_value_range_(self, name, value, rng):
                    self.attrs.append((name, value, rng))

            class _AttrBuilder:
                def initWithString_(self, text):
                    return _FakeAttr(text)

            class _AttrAlloc:
                def alloc(self):
                    return _AttrBuilder()

            class _FakeShadow:
                def __init__(self):
                    self.color = None
                    self.offset = None
                    self.blur = None

                def setShadowColor_(self, color):
                    self.color = color

                def setShadowOffset_(self, offset):
                    self.offset = offset

                def setShadowBlurRadius_(self, radius):
                    self.blur = radius

            class _ShadowBuilder:
                def init(self):
                    return _FakeShadow()

            class _ShadowAlloc:
                def alloc(self):
                    return _ShadowBuilder()

            monkeypatch.setattr(
                sys.modules["AppKit"], "NSMutableAttributedString", _AttrAlloc(), raising=False
            )
            monkeypatch.setattr(sys.modules["AppKit"], "NSShadow", _ShadowAlloc(), raising=False)
            monkeypatch.setattr(
                sys.modules["AppKit"].NSColor,
                "colorWithSRGBRed_green_blue_alpha_",
                staticmethod(lambda r, g, b, a: (r, g, b, a)),
                raising=False,
            )

            overlay, _ = _make_overlay(mock_pyobjc)
            overlay._brightness = 1.0
            overlay._color_phase = 0.75

            frag = overlay._make_response_fragment("Done.")

            fg = next(value for name, value, _ in frag.attrs if name == "NSForegroundColor")
            shadow = next(value for name, value, _ in frag.attrs if name == "NSShadow")

            assert min(fg[:3]) > 0.9
            assert fg[3] == pytest.approx(mod._ASSISTANT_TEXT_ALPHA_MAX)
            assert shadow.blur == pytest.approx(mod._ASSISTANT_BLUR_RADIUS)
            assert shadow.color[:3] != fg[:3]
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_pulse_preserves_crisp_response_foreground_contrast(
        self, mock_pyobjc, monkeypatch
    ):
        sys.modules.pop("spoke.command_overlay", None)
        try:
            mod = importlib.import_module("spoke.command_overlay")

            class _FakeTextStorage:
                def __init__(self, text):
                    self.text = text
                    self.attrs = []

                def length(self):
                    return len(self.text)

                def addAttribute_value_range_(self, name, value, rng):
                    self.attrs.append((name, value, rng))

            monkeypatch.setattr(
                sys.modules["AppKit"].NSColor,
                "colorWithSRGBRed_green_blue_alpha_",
                staticmethod(lambda r, g, b, a: (r, g, b, a)),
                raising=False,
            )

            overlay, _ = _make_overlay(mock_pyobjc)
            overlay._text_view = MagicMock()
            storage = _FakeTextStorage("Prompt\n\nAnswer")
            overlay._text_view.textStorage.return_value = storage
            overlay._brightness = 0.0
            overlay._brightness_target = 0.0
            overlay._utterance_text = "Prompt"
            overlay._response_text = "Answer"
            overlay._cancel_spring = 0.0
            overlay._cancel_spring_target = 0.0
            overlay._cancel_spring_fired = False
            overlay._on_cancel_spring_threshold = None
            overlay._text_punchthrough = False
            overlay._fullscreen_compositor = None
            overlay._apply_surface_theme = MagicMock()
            overlay._apply_backdrop_pulse_style = MagicMock()

            overlay._pulseStepInner()

            response_start = len("Prompt\n\n")
            response_colors = [
                value
                for name, value, rng in storage.attrs
                if name == "NSForegroundColor" and rng[0] >= response_start
            ]

            assert response_colors
            assert min(response_colors[-1][:3]) >= min(mod._ASSISTANT_TEXT_COLOR_DARK)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_pulse_batches_long_response_styling(self, mock_pyobjc, monkeypatch):
        monkeypatch.setenv("SPOKE_COMMAND_RESPONSE_ANIMATION_CHAR_LIMIT", "8")
        sys.modules.pop("spoke.command_overlay", None)
        try:
            mod = importlib.import_module("spoke.command_overlay")

            class _FakeTextStorage:
                def __init__(self, text):
                    self.text = text
                    self.attrs = []

                def length(self):
                    return len(self.text)

                def addAttribute_value_range_(self, name, value, rng):
                    self.attrs.append((name, value, rng))

            monkeypatch.setattr(
                sys.modules["AppKit"].NSColor,
                "colorWithSRGBRed_green_blue_alpha_",
                staticmethod(lambda r, g, b, a: (r, g, b, a)),
                raising=False,
            )

            overlay, _ = _make_overlay(mock_pyobjc)
            overlay._text_view = MagicMock()
            response = "A long answer that should not restyle every character"
            storage = _FakeTextStorage("Prompt\n\n" + response)
            overlay._text_view.textStorage.return_value = storage
            overlay._brightness = 0.0
            overlay._brightness_target = 0.0
            overlay._utterance_text = "Prompt"
            overlay._response_text = response
            overlay._cancel_spring = 0.0
            overlay._cancel_spring_target = 0.0
            overlay._cancel_spring_fired = False
            overlay._on_cancel_spring_threshold = None
            overlay._text_punchthrough = False
            overlay._fullscreen_compositor = None
            overlay._apply_surface_theme = MagicMock()
            overlay._apply_backdrop_pulse_style = MagicMock()

            overlay._pulseStepInner()

            response_start = len("Prompt\n\n")
            response_attrs = [
                rng
                for _name, _value, rng in storage.attrs
                if rng[0] >= response_start
            ]
            assert response_attrs
            assert len(response_attrs) <= 4
            assert any(rng == (response_start, len(response)) for rng in response_attrs)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_punchthrough_hides_live_scroll_text_layer(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay._enable_text_punchthrough(True)

        assert overlay._text_punchthrough is True
        overlay._scroll_view.setHidden_.assert_called_once_with(True)

        overlay._enable_text_punchthrough(False)

        overlay._scroll_view.setHidden_.assert_called_with(False)

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

    def test_update_layout_rebuilds_fill_geometry_when_assistant_overlay_grows(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(280.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj
        overlay._apply_ridge_masks = MagicMock()

        overlay._update_layout()

        overlay._apply_ridge_masks.assert_called_once_with(
            mod._OVERLAY_WIDTH,
            pytest.approx(304.0),
        )

    def test_update_layout_resets_text_geometry_to_match_visible_area(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(280.0)
        container = _FakeTextContainer()
        overlay._text_view.textContainer.return_value = container
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj

        overlay._update_layout()

        doc_frame = overlay._text_view.setFrame_.call_args[0][0]
        assert doc_frame.size.width == pytest.approx(mod._OVERLAY_WIDTH - 24)
        assert doc_frame.size.height == pytest.approx(304.0 - 16)
        assert container.size == (mod._OVERLAY_WIDTH - 24, 1.0e7)


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


class TestSDFCaching:
    """SDF recomputation is skipped when geometry hasn't changed."""

    def test_same_dimensions_reuses_cached_sdf(self, mock_pyobjc, monkeypatch):
        """Brightness-only changes should not recompute the SDF."""
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._spring_tint_layer = None

        import spoke.overlay as ov_mod
        call_count = 0
        original = ov_mod._overlay_rounded_rect_sdf

        def counting_sdf(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", counting_sdf)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: work())

        # First call — SDF must be computed
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        # Second call, same dimensions — SDF should be cached
        overlay._brightness = 0.5
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1, "SDF was recomputed despite identical geometry"

    def test_same_appearance_reuses_fill_image(self, mock_pyobjc, monkeypatch):
        """Repeated show-time geometry checks should not rebuild the fill bitmap."""
        overlay, mod = _make_overlay(mock_pyobjc)

        import spoke.overlay as ov_mod
        call_count = 0

        def counting_fill_image(*_args):
            nonlocal call_count
            call_count += 1
            return "fill-image", b"payload"

        monkeypatch.setattr(ov_mod, "_fill_field_to_image", counting_fill_image)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: work())

        overlay._apply_ridge_masks(600.0, 80.0)
        overlay._apply_ridge_masks(600.0, 80.0)

        assert call_count == 1

    def test_fill_generation_is_not_run_synchronously(self, mock_pyobjc, monkeypatch):
        """The expensive fill bitmap build should happen behind the worker boundary."""
        overlay, mod = _make_overlay(mock_pyobjc)
        queued = []

        import spoke.overlay as ov_mod

        def forbidden_sync_call(*_args):
            raise AssertionError("fill image generation ran on the caller thread")

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", forbidden_sync_call)
        monkeypatch.setattr(mod, "_stadium_signed_distance_field", forbidden_sync_call)
        monkeypatch.setattr(ov_mod, "_fill_field_to_image", forbidden_sync_call)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: queued.append(work))

        overlay._apply_ridge_masks(600.0, 80.0)

        assert len(queued) == 1

    def test_changed_height_recomputes_sdf(self, mock_pyobjc, monkeypatch):
        """A height change must recompute the SDF."""
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._spring_tint_layer = None

        import spoke.overlay as ov_mod
        call_count = 0
        original = ov_mod._overlay_rounded_rect_sdf

        def counting_sdf(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", counting_sdf)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: work())

        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        overlay._apply_ridge_masks(600.0, 200.0)
        assert call_count == 2, "SDF was not recomputed after height change"
