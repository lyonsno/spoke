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
    overlay._wrapper_view = MagicMock()
    overlay._wrapper_view.layer.return_value = MagicMock()
    overlay._content_view = MagicMock()
    overlay._content_view.layer.return_value = MagicMock()
    overlay._scroll_view = MagicMock()
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
    overlay._response_body_started = False
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

    def test_show_clears_attributed_text_storage_before_reuse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.show()

        overlay._text_view.textStorage().setAttributedString_.assert_called_once()

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

        overlay.set_response_text("Done.")

        assert overlay._response_text == "Done."
        # Final rebuild happens in one shot without delegating through append_token().
        overlay._text_view.textStorage().setAttributedString_.assert_called_once()

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

    def test_set_response_text_preserves_tool_indicator_fragments(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "open config"

        fragments = []

        def _tool(token):
            fragments.append(("tool", token))
            return MagicMock()

        def _response(token):
            fragments.append(("response", token))
            return MagicMock()

        overlay._make_tool_indicator_fragment = MagicMock(side_effect=_tool)
        overlay._make_response_fragment = MagicMock(side_effect=_response)

        final_text = '\n[calling read_file…]\n  ["/tmp/config.json" · ~12 tokens]\nDone.'

        overlay.set_response_text(final_text)

        assert "".join(token for _, token in fragments) == final_text
        assert any(kind == "tool" for kind, _ in fragments)
        assert any(kind == "response" for kind, _ in fragments)

    def test_append_token_inserts_separator_before_first_response_after_tool_prelude(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "open config"
        overlay._update_layout = MagicMock()

        overlay.append_token("\n[calling read_file…]\n")
        append_count_after_tool = overlay._text_view.textStorage().appendAttributedString_.call_count

        overlay.append_token("Done.")

        # First prose after tool-only prelude still needs the breathing-room separator.
        assert (
            overlay._text_view.textStorage().appendAttributedString_.call_count
            == append_count_after_tool + 2
        )

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

    def test_assistant_text_alpha_floor_and_ceiling_are_punchier(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._assistant_text_alpha_for_breath(0.0) == pytest.approx(0.75)
            assert mod._assistant_text_alpha_for_breath(1.0) == pytest.approx(1.0)
            assert mod.CommandOverlay._TTS_ALPHA_MIN == pytest.approx(0.75)
            assert mod.CommandOverlay._TTS_ALPHA_MAX == pytest.approx(1.0)
        finally:
            sys.modules.pop("spoke.command_overlay", None)

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

        # First call — SDF must be computed
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        # Second call, same dimensions — SDF should be cached
        overlay._brightness = 0.5
        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1, "SDF was recomputed despite identical geometry"

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

        overlay._apply_ridge_masks(600.0, 80.0)
        assert call_count == 1

        overlay._apply_ridge_masks(600.0, 200.0)
        assert call_count == 2, "SDF was not recomputed after height change"
