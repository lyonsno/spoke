"""Tests for CommandOverlay state machines and timer logic.

Tests cover: thinking timer lifecycle, dismiss animation phases,
show/finish/hide state transitions, and timer cancellation.
All tests use mocked PyObjC — no GUI runtime required.
"""

import importlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spoke.optical_shell_metrics import OpticalShellMetrics

def _make_rect(x, y, width, height):
    return SimpleNamespace(
        origin=SimpleNamespace(x=x, y=y),
        size=SimpleNamespace(width=width, height=height),
    )

def _make_overlay(mock_pyobjc):
    """Create a CommandOverlay with mocked internals."""
    # Mixed compositor/overlay runs can leave real PyObjC packages restored
    # between tests. Re-seat the exact fakes this helper was handed before
    # importing command_overlay so a real Quartz package cannot leak in.
    for name in list(sys.modules):
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in ("objc", "Quartz", "Foundation", "AppKit", "PyObjCTools")
        ):
            sys.modules.pop(name, None)
    sys.modules.update(mock_pyobjc)
    sys.modules["Quartz.CoreGraphics"] = mock_pyobjc["Quartz"]
    assert sys.modules["Quartz"] is mock_pyobjc["Quartz"]
    assert not hasattr(sys.modules["Quartz"], "__path__")
    assert hasattr(sys.modules["Quartz"], "CALayer")
    assert hasattr(sys.modules["Quartz"], "CAShapeLayer")
    assert hasattr(sys.modules["Quartz"], "CGPathCreateWithRoundedRect")
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
    overlay._metrics = OpticalShellMetrics()
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
    overlay._cancel_spring = 0.0
    overlay._cancel_spring_target = 0.0
    overlay._cancel_spring_fired = False
    overlay._on_cancel_spring_threshold = None
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
        self.attributes = []

    def addAttribute_value_range_(self, *args):
        self.attributes.append(args)
        return None

    def appendAttributedString_(self, other):
        offset = len(self.text)
        self.text += other.text
        for name, value, (start, length) in getattr(other, "attributes", []):
            self.attributes.append((name, value, (start + offset, length)))


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


def test_quartz_backdrop_renderer_blurs_snapshot_before_render(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.command_overlay", None)
    mod = importlib.import_module("spoke.command_overlay")
    quartz = sys.modules["Quartz"]

    captured = {}

    class FakeImage:
        def __init__(self, label):
            self.label = label
            self._extent = _make_rect(0.0, 0.0, 680.0, 160.0)

        def extent(self):
            return self._extent

        def imageByClampingToExtent(self):
            captured["clamped"] = self.label
            return self

        def imageByCroppingToRect_(self, rect):
            captured["cropped"] = rect
            return self

    class FakeCIImage:
        @staticmethod
        def imageWithCGImage_(image):
            captured["input_image"] = image
            return FakeImage("ci-snapshot")

    class FakeBlurFilter:
        def __init__(self):
            self.values = {}

        def setDefaults(self):
            captured["defaults"] = True

        def setValue_forKey_(self, value, key):
            self.values[key] = value

        def valueForKey_(self, key):
            assert key == "outputImage"
            captured["blurred_radius"] = self.values["inputRadius"]
            captured["blurred_input"] = self.values["inputImage"].label
            return self.values["inputImage"]

    class FakeCIFilter:
        @staticmethod
        def filterWithName_(name):
            captured["filter"] = name
            return FakeBlurFilter()

    quartz.CGWindowListCreateImage = MagicMock(return_value="sharp-snapshot")
    quartz.kCGWindowListOptionOnScreenBelowWindow = 2
    quartz.CIImage = FakeCIImage
    quartz.CIFilter = FakeCIFilter

    renderer = mod._QuartzBackdropRenderer()
    context = MagicMock()
    context.createCGImage_fromRect_.return_value = "blurred-snapshot"
    renderer._context = MagicMock(return_value=context)

    image = renderer.capture_blurred_image(
        window_number=17,
        capture_rect=_make_rect(100.0, 200.0, 680.0, 160.0),
        blur_radius_points=5.4,
    )

    assert image == "blurred-snapshot"
    assert captured["filter"] == "CIGaussianBlur"
    assert captured["blurred_radius"] == pytest.approx(5.4)
    assert captured["blurred_input"] == "ci-snapshot"
    context.createCGImage_fromRect_.assert_called_once()


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


class TestOpticalShellMaterialization:
    """Assistant optical-shell materialization should be geometry-driven."""

    def test_materialization_starts_as_pressure_slit_before_vertical_bloom(
        self, mock_pyobjc
    ):
        mod = importlib.import_module("spoke.command_overlay")
        base = {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 1200.0,
            "content_height_points": 208.0,
            "corner_radius_points": 32.0,
            "band_width_points": 20.0,
            "tail_width_points": 12.0,
            "ring_amplitude_points": 30.0,
            "tail_amplitude_points": 8.0,
        }

        seed = mod._materialized_optical_shell_config(base, 0.0)
        spread = mod._materialized_optical_shell_config(base, 0.50)
        final = mod._materialized_optical_shell_config(base, 1.0)

        assert seed["center_x"] == pytest.approx(base["center_x"])
        assert seed["center_y"] == pytest.approx(base["center_y"])
        assert seed["content_width_points"] < base["content_width_points"] * 0.20
        assert seed["content_height_points"] < base["content_height_points"] * 0.12
        assert spread["content_width_points"] > base["content_width_points"] * 0.80
        assert spread["content_height_points"] < base["content_height_points"] * 0.45
        assert final == pytest.approx(base)
        assert base["content_width_points"] == pytest.approx(1200.0)

    def test_optical_entrance_waits_for_body_materialization_before_fade(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        compositor = MagicMock()
        compositor.presented_count = 1
        overlay._fullscreen_compositor = compositor
        overlay._fill_hidden_until_signature = None
        overlay._materialization_progress = 0.10

        assert overlay._optical_entrance_ready() is False

        overlay._materialization_progress = 0.75

        assert overlay._optical_entrance_ready() is True

    def test_optical_dismiss_uses_stretched_faster_reverse_timeline(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        shell_config = {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 1200.0,
            "content_height_points": 208.0,
            "corner_radius_points": 32.0,
        }
        scheduled = []
        overlay._fullscreen_compositor = MagicMock()
        overlay._display_local_optical_shell_config = MagicMock(return_value=shell_config)
        overlay._start_materialization_animation = MagicMock()

        def _schedule(interval, _target, selector, _userinfo, _repeats):
            scheduled.append((interval, selector))
            return MagicMock()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_ = MagicMock(
            side_effect=_schedule
        )

        overlay._start_fade_out()

        overlay._start_materialization_animation.assert_called_once_with(
            shell_config,
            direction=-1,
        )
        assert mod._OPTICAL_MATERIALIZATION_DISMISS_S < mod._OPTICAL_MATERIALIZATION_S
        assert mod._OPTICAL_MATERIALIZATION_DISMISS_S == pytest.approx(
            mod._OPTICAL_MATERIALIZATION_S * 0.5
        )
        assert scheduled[-1] == (
            pytest.approx(mod._OPTICAL_MATERIALIZATION_DISMISS_S / mod._FADE_STEPS),
            "fadeStep:",
        )

    def test_toggle_cancel_dismiss_uses_optical_reverse_when_compositor_is_active(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        compositor = MagicMock()
        overlay._fullscreen_compositor = compositor
        overlay._visible = True
        overlay._streaming = True
        overlay._start_fade_out = MagicMock()
        overlay._set_overlay_scale = MagicMock()

        overlay.cancel_dismiss()

        assert overlay._visible is False
        assert overlay._streaming is False
        overlay._window.setAlphaValue_.assert_called_with(1.0)
        overlay._set_overlay_scale.assert_called_once_with(1.0)
        overlay._start_fade_out.assert_called_once()
        assert overlay._cancel_timer_anim is None

    def test_materialization_choreographs_core_magnification_overshoot(self, mock_pyobjc):
        mod = importlib.import_module("spoke.command_overlay")
        base = {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 1200.0,
            "content_height_points": 208.0,
            "corner_radius_points": 32.0,
            "core_magnification": 14.0,
        }

        seed = mod._materialized_optical_shell_config(base, 0.0)
        early = mod._materialized_optical_shell_config(base, 0.20)
        surge = mod._materialized_optical_shell_config(base, 0.62)
        peak = mod._materialized_optical_shell_config(base, 0.72)
        settle = mod._materialized_optical_shell_config(base, 0.90)
        final = mod._materialized_optical_shell_config(base, 1.0)

        assert seed["core_magnification"] < base["core_magnification"] * 0.10
        assert early["core_magnification"] > seed["core_magnification"]
        assert surge["core_magnification"] > base["core_magnification"]
        assert peak["core_magnification"] == pytest.approx(
            base["core_magnification"] * 1.20
        )
        assert settle["core_magnification"] > base["core_magnification"]
        assert settle["core_magnification"] < peak["core_magnification"]
        assert final["core_magnification"] == pytest.approx(base["core_magnification"])

    def test_material_fill_lags_warp_spread_then_blooms_vertically(self, mock_pyobjc):
        mod = importlib.import_module("spoke.command_overlay")

        seed = mod._materialization_fill_state(0.0)
        wide_warp = mod._materialization_fill_state(0.55)
        solid_slit = mod._materialization_fill_state(0.62)
        blooming = mod._materialization_fill_state(0.75)
        full = mod._materialization_fill_state(1.0)

        assert seed["opacity"] == pytest.approx(0.0)
        assert seed["height_frac"] < 0.04
        assert wide_warp["opacity"] == pytest.approx(0.0)
        assert wide_warp["height_frac"] < 0.04
        assert solid_slit["opacity"] > 0.95
        assert solid_slit["height_frac"] < 0.20
        assert blooming["height_frac"] > 0.85
        assert full["opacity"] == pytest.approx(1.0)
        assert full["height_frac"] == pytest.approx(1.0)

    def test_materialization_step_updates_fill_layer_without_window_alpha_fade(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._fullscreen_compositor = MagicMock()
        overlay._materialization_timer = MagicMock()
        overlay._materialization_final_shell_config = {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 1200.0,
            "content_height_points": 208.0,
            "corner_radius_points": 32.0,
        }
        overlay._materialization_direction = 1
        overlay._materialization_started_at = 0.0
        monkeypatch.setattr(mod.time, "perf_counter", lambda: mod._OPTICAL_MATERIALIZATION_S)

        overlay.materializationStep_(overlay._materialization_timer)

        overlay._fill_layer.setFrame_.assert_called()
        overlay._fill_layer.setOpacity_.assert_called()
        assert overlay._window.setAlphaValue_.call_args is None

    def test_optical_fade_out_keeps_window_opaque_until_reverse_collapse_finishes(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._fullscreen_compositor = MagicMock()
        overlay._fade_direction = -1
        overlay._fade_from = 1.0
        overlay._fade_step = 0

        overlay.fadeStep_(None)

        overlay._window.setAlphaValue_.assert_called_with(1.0)

    def test_pulse_does_not_override_fill_opacity_during_materialization(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._fullscreen_compositor = MagicMock()
        overlay._fullscreen_compositor.sampled_brightness = 0.0
        overlay._materialization_timer = MagicMock()
        overlay._materialization_progress = 0.30
        overlay._text_view.textStorage.return_value.length.return_value = 0

        overlay._pulseStepInner()

        assert overlay._fill_layer.setOpacity_.call_args_list
        for call in overlay._fill_layer.setOpacity_.call_args_list:
            assert call.args[0] == pytest.approx(
                mod._materialization_fill_state(0.30)["opacity"]
            )

    def test_pulse_does_not_regenerate_punchthrough_mask_during_materialization(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._fullscreen_compositor = MagicMock()
        overlay._fullscreen_compositor.sampled_brightness = 0.0
        overlay._materialization_timer = MagicMock()
        overlay._materialization_progress = 0.30
        overlay._text_punchthrough = True
        overlay._text_view.textStorage.return_value.length.return_value = 12
        overlay._update_punchthrough_mask = MagicMock()

        overlay._pulseStepInner()

        overlay._update_punchthrough_mask.assert_not_called()

    def test_fill_image_ready_preserves_active_materialization_geometry(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._materialization_timer = MagicMock()
        overlay._materialization_progress = 0.55
        overlay._desired_fill_image_signature = ("sig",)
        overlay._pending_fill_image_signature = ("sig",)

        overlay.fillImageReady_(
            {
                "signature": ("sig",),
                "total_w": 680.0,
                "total_h": 160.0,
                "scale": 2.0,
                "image": "fill-image",
                "payload": b"payload",
                "has_compositor": True,
            }
        )

        last_frame = overlay._fill_layer.setFrame_.call_args[0][0]
        feather = (
            mod._OPTICAL_SHELL_FEATHER
            if mod._COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
            else mod._OUTER_FEATHER
        )
        total_h = 104.0 + 2 * feather
        expected_h = total_h * mod._materialization_fill_state(0.55)["height_frac"]
        assert last_frame[1][1] == pytest.approx(expected_h)

    def test_ridge_mask_refresh_preserves_active_materialization_geometry(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: None)
        overlay._materialization_timer = MagicMock()
        overlay._materialization_progress = 0.55

        overlay._apply_ridge_masks(600.0, 80.0)

        last_frame = overlay._fill_layer.setFrame_.call_args[0][0]
        feather = (
            mod._OPTICAL_SHELL_FEATHER
            if mod._COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED
            else mod._OUTER_FEATHER
        )
        total_h = 104.0 + 2 * feather
        expected_h = total_h * mod._materialization_fill_state(0.55)["height_frac"]
        assert last_frame[1][1] == pytest.approx(expected_h)

    def test_materialization_temporarily_clears_punchthrough_mask(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._fullscreen_compositor = MagicMock()
        overlay._text_punchthrough = True
        overlay._punchthrough_mask_layer = MagicMock()

        overlay._start_materialization_animation(
            {
                "center_x": 640.0,
                "center_y": 1160.0,
                "content_width_points": 1200.0,
                "content_height_points": 208.0,
                "corner_radius_points": 32.0,
            }
        )

        overlay._fill_layer.setMask_.assert_called_with(None)
        assert overlay._punchthrough_mask_layer is None
        assert overlay._punchthrough_mask_dirty is True

    def test_entrance_materialization_rebuilds_punchthrough_mask_after_full_body(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._fullscreen_compositor = MagicMock()
        overlay._materialization_timer = MagicMock()
        overlay._materialization_final_shell_config = {
            "center_x": 640.0,
            "center_y": 1160.0,
            "content_width_points": 1200.0,
            "content_height_points": 208.0,
            "corner_radius_points": 32.0,
        }
        overlay._materialization_direction = 1
        overlay._materialization_started_at = 0.0
        overlay._text_punchthrough = True
        overlay._refresh_punchthrough_mask_if_needed = MagicMock()
        monkeypatch.setattr(mod.time, "perf_counter", lambda: mod._OPTICAL_MATERIALIZATION_S)

        overlay.materializationStep_(overlay._materialization_timer)

        overlay._refresh_punchthrough_mask_if_needed.assert_called_once()


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

    def test_show_does_not_start_independent_brightness_timer(self, mock_pyobjc):
        # Rat 6 fix: CommandOverlay no longer runs its own Quartz screen-capture
        # timer.  GlowOverlay owns the single 1 Hz brightness sample; brightness
        # is pushed here via set_brightness() on amplitude ticks.
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.show()

        assert overlay._brightness_timer is None

    def test_pulse_step_records_display_and_presented_ticks(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)

        overlay.pulseStep_(None)

        snapshot = overlay._metrics.snapshot()
        assert snapshot["display_link_ticks"] == 1
        assert snapshot["presented_frames"] == 1

    def test_show_uses_externally_seeded_brightness_before_first_theme(self, mock_pyobjc, monkeypatch):
        # Rat 7 fix: show() must not do a synchronous Quartz screen capture.
        # When set_brightness() has been called before show() (the normal path
        # via __main__._sync_command_overlay_brightness), that cached value is
        # committed directly.  The _sample_screen_brightness_for_overlay
        # fallback must not fire.
        overlay, mod = _make_overlay(mock_pyobjc)
        # Simulate __main__ calling set_brightness(immediate=True) before show()
        overlay.set_brightness(0.86, immediate=True)
        observed = []
        # Patch the capture function to detect if it fires (it must not)
        capture_calls = []
        monkeypatch.setattr(
            mod, "_sample_screen_brightness_for_overlay",
            lambda _screen: capture_calls.append(True) or 0.0
        )
        overlay._apply_surface_theme = MagicMock(
            side_effect=lambda: observed.append(overlay._brightness)
        )

        overlay.show(start_thinking_timer=False)

        assert capture_calls == [], "synchronous screen capture must not fire when brightness is cached"
        assert observed[0] == pytest.approx(0.86)
        assert overlay._brightness == pytest.approx(0.86)
        assert overlay._brightness_target == pytest.approx(0.86)

    def test_show_uses_neutral_brightness_when_no_external_seed(self, mock_pyobjc, monkeypatch):
        # Rat 7 fix: on first-ever show (no set_brightness() call yet), fall
        # back to 0.5 neutral and skip the sync screen capture.
        overlay, mod = _make_overlay(mock_pyobjc)
        # Do NOT call set_brightness() — _brightness_seeded_externally stays False
        capture_calls = []
        monkeypatch.setattr(
            mod, "_sample_screen_brightness_for_overlay",
            lambda _screen: capture_calls.append(True) or 0.0
        )

        overlay.show(start_thinking_timer=False)

        assert capture_calls == [], "sync screen capture must not fire even on first show"
        assert overlay._brightness == pytest.approx(0.5)
        assert overlay._brightness_target == pytest.approx(0.5)

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

        assert mod._FADE_IN_S <= 0.17

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

    def test_setup_owns_command_overlay_created_log(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")

        setup_source = inspect.getsource(mod.CommandOverlay.setup)
        layer_choice_source = inspect.getsource(mod.CommandOverlay._choose_backdrop_layer_class)

        assert "Command overlay created" in setup_source
        assert "Command overlay created" not in layer_choice_source

    def test_hide_clears_visible_and_streaming(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True

        overlay.hide()
        assert overlay._visible is False
        assert overlay._streaming is False

    def test_hide_cancels_pulse_before_fade_for_heavy_text(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._streaming = True
        pulse_timer = MagicMock()
        overlay._pulse_timer = pulse_timer

        overlay.hide()

        pulse_timer.invalidate.assert_called_once()
        assert overlay._pulse_timer is None

    def test_recording_load_shed_freezes_and_resumes_pulse(self, mock_pyobjc):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        pulse_timer = MagicMock()
        overlay._pulse_timer = pulse_timer
        overlay._start_pulse_timer = MagicMock()

        overlay.set_recording_load_shed(True)

        pulse_timer.invalidate.assert_called_once()
        assert overlay._pulse_timer is None
        assert overlay._recording_load_shed is True

        overlay.set_recording_load_shed(False)

        assert overlay._recording_load_shed is False
        overlay._start_pulse_timer.assert_called_once()

    def test_fade_in_defers_pulse_while_recording_load_shed_active(
        self, mock_pyobjc
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._recording_load_shed = True
        overlay._fade_direction = 1
        overlay._fade_step = mod._FADE_STEPS - 1
        overlay._start_pulse_timer = MagicMock()

        overlay.fadeStep_(None)

        overlay._start_pulse_timer.assert_not_called()

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

    def test_show_with_initial_transcript_builds_response_before_chroma_pulse(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        chroma_states = []

        def _fragment(_token):
            chroma_states.append(getattr(overlay, "_response_chroma_active", None))
            return _FakeAttributedString("")

        overlay._make_response_fragment = MagicMock(side_effect=_fragment)

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert chroma_states == [False]

    def test_optical_show_with_initial_transcript_hides_plain_text_before_front(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        events = []
        overlay._scroll_view.setHidden_.side_effect = (
            lambda hidden: events.append(("scroll_hidden", hidden))
        )
        overlay._window.orderFrontRegardless.side_effect = lambda: events.append(
            ("front", None)
        )

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert ("scroll_hidden", True) in events
        assert events.index(("scroll_hidden", True)) < events.index(("front", None))

    def test_optical_show_with_prompt_only_arms_visual_stack_before_fade(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        events = []

        def _schedule(_interval, _target, selector, _userinfo, _repeats):
            events.append(("timer", selector))
            return MagicMock()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_ = MagicMock(
            side_effect=_schedule
        )
        overlay._scroll_view.setHidden_.side_effect = (
            lambda hidden: events.append(("scroll_hidden", hidden))
        )
        overlay._window.orderFrontRegardless.side_effect = lambda: events.append(
            ("front", None)
        )
        overlay._start_fullscreen_compositor = MagicMock(
            side_effect=lambda: events.append(("compositor", None))
        )
        overlay._refresh_punchthrough_mask_if_needed = MagicMock(
            side_effect=lambda: events.append(("mask", None))
        )

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
        )

        assert overlay._utterance_text == "User prompt"
        assert ("scroll_hidden", True) in events
        assert ("compositor", None) in events
        assert ("mask", None) in events
        assert ("timer", "visualStart:") not in events
        assert events.index(("scroll_hidden", True)) < events.index(("front", None))
        assert events.index(("front", None)) < events.index(("compositor", None))
        assert events.index(("mask", None)) < events.index(("timer", "fadeStep:"))

    def test_optical_show_waits_for_first_compositor_frame_before_entrance(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        events = []

        class PendingCompositor:
            presented_count = 0

        def _schedule(_interval, _target, selector, _userinfo, _repeats):
            events.append(("timer", selector))
            return MagicMock()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_ = MagicMock(
            side_effect=_schedule
        )

        def _start_compositor():
            events.append(("compositor", None))
            overlay._fullscreen_compositor = PendingCompositor()

        overlay._start_fullscreen_compositor = MagicMock(side_effect=_start_compositor)

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
        )

        assert ("compositor", None) in events
        assert ("timer", "visualReadyStep:") in events
        assert ("timer", "_entrancePopStep:") not in events
        assert ("timer", "fadeStep:") not in events

    def test_optical_show_waits_for_materialized_body_even_after_first_compositor_frame(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        events = []

        class PresentedCompositor:
            presented_count = 1
            sampled_brightness = 0.5

            def refresh_brightness(self):
                events.append(("brightness", None))

        def _schedule(_interval, _target, selector, _userinfo, _repeats):
            events.append(("timer", selector))
            return MagicMock()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_ = MagicMock(
            side_effect=_schedule
        )

        def _start_compositor():
            events.append(("compositor", None))
            overlay._fullscreen_compositor = PresentedCompositor()
            overlay._materialization_progress = 0.10

        overlay._start_fullscreen_compositor = MagicMock(side_effect=_start_compositor)

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
        )

        assert ("compositor", None) in events
        assert ("brightness", None) in events
        assert ("timer", "visualReadyStep:") in events
        assert ("timer", "_entrancePopStep:") not in events
        assert ("timer", "fadeStep:") not in events

    def test_optical_show_with_initial_transcript_arms_visual_stack_before_fade(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, mod = _make_overlay(mock_pyobjc)
        events = []

        def _schedule(_interval, _target, selector, _userinfo, _repeats):
            events.append(("timer", selector))
            return MagicMock()

        mod.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_ = MagicMock(
            side_effect=_schedule
        )
        overlay._window.orderFrontRegardless.side_effect = lambda: events.append(
            ("front", None)
        )
        overlay._start_fullscreen_compositor = MagicMock(
            side_effect=lambda: events.append(("compositor", None))
        )
        overlay._refresh_punchthrough_mask_if_needed = MagicMock(
            side_effect=lambda: events.append(("mask", None))
        )

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert ("compositor", None) in events
        assert ("mask", None) in events
        assert ("timer", "visualStart:") not in events
        assert events.index(("front", None)) < events.index(("compositor", None))
        assert events.index(("mask", None)) < events.index(("timer", "fadeStep:"))

    def test_visual_start_refreshes_punchthrough_mask_after_deferred_compositor_start(
        self, mock_pyobjc, monkeypatch
    ):
        monkeypatch.setenv("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", "1")
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        timer = MagicMock()
        overlay._visual_start_timer = timer

        def _start_compositor():
            overlay._fullscreen_compositor = MagicMock()

        overlay._start_fullscreen_compositor = MagicMock(side_effect=_start_compositor)
        overlay._refresh_punchthrough_mask_if_needed = MagicMock()

        overlay.visualStart_(timer)

        overlay._start_fullscreen_compositor.assert_called_once()
        overlay._refresh_punchthrough_mask_if_needed.assert_called_once()

    def test_show_with_initial_transcript_skips_default_shell_fill_build(
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
        calls = []
        overlay._apply_ridge_masks = MagicMock(side_effect=lambda *args: calls.append(args))

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert (mod._OVERLAY_WIDTH, mod._OVERLAY_HEIGHT) not in calls
        assert calls, "initial transcript should still build the resized shell"

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

    def test_show_invalidates_stale_fill_brightness_before_first_theme(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._brightness = 0.86
        overlay._brightness_target = 0.86
        overlay._brightness_seeded_externally = True
        overlay._fill_image_brightness = 0.12
        observed = []
        overlay._apply_surface_theme = MagicMock(
            side_effect=lambda: observed.append(overlay._fill_image_brightness)
        )

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert observed[0] == -1.0

    def test_show_hides_stale_fill_until_first_paint_rebuild_is_ready(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        queued = []
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: queued.append(work))
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._fill_image_signature = ("old-dark-fill",)
        overlay._fill_payload = b"old-fill"
        overlay.set_brightness(0.86, immediate=True)
        overlay._fill_layer.reset_mock()

        overlay.show(
            start_thinking_timer=False,
            initial_utterance="User prompt",
            initial_response="Assistant response",
        )

        assert queued, "first-paint fill rebuild should be pending asynchronously"
        overlay._fill_layer.setHidden_.assert_called_with(True)
        pending_signature = overlay._desired_fill_image_signature
        assert overlay._fill_hidden_until_signature == pending_signature

        overlay.fillImageReady_(
            {
                "signature": pending_signature,
                "geom_key": ("geom",),
                "fallback_alpha": None,
                "raw_interior": None,
                "edge_ridge": None,
                "inside_mask": None,
                "ext_exterior": None,
                "cached_fill_alpha": None,
                "sdf_appearance_b": -1.0,
                "payload": b"fresh-fill",
                "total_w": 680.0,
                "total_h": 160.0,
                "scale": 2.0,
                "image": "fresh-fill-image",
                "has_compositor": False,
            }
        )

        overlay._fill_layer.setHidden_.assert_called_with(False)
        assert overlay._fill_hidden_until_signature is None

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

    def test_late_thinking_topic_updates_collapsed_line_after_response_started(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._utterance_text = "User prompt"
        overlay._collapsed_text = "Thought for 4s"
        overlay._response_text = "Done."
        overlay.append_token = MagicMock()
        _install_fake_attributed_string(monkeypatch)

        overlay.set_thinking_collapsed(" · route planning")

        assert overlay._collapsed_text == "Thought for 4s · route planning"
        combined = (
            overlay._text_view.textStorage()
            .setAttributedString_.call_args[0][0]
            .text
        )
        assert "Thought for 4s · route planning" in combined
        overlay.append_token.assert_called_once_with("Done.")

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

    def test_response_fragment_styles_approval_card_sections(
        self, mock_pyobjc, monkeypatch
    ):
        _install_fake_attributed_string(monkeypatch)
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._brightness = 0.2
        appkit = sys.modules["AppKit"]
        appkit.NSFont.systemFontOfSize_weight_.side_effect = (
            lambda size, weight: ("system", size, weight)
        )
        appkit.NSFont.monospacedSystemFontOfSize_weight_.side_effect = (
            lambda size, weight: ("mono", size, weight)
        )
        text = (
            "Approval needed\n"
            "Enter to run  ·  Delete to cancel  ·  speak or type to revise\n\n"
            "git commit -m x\n\n"
            "reason: command requires approval\n"
            "cwd: /tmp/repo"
        )

        frag = overlay._make_response_fragment(text)
        attrs_by_range = {
            attr_range: value
            for name, value, attr_range in frag.attributes
            if name == "NSFont"
        }

        ranges = mod._approval_card_ranges(text)
        assert attrs_by_range[ranges["header"]] == ("system", 15.0, 0.22)
        assert attrs_by_range[ranges["action"]] == ("system", 12.0, 0.0)
        assert attrs_by_range[ranges["command"]] == ("mono", 16.0, 0.12)
        assert attrs_by_range[ranges["reason"]] == ("system", 12.5, -0.05)
        assert attrs_by_range[ranges["cwd"]] == ("system", 12.0, -0.05)

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
        # _brightness_timer is intentionally None after Rat 6 fix —
        # CommandOverlay no longer owns a brightness sampling timer.
        assert overlay._brightness_timer is None

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

    def test_punchthrough_boost_style_is_bidirectional(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            dark_rgb, dark_opacity = mod._punchthrough_boost_style_for_brightness(0.0)
            light_rgb, light_opacity = mod._punchthrough_boost_style_for_brightness(1.0)

            assert max(dark_rgb) < 0.08
            assert dark_opacity > 0.25
            assert min(light_rgb) > 0.92
            assert light_opacity > dark_opacity
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_brightness_resample_updates_target_while_visible(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._screen = MagicMock()

        monkeypatch.setattr(mod, "_sample_screen_brightness_for_overlay", lambda _screen: 0.83)

        overlay.brightnessResample_(None)

        assert overlay._brightness_target == pytest.approx(0.83)

    def test_brightness_resample_records_sample_count(self, mock_pyobjc, monkeypatch):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._screen = MagicMock()
        monkeypatch.setattr(mod, "_sample_screen_brightness_for_overlay", lambda _screen: 0.83)

        overlay.brightnessResample_(None)

        snapshot = overlay._metrics.snapshot()
        assert snapshot["brightness_samples"] == 1

    def test_brightness_resample_is_ignored_while_compositor_owns_brightness(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._visible = True
        overlay._screen = MagicMock()
        overlay._brightness_target = 0.16
        overlay._fullscreen_compositor = MagicMock()

        monkeypatch.setattr(mod, "_sample_screen_brightness_for_overlay", lambda _screen: 0.91)

        overlay.brightnessResample_(None)

        assert overlay._brightness_target == pytest.approx(0.16)

    def test_brightness_crossing_reaches_contrast_band_in_one_pulse(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._advance_command_brightness(0.0, 1.0) > 0.56
            assert mod._advance_command_brightness(1.0, 0.0) < 0.44
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_brightness_crossing_nearly_settles_in_two_pulses(self, mock_pyobjc):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            up = mod._advance_command_brightness(
                mod._advance_command_brightness(0.0, 1.0),
                1.0,
            )
            down = mod._advance_command_brightness(
                mod._advance_command_brightness(1.0, 0.0),
                0.0,
            )

            assert up > 0.92
            assert down < 0.08
        finally:
            sys.modules.pop("spoke.command_overlay", None)

    def test_compositor_brightness_samples_at_least_every_other_pulse(
        self, mock_pyobjc
    ):
        sys.modules.pop("spoke.command_overlay", None)
        mod = importlib.import_module("spoke.command_overlay")
        try:
            assert mod._BRIGHTNESS_COMPOSITOR_SAMPLE_TICKS <= 2
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

    def test_fullscreen_compositor_start_uses_stable_assistant_client_identity(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        captured = {}

        import spoke.fullscreen_compositor as fullscreen_compositor

        def fake_start_overlay_compositor(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            fullscreen_compositor,
            "start_overlay_compositor",
            fake_start_overlay_compositor,
        )

        overlay._start_fullscreen_compositor()

        assert captured["client_id"] == "assistant.command"
        assert captured["role"] == "assistant"
        assert captured["screen"] is overlay._screen
        assert captured["window"] is overlay._window
        assert captured["content_view"] is overlay._content_view
        assert "shell_config" in captured

    def test_fullscreen_compositor_receives_current_brightness_seed(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        overlay._brightness = 0.07
        captured = {}
        compositor = MagicMock()

        import spoke.fullscreen_compositor as fullscreen_compositor

        def fake_start_overlay_compositor(**kwargs):
            captured.update(kwargs)
            return compositor

        monkeypatch.setattr(
            fullscreen_compositor,
            "start_overlay_compositor",
            fake_start_overlay_compositor,
        )

        overlay._start_fullscreen_compositor()

        assert captured["shell_config"]["initial_brightness"] == pytest.approx(0.07)

    def test_fullscreen_compositor_receives_display_local_shell_center(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        overlay._screen.frame.return_value = _make_rect(1920.0, 100.0, 1280.0, 720.0)
        overlay._window.frame.return_value = _make_rect(2020.0, 160.0, 680.0, 160.0)
        overlay._content_view.frame.return_value = _make_rect(20.0, 30.0, 400.0, 100.0)
        overlay._screen.backingScaleFactor.return_value = 2.0
        captured = {}
        compositor = MagicMock()

        import spoke.fullscreen_compositor as fullscreen_compositor

        def fake_start_overlay_compositor(**kwargs):
            captured.update(kwargs)
            return compositor

        monkeypatch.setattr(
            fullscreen_compositor,
            "start_overlay_compositor",
            fake_start_overlay_compositor,
        )

        overlay._start_fullscreen_compositor()

        shell_config = captured["shell_config"]
        assert shell_config["center_x"] == pytest.approx(640.0)
        assert shell_config["center_y"] == pytest.approx(1160.0)

    def test_fullscreen_compositor_start_arms_brightness_startup_grace(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        overlay._brightness = 0.14
        overlay._brightness_target = 0.21
        overlay._brightness_sample_tick = 11
        compositor = MagicMock()

        import spoke.fullscreen_compositor as fullscreen_compositor

        monkeypatch.setattr(
            fullscreen_compositor,
            "start_overlay_compositor",
            lambda **_kwargs: compositor,
        )

        overlay._start_fullscreen_compositor()

        assert overlay._brightness_sample_tick < 0
        assert overlay._brightness_target == pytest.approx(0.14)

    def test_fullscreen_compositor_start_cancels_legacy_screen_brightness_timer(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        timer = MagicMock()
        overlay._brightness_timer = timer

        import spoke.fullscreen_compositor as fullscreen_compositor

        monkeypatch.setattr(
            fullscreen_compositor,
            "start_overlay_compositor",
            lambda **_kwargs: MagicMock(),
        )

        overlay._start_fullscreen_compositor()

        timer.invalidate.assert_called_once()
        assert overlay._brightness_timer is None

    def test_compositor_startup_grace_does_not_resample_seeded_brightness(
        self, mock_pyobjc
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._text_view.textStorage.return_value.length.return_value = 0
        overlay._brightness = 0.08
        overlay._brightness_target = 0.08
        overlay._brightness_sample_tick = -2
        overlay._cancel_spring = 0.0
        overlay._cancel_spring_target = 0.0
        overlay._cancel_spring_fired = False
        overlay._on_cancel_spring_threshold = None
        overlay._text_punchthrough = False
        overlay._apply_surface_theme = MagicMock()
        overlay._apply_backdrop_pulse_style = MagicMock()

        class _UnstableStartupCompositor:
            def __init__(self):
                self.refresh_calls = 0
                self.sample_reads = 0

            def refresh_brightness(self):
                self.refresh_calls += 1

            @property
            def sampled_brightness(self):
                self.sample_reads += 1
                return 0.92

        compositor = _UnstableStartupCompositor()
        overlay._fullscreen_compositor = compositor

        overlay._pulseStepInner()

        assert compositor.refresh_calls == 0
        assert compositor.sample_reads == 0
        assert overlay._brightness == pytest.approx(0.08)
        assert overlay._brightness_target == pytest.approx(0.08)
        assert overlay._brightness_sample_tick == -1

    def test_dark_punchthrough_uses_dark_glyph_boost_layer(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, _ = _make_overlay(mock_pyobjc)
        overlay._text_view.textStorage.return_value.length.return_value = 0
        overlay._brightness = 0.0
        overlay._brightness_target = 0.0
        overlay._cancel_spring = 0.0
        overlay._cancel_spring_target = 0.0
        overlay._cancel_spring_fired = False
        overlay._on_cancel_spring_threshold = None
        overlay._text_punchthrough = True
        overlay._boost_mask_layer = MagicMock()
        overlay._apply_surface_theme = MagicMock()
        overlay._apply_backdrop_pulse_style = MagicMock()
        overlay._update_punchthrough_mask = MagicMock()
        overlay._fullscreen_compositor = MagicMock(sampled_brightness=0.0)

        monkeypatch.setattr(
            sys.modules["Quartz"],
            "CGColorCreateSRGB",
            lambda r, g, b, a: (r, g, b, a),
            raising=False,
        )

        overlay._pulseStepInner()

        color = overlay._boost_layer.setBackgroundColor_.call_args[0][0]
        assert max(color[:3]) < 0.08
        overlay._boost_layer.setHidden_.assert_called_with(False)

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
            # 3-span bulk path: FG + FN (full range) + up to 3 shadow spans = at most 5.
            # The important invariant is: not O(N) per-character calls.
            assert len(response_attrs) <= 6
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

    def test_pulse_skips_response_text_restyle_when_punchthrough_renders_text(
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

            overlay, _ = _make_overlay(mock_pyobjc)
            overlay._text_view = MagicMock()
            response = "A full paragraph of assistant text that punch-through renders"
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
            overlay._text_punchthrough = True
            overlay._punchthrough_mask_dirty = False
            overlay._fullscreen_compositor = MagicMock(sampled_brightness=0.0)
            overlay._apply_surface_theme = MagicMock()
            overlay._apply_backdrop_pulse_style = MagicMock()
            overlay._update_punchthrough_mask = MagicMock()

            overlay._pulseStepInner()

            response_start = len("Prompt\n\n")
            response_attrs = [
                rng
                for _name, _value, rng in storage.attrs
                if rng[0] >= response_start
            ]
            assert response_attrs == []
        finally:
            sys.modules.pop("spoke.command_overlay", None)

class TestGeometryCaps:
    def test_update_layout_keeps_live_narrator_for_moderate_user_prompt(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(38.0)
        overlay._text_view.textContainer.return_value = object()
        overlay._response_text = ""
        overlay._narrator_label = MagicMock()
        overlay._narrator_label.isHidden.return_value = False
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj

        overlay._update_layout()

        overlay._narrator_label.setHidden_.assert_not_called()

    def test_update_layout_hides_live_narrator_before_wrapped_prompt_overlap(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(56.0)
        overlay._text_view.textContainer.return_value = object()
        overlay._response_text = ""
        overlay._narrator_label = MagicMock()
        overlay._narrator_label.isHidden.return_value = False
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj

        overlay._update_layout()

        overlay._narrator_label.setHidden_.assert_called_with(True)

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

    def test_update_layout_sends_display_local_shell_center_to_compositor(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        overlay._screen.frame.return_value = _make_rect(1920.0, 100.0, 1280.0, 720.0)
        overlay._window.frame.return_value = _make_rect(2020.0, 160.0, 680.0, 160.0)
        overlay._content_view.frame.return_value = _make_rect(20.0, 30.0, 400.0, 100.0)
        overlay._screen.backingScaleFactor.return_value = 2.0
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(280.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj
        compositor = MagicMock()
        overlay._fullscreen_compositor = compositor

        overlay._update_layout()

        shell_config = compositor.update_shell_config.call_args[0][0]
        assert shell_config["center_x"] == pytest.approx(640.0)
        assert shell_config["center_y"] == pytest.approx(1160.0)

    def test_command_overlay_layout_update_publishes_fresh_shell_config_via_session(
        self, mock_pyobjc, monkeypatch
    ):
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
        monkeypatch.setattr(mod, "_COMMAND_BACKDROP_OPTICAL_SHELL_ENABLED", True)
        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._content_view.frame.return_value = _make_rect(28.0, 28.0, 624.0, 104.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(280.0)
        overlay._text_view.textContainer.return_value = object()
        string_obj = MagicMock()
        string_obj.length.return_value = 0
        overlay._text_view.string.return_value = string_obj
        session = MagicMock()
        overlay._fullscreen_compositor = session

        overlay._update_layout()
        first_config = session.update_shell_config.call_args[0][0]
        first_config["center_x"] = -1.0

        overlay._window.frame.return_value = _make_rect(0.0, 260.0, 680.0, 160.0)
        overlay._text_view.layoutManager.return_value = _FakeLayoutManager(320.0)
        overlay._update_layout()
        second_config = session.update_shell_config.call_args[0][0]

        assert session.update_shell_config.call_count == 2
        assert second_config is not first_config
        assert second_config["center_x"] != -1.0

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

    def test_pending_fill_generation_still_updates_visible_frame(
        self, mock_pyobjc, monkeypatch
    ):
        """A resize while a fill worker is pending must not leave the old shell size visible."""
        overlay, mod = _make_overlay(mock_pyobjc)
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: None)

        overlay._apply_ridge_masks(600.0, 80.0)
        overlay._fill_layer.setFrame_.reset_mock()
        overlay._apply_ridge_masks(600.0, 240.0)

        overlay._fill_layer.setFrame_.assert_called_with(
            (
                (0, 0),
                (600.0 + 2 * mod._OUTER_FEATHER, 240.0 + 2 * mod._OUTER_FEATHER),
            )
        )

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

    def test_fill_sdf_uses_command_overlay_corner_radius(self, mock_pyobjc, monkeypatch):
        """The command shell fill should honor the command overlay radius override."""
        overlay, mod = _make_overlay(mock_pyobjc)
        overlay._spring_tint_layer = None

        import numpy as np
        import spoke.overlay as ov_mod

        radii = []

        def capture_sdf(total_w, total_h, width, height, corner_radius, scale):
            radii.append(corner_radius)
            return np.zeros((4, 4), dtype=np.float32)

        monkeypatch.setattr(mod, "_OVERLAY_CORNER_RADIUS", 32.0)
        monkeypatch.setattr(ov_mod, "_OVERLAY_CORNER_RADIUS", 16.0)
        monkeypatch.setattr(ov_mod, "_overlay_rounded_rect_sdf", capture_sdf)
        monkeypatch.setattr(
            ov_mod,
            "_glow_fill_alpha",
            lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32),
        )
        monkeypatch.setattr(ov_mod, "_fill_field_to_image", lambda *_args: ("image", b"payload"))
        monkeypatch.setattr(mod, "_start_overlay_fill_worker", lambda work: work())

        overlay._apply_ridge_masks(600.0, 80.0)

        assert radii == [32.0]
