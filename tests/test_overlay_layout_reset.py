"""Regression tests for overlay layout reset between recording sessions."""

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


class _FakeScreen:
    def __init__(self, width=1440.0, height=900.0):
        self._frame = _make_rect(0.0, 0.0, width, height)

    def frame(self):
        return self._frame


class _FakeWindow:
    def __init__(self):
        self._frame = _make_rect(0.0, 0.0, 680.0, 160.0)
        self.alpha = None
        self.ordered_front = False

    def frame(self):
        return self._frame

    def setFrame_display_animate_(self, frame, display, animate):
        self._frame = frame

    def setAlphaValue_(self, alpha):
        self.alpha = alpha

    def orderFrontRegardless(self):
        self.ordered_front = True


class _FakeView:
    def __init__(self, frame):
        self._frame = frame
        self._layer = _FakeLayer(frame)

    def setFrame_(self, frame):
        self._frame = frame
        self._layer.setFrame_(frame)

    def frame(self):
        return self._frame

    def layer(self):
        return self._layer


class _FakeClipView:
    def __init__(self, y_offset):
        self.origin = SimpleNamespace(x=0.0, y=y_offset)

    def scrollToPoint_(self, point):
        self.origin = SimpleNamespace(x=point[0], y=point[1])

    def setBoundsOrigin_(self, point):
        self.scrollToPoint_(point)

    def bounds(self):
        return SimpleNamespace(origin=self.origin)


class _FakeTextContainer:
    def __init__(self):
        self.container_size = None

    def setContainerSize_(self, size):
        self.container_size = size

    def setWidthTracksTextView_(self, tracks):
        self.width_tracks_text_view = tracks


class _FakeTextView:
    def __init__(self, frame, text):
        self._frame = frame
        self._text = text
        self._text_container = _FakeTextContainer()
        self._layout_manager = None
        self.scrolled_range = None
        self.text_color = None
        self.set_string_calls = []

    def setString_(self, text):
        self.set_string_calls.append(text)
        self._text = text

    def string(self):
        return self._text

    def setTextColor_(self, color):
        self.text_color = color

    def setFrame_(self, frame):
        self._frame = frame

    def frame(self):
        return self._frame

    def textContainer(self):
        return self._text_container

    def layoutManager(self):
        return self._layout_manager

    def set_layout_height(self, height):
        self._layout_manager = _FakeLayoutManager(height)

    def scrollRangeToVisible_(self, visible_range):
        self.scrolled_range = visible_range


class _FakeLayoutManager:
    def __init__(self, height):
        self.height = height

    def ensureLayoutForTextContainer_(self, container):
        self._container = container

    def usedRectForTextContainer_(self, container):
        return _make_rect(0.0, 0.0, 0.0, self.height)


class _FakeLayer:
    def __init__(self, frame):
        self._frame = frame
        self._mask = None
        self._path = None
        self._background_color = None
        self._opacity = None
        self.animations = []

    def setFrame_(self, frame):
        if isinstance(frame, tuple):
            origin, size = frame
            self._frame = _make_rect(origin[0], origin[1], size[0], size[1])
            return
        self._frame = frame

    def frame(self):
        return self._frame

    def setMask_(self, mask):
        self._mask = mask

    def mask(self):
        return self._mask

    def setPath_(self, path):
        self._path = path

    def setBackgroundColor_(self, color):
        self._background_color = color

    def backgroundColor(self):
        return self._background_color

    def setOpacity_(self, opacity):
        self._opacity = opacity

    def opacity(self):
        return self._opacity

    def addAnimation_forKey_(self, animation, key):
        self.animations.append((key, animation))


class _FakeScrollView:
    def __init__(self, frame, text_view, y_offset):
        self._frame = frame
        self._document_view = text_view
        self._clip_view = _FakeClipView(y_offset)
        self.reflected_clip_view = None
        self.hidden = False

    def setFrame_(self, frame):
        self._frame = frame

    def frame(self):
        return self._frame

    def documentView(self):
        return self._document_view

    def contentView(self):
        return self._clip_view

    def reflectScrolledClipView_(self, clip_view):
        self.reflected_clip_view = clip_view

    def setHidden_(self, hidden):
        self.hidden = hidden


def _import_overlay(mock_pyobjc):
    sys.modules.pop("spoke.overlay", None)
    mod = importlib.import_module("spoke.overlay")
    mod._start_overlay_fill_worker = lambda work: work()
    return mod


def test_recording_show_cancels_prior_notice(mock_pyobjc, monkeypatch):
    mod = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
    overlay = mod.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(_make_rect(0, 0, 680, 160))
    overlay._text_view = _FakeTextView(_make_rect(0, 0, 680, 160), "notice")
    overlay._scroll_view = _FakeScrollView(_make_rect(0, 0, 680, 160), overlay._text_view, 0)
    timer = overlay._notice_timer = MagicMock()
    overlay.show()
    timer.invalidate.assert_called_once_with()
    assert overlay._notice_timer is None


@pytest.mark.parametrize("optics_fails", [False, True])
def test_show_attributes_sync_optics_cost_and_keeps_plain_pill_on_failure(
    mock_pyobjc, monkeypatch, optics_fails
):
    mod = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(mod, "NSMakeRect", _make_rect)
    overlay = mod.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(_make_rect(0, 0, 680, 160))
    overlay._text_view = _FakeTextView(_make_rect(0, 0, 680, 160), "")
    overlay._scroll_view = _FakeScrollView(_make_rect(0, 0, 680, 160), overlay._text_view, 0)
    now = [100.0]
    # No sleep or machine-speed assertion: only exact phase attribution.
    monkeypatch.setattr(mod, "time", SimpleNamespace(perf_counter=lambda: now[0]), raising=False)
    def optics(**kwargs):
        now[0] += 0.25
        if optics_fails:
            raise RuntimeError("native optics setup failed")
        return True
    overlay._publish_preview_compositor_snapshot = optics
    overlay.show()
    assert overlay._window.ordered_front
    assert overlay._last_show_timing["optics_ms"] == pytest.approx(250)
    assert overlay._last_show_timing["total_ms"] == pytest.approx(250)
    assert overlay._last_show_timing["optics_status"] == ("failed" if optics_fails else "published")
    now[0] += 0.75
    overlay.fadeStep_(overlay._fade_timer)
    assert overlay._last_show_timing["first_fade_callback_ms"] == pytest.approx(1000)
    assert overlay._last_show_timing["show_generation"] == 1
    assert overlay._window.alpha == pytest.approx(1 / 144)


def test_expired_notice_callback_cannot_hide_new_recording(mock_pyobjc):
    mod = _import_overlay(mock_pyobjc)
    overlay = mod.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay.hide = MagicMock()
    current = overlay._notice_timer = object()
    overlay.noticeTimerDone_(object())
    overlay.hide.assert_not_called()
    assert overlay._notice_timer is current
    overlay.noticeTimerDone_(current)
    overlay.hide.assert_called_once()


def test_show_resets_stale_scroll_and_document_height(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())

    stale_text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, 220.0),
        "old session text that forced the overlay to grow tall",
    )
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(_make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, 220.0))
    overlay._text_view = stale_text_view
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, 204.0),
        stale_text_view,
        y_offset=128.0,
    )

    overlay.show()

    assert overlay._text_view.string() == ""
    assert overlay._text_view.frame().size.height == overlay_module._OVERLAY_HEIGHT - 16
    assert overlay._text_view.textContainer().container_size == (
        overlay_module._OVERLAY_WIDTH - 24,
        1.0e7,
    )
    assert overlay._scroll_view.contentView().bounds().origin.y == 0.0


def test_show_resets_stale_overlay_chrome_height(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())

    stale_text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, 220.0),
        "old session text that forced the overlay to grow tall",
    )
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(_make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, 220.0))
    overlay._text_view = stale_text_view
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, 204.0),
        stale_text_view,
        y_offset=128.0,
    )

    overlay.show()

    default_height = overlay_module._OVERLAY_HEIGHT
    expected_window_height = default_height + 2 * overlay_module._OUTER_FEATHER
    assert overlay._window.frame().size.height == expected_window_height
    assert overlay._content_view.frame().size.height == default_height


def test_show_positions_preview_much_closer_to_screen_bottom(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "stale",
    )
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )

    overlay.show()

    # Window y = _OVERLAY_BOTTOM_MARGIN - _OUTER_FEATHER
    assert overlay._window.frame().origin.y == pytest.approx(
        overlay_module._OVERLAY_BOTTOM_MARGIN - overlay_module._OUTER_FEATHER
    )


def test_show_clears_stale_tray_background_and_fill_override(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "stale",
    )
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )
    overlay._fill_layer = _FakeLayer(
        _make_rect(
            0.0,
            0.0,
            overlay_module._OVERLAY_WIDTH + 2 * overlay_module._OUTER_FEATHER,
            overlay_module._OVERLAY_HEIGHT + 2 * overlay_module._OUTER_FEATHER,
        )
    )
    overlay._content_view.layer().setBackgroundColor_("stale-square")
    overlay._fill_override_rgb = (0.1, 0.1, 0.12)

    overlay.show()

    assert overlay._content_view.layer().backgroundColor() is None
    assert overlay._fill_override_rgb is None
    assert overlay._fill_layer.opacity() == pytest.approx(overlay_module._BG_ALPHA_MIN)


def test_show_tray_keeps_content_background_clear(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "",
    )
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )
    overlay._fill_layer = _FakeLayer(
        _make_rect(
            0.0,
            0.0,
            overlay_module._OVERLAY_WIDTH + 2 * overlay_module._OUTER_FEATHER,
            overlay_module._OVERLAY_HEIGHT + 2 * overlay_module._OUTER_FEATHER,
        )
    )

    overlay.show_tray("saved text", owner="user")

    assert overlay._content_view.layer().backgroundColor() is None
    assert overlay._fill_override_rgb == pytest.approx((0.1, 0.1, 0.12))
    assert overlay._fill_layer.opacity() == pytest.approx(overlay_module._RECOVERY_BG_ALPHA)


def test_show_stack_body_shell_marks_source_pressure_without_entering_tray_mode(
    mock_pyobjc, monkeypatch
):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "live words stay on the bar",
    )
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )
    overlay._fill_layer = _FakeLayer(
        _make_rect(
            0.0,
            0.0,
            overlay_module._OVERLAY_WIDTH + 2 * overlay_module._OUTER_FEATHER,
            overlay_module._OVERLAY_HEIGHT + 2 * overlay_module._OUTER_FEATHER,
        )
    )
    overlay._tray_mode = False

    overlay.show_stack_body_shell(owner="source")

    assert overlay._tray_mode is False
    assert overlay._text_view.string() == "live words stay on the bar"
    assert overlay._content_view.layer().backgroundColor() is None
    assert overlay._fill_override_rgb == pytest.approx((0.10, 0.13, 0.19))
    assert overlay._fill_layer.opacity() == pytest.approx(overlay_module._RECOVERY_BG_ALPHA)
    assert overlay._window.alpha == pytest.approx(1.0)
    assert overlay._window.ordered_front is True


def test_source_stack_body_shell_keeps_live_preview_text_light(
    mock_pyobjc, monkeypatch
):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "live words stay legible on the source shell",
    )
    overlay._typewriter_displayed = "live words stay legible on the source shell"
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )
    overlay._fill_layer = _FakeLayer(
        _make_rect(
            0.0,
            0.0,
            overlay_module._OVERLAY_WIDTH + 2 * overlay_module._OUTER_FEATHER,
            overlay_module._OVERLAY_HEIGHT + 2 * overlay_module._OUTER_FEATHER,
        )
    )
    overlay._brightness = 0.0
    overlay._brightness_target = 0.0
    overlay._text_lum = 0.0
    overlay._fill_image_brightness = 0.0

    overlay.show_stack_body_shell(owner="source")

    assert overlay._text_lum == pytest.approx(1.0)
    assert overlay._text_view.string() == "live words stay legible on the source shell"
    for _ in range(20):
        overlay.update_text_amplitude(10.0)
    assert overlay._text_lum > 0.9


def test_source_stack_body_shell_repaints_text_without_rewriting_preview_content(
    mock_pyobjc, monkeypatch
):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "live preview text",
    )
    overlay._typewriter_displayed = "live preview text"
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )
    overlay._fill_layer = _FakeLayer(
        _make_rect(
            0.0,
            0.0,
            overlay_module._OVERLAY_WIDTH + 2 * overlay_module._OUTER_FEATHER,
            overlay_module._OVERLAY_HEIGHT + 2 * overlay_module._OUTER_FEATHER,
        )
    )

    overlay.show_stack_body_shell(owner="source")

    assert overlay._text_view.string() == "live preview text"
    assert overlay._text_view.set_string_calls == []
    assert overlay._text_view.text_color is not None


def test_operator_ping_token_body_frame_tracks_live_preview_body(
    mock_pyobjc, monkeypatch
):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._content_view = _FakeView(_make_rect(220.0, 220.0, 640.0, 112.0))

    assert overlay.operator_ping_token_body_frame() == pytest.approx(
        (220.0, 220.0, 640.0, 112.0)
    )


def test_update_layout_caps_preview_growth_below_assistant_overlay(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    monkeypatch.setattr(overlay_module, "NSMakeRect", _make_rect)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())
    overlay._window = _FakeWindow()
    overlay._content_view = _FakeView(
        _make_rect(40.0, 40.0, overlay_module._OVERLAY_WIDTH, overlay_module._OVERLAY_HEIGHT)
    )
    overlay._text_view = _FakeTextView(
        _make_rect(0.0, 0.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        "live preview",
    )
    overlay._text_view.set_layout_height(1000.0)
    overlay._scroll_view = _FakeScrollView(
        _make_rect(12.0, 8.0, overlay_module._OVERLAY_WIDTH - 24, overlay_module._OVERLAY_HEIGHT - 16),
        overlay._text_view,
        y_offset=0.0,
    )

    overlay._update_layout()

    f = overlay_module._OUTER_FEATHER
    expected_height = 150.0
    assert overlay._content_view.frame().size.height == pytest.approx(expected_height)
    assert overlay._window.frame().size.height == pytest.approx(expected_height + 2 * f)
