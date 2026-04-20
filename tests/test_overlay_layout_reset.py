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

    def setString_(self, text):
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
    return importlib.import_module("spoke.overlay")


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


def test_init_uses_shared_backdrop_renderer_factory(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    sentinel = object()
    factory = MagicMock(return_value=sentinel)
    monkeypatch.setattr(overlay_module, "make_backdrop_renderer", factory)

    overlay = overlay_module.TranscriptionOverlay.alloc().initWithScreen_(_FakeScreen())

    assert overlay._backdrop_renderer is sentinel
    factory.assert_called_once()


def test_install_backdrop_frame_callback_pushes_live_frames_into_layer(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_layer = MagicMock()

    overlay._install_backdrop_frame_callback()

    callback = overlay._backdrop_renderer.set_frame_callback.call_args[0][0]
    callback("live-frame")

    overlay._backdrop_layer.setContents_.assert_called_once_with("live-frame")


def test_install_backdrop_frame_callback_keeps_image_fallback_for_sample_buffer_layer(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_layer_is_sample_buffer_display = True

    overlay._install_backdrop_frame_callback()

    callback = overlay._backdrop_renderer.set_frame_callback.call_args[0][0]

    assert callback is not None


def test_install_backdrop_sample_buffer_callback_enqueues_live_samples(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_layer = MagicMock()

    overlay._install_backdrop_sample_buffer_callback()

    callback = overlay._backdrop_renderer.set_sample_buffer_callback.call_args[0][0]
    callback("live-sample")

    overlay._backdrop_layer.enqueueSampleBuffer_.assert_called_once_with("live-sample")


def test_preview_backdrop_mask_falloff_width_uses_configured_multiplier(mock_pyobjc, monkeypatch):
    monkeypatch.setenv("SPOKE_PREVIEW_BACKDROP_MASK_WIDTH_MULTIPLIER", "9.0")
    overlay_module = _import_overlay(mock_pyobjc)

    assert overlay_module._preview_backdrop_mask_falloff_width(2.0) == pytest.approx(18.0)


def test_preview_backdrop_rms_style_boosts_blur_when_overlay_comes_forward(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)

    idle = overlay_module._preview_backdrop_rms_style(5.4, 0.0)
    active = overlay_module._preview_backdrop_rms_style(5.4, 1.0)

    assert active > idle


def test_choose_backdrop_layer_uses_display_layer_when_renderer_supports_sample_buffers(mock_pyobjc, monkeypatch):
    overlay_module = _import_overlay(mock_pyobjc)
    sentinel_layer_class = MagicMock()
    monkeypatch.setattr(overlay_module, "_backdrop_display_layer_class", lambda: sentinel_layer_class)

    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.supports_sample_buffer_presentation.return_value = True
    overlay._backdrop_blur_radius_points = 5.4

    assert overlay._choose_backdrop_layer_class() is sentinel_layer_class


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


def test_show_clears_stale_preview_backdrop_before_reuse(mock_pyobjc, monkeypatch):
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
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.capture_blurred_image.return_value = None
    overlay._ridge_scale = 2.0

    timer = object()

    def _timer_factory(*args):
        selector = args[2]
        return timer if selector == "backdropRefreshTick:" else object()

    overlay_module.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.side_effect = _timer_factory

    overlay.show()

    overlay._backdrop_layer.setContents_.assert_called_with(None)
    overlay._backdrop_layer.setMask_.assert_called_with(None)
    assert overlay._backdrop_timer is timer


def test_show_flushes_sample_buffer_preview_backdrop_before_reuse(mock_pyobjc, monkeypatch):
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
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_layer.flushAndRemoveImage = MagicMock()
    overlay._backdrop_layer_is_sample_buffer_display = True
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.capture_blurred_image.return_value = None
    overlay._ridge_scale = 2.0

    overlay.show()

    overlay._backdrop_layer.flushAndRemoveImage.assert_called_once_with()


def test_refresh_preview_backdrop_snapshot_updates_contents_frame_and_mask(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._screen = _FakeScreen(width=1920.0, height=1080.0)
    overlay._window = MagicMock()
    overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
    overlay._window.windowNumber.return_value = 17
    overlay._content_view = MagicMock()
    overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.capture_blurred_image.return_value = "preview-blur"
    overlay._update_backdrop_mask = MagicMock()
    overlay._ridge_scale = 2.0
    overlay._backdrop_capture_overscan_points = 40.0
    overlay._backdrop_blur_radius_points = 9.0

    overlay._refresh_backdrop_snapshot()

    call = overlay._backdrop_renderer.capture_blurred_image.call_args.kwargs
    assert call["blur_radius_points"] == pytest.approx(9.0)
    assert call["capture_rect"].origin.y == pytest.approx(660.0)
    assert call["capture_rect"].size.width == pytest.approx(680.0)
    assert call["capture_rect"].size.height == pytest.approx(160.0)
    overlay._backdrop_layer.setFrame_.assert_called_with(((180.0, 180.0), (680.0, 160.0)))
    overlay._backdrop_layer.setContents_.assert_called_with("preview-blur")
    overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)


def test_refresh_preview_backdrop_snapshot_skips_image_seed_for_blurred_sample_buffer_path(
    mock_pyobjc,
):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._screen = _FakeScreen(width=1920.0, height=1080.0)
    overlay._window = MagicMock()
    overlay._window.frame.return_value = _make_rect(300.0, 80.0, 1040.0, 520.0)
    overlay._window.windowNumber.return_value = 17
    overlay._content_view = MagicMock()
    overlay._content_view.frame.return_value = _make_rect(220.0, 220.0, 600.0, 80.0)
    overlay._backdrop_layer = MagicMock()
    overlay._backdrop_renderer = MagicMock()
    overlay._backdrop_renderer.capture_blurred_image.return_value = None
    overlay._backdrop_renderer.uses_direct_sample_buffers.return_value = True
    overlay._update_backdrop_mask = MagicMock()
    overlay._ridge_scale = 2.0
    overlay._backdrop_capture_overscan_points = 40.0
    overlay._backdrop_blur_radius_points = 5.4

    overlay._refresh_backdrop_snapshot()

    overlay._backdrop_renderer.uses_direct_sample_buffers.assert_called_once_with(5.4)
    overlay._backdrop_layer.setFrame_.assert_called_with(((180.0, 180.0), (680.0, 160.0)))
    overlay._backdrop_layer.setContents_.assert_not_called()
    overlay._update_backdrop_mask.assert_called_once_with(680.0, 160.0)


def test_order_out_stops_live_preview_backdrop_stream(mock_pyobjc):
    overlay_module = _import_overlay(mock_pyobjc)
    overlay = overlay_module.TranscriptionOverlay.__new__(overlay_module.TranscriptionOverlay)
    overlay._window = MagicMock()
    overlay._backdrop_renderer = MagicMock()
    overlay._cancel_tray_capture_flash = MagicMock()
    overlay._cancel_backdrop_refresh = MagicMock()
    overlay._cancel_fade = MagicMock()
    overlay._cancel_typewriter = MagicMock()
    overlay._reset_backdrop_layer = MagicMock()

    overlay.order_out()

    overlay._backdrop_renderer.stop_live_stream.assert_called_once_with()


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
    expected_height = 220.0
    assert overlay._content_view.frame().size.height == pytest.approx(expected_height)
    assert overlay._window.frame().size.height == pytest.approx(expected_height + 2 * f)
