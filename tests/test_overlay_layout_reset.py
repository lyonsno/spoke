"""Regression tests for overlay layout reset between recording sessions."""

import importlib
import sys
from types import SimpleNamespace

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

    def setFrame_(self, frame):
        self._frame = frame

    def frame(self):
        return self._frame


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

    def setString_(self, text):
        self._text = text

    def string(self):
        return self._text

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


class _FakeScrollView:
    def __init__(self, frame, text_view, y_offset):
        self._frame = frame
        self._document_view = text_view
        self._clip_view = _FakeClipView(y_offset)
        self.reflected_clip_view = None

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
