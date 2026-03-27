"""Regression tests for overlay layout reset between recording sessions."""

import importlib
import sys
from types import SimpleNamespace


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
