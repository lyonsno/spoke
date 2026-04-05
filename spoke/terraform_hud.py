"""Terraform HUD: topoi heads-up display panel.

Renders a scrollable sidebar showing active topoi from epistaxis
scoped local state. Each entry shows its semeion name, temperature,
and status snippet.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSGraphicsContext,
    NSPanel,
    NSScreen,
    NSScrollView,
    NSTextField,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
import Quartz

# Style mask constants not always available via PyObjC — use numeric values
_NSWindowStyleMaskTitled = 1 << 0
_NSWindowStyleMaskClosable = 1 << 1
_NSWindowStyleMaskResizable = 1 << 3
_NSWindowStyleMaskUtilityWindow = 1 << 4
from Foundation import (
    NSMakeRect,
    NSObject,
    NSTimer,
)

from .terraform import Topos, load_topoi, format_topos_summary

logger = logging.getLogger(__name__)

# Refresh interval in seconds
_REFRESH_INTERVAL = 5.0

# Visual constants
_PANEL_WIDTH = 320
_PANEL_HEIGHT = 600
_ROW_HEIGHT = 60
_PADDING = 8

# Temperature colors (background tint)
_TEMP_COLORS = {
    "hot": (1.0, 0.3, 0.2, 0.15),      # warm red
    "warm": (1.0, 0.7, 0.2, 0.12),      # amber
    "cool": (0.3, 0.6, 1.0, 0.10),      # cool blue
    "cold": (0.5, 0.5, 0.6, 0.08),      # gray
    "katástasis": (0.3, 0.8, 0.4, 0.10), # settled green
}


def _temp_color(temperature: str | None) -> NSColor:
    """Return a subtle background tint for the given temperature."""
    rgba = _TEMP_COLORS.get(temperature or "", (0.5, 0.5, 0.5, 0.08))
    return NSColor.colorWithRed_green_blue_alpha_(*rgba)


class ToposRowView(NSView):
    """A single row in the topoi list."""

    @classmethod
    def createWithTopos_width_(cls, topos: Topos, width: float) -> ToposRowView:
        view = cls.alloc().initWithFrame_(NSMakeRect(0, 0, width, _ROW_HEIGHT))
        view._topos = topos
        view.setWantsLayer_(True)

        layer = view.layer()
        layer.setCornerRadius_(8.0)
        layer.setMasksToBounds_(True)

        # SDF-style glow: inner color fading to edge
        rgba = _TEMP_COLORS.get(topos.temperature or "", (0.5, 0.5, 0.5, 0.08))
        r, g, b, a = rgba
        # Brighter center, fading to near-transparent edge
        center_color = Quartz.CGColorCreateGenericRGB(r, g, b, a * 3.0)
        mid_color = Quartz.CGColorCreateGenericRGB(r, g, b, a * 1.5)
        edge_color = Quartz.CGColorCreateGenericRGB(r * 0.3, g * 0.3, b * 0.3, a * 0.4)

        gradient = Quartz.CAGradientLayer.layer()
        gradient.setFrame_(((0, 0), (width, _ROW_HEIGHT)))
        gradient.setType_("radial")
        gradient.setColors_([center_color, mid_color, edge_color])
        gradient.setLocations_([0.0, 0.5, 1.0])
        gradient.setStartPoint_((0.5, 0.5))
        gradient.setEndPoint_((1.0, 1.0))
        gradient.setCornerRadius_(8.0)
        layer.addSublayer_(gradient)

        # Subtle border glow
        layer.setBorderWidth_(0.5)
        layer.setBorderColor_(
            Quartz.CGColorCreateGenericRGB(r, g, b, a * 2.0)
        )

        # Semeion name (bold, primary)
        name = topos.semeion or topos.id
        name_label = _make_label(
            name,
            NSMakeRect(_PADDING, _ROW_HEIGHT - 24, width - _PADDING * 2, 18),
            size=13.0,
            bold=True,
            color=NSColor.whiteColor(),
        )
        view.addSubview_(name_label)

        # Temperature + tool badge
        badge_parts = []
        if topos.temperature:
            badge_parts.append(topos.temperature)
        if topos.tool:
            badge_parts.append(topos.tool)
        if badge_parts:
            badge_text = " · ".join(badge_parts)
            badge_label = _make_label(
                badge_text,
                NSMakeRect(_PADDING, _ROW_HEIGHT - 38, width - _PADDING * 2, 14),
                size=10.0,
                bold=False,
                color=NSColor.colorWithWhite_alpha_(0.7, 1.0),
            )
            view.addSubview_(badge_label)

        # Status snippet
        if topos.status:
            snippet = topos.status.split(". ")[0]
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            status_label = _make_label(
                snippet,
                NSMakeRect(_PADDING, 4, width - _PADDING * 2, 14),
                size=10.0,
                bold=False,
                color=NSColor.colorWithWhite_alpha_(0.5, 1.0),
            )
            view.addSubview_(status_label)

        return view


def _make_label(
    text: str,
    frame,
    size: float = 12.0,
    bold: bool = False,
    color: NSColor | None = None,
) -> NSTextField:
    """Create a non-editable text label."""
    label = NSTextField.alloc().initWithFrame_(frame)
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    font = (
        NSFont.boldSystemFontOfSize_(size)
        if bold
        else NSFont.systemFontOfSize_(size)
    )
    label.setFont_(font)
    if color:
        label.setTextColor_(color)
    label.setLineBreakMode_(5)  # NSLineBreakByTruncatingTail
    return label


class TerraformHUD(NSObject):
    """Manages the Terraform topoi HUD panel."""

    def init(self):
        self = objc.super(TerraformHUD, self).init()
        if self is None:
            return None
        self._panel: NSPanel | None = None
        self._content_view: NSView | None = None
        self._scroll_view: NSScrollView | None = None
        self._topoi: list[Topos] = []
        self._timer: NSTimer | None = None
        self._visible = False
        return self

    def setup(self) -> None:
        """Create the HUD panel and populate it."""
        screen = NSScreen.mainScreen()
        screen_frame = screen.visibleFrame()

        # Position on the left side of the screen
        x = screen_frame.origin.x + 20
        y = screen_frame.origin.y + screen_frame.size.height - _PANEL_HEIGHT - 40
        frame = NSMakeRect(x, y, _PANEL_WIDTH, _PANEL_HEIGHT)

        style = (
            _NSWindowStyleMaskResizable
            | _NSWindowStyleMaskUtilityWindow
            | NSWindowStyleMaskNonactivatingPanel
        )

        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False
        )
        self._panel.setTitle_("Terror Form")
        self._panel.setLevel_(3)  # NSFloatingWindowLevel
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self._panel.setFloatingPanel_(True)
        self._panel.setBecomesKeyOnlyIfNeeded_(True)

        # Scroll view for the topoi list — no visible scrollbar,
        # content fades to transparent at top and bottom edges.
        content_frame = self._panel.contentView().bounds()
        self._scroll_view = NSScrollView.alloc().initWithFrame_(content_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setAutoresizingMask_(18)  # width + height flex

        # Document view (the actual content that scrolls)
        self._content_view = NSView.alloc().initWithFrame_(content_frame)
        self._content_view.setWantsLayer_(True)
        self._scroll_view.setDocumentView_(self._content_view)

        # Fade mask: content fades to transparent at top and bottom
        self._scroll_view.setWantsLayer_(True)
        fade_height = content_frame.size.height
        fade_width = content_frame.size.width
        _FADE_FRACTION = 0.08  # 8% of panel height fades at each edge

        mask_layer = Quartz.CAGradientLayer.layer()
        mask_layer.setFrame_(((0, 0), (fade_width, fade_height)))
        clear = Quartz.CGColorCreateGenericRGB(0, 0, 0, 0)
        opaque = Quartz.CGColorCreateGenericRGB(0, 0, 0, 1)
        mask_layer.setColors_([clear, opaque, opaque, clear])
        mask_layer.setLocations_([0.0, _FADE_FRACTION, 1.0 - _FADE_FRACTION, 1.0])
        mask_layer.setStartPoint_((0.5, 0.0))
        mask_layer.setEndPoint_((0.5, 1.0))
        self._scroll_view.layer().setMask_(mask_layer)

        self._panel.contentView().addSubview_(self._scroll_view)

        # Initial load
        self._refresh()

    def show(self) -> None:
        """Show the HUD panel and start auto-refresh."""
        if self._panel is None:
            self.setup()
        self._panel.orderFront_(None)
        self._visible = True
        self._start_timer()

    def hide(self) -> None:
        """Hide the HUD panel and stop auto-refresh."""
        if self._panel is not None:
            self._panel.orderOut_(None)
        self._visible = False
        self._stop_timer()

    def toggle(self) -> None:
        """Toggle HUD visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._stop_timer()
        if self._panel is not None:
            self._panel.close()
            self._panel = None

    # -- private --

    def _start_timer(self) -> None:
        self._stop_timer()
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            _REFRESH_INTERVAL, self, "_timerFired:", None, True
        )

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def _timerFired_(self, timer) -> None:
        self._refresh()

    def _refresh(self) -> None:
        """Reload topoi from epistaxis and rebuild the view."""
        self._topoi = load_topoi()
        self._rebuild_content()

    def _rebuild_content(self) -> None:
        """Rebuild the scrollable content from current topoi."""
        if self._content_view is None:
            return

        # Remove old subviews
        for subview in list(self._content_view.subviews()):
            subview.removeFromSuperview()

        width = self._scroll_view.bounds().size.width - 8  # edge padding
        total_height = max(
            len(self._topoi) * (_ROW_HEIGHT + 4) + _PADDING,
            self._scroll_view.bounds().size.height,
        )

        self._content_view.setFrameSize_((self._scroll_view.bounds().size.width, total_height))

        for i, topos in enumerate(self._topoi):
            y = total_height - (i + 1) * (_ROW_HEIGHT + 4)
            row = ToposRowView.createWithTopos_width_(topos, width)
            row.setFrameOrigin_((4, y))
            self._content_view.addSubview_(row)
