"""Terraform HUD: topoi heads-up display panel.

Renders a scrollable sidebar showing active topoi from epistaxis
scoped local state. Each entry shows its semeion name, temperature,
and status snippet.
"""

from __future__ import annotations

import logging
import math
import subprocess
import time as _time
from pathlib import Path

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSPanel,
    NSScreen,
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

from .terraform import Topos, load_topoi, filter_topoi, sort_topoi, count_attractors, disambiguated_name

logger = logging.getLogger(__name__)

# Refresh interval in seconds
_REFRESH_INTERVAL = 5.0

# Visual constants
_PANEL_WIDTH = 320
_PANEL_HEIGHT = 900
_ROW_HEIGHT = 60
_ROW_CONTAINER_HEIGHT = 64  # tight — bloom bleeds slightly into neighbors
_PADDING = 8
_BLOOM_EXPAND = 4    # how many px the inner bloom ring extends beyond the card
_OUTER_EXPAND = 6    # how many px the outer amber ring extends beyond the card

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
    """A single row in the topoi list — frosted bubble matching overlay language."""

    @classmethod
    def createWithTopos_width_(cls, topos: Topos, width: float) -> ToposRowView:
        # Container is taller than the card to hold bloom overflow
        view = cls.alloc().initWithFrame_(NSMakeRect(0, 0, width, _ROW_CONTAINER_HEIGHT))
        view._topos = topos
        view.setWantsLayer_(True)
        view.layer().setMasksToBounds_(False)

        rgba = _TEMP_COLORS.get(topos.temperature or "", (0.5, 0.5, 0.5, 0.08))
        r, g, b, a = rgba

        # Card is centered vertically in the container
        card_y = (_ROW_CONTAINER_HEIGHT - _ROW_HEIGHT) / 2.0
        card_x = 0.0

        # --- Card: frosted bubble matching overlay language ---
        # The card layer carries both the visible fill AND the glow shadows.
        # No separate invisible glow layers — those were ghosting.
        card_layer = Quartz.CALayer.layer()
        card_layer.setName_("card")
        card_layer.setFrame_(((card_x, card_y), (width, _ROW_HEIGHT)))
        card_layer.setCornerRadius_(10.0)
        # Match overlay bg: desaturated blue-white (0.50, 0.59, 0.84) at 40% sat
        card_layer.setBackgroundColor_(
            Quartz.CGColorCreateGenericRGB(0.38, 0.42, 0.56, 0.82)
        )
        # Temperature-tinted border
        card_layer.setBorderWidth_(1.0)
        card_layer.setBorderColor_(
            Quartz.CGColorCreateGenericRGB(r, g, b, min(a * 4.0, 0.5))
        )
        # Temperature-colored shadow glow — single shadow on the card itself
        # masksToBounds must be False for shadow to render
        card_layer.setMasksToBounds_(False)
        card_layer.setShadowColor_(Quartz.CGColorCreateGenericRGB(r, g, b, 1.0))
        card_layer.setShadowOffset_((0, 0))
        card_layer.setShadowRadius_(6.0)
        card_layer.setShadowOpacity_(min(a * 6.0, 0.5))
        # Explicit shadow path for performance and crisp shape
        shadow_path = Quartz.CGPathCreateWithRoundedRect(
            ((0, 0), (width, _ROW_HEIGHT)), 10.0, 10.0, None
        )
        card_layer.setShadowPath_(shadow_path)
        view.layer().addSublayer_(card_layer)

        # --- Text labels (positioned relative to card_y) ---
        name = disambiguated_name(topos)
        name_label = _make_label(
            name,
            NSMakeRect(_PADDING, card_y + _ROW_HEIGHT - 24, width - _PADDING * 2, 18),
            size=13.0,
            bold=True,
            color=NSColor.whiteColor(),
        )
        view.addSubview_(name_label)

        badge_parts = []
        if topos.temperature:
            badge_parts.append(topos.temperature)
        if topos.tool:
            badge_parts.append(topos.tool)
        if badge_parts:
            badge_text = " · ".join(badge_parts)
            badge_label = _make_label(
                badge_text,
                NSMakeRect(_PADDING, card_y + _ROW_HEIGHT - 38, width - _PADDING * 2, 14),
                size=10.0,
                bold=False,
                color=NSColor.colorWithWhite_alpha_(0.7, 1.0),
            )
            view.addSubview_(badge_label)

        if topos.status:
            snippet = topos.status.split(". ")[0]
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            status_label = _make_label(
                snippet,
                NSMakeRect(_PADDING, card_y + 4, width - _PADDING * 2, 14),
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


class _ManualScrollView(NSView):
    """Plain NSView that scrolls its content by adjusting frame origin.

    Replaces NSScrollView to avoid NSClipView layer-caching ghosts.
    Handles scrollWheel for trackpad scroll, mouseDown/Dragged for
    window repositioning, and applies a gradient fade mask.
    """

    def initWithFrame_(self, frame):
        self = objc.super(_ManualScrollView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._scroll_offset = 0.0  # how far the content is scrolled (positive = scrolled down)
        self._content: NSView | None = None
        self.setWantsLayer_(True)
        self.layer().setMasksToBounds_(True)

        # Fade mask
        h = frame.size.height
        w = frame.size.width
        _FADE_FRACTION = 0.08
        mask = Quartz.CAGradientLayer.layer()
        mask.setFrame_(((0, 0), (w, h)))
        clear = Quartz.CGColorCreateGenericRGB(0, 0, 0, 0)
        opaque = Quartz.CGColorCreateGenericRGB(0, 0, 0, 1)
        mask.setColors_([clear, opaque, opaque, clear])
        mask.setLocations_([0.0, _FADE_FRACTION, 1.0 - _FADE_FRACTION, 1.0])
        mask.setStartPoint_((0.5, 0.0))
        mask.setEndPoint_((0.5, 1.0))
        self.layer().setMask_(mask)
        return self

    def setContent_(self, view):
        """Replace the scrollable content view."""
        if self._content is not None:
            self._content.removeFromSuperview()
        self._content = view
        if view is not None:
            self.addSubview_(view)
            self._apply_scroll()

    def scrollWheel_(self, event):
        if self._content is None:
            return
        # deltaY is positive when scrolling up (content moves down)
        self._scroll_offset -= event.deltaY() * 3.0
        self._clamp_scroll()
        self._apply_scroll()

    def mouseDown_(self, event):
        self._drag_origin = event.locationInWindow()

    def mouseDragged_(self, event):
        origin = getattr(self, '_drag_origin', None)
        if origin is None:
            return
        win = self.window()
        if win is None:
            return
        current = event.locationInWindow()
        dx = current.x - origin.x
        dy = current.y - origin.y
        frame = win.frame()
        win.setFrameOrigin_((frame.origin.x + dx, frame.origin.y + dy))

    def _clamp_scroll(self):
        if self._content is None:
            self._scroll_offset = 0.0
            return
        content_h = self._content.frame().size.height
        visible_h = self.bounds().size.height
        max_scroll = max(0.0, content_h - visible_h)
        self._scroll_offset = max(0.0, min(self._scroll_offset, max_scroll))

    def _apply_scroll(self):
        if self._content is None:
            return
        content_h = self._content.frame().size.height
        visible_h = self.bounds().size.height
        # In flipped-ish terms: offset 0 = top of content visible
        # NSView y=0 is bottom, so we position the content such that
        # the top of the content aligns with the top of the visible area
        # when scroll_offset is 0.
        y = -(content_h - visible_h) + self._scroll_offset
        self._content.setFrameOrigin_((0, y))


class TerraformHUD(NSObject):
    """Manages the Terraform topoi HUD panel."""

    def init(self):
        self = objc.super(TerraformHUD, self).init()
        if self is None:
            return None
        self._panel: NSPanel | None = None
        self._content_view: NSView | None = None
        self._scroll_view: _ManualScrollView | None = None
        self._topoi: list[Topos] = []
        self._timer: NSTimer | None = None
        self._anim_timer: NSTimer | None = None
        self._visible = False
        self._sort_key = "temperature"
        self._hide_katastasis = True
        self._filter_machine: str | None = None
        self._filter_tool: str | None = None
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
        self._panel.setLevel_(1000)  # well above glow/dimmer layer (level 25)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self._panel.setFloatingPanel_(True)
        self._panel.setBecomesKeyOnlyIfNeeded_(True)

        # Manual scroll view — no NSScrollView/NSClipView layer caching
        content_frame = self._panel.contentView().bounds()

        # Stats label at the top of the panel
        _STATS_HEIGHT = 20
        stats_frame = NSMakeRect(8, content_frame.size.height - _STATS_HEIGHT,
                                  content_frame.size.width - 16, _STATS_HEIGHT)
        self._stats_label = _make_label(
            "", stats_frame, size=10.0, bold=False,
            color=NSColor.colorWithWhite_alpha_(0.45, 1.0),
        )
        self._panel.contentView().addSubview_(self._stats_label)

        # Scroll area below stats
        scroll_frame = NSMakeRect(0, 0, content_frame.size.width,
                                   content_frame.size.height - _STATS_HEIGHT)
        self._scroll_view = _ManualScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setAutoresizingMask_(18)  # width + height flex
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
        # ~30fps animation timer for bloom breathing
        self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 30.0, self, "_animTick:", None, True
        )

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        if self._anim_timer is not None:
            self._anim_timer.invalidate()
            self._anim_timer = None

    def _timerFired_(self, timer) -> None:
        self._refresh()

    def _animTick_(self, timer) -> None:
        """Modulate card shadow glow with three phase-shifted sine waves.

        Each card gets a slightly different phase based on its position
        so the breathing drifts across the column.
        """
        if self._content_view is None:
            return
        t = _time.monotonic()
        # Three waves combined into one modulator per card
        wave1 = math.sin(t * 2.0 * math.pi / 9.0)         # 9s period
        wave2 = math.sin(t * 2.0 * math.pi / 7.0 + 2.1)   # 7s period
        wave3 = math.sin(t * 2.0 * math.pi / 11.0 + 4.0)  # 11s period
        # Combined: subtle range 0.6–1.0
        glow_mod = 0.8 + 0.07 * wave1 + 0.07 * wave2 + 0.06 * wave3

        for subview in self._content_view.subviews():
            layer = subview.layer()
            if layer is None:
                continue
            for sublayer in layer.sublayers() or []:
                if sublayer.name() == "card":
                    # Modulate shadow radius for breathing effect
                    # (opacity stored at creation time, don't decay it)
                    sublayer.setShadowRadius_(6.0 * glow_mod)

    def _refresh(self) -> None:
        """Reload topoi from epistaxis, filter, sort, and rebuild the view."""
        raw = load_topoi()
        self._update_stats(raw)
        filtered = filter_topoi(
            raw,
            hide_katastasis=self._hide_katastasis,
            machine=self._filter_machine,
            tool=self._filter_tool,
        )
        self._topoi = sort_topoi(filtered, key=self._sort_key)
        self._rebuild_content()

    def _update_stats(self, topoi: list[Topos]) -> None:
        """Update the stats label with topos + attractor counts."""
        if not hasattr(self, "_stats_label") or self._stats_label is None:
            return
        from collections import Counter
        counts = Counter(t.temperature or "unknown" for t in topoi)

        # Topos counts
        parts = []
        for temp in ("hot", "warm", "cool", "cold", "katástasis"):
            n = counts.get(temp, 0)
            if n > 0:
                parts.append(f"{n} {temp}")
        unknown = counts.get("unknown", 0)
        if unknown:
            parts.append(f"{unknown} ?")
        topos_line = " · ".join(parts) if parts else "no topoi"

        # Attractor counts
        att = count_attractors()
        att_parts = []
        if att.active:
            att_parts.append(f"{att.active} active")
        if att.soak:
            att_parts.append(f"{att.soak} soak")
        if att.smoke:
            att_parts.append(f"{att.smoke} smoke")
        if att.katastasis:
            att_parts.append(f"{att.katastasis} settled")
        if att.unclassified:
            att_parts.append(f"{att.unclassified} unclassified")
        att_line = " · ".join(att_parts) if att_parts else ""
        att_summary = f"  ⫶  {att.total} attractors ({att_line})" if att.total else ""

        self._stats_label.setStringValue_(f"{topos_line}{att_summary}")
        # Re-assert window ordering in case glow used orderFrontRegardless
        if self._panel is not None and self._visible:
            self._panel.orderFront_(None)

    def _rebuild_content(self) -> None:
        """Rebuild the scrollable content from current topoi."""
        if self._scroll_view is None:
            return

        scroll_bounds = self._scroll_view.bounds()
        width = scroll_bounds.size.width - 8  # edge padding
        row_stride = _ROW_CONTAINER_HEIGHT + 2
        total_height = max(
            len(self._topoi) * row_stride + _PADDING,
            scroll_bounds.size.height,
        )

        # Build a fresh content view
        new_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, scroll_bounds.size.width, total_height)
        )

        for i, topos in enumerate(self._topoi):
            y = total_height - (i + 1) * row_stride
            row = ToposRowView.createWithTopos_width_(topos, width)
            row.setFrameOrigin_((4, y))
            new_content.addSubview_(row)

        # Swap content — old view is fully removed and deallocated
        self._scroll_view.setContent_(new_content)
        self._content_view = new_content
