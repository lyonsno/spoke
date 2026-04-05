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

        # --- Outer glow: invisible shape that casts a temperature-colored shadow ---
        outer_glow = Quartz.CALayer.layer()
        outer_glow.setName_("outer_bloom")
        outer_glow.setFrame_(((card_x, card_y), (width, _ROW_HEIGHT)))
        outer_glow.setCornerRadius_(10.0)
        outer_glow.setBackgroundColor_(
            Quartz.CGColorCreateGenericRGB(r, g, b, 0.01)  # near-invisible
        )
        outer_glow.setShadowColor_(Quartz.CGColorCreateGenericRGB(r, g, b, 1.0))
        outer_glow.setShadowOffset_((0, 0))
        outer_glow.setShadowRadius_(8.0)
        outer_glow.setShadowOpacity_(min(a * 5.0, 0.6))
        view.layer().addSublayer_(outer_glow)

        # --- Inner glow: tighter, desaturated toward white ---
        dr = r * 0.5 + 0.5  # pull toward white
        dg = g * 0.5 + 0.5
        db = b * 0.5 + 0.5
        inner_glow = Quartz.CALayer.layer()
        inner_glow.setName_("inner_bloom")
        inner_glow.setFrame_(((card_x, card_y), (width, _ROW_HEIGHT)))
        inner_glow.setCornerRadius_(10.0)
        inner_glow.setBackgroundColor_(
            Quartz.CGColorCreateGenericRGB(dr, dg, db, 0.01)  # near-invisible
        )
        inner_glow.setShadowColor_(Quartz.CGColorCreateGenericRGB(dr, dg, db, 1.0))
        inner_glow.setShadowOffset_((0, 0))
        inner_glow.setShadowRadius_(4.0)
        inner_glow.setShadowOpacity_(min(a * 8.0, 0.5))
        view.layer().addSublayer_(inner_glow)

        # --- Card: frosted bubble matching overlay language ---
        card_layer = Quartz.CALayer.layer()
        card_layer.setName_("card")
        card_layer.setFrame_(((card_x, card_y), (width, _ROW_HEIGHT)))
        card_layer.setCornerRadius_(10.0)
        card_layer.setMasksToBounds_(True)
        # Match overlay bg: desaturated blue-white (0.50, 0.59, 0.84) at 40% sat
        card_layer.setBackgroundColor_(
            Quartz.CGColorCreateGenericRGB(0.38, 0.42, 0.56, 0.82)
        )
        # Temperature-tinted border
        card_layer.setBorderWidth_(1.0)
        card_layer.setBorderColor_(
            Quartz.CGColorCreateGenericRGB(r, g, b, min(a * 4.0, 0.5))
        )
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


class _DraggableScrollView(NSScrollView):
    """NSScrollView that allows click-drag to move the parent window.

    Two-finger trackpad scrolling still works normally via scrollWheel:.
    Single-click drag repositions the panel.
    """

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

        # Scroll view for the topoi list — no visible scrollbar,
        # content fades to transparent at top and bottom edges.
        content_frame = self._panel.contentView().bounds()
        self._scroll_view = _DraggableScrollView.alloc().initWithFrame_(content_frame)
        self._scroll_view.setHasVerticalScroller_(False)
        self._scroll_view.setHasHorizontalScroller_(False)
        self._scroll_view.setDrawsBackground_(False)
        self._scroll_view.setAutoresizingMask_(18)  # width + height flex

        # Document view (the actual content that scrolls) — replaced
        # each refresh cycle to avoid ghost layer accumulation.
        self._content_view = NSView.alloc().initWithFrame_(content_frame)
        self._scroll_view.setDocumentView_(self._content_view)
        # Disable scroll-copy optimization — it caches bitmaps of the
        # document view which become ghosts when the content is replaced.
        self._scroll_view.contentView().setCopiesOnScroll_(False)

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

        # Stats label at the top of the panel (above the scroll view)
        _STATS_HEIGHT = 20
        stats_frame = NSMakeRect(8, content_frame.size.height - _STATS_HEIGHT,
                                  content_frame.size.width - 16, _STATS_HEIGHT)
        self._stats_label = _make_label(
            "", stats_frame, size=10.0, bold=False,
            color=NSColor.colorWithWhite_alpha_(0.45, 1.0),
        )
        self._panel.contentView().addSubview_(self._stats_label)

        # Shrink scroll view to make room for stats
        scroll_frame = NSMakeRect(0, 0, content_frame.size.width,
                                   content_frame.size.height - _STATS_HEIGHT)
        self._scroll_view.setFrame_(scroll_frame)

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
        """Modulate bloom and card opacity with three phase-shifted sine waves."""
        if self._content_view is None:
            return
        t = _time.monotonic()
        # Outer amber ring: 9s period, subtle (0.6–1.0), phase 0
        outer_opacity = 0.8 + 0.2 * math.sin(t * 2.0 * math.pi / 9.0)
        # Inner cool ring: 7s period, subtle (0.65–1.0), phase offset 2.1
        inner_opacity = 0.825 + 0.175 * math.sin(t * 2.0 * math.pi / 7.0 + 2.1)
        # Card (temperature color): 11s period, deeper breathing (0.5–1.0), phase offset 4.0
        # Same trough depth as the rings but peak reaches full opacity
        card_opacity = 0.75 + 0.25 * math.sin(t * 2.0 * math.pi / 11.0 + 4.0)

        for subview in self._content_view.subviews():
            layer = subview.layer()
            if layer is None:
                continue
            for sublayer in layer.sublayers() or []:
                name = sublayer.name()
                if name == "outer_bloom":
                    sublayer.setOpacity_(outer_opacity)
                elif name == "inner_bloom":
                    sublayer.setOpacity_(inner_opacity)
                elif name == "card":
                    sublayer.setOpacity_(card_opacity)

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
        """Rebuild the scrollable content from current topoi.

        Replaces the entire document view each cycle rather than
        mutating in place — prevents ghost layers from accumulating
        when AppKit's layer-backed view teardown leaves orphans.
        """
        if self._scroll_view is None:
            return

        # Preserve scroll position across rebuild
        clip_view = self._scroll_view.contentView()
        old_origin = clip_view.bounds().origin if clip_view else None

        scroll_bounds = self._scroll_view.bounds()
        width = scroll_bounds.size.width - 8  # edge padding
        row_stride = _ROW_CONTAINER_HEIGHT + 2
        total_height = max(
            len(self._topoi) * row_stride + _PADDING,
            scroll_bounds.size.height,
        )

        # Build a fresh document view — no mutation of the old one
        new_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, scroll_bounds.size.width, total_height)
        )

        for i, topos in enumerate(self._topoi):
            y = total_height - (i + 1) * row_stride
            row = ToposRowView.createWithTopos_width_(topos, width)
            row.setFrameOrigin_((4, y))
            new_content.addSubview_(row)

        # Swap document view atomically
        self._scroll_view.setDocumentView_(new_content)
        self._content_view = new_content

        # Force the clip view to flush any cached bitmap from the old document
        clip_view = self._scroll_view.contentView()
        if clip_view is not None:
            clip_view.setNeedsDisplay_(True)
            if clip_view.layer():
                clip_view.layer().setNeedsDisplay()

        # Restore scroll position
        if old_origin is not None and clip_view is not None:
            clip_view.setBoundsOrigin_(old_origin)
