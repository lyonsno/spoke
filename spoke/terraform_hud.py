"""Terraform HUD: topoi heads-up display panel.

Renders a scrollable sidebar showing active topoi from epistaxis
scoped local state. Each entry shows its semeion name, temperature,
and status snippet.
"""

from __future__ import annotations

import json
import logging
import subprocess
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
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
try:
    from .terraform_metal import metal_available, TerraformCardRenderer, CardInfo
    _METAL_OK = metal_available()
except Exception:
    _METAL_OK = False
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

# Persistent state
_PREFS_PATH = Path.home() / "Library" / "Application Support" / "Spoke" / "terraform_hud.json"

# Refresh interval in seconds
_REFRESH_INTERVAL = 5.0  # data check + Metal redraw only; view rebuild only on change

# Visual constants
_GLOW_MARGIN = 120  # extra margin on each side for outer glow spill
_PANEL_WIDTH = 320 + _GLOW_MARGIN * 2
_PANEL_HEIGHT = 900
_ROW_HEIGHT = 60
_ROW_GAP = 16  # breathing room between cards
_PADDING = 8

# Temperature colors (background tint)
_TEMP_COLORS = {
    "hot": (1.0, 0.3, 0.2, 0.15),      # warm red
    "warm": (1.0, 0.7, 0.2, 0.12),      # amber
    "cool": (0.3, 0.6, 1.0, 0.10),      # cool blue
    "cold": (0.5, 0.5, 0.6, 0.08),      # gray
    "katástasis": (0.3, 0.8, 0.4, 0.10), # settled green
}

def _format_observed(raw: str) -> str:
    """Format an Observed timestamp as 12-hour EST with AM/PM."""
    from datetime import datetime, timezone, timedelta
    est = timezone(timedelta(hours=-4))  # EDT
    try:
        if "T" in raw:
            # ISO timestamp — parse and convert to EST 12-hour
            # Handle both -0400 and -04:00 offset formats
            cleaned = raw.strip()
            dt = datetime.fromisoformat(cleaned)
            dt_est = dt.astimezone(est)
            return dt_est.strftime("%-I:%M%p").lower()
        else:
            # Date only
            return raw.strip()
    except Exception:
        return raw.strip()


# Adaptive fill colors — same visual language as overlay.py
_SDF_BASE_DARK = (0.50 * 0.6 + 0.5 * 0.4,
                  0.59 * 0.6 + 0.5 * 0.4,
                  0.84 * 0.6 + 0.5 * 0.4)  # desaturated cornflower on dark bg
_SDF_BASE_LIGHT = (0.10, 0.10, 0.12)       # near-black on light bg
_SDF_CARD_ALPHA = 0.55  # per-card opacity — translucent, not flat


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


def _temp_fill_color(temperature: str | None, brightness: float = 0.0) -> tuple[float, float, float]:
    """Blend adaptive base color with temperature tint for SDF card fill."""
    rgba = _TEMP_COLORS.get(temperature or "", (0.5, 0.5, 0.5, 0.08))
    tr, tg, tb, ta = rgba
    # Interpolate base color by brightness
    br = _lerp(_SDF_BASE_DARK[0], _SDF_BASE_LIGHT[0], brightness)
    bg = _lerp(_SDF_BASE_DARK[1], _SDF_BASE_LIGHT[1], brightness)
    bb = _lerp(_SDF_BASE_DARK[2], _SDF_BASE_LIGHT[2], brightness)
    # Mix: base * (1 - ta) + temp_color * ta  (ta is small, 0.08-0.15)
    return (br * (1.0 - ta) + tr * ta,
            bg * (1.0 - ta) + tg * ta,
            bb * (1.0 - ta) + tb * ta)


def _temp_color(temperature: str | None) -> NSColor:
    """Return a subtle background tint for the given temperature."""
    rgba = _TEMP_COLORS.get(temperature or "", (0.5, 0.5, 0.5, 0.08))
    return NSColor.colorWithRed_green_blue_alpha_(*rgba)


class ToposRowView(NSView):
    """A single row in the topoi list — frosted bubble matching overlay language."""

    @classmethod
    def createWithTopos_width_(cls, topos: Topos, width: float) -> ToposRowView:
        view = cls.alloc().initWithFrame_(NSMakeRect(0, 0, width, _ROW_HEIGHT))
        view._topos = topos
        view.setWantsLayer_(True)
        view.layer().setBackgroundColor_(Quartz.CGColorCreateGenericRGB(0, 0, 0, 0))

        rgba = _TEMP_COLORS.get(topos.temperature or "", (0.5, 0.5, 0.5, 0.08))
        r, g, b, _a = rgba
        # Temperature-tinted text — saturated, reads as color against milky fill
        _sub_color = NSColor.colorWithRed_green_blue_alpha_(
            r * 0.85, g * 0.85, b * 0.85, 0.95
        )
        # Name: white, punched through as cutout when Metal is active
        _text_color = NSColor.colorWithWhite_alpha_(1.0, 0.92)

        name = disambiguated_name(topos)
        name_label = _make_label(
            name,
            NSMakeRect(_PADDING, _ROW_HEIGHT - 24, width - _PADDING * 2, 18),
            size=13.0,
            bold=True,
            color=_text_color,
        )
        # Transparent cutout: text punches through the SDF fill
        if _METAL_OK:
            name_label.setWantsLayer_(True)
            name_label.layer().setCompositingFilter_("destinationOut")
        view.addSubview_(name_label)

        badge_parts = []
        if topos.temperature:
            badge_parts.append(topos.temperature)
        if topos.tool:
            badge_parts.append(topos.tool)
        if topos.observed:
            obs = _format_observed(topos.observed)
            badge_parts.append(obs)
        if badge_parts:
            badge_text = " · ".join(badge_parts)
            badge_label = _make_label(
                badge_text,
                NSMakeRect(_PADDING, _ROW_HEIGHT - 38, width - _PADDING * 2, 14),
                size=10.0,
                bold=False,
                color=_sub_color,
            )
            view.addSubview_(badge_label)

        if topos.status:
            snippet = topos.status.split(". ")[0]
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            status_label = _make_label(
                snippet,
                NSMakeRect(_PADDING, 4, width - _PADDING * 2, 14),
                size=10.0,
                bold=False,
                color=_sub_color,
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


def _animate_alpha(panel, target: float, duration: float) -> None:
    """Animate an NSPanel's alpha value using NSAnimationContext."""
    from AppKit import NSAnimationContext
    NSAnimationContext.beginGrouping()
    NSAnimationContext.currentContext().setDuration_(duration)
    panel.animator().setAlphaValue_(target)
    NSAnimationContext.endGrouping()


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
        self._metal_renderer = None  # set by TerraformHUD.setup()
        self._backing_scale = 2.0
        self.setWantsLayer_(True)
        # No masksToBounds — the gradient mask handles vertical clipping,
        # and we need horizontal frost bleed to extend past the scroll bounds.

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
        # Redraw Metal cards at new scroll position
        if self._metal_renderer is not None:
            self._metal_renderer.set_scroll_offset(self._scroll_offset * self._backing_scale)
            self._metal_renderer.draw_frame()


class TerraformHUD(NSObject):
    """Manages the Terraform topoi HUD panel."""

    def init(self):
        self = objc.super(TerraformHUD, self).init()
        if self is None:
            return None
        self._panel: NSPanel | None = None
        self._content_view: NSView | None = None
        self._scroll_view: _ManualScrollView | None = None
        self._metal_renderer = None
        self._topoi: list[Topos] = []
        self._timer: NSTimer | None = None
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

        # Restore saved position or default to left side of screen
        prefs = self._load_prefs()
        if prefs and "x" in prefs:
            x, y = float(prefs["x"]), float(prefs["y"])
        else:
            x = screen_frame.origin.x + 20
            y = screen_frame.origin.y + screen_frame.size.height - _PANEL_HEIGHT - 40
        frame = NSMakeRect(x, y, _PANEL_WIDTH, _PANEL_HEIGHT)

        if _METAL_OK:
            # Borderless: SDF glow provides visual framing, no title bar needed
            style = NSWindowStyleMaskNonactivatingPanel
        else:
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
        self._panel.setHasShadow_(False)  # shadow renders as a ghost copy on transparent windows
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setIgnoresMouseEvents_(False)
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self._panel.setFloatingPanel_(True)
        self._panel.setBecomesKeyOnlyIfNeeded_(True)

        # Layer-back the content view for scale animations
        self._panel.contentView().setWantsLayer_(True)
        content_frame = self._panel.contentView().bounds()

        # When Metal is active, the panel is wider for glow margin;
        # inset the content area so text stays within the card bounds
        inset = _GLOW_MARGIN if _METAL_OK else 0

        # Stats label at the top of the panel
        _STATS_HEIGHT = 20
        stats_frame = NSMakeRect(inset + 8, content_frame.size.height - _STATS_HEIGHT,
                                  content_frame.size.width - inset * 2 - 16, _STATS_HEIGHT)
        self._stats_label = _make_label(
            "", stats_frame, size=10.0, bold=False,
            color=NSColor.colorWithWhite_alpha_(0.45, 1.0),
        )
        self._panel.contentView().addSubview_(self._stats_label)

        # Scroll area — full width so Metal layer can extend into glow margins
        scroll_frame = NSMakeRect(0, 0, content_frame.size.width,
                                   content_frame.size.height - _STATS_HEIGHT)
        self._scroll_view = _ManualScrollView.alloc().initWithFrame_(scroll_frame)
        self._scroll_view.setAutoresizingMask_(18)  # width + height flex
        self._panel.contentView().addSubview_(self._scroll_view)

        # Metal SDF card renderer — replaces NSVisualEffectView frost
        if _METAL_OK:
            try:
                scale = screen.backingScaleFactor() if hasattr(screen, 'backingScaleFactor') else 2.0
                self._metal_renderer = TerraformCardRenderer(
                    (scroll_frame.size.width, scroll_frame.size.height), scale,
                )
                self._scroll_view._metal_renderer = self._metal_renderer
                self._scroll_view._backing_scale = scale
                # Insert Metal layer at bottom of scroll view's layer stack
                metal_layer = self._metal_renderer.layer()
                self._scroll_view.layer().insertSublayer_atIndex_(metal_layer, 0)
                # Additive blending — same visual language as the overlay
                if hasattr(metal_layer, "setCompositingFilter_"):
                    metal_layer.setCompositingFilter_("plusL")
            except Exception:
                logger.debug("Metal card renderer init failed, using frost fallback", exc_info=True)
                self._metal_renderer = None

        # Initial load
        self._refresh()

    def show(self) -> None:
        """Show the HUD panel with fade+scale pop and start auto-refresh."""
        if self._panel is None:
            self.setup()
        self._panel.setAlphaValue_(0.0)
        self._panel.orderFront_(None)
        self._visible = True
        self._save_position()
        self._start_timer()

        # Animate: fade in + subtle scale pop (0.92 → 1.0)
        content_layer = self._panel.contentView().layer()
        if content_layer is not None:
            content_layer.setAnchorPoint_((0.5, 0.5))
            bounds = self._panel.contentView().bounds()
            content_layer.setPosition_((bounds.size.width / 2, bounds.size.height / 2))

            scale_anim = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
            scale_anim.setFromValue_(0.92)
            scale_anim.setToValue_(1.0)
            scale_anim.setDuration_(0.2)
            scale_anim.setTimingFunction_(
                Quartz.CAMediaTimingFunction.functionWithName_("easeOut")
            )
            content_layer.addAnimation_forKey_(scale_anim, "show_scale")

        # Window-level fade
        _animate_alpha(self._panel, 1.0, 0.2)

    def hide(self) -> None:
        """Hide the HUD panel with fade-out and stop auto-refresh."""
        if self._panel is not None:
            self._save_position()
            self._visible = False
            self._stop_timer()

            # Animate: fade out + scale down (1.0 → 0.92)
            content_layer = self._panel.contentView().layer()
            if content_layer is not None:
                scale_anim = Quartz.CABasicAnimation.animationWithKeyPath_("transform.scale")
                scale_anim.setFromValue_(1.0)
                scale_anim.setToValue_(0.92)
                scale_anim.setDuration_(0.15)
                scale_anim.setTimingFunction_(
                    Quartz.CAMediaTimingFunction.functionWithName_("easeIn")
                )
                scale_anim.setFillMode_("forwards")
                scale_anim.setRemovedOnCompletion_(False)
                content_layer.addAnimation_forKey_(scale_anim, "hide_scale")

            # Fade out then hide
            _animate_alpha(self._panel, 0.0, 0.15)
            # Defer orderOut until animation finishes
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.16, self, "_hideComplete:", None, False
            )
            return
        self._visible = False
        self._stop_timer()

    def _hideComplete_(self, timer) -> None:
        if self._panel is not None and not self._visible:
            self._panel.orderOut_(None)
            # Reset scale for next show
            content_layer = self._panel.contentView().layer()
            if content_layer is not None:
                content_layer.removeAllAnimations()

    def toggle(self) -> None:
        """Toggle HUD visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()

    def set_brightness(self, brightness: float) -> None:
        """Update screen brightness for adaptive compositing.

        On dark backgrounds (< 0.15): additive blending ("plusL"),
        light fill color — cards glow.
        On light backgrounds: normal blending, dark fill — cards read
        as material cutouts.
        """
        self._brightness = min(max(brightness, 0.0), 1.0)
        if self._metal_renderer is None:
            return
        metal_layer = self._metal_renderer.layer()
        if not hasattr(metal_layer, "setCompositingFilter_"):
            return
        if self._brightness < 0.15:
            metal_layer.setCompositingFilter_("plusL")
        else:
            metal_layer.setCompositingFilter_(None)

    def restore_visibility(self) -> None:
        """Restore last saved visibility state (default: visible)."""
        prefs = self._load_prefs()
        if prefs and prefs.get("visible") is False:
            logger.info("Terror Form HUD: restoring hidden state")
        else:
            self.show()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._save_position()
        self._stop_timer()
        if self._panel is not None:
            self._panel.close()
            self._panel = None

    # -- private --

    def _save_position(self) -> None:
        """Persist panel position to disk."""
        if self._panel is None:
            return
        try:
            frame = self._panel.frame()
            _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PREFS_PATH.write_text(json.dumps({
                "x": frame.origin.x,
                "y": frame.origin.y,
                "visible": self._visible,
            }))
        except Exception:
            logger.debug("Failed to save HUD position", exc_info=True)

    @staticmethod
    def _load_prefs() -> dict | None:
        """Load saved HUD prefs (position + visibility), or None."""
        try:
            return json.loads(_PREFS_PATH.read_text())
        except Exception:
            return None

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

        # Skip full view rebuild if topos data hasn't changed
        topos_keys = [(t.id, t.temperature, t.status, t.tool, t.observed) for t in self._topoi]
        if topos_keys == getattr(self, '_last_topos_keys', None):
            # Data unchanged — just redraw Metal if active
            if self._metal_renderer is not None:
                self._metal_renderer.draw_frame()
            return
        self._last_topos_keys = topos_keys

        scroll_bounds = self._scroll_view.bounds()
        inset = _GLOW_MARGIN if self._metal_renderer is not None else 0
        # Card width: panel content minus glow margins and edge padding
        card_width = scroll_bounds.size.width - inset * 2 - 8
        row_stride = _ROW_HEIGHT + _ROW_GAP
        total_height = max(
            len(self._topoi) * row_stride + _PADDING,
            scroll_bounds.size.height,
        )

        # Build a fresh content view
        new_content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, scroll_bounds.size.width, total_height)
        )

        if self._metal_renderer is not None:
            # Metal SDF card surfaces — layer covers full scroll area
            scale = getattr(self._scroll_view, '_backing_scale', 2.0)
            self._metal_renderer.set_geometry(
                scroll_bounds.size.width, scroll_bounds.size.height, scale,
            )

            # Build card info list — inset by glow margin
            cards = []
            for i, topos in enumerate(self._topoi):
                y = total_height - (i + 1) * row_stride
                cr, cg, cb = _temp_fill_color(topos.temperature, getattr(self, '_brightness', 0.0))
                cards.append(CardInfo(
                    x=(inset + 4.0) * scale,
                    y=y * scale,
                    width=card_width * scale,
                    height=_ROW_HEIGHT * scale,
                    r=cr, g=cg, b=cb,
                    alpha=_SDF_CARD_ALPHA,
                ))
            self._metal_renderer.set_cards(cards)
            self._metal_renderer.set_scroll_offset(
                self._scroll_view._scroll_offset * scale,
            )
            self._metal_renderer.draw_frame()
        else:
            # Fallback: single frost layer behind all cards
            frost = NSVisualEffectView.alloc().initWithFrame_(
                NSMakeRect(0, 0, scroll_bounds.size.width, total_height)
            )
            frost.setMaterial_(4)  # NSVisualEffectMaterialDark
            frost.setBlendingMode_(1)  # NSVisualEffectBlendingModeBehindWindow
            frost.setState_(1)  # NSVisualEffectStateActive
            new_content.addSubview_(frost)

        for i, topos in enumerate(self._topoi):
            y = total_height - (i + 1) * row_stride
            row = ToposRowView.createWithTopos_width_(topos, card_width)
            row.setFrameOrigin_((inset + 4, y))
            new_content.addSubview_(row)

        # Swap content — old view is fully removed and deallocated
        self._scroll_view.setContent_(new_content)
        self._content_view = new_content
