"""Live preview optical-shell tuning HUD."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSFont,
    NSPanel,
    NSScreen,
    NSSlider,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject

logger = logging.getLogger(__name__)

_PREFS_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Spoke"
    / "preview_warp_hud.json"
)
_PANEL_WIDTH = 380.0
_PANEL_HEIGHT = 480.0
_PADDING_X = 14.0
_ROW_HEIGHT = 48.0
_TITLE_HEIGHT = 24.0

_NSWindowStyleMaskClosable = 1 << 1
_NSWindowStyleMaskResizable = 1 << 3
_NSWindowStyleMaskUtilityWindow = 1 << 4

_SLIDER_SPECS = [
    ("Core Mag", "core_magnification", 1.0, 2.5, "{:.2f}"),
    ("X Squeeze", "x_squeeze", 0.5, 4.5, "{:.2f}"),
    ("Y Squeeze", "y_squeeze", 0.5, 3.5, "{:.2f}"),
    ("Inflate X", "inflation_x_radii", 0.0, 4.0, "{:.2f}"),
    ("Inflate Y", "inflation_y_radii", 0.0, 4.0, "{:.2f}"),
    ("Bleed", "bleed_zone_frac", 0.0, 1.5, "{:.2f}"),
    ("Exterior Mix", "exterior_mix_width_points", 0.0, 80.0, "{:.1f}"),
    ("Ring Amp", "ring_amplitude_points", 0.0, 40.0, "{:.1f}"),
]


def _make_label(text: str, frame, *, size: float = 12.0, bold: bool = False, color=None):
    label = NSTextField.alloc().initWithFrame_(frame)
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
    label.setFont_(font)
    label.setTextColor_(color or NSColor.colorWithWhite_alpha_(0.15, 1.0))
    return label


class PreviewWarpHUD(NSObject):
    def initWithOverlay_(self, overlay):
        self = objc.super(PreviewWarpHUD, self).init()
        if self is None:
            return None
        self._overlay = overlay
        self._panel = None
        self._visible = False
        self._sliders: dict[str, object] = {}
        self._slider_keys: dict[object, str] = {}
        self._value_labels: dict[str, object] = {}
        self._formats = {key: fmt for _, key, _, _, fmt in _SLIDER_SPECS}
        return self

    def setup(self) -> None:
        if self._panel is not None:
            return
        screen = NSScreen.mainScreen()
        frame = screen.visibleFrame() if screen is not None else NSMakeRect(0, 0, 1440, 900)
        prefs = self._load_prefs()
        x = float(prefs.get("x", frame.origin.x + frame.size.width - _PANEL_WIDTH - 36.0))
        y = float(prefs.get("y", frame.origin.y + frame.size.height - _PANEL_HEIGHT - 72.0))

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, _PANEL_WIDTH, _PANEL_HEIGHT),
            _NSWindowStyleMaskClosable
            | _NSWindowStyleMaskResizable
            | _NSWindowStyleMaskUtilityWindow
            | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Preview Warp Tuner")
        panel.setLevel_(1000)
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.97, 0.94))
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)

        content = panel.contentView()
        title = _make_label(
            "Live Preview Warp Tuner",
            NSMakeRect(_PADDING_X, _PANEL_HEIGHT - 36.0, 220.0, _TITLE_HEIGHT),
            size=14.0,
            bold=True,
        )
        content.addSubview_(title)
        subtitle = _make_label(
            "Adjust live preview warp parameters without relaunching.",
            NSMakeRect(_PADDING_X, _PANEL_HEIGHT - 58.0, _PANEL_WIDTH - 2 * _PADDING_X, 16.0),
            size=10.5,
            color=NSColor.colorWithWhite_alpha_(0.28, 1.0),
        )
        content.addSubview_(subtitle)

        reset_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(_PANEL_WIDTH - 96.0, _PANEL_HEIGHT - 42.0, 80.0, 24.0)
        )
        reset_button.setTitle_("Reset")
        reset_button.setTarget_(self)
        reset_button.setAction_("resetDefaults:")
        content.addSubview_(reset_button)

        top = _PANEL_HEIGHT - 92.0
        for index, (title_text, key, min_value, max_value, _fmt) in enumerate(_SLIDER_SPECS):
            row_y = top - index * _ROW_HEIGHT
            name_label = _make_label(
                title_text,
                NSMakeRect(_PADDING_X, row_y + 20.0, 132.0, 18.0),
                size=11.5,
                bold=True,
            )
            content.addSubview_(name_label)
            value_label = _make_label(
                "",
                NSMakeRect(_PANEL_WIDTH - 84.0, row_y + 20.0, 68.0, 18.0),
                size=11.5,
                color=NSColor.colorWithWhite_alpha_(0.20, 1.0),
            )
            content.addSubview_(value_label)
            slider = NSSlider.alloc().initWithFrame_(
                NSMakeRect(_PADDING_X, row_y, _PANEL_WIDTH - 2 * _PADDING_X, 22.0)
            )
            slider.setMinValue_(min_value)
            slider.setMaxValue_(max_value)
            if hasattr(slider, "setContinuous_"):
                slider.setContinuous_(True)
            slider.setTarget_(self)
            slider.setAction_("sliderChanged:")
            content.addSubview_(slider)
            self._sliders[key] = slider
            self._slider_keys[slider] = key
            self._value_labels[key] = value_label

        self._panel = panel
        self._sync_from_overlay()

    def show(self) -> None:
        if self._panel is None:
            self.setup()
        self._sync_from_overlay()
        self._panel.orderFrontRegardless()
        self._visible = True
        self._save_prefs()

    def hide(self) -> None:
        if self._panel is not None:
            self._save_prefs()
            self._panel.orderOut_(None)
        self._visible = False
        self._save_prefs()

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def restore_visibility(self) -> None:
        prefs = self._load_prefs()
        if prefs.get("visible") is True:
            self.show()

    def cleanup(self) -> None:
        if self._panel is not None:
            self._save_prefs()
            self._panel.orderOut_(None)
            self._panel = None

    def sliderChanged_(self, sender) -> None:
        key = self._slider_keys.get(sender)
        if key is None or self._overlay is None:
            return
        value = float(sender.doubleValue())
        self._overlay.set_preview_warp_tuning_value(key, value)
        self._sync_from_overlay()

    def resetDefaults_(self, sender) -> None:
        if self._overlay is None:
            return
        self._overlay.reset_preview_warp_tuning()
        self._sync_from_overlay()

    def _sync_from_overlay(self) -> None:
        overlay = self._overlay
        if overlay is None or not hasattr(overlay, "preview_warp_tuning_snapshot"):
            return
        snapshot = overlay.preview_warp_tuning_snapshot()
        for key, slider in self._sliders.items():
            value = float(snapshot.get(key, 0.0))
            try:
                slider.setDoubleValue_(value)
            except Exception:
                logger.debug("Failed to sync preview warp slider %s", key, exc_info=True)
            label = self._value_labels.get(key)
            if label is not None:
                label.setStringValue_(self._formats[key].format(value))

    def _load_prefs(self) -> dict:
        try:
            return json.loads(_PREFS_PATH.read_text())
        except Exception:
            return {}

    def _save_prefs(self) -> None:
        if self._panel is None:
            return
        try:
            _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            frame = self._panel.frame()
            payload = {
                "x": float(frame.origin.x),
                "y": float(frame.origin.y),
                "visible": bool(self._visible),
            }
            _PREFS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            logger.debug("Failed to save preview warp HUD prefs", exc_info=True)
