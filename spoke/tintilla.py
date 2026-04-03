"""Vision Quest layer-visibility controls.

Tintilla is a small floating control panel for isolating SDF layer families
without restarting the app. It deliberately starts simple: per-layer toggles
plus all-on/all-off helpers.
"""

from __future__ import annotations

from collections.abc import Callable

import objc
from AppKit import (
    NSApp,
    NSButton,
    NSControlStateValueOff,
    NSControlStateValueOn,
    NSFloatingWindowLevel,
    NSPanel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSSwitchButton,
)
from Foundation import NSMakeRect, NSObject

SCREEN_GLOW_CORE_LAYER_ID = "screen_glow_core"
SCREEN_GLOW_TIGHT_BLOOM_LAYER_ID = "screen_glow_tight_bloom"
SCREEN_GLOW_WIDE_BLOOM_LAYER_ID = "screen_glow_wide_bloom"
SCREEN_VIGNETTE_CORE_LAYER_ID = "screen_vignette_core"
SCREEN_VIGNETTE_MID_LAYER_ID = "screen_vignette_mid"
SCREEN_VIGNETTE_TAIL_LAYER_ID = "screen_vignette_tail"
PREVIEW_FILL_LAYER_ID = "preview_fill"
COMMAND_FILL_LAYER_ID = "command_fill"

LAYER_SPECS: list[tuple[str, str]] = [
    (SCREEN_GLOW_CORE_LAYER_ID, "Screen Glow: Core"),
    (SCREEN_GLOW_TIGHT_BLOOM_LAYER_ID, "Screen Glow: Tight Bloom"),
    (SCREEN_GLOW_WIDE_BLOOM_LAYER_ID, "Screen Glow: Wide Bloom"),
    (SCREEN_VIGNETTE_CORE_LAYER_ID, "Screen Vignette: Core"),
    (SCREEN_VIGNETTE_MID_LAYER_ID, "Screen Vignette: Mid"),
    (SCREEN_VIGNETTE_TAIL_LAYER_ID, "Screen Vignette: Tail"),
    (PREVIEW_FILL_LAYER_ID, "Preview Fill"),
    (COMMAND_FILL_LAYER_ID, "Command Fill"),
]


class LayerVisibilityState:
    """Shared runtime visibility map for Vision Quest surfaces."""

    def __init__(self):
        self._enabled = {layer_id: True for layer_id, _ in LAYER_SPECS}
        self._listeners: list[Callable[["LayerVisibilityState"], None]] = []

    def add_listener(self, listener: Callable[["LayerVisibilityState"], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[["LayerVisibilityState"], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def is_visible(self, layer_id: str) -> bool:
        return self._enabled.get(layer_id, True)

    def set_enabled(self, layer_id: str, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._enabled.get(layer_id, True) == enabled:
            return
        self._enabled[layer_id] = enabled
        self._notify()

    def set_all_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        changed = False
        for layer_id in self._enabled:
            if self._enabled[layer_id] != enabled:
                self._enabled[layer_id] = enabled
                changed = True
        if changed:
            self._notify()

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self)


class TintillaPanelController(NSObject):
    """Floating panel for per-layer Vision Quest visibility toggles."""

    def initWithState_(self, state: LayerVisibilityState):
        self = objc.super(TintillaPanelController, self).init()
        if self is None:
            return None
        self._state = state
        self._panel = None
        self._buttons_by_layer_id: dict[str, object] = {}
        return self

    def setup(self) -> None:
        if self._panel is not None:
            return

        width = 280.0
        row_height = 26.0
        padding = 14.0
        top_controls = 38.0
        height = top_controls + padding * 2 + len(LAYER_SPECS) * row_height
        frame = NSMakeRect(80.0, 120.0, width, height)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable

        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            2,
            False,
        )
        self._panel.setTitle_("Tintilla Panel")
        self._panel.setFloatingPanel_(True)
        self._panel.setLevel_(NSFloatingWindowLevel)
        if hasattr(self._panel, "setReleasedWhenClosed_"):
            self._panel.setReleasedWhenClosed_(False)
        if hasattr(self._panel, "setHidesOnDeactivate_"):
            self._panel.setHidesOnDeactivate_(False)
        if hasattr(self._panel, "setBecomesKeyOnlyIfNeeded_"):
            self._panel.setBecomesKeyOnlyIfNeeded_(False)
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        self._panel.setContentView_(content)

        all_on = NSButton.alloc().initWithFrame_(NSMakeRect(padding, height - 34.0, 80.0, 24.0))
        all_on.setTitle_("All On")
        all_on.setTarget_(self)
        all_on.setAction_("enableAllLayers:")
        content.addSubview_(all_on)

        all_off = NSButton.alloc().initWithFrame_(NSMakeRect(100.0, height - 34.0, 80.0, 24.0))
        all_off.setTitle_("All Off")
        all_off.setTarget_(self)
        all_off.setAction_("disableAllLayers:")
        content.addSubview_(all_off)

        y = height - 64.0
        for index, (layer_id, label) in enumerate(LAYER_SPECS):
            button = NSButton.alloc().initWithFrame_(NSMakeRect(padding, y - index * row_height, width - padding * 2, 22.0))
            button.setButtonType_(NSSwitchButton)
            button.setTitle_(label)
            button.setState_(NSControlStateValueOn if self._state.is_visible(layer_id) else NSControlStateValueOff)
            button.setTarget_(self)
            button.setAction_("toggleLayer:")
            button.setTag_(index)
            content.addSubview_(button)
            self._buttons_by_layer_id[layer_id] = button

    def show(self) -> None:
        self.setup()
        self._refresh_controls()
        if NSApp is not None and hasattr(NSApp, "activateIgnoringOtherApps_"):
            NSApp.activateIgnoringOtherApps_(True)
        if hasattr(self._panel, "makeKeyAndOrderFront_"):
            self._panel.makeKeyAndOrderFront_(None)
        else:
            self._panel.orderFrontRegardless()

    def _refresh_controls(self) -> None:
        for index, (layer_id, _label) in enumerate(LAYER_SPECS):
            button = self._buttons_by_layer_id.get(layer_id)
            if button is None:
                continue
            button.setTag_(index)
            button.setState_(
                NSControlStateValueOn if self._state.is_visible(layer_id) else NSControlStateValueOff
            )

    def toggleLayer_(self, sender) -> None:
        index = sender.tag()
        layer_id, _label = LAYER_SPECS[index]
        self._state.set_enabled(layer_id, bool(sender.state()))
        self._refresh_controls()

    def enableAllLayers_(self, sender) -> None:
        self._state.set_all_enabled(True)
        self._refresh_controls()

    def disableAllLayers_(self, sender) -> None:
        self._state.set_all_enabled(False)
        self._refresh_controls()
