"""Focusable AppKit overlay for filtering and activating live Diauloi."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSPanel,
    NSScreen,
    NSScrollView,
    NSTextField,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWorkspace,
)
from Foundation import NSMakeRect, NSObject

from .diaulos_switcher import (
    DiaulosActivationError,
    DiaulosInventoryError,
    DiaulosSwitcherModel,
    EpistaxisDiaulosClient,
)

logger = logging.getLogger(__name__)

_PANEL_WIDTH = 680.0
_PANEL_HEIGHT = 520.0
_PADDING = 20.0
_TITLE_HEIGHT = 24.0
_SEARCH_HEIGHT = 38.0
_STATUS_HEIGHT = 22.0
_ROW_HEIGHT = 54.0
_WINDOW_LEVEL = 1100
_NSWindowStyleMaskBorderless = 0
_NSApplicationActivateIgnoringOtherApps = 1 << 1
_UP_ARROW_KEYCODE = 126
_DOWN_ARROW_KEYCODE = 125
_ESCAPE_KEYCODE = 53
_ENTER_KEYCODES = {36, 76}


def _label(text: str, frame, *, size: float, color, bold: bool = False):
    label = NSTextField.alloc().initWithFrame_(frame)
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setFont_(
        NSFont.boldSystemFontOfSize_(size)
        if bold
        else NSFont.systemFontOfSize_(size)
    )
    label.setTextColor_(color)
    return label


class _SwitcherPanel(NSPanel):
    def canBecomeKeyWindow(self):
        return True

    def canBecomeMainWindow(self):
        return False


class _SwitcherSearchField(NSTextField):
    def initWithFrame_owner_(self, frame, owner):
        self = objc.super(_SwitcherSearchField, self).initWithFrame_(frame)
        if self is not None:
            self._switcher_owner = owner
        return self

    def keyDown_(self, event):
        keycode = int(event.keyCode())
        if keycode == _UP_ARROW_KEYCODE:
            self._switcher_owner.move_selection(-1)
            return
        if keycode == _DOWN_ARROW_KEYCODE:
            self._switcher_owner.move_selection(1)
            return
        if keycode in _ENTER_KEYCODES:
            self._switcher_owner.activate_selected()
            return
        if keycode == _ESCAPE_KEYCODE:
            self._switcher_owner.hide()
            return
        objc.super(_SwitcherSearchField, self).keyDown_(event)


class DiaulosSwitcherOverlay(NSObject):
    def initWithDelegate_(self, delegate):
        self = objc.super(DiaulosSwitcherOverlay, self).init()
        if self is None:
            return None
        self._delegate = delegate
        self._client = EpistaxisDiaulosClient()
        self._model = DiaulosSwitcherModel([])
        self._panel = None
        self._search_field = None
        self._count_label = None
        self._status_label = None
        self._scroll_view = None
        self._document_view = None
        self._previous_app = None
        self._load_generation = 0
        self._activation_generation = 0
        self.visible = False
        return self

    def setup(self) -> None:
        if self._panel is not None:
            return
        screen = NSScreen.mainScreen()
        visible = (
            screen.visibleFrame()
            if screen is not None
            else NSMakeRect(0, 0, 1440, 900)
        )
        x = visible.origin.x + (visible.size.width - _PANEL_WIDTH) / 2.0
        y = visible.origin.y + visible.size.height - _PANEL_HEIGHT - 72.0
        panel = _SwitcherPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, _PANEL_WIDTH, _PANEL_HEIGHT),
            _NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(_WINDOW_LEVEL)
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.setBackgroundColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(0.045, 0.052, 0.06, 0.985)
        )
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setMovableByWindowBackground_(True)

        content = panel.contentView()
        title_y = _PANEL_HEIGHT - _PADDING - _TITLE_HEIGHT
        content.addSubview_(
            _label(
                "DIAULOI",
                NSMakeRect(_PADDING, title_y, 300.0, _TITLE_HEIGHT),
                size=15.0,
                bold=True,
                color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                    0.94, 0.95, 0.96, 1.0
                ),
            )
        )
        self._count_label = _label(
            "",
            NSMakeRect(
                _PANEL_WIDTH - _PADDING - 160.0,
                title_y,
                160.0,
                _TITLE_HEIGHT,
            ),
            size=12.0,
            color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.56, 0.62, 0.68, 1.0
            ),
        )
        self._count_label.setAlignment_(2)
        content.addSubview_(self._count_label)

        search_y = title_y - 14.0 - _SEARCH_HEIGHT
        self._search_field = (
            _SwitcherSearchField.alloc().initWithFrame_owner_(
                NSMakeRect(
                    _PADDING,
                    search_y,
                    _PANEL_WIDTH - 2.0 * _PADDING,
                    _SEARCH_HEIGHT,
                ),
                self,
            )
        )
        self._search_field.setPlaceholderString_("Find a Diaulos")
        self._search_field.setFont_(NSFont.systemFontOfSize_(16.0))
        self._search_field.setDelegate_(self)
        content.addSubview_(self._search_field)

        status_y = _PADDING
        list_y = status_y + _STATUS_HEIGHT + 8.0
        list_height = search_y - 12.0 - list_y
        self._scroll_view = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(_PADDING, list_y, _PANEL_WIDTH - 2.0 * _PADDING, list_height)
        )
        self._scroll_view.setHasVerticalScroller_(True)
        self._scroll_view.setDrawsBackground_(False)
        self._document_view = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _PANEL_WIDTH - 2.0 * _PADDING, list_height)
        )
        self._scroll_view.setDocumentView_(self._document_view)
        content.addSubview_(self._scroll_view)

        self._status_label = _label(
            "",
            NSMakeRect(_PADDING, status_y, _PANEL_WIDTH - 2.0 * _PADDING, _STATUS_HEIGHT),
            size=11.5,
            color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.56, 0.62, 0.68, 1.0
            ),
        )
        content.addSubview_(self._status_label)
        self._panel = panel

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        self.setup()
        workspace = NSWorkspace.sharedWorkspace()
        self._previous_app = workspace.frontmostApplication()
        self.visible = True
        self._set_status("Loading live Diauloi")
        self._count_label.setStringValue_("")
        self._panel.makeKeyAndOrderFront_(None)
        app = NSApp()
        if app is not None:
            app.activateIgnoringOtherApps_(True)
        self._panel.makeFirstResponder_(self._search_field)
        self._load_generation += 1
        generation = self._load_generation
        threading.Thread(
            target=self._load_worker,
            args=(generation,),
            daemon=True,
        ).start()

    def hide(self, *, restore_previous: bool = True) -> None:
        self._load_generation += 1
        self._activation_generation += 1
        if self._panel is not None:
            self._panel.orderOut_(None)
        self.visible = False
        if restore_previous and self._previous_app is not None:
            try:
                self._previous_app.activateWithOptions_(
                    _NSApplicationActivateIgnoringOtherApps
                )
            except Exception:
                logger.debug("Could not restore prior foreground app", exc_info=True)
        self._previous_app = None

    def cleanup(self) -> None:
        self.hide(restore_previous=False)
        self._panel = None

    def set_dictation_filter(self, text: str) -> None:
        if not self.visible:
            return
        self._search_field.setStringValue_(text)
        self._apply_query(text)
        self._panel.makeKeyAndOrderFront_(None)
        self._panel.makeFirstResponder_(self._search_field)

    def show_error(self, message: str) -> None:
        self._set_status(message, error=True)

    def controlTextDidChange_(self, notification) -> None:
        self._apply_query(str(self._search_field.stringValue() or ""))

    def move_selection(self, delta: int) -> None:
        self._model.move(delta)
        self._render_rows()

    def activate_selected(self) -> None:
        candidate = self._model.selected
        if candidate is None:
            self._set_status("No live Diaulos matches this filter", error=True)
            return
        self._activation_generation += 1
        generation = self._activation_generation
        self._set_status(f"Focusing {candidate.handle}")
        threading.Thread(
            target=self._activation_worker,
            args=(generation, candidate),
            daemon=True,
        ).start()

    def inventoryLoaded_(self, payload: dict) -> None:
        if payload["generation"] != self._load_generation or not self.visible:
            return
        error = payload.get("error")
        if error:
            self._model = DiaulosSwitcherModel([])
            self._render_rows()
            self._set_status(str(error), error=True)
            return
        self._model = DiaulosSwitcherModel(payload["candidates"])
        self._apply_query(str(self._search_field.stringValue() or ""))
        self._set_status(
            f"Live observation {payload['candidates'][0].observed_at}"
            if payload["candidates"]
            else "No verified-live Diauloi"
        )

    def activationFinished_(self, payload: dict) -> None:
        if payload["generation"] != self._activation_generation or not self.visible:
            return
        if payload.get("error"):
            self._set_status(str(payload["error"]), error=True)
            return
        self.hide(restore_previous=False)

    def _load_worker(self, generation: int) -> None:
        try:
            candidates = self._client.load()
            payload = {"generation": generation, "candidates": candidates}
        except DiaulosInventoryError as exc:
            payload = {"generation": generation, "error": str(exc)}
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "inventoryLoaded:",
            payload,
            False,
        )

    def _activation_worker(self, generation: int, candidate) -> None:
        try:
            receipt = self._client.activate(candidate)
            payload = {"generation": generation, "receipt": receipt}
        except DiaulosActivationError as exc:
            payload = {"generation": generation, "error": str(exc)}
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "activationFinished:",
            payload,
            False,
        )

    def _apply_query(self, query: str) -> None:
        self._model.set_query(query)
        self._render_rows()

    def _render_rows(self) -> None:
        if self._document_view is None:
            return
        for view in list(self._document_view.subviews()):
            view.removeFromSuperview()

        width = _PANEL_WIDTH - 2.0 * _PADDING
        viewport_height = float(self._scroll_view.contentSize().height)
        document_height = max(viewport_height, len(self._model.filtered) * _ROW_HEIGHT)
        self._document_view.setFrame_(NSMakeRect(0, 0, width, document_height))
        for index, candidate in enumerate(self._model.filtered):
            y = document_height - (index + 1) * _ROW_HEIGHT
            selected = index == self._model.selected_index
            marker = "> " if selected else "  "
            title_color = (
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.27, 0.85, 0.93, 1.0)
                if selected
                else NSColor.colorWithSRGBRed_green_blue_alpha_(0.91, 0.93, 0.95, 1.0)
            )
            self._document_view.addSubview_(
                _label(
                    marker + candidate.handle,
                    NSMakeRect(6.0, y + 25.0, width - 12.0, 22.0),
                    size=14.0,
                    bold=selected,
                    color=title_color,
                )
            )
            detail = candidate.title or Path(candidate.cwd).name or candidate.cwd
            route = f"pane {candidate.pane_id}"
            if detail:
                route += f"  {detail}"
            self._document_view.addSubview_(
                _label(
                    route,
                    NSMakeRect(26.0, y + 6.0, width - 32.0, 18.0),
                    size=11.0,
                    color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.52, 0.59, 0.65, 1.0
                    ),
                )
            )
            if selected:
                self._document_view.scrollRectToVisible_(
                    NSMakeRect(0, y, width, _ROW_HEIGHT)
                )
        self._count_label.setStringValue_(
            f"{len(self._model.filtered)} live"
            if self._model.query
            else f"{len(self._model.all_candidates)} live"
        )

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if self._status_label is None:
            return
        self._status_label.setStringValue_(text)
        self._status_label.setTextColor_(
            NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.96, 0.36, 0.32, 1.0
            )
            if error
            else NSColor.colorWithSRGBRed_green_blue_alpha_(
                0.56, 0.62, 0.68, 1.0
            )
        )
