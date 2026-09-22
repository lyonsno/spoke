"""Focusable AppKit overlay for filtering and activating live Diauloi."""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from pathlib import Path

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSEvent,
    NSFont,
    NSImage,
    NSImageView,
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
from Foundation import NSMakeRect, NSObject, NSTimer

from .diaulos_switcher import (
    DiaulosActivationError,
    DiaulosInventoryError,
    DiaulosSwitcherModel,
    EpistaxisDiaulosClient,
)

logger = logging.getLogger(__name__)

_PANEL_WIDTH = 720.0
_PANEL_HEIGHT = 560.0
_PADDING = 24.0
_TITLE_HEIGHT = 24.0
_SEARCH_HEIGHT = 38.0
_STATUS_HEIGHT = 22.0
_ROW_HEIGHT = 58.0
_WINDOW_LEVEL = 1100
_NSWindowStyleMaskBorderless = 0
_NSApplicationActivateIgnoringOtherApps = 1 << 1
_NS_KEY_DOWN_MASK = 1 << 10
_UP_ARROW_KEYCODE = 126
_DOWN_ARROW_KEYCODE = 125
_ESCAPE_KEYCODE = 53
_ENTER_KEYCODES = {36, 76}
_WEZTERM_BUNDLE_IDENTIFIER = "com.github.wez.wezterm"


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
        self._load_in_flight = False
        self._prewarm_in_flight = False
        self._pending_inventory_payload = None
        self._pending_inventory_error_payload = None
        self._activation_generation = 0
        self._activation_in_flight = False
        self._activation_handle = None
        self._key_monitor_token = None
        self._key_monitor_handler = None
        self._keyboard_monitor_available = False
        self._last_render_signature = None
        self._shell_host = None
        self._shell_registered = False
        self._shell_unavailable = False
        self._row_buttons = []
        self.visible = False
        self.presentation_generation = 0
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
            NSColor.clearColor()
        )
        panel.setDelegate_(self)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setMovableByWindowBackground_(True)

        content = panel.contentView()
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(8.0)
        content.layer().setMasksToBounds_(True)
        content.layer().setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(0.07, 0.075, 0.075, 0.97).CGColor())
        title_y = _PANEL_HEIGHT - _PADDING - _TITLE_HEIGHT
        mark = NSImageView.alloc().initWithFrame_(NSMakeRect(_PADDING, title_y, 24, 24))
        mark.setImage_(NSImage.imageWithSystemSymbolName_accessibilityDescription_("point.topleft.down.curvedto.point.bottomright.up", "Teleporter"))
        mark.setContentTintColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(0.45, 0.91, 0.75, 1))
        content.addSubview_(mark)
        content.addSubview_(
            _label(
                "Teleporter",
                NSMakeRect(_PADDING + 34, title_y, 300.0, _TITLE_HEIGHT),
                size=18.0,
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
        self._search_field.setFocusRingType_(1)
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

    def prewarm(self) -> None:
        """Build the hidden panel and prime cached rows before the user gesture."""
        self.setup()
        if self._prewarm_in_flight or self._model.all_candidates:
            return
        self._prewarm_in_flight = True
        threading.Thread(
            target=self._prewarm_worker,
            daemon=True,
            name="diaulos-snapshot-prewarm",
        ).start()

    def _prewarm_worker(self) -> None:
        started_at = time.monotonic()
        try:
            payload = {
                "candidates": self._client.load(),
                "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
            }
        except DiaulosInventoryError as exc:
            payload = {
                "error": str(exc),
                "elapsed_ms": (time.monotonic() - started_at) * 1000.0,
            }
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "prewarmFinished:",
            payload,
            False,
        )

    def prewarmFinished_(self, payload: dict) -> None:
        self._prewarm_in_flight = False
        if payload.get("error"):
            logger.info(
                "Diaulos prewarm snapshot unavailable: elapsed_ms=%.1f error=%s",
                float(payload.get("elapsed_ms") or 0.0),
                payload["error"],
            )
            return
        if self.visible or self._load_in_flight or self._model.all_candidates:
            return
        self._model = DiaulosSwitcherModel(payload["candidates"])
        self._render_rows()
        logger.info(
            "Diaulos prewarm complete: elapsed_ms=%.1f rows=%d",
            float(payload.get("elapsed_ms") or 0.0),
            len(self._model.all_candidates),
        )

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def show(self) -> None:
        started_at = time.monotonic()
        self.setup()
        was_visible = self.visible
        workspace = NSWorkspace.sharedWorkspace()
        self._previous_app = workspace.frontmostApplication()
        self._search_field.setStringValue_("")
        self._search_field.setEnabled_(True)
        self._model.set_query("")
        self.visible = True
        if not was_visible:
            self.presentation_generation = (
                getattr(self, "presentation_generation", 0) + 1
            )
        self._install_key_monitor()
        self._set_status(
            "Refreshing live Diauloi"
            if self._model.all_candidates
            else "Loading live Diauloi"
        )
        self._panel.makeKeyAndOrderFront_(None)
        app = NSApp()
        if app is not None:
            app.activateIgnoringOtherApps_(True)
        self._panel.makeFirstResponder_(self._search_field)
        self._render_rows()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.0, self, "publishOpticalShell:", None, False,
        )
        logger.info(
            "Diaulos panel ordered front: elapsed_ms=%.1f cached_rows=%d",
            (time.monotonic() - started_at) * 1000.0,
            len(self._model.all_candidates),
        )
        if not getattr(self, "_load_in_flight", False):
            self._load_generation += 1
            generation = self._load_generation
            self._load_in_flight = True
            threading.Thread(
                target=self._load_worker,
                args=(generation,),
                daemon=True,
            ).start()

    def hide(self, *, restore_previous: bool = True, force: bool = False) -> bool:
        if self._activation_in_flight and not force:
            self._set_status(
                f"Focus committed to {self._activation_handle or 'selected Diaulos'}"
            )
            return False
        was_visible = self.visible
        self._activation_generation += 1
        self._remove_key_monitor()
        host = getattr(self, "_shell_host", None)
        if host is not None:
            host.release_client("spoke.teleporter")
        self._shell_host = None
        self._shell_registered = False
        if self._panel is not None:
            self._panel.orderOut_(None)
        self.visible = False
        if was_visible:
            self.presentation_generation = (
                getattr(self, "presentation_generation", 0) + 1
            )
        if restore_previous and self._previous_app is not None:
            try:
                self._previous_app.activateWithOptions_(
                    _NSApplicationActivateIgnoringOtherApps
                )
            except Exception:
                logger.debug("Could not restore prior foreground app", exc_info=True)
        self._previous_app = None
        return True

    def windowDidMove_(self, notification):
        if self.visible:
            self.publishOpticalShell_(None)

    def publishOpticalShell_(self, timer):
        if not self.visible or os.environ.get("SPOKE_TELEPORTER_OPTICAL_SHELL", "1") == "0":
            return
        registry = getattr(self._delegate, "_overlay_compositor_registry", None)
        if registry is None:
            return
        try:
            from .house_optical_primitive import compile_external_carrier_config
            from .optical_field import OpticalFieldRequest, OpticalFieldProfileRef, OpticalFieldPresentation
            from .perceptasia_throughglass import _display_local_scaled_window_bounds
            screen = self._panel.screen() or NSScreen.mainScreen()
            bounds, coordinates = _display_local_scaled_window_bounds(self._panel.frame(), screen)
            request = OpticalFieldRequest(
                caller_id="spoke.teleporter", continuity_key="spoke.teleporter", bounds=bounds,
                role="hud", state="rest", visible=True, visibility_scope="independent",
                presentation=OpticalFieldPresentation(layer="hud", order=43),
                presentation_layer="hud", layout_recipe="teleporter-native-carrier",
                profile=OpticalFieldProfileRef(base="assistant_shell"), z_index=43,
            )
            config = compile_external_carrier_config(request, carrier="external_native")
            config["optical_field"].update(coordinates)
            host = registry.host_for_screen(screen)
            if self._shell_host is not None and self._shell_host is not host:
                self._shell_host.release_client("spoke.teleporter")
                self._shell_registered = False
            self._shell_host = host
            if self._shell_registered:
                success = host.update_client_config("spoke.teleporter", config)
            else:
                success = host.add_client("spoke.teleporter", self._panel, self._panel.contentView(), config)
            self._shell_registered = bool(success)
            self._shell_unavailable = not success
            if not success:
                self._set_status("Native presentation; optical shell unavailable")
            logger.info("Teleporter House shell: registered=%s display=%s", success, host.display_id)
        except Exception:
            self._shell_unavailable = True
            self._set_status("Native presentation; optical shell unavailable")
            logger.exception("Teleporter optical publication failed; native controls remain usable")

    def selectCandidate_(self, sender):
        if self._activation_in_flight:
            return
        identity = str(sender.identifier())
        for index, current in enumerate(self._model.filtered):
            if json.dumps([current.diaulos_id, current.pane_id, current.thread_id]) == identity:
                self._model.selected_index = index
                self.activate_selected()
                return
        self._set_status("That observation changed; select the current row", error=True)

    def cleanup(self) -> None:
        if getattr(self, "_activation_in_flight", False):
            logger.info(
                "Hiding during shutdown without cancelling committed Diaulos focus"
            )
        self.hide(restore_previous=False, force=True)
        self._panel = None

    def set_dictation_filter(self, text: str) -> int | None:
        if not self.visible or self._activation_in_flight:
            return None
        self._search_field.setStringValue_(text)
        self._apply_query(text)
        self._panel.makeKeyAndOrderFront_(None)
        self._panel.makeFirstResponder_(self._search_field)
        return len(self._model.filtered)

    def show_error(self, message: str) -> None:
        self._set_status(message, error=True)

    def _install_key_monitor(self) -> None:
        if getattr(self, "_key_monitor_token", None) is not None:
            self._keyboard_monitor_available = True
            return

        def _handle(event):
            return self._handle_key_event(event)

        self._key_monitor_handler = _handle
        self._key_monitor_token = (
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                _NS_KEY_DOWN_MASK,
                _handle,
            )
        )
        self._keyboard_monitor_available = self._key_monitor_token is not None
        if self._key_monitor_token is None:
            logger.error("Diaulos switcher keyboard monitor installation failed")
            self._set_status("Keyboard navigation unavailable", error=True)

    def _remove_key_monitor(self) -> None:
        if getattr(self, "_key_monitor_token", None) is not None:
            NSEvent.removeMonitor_(self._key_monitor_token)
        self._key_monitor_token = None
        self._key_monitor_handler = None
        self._keyboard_monitor_available = False

    def _handle_key_event(self, event):
        if not self.visible:
            return event
        try:
            keycode = int(event.keyCode())
            if keycode == _UP_ARROW_KEYCODE:
                self.move_selection(-1)
                return None
            if keycode == _DOWN_ARROW_KEYCODE:
                self.move_selection(1)
                return None
            if keycode in _ENTER_KEYCODES:
                self.activate_selected()
                return None
            if keycode == _ESCAPE_KEYCODE:
                self.hide()
                return None
        except Exception:
            logger.exception("Diaulos switcher key monitor handler failed")
        return event

    def controlTextDidChange_(self, notification) -> None:
        if self._activation_in_flight:
            return
        self._apply_query(str(self._search_field.stringValue() or ""))

    def move_selection(self, delta: int) -> None:
        if self._activation_in_flight:
            return
        self._model.move(delta)
        self._render_rows()

    def activate_selected(self) -> None:
        if self._activation_in_flight:
            self._set_status(
                f"Focus committed to {self._activation_handle or 'selected Diaulos'}"
            )
            return
        candidate = self._model.selected
        if candidate is None:
            self._set_status("No live Diaulos matches this filter", error=True)
            return
        self._activation_in_flight = True
        self._activation_handle = candidate.handle
        self._search_field.setEnabled_(False)
        self._activation_generation += 1
        generation = self._activation_generation
        self._set_status(f"Focusing {candidate.handle} (committed)")
        threading.Thread(
            target=self._activation_worker,
            args=(generation, candidate),
            daemon=True,
        ).start()

    def inventoryLoaded_(self, payload: dict) -> None:
        if payload["generation"] != self._load_generation:
            return
        refreshing = bool(payload.get("refreshing"))
        if not refreshing:
            self._load_in_flight = False
        if getattr(self, "_activation_in_flight", False):
            if payload.get("error"):
                self._pending_inventory_error_payload = payload
            else:
                self._pending_inventory_payload = payload
                if not refreshing:
                    self._pending_inventory_error_payload = None
            logger.info(
                "Diaulos inventory application deferred behind committed activation: "
                "generation=%s refreshing=%s",
                payload["generation"],
                refreshing,
            )
            return
        self._apply_inventory_payload(payload)

    def _apply_inventory_payload(self, payload: dict) -> None:
        refreshing = bool(payload.get("refreshing"))
        error = payload.get("error")
        if error:
            if self.visible:
                suffix = (
                    "; showing last live observation"
                    if self._model.all_candidates
                    else ""
                )
                self._set_status(f"{error}{suffix}", error=True)
            return
        self._model = DiaulosSwitcherModel(payload["candidates"])
        if not self.visible:
            return
        self._apply_query(str(self._search_field.stringValue() or ""))
        if refreshing:
            status = (
                f"Snapshot observation {payload['candidates'][0].observed_at}; refreshing"
                if payload["candidates"]
                else "Snapshot has no verified-live Diauloi; refreshing"
            )
        else:
            status = (
                f"Live observation {payload['candidates'][0].observed_at}"
                if payload["candidates"]
                else "No verified-live Diauloi"
            )
        self._set_status(status)

    def activationFinished_(self, payload: dict) -> None:
        if payload["generation"] != self._activation_generation or not self.visible:
            return
        self._activation_in_flight = False
        self._activation_handle = None
        self._search_field.setEnabled_(True)
        pending_inventory = getattr(self, "_pending_inventory_payload", None)
        pending_inventory_error = getattr(
            self, "_pending_inventory_error_payload", None
        )
        self._pending_inventory_payload = None
        self._pending_inventory_error_payload = None
        if payload.get("error"):
            activation_error = str(payload["error"])
            self._panel.makeFirstResponder_(self._search_field)
            if pending_inventory is not None:
                self._apply_inventory_payload(pending_inventory)
            if pending_inventory_error is not None:
                refresh_error = str(pending_inventory_error["error"])
                self._set_status(
                    f"{activation_error}; inventory refresh failed: {refresh_error}",
                    error=True,
                )
            else:
                self._set_status(activation_error, error=True)
            return
        self.hide(restore_previous=False)
        self._activate_wezterm()
        if pending_inventory is not None:
            self._apply_inventory_payload(pending_inventory)

    def _activate_wezterm(self) -> None:
        workspace = NSWorkspace.sharedWorkspace()
        for app in workspace.runningApplications() or []:
            try:
                if str(app.bundleIdentifier() or "") != _WEZTERM_BUNDLE_IDENTIFIER:
                    continue
                if app.activateWithOptions_(
                    _NSApplicationActivateIgnoringOtherApps
                ):
                    return
            except Exception:
                logger.debug("Could not foreground WezTerm application", exc_info=True)
        logger.error("Focused Diaulos pane but could not foreground WezTerm")

    def _load_worker(self, generation: int) -> None:
        started_at = time.monotonic()
        snapshot_error = None
        try:
            candidates = self._client.load()
            self._publish_inventory(
                {
                    "generation": generation,
                    "candidates": candidates,
                    "refreshing": True,
                }
            )
        except DiaulosInventoryError as exc:
            snapshot_error = str(exc)

        try:
            candidates = self._client.refresh()
            payload = {
                "generation": generation,
                "candidates": candidates,
                "refreshing": False,
            }
        except DiaulosInventoryError as exc:
            error = str(exc)
            if snapshot_error is not None:
                error = f"{snapshot_error}; refresh failed: {error}"
            payload = {
                "generation": generation,
                "error": error,
                "refreshing": False,
            }
        self._publish_inventory(payload)
        logger.info(
            "Diaulos inventory worker complete: generation=%s elapsed_ms=%.1f "
            "outcome=%s",
            generation,
            (time.monotonic() - started_at) * 1000.0,
            "error" if payload.get("error") else "complete",
        )

    def _publish_inventory(self, payload: dict) -> None:
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "inventoryLoaded:",
            payload,
            False,
        )

    def _activation_worker(self, generation: int, candidate) -> None:
        started_at = time.monotonic()
        try:
            receipt = self._client.activate(candidate)
            payload = {"generation": generation, "receipt": receipt}
        except DiaulosActivationError as exc:
            payload = {"generation": generation, "error": str(exc)}
        except Exception as exc:
            logger.exception(
                "Unexpected Diaulos activation failure: generation=%s handle=%s",
                generation,
                candidate.handle,
            )
            payload = {
                "generation": generation,
                "error": f"unexpected activation failure: {type(exc).__name__}: {exc}",
            }
        logger.info(
            "Diaulos activation worker complete: generation=%s handle=%s "
            "elapsed_ms=%.1f outcome=%s",
            generation,
            candidate.handle,
            (time.monotonic() - started_at) * 1000.0,
            "error" if payload.get("error") else "complete",
        )
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
        render_signature = (
            self._model.query,
            self._model.selected_index,
            tuple(
                (
                    candidate.handle,
                    candidate.pane_id,
                    candidate.title,
                    candidate.cwd,
                )
                for candidate in self._model.filtered
            ),
        )
        if getattr(self, "_last_render_signature", None) == render_signature:
            return
        for view in list(self._document_view.subviews()):
            view.removeFromSuperview()
        self._row_buttons = []

        width = _PANEL_WIDTH - 2.0 * _PADDING
        viewport_height = float(self._scroll_view.contentSize().height)
        document_height = max(viewport_height, len(self._model.filtered) * _ROW_HEIGHT)
        self._document_view.setFrame_(NSMakeRect(0, 0, width, document_height))
        for index, candidate in enumerate(self._model.filtered):
            y = document_height - (index + 1) * _ROW_HEIGHT
            selected = index == self._model.selected_index
            if selected:
                background = NSView.alloc().initWithFrame_(NSMakeRect(0, y + 2, width - 3, _ROW_HEIGHT - 4))
                background.setWantsLayer_(True)
                background.layer().setCornerRadius_(6)
                background.layer().setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(0.14, 0.23, 0.20, 1).CGColor())
                self._document_view.addSubview_(background)
            title_color = (
                NSColor.colorWithSRGBRed_green_blue_alpha_(0.56, 0.96, 0.80, 1.0)
                if selected
                else NSColor.colorWithSRGBRed_green_blue_alpha_(0.91, 0.93, 0.95, 1.0)
            )
            self._document_view.addSubview_(
                _label(
                    candidate.handle,
                    NSMakeRect(14.0, y + 28.0, width - 110.0, 22.0),
                    size=14.0,
                    bold=selected,
                    color=title_color,
                )
            )
            detail = candidate.title or Path(candidate.cwd).name or candidate.cwd
            route = f"Pane {candidate.pane_id}"
            if detail:
                route += f"  {detail}"
            self._document_view.addSubview_(
                _label(
                    route,
                    NSMakeRect(14.0, y + 9.0, width - 40.0, 18.0),
                    size=11.0,
                    color=NSColor.colorWithSRGBRed_green_blue_alpha_(
                        0.52, 0.59, 0.65, 1.0
                    ),
                )
            )
            backend = _label(candidate.resume_backend.title(), NSMakeRect(width - 98, y + 29, 80, 18),
                             size=10, color=NSColor.colorWithSRGBRed_green_blue_alpha_(0.82, 0.73, 0.54, 1))
            backend.setAlignment_(2)
            self._document_view.addSubview_(backend)
            button = NSButton.buttonWithTitle_target_action_("", self, "selectCandidate:")
            button.setFrame_(NSMakeRect(0, y, width, _ROW_HEIGHT))
            button.setBordered_(False)
            button.setToolTip_(f"Focus {candidate.handle}")
            button.setIdentifier_(json.dumps([candidate.diaulos_id, candidate.pane_id, candidate.thread_id]))
            self._row_buttons.append(button)
            self._document_view.addSubview_(button)
            if selected:
                self._document_view.scrollRectToVisible_(
                    NSMakeRect(0, y, width, _ROW_HEIGHT)
                )
        self._count_label.setStringValue_(
            f"{len(self._model.filtered)} live"
            if self._model.query
            else f"{len(self._model.all_candidates)} live"
        )
        self._last_render_signature = render_signature

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if self._status_label is None:
            return
        if self.visible and not getattr(self, "_keyboard_monitor_available", False):
            if "Keyboard navigation unavailable" not in text:
                text = f"Keyboard navigation unavailable — {text}"
            error = True
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
