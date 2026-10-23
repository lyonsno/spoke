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
from Foundation import NSMakeRect, NSObject, NSTimer, NSRunLoop
from Quartz import CALayer, CAShapeLayer, CATransaction, CGPathCreateWithRoundedRect

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
_SHELL_CLIENT_ID = "spoke.teleporter"
# Same native-readiness grace as CommandOverlay; not a capture timeout.
_SHELL_START_GRACE_S = 0.25


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
        self._shell_transition = None
        self._shell_final = None
        self._shell_tick_timer = None
        self._shell_epoch = 0
        self._shell_waiting = False
        self._shell_sidecars_registered = False
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

        root = panel.contentView()
        root.setWantsLayer_(True)
        root.layer().setCornerRadius_(8.0)
        root.layer().setMasksToBounds_(True)
        self._shell_body_layer = CALayer.layer()
        self._shell_body_layer.setFrame_(root.bounds())
        self._shell_body_layer.setCornerRadius_(8.0)
        self._shell_body_layer.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(0.07, 0.075, 0.075, 0.97).CGColor())
        root.layer().addSublayer_(self._shell_body_layer)
        content = NSView.alloc().initWithFrame_(root.bounds())
        content.setWantsLayer_(True)
        root.addSubview_(content)
        self._shell_foreground = content
        self._shell_text_mask = CAShapeLayer.layer()
        content.layer().setMask_(self._shell_text_mask)
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
        self._set_shell_planes(dict(opacity=1.0, height_frac=1.0), dict(alpha=1.0, width_frac=1.0, height_frac=1.0))

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
        from .window_focus import WindowFocus
        if not was_visible:
            self._focus_return = WindowFocus.capture()
            self._previous_app = self._focus_return.app if self._focus_return is not None else None
        self._search_field.setStringValue_("")
        self._search_field.setEnabled_(True)
        self._model.set_query("")
        self.visible = True
        if not was_visible:
            self.presentation_generation = (
                getattr(self, "presentation_generation", 0) + 1
            )
            self._prepare_shell_entrance()
        self._panel.setIgnoresMouseEvents_(False)
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
        was_key = bool(self._panel is not None and self._panel.isKeyWindow())
        self._activation_generation += 1
        self._remove_key_monitor()
        self.visible = False
        if was_visible:
            self.presentation_generation = (
                getattr(self, "presentation_generation", 0) + 1
            )
        transition = getattr(self, "_shell_transition", None)
        animate = (was_visible and not force and transition is not None
                   and not getattr(self, "_shell_waiting", False)
                   and getattr(self, "_shell_registered", False)
                   and getattr(self, "_shell_final", None) is not None)
        if animate:
            self._shell_epoch += 1
            transition.dismiss(time.perf_counter(), exposed_text=getattr(self, "_shell_applied_text", None))
            self._panel.setIgnoresMouseEvents_(True)
            self._panel.resignKeyWindow()
            self._start_shell_ticks()
        else:
            self._finish_shell()
        if restore_previous and self._previous_app is not None:
            try:
                from .window_focus import restore_window_focus
                restore_window_focus(self, self._panel, was_key=was_key)
            except Exception:
                logger.debug("Could not restore prior foreground app", exc_info=True)
        self._previous_app = None
        return True

    def _prepare_shell_entrance(self):
        if (os.environ.get("SPOKE_TELEPORTER_OPTICAL_SHELL", "1") == "0"
                or getattr(getattr(self, "_delegate", None), "_overlay_compositor_registry", None) is None):
            return
        from .optical_lifecycle import HouseTransition

        self._shell_epoch = getattr(self, "_shell_epoch", 0) + 1
        now = time.perf_counter()
        transition = getattr(self, "_shell_transition", None)
        reversing = transition is not None and transition.phase == "dismiss" and self._shell_registered
        if not reversing:
            transition = HouseTransition()
        transition.summon(now)
        self._shell_transition = transition
        self._shell_waiting = not reversing
        self._shell_wait_started = now
        self._shell_present_armed = False
        self._shell_applied_text = dict(alpha=0.0, width_frac=1.0, height_frac=0.0)
        self._panel.setHasShadow_(False)
        self._set_shell_planes(dict(opacity=0.0, height_frac=0.0), dict(alpha=0.0, width_frac=1.0, height_frac=0.0))
        self._start_shell_ticks()

    def _start_shell_ticks(self):
        timer = getattr(self, "_shell_tick_timer", None)
        if timer is not None:
            timer.invalidate()
        self._shell_tick_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 60.0, self, "opticalShellStep:", None, True,
        )
        for mode in ("NSRunLoopCommonModes", "NSEventTrackingRunLoopMode"):
            NSRunLoop.currentRunLoop().addTimer_forMode_(self._shell_tick_timer, mode)

    def _stop_shell_ticks(self):
        timer = getattr(self, "_shell_tick_timer", None)
        self._shell_tick_timer = None
        if timer is not None:
            timer.invalidate()

    def _finish_shell(self):
        self._stop_shell_ticks()
        self._shell_epoch = getattr(self, "_shell_epoch", 0) + 1
        self._shell_waiting = False
        host = getattr(self, "_shell_host", None)
        if host is not None:
            for client_id in self._shell_client_ids():
                host.release_client(client_id)
        self._shell_host = None
        self._shell_registered = False
        self._shell_sidecars_registered = False
        self._shell_transition = None
        self._shell_final = None
        if self._panel is not None:
            if not self.visible:
                self._panel.orderOut_(None)
            else:
                self._set_shell_planes(dict(opacity=1.0, height_frac=1.0), dict(alpha=1.0, width_frac=1.0, height_frac=1.0))
                self._panel.setHasShadow_(True)

    def _shell_client_ids(self):
        ids = [_SHELL_CLIENT_ID]
        if getattr(self, "_shell_sidecars_registered", False):
            ids.extend(f"{_SHELL_CLIENT_ID}.{suffix}" for suffix in ("dismiss_seam", "dismiss_radial_pucker"))
        return ids

    def _set_shell_planes(self, body, text):
        foreground = getattr(self, "_shell_foreground", None)
        if foreground is None:
            return
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        try:
            self._shell_body_layer.setValue_forKeyPath_(max(body["height_frac"], .001), "transform.scale.y")
            self._shell_body_layer.setOpacity_(body["opacity"])
            width, height = _PANEL_WIDTH * text["width_frac"], _PANEL_HEIGHT * text["height_frac"]
            rect = ((_PANEL_WIDTH - width) / 2, (_PANEL_HEIGHT - height) / 2), (width, height)
            self._shell_text_mask.setPath_(CGPathCreateWithRoundedRect(rect, min(8.0, height / 2), min(8.0, height / 2), None))
            foreground.layer().setOpacity_(text["alpha"])
        finally:
            CATransaction.commit()

    def _publish_shell_frame(self, now):
        transition = self._shell_transition
        sample_time = self._shell_wait_started if self._shell_waiting else now
        frame = transition.frame(self._shell_final, sample_time)
        if not self._shell_host.update_client_configs(frame.configs):
            raise RuntimeError("House transition publication rejected")
        self._set_shell_planes(frame.body, frame.text)
        self._shell_applied_text = dict(frame.text)
        return frame

    def opticalShellStep_(self, timer):
        if timer is not getattr(self, "_shell_tick_timer", None):
            return
        now = time.perf_counter()
        if getattr(self, "_shell_waiting", False):
            if now - self._shell_wait_started < _SHELL_START_GRACE_S:
                return
            # Never rewind an already usable native panel when capture arrives late.
            self._shell_waiting = False
            self._shell_transition.rest()
        if getattr(self, "_shell_final", None) is None:
            self._finish_shell()
            return
        try:
            frame = self._publish_shell_frame(now)
            if frame.phase == "hidden":
                self._finish_shell()
            elif frame.phase == "rest":
                self._shell_transition.rest()
                self._panel.setHasShadow_(True)
                self._stop_shell_ticks()
            elif frame.phase == "tail":
                self._panel.orderOut_(None)
        except Exception:
            logger.exception("Teleporter transition failed; retaining native controls")
            self._finish_shell()

    def opticalShellDidPresent_(self, payload):
        if (not self.visible or not getattr(self, "_shell_waiting", False)
                or payload.get("epoch") != self._shell_epoch
                or payload.get("host") is not self._shell_host):
            return
        now = time.perf_counter()
        if now - self._shell_wait_started >= _SHELL_START_GRACE_S:
            self._shell_transition.rest()
        else:
            self._shell_transition.started_at = now
        self._shell_waiting = False
        self.opticalShellStep_(self._shell_tick_timer)

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
                for client_id in self._shell_client_ids():
                    self._shell_host.release_client(client_id)
                self._shell_registered = False
                self._shell_sidecars_registered = False
            self._shell_host = host
            if not self._shell_registered:
                from .fullscreen_compositor import OverlayClientIdentity

                identity = OverlayClientIdentity(
                    client_id="spoke.teleporter",
                    display_id=host.display_id,
                    role="hud",
                )

                def capture_state_changed(state, error=None):
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "opticalShellCaptureStateChanged:",
                        {"host": host, "state": state, "error": error},
                        False,
                    )

                self._shell_client = host.register_client(
                    identity,
                    window=self._panel,
                    content_view=self._panel.contentView(),
                    on_capture_state=capture_state_changed,
                )
            transition = getattr(self, "_shell_transition", None)
            if transition is not None:
                if not self._shell_sidecars_registered:
                    from .fullscreen_compositor import OverlayClientIdentity

                    for suffix in ("dismiss_seam", "dismiss_radial_pucker"):
                        host.register_client(
                            OverlayClientIdentity(client_id=f"{_SHELL_CLIENT_ID}.{suffix}", display_id=host.display_id, role="hud"),
                            window=None, content_view=None,
                        )
                    self._shell_sidecars_registered = True
                self._shell_final = config
                self._publish_shell_frame(time.perf_counter())
                success = True
                if self._shell_waiting and not self._shell_present_armed:
                    self._shell_present_armed = True
                    epoch = self._shell_epoch

                    def did_present():
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "opticalShellDidPresent:", {"host": host, "epoch": epoch}, False,
                        )

                    self._shell_client.set_on_first_present(did_present)
            else:
                success = host.update_client_config("spoke.teleporter", config)
            self._shell_registered = bool(success)
            if not success:
                self._shell_unavailable = True
                self._set_status("Native presentation; optical shell unavailable")
            logger.info("Teleporter House shell: registered=%s display=%s", success, host.display_id)
        except Exception:
            self._finish_shell()
            self._shell_unavailable = True
            self._set_status("Native presentation; optical shell unavailable")
            logger.exception("Teleporter optical publication failed; native controls remain usable")

    def opticalShellCaptureStateChanged_(self, payload: dict) -> None:
        if payload.get("host") is not self._shell_host or not self.visible:
            return
        state = payload.get("state")
        error = payload.get("error")
        status_label = getattr(self, "_status_label", None)
        old_status = status_label.stringValue() if status_label is not None else ""
        if state == "failed":
            self._finish_shell()
            self._shell_unavailable = True
            suffix = f" ({error})" if error else ""
            self._set_status(f"Native presentation; optical shell unavailable{suffix}")
        elif state == "pending":
            self._shell_unavailable = True
            self._set_status("Native presentation; optical shell starting")
        elif state == "started":
            self._shell_unavailable = False
            if old_status.startswith("Native presentation; optical shell"):
                self._set_status("")
        elif state == "cancelled":
            self._finish_shell()
            self._shell_unavailable = True
            self._set_status("Native presentation; optical shell stopped")

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

    def accepts_dictation(self) -> bool:
        if not self.visible:
            return False
        if self._activation_in_flight:
            return True
        app = NSApp()
        return bool(
            app is not None
            and app.isActive()
            and self._panel is not None
            and self._panel.isKeyWindow()
        )

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
