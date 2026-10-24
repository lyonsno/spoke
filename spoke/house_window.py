"""Native House choreography shared by Spoke utility windows.

Logical visibility and focus belong to the consumer; this base owns only the
capture admission, optical clients, native reveal mask and animation tail.
"""

import logging
import os
import time

from AppKit import NSScreen
from Foundation import NSObject, NSTimer, NSRunLoop
from Quartz import CAShapeLayer, CATransaction, CGPathCreateWithRoundedRect

logger = logging.getLogger(__name__)
_SHELL_START_GRACE_S = 0.25


class HouseWindowShell(NSObject):
    _shell_client_id = "spoke.teleporter"
    _shell_env = "SPOKE_TELEPORTER_OPTICAL_SHELL"

    def _init_window_shell(self, window, client_id):
        self._panel = window
        self._shell_client_id = client_id
        self._shell_env = "SPOKE_POPUP_OPTICAL_SHELL"
        self.visible = self._popup_visible = False
        self._shell_host = None
        self._shell_registered = False
        self._shell_sidecars_registered = False
        self._shell_transition = self._shell_final = self._shell_tick_timer = None
        self._shell_epoch = 0
        self._shell_waiting = False
        # Include native title-bar controls in the reveal instead of leaving
        # their chrome visible above an otherwise collapsed window.
        foreground = window.contentView().superview()
        foreground.setWantsLayer_(True)
        self._shell_foreground = foreground
        self._shell_body_layer = None
        self._shell_text_mask = CAShapeLayer.layer()
        foreground.layer().setMask_(self._shell_text_mask)
        window.setOpaque_(False)
        # Same native-above-compositor ordering as CommandOverlay. Remain below
        # native menus and attached sheets.
        window.setLevel_(26)
        self._set_shell_planes(dict(opacity=1.0, height_frac=1.0),
                               dict(alpha=1.0, width_frac=1.0, height_frac=1.0))

    def _shell_status(self, text):
        callback = getattr(self, "_set_status", None)
        if callback is not None:
            callback(text)
        elif text:
            logger.info("%s: %s", self._shell_client_id, text)

    def _show_window_shell(self):
        if getattr(self, "_panel", None) is None:
            return
        if not self.visible:
            self.visible = self._popup_visible = True
            self._panel.setIgnoresMouseEvents_(False)
            self._prepare_shell_entrance()
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.0, self, "publishOpticalShell:", None, False,
            )

    def _dismiss_window_shell(self, animate):
        if getattr(self, "_panel", None) is None:
            return False
        was_visible = self.visible
        self.visible = self._popup_visible = False
        transition = getattr(self, "_shell_transition", None)
        if (animate and was_visible and transition is not None
                and not self._shell_waiting and self._shell_registered
                and self._shell_final is not None):
            self._shell_epoch += 1
            transition.dismiss(time.perf_counter(), exposed_text=self._shell_applied_text)
            self._panel.setIgnoresMouseEvents_(True)
            self._panel.resignKeyWindow()
            self._start_shell_ticks()
            return True
        self._finish_shell()
        return False

    def windowDidResize_(self, notification):
        if getattr(self, "visible", False):
            self.publishOpticalShell_(None)

    def _prepare_shell_entrance(self):
        if (os.environ.get(self._shell_env, "1") == "0"
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
        ids = [self._shell_client_id]
        if getattr(self, "_shell_sidecars_registered", False):
            ids.extend(f"{self._shell_client_id}.{suffix}" for suffix in ("dismiss_seam", "dismiss_radial_pucker"))
        return ids

    def _set_shell_planes(self, body, text):
        foreground = getattr(self, "_shell_foreground", None)
        if foreground is None:
            return
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        try:
            body_layer = getattr(self, "_shell_body_layer", None)
            if body_layer is not None:
                body_layer.setValue_forKeyPath_(max(body["height_frac"], .001), "transform.scale.y")
                body_layer.setOpacity_(body["opacity"])
            size = foreground.bounds().size
            width, height = size.width * text["width_frac"], size.height * text["height_frac"]
            rect = ((size.width - width) / 2, (size.height - height) / 2), (width, height)
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
            logger.exception("House window transition failed; retaining native controls")
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
        if getattr(self, "visible", False):
            self.publishOpticalShell_(None)

    def publishOpticalShell_(self, timer):
        if not self.visible or os.environ.get(self._shell_env, "1") == "0":
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
                caller_id=self._shell_client_id, continuity_key=self._shell_client_id, bounds=bounds,
                role="hud", state="rest", visible=True, visibility_scope="independent",
                presentation=OpticalFieldPresentation(layer="hud", order=43),
                presentation_layer="hud", layout_recipe=self._shell_client_id + "-native-carrier",
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
                    client_id=self._shell_client_id,
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
                            OverlayClientIdentity(client_id=f"{self._shell_client_id}.{suffix}", display_id=host.display_id, role="hud"),
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
                success = host.update_client_config(self._shell_client_id, config)
            self._shell_registered = bool(success)
            if not success:
                self._shell_unavailable = True
                self._shell_status("Native presentation; optical shell unavailable")
            logger.info("%s House shell: registered=%s display=%s", self._shell_client_id, success, host.display_id)
        except Exception:
            self._finish_shell()
            self._shell_unavailable = True
            self._shell_status("Native presentation; optical shell unavailable")
            logger.exception("House window optical publication failed; native controls remain usable")

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
            self._shell_status(f"Native presentation; optical shell unavailable{suffix}")
        elif state == "pending":
            self._shell_unavailable = True
            self._shell_status("Native presentation; optical shell starting")
        elif state == "started":
            self._shell_unavailable = False
            if old_status.startswith("Native presentation; optical shell"):
                self._shell_status("")
        elif state == "cancelled":
            self._finish_shell()
            self._shell_unavailable = True
            self._shell_status("Native presentation; optical shell stopped")
