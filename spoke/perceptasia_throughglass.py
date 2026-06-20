"""Perceptasia Throughglass Graft.

This is the first clean Spoke-hosted consumer for Perceptasia as a stack
surface.  The provider is provisional: Spoke owns the window and optical
request contract, while the current Perceptasia localhost viewer is only the
first content source behind that contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
import numbers
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSPanel,
    NSScreen,
    NSView,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject

from .command_overlay_trace import record_command_overlay_trace
from .optical_field import OpticalFieldBounds
from .perceptasia_primitive_passport import (
    PERCEPTASIA_PRIMITIVE_CLIENT_ID,
    build_perceptasia_primitive_request as build_perceptasia_optical_request,
    compile_perceptasia_primitive_carrier_config as compile_perceptasia_shell_config,
)

logger = logging.getLogger(__name__)

_CLIENT_ID = PERCEPTASIA_PRIMITIVE_CLIENT_ID
_DEFAULT_URL = "http://localhost:8742"
_DEFAULT_WIDTH = 980.0
_DEFAULT_HEIGHT = 560.0
_MIN_MARGIN = 32.0
_DISCOVERY_PORTS = (8742, 8753, 8754, 8755, 8764, 8797, 8798, 8799, 8896)

_NSWindowStyleMaskClosable = 1 << 1
_NSWindowStyleMaskResizable = 1 << 3
_NSWindowStyleMaskUtilityWindow = 1 << 4
_THROUGHGLASS_SIBLING_WINDOW_LEVEL = 25
_THROUGHGLASS_PRIMITIVE_CARRIER_WINDOW_LEVEL = 23
_THROUGHGLASS_SHELL_PUBLISH_DELAY_SECONDS = 0.08
_THROUGHGLASS_SHELL_SETTLE_DELAY_SECONDS = 0.12
_THROUGHGLASS_SHELL_DISMISS_DELAY_SECONDS = 0.12
_THROUGHGLASS_SHELL_ANIMATION_FPS = 60.0
_THROUGHGLASS_LIVE_CARRIER_MARGIN_POINTS = 0.0
_NSViewWidthSizable = 1 << 1
_NSViewHeightSizable = 1 << 4
_THROUGHGLASS_UI_DELEGATE = None
_THROUGHGLASS_MEDIA_DECISION_BLOCK_SIGNATURE = b"v@?q"
_THROUGHGLASS_MEDIA_CAPTURE_SELECTOR = (
    b"webView:requestMediaCapturePermissionForOrigin:initiatedByFrame:type:decisionHandler:"
)
_THROUGHGLASS_MEDIA_CAPTURE_SIGNATURE = b"v@:@@@q@?"
_THROUGHGLASS_MEDIA_CAPTURE_METADATA = {
    "retval": {"type": b"v"},
    "arguments": {
        2: {"type": b"@"},
        3: {"type": b"@"},
        4: {"type": b"@"},
        5: {"type": b"q"},
        6: {
            "type": b"@?",
            "callable": {
                "retval": {"type": b"v"},
                "arguments": {
                    0: {"type": b"^v", "null_accepted": True},
                    1: {"type": b"q"},
                },
            },
            "callable_retained": False,
        },
    },
}
_THROUGHGLASS_CONTENT_PROBE_DEFAULT_MAX_ATTEMPTS = 120
_THROUGHGLASS_PRIMITIVE_CAPTURE_CSS = """
:root {
  --spoke-throughglass-radius: 42px;
}
html, body {
  overflow: hidden !important;
  background: #050708 !important;
  border-radius: var(--spoke-throughglass-radius) !important;
  clip-path: inset(0 round var(--spoke-throughglass-radius)) !important;
}
* {
  scrollbar-width: none !important;
}
*::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
body > * {
  border-radius: var(--spoke-throughglass-radius) !important;
  clip-path: inset(0 round var(--spoke-throughglass-radius)) !important;
}
canvas {
  border-radius: var(--spoke-throughglass-radius) !important;
  clip-path: inset(0 round var(--spoke-throughglass-radius)) !important;
}
"""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in {"", "0", "false", "False", "no", "off"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_positive_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def _throughglass_live_carrier_margin_points(width: float, height: float) -> float:
    requested = _env_positive_float(
        "SPOKE_PERCEPTASIA_THROUGHGLASS_LIVE_CARRIER_MARGIN_POINTS",
        _THROUGHGLASS_LIVE_CARRIER_MARGIN_POINTS,
    )
    return min(requested, max(0.0, min(float(width), float(height)) * 0.18))


def _throughglass_live_carrier_aperture(width: float, height: float) -> tuple[float, float, float]:
    margin = _throughglass_live_carrier_margin_points(width, height)
    return margin, max(1.0, float(width) - (2.0 * margin)), max(
        1.0,
        float(height) - (2.0 * margin),
    )


def _usable_selector_scheduler(obj):
    scheduler = getattr(obj, "performSelector_withObject_afterDelay_", None)
    if not callable(scheduler):
        return None
    module_name = getattr(type(scheduler), "__module__", "")
    if module_name.startswith("unittest.mock"):
        return None
    return scheduler


def _throughglass_window_level() -> int:
    return _THROUGHGLASS_SIBLING_WINDOW_LEVEL


def _objc_typed_selector(signature):
    decorator = getattr(objc, "typedSelector", None)
    if callable(decorator):
        return decorator(signature)

    def _decorate(function):
        setattr(function, "__objc_signature__", signature)
        return function

    return _decorate


class _ThroughglassUIDelegate(NSObject):
    """Keep embedded Perceptasia WebKit prompts out of the visual proof surface."""

    @_objc_typed_selector(_THROUGHGLASS_MEDIA_CAPTURE_SIGNATURE)
    def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
        self,
        _webview,
        _origin,
        _frame,
        media_type,
        decision_handler,
    ) -> None:
        try:
            from WebKit import WKPermissionDecisionGrant

            decision = WKPermissionDecisionGrant
        except Exception:
            decision = 1
        logger.info("Perceptasia Throughglass: granting WebKit media capture request type=%s", media_type)
        _decide_webkit_media_capture(decision_handler, decision)


def _register_throughglass_media_metadata() -> None:
    register = getattr(objc, "registerMetaDataForSelector", None)
    if callable(register):
        register(
            b"_ThroughglassUIDelegate",
            _THROUGHGLASS_MEDIA_CAPTURE_SELECTOR,
            _THROUGHGLASS_MEDIA_CAPTURE_METADATA,
        )
        return
    registry = getattr(objc, "_throughglass_metadata_registry", None)
    if registry is None:
        registry = {}
        setattr(objc, "_throughglass_metadata_registry", registry)

    registry[(_ThroughglassUIDelegate, _THROUGHGLASS_MEDIA_CAPTURE_SELECTOR)] = (
        _THROUGHGLASS_MEDIA_CAPTURE_METADATA
    )

    if not hasattr(objc, "_registeredMetadataForSelector"):
        def _registered_metadata_for_selector(class_ref, selector):
            return registry.get((class_ref, selector))

        setattr(objc, "_registeredMetadataForSelector", _registered_metadata_for_selector)


def _decide_webkit_media_capture(decision_handler, decision: int) -> None:
    """Call WebKit's media-capture completion handler even when PyObjC omitted its block signature."""

    try:
        decision_handler(decision)
        return
    except TypeError as exc:
        if "block without a signature" not in str(exc):
            raise
        logger.warning("Perceptasia Throughglass: media capture decision handler lacks signature; seating it")

    try:
        setattr(decision_handler, "__block_signature__", _THROUGHGLASS_MEDIA_DECISION_BLOCK_SIGNATURE)
        decision_handler(decision)
        return
    except Exception as exc:
        logger.error(
            "Perceptasia Throughglass: media capture decision handler could not be called after signature seating",
            exc_info=True,
        )
        raise exc


_register_throughglass_media_metadata()


def _throughglass_ui_delegate():
    global _THROUGHGLASS_UI_DELEGATE
    if _THROUGHGLASS_UI_DELEGATE is None:
        _THROUGHGLASS_UI_DELEGATE = _ThroughglassUIDelegate.alloc().init()
    return _THROUGHGLASS_UI_DELEGATE


@dataclass(frozen=True)
class PerceptasiaThroughglassManifest:
    """Spoke-owned provider contract for a Perceptasia stack surface."""

    schema: str = "spoke.perceptasia-throughglass.provider.v0"
    provider: str = "perceptasia"
    url: str = _DEFAULT_URL
    scene_url: str = f"{_DEFAULT_URL}/scene.json"
    selection_path: str = str(Path.home() / ".local" / "state" / "perceptasia" / "selection.json")

    @classmethod
    def from_env(cls) -> "PerceptasiaThroughglassManifest":
        requested_url = os.environ.get("SPOKE_PERCEPTASIA_THROUGHGLASS_URL", _DEFAULT_URL).rstrip("/")
        url = _resolve_provider_url(requested_url)
        return cls(
            url=url,
            scene_url=os.environ.get(
                "SPOKE_PERCEPTASIA_THROUGHGLASS_SCENE_URL",
                f"{url}/scene.json",
            ),
            selection_path=os.environ.get(
                "SPOKE_PERCEPTASIA_SELECTION_PATH",
                str(Path.home() / ".local" / "state" / "perceptasia" / "selection.json"),
            ),
        )


class PerceptasiaThroughglassGraft(NSObject):
    """Non-activating Spoke window carrying the Perceptasia viewer."""

    def initWithCompositorRegistry_(self, registry):
        self = objc.super(PerceptasiaThroughglassGraft, self).init()
        if self is None:
            return None
        self._registry = registry
        self._host = None
        self._panel = None
        self._content_view = None
        self._visible = False
        self._manifest = PerceptasiaThroughglassManifest.from_env()
        self._content_kind = "uninitialized"
        self._content_verified = False
        self._content_failure = None
        self._content_probe_attempts = 0
        self._content_generation = 0
        self._client_registered = False
        self._pending_show = False
        self._pending_shell_publish = False
        self._pending_shell_rest_publish = False
        self._pending_shell_hide_finish = False
        self._throughglass_shell_animation_direction = 0
        self._throughglass_shell_animation_started_at = 0.0
        self._throughglass_shell_animation_duration = 0.0
        self._carrier_content_width = 0.0
        self._carrier_content_height = 0.0
        self._carrier_clip_view = None
        self._assistant_overlay_parked = False
        self._visibility_callback = None
        return self

    def set_visibility_callback(self, callback) -> None:
        self._visibility_callback = callback

    def setup(self) -> None:
        if self._panel is not None:
            return
        self._content_generation += 1
        logger.info("Perceptasia Throughglass: setup begin url=%s", self._manifest.url)
        provider_reachable = _is_provider_reachable(self._manifest.url)
        if not provider_reachable:
            logger.warning(
                "Perceptasia Throughglass: provider unavailable url=%s",
                self._manifest.url,
            )
        screen = NSScreen.mainScreen()
        screen_frame = screen.visibleFrame() if screen is not None else NSMakeRect(0, 0, 1440, 900)
        x, y, width, height = _default_panel_rect(screen_frame)
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            _NSWindowStyleMaskClosable
            | _NSWindowStyleMaskResizable
            | _NSWindowStyleMaskUtilityWindow
            | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setTitle_("Perceptasia Throughglass Graft")
        # The compositor owns only the optical shell. The live WebView remains
        # an external sibling carrier, otherwise excluded capture makes it
        # invisible behind the fullscreen compositor at rest.
        panel.setLevel_(_throughglass_window_level())
        primitive_shell = _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL")
        # WKWebView/WebGL content is the load-bearing visible surface here.
        # Standalone sibling mode keeps an opaque carrier, while primitive
        # shell mode must not contribute a rectangular panel background around
        # the rounded captured content.
        panel.setOpaque_(not primitive_shell)
        panel.setHasShadow_(False)
        panel.setBackgroundColor_(
            NSColor.colorWithWhite_alpha_(0.0, 0.0 if primitive_shell else 1.0)
        )
        # Throughglass is a live Perceptasia viewer, so pointer input is
        # accepted by default. Click-through remains available for witness/debug
        # runs that only need a visual surface.
        panel.setIgnoresMouseEvents_(
            _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_CLICK_THROUGH")
        )
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setLevel_(_throughglass_window_level())

        content_result = (
            _make_content_view(self._manifest.url, width, height)
            if provider_reachable
            else _make_provider_unavailable_view(self._manifest.url, width, height)
        )
        if isinstance(content_result, tuple) and len(content_result) == 2:
            content, content_kind = content_result
        else:
            content = content_result
            content_kind = "unverified"
        self._content_kind = str(content_kind)
        self._content_verified = False
        self._content_failure = None
        content_root = panel.contentView()
        carrier_margin, carrier_width, carrier_height = _throughglass_live_carrier_aperture(
            width,
            height,
        )
        carrier = NSView.alloc().initWithFrame_(
            NSMakeRect(carrier_margin, carrier_margin, carrier_width, carrier_height)
        )
        _configure_content_carrier(
            content_root,
            carrier,
            content,
            carrier_width,
            carrier_height,
            x=carrier_margin,
            y=carrier_margin,
        )
        carrier.addSubview_(content)
        content_root.addSubview_(carrier)
        self._panel = panel
        self._content_view = content
        self._carrier_clip_view = carrier
        self._carrier_content_width = float(width)
        self._carrier_content_height = float(height)
        if self._content_kind == "webview":
            self.__schedule_content_probe(delay=0.25)
        else:
            self._content_failure = self._content_kind
        logger.info(
            "Perceptasia Throughglass: setup complete x=%.1f y=%.1f w=%.1f h=%.1f content_kind=%s",
            x,
            y,
            width,
            height,
            self._content_kind,
        )

    def show(self) -> bool:
        logger.info("Perceptasia Throughglass: show begin")
        if self._panel is None:
            self.setup()
        if self._panel is None:
            logger.warning("Perceptasia Throughglass: show aborted without panel")
            return False
        if self.__requires_verified_content() and not self._content_verified:
            if self.__should_publish_shell():
                return self.__show_shell_with_quarantined_content()
            self._pending_show = True
            logger.warning(
                "Perceptasia Throughglass: show deferred until content verifies kind=%s failure=%s",
                self._content_kind,
                self._content_failure,
            )
            return False
        return self.__show_verified()

    def __show_verified(self) -> bool:
        if self._panel is None:
            return False
        self.__reassert_live_carrier_window_level()
        was_visible = bool(self._visible)
        if self.__should_publish_shell():
            self.__set_live_carrier_window_exposure(False)
        self._panel.orderFrontRegardless()
        self._visible = True
        self._pending_show = False
        if self.__should_publish_shell():
            self.__schedule_shell_publish_after_carrier_present()
        else:
            logger.info(
                "Perceptasia Throughglass: shell publish skipped for live content carrier"
            )
        logger.info(
            "Perceptasia Throughglass: show complete content_kind=%s content_verified=%s",
            self._content_kind,
            self._content_verified,
        )
        if not was_visible:
            self.__notify_visibility_changed(True)
        return True

    def __show_shell_with_quarantined_content(self) -> bool:
        if self._panel is None:
            return False
        self.__reassert_live_carrier_window_level()
        was_visible = bool(self._visible)
        self.__set_live_carrier_window_exposure(False)
        self._panel.orderFrontRegardless()
        self._visible = True
        self._pending_show = True
        self.__schedule_shell_publish_after_carrier_present()
        logger.info(
            "Perceptasia Throughglass: shell show started before content proof kind=%s failure=%s",
            self._content_kind,
            self._content_failure,
        )
        if not was_visible:
            self.__notify_visibility_changed(True)
        return True

    def hide(self) -> None:
        was_visible = bool(self._visible)
        self._pending_show = False
        self._pending_shell_publish = False
        self._pending_shell_rest_publish = False
        self._visible = False
        self._assistant_overlay_parked = False
        if getattr(self, "_client_registered", False):
            self.__set_live_carrier_window_exposure(False)
            if self.__start_shell_animation(direction=-1):
                if was_visible:
                    self.__notify_visibility_changed(False)
                return
        self.__finish_hide_after_dismiss()
        if was_visible:
            self.__notify_visibility_changed(False)

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def isVisible(self) -> bool:
        return bool(getattr(self, "_visible", False))

    def has_live_carrier(self) -> bool:
        return bool(
            getattr(self, "_panel", None) is not None
            or getattr(self, "_content_view", None) is not None
            or getattr(self, "_client_registered", False)
        )

    def park_for_assistant_overlay(self) -> bool:
        """Temporarily remove the live carrier while assistant owns the screen."""
        if not self.has_live_carrier():
            return False
        panel = getattr(self, "_panel", None)
        self._pending_shell_publish = False
        self._pending_shell_rest_publish = False
        self._throughglass_shell_animation_direction = 0
        self._assistant_overlay_parked = True
        self.__set_live_carrier_human_visible(False)
        if panel is not None:
            panel.orderOut_(None)
        if getattr(self, "_client_registered", False):
            self.__publish_shell_state("hidden", visible=False)
        return True

    def restore_after_assistant_overlay(self) -> bool:
        """Restore a carrier parked for assistant display without reloading it."""
        if not bool(getattr(self, "_assistant_overlay_parked", False)):
            return False
        self._assistant_overlay_parked = False
        if not bool(getattr(self, "_visible", False)):
            return False
        panel = getattr(self, "_panel", None)
        if panel is None:
            return self.show()
        self.__reassert_live_carrier_window_level()
        self.__set_live_carrier_human_visible(True)
        panel.orderFrontRegardless()
        if self.__should_publish_shell():
            self.__schedule_shell_publish_after_carrier_present()
        return True

    def cleanup(self) -> None:
        self.hide()
        self._panel = None
        self._content_view = None

    def mark_content_verified_for_test(self, title: str = "Perceptasia 3D") -> None:
        self.__mark_content_verified({"title": title})

    def probeThroughglassContent_(self, _sender) -> None:
        self.__probe_content_ready()

    def publishThroughglassShellAfterCarrierPresent_(self, _sender) -> None:
        self.__publish_shell_after_carrier_present()

    def publishThroughglassShellRestAfterMaterialize_(self, _sender) -> None:
        self.__publish_shell_rest_after_materialize()

    def finishThroughglassHideAfterDismiss_(self, _sender) -> None:
        self.__finish_hide_after_dismiss(scheduled=True)

    def animateThroughglassShellStep_(self, _sender) -> None:
        self.__animate_shell_step()

    def __requires_verified_content(self) -> bool:
        return _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY") or _env_flag(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE"
        )

    def __should_publish_shell(self) -> bool:
        return _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL")

    def __shell_materialize_settle_delay_seconds(self) -> float:
        return _env_positive_float(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_SETTLE_DELAY_SECONDS",
            _THROUGHGLASS_SHELL_SETTLE_DELAY_SECONDS,
        )

    def __shell_dismiss_delay_seconds(self) -> float:
        return _env_positive_float(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_DISMISS_DELAY_SECONDS",
            _THROUGHGLASS_SHELL_DISMISS_DELAY_SECONDS,
        )

    def __schedule_shell_publish_after_carrier_present(self) -> None:
        self._pending_shell_publish = True
        delay = _env_positive_float(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_PUBLISH_DELAY_SECONDS",
            _THROUGHGLASS_SHELL_PUBLISH_DELAY_SECONDS,
        )
        scheduler = _usable_selector_scheduler(self)
        if callable(scheduler):
            scheduler("publishThroughglassShellAfterCarrierPresent:", None, delay)
            logger.info(
                "Perceptasia Throughglass: shell publish deferred until carrier-present tick delay=%.3f",
                delay,
            )
            return
        self.__publish_shell_after_carrier_present()

    def __schedule_shell_rest_after_materialize(self) -> bool:
        self._pending_shell_rest_publish = True
        delay = self.__shell_materialize_settle_delay_seconds()
        scheduler = _usable_selector_scheduler(self)
        if callable(scheduler):
            scheduler(
                "publishThroughglassShellRestAfterMaterialize:",
                None,
                delay,
            )
            logger.info(
                "Perceptasia Throughglass: shell rest publish deferred after materialize delay=%.3f",
                delay,
            )
            return True
        self.__publish_shell_rest_after_materialize()
        return False

    def __publish_shell_rest_after_materialize(self) -> None:
        if not bool(getattr(self, "_pending_shell_rest_publish", False)):
            return
        self._pending_shell_rest_publish = False
        if (
            not bool(getattr(self, "_visible", False))
            or self._panel is None
            or self._content_view is None
            or not self.__should_publish_shell()
        ):
            logger.info("Perceptasia Throughglass: rest shell publish skipped after state changed")
            return
        self.__publish_shell_rest_state()

    def __schedule_shell_hide_after_dismiss(self) -> bool:
        self._pending_shell_hide_finish = True
        delay = self.__shell_dismiss_delay_seconds()
        scheduler = _usable_selector_scheduler(self)
        if callable(scheduler):
            scheduler(
                "finishThroughglassHideAfterDismiss:",
                None,
                delay,
            )
            logger.info(
                "Perceptasia Throughglass: shell hide deferred after dismiss delay=%.3f",
                delay,
            )
            return True
        self.__finish_hide_after_dismiss()
        return False

    def __finish_hide_after_dismiss(self, *, scheduled: bool = False) -> None:
        if bool(getattr(self, "_visible", False)):
            self._pending_shell_hide_finish = False
            return
        if scheduled and not bool(getattr(self, "_pending_shell_hide_finish", False)):
            return
        self._pending_shell_hide_finish = False
        if getattr(self, "_client_registered", False):
            self.__publish_shell_state("hidden", visible=False)
        if self._panel is not None:
            self._panel.orderOut_(None)
        self.__release_shell_client()
        self.__teardown_content_carrier()

    def __publish_shell_after_carrier_present(self) -> None:
        if not bool(getattr(self, "_pending_shell_publish", False)):
            return
        self._pending_shell_publish = False
        if (
            not bool(getattr(self, "_visible", False))
            or self._panel is None
            or self._content_view is None
            or not self.__should_publish_shell()
        ):
            logger.info("Perceptasia Throughglass: deferred shell publish skipped after state changed")
            return
        self.__reassert_live_carrier_window_level()
        published = self.__start_shell_animation(direction=1)
        if published:
            # Keep the live carrier off the screen-capture source during the
            # transition. It becomes capture-present only when the shell reaches
            # rest, avoiding a full rectangular source plate inside the reveal.
            self.__set_live_carrier_window_exposure(False)

    def __schedule_content_probe(self, *, delay: float) -> None:
        scheduler = _usable_selector_scheduler(self)
        if callable(scheduler):
            try:
                scheduler("probeThroughglassContent:", None, delay)
            except Exception:
                logger.exception("Perceptasia Throughglass: content probe scheduling failed")
                self.__mark_content_failed("probe-scheduler-failed")
        else:
            logger.info("Perceptasia Throughglass: content probe scheduler unavailable")

    def __probe_content_ready(self) -> None:
        view = self._content_view
        generation = self._content_generation
        evaluator = getattr(view, "evaluateJavaScript_completionHandler_", None)
        if not callable(evaluator):
            self.__mark_content_failed("webview-evaluator-unavailable")
            return
        self._content_probe_attempts += 1
        script = (
            "(() => {"
            "const canvases = Array.from(document.querySelectorAll('canvas'));"
            "let sampledPixels = 0;"
            "let visualSignal = 0;"
            "for (const source of canvases) {"
            "const sw = source.width || source.clientWidth || 0;"
            "const sh = source.height || source.clientHeight || 0;"
            "if (sw < 8 || sh < 8) continue;"
            "const w = Math.min(64, Math.max(8, Math.floor(sw)));"
            "const h = Math.min(64, Math.max(8, Math.floor(sh)));"
            "const sample = document.createElement('canvas');"
            "sample.width = w; sample.height = h;"
            "const ctx = sample.getContext('2d', {willReadFrequently: true});"
            "if (!ctx) continue;"
            "try {"
            "ctx.drawImage(source, 0, 0, w, h);"
            "const data = ctx.getImageData(0, 0, w, h).data;"
            "let minL = 255, maxL = 0, chroma = 0, active = 0;"
            "for (let i = 0; i < data.length; i += 4) {"
            "const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];"
            "const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;"
            "minL = Math.min(minL, l);"
            "maxL = Math.max(maxL, l);"
            "chroma += (Math.max(r, g, b) - Math.min(r, g, b)) / 255;"
            "if (a > 8 && l > 8) active += 1;"
            "}"
            "sampledPixels += w * h;"
            "visualSignal = Math.max(visualSignal, (maxL - minL) / 255 + chroma / (w * h) + active / (w * h) * 0.25);"
            "} catch (e) {}"
            "}"
            "return {"
            "title: document.title || '',"
            "readyState: document.readyState || '',"
            "bodyText: (document.body && document.body.innerText || '').slice(0, 512),"
            "canvasCount: canvases.length,"
            "canvasSampledPixels: sampledPixels,"
            "canvasVisualSignal: visualSignal"
            "};"
            "})()"
        )

        def _completion(result, error):
            if generation != self._content_generation or view is not self._content_view:
                logger.info("Perceptasia Throughglass: stale content probe ignored")
                return
            if error is not None:
                self.__mark_content_failed(f"javascript-error:{error}")
                return
            if self.__content_probe_matches_perceptasia(result):
                self.__mark_content_verified(result)
                return
            max_attempts = _env_positive_int(
                "SPOKE_PERCEPTASIA_THROUGHGLASS_CONTENT_PROBE_ATTEMPTS",
                _THROUGHGLASS_CONTENT_PROBE_DEFAULT_MAX_ATTEMPTS,
            )
            if self._content_probe_attempts < max_attempts:
                self.__schedule_content_probe(delay=0.25)
                return
            self.__mark_content_failed(f"probe-mismatch:{result!r}")

        evaluator(script, _completion)

    def __content_probe_matches_perceptasia(self, result) -> bool:
        if not isinstance(result, Mapping):
            return False
        haystack = " ".join(
            str(result.get(key, ""))
            for key in ("title", "readyState", "bodyText")
        ).lower()
        body_text = str(result.get("bodyText", "")).lower()
        canvas_count = result.get("canvasCount", 0)
        try:
            canvas_count = int(canvas_count)
        except (TypeError, ValueError):
            canvas_count = 0
        sampled_pixels = result.get("canvasSampledPixels", 0)
        visual_signal = result.get("canvasVisualSignal", 0.0)
        try:
            sampled_pixels = int(sampled_pixels)
        except (TypeError, ValueError):
            sampled_pixels = 0
        try:
            visual_signal = float(visual_signal)
        except (TypeError, ValueError):
            visual_signal = 0.0
        has_perceptasia_identity = "perceptasia" in haystack
        canvas_proves_content = (
            canvas_count >= 1
            and sampled_pixels >= 64
            and visual_signal >= 0.015
        )
        dom_markers = (
            "start hand control",
            "native stream",
            "frame ",
            "authority",
            "orbit",
            "spring",
            "witness",
            "reticule",
            "command",
            "structure",
            "visibility",
            "labels",
            "edges",
            "atmosphere",
            "context",
            "physics",
            "works_on",
            "belongs_to",
            "attracted_to",
        )
        live_dom_marker_count = sum(1 for marker in dom_markers if marker in body_text)
        dom_proves_content = (
            canvas_count >= 1
            and sampled_pixels >= 64
            and live_dom_marker_count >= 4
            and (not self.__should_publish_shell() or canvas_proves_content)
        )
        return has_perceptasia_identity and (canvas_proves_content or dom_proves_content)

    def __mark_content_verified(self, result) -> None:
        self._content_verified = True
        self._content_failure = None
        result_title = result.get("title") if isinstance(result, Mapping) else result
        canvas_count = result.get("canvasCount") if isinstance(result, Mapping) else None
        sampled_pixels = result.get("canvasSampledPixels") if isinstance(result, Mapping) else None
        visual_signal = result.get("canvasVisualSignal") if isinstance(result, Mapping) else None
        logger.info(
            "Perceptasia Throughglass: content verified title=%r canvas_count=%s canvas_sampled_pixels=%s canvas_signal=%s",
            result_title,
            canvas_count,
            sampled_pixels,
            visual_signal,
        )
        if self._pending_show:
            if self.__should_publish_shell() and bool(getattr(self, "_visible", False)):
                self._pending_show = False
                self.__reassert_live_carrier_window_level()
                return
            self.__show_verified()

    def __mark_content_failed(self, reason: str) -> None:
        self._content_verified = False
        self._content_failure = reason
        logger.warning("Perceptasia Throughglass: content verification failed reason=%s", reason)

    def __release_shell_client(self) -> None:
        if self._host is None or not getattr(self, "_client_registered", False):
            return
        release = getattr(self._host, "release_client", None)
        if callable(release):
            release(_CLIENT_ID)
        self._client_registered = False

    def __notify_visibility_changed(self, visible: bool) -> None:
        callback = getattr(self, "_visibility_callback", None)
        if not callable(callback):
            return
        try:
            callback(bool(visible))
        except Exception:
            logger.debug("Perceptasia Throughglass: visibility callback failed", exc_info=True)

    def __reassert_live_carrier_window_level(self) -> None:
        panel = self._panel
        if panel is None:
            return
        set_level = getattr(panel, "setLevel_", None)
        if callable(set_level):
            set_level(_throughglass_window_level())

    def __publish_shell_rest_state(self) -> bool:
        self.__reassert_live_carrier_window_level()
        self.__apply_live_carrier_shell_phase("rest", 1.0)
        self.__set_live_carrier_window_exposure(True)
        panel = self._panel
        order_front = getattr(panel, "orderFrontRegardless", None) if panel is not None else None
        if callable(order_front):
            order_front()
        published = self.__publish_shell_state("rest")
        if not published:
            self.__set_live_carrier_window_exposure(False)
        return published

    def __set_live_carrier_human_visible(self, visible: bool) -> None:
        panel = self._panel
        if panel is None:
            return
        self.__set_live_carrier_window_exposure(visible)
        content_root_getter = getattr(panel, "contentView", None)
        content_root = content_root_getter() if callable(content_root_getter) else None
        for view in (content_root, self._content_view):
            set_hidden = getattr(view, "setHidden_", None)
            if callable(set_hidden):
                set_hidden(not visible)

    def __set_live_carrier_window_exposure(self, visible: bool) -> None:
        panel = self._panel
        if panel is None:
            return
        set_alpha = getattr(panel, "setAlphaValue_", None)
        if callable(set_alpha):
            set_alpha(1.0 if visible else 0.0)
        set_ignores_mouse = getattr(panel, "setIgnoresMouseEvents_", None)
        if callable(set_ignores_mouse):
            set_ignores_mouse(
                _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_CLICK_THROUGH")
                if visible
                else True
            )

    def __apply_live_carrier_shell_phase(self, state: str, progress: float | None) -> None:
        content = self._content_view
        carrier = getattr(self, "_carrier_clip_view", None)
        if content is None or carrier is None:
            return
        final_width, final_height = self.__live_carrier_content_size()
        if final_width <= 0.0 or final_height <= 0.0:
            return
        carrier_margin, carrier_width, carrier_height = _throughglass_live_carrier_aperture(
            final_width,
            final_height,
        )
        if state == "rest":
            target_width = carrier_width
            target_height = carrier_height
        else:
            config = compile_perceptasia_shell_config(
                OpticalFieldBounds(0.0, 0.0, carrier_width, carrier_height),
                state=state,
                visible=True,
                materialization_progress=progress,
            )
            target_width = min(
                carrier_width,
                max(1.0, float(config.get("content_width_points", carrier_width))),
            )
            target_height = min(
                carrier_height,
                max(1.0, float(config.get("content_height_points", carrier_height))),
            )
        origin_x = carrier_margin + ((carrier_width - target_width) * 0.5)
        origin_y = carrier_margin + ((carrier_height - target_height) * 0.5)
        carrier_setter = getattr(carrier, "setFrame_", None)
        if callable(carrier_setter):
            carrier_setter(NSMakeRect(origin_x, origin_y, target_width, target_height))
        # Keep the WKWebView/WebGL surface at its final size so WebKit's canvas
        # viewport does not collapse to the seed slit and leave a black backing
        # plate at rest. The carrier view supplies the animated clipping aperture.
        content_setter = getattr(content, "setFrame_", None)
        if callable(content_setter):
            content_setter(
                NSMakeRect(
                    -(origin_x - carrier_margin),
                    -(origin_y - carrier_margin),
                    carrier_width,
                    carrier_height,
                )
            )
        radius = _throughglass_carrier_corner_radius(target_width, target_height)
        layer_getter = getattr(carrier, "layer", None)
        layer = layer_getter() if callable(layer_getter) else None
        _shape_throughglass_carrier_layer(layer, radius=radius)
        for view in (carrier, content):
            display_setter = getattr(view, "setNeedsDisplay_", None)
            if callable(display_setter):
                display_setter(True)
            layout_setter = getattr(view, "setNeedsLayout_", None)
            if callable(layout_setter):
                layout_setter(True)
        self.__set_live_carrier_phase_exposure(state, progress)

    def __live_carrier_content_size(self) -> tuple[float, float]:
        width = float(getattr(self, "_carrier_content_width", 0.0) or 0.0)
        height = float(getattr(self, "_carrier_content_height", 0.0) or 0.0)
        root = None
        panel = self._panel
        content_root_getter = getattr(panel, "contentView", None) if panel is not None else None
        if callable(content_root_getter):
            root = content_root_getter()
        frame_getter = getattr(root, "frame", None)
        if callable(frame_getter):
            try:
                _x, _y, frame_width, frame_height = _rect_numbers(frame_getter())
            except Exception:
                frame_width = frame_height = 0.0
            if frame_width > 0.0 and frame_height > 0.0:
                width = frame_width
                height = frame_height
        return width, height

    def __set_live_carrier_phase_exposure(self, state: str, progress: float | None) -> None:
        panel = self._panel
        if panel is None:
            return
        if state == "rest":
            alpha = 1.0
            accepts_mouse = not _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_CLICK_THROUGH")
        else:
            p = _clamp01_float(0.0 if progress is None else float(progress))
            if state == "materialize":
                alpha = _clamp01_float((p - 0.12) / 0.50)
            else:
                alpha = _clamp01_float((p - 0.08) / 0.32)
            accepts_mouse = False
        set_alpha = getattr(panel, "setAlphaValue_", None)
        if callable(set_alpha):
            set_alpha(alpha)
        set_ignores_mouse = getattr(panel, "setIgnoresMouseEvents_", None)
        if callable(set_ignores_mouse):
            set_ignores_mouse(not accepts_mouse)

    def __teardown_content_carrier(self) -> None:
        content = self._content_view
        panel = self._panel
        carrier = getattr(self, "_carrier_clip_view", None)
        if content is not None:
            stop = getattr(content, "stopLoading", None)
            if callable(stop):
                stop()
            load_blank = getattr(content, "loadHTMLString_baseURL_", None)
            if callable(load_blank):
                load_blank("", None)
            remove = getattr(content, "removeFromSuperview", None)
            if callable(remove):
                remove()
        if carrier is not None:
            remove_carrier = getattr(carrier, "removeFromSuperview", None)
            if callable(remove_carrier):
                remove_carrier()
        self._content_generation += 1
        self._panel = None
        self._content_view = None
        self._carrier_clip_view = None
        self._content_kind = "uninitialized"
        self._content_verified = False
        self._content_failure = None
        self._carrier_content_width = 0.0
        self._carrier_content_height = 0.0
        self._content_probe_attempts = 0
        if panel is not None:
            logger.info("Perceptasia Throughglass: content carrier torn down")

    def _bounds(self) -> OpticalFieldBounds:
        bounds, _metadata = self.__bounds_and_coordinate_metadata()
        return bounds

    def __bounds_and_coordinate_metadata(self) -> tuple[OpticalFieldBounds, dict[str, object]]:
        if self._panel is None:
            return OpticalFieldBounds(0.0, 0.0, _DEFAULT_WIDTH, _DEFAULT_HEIGHT), {
                "source_coordinate_space": "display_local",
            }
        content_root_getter = getattr(self._panel, "contentView", None)
        content_root = content_root_getter() if callable(content_root_getter) else None
        frame = _content_view_screen_frame(self._panel, content_root)
        source_rect_basis = "content_root" if frame is not None else "panel_frame"
        if frame is None:
            frame = self._panel.frame()
        screen = NSScreen.mainScreen()
        bounds, metadata = _display_local_scaled_window_bounds(frame, screen)
        metadata["source_rect_basis"] = source_rect_basis
        return bounds, metadata

    def __shell_bounds_and_coordinate_metadata(
        self,
        state: str,
    ) -> tuple[OpticalFieldBounds, dict[str, object]]:
        if state in {"materialize", "dismiss"} and self._panel is not None:
            carrier = getattr(self, "_carrier_clip_view", None)
            frame = _content_view_screen_frame(
                self._panel,
                carrier,
                fallback_to_content_root=False,
            )
            if frame is not None:
                screen = NSScreen.mainScreen()
                bounds, metadata = _display_local_scaled_window_bounds(frame, screen)
                metadata["source_rect_basis"] = "carrier_clip"
                return bounds, metadata
        return self.__bounds_and_coordinate_metadata()

    def __animation_frame_delay_seconds(self) -> float:
        return 1.0 / _env_positive_float(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_ANIMATION_FPS",
            _THROUGHGLASS_SHELL_ANIMATION_FPS,
        )

    def __schedule_shell_animation_step(self) -> bool:
        scheduler = _usable_selector_scheduler(self)
        delay = self.__animation_frame_delay_seconds()
        if callable(scheduler):
            scheduler("animateThroughglassShellStep:", None, delay)
            return True
        self.__finish_shell_animation_immediately()
        return False

    def __finish_shell_animation_immediately(self) -> None:
        direction = int(getattr(self, "_throughglass_shell_animation_direction", 0))
        if direction == 0:
            return
        self._throughglass_shell_animation_direction = 0
        self._throughglass_shell_animation_duration = 0.0
        if direction > 0:
            self.__publish_shell_state("rest")
        else:
            self.__finish_hide_after_dismiss()

    def __start_shell_animation(self, *, direction: int) -> bool:
        self._pending_shell_rest_publish = False
        self._pending_shell_hide_finish = direction < 0
        self._throughglass_shell_animation_direction = 1 if direction >= 0 else -1
        self._throughglass_shell_animation_duration = (
            self.__shell_materialize_settle_delay_seconds()
            if self._throughglass_shell_animation_direction > 0
            else self.__shell_dismiss_delay_seconds()
        )
        self._throughglass_shell_animation_started_at = time.perf_counter()
        progress = 0.0 if self._throughglass_shell_animation_direction > 0 else 1.0
        state = "materialize" if self._throughglass_shell_animation_direction > 0 else "dismiss"
        self.__apply_live_carrier_shell_phase(state, progress)
        published = self.__publish_shell_state(
            state,
            visible=True,
            materialization_progress=progress,
        )
        if published:
            self.__schedule_shell_animation_step()
        return published

    def __animate_shell_step(self) -> None:
        direction = int(getattr(self, "_throughglass_shell_animation_direction", 0))
        if direction == 0:
            return
        duration = max(float(getattr(self, "_throughglass_shell_animation_duration", 0.0)), 1e-6)
        elapsed = max(time.perf_counter() - float(getattr(self, "_throughglass_shell_animation_started_at", 0.0)), 0.0)
        raw = min(elapsed / duration, 1.0)
        progress = raw if direction > 0 else 1.0 - raw
        state = "materialize" if direction > 0 else "dismiss"
        if raw >= 1.0:
            self._throughglass_shell_animation_direction = 0
            self._throughglass_shell_animation_duration = 0.0
            if direction > 0:
                self.__publish_shell_rest_state()
            else:
                self.__finish_hide_after_dismiss()
            return
        self.__apply_live_carrier_shell_phase(state, progress)
        self.__publish_shell_state(
            state,
            visible=True,
            materialization_progress=progress,
        )
        self.__schedule_shell_animation_step()

    def __publish_shell_state(
        self,
        state: str,
        *,
        visible: bool = True,
        materialization_progress: float | None = None,
    ) -> bool:
        if self._registry is None or self._panel is None or self._content_view is None:
            logger.info("Perceptasia Throughglass: publish skipped state=%s", state)
            return False
        if self._host is None:
            host_for_screen = getattr(self._registry, "host_for_screen", None)
            if not callable(host_for_screen):
                logger.info("Perceptasia Throughglass: registry has no host_for_screen")
                return False
            self._host = host_for_screen(NSScreen.mainScreen())
        bounds, coordinate_metadata = self.__shell_bounds_and_coordinate_metadata(state)
        config = compile_perceptasia_shell_config(
            bounds,
            state=state,
            visible=visible,
            materialization_progress=materialization_progress,
        )
        if (
            state in {"materialize", "dismiss"}
            and coordinate_metadata.get("source_rect_basis") == "carrier_clip"
        ):
            _pin_shell_geometry_to_bounds(config, bounds)
        _annotate_shell_coordinate_metadata(config, coordinate_metadata)
        if not getattr(self, "_client_registered", False):
            carrier_view = getattr(self, "_carrier_clip_view", None) or self._content_view
            added = self._host.add_client(_CLIENT_ID, self._panel, carrier_view, config)
            self._client_registered = bool(added)
            logger.info(
                "Perceptasia Throughglass: publish state=%s registered=%s",
                state,
                self._client_registered,
            )
            record_command_overlay_trace(
                f"throughglass.publish.{state}",
                visible=visible,
                materialization_progress=materialization_progress,
                updated=bool(added),
                registered=self._client_registered,
                carrier=config.get("throughglass_content_carrier"),
                include_carrier_window_in_capture=bool(
                    config.get("include_carrier_window_in_capture", False)
                ),
                clip_captured_carrier_to_shell=bool(
                    config.get("clip_captured_carrier_to_shell", False)
                ),
                x=bounds.x,
                y=bounds.y,
                width=bounds.width,
                height=bounds.height,
            )
            return bool(added)
        updated = bool(self._host.update_client_config(_CLIENT_ID, config))
        logger.info("Perceptasia Throughglass: publish state=%s updated=%s", state, updated)
        record_command_overlay_trace(
            f"throughglass.publish.{state}",
            visible=visible,
            materialization_progress=materialization_progress,
            updated=updated,
            registered=True,
            carrier=config.get("throughglass_content_carrier"),
            include_carrier_window_in_capture=bool(
                config.get("include_carrier_window_in_capture", False)
            ),
            clip_captured_carrier_to_shell=bool(
                config.get("clip_captured_carrier_to_shell", False)
            ),
            x=bounds.x,
            y=bounds.y,
            width=bounds.width,
            height=bounds.height,
        )
        return updated


def _default_panel_rect(frame) -> tuple[float, float, float, float]:
    width = min(_DEFAULT_WIDTH, max(480.0, float(frame.size.width) - 2 * _MIN_MARGIN))
    height = min(_DEFAULT_HEIGHT, max(320.0, float(frame.size.height) - 2 * _MIN_MARGIN))
    x = float(frame.origin.x) + (float(frame.size.width) - width) * 0.5
    y = float(frame.origin.y) + max(
        _MIN_MARGIN,
        float(frame.size.height) - height - _MIN_MARGIN,
    )
    return x, y, width, height


def _real_attr(value, attr: str, default: float = 0.0) -> float:
    candidate = getattr(value, attr, default)
    if isinstance(candidate, numbers.Real):
        return float(candidate)
    return float(default)


def _clamp01_float(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _rect_numbers(rect) -> tuple[float, float, float, float]:
    origin = getattr(rect, "origin", None)
    size = getattr(rect, "size", None)
    return (
        _real_attr(origin, "x"),
        _real_attr(origin, "y"),
        _real_attr(size, "width", _DEFAULT_WIDTH),
        _real_attr(size, "height", _DEFAULT_HEIGHT),
    )


def _rect_has_positive_size(rect) -> bool:
    size = getattr(rect, "size", None)
    width = getattr(size, "width", None)
    height = getattr(size, "height", None)
    return (
        isinstance(width, numbers.Real)
        and isinstance(height, numbers.Real)
        and float(width) > 0.0
        and float(height) > 0.0
    )


def _content_view_screen_frame(panel, content_view, *, fallback_to_content_root: bool = True):
    """Return the live payload rect in screen points when AppKit can prove it."""

    if panel is None or content_view is None:
        return None
    convert_to_screen = getattr(panel, "convertRectToScreen_", None)
    if not callable(convert_to_screen):
        return None
    candidate_views = [content_view]
    content_root_getter = getattr(panel, "contentView", None)
    content_root = content_root_getter() if callable(content_root_getter) else None
    if fallback_to_content_root and content_root is not None and content_root is not content_view:
        candidate_views.append(content_root)
    for view in candidate_views:
        bounds_getter = getattr(view, "bounds", None)
        convert_to_window = getattr(view, "convertRect_toView_", None)
        if not callable(bounds_getter) or not callable(convert_to_window):
            continue
        try:
            bounds = bounds_getter()
            if not _rect_has_positive_size(bounds):
                continue
            window_rect = convert_to_window(bounds, None)
            if not _rect_has_positive_size(window_rect):
                continue
            screen_rect = convert_to_screen(window_rect)
        except Exception:
            logger.debug("Perceptasia Throughglass: content-view bounds conversion failed", exc_info=True)
            continue
        if _rect_has_positive_size(screen_rect):
            return screen_rect
    frame_getter = getattr(content_view, "frame", None)
    if callable(frame_getter):
        try:
            frame = frame_getter()
            if _rect_has_positive_size(frame):
                screen_rect = convert_to_screen(frame)
                if _rect_has_positive_size(screen_rect):
                    return screen_rect
        except Exception:
            logger.debug("Perceptasia Throughglass: content-view frame conversion failed", exc_info=True)
    return None


def _screen_frame_for_bounds(screen):
    if screen is None:
        return NSMakeRect(0, 0, 1440, 900)
    for selector in ("frame", "visibleFrame"):
        getter = getattr(screen, selector, None)
        if not callable(getter):
            continue
        try:
            frame = getter()
        except Exception:
            continue
        if not _rect_has_positive_size(frame):
            continue
        _x, _y, width, height = _rect_numbers(frame)
        if width > 0.0 and height > 0.0:
            return frame
    return NSMakeRect(0, 0, 1440, 900)


def _screen_backing_scale(screen) -> float:
    getter = getattr(screen, "backingScaleFactor", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
        if isinstance(value, numbers.Real) and value > 0.0:
            return float(value)
    return 2.0


def _display_local_scaled_window_bounds(
    window_frame,
    screen,
) -> tuple[OpticalFieldBounds, dict[str, object]]:
    window_x, window_y, window_width, window_height = _rect_numbers(window_frame)
    screen_frame = _screen_frame_for_bounds(screen)
    screen_x, screen_y, _screen_width, screen_height = _rect_numbers(screen_frame)
    scale = _screen_backing_scale(screen)
    display_x = window_x - screen_x
    display_y_top = screen_y + screen_height - (window_y + window_height)
    bounds = OpticalFieldBounds(
        x=display_x * scale,
        y=display_y_top * scale,
        width=window_width * scale,
        height=window_height * scale,
    )
    return bounds, {
        "source_coordinate_space": "screen_points",
        "normalized_coordinate_space": "display_local_backing_pixels",
        "backing_scale": scale,
        "display_origin": (screen_x, screen_y),
    }


def _annotate_shell_coordinate_metadata(config: dict[str, object], metadata: dict[str, object]) -> None:
    optical_field = config.get("optical_field")
    if isinstance(optical_field, dict):
        optical_field.update(metadata)


def _pin_shell_geometry_to_bounds(config: dict[str, object], bounds: OpticalFieldBounds) -> None:
    """Keep the compositor shell on an already-animated carrier aperture."""

    config["content_width_points"] = float(bounds.width)
    config["content_height_points"] = float(bounds.height)
    config["center_x"] = float(bounds.center_x)
    config["center_y"] = float(bounds.center_y)
    config["gpu_material_base_width_points"] = float(bounds.width)
    config["gpu_material_base_height_points"] = float(bounds.height)
    if isinstance(config.get("optical_field"), dict):
        optical_field = dict(config["optical_field"])
        optical_field["bounds"] = bounds.to_payload()
        optical_field["content_frame"] = bounds.to_payload()
        config["optical_field"] = optical_field


def _discovery_ports() -> tuple[int, ...]:
    raw = os.environ.get("SPOKE_PERCEPTASIA_THROUGHGLASS_DISCOVERY_PORTS", "")
    if not raw.strip():
        return _DISCOVERY_PORTS
    ports: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ports.append(int(item))
        except ValueError:
            logger.warning("Perceptasia Throughglass: ignoring invalid discovery port %r", item)
    return tuple(ports) or _DISCOVERY_PORTS


def _candidate_provider_urls(requested_url: str) -> tuple[str, ...]:
    candidates = [requested_url.rstrip("/"), _DEFAULT_URL]
    candidates.extend(f"http://localhost:{port}" for port in _discovery_ports())
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return tuple(unique)


def _is_provider_reachable(url: str, *, timeout: float = 0.35) -> bool:
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            if not 200 <= status < 400:
                return False
            read = getattr(response, "read", None)
            if not callable(read):
                return True
            body = read(65536)
            if not body:
                return True
            marker = body.decode("utf-8", errors="ignore").lower()
            return "perceptasia" in marker or "scene.json" in marker
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _resolve_provider_url(requested_url: str) -> str:
    for candidate in _candidate_provider_urls(requested_url):
        if _is_provider_reachable(candidate):
            if candidate != requested_url:
                logger.info(
                    "Perceptasia Throughglass: resolved provider %s from requested %s",
                    candidate,
                    requested_url,
                )
            return candidate
    return requested_url


def _make_content_view(url: str, width: float, height: float):
    try:
        from Foundation import NSURL, NSURLRequest
        from WebKit import WKWebView

        logger.info("Perceptasia Throughglass: creating WKWebView")
        rect = NSMakeRect(0, 0, width, height)
        webview_alloc = WKWebView.alloc()
        configuration = _make_webview_configuration()
        configured_init = getattr(webview_alloc, "initWithFrame_configuration_", None)
        if configuration is not None and callable(configured_init):
            view = configured_init(rect, configuration)
        else:
            view = webview_alloc.initWithFrame_(rect)
        # Do not install the Python media UIDelegate by default. WebKit's
        # camera permission completion block can arrive without a PyObjC method
        # signature, and failing to call it synchronously aborts the process.
        # The live graft should leave getUserMedia on WebKit's native path.
        if _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_MEDIA_DELEGATE") and hasattr(
            view, "setUIDelegate_"
        ):
            view.setUIDelegate_(_throughglass_ui_delegate())
        _set_view_autoresizing(view)
        request = NSURLRequest.requestWithURL_(NSURL.URLWithString_(url))
        view.loadRequest_(request)
        logger.info("Perceptasia Throughglass: WKWebView request loaded")
        return view, "webview"
    except Exception:
        logger.warning("Perceptasia Throughglass: WKWebView unavailable, using fallback", exc_info=True)
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        label.setStringValue_(f"Perceptasia provider: {url}")
        label.setBezeled_(False)
        label.setDrawsBackground_(True)
        label.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.08, 0.88))
        label.setTextColor_(NSColor.colorWithWhite_alpha_(0.86, 1.0))
        label.setEditable_(False)
        label.setSelectable_(True)
        _set_view_autoresizing(label)
        return label, "webkit-fallback"


def _make_webview_configuration():
    try:
        from WebKit import WKWebViewConfiguration

        configuration = WKWebViewConfiguration.alloc().init()
        if _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL"):
            _install_primitive_capture_styles(configuration)
        return configuration
    except Exception:
        logger.warning(
            "Perceptasia Throughglass: WKWebView configuration unavailable",
            exc_info=True,
        )
        return None


def _install_primitive_capture_styles(configuration) -> None:
    try:
        from WebKit import (
            WKUserContentController,
            WKUserScript,
            WKUserScriptInjectionTimeAtDocumentStart,
        )
    except Exception:
        logger.warning(
            "Perceptasia Throughglass: primitive capture stylesheet unavailable",
            exc_info=True,
        )
        return
    controller = WKUserContentController.alloc().init()
    source = (
        "(() => {"
        f"const css = {json.dumps(_THROUGHGLASS_PRIMITIVE_CAPTURE_CSS)};"
        "const style = document.createElement('style');"
        "style.dataset.spokeThroughglassPrimitiveCapture = 'true';"
        "style.textContent = css;"
        "(document.head || document.documentElement).appendChild(style);"
        "})();"
    )
    script = WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
        source,
        WKUserScriptInjectionTimeAtDocumentStart,
        True,
    )
    add_script = getattr(controller, "addUserScript_", None)
    set_controller = getattr(configuration, "setUserContentController_", None)
    if callable(add_script) and callable(set_controller):
        add_script(script)
        set_controller(controller)


def _make_provider_unavailable_view(url: str, width: float, height: float):
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    label.setStringValue_(f"Perceptasia provider unavailable: {url}")
    label.setBezeled_(False)
    label.setDrawsBackground_(True)
    label.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.08, 0.88))
    label.setTextColor_(NSColor.colorWithWhite_alpha_(0.86, 1.0))
    label.setEditable_(False)
    label.setSelectable_(True)
    _set_view_autoresizing(label)
    return label, "provider-unavailable"


def _set_view_autoresizing(view) -> None:
    setter = getattr(view, "setAutoresizingMask_", None)
    if callable(setter):
        setter(_NSViewWidthSizable | _NSViewHeightSizable)


def _throughglass_carrier_corner_radius(width: float, height: float) -> float:
    bounds = OpticalFieldBounds(0.0, 0.0, float(width), float(height))
    config = compile_perceptasia_shell_config(bounds, state="rest")
    return float(config.get("corner_radius_points", min(float(width), float(height)) * 0.25))


def _shape_throughglass_carrier_layer(layer, *, radius: float, background_alpha: float | None = None) -> None:
    if layer is None:
        return
    masks_setter = getattr(layer, "setMasksToBounds_", None)
    if callable(masks_setter):
        masks_setter(True)
    radius_setter = getattr(layer, "setCornerRadius_", None)
    if callable(radius_setter):
        radius_setter(float(radius))
    if background_alpha is not None:
        background_setter = getattr(layer, "setBackgroundColor_", None)
        cg_color_getter = getattr(
            NSColor.colorWithWhite_alpha_(0.0, float(background_alpha)), "CGColor", None
        )
        if callable(background_setter) and callable(cg_color_getter):
            background_setter(cg_color_getter())


def _configure_content_carrier(
    content_root,
    carrier,
    content,
    width: float,
    height: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
) -> None:
    carrier_frame_setter = getattr(carrier, "setFrame_", None)
    if callable(carrier_frame_setter):
        carrier_frame_setter(NSMakeRect(x, y, width, height))
    content_frame_setter = getattr(content, "setFrame_", None)
    if callable(content_frame_setter):
        content_frame_setter(NSMakeRect(0, 0, width, height))
    _set_view_autoresizing(carrier)
    primitive_shell = _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL")
    corner_radius = _throughglass_carrier_corner_radius(width, height)
    root_layer_setter = getattr(content_root, "setWantsLayer_", None)
    if callable(root_layer_setter):
        root_layer_setter(True)
    root_layer_getter = getattr(content_root, "layer", None)
    root_layer = root_layer_getter() if callable(root_layer_getter) else None
    _shape_throughglass_carrier_layer(
        root_layer,
        radius=corner_radius,
        background_alpha=0.0 if primitive_shell else 1.0,
    )
    carrier_layer_setter = getattr(carrier, "setWantsLayer_", None)
    if callable(carrier_layer_setter):
        carrier_layer_setter(True)
    carrier_layer_getter = getattr(carrier, "layer", None)
    carrier_layer = carrier_layer_getter() if callable(carrier_layer_getter) else None
    _shape_throughglass_carrier_layer(carrier_layer, radius=corner_radius)
    content_layer_setter = getattr(content, "setWantsLayer_", None)
    if callable(content_layer_setter):
        content_layer_setter(True)
    content_layer_getter = getattr(content, "layer", None)
    content_layer = content_layer_getter() if callable(content_layer_getter) else None
    _shape_throughglass_carrier_layer(content_layer, radius=corner_radius)
