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
import urllib.error
import urllib.request

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSPanel,
    NSScreen,
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
_THROUGHGLASS_PRIMITIVE_CAPTURE_CSS = """
html, body {
  overflow: hidden !important;
  background: transparent !important;
}
* {
  scrollbar-width: none !important;
}
*::-webkit-scrollbar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}
"""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in {"", "0", "false", "False", "no", "off"}


def _throughglass_window_level() -> int:
    if _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL"):
        return _THROUGHGLASS_PRIMITIVE_CARRIER_WINDOW_LEVEL
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
        # In primitive-shell mode the WebView must sit under the compositor so
        # the optical field captures its pixels instead of excluding them. When
        # shell publication is off, keep the WebView as a normal sibling panel.
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
        _configure_content_carrier(panel.contentView(), content, width, height)
        panel.contentView().addSubview_(content)
        self._panel = panel
        self._content_view = content
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
        self._panel.orderFrontRegardless()
        self._visible = True
        self._pending_show = False
        if self.__should_publish_shell():
            self.__publish_shell_state("materialize")
            self.__publish_shell_state("rest")
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

    def hide(self) -> None:
        was_visible = bool(self._visible)
        self._pending_show = False
        self._visible = False
        self._assistant_overlay_parked = False
        if getattr(self, "_client_registered", False):
            self.__publish_shell_state("dismiss")
            self.__publish_shell_state("hidden", visible=False)
        if self._panel is not None:
            self._panel.orderOut_(None)
        self.__release_shell_client()
        self.__teardown_content_carrier()
        if was_visible:
            self.__notify_visibility_changed(False)

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def isVisible(self) -> bool:
        return bool(getattr(self, "_visible", False))

    def park_for_assistant_overlay(self) -> bool:
        """Temporarily remove the live carrier while assistant owns the screen."""
        if not bool(getattr(self, "_visible", False)):
            return False
        panel = getattr(self, "_panel", None)
        if panel is not None:
            panel.orderOut_(None)
        self._assistant_overlay_parked = True
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
        panel.orderFrontRegardless()
        if self.__should_publish_shell():
            self.__publish_shell_state("materialize")
            self.__publish_shell_state("rest")
        return True

    def cleanup(self) -> None:
        self.hide()
        self._panel = None
        self._content_view = None

    def mark_content_verified_for_test(self, title: str = "Perceptasia 3D") -> None:
        self.__mark_content_verified({"title": title})

    def probeThroughglassContent_(self, _sender) -> None:
        self.__probe_content_ready()

    def __requires_verified_content(self) -> bool:
        return _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY") or _env_flag(
            "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE"
        )

    def __should_publish_shell(self) -> bool:
        return _env_flag("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL")

    def __schedule_content_probe(self, *, delay: float) -> None:
        scheduler = getattr(self, "performSelector_withObject_afterDelay_", None)
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
            if self._content_probe_attempts < 10:
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

    def __teardown_content_carrier(self) -> None:
        content = self._content_view
        panel = self._panel
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
        self._content_generation += 1
        self._panel = None
        self._content_view = None
        self._content_kind = "uninitialized"
        self._content_verified = False
        self._content_failure = None
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
        frame = self._panel.frame()
        screen = NSScreen.mainScreen()
        return _display_local_scaled_window_bounds(frame, screen)

    def __publish_shell_state(self, state: str, *, visible: bool = True) -> bool:
        if self._registry is None or self._panel is None or self._content_view is None:
            logger.info("Perceptasia Throughglass: publish skipped state=%s", state)
            return False
        if self._host is None:
            host_for_screen = getattr(self._registry, "host_for_screen", None)
            if not callable(host_for_screen):
                logger.info("Perceptasia Throughglass: registry has no host_for_screen")
                return False
            self._host = host_for_screen(NSScreen.mainScreen())
        bounds, coordinate_metadata = self.__bounds_and_coordinate_metadata()
        config = compile_perceptasia_shell_config(bounds, state=state, visible=visible)
        _annotate_shell_coordinate_metadata(config, coordinate_metadata)
        if not getattr(self, "_client_registered", False):
            added = self._host.add_client(_CLIENT_ID, self._panel, self._content_view, config)
            self._client_registered = bool(added)
            logger.info(
                "Perceptasia Throughglass: publish state=%s registered=%s",
                state,
                self._client_registered,
            )
            record_command_overlay_trace(
                f"throughglass.publish.{state}",
                visible=visible,
                updated=bool(added),
                registered=self._client_registered,
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
            updated=updated,
            registered=True,
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
    y = float(frame.origin.y) + (float(frame.size.height) - height) * 0.5
    return x, y, width, height


def _real_attr(value, attr: str, default: float = 0.0) -> float:
    candidate = getattr(value, attr, default)
    if isinstance(candidate, numbers.Real):
        return float(candidate)
    return float(default)


def _rect_numbers(rect) -> tuple[float, float, float, float]:
    origin = getattr(rect, "origin", None)
    size = getattr(rect, "size", None)
    return (
        _real_attr(origin, "x"),
        _real_attr(origin, "y"),
        _real_attr(size, "width", _DEFAULT_WIDTH),
        _real_attr(size, "height", _DEFAULT_HEIGHT),
    )


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


def _configure_content_carrier(content_root, content, width: float, height: float) -> None:
    frame_setter = getattr(content, "setFrame_", None)
    if callable(frame_setter):
        frame_setter(NSMakeRect(0, 0, width, height))
    _set_view_autoresizing(content)
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
    content_layer_setter = getattr(content, "setWantsLayer_", None)
    if callable(content_layer_setter):
        content_layer_setter(True)
    content_layer_getter = getattr(content, "layer", None)
    content_layer = content_layer_getter() if callable(content_layer_getter) else None
    _shape_throughglass_carrier_layer(content_layer, radius=corner_radius)
