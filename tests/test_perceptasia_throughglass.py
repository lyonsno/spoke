from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spoke.optical_field import OpticalFieldBounds


def _connection_refused(*_args, **_kwargs):
    raise OSError("connection refused")


def test_throughglass_declares_webkit_pyobjc_framework_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = {
        dependency.split(";", 1)[0].split(">=", 1)[0].split("==", 1)[0].strip().lower()
        for dependency in pyproject["project"]["dependencies"]
    }

    assert "pyobjc-framework-webkit" in dependencies


def test_manifest_defaults_to_current_local_perceptasia_provider(mock_pyobjc, monkeypatch):
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.delenv("SPOKE_PERCEPTASIA_THROUGHGLASS_URL", raising=False)
    monkeypatch.setattr(module.urllib.request, "urlopen", _connection_refused)
    manifest = module.PerceptasiaThroughglassManifest.from_env()

    assert manifest.schema == "spoke.perceptasia-throughglass.provider.v0"
    assert manifest.url == "http://localhost:8742"
    assert manifest.scene_url == "http://localhost:8742/scene.json"
    assert manifest.selection_path.endswith(".local/state/perceptasia/selection.json")


def test_manifest_env_override_is_provider_contract_not_window_lifecycle(mock_pyobjc, monkeypatch):
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_URL", "http://localhost:9999/")
    monkeypatch.setattr(module.urllib.request, "urlopen", _connection_refused)
    manifest = module.PerceptasiaThroughglassManifest.from_env()

    assert manifest.url == "http://localhost:9999"
    assert manifest.scene_url == "http://localhost:9999/scene.json"


def test_manifest_discovers_live_local_provider_when_requested_port_is_dead(mock_pyobjc, monkeypatch):
    module = importlib.import_module("spoke.perceptasia_throughglass")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_URL", "http://localhost:8742")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_DISCOVERY_PORTS", "8753")

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout=None):
        if request.full_url == "http://localhost:8753":
            return _Response()
        raise OSError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    manifest = module.PerceptasiaThroughglassManifest.from_env()

    assert manifest.url == "http://localhost:8753"
    assert manifest.scene_url == "http://localhost:8753/scene.json"


def test_manifest_skips_non_perceptasia_directory_listing(mock_pyobjc, monkeypatch):
    module = importlib.import_module("spoke.perceptasia_throughglass")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_URL", "http://localhost:8742")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_DISCOVERY_PORTS", "8797,8798")

    class _Response:
        status = 200

        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return self._body

    def fake_urlopen(request, timeout=None):
        if request.full_url == "http://localhost:8797":
            return _Response(b"<title>Directory listing for /</title>")
        if request.full_url == "http://localhost:8798":
            return _Response(b"<title>Perceptasia 3D</title>")
        raise OSError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    manifest = module.PerceptasiaThroughglassManifest.from_env()

    assert manifest.url == "http://localhost:8798"


def test_throughglass_request_is_independent_sibling_without_progress_custody(mock_pyobjc):
    from spoke.perceptasia_throughglass import build_perceptasia_optical_request

    bounds = OpticalFieldBounds(100.0, 80.0, 900.0, 520.0)
    request = build_perceptasia_optical_request(bounds, state="rest")

    assert request.caller_id == "perceptasia.throughglass"
    assert request.role == "hud"
    assert request.visibility_scope == "independent"
    assert request.layout_recipe == "perceptasia-primitive-passport"
    assert request.profile.base == "captured_scene"
    assert request.presentation.layer == "hud"
    assert request.presentation.order == 42


def test_throughglass_compiles_to_public_optical_field_shell_config(mock_pyobjc):
    from spoke.perceptasia_throughglass import compile_perceptasia_shell_config

    bounds = OpticalFieldBounds(100.0, 80.0, 900.0, 520.0)
    config = compile_perceptasia_shell_config(bounds, state="materialize")

    assert config["client_id"] == "perceptasia.throughglass"
    assert config["role"] == "hud"
    assert config["presentation_layer"] == "hud"
    assert config["presentation_order"] == 42
    assert config["optical_field"]["layout_recipe"] == "perceptasia-primitive-passport"
    assert config["optical_field"]["state"] == "materialize"
    assert "progress" not in config["optical_field"]
    assert "phase" not in config["optical_field"]


def test_throughglass_shell_preserves_live_webview_content_but_still_draws_perimeter(mock_pyobjc):
    from spoke.perceptasia_throughglass import compile_perceptasia_shell_config

    bounds = OpticalFieldBounds(100.0, 80.0, 900.0, 520.0)
    config = compile_perceptasia_shell_config(bounds, state="rest")

    assert config["client_id"] == "perceptasia.throughglass"
    assert config["visible"] is True
    assert config["mip_blur_strength"] == pytest.approx(0.0)
    assert config["gpu_material_enabled"] == pytest.approx(1.0)
    assert 0.0 < config["gpu_material_opacity"] <= 0.45
    assert config["gpu_material_feather_points"] >= 90.0


def test_throughglass_default_panel_rect_seats_independent_consumer_in_top_band(mock_pyobjc):
    module = importlib.import_module("spoke.perceptasia_throughglass")
    frame = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )

    x, y, width, height = module._default_panel_rect(frame)

    assert width == pytest.approx(980.0)
    assert height == pytest.approx(560.0)
    assert x == pytest.approx(230.0)
    assert y == pytest.approx(308.0)


def test_throughglass_real_pyobjc_import_accepts_private_helpers():
    if (
        importlib.util.find_spec("objc") is None
        or importlib.util.find_spec("AppKit") is None
        or importlib.util.find_spec("WebKit") is None
    ):
        pytest.skip("PyObjC/AppKit unavailable")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from WebKit import WKWebView; "
            "from spoke.perceptasia_throughglass import PerceptasiaThroughglassGraft; "
            "print(PerceptasiaThroughglassGraft.__name__)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "PerceptasiaThroughglassGraft" in result.stdout


def test_throughglass_ui_delegate_grants_webkit_media_capture_permission(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    decisions = []

    delegate = module._ThroughglassUIDelegate.alloc().init()
    delegate.webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        0,
        decisions.append,
    )

    assert decisions == [1]


def test_throughglass_ui_delegate_uses_explicit_block_call_for_opaque_webkit_handler(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    class OpaqueDecisionHandler:
        def __init__(self):
            self.__block_signature__ = None
            self.decisions = []

        def __call__(self, _decision):
            if self.__block_signature__ is None:
                raise TypeError("cannot call block without a signature")
            self.decisions.append(_decision)

    handler = OpaqueDecisionHandler()

    delegate = module._ThroughglassUIDelegate.alloc().init()
    delegate.webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        0,
        handler,
    )

    assert handler.__block_signature__ == b"v@?q"
    assert handler.decisions == [1]


def test_throughglass_ui_delegate_fails_loud_when_webkit_handler_still_uncallable(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    class UncallableDecisionHandler:
        def __call__(self, _decision):
            raise TypeError("cannot call block without a signature")

    delegate = module._ThroughglassUIDelegate.alloc().init()
    with pytest.raises(TypeError, match="cannot call block without a signature"):
        delegate.webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            0,
            UncallableDecisionHandler(),
        )


def test_throughglass_ui_delegate_registers_decision_handler_block_metadata(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    objc = sys.modules["objc"]

    metadata = objc._registeredMetadataForSelector(
        module._ThroughglassUIDelegate,
        b"webView:requestMediaCapturePermissionForOrigin:initiatedByFrame:type:decisionHandler:",
    )

    assert metadata is not None
    assert metadata["arguments"][5]["type"] == b"q"
    handler = metadata["arguments"][6]
    assert handler["type"] == b"@?"
    assert handler["callable"]["retval"]["type"] == b"v"
    assert handler["callable"]["arguments"][1]["type"] == b"q"


def test_throughglass_webview_leaves_media_capture_on_native_webkit_path_by_default(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    foundation = sys.modules["Foundation"]
    foundation.NSURL = SimpleNamespace(URLWithString_=MagicMock(return_value="url"))
    foundation.NSURLRequest = SimpleNamespace(requestWithURL_=MagicMock(return_value="request"))

    view = MagicMock()
    config = MagicMock()
    webkit = types.ModuleType("WebKit")
    webkit.WKWebViewConfiguration = MagicMock()
    webkit.WKWebViewConfiguration.alloc.return_value.init.return_value = config
    webkit.WKWebView = MagicMock()
    webkit.WKWebView.alloc.return_value.initWithFrame_.return_value = view
    webkit.WKWebView.alloc.return_value.initWithFrame_configuration_.return_value = view
    monkeypatch.setitem(sys.modules, "WebKit", webkit)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    content, kind = module._make_content_view("http://localhost:8753", 900.0, 520.0)

    assert content is view
    assert kind == "webview"
    webkit.WKWebView.alloc.return_value.initWithFrame_configuration_.assert_called_once_with(
        foundation.NSMakeRect.return_value,
        config,
    )
    assert not hasattr(webkit, "WKUserScript") or not webkit.WKUserScript.called
    view.setUIDelegate_.assert_not_called()


def test_throughglass_primitive_shell_hides_webkit_scrollbars_before_capture(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    foundation = sys.modules["Foundation"]
    foundation.NSURL = SimpleNamespace(URLWithString_=MagicMock(return_value="url"))
    foundation.NSURLRequest = SimpleNamespace(requestWithURL_=MagicMock(return_value="request"))

    view = MagicMock()
    config = MagicMock()
    controller = MagicMock()
    script = MagicMock()
    webkit = types.ModuleType("WebKit")
    webkit.WKWebViewConfiguration = MagicMock()
    webkit.WKWebViewConfiguration.alloc.return_value.init.return_value = config
    webkit.WKUserContentController = MagicMock()
    webkit.WKUserContentController.alloc.return_value.init.return_value = controller
    webkit.WKUserScript = MagicMock()
    webkit.WKUserScript.alloc.return_value.initWithSource_injectionTime_forMainFrameOnly_.return_value = script
    webkit.WKUserScriptInjectionTimeAtDocumentStart = 0
    webkit.WKWebView = MagicMock()
    webkit.WKWebView.alloc.return_value.initWithFrame_configuration_.return_value = view
    monkeypatch.setitem(sys.modules, "WebKit", webkit)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    content, kind = module._make_content_view("http://localhost:8753", 900.0, 520.0)

    assert content is view
    assert kind == "webview"
    config.setUserContentController_.assert_called_once_with(controller)
    controller.addUserScript_.assert_called_once_with(script)
    source = webkit.WKUserScript.alloc.return_value.initWithSource_injectionTime_forMainFrameOnly_.call_args.args[0]
    assert "::-webkit-scrollbar" in source
    assert "overflow: hidden" in source
    assert "background: #050708" in source


def test_throughglass_primitive_capture_css_uses_dark_scene_backing_not_source_plate(
    mock_pyobjc,
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    css = module._THROUGHGLASS_PRIMITIVE_CAPTURE_CSS

    assert "#050708" in css
    assert "background: transparent" not in css


def test_throughglass_shell_probe_rejects_dom_controls_without_scene_pixels(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    result = {
        "title": "Perceptasia 3D",
        "readyState": "complete",
        "bodyText": (
            "Start Hand Control Native Stream Off Frame Low Authority WiLoR "
            "Orbit On Spring Absolute Witness Off"
        ),
        "canvasCount": 1,
        "canvasSampledPixels": 4096,
        "canvasVisualSignal": 0.0,
    }

    assert graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(result) is False


def test_throughglass_shell_probe_accepts_scene_pixels_without_dom_controls(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    result = {
        "title": "Perceptasia 3D",
        "readyState": "complete",
        "bodyText": "",
        "canvasCount": 1,
        "canvasSampledPixels": 4096,
        "canvasVisualSignal": 0.08,
    }

    assert graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(result) is True


def test_throughglass_panel_accepts_pointer_input_by_default(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.delenv("SPOKE_PERCEPTASIA_THROUGHGLASS_CLICK_THROUGH", raising=False)
    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    panel.setLevel_.assert_called_with(25)
    panel.setIgnoresMouseEvents_.assert_called_once_with(False)


def test_throughglass_panel_level_is_reasserted_after_floating_panel_setup(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    events = []
    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.setFloatingPanel_.side_effect = lambda value: events.append(("floating", value))
    panel.setLevel_.side_effect = lambda level: events.append(("level", level))
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    assert events[-2:] == [("floating", True), ("level", 25)]


def test_throughglass_shell_publish_places_webview_under_compositor_for_capture(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    events = []
    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.setFloatingPanel_.side_effect = lambda value: events.append(("floating", value))
    panel.setLevel_.side_effect = lambda level: events.append(("level", level))
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    assert ("level", 23) in events
    assert events[-2:] == [("floating", True), ("level", 23)]


def test_throughglass_panel_can_opt_into_click_through_debug_mode(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_CLICK_THROUGH", "1")
    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    panel.setIgnoresMouseEvents_.assert_called_once_with(True)


def test_throughglass_webview_panel_is_opaque_content_carrier(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)
    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    panel.setOpaque_.assert_called_once_with(True)
    module.NSColor.colorWithWhite_alpha_.assert_any_call(0.0, 1.0)
    panel.setBackgroundColor_.assert_called()


def test_throughglass_primitive_shell_masks_live_carrier_to_rounded_optical_body(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    root_layer = MagicMock()
    content_layer = MagicMock()
    content_root = MagicMock()
    content_root.layer.return_value = root_layer
    panel = MagicMock()
    panel.contentView.return_value = content_root
    content = MagicMock()
    content.layer.return_value = content_layer
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: content)

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()

    panel.setOpaque_.assert_called_once_with(False)
    module.NSColor.colorWithWhite_alpha_.assert_any_call(0.0, 0.0)
    root_layer.setMasksToBounds_.assert_called_once_with(True)
    content_layer.setMasksToBounds_.assert_called_once_with(True)
    assert root_layer.setCornerRadius_.call_args.args[0] == pytest.approx(
        content_layer.setCornerRadius_.call_args.args[0]
    )
    assert root_layer.setCornerRadius_.call_args.args[0] > 1.0


def test_throughglass_smoke_defers_unverified_webview_content(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    panel.orderFrontRegardless.assert_not_called()
    host.add_client.assert_not_called()


def test_throughglass_content_verification_releases_deferred_smoke(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")

    assert panel.orderFrontRegardless.call_count == 1
    host.add_client.assert_not_called()
    host.update_client_config.assert_not_called()


def test_throughglass_optical_shell_is_explicit_opt_in_for_live_webview(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    assert panel.orderFrontRegardless.call_count == 1
    assert panel.setLevel_.call_count >= 3
    assert host.add_client.call_count == 1
    assert host.update_client_config.call_count == 1


def test_throughglass_shell_publish_waits_for_carrier_present_tick(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)
    scheduled = []
    graft.performSelector_withObject_afterDelay_ = (
        lambda selector, obj, delay: scheduled.append((selector, obj, delay))
    )

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")

    assert panel.orderFrontRegardless.call_count == 1
    host.add_client.assert_not_called()
    host.update_client_config.assert_not_called()
    assert scheduled == [
        ("publishThroughglassShellAfterCarrierPresent:", None, pytest.approx(0.08))
    ]

    graft.publishThroughglassShellAfterCarrierPresent_(None)

    assert host.add_client.call_count == 1
    host.update_client_config.assert_not_called()
    assert scheduled[-1] == (
        "publishThroughglassShellRestAfterMaterialize:",
        None,
        pytest.approx(0.12),
    )


def test_throughglass_shell_materialize_survives_until_settle_tick(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)
    scheduled = []
    graft.performSelector_withObject_afterDelay_ = (
        lambda selector, obj, delay: scheduled.append((selector, obj, delay))
    )

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    assert host.add_client.call_count == 1
    assert host.add_client.call_args.args[3]["optical_field"]["state"] == "materialize"
    host.update_client_config.assert_not_called()
    assert scheduled[-1] == (
        "publishThroughglassShellRestAfterMaterialize:",
        None,
        pytest.approx(0.12),
    )

    graft.publishThroughglassShellRestAfterMaterialize_(None)

    assert host.update_client_config.call_count == 1
    assert host.update_client_config.call_args.args[1]["optical_field"]["state"] == "rest"


def test_throughglass_shell_uses_content_view_bounds_not_outer_panel_frame(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    outer_frame = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    content_bounds = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=880.0, height=500.0),
    )
    content_window_rect = SimpleNamespace(
        origin=SimpleNamespace(x=10.0, y=12.0),
        size=SimpleNamespace(width=880.0, height=500.0),
    )
    content_screen_rect = SimpleNamespace(
        origin=SimpleNamespace(x=110.0, y=92.0),
        size=SimpleNamespace(width=880.0, height=500.0),
    )
    content_root = MagicMock()
    content = MagicMock()
    content.bounds.return_value = content_bounds
    content.convertRect_toView_.return_value = content_window_rect
    panel.contentView.return_value = content_root
    panel.frame.return_value = outer_frame
    panel.convertRectToScreen_.return_value = content_screen_rect
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    screen = module.NSScreen.mainScreen.return_value
    screen.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: content)

    host = MagicMock()
    host.add_client.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    config = host.add_client.call_args.args[3]
    assert config["content_width_points"] == pytest.approx(1760.0)
    assert config["content_height_points"] == pytest.approx(1000.0)
    assert config["center_x"] == pytest.approx(1100.0)
    assert config["center_y"] == pytest.approx(1116.0)
    assert config["optical_field"]["source_rect_basis"] == "content_view"


def test_throughglass_shell_materialize_hold_can_be_widened_for_visual_witness(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setenv(
        "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_SETTLE_DELAY_SECONDS",
        "0.42",
    )
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)
    scheduled = []
    graft.performSelector_withObject_afterDelay_ = (
        lambda selector, obj, delay: scheduled.append((selector, obj, delay))
    )

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    assert scheduled[-1] == (
        "publishThroughglassShellRestAfterMaterialize:",
        None,
        pytest.approx(0.42),
    )


def test_throughglass_shell_publish_emits_trace_receipts_for_visual_witness(
    mock_pyobjc, monkeypatch, tmp_path
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "throughglass.publish.materialize",
        "throughglass.publish.rest",
    ]
    assert events[0]["visible"] is True
    assert events[0]["updated"] is True
    assert events[0]["width"] == 1800.0


def test_throughglass_publishes_display_local_scaled_geometry_for_primitive_shell(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_PUBLISH_SHELL", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=230.0, y=90.0),
        size=SimpleNamespace(width=980.0, height=560.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    screen = module.NSScreen.mainScreen.return_value
    screen.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: MagicMock())

    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(registry)

    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.publishThroughglassShellAfterCarrierPresent_(None)

    published_config = host.add_client.call_args.args[3]
    assert published_config["center_x"] == pytest.approx(1440.0)
    assert published_config["center_y"] == pytest.approx(1060.0)
    assert published_config["content_width_points"] == pytest.approx(1960.0)
    assert published_config["content_height_points"] == pytest.approx(1120.0)
    assert published_config["corner_radius_points"] <= 560.0
    assert published_config["optical_field"]["source_coordinate_space"] == "screen_points"
    assert published_config["optical_field"]["backing_scale"] == pytest.approx(2.0)


def test_throughglass_exposes_visible_state_for_menu_toggle(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)

    assert graft.isVisible() is False

    graft._visible = True
    assert graft.isVisible() is True


def test_throughglass_reports_display_visibility_independent_of_assistant_overlay(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft._panel = MagicMock()
    events = []

    graft.set_visibility_callback(lambda visible: events.append(bool(visible)))

    assert graft.show() is True
    graft.hide()

    assert events == [True, False]


def test_throughglass_can_park_without_destroying_live_webview(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    panel = MagicMock()
    webview = MagicMock()
    graft._panel = panel
    graft._content_view = webview
    graft._visible = True

    assert graft.park_for_assistant_overlay() is True

    panel.orderOut_.assert_called_once_with(None)
    assert graft.isVisible() is True
    assert graft._content_view is webview


def test_throughglass_dismiss_survives_until_hide_finish_tick(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    events = []
    panel = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    panel.orderOut_.side_effect = lambda _sender: events.append(("order_out", None))
    screen = module.NSScreen.mainScreen.return_value
    screen.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    host = MagicMock()
    host.update_client_config.side_effect = (
        lambda _client_id, config: events.append(
            ("publish", config["optical_field"]["state"])
        )
        or True
    )
    host.release_client.side_effect = lambda _client_id: events.append(("release", None))
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft._registry = registry
    graft._panel = panel
    graft._content_view = MagicMock()
    graft._host = host
    graft._client_registered = True
    graft._visible = True
    scheduled = []
    graft.performSelector_withObject_afterDelay_ = (
        lambda selector, obj, delay: scheduled.append((selector, obj, delay))
    )

    graft.hide()

    assert events == [("publish", "dismiss")]
    assert scheduled[-1] == (
        "finishThroughglassHideAfterDismiss:",
        None,
        pytest.approx(0.12),
    )

    graft.finishThroughglassHideAfterDismiss_(None)

    assert events[:3] == [
        ("publish", "dismiss"),
        ("publish", "hidden"),
        ("order_out", None),
    ]
    assert events[-1] == ("release", None)


def test_throughglass_dismiss_hold_can_be_widened_for_visual_witness(
    mock_pyobjc, monkeypatch
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    monkeypatch.setenv(
        "SPOKE_PERCEPTASIA_THROUGHGLASS_SHELL_DISMISS_DELAY_SECONDS",
        "0.46",
    )
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    panel = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    screen = module.NSScreen.mainScreen.return_value
    screen.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    host = MagicMock()
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft._registry = registry
    graft._panel = panel
    graft._content_view = MagicMock()
    graft._host = host
    graft._client_registered = True
    graft._visible = True
    scheduled = []
    graft.performSelector_withObject_afterDelay_ = (
        lambda selector, obj, delay: scheduled.append((selector, obj, delay))
    )

    graft.hide()

    assert scheduled[-1] == (
        "finishThroughglassHideAfterDismiss:",
        None,
        pytest.approx(0.46),
    )


def test_throughglass_park_quarantines_carrier_before_hidden_publish(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    events = []
    panel_root = MagicMock()
    panel_root.setHidden_.side_effect = lambda hidden: events.append(("root_hidden", hidden))
    panel = MagicMock()
    panel.contentView.return_value = panel_root
    panel.setAlphaValue_.side_effect = lambda alpha: events.append(("panel_alpha", alpha))
    panel.setIgnoresMouseEvents_.side_effect = lambda ignored: events.append(
        ("mouse_ignored", ignored)
    )
    panel.orderOut_.side_effect = lambda _sender: events.append(("order_out", None))
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    content = MagicMock()
    content.setHidden_.side_effect = lambda hidden: events.append(("content_hidden", hidden))
    screen = module.NSScreen.mainScreen.return_value
    screen.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    host = MagicMock()
    host.update_client_config.side_effect = (
        lambda _client_id, _config: events.append(("publish_hidden", None)) or True
    )
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft._registry = registry
    graft._panel = panel
    graft._content_view = content
    graft._host = host
    graft._client_registered = True
    graft._visible = True

    assert graft.park_for_assistant_overlay() is True

    assert events.index(("panel_alpha", 0.0)) < events.index(("publish_hidden", None))
    assert events.index(("content_hidden", True)) < events.index(("publish_hidden", None))
    assert events.index(("order_out", None)) < events.index(("publish_hidden", None))


def test_throughglass_park_hides_registered_shell_when_visibility_is_stale(
    mock_pyobjc,
):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    panel = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    screen = module.NSScreen.mainScreen.return_value
    screen.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    host = MagicMock()
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))
    graft._registry = registry
    graft._panel = panel
    graft._content_view = MagicMock()
    graft._host = host
    graft._client_registered = True
    graft._visible = False

    assert graft.park_for_assistant_overlay() is True

    panel.orderOut_.assert_called_once_with(None)
    config = host.update_client_config.call_args.args[1]
    assert config["visible"] is False
    assert config["optical_field"]["state"] == "hidden"


def test_throughglass_restores_after_assistant_overlay_park(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    panel = MagicMock()
    graft._panel = panel
    graft._visible = True
    graft._assistant_overlay_parked = True

    assert graft.restore_after_assistant_overlay() is True

    panel.orderFrontRegardless.assert_called_once_with()
    assert graft._assistant_overlay_parked is False


def test_throughglass_hide_unloads_live_webview_carrier(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel = MagicMock()
    panel.contentView.return_value = MagicMock()
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    webview = MagicMock()
    monkeypatch.setattr(module, "_make_content_view", lambda url, width, height: (webview, "webview"))

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    graft.setup()
    graft.mark_content_verified_for_test("Perceptasia 3D")
    assert graft.show() is True

    graft.hide()

    panel.orderOut_.assert_called_once_with(None)
    webview.stopLoading.assert_called_once_with()
    webview.loadHTMLString_baseURL_.assert_called_once_with("", None)
    webview.removeFromSuperview.assert_called_once_with()
    assert graft._panel is None
    assert graft._content_view is None
    assert graft._content_verified is False


def test_throughglass_rehydrates_content_after_hide(mock_pyobjc, monkeypatch):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")

    monkeypatch.setenv("SPOKE_PERCEPTASIA_THROUGHGLASS_REQUIRE_CONTENT_READY", "1")
    monkeypatch.setattr(module, "_is_provider_reachable", lambda _url: True)

    panel1 = MagicMock()
    panel1.contentView.return_value = MagicMock()
    panel1.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=900.0, height=520.0),
    )
    panel2 = MagicMock()
    panel2.contentView.return_value = MagicMock()
    panel2.frame.return_value = panel1.frame.return_value
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.side_effect = [
        panel1,
        panel2,
    ]
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    webviews = [MagicMock(), MagicMock()]

    def make_content(_url, _width, _height):
        return webviews.pop(0), "webview"

    monkeypatch.setattr(module, "_make_content_view", make_content)

    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)
    assert graft.show() is False
    graft.mark_content_verified_for_test("Perceptasia 3D")
    graft.hide()

    assert graft.show() is False

    panel1.orderOut_.assert_called_once_with(None)
    assert panel2.orderFrontRegardless.call_count == 0
    assert graft._pending_show is True
    assert graft._content_verified is False
    assert graft._content_view is not None


def test_throughglass_probe_rejects_canvas_count_without_pixel_signal(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)

    matches = graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(
        {
            "title": "Perceptasia 3D",
            "readyState": "complete",
            "bodyText": "Perceptasia",
            "canvasCount": 2,
            "canvasSampledPixels": 1024,
            "canvasVisualSignal": 0.0,
        }
    )

    assert matches is False


def test_throughglass_probe_accepts_live_perceptasia_dom_when_canvas_readback_is_blank(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)

    matches = graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(
        {
            "title": "Perceptasia 3D",
            "readyState": "complete",
            "bodyText": (
                "Perceptasia 3D - loading... reticule: scout none locked none "
                "Start Hand Control Native Stream Off Frame Low Authority WiLoR "
                "Orbit On Spring Absolute Witness Off command idle"
            ),
            "canvasCount": 3,
            "canvasSampledPixels": 12288,
            "canvasVisualSignal": 0.0,
        }
    )

    assert matches is True


def test_throughglass_probe_accepts_current_perceptasia_viewer_dom_when_canvas_readback_is_blank(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)

    matches = graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(
        {
            "title": "Perceptasia 3D",
            "readyState": "complete",
            "bodyText": (
                "Perceptasia 3D - 1763 primitives, 3414 edges - remote - origin/main "
                "reticule: scout none locked none Structure Visibility Labels Edges "
                "Atmosphere Reticule Context Physics works_on belongs_to attracted_to"
            ),
            "canvasCount": 2,
            "canvasSampledPixels": 8192,
            "canvasVisualSignal": 0.0,
        }
    )

    assert matches is True


def test_throughglass_probe_accepts_perceptasia_canvas_with_pixel_signal(mock_pyobjc):
    sys.modules.pop("spoke.perceptasia_throughglass", None)
    module = importlib.import_module("spoke.perceptasia_throughglass")
    graft = module.PerceptasiaThroughglassGraft.alloc().initWithCompositorRegistry_(None)

    matches = graft._PerceptasiaThroughglassGraft__content_probe_matches_perceptasia(
        {
            "title": "Perceptasia 3D",
            "readyState": "complete",
            "bodyText": "Perceptasia",
            "canvasCount": 2,
            "canvasSampledPixels": 1024,
            "canvasVisualSignal": 0.037,
        }
    )

    assert matches is True
