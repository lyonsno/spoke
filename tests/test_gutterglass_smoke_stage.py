from __future__ import annotations

import importlib
import json
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _install_webkit(monkeypatch, webview):
    webkit = types.ModuleType("WebKit")
    webkit.WKWebView = MagicMock()
    webkit.WKWebView.alloc.return_value.initWithFrame_.return_value = webview
    monkeypatch.setitem(sys.modules, "WebKit", webkit)
    return webkit


def _request_payload(**overrides):
    payload = {
        "schema": "spoke.gutterglass-smoke-stage.request.v0",
        "source_sign": "diaulos:smoke-scout",
        "title": "Smoke Scout Report",
        "content_kind": "html",
        "path": "report.html",
        "created_at": 1_800.0,
        "lifecycle": "ephemeral",
        "receipt_refs": ["spoke://smoke/123"],
    }
    payload.update(overrides)
    return payload


def test_stage_request_accepts_flexible_local_artifact_with_source_provenance(
    mock_pyobjc,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    report = tmp_path / "report.html"
    report.write_text("<h1>smoke</h1>", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")

    document = module.load_gutterglass_request(
        request_path,
        now=1_820.0,
        max_age_seconds=300.0,
    )

    assert document.request.source_sign == "diaulos:smoke-scout"
    assert document.request.content_kind == "html"
    assert document.request.target_url == report.resolve().as_uri()
    assert document.request.receipt_refs == ("spoke://smoke/123",)
    assert document.is_stale is False
    assert document.provenance_label == "diaulos:smoke-scout · html · spoke://smoke/123"


def test_stage_request_marks_stale_without_erasing_provenance(mock_pyobjc, tmp_path):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="url",
                uri="http://localhost:8899/report",
                created_at=100.0,
            )
        ),
        encoding="utf-8",
    )

    document = module.load_gutterglass_request(
        request_path,
        now=1_000.0,
        max_age_seconds=60.0,
    )

    assert document.is_stale is True
    assert document.request.source_sign == "diaulos:smoke-scout"
    assert "stale" in document.status_message.lower()
    assert "diaulos:smoke-scout" in document.status_message


def test_stage_writer_publishes_round_trippable_source_signed_request(
    mock_pyobjc,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")
    output_path = tmp_path / "stage" / "request.json"

    module.write_gutterglass_request(
        output_path,
        source_sign="diaulos:frame-hunter",
        title="Frame Hunter",
        content_kind="image",
        target=str(frame),
        receipt_refs=("retina://capture/11", "spoke://trace/abc"),
        created_at=2_000.0,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == module.SCHEMA
    assert payload["source_sign"] == "diaulos:frame-hunter"
    assert payload["path"] == str(frame)
    assert payload["receipt_refs"] == ["retina://capture/11", "spoke://trace/abc"]

    document = module.load_gutterglass_request(
        output_path,
        now=2_001.0,
        max_age_seconds=300.0,
    )
    assert document.request.content_kind == "image"
    assert document.request.target_url == frame.resolve().as_uri()


def test_stage_cli_publish_writes_requested_path(mock_pyobjc, tmp_path):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    report = tmp_path / "report.md"
    report.write_text("# smoke\n", encoding="utf-8")
    output_path = tmp_path / "stage.json"

    exit_code = module.main(
        [
            "publish",
            "--request-path",
            str(output_path),
            "--source-sign",
            "diaulos:cli-smoke",
            "--title",
            "CLI Smoke",
            "--kind",
            "markdown",
            "--target",
            str(report),
            "--receipt-ref",
            "spoke://cli/1",
        ]
    )

    assert exit_code == 0
    document = module.load_gutterglass_request(
        output_path,
        now=time.time(),
        max_age_seconds=300.0,
    )
    assert document.request.source_sign == "diaulos:cli-smoke"
    assert document.request.content_kind == "markdown"
    assert document.request.receipt_refs == ("spoke://cli/1",)


def test_stage_panel_renders_text_file_with_provenance_header(
    mock_pyobjc,
    monkeypatch,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    text_file = tmp_path / "receipt.txt"
    text_file.write_text("the harness saw the button", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="text",
                path=str(text_file),
                created_at=time.time(),
                title="Button Smoke",
                receipt_refs=["retina://frame/9"],
            )
        ),
        encoding="utf-8",
    )

    webview = MagicMock(name="webview")
    _install_webkit(monkeypatch, webview)
    panel = MagicMock(name="panel")
    panel.contentView.return_value = MagicMock(name="content-root")
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )

    stage = module.GutterglassSmokeStage.alloc().initWithRequestPath_(request_path)
    assert stage.show() is True

    html = webview.loadHTMLString_baseURL_.call_args.args[0]
    assert "Button Smoke" in html
    assert "diaulos:smoke-scout · text · retina://frame/9" in html
    assert "the harness saw the button" in html
    panel.orderFrontRegardless.assert_called_once_with()


def test_stage_panel_close_marks_hidden_and_keeps_panel_reusable(
    mock_pyobjc,
    monkeypatch,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    text_file = tmp_path / "receipt.txt"
    text_file.write_text("close me without stranding state", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="text",
                path=str(text_file),
                created_at=time.time(),
            )
        ),
        encoding="utf-8",
    )

    webview = MagicMock(name="webview")
    _install_webkit(monkeypatch, webview)
    panel = MagicMock(name="panel")
    panel.contentView.return_value = MagicMock(name="content-root")
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )

    stage = module.GutterglassSmokeStage.alloc().initWithRequestPath_(request_path)

    assert stage.show() is True
    assert stage.isVisible() is True
    panel.setReleasedWhenClosed_.assert_called_once_with(False)
    panel.setDelegate_.assert_called_once_with(stage)

    stage.windowWillClose_(None)

    assert stage.isVisible() is False
    stage.toggle()
    assert stage.isVisible() is True
    assert panel.orderFrontRegardless.call_count == 2


def test_stage_panel_loads_remote_url_request_in_webkit(
    mock_pyobjc,
    monkeypatch,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="url",
                uri="http://localhost:8888/smoke",
                created_at=time.time(),
                title="Live Browser Smoke",
            )
        ),
        encoding="utf-8",
    )

    webview = MagicMock(name="webview")
    _install_webkit(monkeypatch, webview)
    panel = MagicMock(name="panel")
    panel.contentView.return_value = MagicMock(name="content-root")
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    module.NSScreen.mainScreen.return_value.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )

    stage = module.GutterglassSmokeStage.alloc().initWithRequestPath_(request_path)
    assert stage.show() is True

    webview.loadRequest_.assert_called_once()
    request_arg = webview.loadRequest_.call_args.args[0]
    assert "localhost:8888/smoke" in repr(request_arg)


def test_stage_panel_show_registers_house_primitive_shell(
    mock_pyobjc,
    monkeypatch,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    from spoke.gutterglass_primitive_passport import GUTTERGLASS_PRIMITIVE_CLIENT_ID

    text_file = tmp_path / "receipt.txt"
    text_file.write_text("primitive smoke", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="text",
                path=str(text_file),
                created_at=time.time(),
            )
        ),
        encoding="utf-8",
    )

    webview = MagicMock(name="webview")
    _install_webkit(monkeypatch, webview)
    panel = MagicMock(name="panel")
    content_root = MagicMock(name="content-root")
    panel.contentView.return_value = content_root
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=1040.0, height=620.0),
    )
    screen = module.NSScreen.mainScreen.return_value
    screen.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    panel.screen.return_value = screen
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    host = MagicMock()
    host.add_client.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))

    stage = module.GutterglassSmokeStage.alloc().initWithRequestPath_compositorRegistry_(
        request_path,
        registry,
    )
    stage.performSelector_withObject_afterDelay_ = lambda *_args: None

    assert stage.show() is True

    host.add_client.assert_called()
    client_id, window, content_view, config = host.add_client.call_args.args
    assert client_id == GUTTERGLASS_PRIMITIVE_CLIENT_ID
    assert window is panel
    assert content_view is webview
    assert config["client_id"] == GUTTERGLASS_PRIMITIVE_CLIENT_ID
    assert config["optical_field"]["profile"] == "assistant_shell"
    assert config["optical_field"]["source_rect_basis"] == "gutterglass_panel"
    assert config["content_width_points"] < 1040.0 * 2.0
    assert config["content_height_points"] < 620.0 * 2.0
    assert config["gpu_material_base_width_points"] > 1040.0 * 2.0
    assert config["gpu_material_base_height_points"] > 620.0 * 2.0


def test_stage_panel_hide_uses_shared_radial_pucker_before_release(
    mock_pyobjc,
    monkeypatch,
    tmp_path,
):
    module = importlib.import_module("spoke.gutterglass_smoke_stage")
    from spoke.gutterglass_primitive_passport import (
        GUTTERGLASS_PRIMITIVE_CLIENT_ID,
        GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID,
    )

    text_file = tmp_path / "receipt.txt"
    text_file.write_text("dismiss me with the oscillator", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request_payload(
                content_kind="text",
                path=str(text_file),
                created_at=time.time(),
            )
        ),
        encoding="utf-8",
    )

    webview = MagicMock(name="webview")
    _install_webkit(monkeypatch, webview)
    panel = MagicMock(name="panel")
    panel.contentView.return_value = MagicMock(name="content-root")
    panel.frame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=100.0, y=80.0),
        size=SimpleNamespace(width=1040.0, height=620.0),
    )
    screen = module.NSScreen.mainScreen.return_value
    screen.visibleFrame.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0.0, y=0.0),
        size=SimpleNamespace(width=1440.0, height=900.0),
    )
    screen.backingScaleFactor.return_value = 2.0
    panel.screen.return_value = screen
    module.NSPanel.alloc.return_value.initWithContentRect_styleMask_backing_defer_.return_value = panel
    host = MagicMock()
    host.add_client.return_value = True
    host.update_client_config.return_value = True
    registry = SimpleNamespace(host_for_screen=MagicMock(return_value=host))

    stage = module.GutterglassSmokeStage.alloc().initWithRequestPath_compositorRegistry_(
        request_path,
        registry,
    )
    stage.performSelector_withObject_afterDelay_ = lambda *_args: None
    assert stage.show() is True

    stage.hide()

    radial_calls = [
        call
        for call in host.add_client.call_args_list
        if call.args[0] == GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID
    ]
    assert radial_calls
    radial_config = radial_calls[-1].args[3]
    assert radial_config["client_id"] == GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID
    assert radial_config["client_id"] != "assistant.command.dismiss_radial_pucker"
    assert radial_config["role"] == "hud"
    assert radial_config["warp_mode"] == 2.0
    panel.orderOut_.assert_not_called()

    stage._stage_shell_animation_started_at -= stage._stage_shell_animation_duration + 0.01
    stage.animateGutterglassShellStep_(None)

    panel.orderOut_.assert_called_once_with(None)
    host.release_client.assert_any_call(GUTTERGLASS_PRIMITIVE_CLIENT_ID)
    host.release_client.assert_any_call(GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID)
