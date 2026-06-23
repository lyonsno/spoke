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
