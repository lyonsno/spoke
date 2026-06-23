"""Gutterglass Smoke Stage.

Narrow Spoke-hosted aperture for source-signed agent smoke/artifact receipts.
This first slice is intentionally a latest-request panel, not a general window
manager and not a claim of operation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import html
import json
import logging
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSPanel,
    NSScreen,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject

logger = logging.getLogger(__name__)

SCHEMA = "spoke.gutterglass-smoke-stage.request.v0"
DEFAULT_MAX_AGE_SECONDS = 12 * 60 * 60
_DEFAULT_WIDTH = 1040.0
_DEFAULT_HEIGHT = 620.0
_MIN_MARGIN = 32.0
_NSWindowStyleMaskClosable = 1 << 1
_NSWindowStyleMaskResizable = 1 << 3
_NSWindowStyleMaskUtilityWindow = 1 << 4
_NSViewWidthSizable = 1 << 1
_NSViewHeightSizable = 1 << 4
_SUPPORTED_KINDS = {
    "url",
    "html",
    "image",
    "text",
    "markdown",
    "json",
    "directory",
    "filmstrip",
    "report",
}


@dataclass(frozen=True)
class GutterglassSmokeStageRequest:
    """Normalized source-signed request for the smoke stage."""

    source_sign: str
    title: str
    content_kind: str
    target_url: str
    created_at: float
    lifecycle: str
    receipt_refs: tuple[str, ...]
    path: Path | None = None
    summary: str = ""
    authority: str = "none"

    @classmethod
    def from_mapping(
        cls,
        payload: dict,
        *,
        base_dir: Path,
    ) -> "GutterglassSmokeStageRequest":
        schema = str(payload.get("schema", "")).strip()
        if schema != SCHEMA:
            raise ValueError(f"unsupported Gutterglass schema: {schema or '<missing>'}")

        source_sign = str(payload.get("source_sign", "")).strip()
        if not source_sign:
            raise ValueError("Gutterglass request missing source_sign")

        content_kind = str(payload.get("content_kind", "")).strip().lower()
        if content_kind not in _SUPPORTED_KINDS:
            raise ValueError(f"unsupported Gutterglass content_kind: {content_kind or '<missing>'}")

        uri = payload.get("uri")
        target_path = (
            None
            if uri is not None and str(uri).strip()
            else _resolve_request_path(payload.get("path"), base_dir=base_dir)
        )
        target_url = _resolve_target_url(uri, target_path)
        if not target_url:
            raise ValueError("Gutterglass request requires uri or path")

        return cls(
            source_sign=source_sign,
            title=str(payload.get("title") or "Gutterglass Smoke Stage"),
            content_kind=content_kind,
            target_url=target_url,
            created_at=_parse_created_at(payload.get("created_at")),
            lifecycle=str(payload.get("lifecycle") or "ephemeral"),
            receipt_refs=tuple(str(ref) for ref in payload.get("receipt_refs", ()) if str(ref)),
            path=target_path,
            summary=str(payload.get("summary") or ""),
            authority=str(payload.get("authority") or "none"),
        )

    @property
    def provenance_label(self) -> str:
        parts = [self.source_sign, self.content_kind]
        parts.extend(self.receipt_refs[:1])
        return " · ".join(parts)


@dataclass(frozen=True)
class GutterglassSmokeStageDocument:
    """Loaded request plus freshness/status metadata."""

    request: GutterglassSmokeStageRequest
    is_stale: bool
    status_message: str

    @property
    def provenance_label(self) -> str:
        return self.request.provenance_label


class _URLRequest:
    def __init__(self, url: str):
        self.url = url

    def __repr__(self) -> str:
        return f"<GutterglassURLRequest {self.url!r}>"


def default_request_path() -> Path:
    configured = os.environ.get("SPOKE_GUTTERGLASS_STAGE_REQUEST_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("~/.local/state/spoke/gutterglass_smoke_stage/request.json").expanduser()


def load_gutterglass_request(
    path: str | os.PathLike | None = None,
    *,
    now: float | None = None,
    max_age_seconds: float | None = None,
) -> GutterglassSmokeStageDocument:
    request_path = Path(path).expanduser() if path is not None else default_request_path()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    request = GutterglassSmokeStageRequest.from_mapping(payload, base_dir=request_path.parent)
    age = max(0.0, (time.time() if now is None else float(now)) - request.created_at)
    max_age = _env_max_age_seconds() if max_age_seconds is None else float(max_age_seconds)
    stale = age > max_age and request.lifecycle != "persistent"
    if stale:
        status = (
            f"Stale Gutterglass request from {request.source_sign}: "
            f"{age:.0f}s old, max {max_age:.0f}s"
        )
    else:
        status = f"Gutterglass request from {request.provenance_label}"
    return GutterglassSmokeStageDocument(request=request, is_stale=stale, status_message=status)


def write_gutterglass_request(
    path: str | os.PathLike | None = None,
    *,
    source_sign: str,
    title: str,
    content_kind: str,
    target: str | os.PathLike,
    receipt_refs=(),
    created_at: float | str | None = None,
    lifecycle: str = "ephemeral",
    summary: str = "",
    authority: str = "none",
) -> Path:
    """Publish a round-trippable latest request for the hosted smoke stage."""

    output_path = Path(path).expanduser() if path is not None else default_request_path()
    target_text = str(target).strip()
    if not target_text:
        raise ValueError("Gutterglass target must not be empty")

    payload = {
        "schema": SCHEMA,
        "source_sign": source_sign,
        "title": title,
        "content_kind": content_kind,
        "created_at": created_at if created_at is not None else time.time(),
        "lifecycle": lifecycle,
        "receipt_refs": [str(ref) for ref in receipt_refs if str(ref)],
        "summary": summary,
        "authority": authority,
    }
    parsed = urlparse(target_text)
    if parsed.scheme in {"http", "https", "file", "retina", "spoke"}:
        payload["uri"] = target_text
    else:
        payload["path"] = target_text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


class GutterglassSmokeStage(NSObject):
    """Non-activating panel for agent smoke/artifact receipts."""

    def init(self):
        return self.initWithRequestPath_(None)

    def initWithRequestPath_(self, request_path):
        self = objc.super(GutterglassSmokeStage, self).init()
        if self is None:
            return None
        self._request_path = Path(request_path).expanduser() if request_path else default_request_path()
        self._panel = None
        self._webview = None
        self._visible = False
        self._last_document = None
        return self

    def setup(self) -> None:
        if self._panel is not None:
            return
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
        panel.setTitle_("Gutterglass Smoke Stage")
        panel.setLevel_(28)
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.setBackgroundColor_(NSColor.colorWithWhite_alpha_(0.0, 0.88))
        panel.setIgnoresMouseEvents_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)

        webview = _make_webview(width, height)
        content_root = panel.contentView()
        if hasattr(webview, "setAutoresizingMask_"):
            webview.setAutoresizingMask_(_NSViewWidthSizable | _NSViewHeightSizable)
        content_root.addSubview_(webview)
        self._panel = panel
        self._webview = webview

    def show(self) -> bool:
        self.setup()
        if self._panel is None or self._webview is None:
            return False
        document = self._load_current_document()
        self._last_document = document
        if document is None:
            _load_html(
                self._webview,
                _render_status_html(
                    "No Gutterglass request",
                    "No source-signed smoke/artifact request is present.",
                    provenance="source: none · kind: none",
                ),
            )
        else:
            self._load_document(document)
        self._panel.orderFrontRegardless()
        self._visible = True
        return True

    def hide(self) -> None:
        if self._panel is not None:
            self._panel.orderOut_(None)
        self._visible = False

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def isVisible(self) -> bool:
        return bool(self._visible)

    def _load_current_document(self) -> GutterglassSmokeStageDocument | None:
        try:
            return load_gutterglass_request(self._request_path)
        except FileNotFoundError:
            logger.info("Gutterglass request absent at %s", self._request_path)
            return None
        except Exception as exc:
            logger.exception("Failed to load Gutterglass request")
            request = GutterglassSmokeStageRequest(
                source_sign="invalid-request",
                title="Invalid Gutterglass Request",
                content_kind="error",
                target_url="about:blank",
                created_at=time.time(),
                lifecycle="ephemeral",
                receipt_refs=(),
                summary=str(exc),
            )
            return GutterglassSmokeStageDocument(
                request=request,
                is_stale=False,
                status_message=f"Invalid Gutterglass request: {exc}",
            )

    def _load_document(self, document: GutterglassSmokeStageDocument) -> None:
        request = document.request
        if document.is_stale:
            _load_html(
                self._webview,
                _render_status_html(
                    f"Stale: {request.title}",
                    document.status_message,
                    provenance=request.provenance_label,
                ),
            )
            return

        if request.content_kind == "url":
            _load_url(self._webview, request.target_url)
            return

        if request.content_kind in {"html", "report"} and request.path is not None:
            _load_url(self._webview, request.target_url)
            return

        _load_html(self._webview, _render_artifact_html(request))


def _env_max_age_seconds() -> float:
    try:
        value = float(os.environ.get("SPOKE_GUTTERGLASS_STAGE_MAX_AGE_SECONDS", ""))
    except ValueError:
        return DEFAULT_MAX_AGE_SECONDS
    return value if value > 0.0 else DEFAULT_MAX_AGE_SECONDS


def _parse_created_at(value) -> float:
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return time.time()
    try:
        return float(text)
    except ValueError:
        pass
    normalized = text.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()


def _resolve_request_path(value, *, base_dir: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Gutterglass artifact path does not exist: {resolved}")
    return resolved


def _resolve_target_url(uri, path: Path | None) -> str:
    if uri is not None and str(uri).strip():
        text = str(uri).strip()
        parsed = urlparse(text)
        if parsed.scheme:
            return text
        return Path(text).expanduser().resolve().as_uri()
    if path is not None:
        return path.as_uri()
    return ""


def _default_panel_rect(frame) -> tuple[float, float, float, float]:
    width = min(_DEFAULT_WIDTH, max(360.0, float(frame.size.width) - (2.0 * _MIN_MARGIN)))
    height = min(_DEFAULT_HEIGHT, max(260.0, float(frame.size.height) - (2.0 * _MIN_MARGIN)))
    x = float(frame.origin.x) + (float(frame.size.width) - width) / 2.0
    y = float(frame.origin.y) + (float(frame.size.height) - height) / 2.0
    return x, y, width, height


def _make_webview(width: float, height: float):
    try:
        from WebKit import WKWebView

        return WKWebView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
    except Exception:
        logger.warning("WKWebView unavailable, using AppKit fallback", exc_info=True)
        fallback = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        return fallback


def _load_url(webview, url: str) -> None:
    load_request = getattr(webview, "loadRequest_", None)
    if callable(load_request):
        load_request(_make_url_request(url))
        return
    _load_html(webview, _render_status_html("Open externally", url, provenance="url fallback"))


def _make_url_request(url: str):
    try:
        from Foundation import NSURL, NSURLRequest

        ns_url = NSURL.URLWithString_(url)
        return NSURLRequest.requestWithURL_(ns_url)
    except Exception:
        return _URLRequest(url)


def _load_html(webview, source: str) -> None:
    loader = getattr(webview, "loadHTMLString_baseURL_", None)
    if callable(loader):
        loader(source, None)


def _render_artifact_html(request: GutterglassSmokeStageRequest) -> str:
    body = ""
    if request.path is not None and request.content_kind in {"text", "markdown", "json"}:
        body = f"<pre>{html.escape(request.path.read_text(encoding='utf-8', errors='replace'))}</pre>"
    elif request.content_kind in {"image", "filmstrip"}:
        body = f'<img class="artifact-image" src="{html.escape(request.target_url)}" alt="">'
    elif request.content_kind == "directory":
        body = f"<pre>{html.escape(_render_directory_listing(request.path))}</pre>"
    else:
        body = f"<p>{html.escape(request.summary or request.target_url)}</p>"
    return _wrap_html(request.title, body, provenance=request.provenance_label)


def _render_directory_listing(path: Path | None) -> str:
    if path is None:
        return "No directory path supplied."
    return "\n".join(sorted(child.name for child in path.iterdir()))


def _render_status_html(title: str, message: str, *, provenance: str) -> str:
    body = f"<p>{html.escape(message)}</p>"
    return _wrap_html(title, body, provenance=provenance)


def _wrap_html(title: str, body: str, *, provenance: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0;
    min-height: 100vh;
    box-sizing: border-box;
    padding: 28px 34px;
    background: #101214;
    color: #eceff3;
    font: 16px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
  }}
  header {{
    margin-bottom: 24px;
    color: #9ea8b3;
    font-size: 12px;
    letter-spacing: 0;
    text-transform: none;
  }}
  h1 {{
    margin: 0 0 8px;
    color: #f6f7f9;
    font-size: 22px;
    line-height: 1.2;
  }}
  pre {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.42;
  }}
  .artifact-image {{
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 120px);
    object-fit: contain;
  }}
</style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div>{html.escape(provenance)}</div>
  </header>
  <main>{body}</main>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spoke-gutterglass-stage",
        description="Publish source-signed artifact requests for Spoke's Gutterglass Smoke Stage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish", help="write a latest smoke-stage request")
    publish.add_argument("--request-path", default=None)
    publish.add_argument("--source-sign", required=True)
    publish.add_argument("--title", required=True)
    publish.add_argument("--kind", required=True, choices=sorted(_SUPPORTED_KINDS))
    publish.add_argument("--target", required=True)
    publish.add_argument("--receipt-ref", action="append", default=[])
    publish.add_argument("--lifecycle", default="ephemeral")
    publish.add_argument("--summary", default="")
    publish.add_argument("--authority", default="none")

    args = parser.parse_args(argv)
    if args.command == "publish":
        written = write_gutterglass_request(
            args.request_path,
            source_sign=args.source_sign,
            title=args.title,
            content_kind=args.kind,
            target=args.target,
            receipt_refs=tuple(args.receipt_ref),
            lifecycle=args.lifecycle,
            summary=args.summary,
            authority=args.authority,
        )
        print(written)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
