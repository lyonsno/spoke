from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from spoke.greenroom_footer import GreenroomAttentionReader, attention_snapshot_path


def snapshot():
    return {
        "schema": "gpu-greenroom.operator-attention.v1",
        "observed_at": "2026-11-06T12:00:00Z",
        "valid_until": "2026-11-06T12:01:00Z",
        "monitor_url": "http://localhost:8768/",
        "queue": {"running": 1, "waiting": 3},
        "items": [
            {"id": str(i), "title": f"Request {i}", "status": "pending",
             "target_url": f"http://localhost:8768/#attention-{i}"}
            for i in range(3)
        ],
        "future_additive_field": {"allowed": True},
    }


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def test_projection_cache_and_producer_deadline(tmp_path, monkeypatch):
    path = tmp_path / "operator-attention.json"
    path.write_text(json.dumps(snapshot()))
    reader = GreenroomAttentionReader(path)
    original = Path.read_text
    reads = []
    def read(p, *args, **kwargs):
        reads.append(p)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    now = instant("2026-11-06T12:00:10Z")
    current = reader.read(now)
    assert current.queue_text == "Greenroom · 1 running · 3 waiting"
    assert current.attention_text == "⚠ 3 · Request 0"
    assert current.attention_url.endswith("#attention-0")
    assert reader.read(now + 10) == current
    assert len(reads) == 1
    expired = reader.read(now + 50)
    assert expired.freshness == "stale"
    assert expired.attention_url is None
    assert expired.monitor_url == current.monitor_url
    assert len(reads) == 1, "deadline checks must not reparse unchanged files"
    newer = snapshot()
    newer["items"] = []
    path.write_text(json.dumps(newer))
    assert reader.read(now).attention_text == "✓ No requests"
    assert len(reads) == 2
    path.unlink()
    absent = reader.read(now)
    assert absent.freshness == "unavailable"
    assert absent.monitor_url is None and absent.attention_url is None


@pytest.mark.parametrize("case", ["schema", "items", "queue", "count", "timestamp", "timezone", "deadline", "item", "duplicate", "unsafe-url", "blank"])
def test_invalid_snapshot_never_reuses_current_attention(tmp_path, case):
    path = tmp_path / "attention.json"
    payload = snapshot()
    path.write_text(json.dumps(payload))
    reader = GreenroomAttentionReader(path)
    now = instant("2026-11-06T12:00:10Z")
    assert reader.read(now).attention_url
    invalid = copy.deepcopy(payload)
    if case == "schema": invalid["schema"] = "v2"
    elif case == "items": invalid["items"] = None
    elif case == "queue": invalid.pop("queue")
    elif case == "count": invalid["queue"]["running"] = True
    elif case == "timestamp": invalid["observed_at"] = 0
    elif case == "timezone": invalid["observed_at"] = "2026-11-06T12:00:00"
    elif case == "deadline": invalid["valid_until"] = "2026-11-06T11:00:00Z"
    elif case == "item": invalid["items"][1].pop("id")
    elif case == "duplicate": invalid["items"][1]["id"] = "0"
    elif case == "unsafe-url": invalid["items"][0]["target_url"] = "file:///tmp/action"
    path.write_text("" if case == "blank" else json.dumps(invalid))
    footer = reader.read(now)
    assert footer.freshness == "unavailable"
    assert footer.attention_url is None and footer.monitor_url is None


def test_path_override_and_clock_mismatch(tmp_path, monkeypatch):
    monkeypatch.delenv("SPOKE_GREENROOM_ATTENTION_PATH", raising=False)
    monkeypatch.setenv("GPU_GREENROOM_DIR", str(tmp_path))
    assert attention_snapshot_path() == tmp_path / "operator-attention.json"
    monkeypatch.setenv("SPOKE_GREENROOM_ATTENTION_PATH", str(tmp_path / "override.json"))
    assert attention_snapshot_path().name == "override.json"
    assert attention_snapshot_path(tmp_path / "explicit.json").name == "explicit.json"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(snapshot()))
    footer = GreenroomAttentionReader(path).read(instant("2026-11-06T11:59:59Z"))
    assert footer.freshness == "clock mismatch"
    assert footer.attention_url is None


def test_native_footer_is_pinned_below_lane_scroll():
    if sys.platform != "darwin":
        pytest.skip("native sidebar")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent('''
        from AppKit import NSApplication
        import spoke.wezterm_lanes_sidebar as sidebar
        NSApplication.sharedApplication()
        hud = sidebar.build_sidebar_hud_controller()
        hud.setup()
        try:
            labels = [str(v.stringValue()) for v in hud._foreground.subviews() if hasattr(v, "stringValue")]
            buttons = [str(v.title()) for v in hud._foreground.subviews() if hasattr(v, "title")]
            assert any("Greenroom" in s for s in labels + buttons), "pinned footer has no Greenroom status"
            assert hud._greenroom_button.superview() == hud._foreground
            assert hud._attention_button.superview() == hud._foreground
            assert hud._scroll_view.frame().origin.y >= hud._greenroom_button.frame().origin.y + hud._greenroom_button.frame().size.height
        finally:
            hud.cleanup()
    ''')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_footer_click_routes_current_item_without_browser(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("native sidebar")
    path = tmp_path / "native.json"
    raw = snapshot()
    # Actual wall clock belongs to this host-native exercise, not the model clock.
    from datetime import timedelta, timezone
    now = datetime.now(timezone.utc)
    raw.update(observed_at=now.isoformat(), valid_until=(now + timedelta(minutes=10)).isoformat())
    path.write_text(json.dumps(raw))
    script = '''
        import json, sys
        import AppKit
        import spoke.wezterm_lanes_sidebar as sidebar
        from pathlib import Path
        AppKit.NSApplication.sharedApplication()
        opened = []
        class Workspace:
            @classmethod
            def sharedWorkspace(cls): return cls()
            def openURL_(self, url): opened.append(str(url.absoluteString())); return True
        AppKit.NSWorkspace = Workspace
        path = Path(sys.argv[1])
        hud = sidebar.build_sidebar_hud_controller(path)
        hud.setup()
        try:
            assert hud._attention_button.isEnabled()
            hud._attention_button.performClick_(None)
            assert opened == ["http://localhost:8768/#attention-0"], opened
            hud._greenroom_button.performClick_(None)
            assert opened[-1] == "http://localhost:8768/"
            # Resolution between presentation and click must not open an old item.
            raw = json.loads(path.read_text()); raw["items"] = []; path.write_text(json.dumps(raw))
            before = len(opened)
            hud._attention_button.performClick_(None)
            assert len(opened) == before
            assert hud._attention_button.title() == "✓ No requests"
            assert not hud._attention_button.isEnabled()
            for height in (428., 1054.):
                sidebar.query_wezterm_window_bounds = lambda: {"X":0., "Y":0., "Width":900., "Height":height + 43.}
                hud._sync_geometry()
                hud._render_lanes([sidebar.LaneRow(i,i,str(i),f"lane-{i}","done","",i==35) for i in range(1,36)])
                assert hud._greenroom_button.frame().origin.y == 48.
                assert hud._scroll_view.frame().origin.y == 72.
        finally:
            hud.cleanup()
    '''
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field", ["monitor_url", "item"])
@pytest.mark.parametrize("url", ["http://localhost:bad/", "http://local host/", "http://localhost:65536/"])
def test_malformed_web_urls_clear_navigation(tmp_path, field, url):
    raw = snapshot()
    if field == "monitor_url": raw[field] = url
    else: raw["items"][0]["target_url"] = url
    path = tmp_path / "malformed-url.json"
    path.write_text(json.dumps(raw))
    footer = GreenroomAttentionReader(path).read(instant("2026-11-06T12:00:10Z"))
    assert footer.freshness == "unavailable", "malformed URL is still actionable"
    assert footer.monitor_url is None and footer.attention_url is None


@pytest.mark.parametrize("selection", ["cli", "path-env", "queue-env"])
def test_daemon_keeps_caller_relative_snapshot_path(tmp_path, monkeypatch, selection):
    import spoke.wezterm_lanes_sidebar as sidebar
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sidebar, "DAEMON_LOCK_PATH", tmp_path / "helper.lock")
    monkeypatch.delenv("SPOKE_GREENROOM_ATTENTION_PATH", raising=False)
    expected = tmp_path / "relative" / "operator-attention.json"
    if selection == "cli":
        selected = "relative/operator-attention.json"
    elif selection == "path-env":
        selected = None
        monkeypatch.setenv("SPOKE_GREENROOM_ATTENTION_PATH", "relative/operator-attention.json")
    else:
        selected = None
        monkeypatch.setenv("GPU_GREENROOM_DIR", "relative")
    calls = []
    monkeypatch.setattr(sidebar.subprocess, "Popen", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    sidebar.ensure_daemon_running(selected)
    command, options = calls[0]
    assert "--greenroom-attention-path" in command, "spawn lost caller's snapshot selection"
    actual = command[command.index("--greenroom-attention-path") + 1]
    assert Path(actual).is_absolute(), "daemon changes cwd before resolving caller's relative path"
    assert Path(actual) == expected


def test_native_queue_text_contrast_survives_light_appearance(tmp_path):
    if sys.platform != "darwin": pytest.skip("native sidebar raster")
    script = '''
        from AppKit import NSApplication, NSAppearance
        from Foundation import NSMakeRect
        import spoke.wezterm_lanes_sidebar as sidebar
        app=NSApplication.sharedApplication()
        app.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameAqua"))
        hud=sidebar.build_sidebar_hud_controller()
        hud.setup()
        try:
            hud._apply_planes({"phase":"rest","height_frac":1.,"width_frac":1.,"opacity":1.,"text_alpha":1.})
            # Both enabled and unavailable/disabled queue text must be readable.
            for enabled in (True,False):
                hud._greenroom_button.setEnabled_(enabled)
                rect=NSMakeRect(10.,48.,232.,20.)
                view=hud._panel.contentView()
                rep=view.bitmapImageRepForCachingDisplayInRect_(rect)
                view.cacheDisplayInRect_toBitmapImageRep_(rect,rep)
                bright=0
                for y in range(rep.pixelsHigh()):
                    for x in range(rep.pixelsWide()):
                        color=rep.colorAtX_y_(x,y)
                        if min(color.redComponent(),color.greenComponent(),color.blueComponent())>0.4: bright+=1
                assert bright>40, f"queue status disappears against dark sidebar in Aqua; enabled={enabled}, bright pixels={bright}"
        finally: hud.cleanup()
    '''
    result=subprocess.run([sys.executable,"-c",textwrap.dedent(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
