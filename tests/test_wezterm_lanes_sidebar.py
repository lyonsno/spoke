from __future__ import annotations

import json
from pathlib import Path

import pytest

import spoke.wezterm_lanes_sidebar as sidebar_mod
from spoke.wezterm_lanes_sidebar import (
    LaneRow,
    SidebarHouseTransition,
    activate_wezterm_lane,
    compute_sidebar_frame_for_wezterm_bounds,
    is_wezterm_sidebar_enabled,
    load_wezterm_lanes_state,
)


def test_load_lanes_state_and_gutter_geometry_and_house_transition(tmp_path: Path) -> None:
    state_file = tmp_path / "wezterm-lanes-state.json"
    state_file.write_text(
        json.dumps(
            {
                "updated_at": 1793848000,
                "lanes": [
                    {
                        "index": 1,
                        "tab_id": 11,
                        "pane_id": "12",
                        "name": "Thomas the jank engine",
                        "state": "working",
                        "badge": "2t",
                        "active": False,
                    },
                    {
                        "index": 23,
                        "tab_id": 7,
                        "pane_id": "7",
                        "name": "pimp-my-wezterm",
                        "state": "done",
                        "badge": "",
                        "active": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    lanes, updated = load_wezterm_lanes_state(state_file)
    assert updated == 1793848000
    assert len(lanes) == 2
    assert lanes[0] == LaneRow(
        index=1,
        tab_id=11,
        pane_id="12",
        name="Thomas the jank engine",
        state="working",
        badge="2t",
        active=False,
    )
    assert lanes[1].active is True

    # Partial/empty read under load retains last known lanes
    state_file.write_text("", encoding="utf-8")
    lanes_after, updated_after = load_wezterm_lanes_state(state_file)
    assert len(lanes_after) == 2
    assert updated_after == 1793848000

    enabled_file = tmp_path / "wezterm-sidebar-enabled"
    enabled_file.write_text("1\n", encoding="utf-8")
    assert is_wezterm_sidebar_enabled(enabled_file) is True
    enabled_file.write_text("0\n", encoding="utf-8")
    assert is_wezterm_sidebar_enabled(enabled_file) is False

    # Quartz top-left bounds -> AppKit bottom-left frame inside the 252pt WezTerm left padding gutter
    # Windowed below 38pt menu bar: clears 25pt WezTerm top tab viewer bar
    x, y, w, h = compute_sidebar_frame_for_wezterm_bounds(
        {"X": 0.0, "Y": 38.0, "Width": 1708.0, "Height": 1056.0},
        screen_height=1117.0,
        width_pt=252.0,
    )
    assert (x, y, w, h) == (0.0, 23.0, 252.0, 1031.0)

    # Full-height Y=0 window: clears both top margin + 25pt WezTerm top tab viewer bar (43pt total)
    x0, y0, w0, h0 = compute_sidebar_frame_for_wezterm_bounds(
        {"X": 0.0, "Y": 0.0, "Width": 1728.0, "Height": 1117.0},
        screen_height=1117.0,
        width_pt=252.0,
    )
    assert (x0, y0, w0, h0) == (0.0, 0.0, 252.0, 1074.0)

    # House choreography summon -> rest -> dismiss
    tr = SidebarHouseTransition()
    tr.summon(100.0)
    mid = tr.sample(100.15)
    assert mid["phase"] == "summon"
    assert 0.0 < float(mid["height_frac"]) <= 1.0
    rest = tr.sample(102.0)
    assert rest["phase"] == "rest"
    assert float(rest["height_frac"]) == 1.0
    assert float(rest["text_alpha"]) == 1.0

    tr.dismiss(103.0)
    closing = tr.sample(103.1)
    assert closing["phase"] == "dismiss"
    assert float(closing["height_frac"]) < 1.0
    hidden = tr.sample(105.0)
    assert hidden["phase"] == "hidden"
    assert float(hidden["opacity"]) == 0.0

    # Activation calls only activate-pane, never split-pane
    recorded: list[list[str]] = []
    orig_run = getattr(sidebar_mod, "_run_subprocess", None)
    try:
        def fake_run(cmd: list[str]) -> int:
            recorded.append(list(cmd))
            return 0

        sidebar_mod._run_subprocess = fake_run  # type: ignore[attr-defined]
        assert activate_wezterm_lane("12", sync=True) is True
    finally:
        if orig_run is not None:
            sidebar_mod._run_subprocess = orig_run  # type: ignore[attr-defined]

    assert len(recorded) == 1
    assert recorded[0][-3:] == ["activate-pane", "--pane-id", "12"]
    assert not any("split-pane" in part for part in recorded[0])


def test_window_bounds_follow_frontmost_wezterm_window(monkeypatch) -> None:
    import sys
    import types

    small = {"X": 606.0, "Y": 224.0, "Width": 574.0, "Height": 453.0}
    large = {"X": 0.0, "Y": 0.0, "Width": 1728.0, "Height": 1117.0}
    windows = [
        {"kCGWindowOwnerName": "Spoke", "kCGWindowLayer": 0, "kCGWindowBounds": large},
        {"kCGWindowOwnerName": "WezTerm", "kCGWindowLayer": 3, "kCGWindowBounds": large},
        {"kCGWindowOwnerName": "WezTerm", "kCGWindowLayer": 0, "kCGWindowBounds": small},
        {"kCGWindowOwnerName": "WezTerm", "kCGWindowLayer": 0, "kCGWindowBounds": large},
    ]
    monkeypatch.setitem(sys.modules, "Quartz", types.SimpleNamespace(
        kCGWindowListOptionOnScreenOnly=1,
        kCGWindowListExcludeDesktopElements=2,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_args: windows,
    ))
    assert sidebar_mod.query_wezterm_window_bounds() == small


def test_native_mask_and_scroll_follow_small_then_large_window(tmp_path: Path) -> None:
    import subprocess
    import sys
    import textwrap
    import pytest
    if sys.platform != "darwin":
        pytest.skip("native AppKit/CoreAnimation contract")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import spoke.wezterm_lanes_sidebar as sidebar
        from AppKit import NSApplication
        from Quartz import CGPathGetBoundingBox
        NSApplication.sharedApplication()
        hud = sidebar.build_sidebar_hud_controller()
        hud.setup()
        try:
            sidebar.query_wezterm_window_bounds = lambda: {"X":606.,"Y":224.,"Width":918.,"Height":453.}
            assert hud._sync_geometry()
            hud._apply_planes({"phase":"rest","height_frac":1.,"width_frac":1.,"opacity":1.,"text_alpha":1.})
            lanes = [sidebar.LaneRow(i,i,str(i),f"lane-{i}","done","",i==29) for i in range(1,32)]
            hud._render_lanes(lanes)
            sidebar.query_wezterm_window_bounds = lambda: {"X":0.,"Y":38.,"Width":1728.,"Height":1079.}
            assert hud._sync_geometry()
            hud._render_lanes(lanes)
            mask = CGPathGetBoundingBox(hud._text_mask.path())
            height = hud._foreground.bounds().size.height
            assert abs(mask.size.height-height)<0.01, f"stale mask clips upper rows: mask={mask.size.height}, foreground={height}"
            assert hud._header_title.stringValue()=="LANES (31)"
            assert abs(hud._scroll_view.contentView().bounds().origin.y)<0.01, "fitting list retained a scrolled offset"
        finally:
            hud.cleanup()
    """)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_focused_snapshot_visibility_overrides_legacy_global_flag(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    legacy = tmp_path / "enabled"
    legacy.write_text("1\n")
    monkeypatch.setattr(sidebar_mod, "DEFAULT_STATE_PATH", state)
    monkeypatch.setattr(sidebar_mod, "DEFAULT_ENABLED_PATH", legacy)
    payload = {"window_id":"1", "sidebar_enabled":False, "lanes":[{"index":1,"pane_id":"101","name":"secondary"}]}
    state.write_text(json.dumps(payload))
    assert sidebar_mod.is_wezterm_sidebar_enabled() is False
    payload.update(window_id="0", sidebar_enabled=True)
    legacy.write_text("0\n")
    state.write_text(json.dumps(payload))
    assert sidebar_mod.is_wezterm_sidebar_enabled() is True


def test_invalid_window_visibility_does_not_restore_global_visibility(tmp_path, monkeypatch, caplog):
    state=tmp_path/"invalid-visibility.json"
    enabled=tmp_path/"global-enabled"
    enabled.write_text("1\n")
    monkeypatch.setattr(sidebar_mod,"DEFAULT_STATE_PATH",state)
    monkeypatch.setattr(sidebar_mod,"DEFAULT_ENABLED_PATH",enabled)
    state.write_text(json.dumps({"window_id":"1","sidebar_enabled":"false","lanes":[{"pane_id":"101"}]}))
    assert sidebar_mod.is_wezterm_sidebar_enabled() is False
    assert "sidebar_enabled" in caplog.text


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("row_shape", ["missing", "null", "object", "bad-index"])
@pytest.mark.parametrize("scoped", [False, "false"])
def test_scoped_visibility_is_checked_before_malformed_rows(tmp_path, monkeypatch, warm, row_shape, scoped):
    state=tmp_path/"malformed-rows.json"
    legacy=tmp_path/"enabled"
    legacy.write_text("1\n")
    monkeypatch.setattr(sidebar_mod,"DEFAULT_STATE_PATH",state)
    monkeypatch.setattr(sidebar_mod,"DEFAULT_ENABLED_PATH",legacy)
    if warm:
        state.write_text(json.dumps({"updated_at":3,"window_id":"0","sidebar_enabled":True,"lanes":[{"pane_id":"1"}]}))
        assert sidebar_mod.is_wezterm_sidebar_enabled() is True
    raw={"window_id":"1","sidebar_enabled":scoped}
    if row_shape!="missing":raw["lanes"] = ([{"index":[],"pane_id":"2"}] if row_shape=="bad-index" else None if row_shape=="null" else {})
    state.write_text(json.dumps(raw))
    assert sidebar_mod.is_wezterm_sidebar_enabled() is False
    rows,updated=sidebar_mod.load_wezterm_lanes_state()
    assert len(rows)==(1 if warm else 0), "retain row evidence independently of visibility"
    if warm:assert rows[0].pane_id=="1" and updated==3
