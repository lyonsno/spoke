from __future__ import annotations

import json
from pathlib import Path

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
