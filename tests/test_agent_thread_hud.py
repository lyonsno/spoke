"""Tests for compositor-facing Agent Shell partyline HUD render data."""

from __future__ import annotations


def test_hud_surfaces_preserve_selected_resting_response_and_compact_inactive_cards():
    from spoke.agent_thread_hud import build_agent_thread_hud

    cards = [
        {
            "provider_session_id": "codex-thread-1",
            "thread_id": "codex-thread-1",
            "provider": "codex",
            "title": "active thread",
            "readiness": "ready",
            "bearing": "Session: active work",
            "activity_line": "Ready to read",
            "latest_response": "The real selected response stays in the transcript.",
            "selected": True,
        },
        {
            "provider_session_id": "codex-thread-2",
            "thread_id": "codex-thread-2",
            "provider": "codex",
            "title": "inactive thread",
            "readiness": "working",
            "bearing": "Session: inactive work\nImmediate next step: run tests",
            "activity_line": "Running focused tests",
            "latest_response": "Hidden inactive response.",
            "selected": False,
        },
    ]

    hud = build_agent_thread_hud(
        cards,
        content_width_points=640.0,
        content_height_points=180.0,
    )

    assert hud["surface_kind"] == "agent_shell_partyline"
    assert hud["selected_thread_id"] == "codex-thread-1"
    assert hud["main_transcript"] == {
        "thread_id": "codex-thread-1",
        "show_latest_response": True,
        "text": "The real selected response stays in the transcript.",
    }
    assert [surface["thread_id"] for surface in hud["cards"]] == [
        "codex-thread-2",
        "codex-thread-1",
    ]
    inactive = hud["cards"][0]
    selected = hud["cards"][1]
    assert inactive["role"] == "inactive_card"
    assert inactive["text"] == "inactive thread · Running focused tests"
    assert inactive["show_latest_response"] is False
    assert inactive["latest_response"] == ""
    assert inactive["frame"]["x"] < selected["frame"]["x"]
    assert selected["role"] == "selected_summary"
    assert selected["show_latest_response"] is False
    assert selected["text"] == "active thread · Ready to read"


def test_hud_surfaces_are_deterministically_bounded_and_clipped():
    from spoke.agent_thread_hud import build_agent_thread_hud

    cards = [
        {
            "provider_session_id": f"codex-thread-{index}",
            "thread_id": f"codex-thread-{index}",
            "provider": "codex",
            "title": f"thread {index}",
            "readiness": "ready",
            "bearing": "bearing",
            "activity_line": "Ready to read",
            "latest_response": f"response {index}",
            "selected": index == 0,
        }
        for index in range(8)
    ]

    hud = build_agent_thread_hud(
        cards,
        content_width_points=420.0,
        content_height_points=140.0,
    )

    assert len(hud["cards"]) == 4
    frames = [surface["frame"] for surface in hud["cards"]]
    assert all(frame["width"] >= 92.0 for frame in frames)
    assert all(frame["height"] == 44.0 for frame in frames)
    assert all(frame["x"] + frame["width"] <= 420.0 for frame in frames)
    assert all(frame["y"] + frame["height"] <= 140.0 for frame in frames)
    assert [surface["thread_id"] for surface in hud["cards"]] == [
        "codex-thread-1",
        "codex-thread-2",
        "codex-thread-3",
        "codex-thread-0",
    ]


def test_hud_surfaces_receive_initial_placement_geometry_for_renderer_consumption():
    from spoke.agent_thread_hud import build_agent_thread_hud

    cards = [
        {
            "provider_session_id": f"codex-thread-{index}",
            "thread_id": f"codex-thread-{index}",
            "provider": "codex",
            "title": f"thread {index}",
            "readiness": "ready",
            "bearing": "bearing",
            "activity_line": "Ready to read",
            "latest_response": f"response {index}",
            "selected": index == 0,
        }
        for index in range(3)
    ]

    hud = build_agent_thread_hud(
        cards,
        content_width_points=420.0,
        content_height_points=140.0,
    )

    assert [surface["geometry"]["anchor"] for surface in hud["cards"]] == [
        "bottom",
        "bottom",
        "right",
    ]
    assert [surface["geometry"]["priority"] for surface in hud["cards"]] == [
        100,
        101,
        0,
    ]
