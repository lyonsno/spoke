"""Tests for Agent Shell primitive card render payloads."""

from __future__ import annotations


def _primitive(
    primitive_id: str,
    *,
    selected: bool = False,
    priority: int = 0,
    show_latest_response: bool = False,
) -> dict:
    return {
        "id": primitive_id,
        "kind": "selected_thread" if selected else "thread_card",
        "provider": "codex",
        "provider_session_id": primitive_id,
        "selected": selected,
        "readiness": "ready" if not selected else "working",
        "title": f"title {primitive_id}",
        "bearing": f"bearing {primitive_id}",
        "activity_line": f"activity {primitive_id}",
        "latest_response": f"latest response {primitive_id}",
        "display": {
            "display_state": "selected" if selected else "inactive",
            "primary_text": f"primary {primitive_id}",
            "secondary_text": f"secondary {primitive_id}",
            "show_latest_response": show_latest_response,
        },
        "geometry": {
            "anchor": "top",
            "priority": priority,
            "preferred_width": 156.0,
            "preferred_height": 48.0 if not selected else 64.0,
            "min_width": 96.0,
            "min_height": 42.0,
        },
        "material": {
            "style": "thread_card",
            "prominence": "selected" if selected else "inactive",
            "optical_displacement": 0.18 if selected else 0.06,
            "corner_radius": 10.0,
        },
    }


def test_card_renderer_consumes_primitives_without_reinterpreting_display_contract():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    primitives = [
        _primitive("codex-inactive-2", priority=20),
        _primitive("codex-selected", selected=True, priority=10, show_latest_response=True),
        _primitive("codex-inactive-1", priority=5),
    ]

    payload = build_agent_shell_card_render_payload(
        primitives,
        content_width_points=460.0,
        content_height_points=180.0,
    )

    assert payload["surface_kind"] == "agent_shell_card_primitives"
    assert payload["selected_primitive_id"] == "codex-selected"
    assert payload["main_transcript"] == {
        "primitive_id": "codex-selected",
        "show_latest_response": True,
        "text": "latest response codex-selected",
    }
    assert [surface["primitive_id"] for surface in payload["cards"]] == [
        "codex-inactive-1",
        "codex-inactive-2",
        "codex-selected",
    ]
    inactive = payload["cards"][0]
    selected = payload["cards"][-1]
    assert inactive["selected"] is False
    assert inactive["readiness"] == "ready"
    assert inactive["show_latest_response"] is False
    assert inactive["latest_response"] == ""
    assert inactive["primary_text"] == "primary codex-inactive-1"
    assert selected["selected"] is True
    assert selected["readiness"] == "working"
    assert selected["material"]["prominence"] == "selected"


def test_card_renderer_bounds_cards_above_transcript_without_overlap():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    primitives = [_primitive(f"thread-{index}", priority=index) for index in range(7)]
    primitives.append(_primitive("selected", selected=True, priority=100))

    payload = build_agent_shell_card_render_payload(
        primitives,
        content_width_points=360.0,
        content_height_points=154.0,
    )

    assert len(payload["cards"]) == 3
    assert [surface["primitive_id"] for surface in payload["cards"]] == [
        "thread-0",
        "thread-1",
        "selected",
    ]
    transcript = payload["transcript_frame"]
    assert transcript["height"] > 0.0
    for surface in payload["cards"]:
        frame = surface["frame"]
        assert frame["x"] >= 12.0
        assert frame["x"] + frame["width"] <= 360.0
        assert frame["y"] + frame["height"] <= 154.0
        assert surface["clip"] is True
        assert surface["material"]["style"] in {"thread_card", "quiet_chip"}
        assert transcript["y"] + transcript["height"] <= frame["y"]


def test_card_renderer_builds_optical_field_requests_from_rendered_cards():
    from spoke.agent_shell_card_renderer import (
        build_agent_shell_card_optical_field_payload,
        build_agent_shell_card_render_payload,
    )

    primitives = [
        _primitive("codex-inactive", priority=5),
        _primitive("codex-selected", selected=True, show_latest_response=True),
    ]
    payload = build_agent_shell_card_render_payload(
        primitives,
        content_width_points=420.0,
        content_height_points=160.0,
    )

    optical = build_agent_shell_card_optical_field_payload(payload)

    assert optical["surface_kind"] == "agent_shell_card_optical_fields"
    requests = optical["requests"]
    assert [request["caller_id"] for request in requests] == [
        "agent.card.codex-inactive",
        "agent.card.codex-selected",
    ]
    inactive, selected = requests
    assert inactive["profile"] == "agent_card"
    assert inactive["role"] == "agent_card"
    assert inactive["compiled_shell_config"]["optical_field"]["profile"] == "agent_card"
    assert inactive["compiled_shell_config"]["center_x"] == inactive["bounds"]["x"] + inactive["bounds"]["width"] * 0.5
    assert selected["profile"] == "agent_card"
    assert selected["role"] == "selected_thread"
    assert selected["z_index"] > inactive["z_index"]
    assert selected["disturbances"][0]["kind"] == "readiness_pulse"
    assert selected["compiled_shell_config"]["optical_field"]["disturbances"] == (
        "readiness.codex-selected",
    )
