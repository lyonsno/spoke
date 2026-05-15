"""Tests for Agent Shell primitive card render payloads."""

from __future__ import annotations


def _primitive(
    primitive_id: str,
    *,
    selected: bool = False,
    priority: int = 0,
    show_latest_response: bool = False,
    primary_text: str | None = None,
    secondary_text: str | None = None,
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
            "primary_text": primary_text or f"primary {primitive_id}",
            "secondary_text": secondary_text or f"secondary {primitive_id}",
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
    ]
    inactive = payload["cards"][-1]
    assert inactive["selected"] is False
    assert inactive["readiness"] == "ready"


def test_selected_resting_response_stays_in_main_transcript_not_duplicate_card():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    payload = build_agent_shell_card_render_payload(
        [
            _primitive(
                "codex-inactive",
                priority=5,
                primary_text="Codex idle lane · Ready to read",
                secondary_text="",
            ),
            _primitive(
                "codex-selected",
                selected=True,
                show_latest_response=True,
                primary_text="The selected thread response is already in the transcript.",
                secondary_text="selected resting · Ready",
            ),
        ],
        content_width_points=720.0,
        content_height_points=260.0,
    )

    assert payload["main_transcript"] == {
        "primitive_id": "codex-selected",
        "show_latest_response": True,
        "text": "latest response codex-selected",
    }
    assert [card["primitive_id"] for card in payload["cards"]] == ["codex-inactive"]


def test_card_renderer_places_sibling_surfaces_inside_visible_content_bounds():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    primitives = [_primitive(f"thread-{index}", priority=index) for index in range(7)]
    primitives.append(_primitive("selected", selected=True, priority=100))

    payload = build_agent_shell_card_render_payload(
        primitives,
        content_width_points=360.0,
        content_height_points=154.0,
    )

    assert len(payload["cards"]) == 1
    assert [surface["primitive_id"] for surface in payload["cards"]] == [
        "selected",
    ]
    transcript = payload["transcript_frame"]
    assert transcript == {
        "x": 12.0,
        "y": 12.0,
        "width": 336.0,
        "height": 130.0,
    }
    for surface in payload["cards"]:
        frame = surface["frame"]
        assert 0.0 <= frame["x"] <= 360.0 - frame["width"]
        assert 0.0 <= frame["y"] <= 154.0 - frame["height"]
        assert surface["clip"] is True
        assert surface["movable"] is True
        assert surface["surface_attachment"] == "sibling"
        assert surface["material"]["style"] in {"thread_card", "quiet_chip"}
    selected = payload["cards"][-1]
    assert selected["frame"]["width"] >= 1.0
    assert selected["frame"]["height"] >= 1.0


def test_card_renderer_keeps_placeholder_cards_readable_and_non_overlapping():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    primitives = [
        _primitive(
            f"thread-{index}",
            priority=index,
            primary_text=f"Readable thread card {index} with enough text",
            secondary_text=f"ready · branch-{index}",
        )
        for index in range(6)
    ]
    primitives.append(
        _primitive(
            "selected",
            selected=True,
            priority=100,
            primary_text="Selected agent thread with real visible response text",
            secondary_text="working · gpt-5.5",
        )
    )

    payload = build_agent_shell_card_render_payload(
        primitives,
        content_width_points=720.0,
        content_height_points=260.0,
    )

    assert len(payload["cards"]) >= 2
    frames = [surface["frame"] for surface in payload["cards"]]
    assert all(frame["width"] >= 300.0 for frame in frames)
    assert all(frame["height"] >= 72.0 for frame in frames)
    assert len({(frame["x"], frame["y"]) for frame in frames}) == len(frames)


def test_selected_placeholder_card_stays_readable_when_parent_body_is_short():
    from spoke.agent_shell_card_renderer import build_agent_shell_card_render_payload

    payload = build_agent_shell_card_render_payload(
        [
            _primitive(
                "selected",
                selected=True,
                primary_text="Selected agent thread with readable smoke text",
                secondary_text="working · claude-opus",
            ),
        ],
        content_width_points=600.0,
        content_height_points=80.0,
    )

    selected = payload["cards"][0]
    assert selected["frame"]["width"] >= 420.0
    assert selected["frame"]["height"] >= 120.0


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
    ]
    (inactive,) = requests
    assert inactive["profile"] == "agent_card"
    assert inactive["role"] == "agent_card"
    assert inactive["presentation_layer"] == "agent_card"
    assert inactive["layout_recipe"] == "agent-thread-card"
    assert inactive["visibility_scope"] == "independent"
    assert inactive["z_index"] < 200
    assert inactive["disturbances"] == []
    assert inactive["compiled_shell_config"]["optical_field"]["bounds"] == inactive["bounds"]
    assert inactive["compiled_shell_config"]["optical_field"]["content_frame"] == inactive["bounds"]
    assert inactive["compiled_shell_config"]["presentation_layer"] == "agent_card"
    assert inactive["compiled_shell_config"]["visibility_scope"] == "independent"
    assert inactive["compiled_shell_config"]["gpu_material_enabled"] == 1.0


def test_placeholder_cards_are_smoke_readable_and_carry_text_payloads():
    from spoke.agent_shell_card_renderer import (
        build_agent_shell_card_optical_field_payload,
        build_agent_shell_card_render_payload,
    )

    payload = build_agent_shell_card_render_payload(
        [
            _primitive(
                "codex-inactive",
                primary_text="Codex lane reading Epistaxis semantic state",
                secondary_text="ready · /private/tmp/spoke-house-ten-thousand-ghosts",
            ),
            _primitive(
                "claude-selected",
                selected=True,
                show_latest_response=True,
                primary_text="Claude lane has a settled answer for the operator",
                secondary_text="working · claude-opus-4-6",
            ),
        ],
        content_width_points=720.0,
        content_height_points=260.0,
    )

    (inactive,) = payload["cards"]
    assert inactive["frame"]["width"] >= 300.0
    assert inactive["frame"]["height"] >= 72.0

    optical = build_agent_shell_card_optical_field_payload(payload)
    (inactive_request,) = optical["requests"]
    assert inactive_request["text"]["primary"] == inactive["primary_text"]
    assert inactive_request["text"]["secondary"] == inactive["secondary_text"]
