"""Tests for deterministic Agent Shell placement placeholders."""

from __future__ import annotations


def _primitive(
    primitive_id: str,
    *,
    selected: bool = False,
    kind: str = "thread_card",
    anchor: str | None = None,
) -> dict[str, object]:
    primitive: dict[str, object] = {
        "id": primitive_id,
        "kind": "selected_thread" if selected else kind,
        "provider": "codex",
        "provider_session_id": primitive_id,
        "selected": selected,
        "readiness": "ready",
        "title": f"title {primitive_id}",
        "bearing": f"bearing {primitive_id}",
        "activity_line": "Ready to read",
        "display": {
            "display_state": "selected_resting" if selected else "inactive",
            "primary_text": f"primary {primitive_id}",
            "show_latest_response": selected,
        },
        "geometry": {
            "preferred_width": 180.0,
            "preferred_height": 44.0,
            "min_width": 92.0,
            "min_height": 44.0,
        },
    }
    if anchor is not None:
        geometry = primitive["geometry"]
        assert isinstance(geometry, dict)
        geometry["anchor"] = anchor
    return primitive


def test_initial_placement_assigns_stable_anchors_and_priorities_without_changing_identity():
    from spoke.agent_shell_placement import apply_initial_agent_shell_placement

    selected = _primitive("codex-selected", selected=True)
    inactive_a = _primitive("codex-inactive-a")
    inactive_b = _primitive("codex-inactive-b")
    before_identity = [
        {
            key: primitive[key]
            for key in (
                "id",
                "kind",
                "provider",
                "provider_session_id",
                "selected",
                "readiness",
                "display",
            )
        }
        for primitive in (inactive_a, selected, inactive_b)
    ]

    placed = apply_initial_agent_shell_placement([inactive_a, selected, inactive_b])

    assert [
        {
            key: primitive[key]
            for key in (
                "id",
                "kind",
                "provider",
                "provider_session_id",
                "selected",
                "readiness",
                "display",
            )
        }
        for primitive in placed
    ] == before_identity
    assert [primitive["geometry"]["anchor"] for primitive in placed] == [
        "bottom",
        "right",
        "bottom",
    ]
    assert [primitive["geometry"]["priority"] for primitive in placed] == [
        100,
        0,
        101,
    ]

    replaced = apply_initial_agent_shell_placement([inactive_a, selected, inactive_b])
    assert [primitive["geometry"] for primitive in replaced] == [
        primitive["geometry"] for primitive in placed
    ]


def test_semantic_anchor_survives_as_placeholder_with_fallback_and_contract_payload():
    from spoke.agent_shell_placement import apply_initial_agent_shell_placement

    semantic = _primitive("codex-semantic", anchor="semantic")

    placed = apply_initial_agent_shell_placement([semantic])

    geometry = placed[0]["geometry"]
    assert geometry["anchor"] == "semantic"
    assert geometry["priority"] == 100
    assert geometry["semantic_placement"] == {
        "status": "placeholder",
        "fallback_anchor": "bottom",
        "provider_session_id": "codex-semantic",
        "primitive_id": "codex-semantic",
    }


def test_placement_skips_non_mapping_values_without_mutating_input_primitives():
    from spoke.agent_shell_placement import apply_initial_agent_shell_placement

    original = _primitive("codex-thread")

    placed = apply_initial_agent_shell_placement([original, "not-a-primitive"])

    assert len(placed) == 1
    assert placed[0] is not original
    assert "anchor" not in original["geometry"]
    assert placed[0]["geometry"]["anchor"] == "bottom"
