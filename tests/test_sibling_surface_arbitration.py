from __future__ import annotations

import pytest

from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldPlaceholderBackend,
    OpticalFieldPresentation,
    OpticalFieldRequest,
    OpticalFieldSelectedHandoff,
)


def _bounds(x: float = 0.0, y: float = 0.0) -> OpticalFieldBounds:
    return OpticalFieldBounds(x=x, y=y, width=240.0, height=80.0)


def test_sibling_surfaces_compile_independently_of_assistant_visibility():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.command",
            bounds=_bounds(),
            role="assistant_shell",
            presentation=OpticalFieldPresentation(layer="assistant_shell", order=20),
            visible=False,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="preview.transcription",
            bounds=_bounds(10.0, 12.0),
            role="user_preview",
            presentation=OpticalFieldPresentation(layer="user_preview", order=30),
            visible=True,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.codex-1",
            bounds=_bounds(20.0, 24.0),
            role="agent_card",
            presentation=OpticalFieldPresentation(layer="agent_card", order=10),
            visible=True,
        )
    )

    configs = backend.compile_shell_configs()

    assert [config["client_id"] for config in configs] == [
        "agent.card.codex-1",
        "preview.transcription",
    ]
    assert all(config["visibility_scope"] == "independent" for config in configs)
    assert {config["optical_field"]["role"] for config in configs} == {
        "agent_card",
        "user_preview",
    }


def test_presentation_layer_order_is_data_not_window_level_surgery():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="preview.transcription",
            bounds=_bounds(),
            role="user_preview",
            presentation=OpticalFieldPresentation(layer="user_preview", order=40),
            z_index=0,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.command",
            bounds=_bounds(),
            role="assistant_shell",
            presentation=OpticalFieldPresentation(layer="assistant_shell", order=20),
            z_index=99,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.codex-1",
            bounds=_bounds(),
            role="agent_card",
            presentation=OpticalFieldPresentation(layer="agent_card", order=20),
            z_index=5,
        )
    )

    configs = backend.compile_shell_configs()

    assert [config["client_id"] for config in configs] == [
        "agent.card.codex-1",
        "assistant.command",
        "preview.transcription",
    ]
    assert [config["presentation_layer"] for config in configs] == [
        "agent_card",
        "assistant_shell",
        "user_preview",
    ]
    assert configs[0]["presentation_order"] == 20
    assert configs[0]["z_index"] == 5
    assert configs[1]["presentation_order"] == 20
    assert configs[1]["z_index"] == 99


def test_selected_handoff_metadata_round_trips_without_card_truth_mutation():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.codex-1",
            bounds=_bounds(),
            role="agent_card",
            presentation=OpticalFieldPresentation(layer="agent_card", order=20),
            selected_handoff=OpticalFieldSelectedHandoff(
                from_caller_id="agent.card.codex-1",
                to_caller_id="assistant.command",
                continuity_key="codex-session-1",
                mode="handoff",
            ),
        )
    )

    (config,) = backend.compile_shell_configs()

    assert config["optical_field"]["selected_handoff"] == {
        "from_caller_id": "agent.card.codex-1",
        "to_caller_id": "assistant.command",
        "continuity_key": "codex-session-1",
        "mode": "handoff",
    }
    assert config["client_id"] == "agent.card.codex-1"
    assert config["role"] == "agent_card"


def test_presentation_rejects_assistant_dependent_sibling_visibility():
    with pytest.raises(ValueError, match="sibling surfaces must use independent visibility"):
        OpticalFieldRequest(
            caller_id="preview.transcription",
            bounds=_bounds(),
            role="user_preview",
            presentation=OpticalFieldPresentation(layer="user_preview", order=30),
            visibility_scope="follows_assistant_shell",
        )
