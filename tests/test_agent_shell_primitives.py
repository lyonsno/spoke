"""Tests for provider-agnostic Agent Shell card primitives."""

from __future__ import annotations


def test_selected_resting_primitive_preserves_display_contract_and_session_identity():
    from spoke.agent_shell_primitives import build_agent_shell_primitives

    primitives = build_agent_shell_primitives(
        [
            {
                "thread_id": "codex-session-1",
                "provider_session_id": "codex-session-1",
                "provider": "codex",
                "title": "active lane",
                "readiness": "ready",
                "bearing": "Session: active lane\nImmediate next step: ship it",
                "activity_line": "Ready to read",
                "latest_response": "The selected resting response is real transcript text.",
                "selected": True,
                "topos": "codex-active-lane",
                "worktree": "/tmp/spoke-active-lane",
                "model": "gpt-5",
            }
        ]
    )

    assert len(primitives) == 1
    primitive = primitives[0]
    assert primitive["kind"] == "selected_thread"
    assert primitive["id"] == "codex-session-1"
    assert primitive["provider_session_id"] == "codex-session-1"
    assert primitive["provider"] == "codex"
    assert primitive["selected"] is True
    assert primitive["readiness"] == "ready"
    assert primitive["latest_response"] == "The selected resting response is real transcript text."
    assert primitive["display"] == {
        "display_state": "selected_resting",
        "primary_text": "The selected resting response is real transcript text.",
        "secondary_text": "active lane · Ready to read",
        "show_latest_response": True,
    }
    assert primitive["chrome"] == {
        "header": "active lane",
        "footer": "Ready to read",
        "topos": "codex-active-lane",
        "worktree": "/tmp/spoke-active-lane",
        "model": "gpt-5",
    }


def test_selected_working_primitive_hides_latest_response_and_keeps_status_primary():
    from spoke.agent_shell_primitives import build_agent_shell_primitives

    primitive = build_agent_shell_primitives(
        [
            {
                "thread_id": "claude-session-1",
                "provider_session_id": "claude-session-1",
                "provider": "claude-code",
                "title": "working lane",
                "readiness": "working",
                "bearing": "Session: working lane",
                "activity_line": "Running tests",
                "latest_response": "A stale partial response must not surface while working.",
                "selected": True,
            }
        ]
    )[0]

    assert primitive["kind"] == "selected_thread"
    assert primitive["provider"] == "claude-code"
    assert primitive["provider_session_id"] == "claude-session-1"
    assert primitive["latest_response"] == ""
    assert primitive["display"] == {
        "display_state": "selected_working",
        "primary_text": "working lane · Running tests",
        "secondary_text": "working lane · Running tests",
        "show_latest_response": False,
    }


def test_inactive_primitives_hide_response_without_provider_specific_special_cases():
    from spoke.agent_shell_primitives import build_agent_shell_primitives

    primitives = build_agent_shell_primitives(
        [
            {
                "thread_id": "gemini-session-1",
                "provider": "gemini-cli",
                "title": "inactive lane",
                "readiness": "ready",
                "bearing": "Session: inactive lane",
                "activity_line": "Ready to read",
                "latest_response": "Inactive cards must not expose full transcript text.",
                "selected": False,
            },
            {
                "id": "generated-stable-id",
                "provider": "",
                "title": "fallback identity lane",
                "readiness": "waiting",
                "bearing": "Session: fallback identity lane",
                "activity_line": "Waiting",
                "latest_response": "Hidden fallback response.",
                "selected": False,
            },
        ]
    )

    assert [primitive["kind"] for primitive in primitives] == ["thread_card", "thread_card"]
    assert [primitive["provider"] for primitive in primitives] == ["gemini-cli", ""]
    assert [primitive["provider_session_id"] for primitive in primitives] == [
        "gemini-session-1",
        "generated-stable-id",
    ]
    assert [primitive["id"] for primitive in primitives] == [
        "gemini-session-1",
        "generated-stable-id",
    ]
    assert all(primitive["latest_response"] == "" for primitive in primitives)
    assert all(primitive["display"]["show_latest_response"] is False for primitive in primitives)
    assert primitives[0]["display"]["display_state"] == "inactive"
    assert primitives[0]["display"]["primary_text"] == "inactive lane · Ready to read"
    assert primitives[1]["display"]["display_state"] == "inactive"
    assert primitives[1]["display"]["primary_text"] == "fallback identity lane · Waiting"
