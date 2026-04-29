"""Tests for Agent Shell thread-card waypoint extraction and card state."""

from __future__ import annotations


def test_extracts_current_intent_waypoint_from_agent_text():
    from spoke.agent_thread_cards import extract_thread_waypoints_from_text

    waypoints = extract_thread_waypoints_from_text(
        """Some response.

**Current intent**
Session: build Agent Shell cards.
Repo/task: Spoke Spinal Tap worktree.
Write-local: no write.
Immediate next step: run the focused tests.

More prose.""",
        sequence=7,
        source="agent_message",
    )

    assert len(waypoints) == 1
    assert waypoints[0].kind == "current_intent"
    assert waypoints[0].sequence == 7
    assert waypoints[0].source == "agent_message"
    assert "build Agent Shell cards" in waypoints[0].text
    assert "run the focused tests" in waypoints[0].text
    assert "More prose" not in waypoints[0].text


def test_extracts_anagnosis_waypoint_from_agent_text():
    from spoke.agent_thread_cards import extract_thread_waypoints_from_text

    waypoints = extract_thread_waypoints_from_text(
        """**Anagnosis**
The lane is turning Codex sessions into operator-thread cards.
The next useful slice is read-side waypoint extraction.

Then the agent starts working.""",
        sequence=3,
        source="codex-log",
    )

    assert [(waypoint.kind, waypoint.text) for waypoint in waypoints] == [
        (
            "anagnosis",
            "The lane is turning Codex sessions into operator-thread cards.\n"
            "The next useful slice is read-side waypoint extraction.",
        )
    ]


def test_thread_card_uses_orientation_as_bearing_and_completed_state_as_ready():
    from spoke.agent_thread_cards import build_agent_thread_card

    card = build_agent_thread_card(
        {
            "id": "agent-backend-codex-1",
            "provider": "codex",
            "state": "completed",
            "prompt": "build the thread card",
            "result": "Done. The card contract is implemented.",
            "backend_events": [
                {
                    "sequence": 1,
                    "kind": "agent_message",
                    "text": (
                        "**Current intent**\n"
                        "Session: build Agent Shell cards.\n"
                        "Repo/task: Spoke Spinal Tap worktree.\n"
                        "Immediate next step: verify tests."
                    ),
                    "data": {"id": "msg-1"},
                },
                {
                    "sequence": 2,
                    "kind": "command_execution",
                    "data": {
                        "command": "uv run pytest -q tests/test_agent_thread_cards.py",
                        "status": "completed",
                    },
                },
            ],
        }
    )

    assert card.thread_id == "agent-backend-codex-1"
    assert card.readiness == "ready"
    assert card.title == "build Agent Shell cards"
    assert card.bearing.startswith("Session: build Agent Shell cards.")
    assert card.activity_line == "Command completed"
    assert card.latest_response == "Done. The card contract is implemented."


def test_thread_card_keeps_running_activity_under_anagnosis_anchor():
    from spoke.agent_thread_cards import build_agent_thread_card

    card = build_agent_thread_card(
        {
            "id": "agent-backend-codex-2",
            "provider": "codex",
            "state": "running",
            "prompt": "fix the overlay",
            "backend_events": [
                {
                    "sequence": 1,
                    "kind": "agent_message",
                    "text": (
                        "**Anagnosis**\n"
                        "The lane is fixing Agent Shell overlay cards.\n"
                        "The current approach is pure extraction before UI polish."
                    ),
                },
                {
                    "sequence": 2,
                    "kind": "file_change",
                    "text": "spoke/agent_thread_cards.py",
                    "data": {"changes": [{"path": "spoke/agent_thread_cards.py"}]},
                },
            ],
        }
    )

    assert card.readiness == "working"
    assert card.title == "fixing Agent Shell overlay cards"
    assert card.activity_line == "Edited spoke/agent_thread_cards.py"
    assert "pure extraction before UI polish" in card.bearing
