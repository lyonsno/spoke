"""Tests for source-attributed Agent Shell waypoint packets."""

from __future__ import annotations

import json


def _write_jsonl(path, events):
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_codex_log_packet_extracts_waypoints_from_assistant_messages(tmp_path):
    from spoke.agent_thread_waypoint_sources import waypoint_packet_from_codex_log

    log_path = tmp_path / "rollout-2026-04-29T19-36-20-thread-1.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "timestamp": "2026-04-29T23:36:23.006Z",
                "type": "session_meta",
                "payload": {
                    "id": "thread-1",
                    "cwd": "/work/spoke",
                    "agent_role": "epistaxis-log-gut-intent-scavenger-spinaltap",
                    "agent_nickname": "Log-Gut Intent Scavenger SpinalTap",
                },
            },
            {
                "timestamp": "2026-04-29T23:39:10.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "**Current intent**\n"
                                "- Session: harvest waypoints from Codex logs.\n"
                                "- Repo/task: Spoke child branch.\n"
                                "- Immediate next step: write failing tests.\n\n"
                                "Ordinary response prose continues here."
                            ),
                        }
                    ],
                },
            },
            {
                "timestamp": "2026-04-29T23:40:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "**Anagnosis**\nThis tool output must not become identity.",
                },
            },
            {
                "timestamp": "2026-04-29T23:41:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": "ignored non-list content",
                },
            },
            {
                "timestamp": "2026-04-29T23:42:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "**Anagnosis**\n"
                                "The lane now thinks the next useful slice is archive fallback."
                            ),
                        }
                    ],
                },
            },
        ],
    )

    packet = waypoint_packet_from_codex_log(log_path)

    assert packet.status == "ok"
    assert packet.provider == "codex"
    assert packet.thread_id == "thread-1"
    assert packet.source_path == str(log_path)
    assert packet.source_hint == "codex-log"
    assert packet.metadata["cwd"] == "/work/spoke"
    assert packet.metadata["agent_role"] == "epistaxis-log-gut-intent-scavenger-spinaltap"
    assert [(waypoint.kind, waypoint.sequence) for waypoint in packet.waypoints] == [
        ("current_intent", 1),
        ("anagnosis", 4),
    ]
    assert packet.waypoints[0].source == f"codex-log:{log_path}"
    assert "tool output" not in "\n".join(waypoint.text for waypoint in packet.waypoints)
    assert packet.to_events() == [
        {
            "kind": "thread_waypoint",
            "text": packet.waypoints[0].text,
            "sequence": 1,
            "data": packet.waypoints[0].to_event_data(),
        },
        {
            "kind": "thread_waypoint",
            "text": packet.waypoints[1].text,
            "sequence": 4,
            "data": packet.waypoints[1].to_event_data(),
        },
    ]


def test_codex_session_lookup_degrades_explicitly_for_missing_and_ambiguous(tmp_path):
    from spoke.agent_thread_waypoint_sources import waypoint_packet_from_codex_session

    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()

    missing = waypoint_packet_from_codex_session(
        "missing-thread",
        sessions_root=sessions_root,
    )

    assert missing.status == "missing_archive"
    assert missing.thread_id == "missing-thread"
    assert missing.provider == "codex"
    assert missing.waypoints == ()
    assert "No Codex session log matched" in missing.degradation_reason
    assert missing.to_events() == []

    first = sessions_root / "rollout-a-maybe-thread.jsonl"
    second = sessions_root / "nested" / "rollout-b-maybe-thread.jsonl"
    second.parent.mkdir()
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    ambiguous = waypoint_packet_from_codex_session(
        "maybe-thread",
        sessions_root=sessions_root,
    )

    assert ambiguous.status == "ambiguous_archive"
    assert ambiguous.thread_id == "maybe-thread"
    assert ambiguous.waypoints == ()
    assert "2 Codex session logs matched" in ambiguous.degradation_reason
    assert ambiguous.metadata["candidate_count"] == 2
    assert "agent" not in ambiguous.metadata


def test_epistaxis_markdown_packet_preserves_drift_between_current_intents(tmp_path):
    from spoke.agent_thread_waypoint_sources import waypoint_packet_from_epistaxis_markdown

    epistaxis_path = tmp_path / "epistaxis.md"
    epistaxis_path.write_text(
        """# Spoke Epistaxis

**Current intent**
- Session: old lane thought it was polishing the HUD.
- Repo/task: branch `old`.
- Immediate next step: smoke the visual surface.

## Scoped Local State

### codex-log-gut-intent-scavenger-spinaltap-0429
- Lane: `anakrisis-waypoint-harvester`
- Status: **Lane launched.** Extracting source-attributed waypoints.

**Current intent**
- Session: new lane is harvesting Codex and Epistaxis waypoints.
- Repo/task: branch `new`.
- Immediate next step: hand packets to compiler lanes.
""",
        encoding="utf-8",
    )

    packet = waypoint_packet_from_epistaxis_markdown(
        epistaxis_path,
        thread_id="spoke-epistaxis",
    )

    assert packet.status == "ok"
    assert packet.provider == "epistaxis"
    assert [waypoint.kind for waypoint in packet.waypoints] == [
        "current_intent",
        "current_intent",
    ]
    assert "old lane thought" in packet.waypoints[0].text
    assert "new lane is harvesting" in packet.waypoints[1].text
    assert packet.waypoints[0].sequence < packet.waypoints[1].sequence
    assert packet.waypoints[0].source == f"epistaxis:{epistaxis_path}"


def test_epistaxis_missing_archive_degrades_without_identity_guess(tmp_path):
    from spoke.agent_thread_waypoint_sources import waypoint_packet_from_epistaxis_markdown

    missing_path = tmp_path / "missing.md"

    packet = waypoint_packet_from_epistaxis_markdown(
        missing_path,
        thread_id="spoke-epistaxis",
    )

    assert packet.status == "missing_archive"
    assert packet.provider == "epistaxis"
    assert packet.thread_id == "spoke-epistaxis"
    assert packet.title == ""
    assert packet.waypoints == ()
    assert "No Epistaxis markdown archive found" in packet.degradation_reason
