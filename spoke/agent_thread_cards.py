"""Agent Shell thread-card waypoints and deterministic card construction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentThreadWaypoint:
    kind: str
    text: str
    sequence: int = 0
    source: str = ""

    def to_event_data(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentThreadCard:
    thread_id: str
    provider: str
    title: str
    readiness: str
    bearing: str
    activity_line: str
    latest_response: str
    updated_sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CURRENT_INTENT_RE = re.compile(
    r"^\s*(?:\*\*)?Current intent(?:\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)
_ANAGNOSIS_RE = re.compile(
    r"^\s*(?:\*\*)?(?:Anagnosis|Ἀνάγνωσις)(?:\*\*)?\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _capture_block(lines: list[str], start: int, *, max_lines: int = 8) -> tuple[str, int]:
    block: list[str] = []
    index = start + 1
    while index < len(lines) and len(block) < max_lines:
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            break
        if block and re.match(r"^\s*#{1,6}\s+\S+", line):
            break
        if block and re.match(r"^\s*(?:\*\*)?[A-Z][A-Za-z ]+(?:\*\*)?\s*:?\s*$", line):
            break
        block.append(line)
        index += 1
    return _clean_text("\n".join(block)), index


def extract_thread_waypoints_from_text(
    text: str,
    *,
    sequence: int = 0,
    source: str = "",
) -> list[AgentThreadWaypoint]:
    """Extract self-orientation waypoints from agent-authored text."""
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    waypoints: list[AgentThreadWaypoint] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _CURRENT_INTENT_RE.match(line):
            block, index = _capture_block(lines, index)
            if block:
                waypoints.append(
                    AgentThreadWaypoint(
                        kind="current_intent",
                        text=block,
                        sequence=sequence,
                        source=source,
                    )
                )
            continue
        if _ANAGNOSIS_RE.match(line):
            block, index = _capture_block(lines, index)
            if block:
                waypoints.append(
                    AgentThreadWaypoint(
                        kind="anagnosis",
                        text=block,
                        sequence=sequence,
                        source=source,
                    )
                )
            continue
        index += 1
    return waypoints


def _event_sequence(event: dict[str, Any]) -> int:
    value = event.get("sequence")
    return value if isinstance(value, int) else 0


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _waypoints_from_event(event: dict[str, Any]) -> list[AgentThreadWaypoint]:
    kind = _string(event.get("kind"))
    sequence = _event_sequence(event)
    text = _string(event.get("text"))
    data = _event_data(event)
    if kind == "thread_waypoint":
        waypoint_kind = _string(data.get("kind")) or "waypoint"
        source = _string(data.get("source")) or "thread_waypoint"
        return [
            AgentThreadWaypoint(
                kind=waypoint_kind,
                text=text,
                sequence=sequence,
                source=source,
            )
        ] if text else []
    if kind == "agent_message":
        return extract_thread_waypoints_from_text(
            text or _string(data.get("text")),
            sequence=sequence,
            source="agent_message",
        )
    return []


def _orientation_waypoint(events: list[dict[str, Any]]) -> AgentThreadWaypoint | None:
    waypoints: list[AgentThreadWaypoint] = []
    for event in events:
        waypoints.extend(_waypoints_from_event(event))
    orienting = [
        waypoint
        for waypoint in waypoints
        if waypoint.kind in {"anagnosis", "current_intent"}
        and waypoint.text.strip()
    ]
    if not orienting:
        return None
    return max(orienting, key=lambda waypoint: waypoint.sequence)


def _readiness(state: str, *, has_response: bool) -> str:
    if state == "completed":
        return "ready" if has_response else "resting"
    if state in {"queued", "running", "cancelling"}:
        return "working"
    if state == "failed":
        return "failed"
    if state == "cancelled":
        return "cancelled"
    return state or "unknown"


def _strip_sentence(value: str) -> str:
    value = " ".join(value.split()).strip()
    return value.rstrip(".")


def _title_from_bearing(bearing: str, prompt: str) -> str:
    for line in bearing.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        session_match = re.match(r"^Session:\s*(.+)$", stripped, re.IGNORECASE)
        if session_match:
            return _strip_sentence(session_match.group(1))
        lowered = stripped.casefold()
        for prefix in ("the lane is ", "this lane is ", "working on "):
            if lowered.startswith(prefix):
                return _strip_sentence(stripped[len(prefix):])
        return _strip_sentence(stripped)
    return _strip_sentence(prompt) or "Agent thread"


def _paths_from_file_change(data: dict[str, Any], text: str) -> list[str]:
    changes = data.get("changes")
    if isinstance(changes, list):
        paths = [
            change.get("path")
            for change in changes
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        ]
        if paths:
            return paths
    return [part.strip() for part in text.split(",") if part.strip()]


def _activity_line(events: list[dict[str, Any]], readiness: str) -> str:
    for event in sorted(events, key=_event_sequence, reverse=True):
        kind = _string(event.get("kind"))
        data = _event_data(event)
        text = _string(event.get("text"))
        if kind == "error":
            return text or _string(data.get("message")) or "Backend error"
        if kind == "command_execution":
            status = _string(data.get("status"))
            command = _string(data.get("command"))
            if status == "in_progress":
                return f"Running: {command}" if command else "Running command"
            if status == "failed":
                return "Command failed"
            if status == "completed":
                return "Command completed"
        if kind == "file_change":
            paths = _paths_from_file_change(data, text)
            if len(paths) == 1:
                return f"Edited {paths[0]}"
            if len(paths) > 1:
                return f"Edited {len(paths)} files"
        if kind == "reasoning" and text:
            return " ".join(text.split())
    if readiness == "ready":
        return "Ready to read"
    if readiness == "working":
        return "Working"
    return readiness.capitalize() if readiness else "Unknown"


def build_agent_thread_card(session: dict[str, Any]) -> AgentThreadCard:
    """Build an operator attention card from a public Agent Shell session."""
    events = session.get("backend_events") or session.get("events") or []
    events = [event for event in events if isinstance(event, dict)]
    result = _string(session.get("result"))
    state = _string(session.get("state"))
    readiness = _readiness(state, has_response=bool(result))
    orientation = _orientation_waypoint(events)
    bearing = orientation.text if orientation is not None else _string(session.get("prompt"))
    title = _title_from_bearing(bearing, _string(session.get("prompt")))
    updated_sequence = max([_event_sequence(event) for event in events] or [0])
    return AgentThreadCard(
        thread_id=_string(session.get("id")),
        provider=_string(session.get("provider")),
        title=title,
        readiness=readiness,
        bearing=bearing,
        activity_line=_activity_line(events, readiness),
        latest_response=result,
        updated_sequence=updated_sequence,
    )
