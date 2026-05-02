"""Deterministic Agent Shell narrator state.

This module builds the selected-thread semantic surface from state Spoke already
custodies. Local model refinement can sit on top later; this first contract is
pure, bounded, and source-attached.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentThreadNarratorState:
    provider: str
    thread_id: str
    provider_session_id: str
    updated_sequence: int
    source_event_ids: tuple[str, ...]
    latest_user_prompt: str
    bearing: str
    since_user_prompt: str
    latest_verbatim_tail: tuple[str, ...]
    current_activity: str
    readiness: str
    operator_needed: bool
    confidence: str
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_event_ids"] = list(self.source_event_ids)
        payload["latest_verbatim_tail"] = list(self.latest_verbatim_tail)
        payload["provenance"] = list(self.provenance)
        return payload


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _event_sequence(event: dict[str, Any]) -> int:
    sequence = event.get("sequence")
    return sequence if isinstance(sequence, int) else 0


def _normalize_line(text: str) -> str:
    return " ".join(text.split()).strip()


def _clamp(text: str, limit: int) -> str:
    text = _normalize_line(text)
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def _card_mapping(session: dict[str, Any]) -> dict[str, Any]:
    card = session.get("thread_card")
    return card if isinstance(card, dict) else {}


def _events(session: dict[str, Any]) -> list[dict[str, Any]]:
    raw = session.get("backend_events") or session.get("events") or []
    if not isinstance(raw, list):
        return []
    return [event for event in raw if isinstance(event, dict)]


def _source_event_id(event: dict[str, Any]) -> str:
    kind = _string(event.get("kind")) or "event"
    sequence = _event_sequence(event)
    return f"{kind}:{sequence}" if sequence else kind


def _latest_user_prompt(session: dict[str, Any]) -> str:
    return (
        _string(session.get("prompt"))
        or _string(session.get("last_utterance"))
        or _string(session.get("utterance"))
    )


def _latest_agent_text(session: dict[str, Any], card: dict[str, Any]) -> str:
    return (
        _string(session.get("result"))
        or _string(session.get("last_response"))
        or _string(card.get("latest_response"))
    )


def _latest_verbatim_tail(text: str, *, max_lines: int = 8) -> tuple[str, ...]:
    lines = [
        _clamp(line, 260)
        for line in text.splitlines()
        if _normalize_line(line)
    ]
    return tuple(lines[-max(1, max_lines):])


def _is_prompt_echo(candidate: str, latest_user_prompt: str) -> bool:
    candidate = _normalize_line(candidate)
    latest_user_prompt = _normalize_line(latest_user_prompt)
    if not candidate or not latest_user_prompt:
        return False
    if candidate == latest_user_prompt:
        return True
    return len(latest_user_prompt) >= 48 and candidate.startswith(latest_user_prompt[:48])


def _bearing(session: dict[str, Any], card: dict[str, Any], events: list[dict[str, Any]]) -> str:
    waypoints = [
        event
        for event in events
        if _string(event.get("kind")) == "thread_waypoint"
        and (_string(event.get("text")) or _string(_event_data(event).get("text")))
    ]
    if waypoints:
        waypoint = max(waypoints, key=_event_sequence)
        return _clamp(
            _string(waypoint.get("text")) or _string(_event_data(waypoint).get("text")),
            220,
        )
    latest_user_prompt = _latest_user_prompt(session)
    for candidate in (_string(card.get("bearing")), _string(card.get("title"))):
        if candidate and not _is_prompt_echo(candidate, latest_user_prompt):
            return _clamp(candidate, 220)
    return "No durable bearing captured yet"


def _readiness(session: dict[str, Any], card: dict[str, Any], latest_text: str) -> str:
    readiness = _string(card.get("readiness")).strip()
    if readiness:
        return readiness
    state = _string(session.get("state")).strip()
    if state == "completed":
        return "ready" if latest_text else "resting"
    if state in {"queued", "running", "cancelling"}:
        return "working"
    if state in {"failed", "cancelled"}:
        return state
    return state or "unknown"


def _activity_line(card: dict[str, Any], events: list[dict[str, Any]], readiness: str) -> str:
    activity = _string(card.get("activity_line")).strip()
    if activity:
        return activity
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
            if status == "completed":
                return "Command completed"
            if status == "failed":
                return "Command failed"
        if kind == "reasoning" and text:
            return _clamp(text, 96)
    if readiness == "ready":
        return "Ready to read"
    if readiness == "working":
        return "Working"
    return readiness.capitalize() if readiness else "Unknown"


def _event_fact(event: dict[str, Any]) -> str:
    kind = _string(event.get("kind"))
    data = _event_data(event)
    if kind == "command_execution":
        status = _string(data.get("status"))
        command = _string(data.get("command"))
        if status == "completed":
            return "Command completed"
        if status == "failed":
            return "Command failed"
        if status == "in_progress":
            return f"Command running: {command}" if command else "Command running"
    if kind == "file_change":
        changes = data.get("changes")
        if isinstance(changes, list):
            count = len([change for change in changes if isinstance(change, dict)])
            if count == 1:
                return "Edited 1 file"
            if count > 1:
                return f"Edited {count} files"
        text = _string(event.get("text"))
        if text:
            return f"Edited {text}"
    if kind == "error":
        return "Backend error"
    if kind == "mcp_tool_call":
        status = _string(data.get("status"))
        tool = ".".join(
            part for part in (_string(data.get("server")), _string(data.get("tool"))) if part
        )
        if status == "completed":
            return f"Tool completed: {tool}" if tool else "Tool completed"
        if status == "failed":
            return f"Tool failed: {tool}" if tool else "Tool failed"
        if status == "in_progress":
            return f"Tool running: {tool}" if tool else "Tool running"
    return ""


def _since_user_prompt(events: list[dict[str, Any]], tail: tuple[str, ...]) -> str:
    facts: list[str] = []
    for event in sorted(events, key=_event_sequence):
        fact = _event_fact(event)
        if fact and fact not in facts:
            facts.append(fact)
    if tail and facts:
        facts.append(
            f"assistant produced {len(tail)} recent line"
            f"{'' if len(tail) == 1 else 's'}"
        )
    if tail and not facts:
        return (
            f"Assistant replied with {len(tail)} visible line"
            f"{'' if len(tail) == 1 else 's'}; "
            "no tool or state events captured in this slice."
        )
    if not facts:
        return "No new activity captured yet."
    sentence = "; ".join(facts)
    return sentence[:1].upper() + sentence[1:] + "."


def _provenance(
    session: dict[str, Any],
    card: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[str, ...]:
    provenance: list[str] = []
    if _latest_user_prompt(session):
        provenance.append("latest_user_prompt")
    if _latest_agent_text(session, card):
        provenance.append("latest_agent_text")
    if card:
        provenance.append("thread_card")
    for event in events:
        source = _source_event_id(event)
        if source not in provenance:
            provenance.append(source)
    return tuple(provenance)


def build_agent_thread_narrator_state(session: dict[str, Any]) -> AgentThreadNarratorState:
    card = _card_mapping(session)
    events = _events(session)
    provider = _string(session.get("provider")) or _string(card.get("provider"))
    provider_session_id = (
        _string(session.get("provider_session_id"))
        or _string(card.get("provider_session_id"))
        or _string(session.get("id"))
        or _string(card.get("thread_id"))
    )
    thread_id = (
        _string(card.get("thread_id"))
        or provider_session_id
        or _string(session.get("id"))
    )
    latest_text = _latest_agent_text(session, card)
    tail = _latest_verbatim_tail(latest_text)
    readiness = _readiness(session, card, latest_text)
    updated_sequence = max(
        [_event_sequence(event) for event in events]
        + [value for value in (card.get("updated_sequence"),) if isinstance(value, int)]
        + [0]
    )
    return AgentThreadNarratorState(
        provider=provider,
        thread_id=thread_id,
        provider_session_id=provider_session_id,
        updated_sequence=updated_sequence,
        source_event_ids=tuple(_source_event_id(event) for event in events),
        latest_user_prompt=_latest_user_prompt(session),
        bearing=_bearing(session, card, events),
        since_user_prompt=_since_user_prompt(events, tail),
        latest_verbatim_tail=tail,
        current_activity=_activity_line(card, events, readiness),
        readiness=readiness,
        operator_needed=readiness in {"ready", "failed", "cancelled"},
        confidence="deterministic",
        provenance=_provenance(session, card, events),
    )


def format_selected_thread_narrator_response(
    state: AgentThreadNarratorState | dict[str, Any],
) -> str:
    if isinstance(state, AgentThreadNarratorState):
        payload = state.to_dict()
    else:
        payload = state if isinstance(state, dict) else {}
    bearing = _string(payload.get("bearing"))
    since = _string(payload.get("since_user_prompt"))
    raw_tail = payload.get("latest_verbatim_tail")
    tail = [line for line in raw_tail if isinstance(line, str) and line.strip()] if isinstance(raw_tail, list) else []
    lines: list[str] = []
    if bearing:
        lines.append(f"Bearing: {bearing}")
    if since:
        lines.append(f"Since your prompt: {since}")
    if tail:
        if lines:
            lines.append("")
        lines.append("Recent output:")
        lines.extend(tail)
    return "\n".join(lines).strip()
