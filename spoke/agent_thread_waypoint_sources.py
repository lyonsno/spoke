"""Read-only source adapters for Agent Shell thread waypoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .agent_thread_cards import (
    AgentThreadWaypoint,
    extract_thread_waypoints_from_text,
)


@dataclass(frozen=True)
class AgentThreadWaypointPacket:
    """Small source-attributed waypoint packet for compiler/HUD consumers."""

    provider: str
    thread_id: str
    status: str
    source_hint: str
    source_path: str = ""
    title: str = ""
    waypoints: tuple[AgentThreadWaypoint, ...] = ()
    degradation_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_events(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": "thread_waypoint",
                "text": waypoint.text,
                "sequence": waypoint.sequence,
                "data": waypoint.to_event_data(),
            }
            for waypoint in self.waypoints
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["waypoints"] = [waypoint.to_event_data() for waypoint in self.waypoints]
        data["events"] = self.to_events()
        return data


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _jsonl_events(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield index, event


def _codex_message_text(payload: dict[str, Any]) -> str:
    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return ""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _metadata_from_codex_session(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("cwd", "agent_role", "agent_nickname", "model", "cli_version"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    return metadata


def waypoint_packet_from_codex_log(path: str | Path) -> AgentThreadWaypointPacket:
    """Extract waypoints from a Codex rollout JSONL file without mutating it."""

    log_path = Path(path).expanduser()
    source = f"codex-log:{log_path}"
    if not log_path.exists():
        return AgentThreadWaypointPacket(
            provider="codex",
            thread_id="",
            status="missing_archive",
            source_hint="codex-log",
            source_path=str(log_path),
            degradation_reason=f"No Codex session log found at {log_path}",
        )

    thread_id = ""
    metadata: dict[str, Any] = {}
    waypoints: list[AgentThreadWaypoint] = []
    try:
        for sequence, event in _jsonl_events(log_path):
            event_type = event.get("type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if event_type == "session_meta":
                candidate_id = payload.get("id")
                if isinstance(candidate_id, str) and candidate_id:
                    thread_id = candidate_id
                metadata.update(_metadata_from_codex_session(payload))
                continue
            if event_type != "response_item":
                continue
            text = _codex_message_text(payload)
            if not text:
                continue
            waypoints.extend(
                extract_thread_waypoints_from_text(
                    text,
                    sequence=sequence,
                    source=source,
                )
            )
    except OSError as exc:
        return AgentThreadWaypointPacket(
            provider="codex",
            thread_id=thread_id,
            status="unreadable_archive",
            source_hint="codex-log",
            source_path=str(log_path),
            degradation_reason=f"Could not read Codex session log {log_path}: {exc}",
            metadata=metadata,
        )

    status = "ok" if waypoints else "no_waypoints"
    return AgentThreadWaypointPacket(
        provider="codex",
        thread_id=thread_id,
        status=status,
        source_hint="codex-log",
        source_path=str(log_path),
        title=metadata.get("agent_nickname", ""),
        waypoints=tuple(waypoints),
        metadata=metadata,
    )


def _matching_codex_logs(session_id: str, sessions_root: Path) -> list[Path]:
    if not session_id.strip() or not sessions_root.exists():
        return []
    return sorted(sessions_root.rglob(f"*{session_id.strip()}*.jsonl"))


def waypoint_packet_from_codex_session(
    session_id: str,
    *,
    sessions_root: str | Path | None = None,
) -> AgentThreadWaypointPacket:
    """Find a Codex session log by id and return a waypoint packet or degradation."""

    root = (
        Path(sessions_root).expanduser()
        if sessions_root is not None
        else Path.home() / ".codex" / "sessions"
    )
    matches = _matching_codex_logs(session_id, root)
    if not matches:
        return AgentThreadWaypointPacket(
            provider="codex",
            thread_id=session_id,
            status="missing_archive",
            source_hint="codex-log",
            degradation_reason=(
                f"No Codex session log matched {session_id!r} under {root}"
            ),
            metadata={"sessions_root": str(root)},
        )
    if len(matches) > 1:
        return AgentThreadWaypointPacket(
            provider="codex",
            thread_id=session_id,
            status="ambiguous_archive",
            source_hint="codex-log",
            degradation_reason=(
                f"{len(matches)} Codex session logs matched {session_id!r} under {root}"
            ),
            metadata={
                "sessions_root": str(root),
                "candidate_count": len(matches),
                "candidates": [str(path) for path in matches],
            },
        )
    packet = waypoint_packet_from_codex_log(matches[0])
    return AgentThreadWaypointPacket(
        provider=packet.provider,
        thread_id=packet.thread_id or session_id,
        status=packet.status,
        source_hint=packet.source_hint,
        source_path=packet.source_path,
        title=packet.title,
        waypoints=packet.waypoints,
        degradation_reason=packet.degradation_reason,
        metadata=packet.metadata,
    )


def _waypoints_from_markdown_text(
    text: str,
    *,
    source: str,
) -> tuple[AgentThreadWaypoint, ...]:
    lines = text.splitlines()
    waypoints: list[AgentThreadWaypoint] = []
    for index, line in enumerate(lines):
        lowered = line.strip().strip("*").strip(":").casefold()
        if lowered not in {"current intent", "anagnosis", "ἁνάγνωσις", "ἀνάγνωσις"}:
            continue
        snippet = "\n".join(lines[index : index + 10])
        waypoints.extend(
            extract_thread_waypoints_from_text(
                snippet,
                sequence=index,
                source=source,
            )
        )
    return tuple(waypoints)


def waypoint_packet_from_epistaxis_markdown(
    path: str | Path,
    *,
    thread_id: str,
) -> AgentThreadWaypointPacket:
    """Extract self-orientation waypoints from a read-only Epistaxis markdown file."""

    archive_path = Path(path).expanduser()
    source = f"epistaxis:{archive_path}"
    if not archive_path.exists():
        return AgentThreadWaypointPacket(
            provider="epistaxis",
            thread_id=thread_id,
            status="missing_archive",
            source_hint="epistaxis-markdown",
            source_path=str(archive_path),
            degradation_reason=f"No Epistaxis markdown archive found at {archive_path}",
        )
    try:
        text = archive_path.read_text(encoding="utf-8")
    except OSError as exc:
        return AgentThreadWaypointPacket(
            provider="epistaxis",
            thread_id=thread_id,
            status="unreadable_archive",
            source_hint="epistaxis-markdown",
            source_path=str(archive_path),
            degradation_reason=(
                f"Could not read Epistaxis markdown archive {archive_path}: {exc}"
            ),
        )
    waypoints = _waypoints_from_markdown_text(text, source=source)
    return AgentThreadWaypointPacket(
        provider="epistaxis",
        thread_id=thread_id,
        status="ok" if waypoints else "no_waypoints",
        source_hint="epistaxis-markdown",
        source_path=str(archive_path),
        waypoints=waypoints,
    )
