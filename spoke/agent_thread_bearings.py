"""Local Agent Shell thread-bearing compilation and scheduling."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .agent_thread_cards import AgentThreadWaypoint


class ThreadBearingModelClient(Protocol):
    def compile_bearing(self, prompt: str) -> dict[str, str] | str:
        """Return a compact bearing for a local prompt."""


class ThreadBearingCompiler(Protocol):
    def compile(self, item: "ThreadBearingInput") -> "CompiledThreadBearing":
        """Compile a source-attached operator bearing."""


@dataclass
class ThreadBearingInput:
    thread_id: str
    provider: str
    waypoints: tuple[AgentThreadWaypoint, ...] = ()
    recent_events: tuple[dict[str, Any], ...] = ()
    title: str = ""
    current_bearing: str = ""
    current_activity_line: str = ""
    updated_sequence: int = 0

    def __post_init__(self) -> None:
        self.waypoints = tuple(self.waypoints)
        self.recent_events = tuple(
            event for event in self.recent_events if isinstance(event, dict)
        )
        if self.updated_sequence <= 0:
            waypoint_sequences = [waypoint.sequence for waypoint in self.waypoints]
            event_sequences = [
                event.get("sequence", 0)
                for event in self.recent_events
                if isinstance(event.get("sequence"), int)
            ]
            self.updated_sequence = max(waypoint_sequences + event_sequences + [0])

    def input_signature(self) -> str:
        payload = {
            "thread_id": self.thread_id,
            "provider": self.provider,
            "waypoints": [waypoint.to_event_data() for waypoint in self.waypoints],
            "recent_events": self.recent_events,
            "current_bearing": self.current_bearing,
            "current_activity_line": self.current_activity_line,
            "updated_sequence": self.updated_sequence,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompiledThreadBearing:
    thread_id: str
    provider: str
    bearing: str
    activity_line: str
    updated_sequence: int
    sources: tuple[str, ...]
    prompt: str
    inference_status: str
    input_signature: str

    def to_card_updates(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "provider": self.provider,
            "bearing": self.bearing,
            "activity_line": self.activity_line,
            "updated_sequence": self.updated_sequence,
            "bearing_sources": list(self.sources),
            "bearing_inference_status": self.inference_status,
            "bearing_input_signature": self.input_signature,
        }


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _event_sequence(event: dict[str, Any]) -> int:
    value = event.get("sequence")
    return value if isinstance(value, int) else 0


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _normalize_line(text: str) -> str:
    return " ".join(text.split()).strip()


def _clamp(text: str, limit: int) -> str:
    text = _normalize_line(text)
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def _activity_line_from_events(events: tuple[dict[str, Any], ...]) -> str:
    for event in sorted(events, key=_event_sequence, reverse=True):
        kind = _string(event.get("kind"))
        text = _string(event.get("text"))
        data = _event_data(event)
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
            return f"Command: {command}" if command else "Command event"
        if kind == "file_change":
            changes = data.get("changes")
            paths: list[str] = []
            if isinstance(changes, list):
                paths = [
                    change.get("path")
                    for change in changes
                    if isinstance(change, dict) and isinstance(change.get("path"), str)
                ]
            if not paths:
                paths = [part.strip() for part in text.split(",") if part.strip()]
            if len(paths) == 1:
                return f"Edited {paths[0]}"
            if len(paths) > 1:
                return f"Edited {len(paths)} files"
        if kind == "reasoning" and text:
            return _clamp(text, 80)
        if text:
            return _clamp(text, 80)
    return ""


def _source_labels(item: ThreadBearingInput) -> tuple[str, ...]:
    labels: list[str] = []
    for waypoint in item.waypoints:
        label = waypoint.source or waypoint.kind
        if label and label not in labels:
            labels.append(label)
    for event in item.recent_events:
        kind = _string(event.get("kind"))
        sequence = _event_sequence(event)
        if kind and kind != "thread_waypoint":
            label = f"{kind}:{sequence}" if sequence else kind
            if label not in labels:
                labels.append(label)
    return tuple(labels)


class DeterministicBearingCompiler:
    """Compile thread bearings with optional local-model help and safe fallback."""

    def __init__(
        self,
        *,
        model_client: ThreadBearingModelClient | None = None,
        max_bearing_chars: int = 180,
        max_activity_chars: int = 96,
    ) -> None:
        self.model_client = model_client
        self.max_bearing_chars = max_bearing_chars
        self.max_activity_chars = max_activity_chars

    def build_prompt(self, item: ThreadBearingInput) -> str:
        waypoint_lines = [
            (
                f"- {waypoint.kind} seq={waypoint.sequence} "
                f"source={waypoint.source or 'unknown'}: {_clamp(waypoint.text, 420)}"
            )
            for waypoint in item.waypoints[-4:]
        ] or ["- none"]
        event_lines = [
            (
                f"- {_string(event.get('kind')) or 'event'} "
                f"seq={_event_sequence(event)}: {_clamp(_string(event.get('text')) or json.dumps(_event_data(event), sort_keys=True, default=str), 240)}"
            )
            for event in sorted(item.recent_events, key=_event_sequence)[-6:]
        ] or ["- none"]
        return "\n".join(
            [
                "Compile a compact Agent Shell operator bearing.",
                "Keep it source-attached, status-shaped, and non-surveillant.",
                "Do not replace backend latest_response or redefine card display rules.",
                f"THREAD {item.thread_id}",
                f"PROVIDER {item.provider}",
                f"TITLE {item.title or '(none)'}",
                f"CURRENT_BEARING {_clamp(item.current_bearing, 240) or '(none)'}",
                f"CURRENT_ACTIVITY {_clamp(item.current_activity_line, 160) or '(none)'}",
                "WAYPOINTS",
                *waypoint_lines,
                "RECENT_EVENTS",
                *event_lines,
                "Return JSON with keys bearing and activity_line.",
            ]
        )

    def compile(self, item: ThreadBearingInput) -> CompiledThreadBearing:
        prompt = self.build_prompt(item)
        sources = _source_labels(item)
        activity_line = _activity_line_from_events(item.recent_events) or item.current_activity_line
        bearing = self._fallback_bearing(item)
        inference_status = "fallback"
        if self.model_client is not None:
            try:
                result = self.model_client.compile_bearing(prompt)
            except Exception:
                result = None
            model_bearing, model_activity = self._parse_model_result(result)
            if model_bearing:
                bearing = model_bearing
                inference_status = "model"
            if model_activity:
                activity_line = model_activity

        source_prefix = f"[{item.provider or 'agent'}/{item.thread_id or 'unknown'}]"
        if not bearing.startswith(source_prefix):
            bearing = f"{source_prefix} {bearing}"
        return CompiledThreadBearing(
            thread_id=item.thread_id,
            provider=item.provider,
            bearing=_clamp(bearing, self.max_bearing_chars),
            activity_line=_clamp(activity_line or "No recent activity", self.max_activity_chars),
            updated_sequence=item.updated_sequence,
            sources=sources,
            prompt=prompt,
            inference_status=inference_status,
            input_signature=item.input_signature(),
        )

    def _fallback_bearing(self, item: ThreadBearingInput) -> str:
        if item.waypoints:
            waypoint = max(item.waypoints, key=lambda candidate: candidate.sequence)
            text = waypoint.text
        else:
            text = item.current_bearing or item.title or "No waypoint packet yet"
        title = item.title.strip()
        if title:
            text = f"{title}: {text}"
        return _clamp(text, self.max_bearing_chars)

    @staticmethod
    def _parse_model_result(result: dict[str, str] | str | None) -> tuple[str, str]:
        if isinstance(result, str):
            return result, ""
        if isinstance(result, dict):
            return _string(result.get("bearing")), _string(result.get("activity_line"))
        return "", ""


@dataclass
class _PendingThreadBearing:
    item: ThreadBearingInput
    first_seen_at: float


class ThreadBearingScheduler:
    """Coalesce fast thread events and compile only changed packet state."""

    def __init__(
        self,
        *,
        compiler: ThreadBearingCompiler | None = None,
        min_interval_s: float = 2.0,
        max_batch_size: int = 8,
        clock: Any = time.monotonic,
    ) -> None:
        self.compiler = compiler or DeterministicBearingCompiler()
        self.min_interval_s = min_interval_s
        self.max_batch_size = max(1, max_batch_size)
        self.clock = clock
        self._pending: dict[str, _PendingThreadBearing] = {}
        self._last_compiled_signature: dict[str, str] = {}

    def enqueue(self, item: ThreadBearingInput) -> None:
        signature = item.input_signature()
        if self._last_compiled_signature.get(item.thread_id) == signature:
            return
        existing = self._pending.get(item.thread_id)
        first_seen_at = existing.first_seen_at if existing is not None else float(self.clock())
        self._pending[item.thread_id] = _PendingThreadBearing(
            item=item,
            first_seen_at=first_seen_at,
        )

    def flush_due(self) -> list[CompiledThreadBearing]:
        now = float(self.clock())
        due = [
            pending
            for pending in self._pending.values()
            if now - pending.first_seen_at >= self.min_interval_s
        ]
        due.sort(key=lambda pending: (pending.item.updated_sequence, pending.item.thread_id))
        compiled: list[CompiledThreadBearing] = []
        for pending in due[: self.max_batch_size]:
            item = pending.item
            signature = item.input_signature()
            self._pending.pop(item.thread_id, None)
            if self._last_compiled_signature.get(item.thread_id) == signature:
                continue
            bearing = self.compiler.compile(item)
            self._last_compiled_signature[item.thread_id] = signature
            compiled.append(bearing)
        return compiled

    def pending_count(self) -> int:
        return len(self._pending)


def thread_bearing_input_from_session(session: dict[str, Any]) -> ThreadBearingInput:
    events = tuple(event for event in session.get("backend_events", ()) if isinstance(event, dict))
    waypoints: list[AgentThreadWaypoint] = []
    for event in events:
        if _string(event.get("kind")) != "thread_waypoint":
            continue
        data = _event_data(event)
        text = _string(event.get("text"))
        if not text:
            continue
        waypoints.append(
            AgentThreadWaypoint(
                kind=_string(data.get("kind")) or "waypoint",
                text=text,
                sequence=_event_sequence(event),
                source=_string(data.get("source")) or "thread_waypoint",
            )
        )
    thread_card = session.get("thread_card")
    card = thread_card if isinstance(thread_card, dict) else {}
    return ThreadBearingInput(
        thread_id=_string(session.get("id")),
        provider=_string(session.get("provider")),
        title=_string(card.get("title")),
        waypoints=tuple(waypoints),
        recent_events=events,
        current_bearing=_string(card.get("bearing")),
        current_activity_line=_string(card.get("activity_line")),
    )
