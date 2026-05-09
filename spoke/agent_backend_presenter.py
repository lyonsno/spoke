"""Presentation policy for local agent backend events.

This module is intentionally pure so the Agent Shell, assistant operator, and
future scrollback surfaces can share one compact event contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentBackendPresentationState:
    message_text_by_id: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    model: str = ""
    five_hour_percent: float | None = None
    seven_day_percent: float | None = None
    plan_type: str = ""
    topos_name: str = ""


@dataclass(frozen=True)
class AgentBackendPresentation:
    kind: str
    text: str = ""
    active: bool | None = None


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_percent(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"


def _metadata_footer_text(state: AgentBackendPresentationState) -> str:
    parts: list[str] = []
    if state.model:
        parts.append(f"model {state.model}")
    if state.cwd:
        parts.append(f"cwd {state.cwd}")
    five_hour = _format_percent(state.five_hour_percent)
    if five_hour:
        parts.append(f"5h {five_hour}")
    seven_day = _format_percent(state.seven_day_percent)
    if seven_day:
        parts.append(f"7d {seven_day}")
    if state.plan_type:
        parts.append(state.plan_type)
    return " | ".join(parts)


_TOPOS_PATTERNS = (
    re.compile(
        r"\btopos\s*[:=]\s*`?([A-Za-z0-9][A-Za-z0-9_.-]*-[A-Za-z0-9_.-]*)`?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:created|opened|registered|recorded)\s+(?:a\s+)?topos\s+`?([A-Za-z0-9][A-Za-z0-9_.-]*-[A-Za-z0-9_.-]*)`?",
        re.IGNORECASE,
    ),
)


def _extract_topos_name(text: str) -> str:
    for pattern in _TOPOS_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).rstrip(".,;:")
    return ""


def _item_id(data: dict[str, Any], fallback: str) -> str:
    item_id = data.get("id")
    return item_id if isinstance(item_id, str) and item_id else fallback


def _agent_message_action(
    event: dict[str, Any],
    data: dict[str, Any],
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    item_id = _item_id(data, str(event.get("sequence", "")))
    text = _string(data.get("text")) or _string(event.get("text"))
    previous = state.message_text_by_id.get(item_id, "")
    if not text or text == previous:
        return []
    delta = text[len(previous):] if text.startswith(previous) else text
    state.message_text_by_id[item_id] = text
    return [AgentBackendPresentation(kind="response_delta", text=delta)]


def _command_execution_actions(data: dict[str, Any]) -> list[AgentBackendPresentation]:
    command = _string(data.get("command")).strip()
    status = _string(data.get("status")).strip()
    if not command:
        return []
    if status == "in_progress":
        return [
            AgentBackendPresentation(kind="tool_start"),
            AgentBackendPresentation(kind="status", text=f"Codex running: {command}"),
        ]
    if status == "completed":
        return [
            AgentBackendPresentation(kind="tool_end"),
            AgentBackendPresentation(kind="status", text="Codex command completed"),
        ]
    if status == "failed":
        return [
            AgentBackendPresentation(kind="tool_end"),
            AgentBackendPresentation(kind="status", text="Codex command failed"),
        ]
    return [AgentBackendPresentation(kind="status", text=f"Codex command: {command}")]


def _mcp_tool_actions(data: dict[str, Any]) -> list[AgentBackendPresentation]:
    server = _string(data.get("server")).strip()
    tool = _string(data.get("tool")).strip()
    label = ".".join(part for part in (server, tool) if part) or "tool"
    status = _string(data.get("status")).strip()
    if status == "in_progress":
        return [
            AgentBackendPresentation(kind="tool_start"),
            AgentBackendPresentation(kind="status", text=f"Codex tool: {label}"),
        ]
    if status == "completed":
        return [
            AgentBackendPresentation(kind="tool_end"),
            AgentBackendPresentation(kind="status", text=f"Codex tool completed: {label}"),
        ]
    if status == "failed":
        return [
            AgentBackendPresentation(kind="tool_end"),
            AgentBackendPresentation(kind="status", text=f"Codex tool failed: {label}"),
        ]
    return [AgentBackendPresentation(kind="status", text=f"Codex tool: {label}")]


def _file_change_actions(data: dict[str, Any]) -> list[AgentBackendPresentation]:
    changes = data.get("changes")
    if not isinstance(changes, list) or not changes:
        return []
    count = len([change for change in changes if isinstance(change, dict)])
    if count <= 0:
        return []
    noun = "file" if count == 1 else "files"
    return [AgentBackendPresentation(kind="status", text=f"Codex edited {count} {noun}")]


def _metadata_actions(
    data: dict[str, Any],
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    cwd = _string(data.get("cwd")).strip()
    if cwd:
        state.cwd = cwd
    model = _string(data.get("model")).strip()
    if model:
        state.model = model
    footer = _metadata_footer_text(state)
    if not footer:
        return []
    return [AgentBackendPresentation(kind="metadata_footer", text=footer)]


def _usage_actions(
    data: dict[str, Any],
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    five_hour = _number(data.get("five_hour_percent"))
    if five_hour is not None:
        state.five_hour_percent = five_hour
    seven_day = _number(data.get("seven_day_percent"))
    if seven_day is not None:
        state.seven_day_percent = seven_day
    plan_type = _string(data.get("plan_type")).strip()
    if plan_type:
        state.plan_type = plan_type
    footer = _metadata_footer_text(state)
    if not footer:
        return []
    return [AgentBackendPresentation(kind="metadata_footer", text=footer)]


def _topos_actions(
    text: str,
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    name = _extract_topos_name(text)
    if not name or name == state.topos_name:
        return []
    state.topos_name = name
    return [AgentBackendPresentation(kind="metadata_header", text=f"Topos: {name}")]


def present_backend_events(
    events: list[dict[str, Any]],
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    actions: list[AgentBackendPresentation] = []
    for event in events:
        kind = _string(event.get("kind"))
        data = _event_data(event)
        identity_text = "\n".join(
            part
            for part in (
                _string(event.get("text")),
                _string(data.get("text")),
                _string(data.get("aggregated_output")),
                _string(data.get("message")),
            )
            if part
        )
        if identity_text:
            actions.extend(_topos_actions(identity_text, state))
        if kind == "agent_message":
            actions.extend(_agent_message_action(event, data, state))
        elif kind == "reasoning":
            text = _string(data.get("text")) or _string(event.get("text"))
            if text:
                actions.append(AgentBackendPresentation(kind="narrator_summary", text=text))
        elif kind == "command_execution":
            actions.extend(_command_execution_actions(data))
        elif kind == "mcp_tool_call":
            actions.extend(_mcp_tool_actions(data))
        elif kind == "file_change":
            actions.extend(_file_change_actions(data))
        elif kind == "session_metadata":
            actions.extend(_metadata_actions(data, state))
        elif kind == "usage_limits":
            actions.extend(_usage_actions(data, state))
        elif kind == "topos_identity":
            name = _string(data.get("name")).strip()
            if name:
                actions.extend(_topos_actions(f"Topos: {name}", state))
        elif kind == "web_search":
            query = _string(data.get("query")).strip()
            if query:
                actions.append(
                    AgentBackendPresentation(kind="status", text=f"Codex web search: {query}")
                )
        elif kind == "error":
            text = _string(data.get("message")) or _string(event.get("text"))
            if text:
                actions.append(AgentBackendPresentation(kind="error", text=text))
    return actions


def present_backend_liveness(label: str) -> list[AgentBackendPresentation]:
    provider = label.strip() or "Agent"
    return [
        AgentBackendPresentation(kind="narrator_summary", text=f"{provider} thinking"),
        AgentBackendPresentation(kind="narrator_shimmer", active=True),
    ]


def present_backend_idle() -> list[AgentBackendPresentation]:
    return [AgentBackendPresentation(kind="narrator_shimmer", active=False)]
