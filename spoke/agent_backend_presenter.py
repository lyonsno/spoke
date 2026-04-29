"""Presentation policy for local agent backend events.

This module is intentionally pure so the Agent Shell, assistant operator, and
future scrollback surfaces can share one compact event contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentBackendPresentationState:
    message_text_by_id: dict[str, str] = field(default_factory=dict)


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


def present_backend_events(
    events: list[dict[str, Any]],
    state: AgentBackendPresentationState,
) -> list[AgentBackendPresentation]:
    actions: list[AgentBackendPresentation] = []
    for event in events:
        kind = _string(event.get("kind"))
        data = _event_data(event)
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
