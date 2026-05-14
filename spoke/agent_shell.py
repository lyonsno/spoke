"""Modal Agent Shell routing.

Agent Shell mode routes operator input to a selected local-auth agent backend
session while keeping explicit Spoke-owned control input under the operator
shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentShellState:
    active: bool
    provider: str | None = None
    spoke_session_id: str | None = None
    provider_session_id: str | None = None
    cwd: str | None = None


@dataclass(frozen=True)
class AgentShellRoutingDecision:
    kind: str
    text: str
    provider: str | None = None
    spoke_session_id: str | None = None
    provider_session_id: str | None = None
    cwd: str | None = None
    control_action: str | None = None


def _normalized_text(text: str) -> str:
    return text.casefold().strip()


def _provider_switch(text: str) -> str | None:
    normalized = _normalized_text(text)
    if re.search(r"\b(?:switch|change|route|use)\s+(?:to\s+)?claude\s+code\b", normalized):
        return "claude-code"
    match = re.search(r"\b(?:switch|change|route|use)\s+(?:to\s+)?(codex)\b", normalized)
    if match:
        return match.group(1)
    return None


def route_agent_shell_input(
    text: str,
    state: AgentShellState | None,
) -> AgentShellRoutingDecision:
    """Classify input under Agent Shell mode."""
    stripped = text.strip() if isinstance(text, str) else ""
    if state is None or not state.active:
        return AgentShellRoutingDecision(kind="normal_assistant", text=stripped)

    provider = _provider_switch(stripped)
    if provider is not None:
        return AgentShellRoutingDecision(
            kind="mode_control",
            text=stripped,
            provider=provider,
            control_action="switch_provider",
        )

    return AgentShellRoutingDecision(
        kind="provider_message",
        text=stripped,
        provider=state.provider,
        spoke_session_id=state.spoke_session_id,
        provider_session_id=state.provider_session_id,
        cwd=state.cwd,
    )
