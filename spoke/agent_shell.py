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
    if re.search(r"\b(?:switch|change|route|use)\s+(?:to\s+)?gemini(?:\s+cli)?\b", normalized):
        return "gemini-cli"
    match = re.search(r"\b(?:switch|change|route|use)\s+(?:to\s+)?(codex)\b", normalized)
    if match:
        return match.group(1)
    return None


def _cancel_active_run(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", _normalized_text(text))
    if not tokens:
        return False
    verbs = {"cancel", "stop", "interrupt", "kill"}
    objects = {
        ("agent",),
        ("backend",),
        ("run",),
        ("session",),
        ("generation",),
        ("agent", "run"),
        ("agent", "session"),
        ("agent", "cli"),
        ("agent", "cli", "run"),
        ("agent", "cli", "session"),
        ("backend", "run"),
        ("backend", "session"),
        ("cli", "run"),
        ("cli", "session"),
    }
    fillers = {"the", "this", "active", "current", "running", "selected"}
    if tokens[0] not in verbs:
        return False
    rest = [token for token in tokens[1:] if token not in fillers]
    return tuple(rest) in objects


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

    if _cancel_active_run(stripped):
        return AgentShellRoutingDecision(
            kind="mode_control",
            text=stripped,
            provider=state.provider,
            spoke_session_id=state.spoke_session_id,
            provider_session_id=state.provider_session_id,
            cwd=state.cwd,
            control_action="cancel_active_run",
        )

    return AgentShellRoutingDecision(
        kind="provider_message",
        text=stripped,
        provider=state.provider,
        spoke_session_id=state.spoke_session_id,
        provider_session_id=state.provider_session_id,
        cwd=state.cwd,
    )
