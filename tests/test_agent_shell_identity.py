"""Tests for Epistaxis-backed Agent Shell identity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Topos:
    name: str
    repo: str = "spoke"
    branch: str = ""
    worktree: str = ""
    session_id: str = ""
    content: str = ""
    lifecycle: str = "active"


def test_resolver_prefers_epistaxis_session_id_over_transcript_regex():
    from spoke.agent_shell_identity import (
        AgentShellIdentity,
        resolve_agent_shell_identity,
    )

    identity = resolve_agent_shell_identity(
        provider="codex",
        provider_session_id="codex-thread-1",
        cwd="/private/tmp/spoke-agent-sdk-operator-shell-0428",
        topoi_loader=lambda _root: [
            _Topos(name="codex-real-epistaxis-topos", session_id="codex-thread-1")
        ],
        archive_session_loader=lambda *_args, **_kwargs: None,
        archive_resolver=lambda *_args, **_kwargs: None,
        transcript_text="Created Topos: regex-fallback-wrong",
    )

    assert identity == AgentShellIdentity(
        topos_name="codex-real-epistaxis-topos",
        source="epistaxis-session-id",
        confidence="exact",
    )


def test_resolver_uses_epistaxis_worktree_when_session_id_is_absent():
    from spoke.agent_shell_identity import resolve_agent_shell_identity

    identity = resolve_agent_shell_identity(
        provider="codex",
        provider_session_id=None,
        cwd="/private/tmp/spoke-agent-sdk-operator-shell-0428/subdir",
        topoi_loader=lambda _root: [
            _Topos(
                name="codex-worktree-owned-topos",
                worktree="/private/tmp/spoke-agent-sdk-operator-shell-0428",
            )
        ],
        archive_session_loader=lambda *_args, **_kwargs: None,
        archive_resolver=lambda *_args, **_kwargs: None,
    )

    assert identity is not None
    assert identity.topos_name == "codex-worktree-owned-topos"
    assert identity.source == "epistaxis-worktree"
    assert identity.confidence == "exact"


def test_resolver_uses_anakrisis_archive_metadata_before_regex_fallback():
    from spoke.agent_shell_identity import resolve_agent_shell_identity

    archive_calls = []

    def archive_session_loader(provider, session_id, epistaxis_root):
        archive_calls.append((provider, session_id, epistaxis_root))
        return {
            "tool": "codex",
            "session_id": session_id,
            "repo": "spoke",
            "git_branch": "cc/agent-sdk-operator-shell-0428",
            "cwd": "/private/tmp/spoke-agent-sdk-operator-shell-0428",
            "events": [
                {"role": "user", "text": "repair agent shell identity"}
            ],
            "event_count": 1,
        }

    def archive_resolver(session, topoi, epistaxis_root):
        assert session["session_id"] == "codex-thread-archive"
        assert topoi[0].name == "codex-archive-owned-topos"
        return topoi[0]

    identity = resolve_agent_shell_identity(
        provider="codex",
        provider_session_id="codex-thread-archive",
        cwd="/Users/noahlyons/dev/spoke",
        epistaxis_root=Path("/tmp/epistaxis"),
        topoi_loader=lambda _root: [_Topos(name="codex-archive-owned-topos")],
        archive_session_loader=archive_session_loader,
        archive_resolver=archive_resolver,
        transcript_text="Created Topos: regex-fallback-wrong",
    )

    assert archive_calls == [
        ("codex", "codex-thread-archive", Path("/tmp/epistaxis"))
    ]
    assert identity is not None
    assert identity.topos_name == "codex-archive-owned-topos"
    assert identity.source == "epistaxis-archive-metadata"
    assert identity.confidence == "semantic"


def test_resolver_allows_regex_only_as_weak_fallback():
    from spoke.agent_shell_identity import resolve_agent_shell_identity

    identity = resolve_agent_shell_identity(
        provider="codex",
        provider_session_id=None,
        cwd="/tmp/no-match",
        topoi_loader=lambda _root: [],
        archive_session_loader=lambda *_args, **_kwargs: None,
        archive_resolver=lambda *_args, **_kwargs: None,
        transcript_text="Created Topos: regex-fallback-topos",
    )

    assert identity is not None
    assert identity.topos_name == "regex-fallback-topos"
    assert identity.source == "transcript-regex-fallback"
    assert identity.confidence == "weak"

