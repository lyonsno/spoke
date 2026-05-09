"""Read-only Epistaxis identity resolution for Agent Shell sessions."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class AgentShellIdentity:
    topos_name: str
    source: str
    confidence: str


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


def _default_epistaxis_root() -> Path:
    return Path.home() / "dev" / "epistaxis"


def _ensure_epistaxis_import_path(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _load_topoi(epistaxis_root: Path) -> list[Any]:
    _ensure_epistaxis_import_path(epistaxis_root)
    from tools.epistaxis_lysis import build_topos_index

    return build_topos_index(epistaxis_root)


def _load_archive_session(
    provider: str,
    session_id: str,
    epistaxis_root: Path,
) -> dict[str, Any] | None:
    if provider != "codex":
        return None
    _ensure_epistaxis_import_path(epistaxis_root)
    from tools.orwell_anakrisis import find_codex_rollouts, parse_codex_rollout

    matches = find_codex_rollouts(session_id=session_id)
    if not matches:
        return None
    return parse_codex_rollout(max(matches, key=lambda path: path.stat().st_mtime))


def _resolve_archive_session(
    session: dict[str, Any],
    topoi: list[Any],
    epistaxis_root: Path,
) -> Any | None:
    _ensure_epistaxis_import_path(epistaxis_root)
    from tools.epistaxis_archive import build_session_meta, resolve_tab_via_archive

    return resolve_tab_via_archive(build_session_meta(session), topoi)


def _clean_path(value: str) -> Path | None:
    if not value:
        return None
    try:
        return Path(value).expanduser().resolve()
    except OSError:
        return Path(value).expanduser()


def _unique_identity(
    matches: list[Any],
    *,
    source: str,
    confidence: str,
) -> AgentShellIdentity | None:
    names = sorted({str(getattr(match, "name", "")).strip() for match in matches})
    names = [name for name in names if name]
    if len(names) != 1:
        return None
    return AgentShellIdentity(
        topos_name=names[0],
        source=source,
        confidence=confidence,
    )


def _identity_from_session_id(
    topoi: list[Any],
    provider_session_id: str | None,
) -> AgentShellIdentity | None:
    if not provider_session_id:
        return None
    matches = [
        entry
        for entry in topoi
        if str(getattr(entry, "session_id", "")).strip() == provider_session_id
    ]
    return _unique_identity(
        matches,
        source="epistaxis-session-id",
        confidence="exact",
    )


def _identity_from_worktree(
    topoi: list[Any],
    cwd: str | None,
) -> AgentShellIdentity | None:
    cwd_path = _clean_path(cwd or "")
    if cwd_path is None:
        return None
    matches: list[Any] = []
    for entry in topoi:
        worktree = str(getattr(entry, "worktree", "")).strip()
        worktree_path = _clean_path(worktree)
        if worktree_path is None:
            continue
        if cwd_path == worktree_path:
            matches.append(entry)
            continue
        try:
            cwd_path.relative_to(worktree_path)
        except ValueError:
            continue
        matches.append(entry)
    return _unique_identity(
        matches,
        source="epistaxis-worktree",
        confidence="exact",
    )


def _extract_transcript_topos_name(text: str) -> str:
    for pattern in _TOPOS_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).rstrip(".,;:")
    return ""


def _identity_from_archive(
    *,
    provider: str,
    provider_session_id: str | None,
    topoi: list[Any],
    epistaxis_root: Path,
    archive_session_loader: Callable[[str, str, Path], dict[str, Any] | None],
    archive_resolver: Callable[[dict[str, Any], list[Any], Path], Any | None],
) -> AgentShellIdentity | None:
    if not provider_session_id:
        return None
    session = archive_session_loader(provider, provider_session_id, epistaxis_root)
    if not session:
        return None
    match = archive_resolver(session, topoi, epistaxis_root)
    name = str(getattr(match, "name", "")).strip() if match is not None else ""
    if not name:
        return None
    return AgentShellIdentity(
        topos_name=name,
        source="epistaxis-archive-metadata",
        confidence="semantic",
    )


def resolve_agent_shell_identity(
    *,
    provider: str,
    provider_session_id: str | None,
    cwd: str | None,
    epistaxis_root: Path | None = None,
    transcript_text: str = "",
    topoi_loader: Callable[[Path], list[Any]] = _load_topoi,
    archive_session_loader: Callable[
        [str, str, Path], dict[str, Any] | None
    ] = _load_archive_session,
    archive_resolver: Callable[
        [dict[str, Any], list[Any], Path], Any | None
    ] = _resolve_archive_session,
) -> AgentShellIdentity | None:
    """Resolve the owning Epistaxis topos for an Agent Shell backend session.

    The authoritative path is Epistaxis state first, then Anakrisis/archive
    metadata. Transcript regex is retained only as a weak display fallback for
    smoke sessions that mention their new topos before Epistaxis state is
    indexed or exact identity is available.
    """
    root = epistaxis_root or _default_epistaxis_root()
    try:
        topoi = topoi_loader(root)
    except Exception:
        topoi = []

    identity = _identity_from_session_id(topoi, provider_session_id)
    if identity is not None:
        return identity

    identity = _identity_from_worktree(topoi, cwd)
    if identity is not None:
        return identity

    try:
        identity = _identity_from_archive(
            provider=provider,
            provider_session_id=provider_session_id,
            topoi=topoi,
            epistaxis_root=root,
            archive_session_loader=archive_session_loader,
            archive_resolver=archive_resolver,
        )
    except Exception:
        identity = None
    if identity is not None:
        return identity

    fallback = _extract_transcript_topos_name(transcript_text)
    if fallback:
        return AgentShellIdentity(
            topos_name=fallback,
            source="transcript-regex-fallback",
            confidence="weak",
        )
    return None

