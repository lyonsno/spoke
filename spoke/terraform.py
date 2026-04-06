"""Terraform: epistaxis topoi parser and HUD panel.

Parses scoped local state from an epistaxis project note and renders
a scrollable sidebar showing active topoi with semeion names, status,
and current intent.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_EPISTAXIS_NOTE = (
    Path.home() / "dev" / "epistaxis" / "projects" / "spoke" / "epistaxis.md"
)


@dataclass
class Topos:
    """A single parsed topos from epistaxis scoped local state."""

    id: str
    semeion: str | None = None
    branch: str | None = None
    resume_cmd: str | None = None
    status: str | None = None
    temperature: str | None = None
    attractors: list[str] = field(default_factory=list)
    machine: str | None = None
    tool: str | None = None
    observed: str | None = None
    all_semeions: list[str] = field(default_factory=list)


_TAG_PATTERNS = re.compile(
    r"^(reboot|consult |see |pull |check |blocked|shared-surface|leaf \d|"
    r"operator-chord|launcher-teardown|finger-flounder-coverage|"
    r"test-isolation)",
    re.IGNORECASE,
)


def _is_tag_semeion(name: str) -> bool:
    """Return True if this semeion looks like an operational tag, not a name."""
    return bool(_TAG_PATTERNS.search(name.strip()))


def _clean_status(raw: str) -> str:
    """Strip markdown artifacts from status text."""
    s = re.sub(r"\*\*", "", raw)  # bold
    s = re.sub(r"`([^`]*)`", r"\1", s)  # backticks
    s = re.sub(r"~~([^~]*)~~", r"\1", s)  # strikethrough
    return s.strip()


def parse_topoi(text: str) -> list[Topos]:
    """Parse scoped local state entries from an epistaxis note.

    Looks for the ``## Scoped Local State`` section and extracts each
    ``### <id>`` subsection into a :class:`Topos`.
    """
    # Find the scoped local state section
    scoped_match = re.search(
        r"^## Scoped Local State\s*$", text, re.MULTILINE
    )
    if not scoped_match:
        return []

    scoped_start = scoped_match.end()

    # Find the next ## section (or end of file)
    next_section = re.search(r"^## (?!Scoped Local State)", text[scoped_start:], re.MULTILINE)
    scoped_text = text[scoped_start : scoped_start + next_section.start()] if next_section else text[scoped_start:]

    # Split into individual topos entries
    entries = re.split(r"^### ", scoped_text, flags=re.MULTILINE)
    topoi: list[Topos] = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        lines = entry.split("\n")
        topos_id = lines[0].strip()

        # Skip non-topos content (italic notes, plain text, etc.)
        # Topos IDs start with a letter or digit, not underscore
        if not re.match(r"^[a-zA-Z0-9][\w-]*", topos_id):
            continue
        topos = Topos(id=topos_id)

        body = "\n".join(lines[1:])

        # Extract semeion names from [Sēmeion: ...] markers
        # Grab only the name part before any em-dash qualifier
        semeion_matches = re.findall(
            r"\[Sēmeion:\s*`?([^]`—]+)`?\s*(?:—[^]]*)?]", body
        )
        topos.all_semeions = [s.strip() for s in semeion_matches]
        # Pick the first semeion that looks like a name, not an instruction/tag
        for s in topos.all_semeions:
            if not _is_tag_semeion(s):
                topos.semeion = s
                break

        # Extract fields from bullet lines
        for line in lines[1:]:
            line = line.strip()
            if not line.startswith("- "):
                continue
            content = line[2:]

            # Branch
            branch_m = re.search(r"Branch:\s*`([^`]+)`", content)
            if branch_m and not topos.branch:
                topos.branch = branch_m.group(1)

            # Resume / Continuation
            resume_m = re.search(
                r"(?:Resume|Continuation):\s*`([^`]+)`", content
            )
            if resume_m and not topos.resume_cmd:
                topos.resume_cmd = resume_m.group(1)

            # Temperature (may be in backticks or bold)
            temp_m = re.search(r"Temperature:\s*\**`?([^`*\s]+)`?\**", content)
            if temp_m and not topos.temperature:
                topos.temperature = temp_m.group(1)

            # Machine
            machine_m = re.search(r"Machine:\s*`([^`]+)`", content)
            if machine_m and not topos.machine:
                topos.machine = machine_m.group(1)

            # Tool
            tool_m = re.search(r"Tool:\s*`?([^`|]+)`?", content)
            if tool_m and not topos.tool:
                topos.tool = tool_m.group(1).strip()

            # Observed timestamp
            obs_m = re.search(r"Observed:\s*`?([^`\n]+)`?", content)
            if obs_m and not topos.observed:
                topos.observed = obs_m.group(1).strip()

            # Status (the last "Status:" line wins)
            if content.startswith("Status:") or "Status:" in content:
                status_m = re.search(r"Status:\s*\**(.+)", content)
                if status_m:
                    topos.status = _clean_status(status_m.group(1))
                    # Override temperature when status says settled/katástasis,
                    # even if an explicit Temperature: field was set — the status
                    # is the more recent signal.
                    status_lower = topos.status.lower()
                    if ("katástasis" in status_lower
                            or "κατάστασις" in status_lower
                            or "katastasis" in status_lower
                            or "settled" in status_lower):
                        topos.temperature = "katástasis"

            # Attractors
            attr_m = re.search(r"Attractors?:\s*(.+)", content)
            if attr_m and not topos.attractors:
                raw = attr_m.group(1)
                # Split on commas, strip backticks and strikethrough
                parts = re.split(r",\s*", raw)
                for part in parts:
                    clean = re.sub(r"[`~]", "", part).strip()
                    # Remove parenthetical suffixes
                    clean = re.sub(r"\s*\([^)]*\)\s*$", "", clean)
                    if clean:
                        topos.attractors.append(clean)

        topoi.append(topos)

    # Also parse the Katastasis section — entries there are settled regardless
    # of their format. They appear as bullet points: - **id** (date): text...
    kata_match = re.search(r"^## Katastasis\s*$", text, re.MULTILINE)
    if kata_match:
        kata_start = kata_match.end()
        kata_next = re.search(r"^## ", text[kata_start:], re.MULTILINE)
        kata_text = text[kata_start : kata_start + kata_next.start()] if kata_next else text[kata_start:]

        # Known active topos IDs — skip duplicates
        active_ids = {t.id.split(" ")[0].split("—")[0].strip() for t in topoi}

        for m in re.finditer(
            r"^- \*\*([a-zA-Z0-9][\w-]*)\*\*\s*(?:\(([^)]*)\))?:\s*(.*)",
            kata_text,
            re.MULTILINE,
        ):
            raw_id = m.group(1).strip()
            if raw_id in active_ids:
                continue  # already in scoped local state
            date = m.group(2) or ""
            description = m.group(3).strip()
            # Truncate long descriptions
            if len(description) > 80:
                description = description[:77] + "..."
            topos = Topos(
                id=raw_id,
                temperature="katástasis",
                status=description or f"Settled {date}".strip(),
                observed=date or None,
            )
            topoi.append(topos)

    return topoi


_DEFAULT_EPISTAXIS_REPO = Path.home() / "dev" / "epistaxis"
_DEFAULT_EPISTAXIS_REL = "projects/spoke/epistaxis.md"


def _fetch_remote_text(repo: Path, rel_path: str) -> str | None:
    """Fetch a file from origin/main without touching the worktree.

    Runs ``git fetch origin main`` (quiet, fast) then reads the file
    via ``git show origin/main:<path>``. Returns None on any failure.
    """
    try:
        subprocess.run(
            ["git", "-C", str(repo), "fetch", "--quiet", "origin", "main"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        logger.debug("git fetch failed for %s", repo, exc_info=True)
    # Even if fetch fails, try git show — we might have a recent fetch
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"origin/main:{rel_path}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        logger.debug("git show failed for %s:%s", repo, rel_path, exc_info=True)
    return None


def load_topoi(
    path: str | Path | None = None,
) -> list[Topos]:
    """Load and parse topoi from epistaxis.

    By default fetches from ``origin/main`` in the epistaxis repo so the
    HUD always reflects the pushed (shared) state. Falls back to reading
    the local file if the remote fetch fails.

    Parameters
    ----------
    path : str or Path, optional
        Override path to a local epistaxis note (bypasses remote fetch).
    """
    env_override = os.environ.get("SPOKE_EPISTAXIS_NOTE")
    if path or env_override:
        note_path = Path(path or env_override)
        if not note_path.exists():
            logger.warning("epistaxis note not found: %s", note_path)
            return []
        text = note_path.read_text(encoding="utf-8")
        return parse_topoi(text)

    # Try remote first
    text = _fetch_remote_text(_DEFAULT_EPISTAXIS_REPO, _DEFAULT_EPISTAXIS_REL)
    if text:
        return parse_topoi(text)

    # Fall back to local
    local = _DEFAULT_EPISTAXIS_NOTE
    if local.exists():
        logger.info("Remote fetch failed — falling back to local epistaxis")
        return parse_topoi(local.read_text(encoding="utf-8"))

    logger.warning("epistaxis note not found (remote or local)")
    return []


# Temperature sort order: hot first, then warm, cool, cold, katastasis last.
# Unknown temperatures sort between cold and katastasis.
_TEMP_SORT_ORDER = {
    "hot": 0,
    "warm": 1,
    "cool": 2,
    "cold": 3,
    "katástasis": 5,
}
_TEMP_UNKNOWN = 4


def sort_topoi(
    topoi: list[Topos],
    key: str = "temperature",
) -> list[Topos]:
    """Sort topoi by the given key.

    Supported keys:

    - ``"temperature"`` (default): hot → warm → cool → cold → unknown → katastasis
    - ``"semeion"``: alphabetical by display name (semeion or id)
    - ``"machine"``: group by machine, then by temperature within each group
    """
    if key == "temperature":
        return sorted(
            topoi,
            key=lambda t: _TEMP_SORT_ORDER.get(t.temperature or "", _TEMP_UNKNOWN),
        )
    elif key == "semeion":
        return sorted(
            topoi,
            key=lambda t: (t.semeion or t.id).lower(),
        )
    elif key == "machine":
        return sorted(
            topoi,
            key=lambda t: (
                t.machine or "zzz-unknown",
                _TEMP_SORT_ORDER.get(t.temperature or "", _TEMP_UNKNOWN),
            ),
        )
    return topoi


def filter_topoi(
    topoi: list[Topos],
    *,
    hide_katastasis: bool = False,
    machine: str | None = None,
    tool: str | None = None,
    temperature: str | None = None,
) -> list[Topos]:
    """Filter topoi by criteria.

    Parameters
    ----------
    hide_katastasis : bool
        If True, exclude topoi with temperature "katástasis".
    machine : str, optional
        If set, only include topoi from this machine (substring match).
    tool : str, optional
        If set, only include topoi using this tool (substring match, case-insensitive).
    temperature : str, optional
        If set, only include topoi with this exact temperature.
    """
    result = topoi
    if hide_katastasis:
        result = [t for t in result if t.temperature != "katástasis"]
    if machine:
        machine_lower = machine.lower()
        result = [
            t for t in result
            if t.machine and machine_lower in t.machine.lower()
        ]
    if tool:
        tool_lower = tool.lower()
        result = [
            t for t in result
            if t.tool and tool_lower in t.tool.lower()
        ]
    if temperature:
        result = [t for t in result if t.temperature == temperature]
    return result


@dataclass
class AttractorStats:
    """Counts of attractors by status."""

    total: int = 0
    active: int = 0
    unclassified: int = 0
    soak: int = 0
    smoke: int = 0
    katastasis: int = 0


def count_attractors(
    epistaxis_root: str | Path | None = None,
) -> AttractorStats:
    """Count attractors by status from frontmatter.

    Scans both ``attractors/`` (top-level, cross-repo) and
    ``projects/spoke/attractors/`` for ``.md`` files with YAML frontmatter.
    Files without a ``status:`` field are counted as active.
    """
    root = Path(epistaxis_root) if epistaxis_root else Path.home() / "dev" / "epistaxis"
    env_override = os.environ.get("SPOKE_EPISTAXIS_ROOT")
    if env_override:
        root = Path(env_override)

    stats = AttractorStats()
    dirs = [
        root / "attractors",
        root / "projects" / "spoke" / "attractors",
    ]
    seen: set[str] = set()  # dedup by filename

    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.glob("*.md"):
            if f.name in seen:
                continue
            seen.add(f.name)
            stats.total += 1

            # Try to read status from frontmatter
            status = None  # None means no frontmatter status found
            try:
                head = f.read_text(encoding="utf-8", errors="replace")[:500]
                if head.startswith("---"):
                    end = head.find("---", 3)
                    if end != -1:
                        fm = head[3:end]
                        m = re.search(r"^status:\s*(.+)$", fm, re.MULTILINE)
                        if m:
                            status = m.group(1).strip().lower()
            except OSError:
                pass

            if status is None:
                stats.unclassified += 1
            elif status in ("soak", "soaking"):
                stats.soak += 1
            elif status in ("smoke", "smoking"):
                stats.smoke += 1
            elif status in ("katástasis", "katastasis", "settled"):
                stats.katastasis += 1
            else:
                stats.active += 1

    return stats


def format_topos_summary(topos: Topos) -> str:
    """One-line summary for display."""
    name = topos.semeion or topos.id
    temp = f" [{topos.temperature}]" if topos.temperature else ""
    status_snippet = ""
    if topos.status:
        # Take first sentence or first 80 chars
        first_sentence = topos.status.split(". ")[0]
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        status_snippet = f" — {first_sentence}"
    return f"{name}{temp}{status_snippet}"


def disambiguated_name(topos: Topos) -> str:
    """Display name with machine/tool suffix when semeion is shared or absent."""
    name = topos.semeion or topos.id
    parts = []
    if topos.machine:
        # Short hostname: "MacBook-Pro-2.local" → "MacBook-Pro-2"
        parts.append(topos.machine.split(".")[0])
    if topos.tool:
        # Short tool: "Claude Code (Opus 4.6)" → "Claude Code"
        parts.append(topos.tool.split("(")[0].strip())
    suffix = f"  ({', '.join(parts)})" if parts else ""
    return f"{name}{suffix}"
