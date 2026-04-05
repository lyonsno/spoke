"""Terraform: epistaxis topoi parser and HUD panel.

Parses scoped local state from an epistaxis project note and renders
a scrollable sidebar showing active topoi with semeion names, status,
and current intent.
"""

from __future__ import annotations

import logging
import os
import re
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
    all_semeions: list[str] = field(default_factory=list)


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
        if topos.all_semeions:
            topos.semeion = topos.all_semeions[0]

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

            # Status (the last "Status:" line wins)
            if content.startswith("Status:") or "Status:" in content:
                status_m = re.search(r"Status:\s*\**(.+)", content)
                if status_m:
                    # Strip bold markers
                    topos.status = re.sub(r"\*\*", "", status_m.group(1)).strip()

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

    return topoi


def load_topoi(
    path: str | Path | None = None,
) -> list[Topos]:
    """Load and parse topoi from an epistaxis note file.

    Parameters
    ----------
    path : str or Path, optional
        Override path to the epistaxis note. Defaults to
        ``~/dev/epistaxis/projects/spoke/epistaxis.md``.
    """
    note_path = Path(path) if path else _DEFAULT_EPISTAXIS_NOTE
    env_override = os.environ.get("SPOKE_EPISTAXIS_NOTE")
    if env_override:
        note_path = Path(env_override)

    if not note_path.exists():
        logger.warning("epistaxis note not found: %s", note_path)
        return []

    text = note_path.read_text(encoding="utf-8")
    return parse_topoi(text)


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
            status = "active"  # default if no frontmatter
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

            if status in ("soak", "soaking"):
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
