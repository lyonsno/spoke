#!/usr/bin/env python3
"""Converge glossary builder — produce a cheap one-liner index of all attractors.

The two-layer attractor architecture:
- Full attractors: rich markdown files (epistaxis project + personal)
- Glossary: single JSON file with one-line summaries, cheap to load into context

The glossary is the working surface for guided compaction. Full attractors are
consulted only when the model needs depth on a specific matched attractor.

Usage:
    uv run scripts/converge-build-glossary.py            # rebuild glossary
    uv run scripts/converge-build-glossary.py --print    # print to stdout instead of writing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ATTRACTOR_SOURCES = [
    ("project", Path.home() / "dev" / "epistaxis" / "attractors"),
    ("personal", Path.home() / ".config" / "spoke" / "attractors"),
]
_GLOSSARY_PATH = Path.home() / ".config" / "spoke" / "attractor-glossary.json"


def _extract_one_liner(path: Path) -> str:
    """Extract a one-line summary from an attractor markdown file.

    Strategy: use the first bullet's text (after '- Stimulus:' or '- Evidence:'),
    or failing that the first non-heading non-empty line, truncated to 120 chars.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "(unreadable)"

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip leading list marker and field label
        cleaned = re.sub(r"^-\s*(?:Stimulus|Evidence|Source|Satisfaction|Strength|Observed):\s*", "", line)
        if cleaned:
            return cleaned[:120]

    return "(empty)"


def build_glossary() -> list[dict]:
    """Build the glossary from all attractor sources."""
    entries = []
    for source_label, adir in _ATTRACTOR_SOURCES:
        if not adir.is_dir():
            continue
        for f in sorted(adir.iterdir()):
            if f.is_file() and f.suffix == ".md":
                entries.append({
                    "source": source_label,
                    "slug": f.stem,
                    "summary": _extract_one_liner(f),
                })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Converge glossary builder")
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="Print glossary to stdout instead of writing to disk")
    args = parser.parse_args()

    entries = build_glossary()

    if args.print_only:
        print(json.dumps(entries, indent=2))
        print(f"\n({len(entries)} entries)", file=__import__("sys").stderr)
        return

    _GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GLOSSARY_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Glossary written: {_GLOSSARY_PATH} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
