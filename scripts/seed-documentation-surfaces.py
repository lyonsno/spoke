#!/usr/bin/env python3
"""Seed documentation surface routes from existing markdown surfaces.

Usage:
    uv run python scripts/seed-documentation-surfaces.py docs/*.md

The script adds missing entries to docs/documentation_surfaces.toml by turning
second-level markdown headings into draft capability routes. Existing manifest
entries are preserved so hand-tuned routes are not overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spoke.documentation_surfaces import load_manifest, merge_seeded_entries, render_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="Markdown files to scan for capability sections.")
    parser.add_argument(
        "--output",
        default="docs/documentation_surfaces.toml",
        help="Manifest path to update. Defaults to docs/documentation_surfaces.toml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merged manifest instead of writing it back to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    manifest = load_manifest(output_path)

    for source in args.sources:
        source_path = Path(source)
        manifest = merge_seeded_entries(
            manifest=manifest,
            source_path=source_path,
            markdown=source_path.read_text(encoding="utf-8"),
        )

    rendered = render_manifest(manifest)
    if args.dry_run:
        print(rendered, end="")
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Updated {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
