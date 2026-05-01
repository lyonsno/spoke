#!/usr/bin/env python3
"""Build an optical-shell residency budget from JSONL measurement samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spoke.optical_shell_baseline import build_optical_shell_budget, sample_from_mapping


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Spoke optical-shell baseline budget JSON")
    parser.add_argument("--input", required=True, type=Path, help="JSONL sample file")
    parser.add_argument("--output", type=Path, help="Budget JSON output path")
    parser.add_argument("--machine", required=True, help="Machine name used for the run")
    parser.add_argument("--display-hz", default=120.0, type=float, help="Display refresh rate")
    parser.add_argument("--branch", help="Spoke branch measured")
    parser.add_argument("--commit", help="Spoke commit measured")
    parser.add_argument("--captured-at", help="Capture timestamp")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any packet-required scenario is missing",
    )
    return parser.parse_args(argv)


def _load_samples(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            yield sample_from_mapping(payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    budget = build_optical_shell_budget(
        _load_samples(args.input),
        machine=args.machine,
        display_hz=args.display_hz,
        branch=args.branch,
        commit=args.commit,
        captured_at=args.captured_at,
    )
    text = json.dumps(budget, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    if args.require_complete and budget["missing_scenarios"]:
        sys.stderr.write(
            "missing optical-shell baseline scenarios: "
            + ", ".join(budget["missing_scenarios"])
            + "\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
