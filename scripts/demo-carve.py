#!/usr/bin/env python3
"""Demo carve harness: feed turns through a real TurnCarver pointed at temp
directories and inspect what it produces.

Runs the full pipeline: context buffer, cadence, four passes, beast filter,
bearing updates, file writes. All output goes to a temp directory tree.

Usage:
    uv run scripts/demo-carve.py                     # use today's Grapheus log
    uv run scripts/demo-carve.py --date 2026-04-20   # specific day
    uv run scripts/demo-carve.py --turns 10           # limit to N turns
    uv run scripts/demo-carve.py --start 15           # start from entry N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_grapheus_turns(log_date: str | None = None, start: int = 0, limit: int = 0) -> list[dict]:
    """Load user/assistant pairs from a Grapheus daily log, skipping carve requests."""
    d = log_date or date.today().isoformat()
    log_path = Path.home() / "dev" / "grapheus" / "logs" / f"grapheus-{d}.jsonl"
    if not log_path.exists():
        print(f"No Grapheus log at {log_path}", file=sys.stderr)
        return []

    pairs = []
    seen_user = set()  # dedup retries
    with open(log_path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            entry = json.loads(line)
            req = entry.get("request", {})
            msgs = req.get("messages", [])

            # Skip carve requests
            sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
            if any("attractor carver" in (m.get("content") or "").lower() for m in sys_msgs):
                continue
            if any("anamnesis carver" in (m.get("content") or "").lower() for m in sys_msgs):
                continue
            if any("tópos carver" in (m.get("content") or "").lower() for m in sys_msgs):
                continue
            if any("policy observer" in (m.get("content") or "").lower() for m in sys_msgs):
                continue
            if any("species classifier" in (m.get("content") or "").lower() for m in sys_msgs):
                continue
            if any("bearing" in (m.get("content") or "").lower() and "navigational" in (m.get("content") or "").lower() for m in sys_msgs):
                continue

            # Also skip by system prompt keywords that may appear in older carve tools
            if any("substrate carver" in (m.get("content") or "").lower() for m in sys_msgs):
                continue

            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            if not user_msgs:
                continue
            last_user = user_msgs[-1].get("content", "")
            if not last_user.strip():
                continue

            # Skip old carve-script transcripts that appear as user messages
            if last_user.strip().startswith("Here is a transcript of"):
                continue

            # Skip carve prompts that appear as user messages
            if "Identify attractor operations for this utterance" in last_user:
                continue
            if "User utterance from a voice interaction" in last_user:
                continue

            # Dedup retries (same user message)
            user_key = last_user[:200]
            if user_key in seen_user:
                continue
            seen_user.add(user_key)

            resp = entry.get("response", {})
            assembled = resp.get("assembled_content", "")
            if not assembled:
                choices = resp.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    assembled = msg.get("content", "")

            pairs.append({
                "index": i,
                "user": last_user,
                "assistant": assembled,
            })

            if limit and len(pairs) >= limit:
                break

    return pairs


def _print_surface(surface_dir: Path, label: str) -> int:
    """Print contents of a surface directory. Returns entry count."""
    if not surface_dir.is_dir():
        return 0
    files = sorted(f for f in surface_dir.iterdir() if f.is_file() and f.suffix == ".md")
    if not files:
        print(f"  {label}: (empty)")
        return 0
    print(f"  {label}: ({len(files)} entries)")
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        # Show first meaningful line after title
        lines = content.split("\n")
        title = f.stem
        preview = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.strip() and not line.startswith("-") and not line.startswith("#"):
                preview = line.strip()[:100]
                break
        if preview:
            print(f"    {title}: {preview}")
        else:
            print(f"    {title}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Demo carve harness")
    parser.add_argument("--date", type=str, default=None, help="Grapheus log date (YYYY-MM-DD)")
    parser.add_argument("--turns", type=int, default=6, help="Number of turns to process")
    parser.add_argument("--start", type=int, default=0, help="Start from entry N in log")
    args = parser.parse_args()

    turns = _load_grapheus_turns(args.date, start=args.start, limit=args.turns)
    if not turns:
        print("No turns found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(turns)} turns from Grapheus\n")

    # Create temp directory tree
    with tempfile.TemporaryDirectory(prefix="spoke-demo-carve-") as tmpdir:
        tmp = Path(tmpdir)
        attractors_dir = tmp / "attractors"
        anamnesis_dir = tmp / "anamnesis"
        topoi_dir = tmp / "topoi"
        policy_dir = tmp / "policy"
        trace_path = tmp / "converge-trace.jsonl"
        bearing_path = tmp / "converge-bearing.md"

        for d in (attractors_dir, anamnesis_dir, topoi_dir, policy_dir):
            d.mkdir()

        # Monkey-patch module constants to use temp dirs
        import spoke.converge as mod
        original_attractors = mod._ATTRACTORS_DIR
        original_anamnesis = mod._ANAMNESIS_DIR
        original_topoi = mod._TOPOI_DIR
        original_policy = mod._POLICY_DIR
        original_trace = mod._TRACE_PATH
        original_bearing = mod._BEARING_PATH
        original_stagger = mod._PREFILL_STAGGER_S

        mod._ATTRACTORS_DIR = attractors_dir
        mod._ANAMNESIS_DIR = anamnesis_dir
        mod._TOPOI_DIR = topoi_dir
        mod._POLICY_DIR = policy_dir
        mod._TRACE_PATH = trace_path
        mod._BEARING_PATH = bearing_path
        mod._PREFILL_STAGGER_S = 0.5

        try:
            carver = mod.TurnCarver()

            print(f"Processing {len(turns)} turns through full pipeline...\n")

            run_start = time.time()
            for i, turn in enumerate(turns):
                words = len(turn["user"].split())
                print(f"[{i+1}/{len(turns)}] Entry {turn['index']} ({words}w): {turn['user'][:80]}...")

                t0 = time.time()
                carver.on_turn_complete(turn["user"], turn["assistant"])

                # Wait for background work to complete — _drain_sync runs
                # the background loop synchronously, processing all pending
                # carve and embed work before returning.
                carver._drain_sync()

                # Also wait for any thread that on_turn_complete may have
                # started before _drain_sync could grab the work.
                if carver._thread is not None and carver._thread.is_alive():
                    carver._thread.join(timeout=300)

                elapsed = time.time() - t0
                # Show carve count and timing
                print(f"  carve_count: {carver._carve_count}  ({elapsed:.1f}s)")

            total_elapsed = time.time() - run_start
            print(f"\n{'='*60}")
            print(f"RESULTS  (total: {total_elapsed:.0f}s / {total_elapsed/60:.1f}min)")
            print(f"{'='*60}\n")

            total = 0
            total += _print_surface(attractors_dir, "ATTRACTORS")
            total += _print_surface(anamnesis_dir, "ANAMNESIS")
            total += _print_surface(topoi_dir, "TOPOI")
            total += _print_surface(policy_dir, "POLICY")

            print(f"\n  Total entries: {total}")
            print(f"  Carve events: {carver._carve_count}")

            # Show bearing
            if bearing_path.exists():
                bearing = bearing_path.read_text(encoding="utf-8").strip()
                print(f"\n  BEARING ({len(bearing)} chars):")
                for line in bearing.split("\n"):
                    print(f"    {line}")
            else:
                print("\n  BEARING: (none)")

            # Show trace summary
            if trace_path.exists():
                events = []
                with open(trace_path) as f:
                    for line in f:
                        events.append(json.loads(line))
                beast_events = [e for e in events if e.get("event") == "beast_complete"]
                bearing_events = [e for e in events if e.get("event") == "bearing_update"]
                candidate_events = [e for e in events if e.get("event") == "carve_candidates"]
                empty_events = [e for e in events if e.get("event") == "carve_empty"]

                print(f"\n  TRACE SUMMARY:")
                print(f"    Total events: {len(events)}")
                print(f"    Carve candidate batches: {len(candidate_events)}")
                print(f"    Empty carves: {len(empty_events)}")
                print(f"    Beast passes: {len(beast_events)}")
                print(f"    Bearing updates: {len(bearing_events)}")

                for be in beast_events:
                    kills = be.get("kill_details", [])
                    reroutes = be.get("reroute_details", [])
                    if kills or reroutes:
                        print(f"    Beast: {be.get('survivors', 0)} survived, {be.get('killed', 0)} killed, {be.get('rerouted', 0)} rerouted")
                        for k in kills:
                            print(f"      X {k.get('surface', '?')}:{k.get('slug', '?')} — {k.get('reason', '')[:80]}")
                        for r in reroutes:
                            print(f"      > {r.get('from', '?')}→{r.get('to', '?')} {r.get('slug', '?')} — {r.get('reason', '')[:80]}")

            print(f"\n  Temp dir: {tmpdir}")
            print("  (will be deleted when this script exits)\n")

            # Pause so user can inspect if they want
            input("  Press Enter to clean up and exit...")

        finally:
            mod._ATTRACTORS_DIR = original_attractors
            mod._ANAMNESIS_DIR = original_anamnesis
            mod._TOPOI_DIR = original_topoi
            mod._POLICY_DIR = original_policy
            mod._TRACE_PATH = original_trace
            mod._BEARING_PATH = original_bearing
            mod._PREFILL_STAGGER_S = original_stagger


if __name__ == "__main__":
    main()
