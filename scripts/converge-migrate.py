#!/usr/bin/env python3
"""Converge migration runner: process historical Grapheus transcripts through
the full carving pipeline (four passes + beast + bearing + recompile).

Runs on the cold path — processes accumulated history when the server is idle,
pauses when inference is active, resumes when idle again.

Usage:
    # Process all history from scratch into temp dirs
    uv run scripts/converge-migrate.py --clean

    # Process specific date range
    uv run scripts/converge-migrate.py --from 2026-04-20 --to 2026-04-23

    # Process into real dirs (careful — this writes to ~/.config/spoke/)
    uv run scripts/converge-migrate.py --clean --live

    # Use OpenRouter for fast iteration
    SPOKE_COMMAND_URL=https://openrouter.ai/api SPOKE_COMMAND_API_KEY=sk-... \
        SPOKE_COMMAND_MODEL=qwen/qwen3.6-35b-a3b \
        uv run scripts/converge-migrate.py --clean

    # Resume from where we left off
    uv run scripts/converge-migrate.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_GRAPHEUS_LOG_DIR = Path.home() / "dev" / "grapheus" / "logs"
_SPOKE_CONFIG = Path.home() / ".config" / "spoke"
_HEARTBEAT_INTERVAL_S = 300  # 5 minutes between idle checks
_IDLE_CHECK_TIMEOUT_S = 5  # timeout for the idle check request


def _list_grapheus_logs(from_date: str | None, to_date: str | None) -> list[Path]:
    """List Grapheus log files in date order, optionally filtered by range."""
    logs = sorted(_GRAPHEUS_LOG_DIR.glob("grapheus-*.jsonl"))
    if from_date:
        logs = [l for l in logs if l.stem >= f"grapheus-{from_date}"]
    if to_date:
        logs = [l for l in logs if l.stem <= f"grapheus-{to_date}"]
    return logs


def _load_turns_from_log(log_path: Path) -> list[dict]:
    """Load user/assistant pairs from a single Grapheus log, filtering noise."""
    pairs = []
    seen_user = set()

    with open(log_path) as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            req = entry.get("request", {})
            msgs = req.get("messages", [])

            # Skip carve/beast/bearing requests by system prompt
            sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
            sys_content = " ".join((m.get("content") or "").lower() for m in sys_msgs)
            skip_keywords = [
                "attractor carver", "anamnesis carver", "tópos carver",
                "policy observer", "species classifier", "conversational bearing",
                "substrate carver", "file recompiler",
            ]
            if any(kw in sys_content for kw in skip_keywords):
                continue

            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
            if not user_msgs:
                continue
            last_user = user_msgs[-1].get("content", "")
            if not last_user.strip():
                continue

            # Skip carve-script transcripts in user messages
            if last_user.strip().startswith("Here is a transcript of"):
                continue
            if "Identify attractor operations for this utterance" in last_user:
                continue
            if "User utterance from a voice interaction" in last_user:
                continue

            # Dedup retries
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
                "log": log_path.name,
                "index": i,
                "user": last_user,
                "assistant": assembled,
            })

    return pairs


def _check_server_idle(base_url: str, api_key: str) -> bool:
    """Quick check if the server seems idle. Returns True if idle or unreachable."""
    import urllib.request
    try:
        url = f"{base_url.rstrip('/')}/v1/models"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=_IDLE_CHECK_TIMEOUT_S):
            pass
        return True  # Server responded quickly — likely idle
    except Exception:
        return True  # Can't reach server — treat as idle (will fail on carve if really down)


def _save_progress(progress_path: Path, log_name: str, turn_index: int):
    """Save resume point."""
    progress_path.write_text(json.dumps({
        "log": log_name,
        "turn_index": turn_index,
        "timestamp": datetime.now().isoformat(),
    }))


def _load_progress(progress_path: Path) -> dict | None:
    """Load resume point."""
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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
        lines = content.split("\n")
        title = f.stem
        preview = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.strip() and not line.startswith("-") and not line.startswith("#"):
                preview = line.strip()[:80]
                break
        if preview:
            print(f"    {title}: {preview}")
        else:
            print(f"    {title}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Converge migration runner")
    parser.add_argument("--from", dest="from_date", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--clean", action="store_true",
                        help="Start from clean slate (empty surface dirs)")
    parser.add_argument("--live", action="store_true",
                        help="Write to real ~/.config/spoke/ dirs (default: temp dirs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved progress")
    parser.add_argument("--no-idle-check", action="store_true",
                        help="Skip server idle checks (for OpenRouter or testing)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size: number of exchanges per carve event (overrides cadence + buffer)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (requires --batch). Window of --batch, advance by --stride. "
                             "E.g. --batch 8 --stride 6 means 8-turn windows overlapping by 2.")
    parser.add_argument("--stagger", type=float, default=None,
                        help="Seconds between pass launches (default 0.5). Higher = fewer concurrent "
                             "inference calls. E.g. --stagger 30 limits to ~2 concurrent.")
    args = parser.parse_args()

    if args.stride and not args.batch:
        parser.error("--stride requires --batch")

    # Determine output directories
    if args.live:
        base_dir = _SPOKE_CONFIG
        print("*** LIVE MODE — writing to ~/.config/spoke/ ***\n")
    else:
        tmp = tempfile.mkdtemp(prefix="spoke-migrate-")
        base_dir = Path(tmp)
        print(f"Temp output: {base_dir}\n")

    attractors_dir = base_dir / "attractors"
    anamnesis_dir = base_dir / "anamnesis"
    topoi_dir = base_dir / "topoi"
    policy_dir = base_dir / "policy"
    bearing_path = base_dir / "converge-bearing.md"
    trace_path = base_dir / "converge-trace.jsonl"
    progress_path = base_dir / "converge-migrate-progress.json"

    if args.clean:
        for d in (attractors_dir, anamnesis_dir, topoi_dir, policy_dir):
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        if bearing_path.exists():
            bearing_path.unlink()
        if trace_path.exists():
            trace_path.unlink()
        if progress_path.exists():
            progress_path.unlink()
        print("Clean slate — all surfaces cleared.\n")
    else:
        for d in (attractors_dir, anamnesis_dir, topoi_dir, policy_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Load resume point
    resume_from = None
    if args.resume:
        resume_from = _load_progress(progress_path)
        if resume_from:
            print(f"Resuming from {resume_from['log']} turn {resume_from['turn_index']}\n")

    # Monkey-patch module constants
    import spoke.converge as mod
    originals = {
        "_ATTRACTORS_DIR": mod._ATTRACTORS_DIR,
        "_ANAMNESIS_DIR": mod._ANAMNESIS_DIR,
        "_TOPOI_DIR": mod._TOPOI_DIR,
        "_POLICY_DIR": mod._POLICY_DIR,
        "_TRACE_PATH": mod._TRACE_PATH,
        "_BEARING_PATH": mod._BEARING_PATH,
    }
    mod._ATTRACTORS_DIR = attractors_dir
    mod._ANAMNESIS_DIR = anamnesis_dir
    mod._TOPOI_DIR = topoi_dir
    mod._POLICY_DIR = policy_dir
    mod._TRACE_PATH = trace_path
    mod._BEARING_PATH = bearing_path

    # Batch size override
    if args.batch:
        mod._CARVE_CADENCE = args.batch
        mod._MAX_CONTEXT_BUFFER = args.batch

    # Stagger override — higher values = fewer concurrent inference calls
    if args.stagger is not None:
        mod._PREFILL_STAGGER_S = args.stagger
        print(f"Stagger: {args.stagger}s between pass launches\n")
        print(f"Batch size: {args.batch} (cadence={args.batch}, buffer={args.batch})\n")

    # Graceful shutdown
    shutdown = [False]
    def _handle_signal(sig, frame):
        print("\n\nShutting down gracefully (finishing current carve)...")
        shutdown[0] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    base_url = os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8090")
    api_key = (os.environ.get("SPOKE_COMMAND_API_KEY")
               or os.environ.get("OMLX_SERVER_API_KEY", ""))

    try:
        carver = mod.TurnCarver()
        logs = _list_grapheus_logs(args.from_date, args.to_date)
        print(f"Found {len(logs)} Grapheus logs to process.\n")

        total_turns = 0
        total_carves = 0
        run_start = time.time()
        skipping = resume_from is not None

        for log_path in logs:
            if shutdown[0]:
                break

            turns = _load_turns_from_log(log_path)
            print(f"--- {log_path.name}: {len(turns)} turns ---")

            if args.stride:
                # Sliding window mode: bypass the carver's accumulation
                # machinery entirely. We construct each window's context
                # directly and call _carve_single with it, avoiding the
                # double-counting and state corruption that happens when
                # overlap turns are fed through on_turn_complete twice.
                import spoke.converge as _mod
                window_size = args.batch or 8
                stride = args.stride
                total_turns += len(turns)

                # Build windows
                windows = []
                pos = 0
                while pos < len(turns):
                    window = turns[pos:pos + window_size]
                    windows.append((pos, window))
                    pos += stride

                print(f"  Sliding window: size={window_size}, stride={stride}, "
                      f"windows={len(windows)}, overlap={window_size - stride}")

                for wi, (start_pos, window) in enumerate(windows):
                    if shutdown[0]:
                        break

                    # Build the context buffer directly from the window turns,
                    # without going through on_turn_complete's accumulation.
                    context = []
                    for j, turn in enumerate(window):
                        assistant_ctx = _mod._middle_out_truncate(
                            turn["assistant"] or "",
                            _mod._ASSISTANT_KEEP_HEAD,
                            _mod._ASSISTANT_KEEP_TAIL,
                        )
                        context.append({
                            "user": turn["user"],
                            "assistant": assistant_ctx,
                            "_seq": start_pos + j,  # stable seq based on position in log
                        })

                    # Also feed turns through on_turn_complete for embedding
                    # only — but skip any turns we've already embedded in a
                    # previous window.
                    for turn in window[max(0, stride * wi - (window_size - stride)):] if wi > 0 else window:
                        carver._embed_pending.append(turn["user"])

                    # Call _carve_single directly with our constructed context
                    t0 = time.time()
                    carver._carve_single(
                        window[-1]["user"],
                        context=context,
                        current_seq=start_pos + len(window) - 1,
                    )
                    # Drain any embed work
                    carver._drain_sync()
                    elapsed = time.time() - t0

                    total_carves = carver._carve_count
                    end_pos = min(start_pos + window_size, len(turns))
                    print(f"  window [{start_pos}:{end_pos}] ({len(window)} turns) "
                          f"{elapsed:.0f}s CARVE {total_carves}  "
                          f"{window[-1]['user'][:50]}...")

                    _save_progress(progress_path, log_path.name, end_pos)

            else:
                # Standard mode: feed turns one at a time, let cadence decide
                for i, turn in enumerate(turns):
                    if shutdown[0]:
                        break

                    # Handle resume
                    if skipping:
                        if (log_path.name == resume_from["log"]
                                and i >= resume_from["turn_index"]):
                            skipping = False
                            print(f"  (resumed at turn {i})")
                        else:
                            continue

                    # Server idle check (skip for OpenRouter / --no-idle-check)
                    if not args.no_idle_check and total_turns > 0 and total_turns % 20 == 0:
                        if not _check_server_idle(base_url, api_key):
                            print(f"  Server busy — waiting {_HEARTBEAT_INTERVAL_S}s...")
                            time.sleep(_HEARTBEAT_INTERVAL_S)

                    words = len(turn["user"].split())
                    total_turns += 1
                    prev_carves = carver._carve_count

                    t0 = time.time()
                    carver.on_turn_complete(turn["user"], turn["assistant"])
                    carver._drain_sync()
                    if carver._thread is not None and carver._thread.is_alive():
                        carver._thread.join(timeout=600)
                    elapsed = time.time() - t0

                    carved = carver._carve_count > prev_carves
                    total_carves = carver._carve_count
                    marker = f"CARVE {total_carves}" if carved else ""

                    if carved or words >= 20:
                        print(f"  [{i}] ({words}w) {elapsed:.0f}s {marker}  {turn['user'][:60]}...")

                    # Save progress periodically
                    if total_turns % 10 == 0:
                        _save_progress(progress_path, log_path.name, i)

            # Save progress at end of each log
            _save_progress(progress_path, log_path.name, len(turns))

        total_elapsed = time.time() - run_start

        # Print results
        batch_label = f"batch={args.batch}" if args.batch else "batch=default(2/4)"
        print(f"\n{'='*60}")
        print(f"RESULTS  ({batch_label}, {total_turns} turns, {total_carves} carves, "
              f"{total_elapsed:.0f}s / {total_elapsed/60:.1f}min)")
        print(f"{'='*60}\n")

        total_entries = 0
        total_entries += _print_surface(attractors_dir, "ATTRACTORS")
        total_entries += _print_surface(anamnesis_dir, "ANAMNESIS")
        total_entries += _print_surface(topoi_dir, "TOPOI")
        total_entries += _print_surface(policy_dir, "POLICY")

        print(f"\n  Total entries: {total_entries}")

        if bearing_path.exists():
            bearing = bearing_path.read_text(encoding="utf-8").strip()
            print(f"\n  BEARING ({len(bearing)} chars):")
            for line in bearing.split("\n"):
                print(f"    {line}")

        # Trace summary
        if trace_path.exists():
            events = [json.loads(l) for l in open(trace_path)]
            beast_events = [e for e in events if e.get("event") == "beast_complete"]
            total_killed = sum(e.get("killed", 0) for e in beast_events)
            total_rerouted = sum(e.get("rerouted", 0) for e in beast_events)
            total_survived = sum(e.get("survivors", 0) for e in beast_events)
            print(f"\n  TRACE: {len(events)} events, {len(beast_events)} beast passes")
            print(f"  Beast totals: {total_survived} survived, {total_killed} killed, {total_rerouted} rerouted")

        if not args.live:
            print(f"\n  Output: {base_dir}")
            print("  (temp dir — will persist until manually deleted)\n")

    finally:
        # Restore module constants
        for k, v in originals.items():
            setattr(mod, k, v)


if __name__ == "__main__":
    main()
