#!/usr/bin/env python3
"""Converge personal attractor carver — infer durable user concerns from conversation.

Reads the spoke operator ring buffer or Grapheus logs and asks the model to
identify personal attractors: recurring themes, concerns, and interests that
persist across conversations. These are NOT project attractors (which live in
epistaxis) — they're personal patterns that help guided compaction know what
matters to this user regardless of project context.

Output: markdown files at ~/.config/spoke/attractors/<slug>.md

Usage:
    uv run scripts/converge-carve-attractors.py                 # from ring buffer
    uv run scripts/converge-carve-attractors.py --grapheus      # from today's Grapheus log
    uv run scripts/converge-carve-attractors.py --dry-run       # print prompt only
    uv run scripts/converge-carve-attractors.py --merge         # merge with existing attractors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

_HISTORY_PATH = Path.home() / ".config" / "spoke" / "history.json"
_GRAPHEUS_LOG_DIR = Path.home() / "dev" / "grapheus" / "logs"
_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"

_CARVE_SYSTEM_PROMPT = """\
You are a personal attractor carver for the Converge system. Your job is to
read a transcript of recent voice interactions and identify the user's durable
concerns — recurring themes, interests, and priorities that persist across
conversations and aren't tied to any single project task.

Personal attractors are distinct from project attractors (which track specific
work items). Personal attractors capture things like:
- Recurring aesthetic preferences (how things should look, feel, sound)
- Persistent technical interests (architectures, tools, approaches they favor)
- Communication and workflow patterns (how they prefer to work)
- Quality standards they consistently enforce
- Topics they keep returning to regardless of the current task

Produce a JSON array of personal attractors. Each attractor has:
- "slug": kebab-case identifier (e.g., "prefers-minimal-abstractions")
- "title": short human-readable title
- "evidence": 1-2 sentence description of what you observed
- "strength": "strong" (appeared multiple times, explicitly stated) or
  "tentative" (appeared once, inferred)

Rules:
- Only extract attractors that are PERSONAL and DURABLE — not project-specific,
  not ephemeral, not about a specific bug or feature.
- If the conversation is too short or too task-specific to reveal personal
  patterns, return an empty array. That's fine.
- Be concrete. "Cares about code quality" is too vague. "Prefers explicit
  error paths over catch-all handlers" is specific enough.
- Maximum 5 attractors per carve. Better to have 3 strong ones than 5 weak ones.
- Output ONLY the JSON array. No markdown, no commentary.
"""


def _load_history_utterances() -> list[dict]:
    try:
        data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        return [{"user": u, "assistant": a} for u, a in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_grapheus_utterances(log_date: str | None = None) -> list[dict]:
    d = log_date or date.today().isoformat()
    log_path = _GRAPHEUS_LOG_DIR / f"grapheus-{d}.jsonl"
    if not log_path.exists():
        print(f"No Grapheus log at {log_path}", file=sys.stderr)
        return []

    pairs = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            req = entry.get("request", {})
            messages = req.get("messages", [])
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            if not user_msgs:
                continue
            last_user = user_msgs[-1].get("content", "")
            resp = entry.get("response", {})
            assembled = resp.get("assembled_content", "")
            if not assembled:
                choices = resp.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    assembled = msg.get("content", "")
            pairs.append({"user": last_user, "assistant": assembled})
    return pairs


def _build_transcript(pairs: list[dict]) -> str:
    lines = []
    for i, p in enumerate(pairs, 1):
        lines.append(f"[Turn {i}]")
        lines.append(f"User: {p['user']}")
        if p.get("assistant"):
            a = p["assistant"]
            if len(a) > 500:
                a = a[:500] + "..."
            lines.append(f"Assistant: {a}")
        lines.append("")
    return "\n".join(lines)


def _call_model(system: str, user: str) -> str:
    base_url = os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8090")
    api_key = (
        os.environ.get("SPOKE_COMMAND_API_KEY")
        or os.environ.get("OMLX_SERVER_API_KEY", "")
    )
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": os.environ.get("SPOKE_COMMAND_MODEL", "Qwen3.6-35B-A3B-bf16"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"]


def _load_existing_attractors() -> dict[str, dict]:
    """Load existing personal attractors by slug."""
    existing = {}
    if _ATTRACTORS_DIR.is_dir():
        for f in sorted(_ATTRACTORS_DIR.iterdir()):
            if f.is_file() and f.suffix == ".md":
                existing[f.stem] = {"path": f, "content": f.read_text(encoding="utf-8")}
    return existing


def _write_attractor(slug: str, title: str, evidence: str, strength: str) -> Path:
    """Write a personal attractor file."""
    _ATTRACTORS_DIR.mkdir(parents=True, exist_ok=True)
    path = _ATTRACTORS_DIR / f"{slug}.md"
    today = date.today().isoformat()
    content = f"# {title}\n\n- Evidence: {evidence}\n- Strength: {strength}\n- Observed: {today}\n"
    path.write_text(content, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Converge personal attractor carver")
    parser.add_argument("--grapheus", action="store_true",
                        help="Read from Grapheus log instead of history.json")
    parser.add_argument("--date", type=str, default=None,
                        help="Grapheus log date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt without calling the model")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing attractors (update strength if re-observed)")
    args = parser.parse_args()

    if args.grapheus:
        pairs = _load_grapheus_utterances(args.date)
        source = "Grapheus"
    else:
        pairs = _load_history_utterances()
        source = "history.json"

    if not pairs:
        print(f"No conversation data found from {source}.", file=sys.stderr)
        sys.exit(1)

    # Include existing attractors in context if merging
    existing = _load_existing_attractors()
    existing_context = ""
    if args.merge and existing:
        existing_context = (
            "\n\nExisting personal attractors (update strength to 'strong' if "
            "you see evidence again, or leave them out of your output if they "
            "don't appear in this transcript):\n"
        )
        for slug, info in existing.items():
            existing_context += f"- {slug}: {info['content'][:200]}\n"

    transcript = _build_transcript(pairs)
    user_prompt = (
        f"Here is a transcript of {len(pairs)} recent voice interactions "
        f"between the user and their spoke operator assistant.\n\n"
        f"---\n{transcript}\n---\n\n"
        f"Identify personal attractors from this conversation.{existing_context}"
    )

    if args.dry_run:
        print("=== SYSTEM PROMPT ===")
        print(_CARVE_SYSTEM_PROMPT)
        print()
        print("=== USER PROMPT ===")
        print(user_prompt)
        print()
        print(f"({len(pairs)} turns, {len(transcript)} chars)")
        return

    print(f"Carving personal attractors from {source} ({len(pairs)} turns)...",
          file=sys.stderr)
    result = _call_model(_CARVE_SYSTEM_PROMPT, user_prompt)

    # Parse JSON from response (model may wrap in ```json blocks)
    cleaned = result.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        attractors = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Failed to parse model response as JSON:", file=sys.stderr)
        print(result, file=sys.stderr)
        sys.exit(1)

    if not attractors:
        print("No personal attractors identified in this conversation.", file=sys.stderr)
        return

    written = 0
    for a in attractors:
        slug = a["slug"]
        title = a["title"]
        evidence = a["evidence"]
        strength = a.get("strength", "tentative")

        if slug in existing and not args.merge:
            print(f"  skip (exists): {slug}", file=sys.stderr)
            continue

        if slug in existing and args.merge and strength == "strong":
            # Upgrade existing attractor
            path = _write_attractor(slug, title, evidence, "strong")
            print(f"  upgraded: {path}", file=sys.stderr)
            written += 1
        elif slug not in existing:
            path = _write_attractor(slug, title, evidence, strength)
            print(f"  created: {path}", file=sys.stderr)
            written += 1
        else:
            print(f"  unchanged: {slug}", file=sys.stderr)

    print(f"\n{written} attractor(s) written to {_ATTRACTORS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
