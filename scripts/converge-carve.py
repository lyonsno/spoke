#!/usr/bin/env python3
"""Converge carving prototype — extract substrate diffs from conversation history.

Reads the spoke operator ring buffer (persisted at ~/.config/spoke/history.json)
and sends the user utterances to the local model with a carving prompt. The model
produces substrate diffs: observations, intent signals, attractor references, and
open questions that a future session could use to pick up where the user left off.

Usage:
    uv run scripts/converge-carve.py                    # carve from persisted history
    uv run scripts/converge-carve.py --grapheus         # carve from today's Grapheus log
    uv run scripts/converge-carve.py --grapheus --date 2026-04-20
    uv run scripts/converge-carve.py --dry-run           # print the prompt, don't call the model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

_HISTORY_PATH = Path.home() / ".config" / "spoke" / "history.json"
_GRAPHEUS_LOG_DIR = Path.home() / "dev" / "grapheus" / "logs"

_CARVE_SYSTEM_PROMPT = """\
You are a substrate carver for the Converge system. Your job is to read a
transcript of recent voice interactions between a user and their spoke operator
assistant, and extract durable observations that a future session could use to
understand what the user was doing, thinking about, and trying to accomplish.

Produce a structured extraction with these sections:

## Active intent
What is the user currently trying to do? One or two sentences. Be specific —
name repos, branches, tools, operation codenames if you can identify them.

## Observations
Bullet list of concrete facts surfaced in the conversation. Things learned,
decisions made, results observed. Not summaries of what was said — distillations
of what is now known that wasn't known before.

## Attractor references
Any references to named units of work — operation codenames, attractor titles,
tópos names, sēmeia, lane handles. List them with the context in which they
appeared. If a reference is ambiguous or mangled (e.g., whisper transcription
artifacts), note the likely canonical form.

## Open questions
Things the user expressed uncertainty about, or things that were left unresolved.
These are the pickup points for a future session.

## Substrate diff
If you were writing a one-paragraph update to this user's epistaxis scoped local
state, what would it say? Write it as if you are updating a topos status field.

Rules:
- Extract, don't summarize. The transcript is the record; your job is to pull
  out the load-bearing signal.
- Be concrete. Names, numbers, file paths, branch names, model names.
- If something is ambiguous, say so. Don't resolve ambiguity silently.
- The user speaks via voice dictation. Expect transcription artifacts, informal
  language, and half-finished sentences. Read through them.
"""


def _load_history_utterances() -> list[dict]:
    """Load user/assistant pairs from the persisted ring buffer."""
    try:
        data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        return [{"user": u, "assistant": a} for u, a in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_grapheus_utterances(log_date: str | None = None) -> list[dict]:
    """Load user utterances from a Grapheus daily log."""
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
            user_msgs = [m for m in messages if m.get("role") == "user"]
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
    """Format pairs into a readable transcript."""
    lines = []
    for i, p in enumerate(pairs, 1):
        lines.append(f"[Turn {i}]")
        lines.append(f"User: {p['user']}")
        if p.get("assistant"):
            # Truncate very long assistant responses
            a = p["assistant"]
            if len(a) > 500:
                a = a[:500] + "..."
            lines.append(f"Assistant: {a}")
        lines.append("")
    return "\n".join(lines)


def _call_model(system: str, user: str) -> str:
    """Send to the local OMLX endpoint and return the response."""
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


def main():
    parser = argparse.ArgumentParser(description="Converge carving prototype")
    parser.add_argument("--grapheus", action="store_true",
                        help="Read from Grapheus log instead of history.json")
    parser.add_argument("--date", type=str, default=None,
                        help="Grapheus log date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt without calling the model")
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

    transcript = _build_transcript(pairs)
    user_prompt = (
        f"Here is a transcript of {len(pairs)} recent voice interactions "
        f"between the user and their spoke operator assistant.\n\n"
        f"---\n{transcript}\n---\n\n"
        f"Extract the substrate diff."
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

    print(f"Carving from {source} ({len(pairs)} turns)...", file=sys.stderr)
    result = _call_model(_CARVE_SYSTEM_PROMPT, user_prompt)
    print(result)


if __name__ == "__main__":
    main()
