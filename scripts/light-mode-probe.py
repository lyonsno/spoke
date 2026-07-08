#!/usr/bin/env python3
"""Light-mode probe: single combined pass that routes to attractor, rough,
or nothing. No beast, no separate surfaces. One call per carve.

Usage:
    uv run scripts/light-mode-probe.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

_GRAPHEUS_DIR = Path.home() / "dev" / "grapheus" / "logs"
_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"

# Subset: one clearly ephemeral, one with real signal, one mixed,
# one cultural/relational, one tool-design discussion
SAMPLES_BY_DAY = {
    "2026-04-20": [20, 78, 135],
    "2026-04-23": [19, 178],
}

_LIGHT_MODE_PROMPT = """\
You are a conversational carver. You observe voice interactions and extract
durable signal into two categories:

ATTRACTOR — a force pulling work into existence, with an extinguishable
satisfaction condition. Once satisfied, you stop caring about it. If the
thing were true right now, the pressure would be gone.
Example: "Tool descriptions clearly communicate async wait behavior."

ROUGH — anything worth remembering that is NOT an attractor. Facts about
the environment, relational observations, state of ongoing work, reasoning
and rationales, operational knowledge. These are observations, not forces.
Examples:
- "The OMLX server runs on port 8001" (fact)
- "The user calls the agents 'the boys'" (relational)
- "Working on the context window branch" (work state)
- "Append-only is not stable for durable state" (reasoning)
- "Development always happens in worktrees" (standing rule / policy)

NOTHING — ephemeral commands, test messages, task execution with no
durable signal. "Compact the context." "Merge this into main." "Find
that file." Return [] for these.

Apply the EXTINGUISHMENT TEST for attractors: if satisfying the concern
would make you stop caring about it, it is an attractor. If you would still
need to keep enforcing it forever, it belongs in rough (as policy/standing
rule). If it can be completed by a single action right now, it is nothing.

The user speaks via voice dictation with transcription artifacts. Read
through them to the intent. "Tractor" is "attractor." "operator memory" is
correct Greek.

You will be given EXISTING attractors and rough entries. Do not duplicate
what already exists. Prefer updating existing entries over creating new ones.

Output ONLY a JSON array:
- {"surface": "attractor", "op": "create", "slug": "kebab-case", "title": "Short title", "evidence": "One sentence"}
- {"surface": "attractor", "op": "reinforce", "slug": "<existing>", "evidence": "New evidence"}
- {"surface": "rough", "op": "create", "slug": "kebab-case", "content": "The observation"}
- {"surface": "rough", "op": "update", "slug": "<existing>", "content": "Updated observation"}
- [] when there is nothing worth recording
"""


def _load_samples():
    samples = []
    for day, indices in SAMPLES_BY_DAY.items():
        log = _GRAPHEUS_DIR / f"grapheus-{day}.jsonl"
        if not log.exists():
            continue
        with open(log) as f:
            for i, line in enumerate(f):
                if i not in indices:
                    continue
                entry = json.loads(line)
                req = entry.get("request", {})
                msgs = req.get("messages", [])
                sys_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
                if any("attractor carver" in (m.get("content") or "") for m in sys_msgs):
                    continue
                user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if not user_msgs:
                    continue
                samples.append({
                    "label": f"{day[-5:]}/{i}",
                    "user": user_msgs[-1].get("content", ""),
                })
    return samples


def _load_existing():
    if not _ATTRACTORS_DIR.is_dir():
        return ""
    lines = []
    for f in sorted(_ATTRACTORS_DIR.iterdir()):
        if f.is_file() and f.suffix == ".md":
            text = f.read_text(encoding="utf-8")
            title = f.stem
            for line in text.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            lines.append(f"- {f.stem}: {title}")
    return "Existing attractors:\n" + "\n".join(lines) if lines else ""


def _call(system, user):
    t0 = time.time()
    base_url = os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8090")
    api_key = os.environ.get("SPOKE_COMMAND_API_KEY") or os.environ.get("OMLX_SERVER_API_KEY", "")
    payload = json.dumps({
        "model": os.environ.get("SPOKE_COMMAND_MODEL", "Qwen3.6-35B-A3B-oQ8"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False, "temperature": 0.8, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload, headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"], time.time() - t0


def _parse(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        r = json.loads(cleaned)
        return [r] if isinstance(r, dict) else r
    except json.JSONDecodeError:
        return None


def main():
    samples = _load_samples()
    existing = _load_existing()
    print(f"Samples: {len(samples)}, 1 call each (light mode)\n")

    for sample in samples:
        print(f"{'='*70}")
        print(f"SAMPLE {sample['label']} ({len(sample['user'].split())}w)")
        print(f"  {sample['user'][:150]}...")
        print()

        user_prompt = f"Current user utterance:\n\n\"{sample['user']}\"\n\n"
        if existing:
            user_prompt += f"{existing}\n\n"
        user_prompt += "Extract durable signal from this utterance."

        print(f"  light: ", end="", flush=True)
        try:
            raw, elapsed = _call(_LIGHT_MODE_PROMPT, user_prompt)
            ops = _parse(raw)
            if not ops:
                print(f"[]  ({elapsed:.1f}s)")
                continue

            print(f"({elapsed:.1f}s)")
            for op in ops:
                surface = op.get("surface", "?")
                op_type = op.get("op", "?")
                slug = op.get("slug", "?")
                content = op.get("content", op.get("evidence", ""))
                sym = "A" if surface == "attractor" else "I"
                print(f"    [{sym}] {surface:10s} {op_type:10s} {slug}")
                if content:
                    print(f"        {content[:120]}")
        except Exception as e:
            print(f"ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
