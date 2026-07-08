#!/usr/bin/env python3
"""Four-pass probe: run real Grapheus samples through all four carve surfaces
and print what each surface produces.

Usage:
    uv run scripts/four-pass-probe.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import hashlib
import urllib.request
from datetime import date
from pathlib import Path

_GRAPHEUS_LOG = Path.home() / "dev" / "grapheus" / "logs" / "grapheus-2026-04-23.jsonl"
_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"

# Three diverse samples:
#   19: subagent async reasoning (likely real attractor + maybe policy)
#  143: "look in the attractors directory" (ephemeral — should be [] on most surfaces)
#  178: "compact our context, merge into operator-memory" (ephemeral + maybe topos)
_SAMPLE_INDICES = [19, 143, 178]

# Import the prompts from the branch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spoke.converge import (
    _CARVE_SYSTEM_PROMPT,
    _ANAMNESIS_SYSTEM_PROMPT,
    _TOPOS_SYSTEM_PROMPT,
    _POLICY_SYSTEM_PROMPT,
)

SURFACES = [
    ("attractor", _CARVE_SYSTEM_PROMPT, "Identify attractor operations for this utterance."),
    ("anamnesis", _ANAMNESIS_SYSTEM_PROMPT, "Extract factual observations worth remembering from this utterance."),
    ("topos", _TOPOS_SYSTEM_PROMPT, "Extract changes to the state of ongoing work from this utterance."),
    ("policy", _POLICY_SYSTEM_PROMPT, "Extract reasoning, rationales, or operational principles from this utterance."),
]


def _load_samples() -> list[dict]:
    samples = []
    with open(_GRAPHEUS_LOG) as f:
        for i, line in enumerate(f):
            if i not in _SAMPLE_INDICES:
                continue
            entry = json.loads(line)
            req = entry.get("request", {})
            msgs = req.get("messages", [])
            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
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

            samples.append({"index": i, "user": last_user, "assistant": assembled})
    return samples


def _load_existing_attractors() -> str:
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
                elif "Evidence:" in line:
                    evidence = line.split("Evidence:", 1)[1].strip()
                    lines.append(f"- {f.stem}: {title} — {evidence}")
                    break
            else:
                lines.append(f"- {f.stem}: {title}")
    if not lines:
        return ""
    return "Existing entries:\n" + "\n".join(lines)


def _call_model(system: str, user: str) -> tuple[str, float]:
    import time
    t0 = time.time()
    base_url = os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8090")
    api_key = (
        os.environ.get("SPOKE_COMMAND_API_KEY")
        or os.environ.get("OMLX_SERVER_API_KEY", "")
    )
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": os.environ.get("SPOKE_COMMAND_MODEL", "Qwen3.6-35B-A3B-oQ8"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0.8, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    return body["choices"][0]["message"]["content"], elapsed


def _parse_ops(raw: str) -> list[dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return [{"_parse_error": cleaned[:100]}]
    if isinstance(result, dict):
        return [result]
    return result


def _summarize_ops(ops: list[dict]) -> str:
    if not ops:
        return "[]"
    parts = []
    for op in ops:
        if "_parse_error" in op:
            return f"PARSE_ERROR: {op['_parse_error']}"
        op_type = op.get("op", "?")
        slug = op.get("slug", "?")
        content = op.get("content", op.get("evidence", op.get("new_evidence", "")))
        parts.append(f"{op_type}:{slug}")
    return ", ".join(parts)


def main():
    samples = _load_samples()
    existing = _load_existing_attractors()
    total_calls = len(samples) * len(SURFACES)
    print(f"Samples: {len(samples)}, Surfaces: {len(SURFACES)}, Total calls: {total_calls}")
    print(f"All calls are serial.\n")

    for sample in samples:
        print(f"{'='*70}")
        print(f"SAMPLE {sample['index']} ({len(sample['user'].split())}w)")
        print(f"  {sample['user'][:150]}...")
        print()

        for surface_name, system_prompt, instruction in SURFACES:
            user_prompt = f"Current user utterance:\n\n\"{sample['user']}\"\n\n"
            if existing:
                user_prompt += f"{existing}\n\n"
            user_prompt += instruction

            prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
            print(f"  {surface_name} [{prompt_hash}]: ", end="", flush=True)
            try:
                raw, elapsed = _call_model(system_prompt, user_prompt)
                ops = _parse_ops(raw)
                summary = _summarize_ops(ops)
                print(f"{summary}  ({elapsed:.1f}s)")

                # Print details for non-empty results
                for op in ops:
                    if not op or "_parse_error" in op:
                        continue
                    detail = op.get("content", op.get("evidence", op.get("new_evidence", "")))
                    if detail:
                        print(f"    → {detail[:120]}")
            except Exception as e:
                print(f"ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
