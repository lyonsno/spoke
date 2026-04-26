"""Converge — per-turn attractor carving and embedding via OMLX batch parallel.

After each command response completes, fires an async request to the local
model (same endpoint, same model) asking it to identify personal attractors
from that turn. Also embeds the user utterance via the local embedding model
and appends to a rolling turn-embedding cache so that guided compaction can
do pure-numpy cosine search without loading any model at tool-call time.

OMLX's batch parallel scheduling handles contention with interactive command
requests — carve/embed requests simply wait in the queue when the user is
actively talking.

Outputs:
- Personal attractors: ~/.config/spoke/attractors/
- Turn embedding cache: ~/.config/spoke/turn-embeddings.npz
- Trace log: ~/.config/spoke/converge-trace.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import hashlib
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"
_ANAMNESIS_DIR = Path.home() / ".config" / "spoke" / "anamnesis"
_TOPOI_DIR = Path.home() / ".config" / "spoke" / "topoi"
_POLICY_DIR = Path.home() / ".config" / "spoke" / "policy"
_ATTRACTORS_ARCHIVE_DIR = Path.home() / ".config" / "spoke" / "attractors-archive"
_ATTRACTOR_INDEX_PATH = Path.home() / ".config" / "spoke" / "attractor-index.npz"
_TRACE_PATH = Path.home() / ".config" / "spoke" / "converge-trace.jsonl"
_TURN_EMBEDDINGS_PATH = Path.home() / ".config" / "spoke" / "turn-embeddings.npz"
_MAX_CACHED_EMBEDDINGS = 100  # rolling window of recent turn embeddings
_MAX_CONTEXT_BUFFER = 4  # recent turns kept as conversational context for carving
_CARVE_DEBOUNCE_S = 30.0  # seconds to wait after assistant response before carving
_CARVE_CADENCE = 2  # carve every Nth substantive turn
_ASSISTANT_TRUNCATE_THRESHOLD = 500  # chars; assistant turns longer than this get middle-out truncated
_ASSISTANT_KEEP_HEAD = 250  # chars to keep from the start of long assistant turns
_ASSISTANT_KEEP_TAIL = 250  # chars to keep from the end of long assistant turns
_PREFILL_STAGGER_S = 0.5  # seconds between pass launches to avoid concurrent prefills
_BEARING_PATH = Path.home() / ".config" / "spoke" / "converge-bearing.md"
_BEARING_MAX_TOKENS = 400  # rough target for bearing length

_CARVE_SYSTEM_PROMPT = """\
You are a personal attractor carver. An attractor is a force pulling work
into existence — a durable concern with a satisfaction condition that, once
met, EXTINGUISHES the attractor. The pressure goes away because the thing
got built, fixed, or resolved.

Apply the EXTINGUISHMENT TEST before carving:
- If the thing were true right now, would you STOP CARING about it? If yes,
  it is an attractor. "The carver context window includes recent turns" —
  once landed, done, the pressure is gone.
- If the thing were true right now, would you STILL NEED TO KEEP ENFORCING
  it forever? If yes, it is NOT an attractor — it is policy or a standing
  rule. "Development always happens in worktrees" is never done; every
  future action either complies or doesn't. Do not carve it.
- If the thing can be completed by a single action right now, it is an
  ephemeral command, not an attractor. "Compact the context" is done the
  moment you compact. Do not carve it.

The user speaks via voice dictation with transcription artifacts. Read through
them to the intent. "Tractor" is almost certainly "attractor." "Epístaxis"
is correct Greek, not a typo.

Look across ALL recent turns together, not just the current utterance in
isolation. If the user's concern evolved across multiple turns — started
with one framing and refined it — carve the mature form, not each step.
If the arc across turns reveals something that no single turn states
explicitly, that emergent signal is the most valuable thing to carve.

You will be given EXISTING personal attractors. These are shown so you can
avoid duplicates and reinforce existing entries when appropriate. Do NOT
reference the content of existing entries in new candidates — carve ONLY
from what appears in the current utterance and recent conversation context.
Prefer reinforce/expand over create. Be skeptical — most turns are task
execution and reveal nothing durable.

Your response is a JSON array of operations:

1. REINFORCE an existing attractor (re-observed evidence):
   {"op": "reinforce", "slug": "<existing-slug>", "evidence": "New evidence observed"}

2. EXPAND an existing attractor (broaden its scope with new detail):
   {"op": "expand", "slug": "<existing-slug>", "new_evidence": "Additional detail", "new_title": "Optional broader title"}

3. CORRECT an existing attractor (fix a transcription error or misattribution):
   {"op": "correct", "slug": "<existing-slug>", "corrected_slug": "fixed-slug", "corrected_title": "Fixed title", "reason": "Why"}

4. CREATE a new attractor (genuinely novel signal not covered by any existing one):
   {"op": "create", "slug": "kebab-case-id", "title": "Short title", "evidence": "One sentence"}

Rules:
- Output ONLY the JSON array. No markdown, no commentary.
- Return [] when nothing durable is revealed. Most turns are just task execution.
"""

_RECOMPILE_SYSTEM_PROMPT = """\
You are a file recompiler. You are given an existing file and new evidence
from a recent conversation. Your job is to produce an UPDATED version that
integrates the new evidence into a coherent current-state description.

Rules:
- The output replaces the entire file.
- Do NOT append dated lines. Integrate the new evidence into the description.
- Keep the file roughly the same length or shorter unless the scope genuinely
  expanded.
- Preserve the file's core identity — do not drift the meaning.
- Preserve the file's existing format and metadata fields (Strength, Last
  observed, or whatever fields the file already uses). Do not add fields
  that are not already present.
- Use a # Title as the first line.

Output ONLY the markdown file content. No commentary.
"""

_ANAMNESIS_SYSTEM_PROMPT = """\
You are an anamnesis carver. You observe voice interactions and extract
factual observations worth remembering — things that are true about the
user, the environment, relationships between systems, or operational
knowledge that a future session would benefit from knowing.

Anamnesis is NOT:
- Commands or requests (those are ephemeral)
- Preferences with extinguishable satisfaction conditions (those are attractors)
- State of ongoing work (those are tópoi)
- Reasoning about why something should be a certain way (that is policy)

Anamnesis IS:
- Facts about the environment: ports, paths, model names, server topology
- Facts about the user: how they refer to things, what tools they use
- Relational knowledge: what connects to what, what depends on what
- Operational knowledge: things learned from incidents or debugging

Look across ALL recent turns together. A fact may only become clear from
the combination of multiple turns — the user says one thing, the assistant
responds, and the user's correction or elaboration reveals the real fact.
Carve the synthesized observation, not each fragment.

You will be given EXISTING anamnesis entries. These are shown so you can
avoid duplicates — do NOT reference the content of existing entries in new
candidates. Carve ONLY from what appears in the current utterance and recent
conversation context. If an observation is already captured, return []. If
it updates an existing entry, return an update op.

The user speaks via voice dictation with transcription artifacts. Read through
them to the intent.

Output ONLY a JSON array:
- {"op": "create", "slug": "kebab-case", "content": "The factual observation"}
- {"op": "update", "slug": "<existing-slug>", "content": "Updated observation"}
- [] when there is nothing new worth remembering
"""

_TOPOS_SYSTEM_PROMPT = """\
You are a tópos carver. You observe voice interactions and extract changes
to the state of work that the OPERATOR ASSISTANT is actively doing.

Only carve tópoi for work the operator is the AGENT of — searches it
launched, tasks it is mid-way through, things it is actively waiting on.
If the user mentions work that is happening elsewhere (other sessions,
other tools, other lanes), that is something the operator WITNESSED, not
something it is doing. Witnessed work is anamnesis (a fact learned), not
a tópos.

A tópos captures the current state of a unit of work the operator owns —
not the history of how it got there, just where it is now. Tópoi decay
naturally as work completes or goes stale.

Tópoi are NOT:
- Work the user mentions doing in other tools or sessions (those are anamnesis)
- Durable preferences or forces (those are attractors)
- Facts about the environment (those are anamnesis)
- Reasoning about why (that is policy)
- Ephemeral commands (those are nothing)

Tópoi ARE:
- "Dispatched subagent to search for the cancel-generation attractor"
- "Waiting for subagent result on attractor search"
- "Creating a spoke worktree off remote main"

Look across ALL recent turns together. Work state often evolves across
turns — the user starts something, hits a problem, pivots. Carve the
current state after the arc, not each intermediate step.

You will be given EXISTING tópoi. These are shown so you can avoid duplicates
and update existing entries — do NOT reference the content of existing entries
in new candidates. Carve ONLY from what appears in the current utterance and
recent conversation context. If the state of an existing tópos changed,
return an update op. If a new unit of work appeared, create it. If nothing
about the state of work changed, return [].

The user speaks via voice dictation with transcription artifacts. Read through
them to the intent.

Output ONLY a JSON array:
- {"op": "create", "slug": "kebab-case", "content": "Current state of this work"}
- {"op": "update", "slug": "<existing-slug>", "content": "Updated current state"}
- [] when the state of work did not change
"""

_POLICY_SYSTEM_PROMPT = """\
You are a policy observer. You observe voice interactions and extract
reasoning, rationales, and operational principles the user articulated.

Policy is distinct from attractors. An attractor has a satisfaction condition
that EXTINGUISHES it — once the thing is built or fixed, the pressure goes
away. Policy has COMPLIANCE, not satisfaction. You can comply with policy or
violate it, but you can never finish it. Satisfying an attractor negates the
need for the attractor. Satisfying policy just means you did not violate it
this time.

You are OBSERVING policy, not ENFORCING it. Document policy-shaped reasoning
so it can be reviewed later — do not make it active.

Policy is NOT:
- Commands (those are ephemeral)
- Concerns with extinguishable endpoints (those are attractors)
- Facts (those are anamnesis)
- State of work (those are tópoi)

Policy IS:
- "Append-only is not stable for durable state"
- "Development should always happen in worktrees" (never done, always enforced)
- "Four parallel passes are better than one multi-routing pass"
- "The satisfaction condition test should distinguish action-shaped from state-shaped"

Policy often emerges across multiple turns of reasoning — the user
proposes something, tests it, refines it, and the mature principle is only
visible from the full arc. Look across ALL recent turns for reasoning that
converged on a principle, not just single-turn declarations. The most
valuable policy carves are ones where the user arrived at a standing rule
through a multi-turn discussion that no single turn captures alone.

You will be given EXISTING policy observations. These are shown so you can
avoid duplicates — do NOT reference the content of existing entries in new
candidates. Carve ONLY from what appears in the current utterance and recent
conversation context. If a principle is already captured, return []. If a
new observation refines or supersedes an existing one, return an update op.

The user speaks via voice dictation with transcription artifacts. Read through
them to the intent.

Output ONLY a JSON array:
- {"op": "create", "slug": "kebab-case", "content": "The observed principle or rationale"}
- {"op": "update", "slug": "<existing-slug>", "content": "Refined principle"}
- [] when no policy-shaped reasoning was articulated
"""

_BEAST_SPECIES_PROMPT = """\
You are a species classifier for a multi-surface carving system. You are
given a user utterance and a set of candidate carves that four independent
passes produced. Each candidate is tagged with its claimed surface:
attractor, anamnesis, topos, or policy.

Your job is to review each candidate and decide: is it correctly routed to
the claimed surface, or is it the wrong species?

The surfaces are:
- ATTRACTOR: a force pulling work into existence, with an EXTINGUISHABLE
  satisfaction condition. Once satisfied, you stop caring about it.
  "Tool descriptions clearly communicate async behavior" — once done, done.
- ANAMNESIS: a factual observation worth remembering. No satisfaction
  condition. "The server runs on port 8001." "The user calls agents 'the boys'."
- TOPOS: current state of a unit of work. Decays naturally. "Working on
  the context window branch." "Merging into main."
- POLICY: a standing rule or principle with COMPLIANCE, not satisfaction.
  You can comply or violate but never finish it. "Development always
  happens in worktrees." "Append-only is not stable for durable state."

For each candidate, output one of:
- "pass" — correctly routed, write it
- "kill" — wrong species, do not write it
- "reroute:<surface>" — belongs on a different surface (e.g., "reroute:anamnesis")

Output ONLY a JSON array with one entry per candidate, in the same order:
[
  {"index": 0, "verdict": "pass"},
  {"index": 1, "verdict": "kill", "reason": "This is an ephemeral command, not an attractor"},
  {"index": 2, "verdict": "reroute:anamnesis", "reason": "This is a fact, not policy"}
]
"""

_BEARING_SYSTEM_PROMPT = """\
You maintain a conversational bearing — 2-3 sentences capturing the
direction of an ongoing voice conversation. Not a summary. Not details.
Just the heading: where is this conversation pointed?

Rules:
- Output ONLY the bearing. No preamble.
- Maximum 3 sentences. Be abstract. No proper nouns, no specifics, no
  tool names, no project names, no technical details. Just the direction.
- Example good bearing: "Exploring how to make background processes
  smarter about what they capture. Shifting from mechanism to quality."
- Example bad bearing: "Working on the Converge carver with Panopticon
  review on the cc/converge-anamnesis branch using Qwen 3.6."
- If the direction changed, replace. Do not average old and new.
- If nothing changed, return the bearing unchanged.
"""


def _import_numpy(feature: str):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"numpy is required for {feature}. Install the Converge runtime "
            "dependencies before using guided compaction or turn embeddings."
        ) from exc
    return np


def _url_has_version_prefix(raw_url: str) -> bool:
    path = urlparse(raw_url).path.rstrip("/")
    return any(
        seg.startswith("v") and seg[1:].replace("beta", "").isdigit()
        for seg in path.split("/")
        if seg
    )


def _openai_endpoint(base_url: str, suffix: str) -> str:
    normalized = base_url.rstrip("/")
    if _url_has_version_prefix(normalized):
        return f"{normalized}/{suffix}"
    return f"{normalized}/v1/{suffix}"


def _middle_out_truncate(text: str, head: int, tail: int) -> str:
    """Truncate by cutting the middle, preserving head and tail.

    Long assistant turns are typically agent loops where the intent is at the
    start and the conclusion at the end.  The middle is tool calls and
    intermediate reasoning — least useful for attractor carving context.
    """
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n[...]\n" + text[-tail:]


def _append_trace(path: Path, event: str, **kwargs) -> None:
    try:
        entry = {"timestamp": datetime.now().isoformat(), "event": event, **kwargs}
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _build_turn_preview(history: list, target: int) -> list[str]:
    turn_preview = []
    for i in range(target):
        turn = history[i]
        if isinstance(turn, (list, tuple)) and len(turn) >= 2 and isinstance(turn[0], str):
            turn_preview.append(f"Turn {i+1}: user: {turn[0][:200]}")
            continue
        if not isinstance(turn, list):
            continue
        parts = []
        for message in turn:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content", "")
            if role in ("user", "assistant") and content:
                parts.append(f"{role}: {content[:200]}")
        if parts:
            turn_preview.append(f"Turn {i+1}: " + " | ".join(parts))
    return turn_preview


def _guided_compaction(
    history: list,
    target: int,
    arguments: dict[str, Any],
    *,
    index_path: Path,
    trace_path: Path,
    turn_embeddings_loader: Callable[[], tuple[Any, list[str]] | None],
) -> dict[str, Any]:
    turn_preview = _build_turn_preview(history, target)
    if not index_path.is_file():
        return {
            "status": "error",
            "error": "attractor-index.npz not found. Run: uv run scripts/converge-embed.py build",
            "turn_preview": turn_preview[:5],
        }

    turn_cache = turn_embeddings_loader()
    if turn_cache is None or turn_cache[0].shape[0] == 0:
        return {
            "status": "error",
            "error": (
                "No turn embeddings cached yet. The background carver embeds turns "
                "after each response — try again after a few more exchanges."
            ),
            "turn_preview": turn_preview[:5],
        }

    np = _import_numpy("guided compaction")
    turn_embeddings, turn_texts = turn_cache
    t0 = time.time()

    try:
        data = np.load(index_path, allow_pickle=False)
        full_emb = data["full_embeddings"]
        metadata = json.loads(str(data["metadata"]))
    except Exception as exc:
        return {"status": "error", "error": f"index load failed: {exc}"}

    full_scores = full_emb @ turn_embeddings.T
    combined = full_scores.max(axis=1)

    top_k = arguments.get("top_k", 10)
    threshold = arguments.get("threshold", 0.35)
    top_indices = np.argsort(combined)[::-1][:top_k]
    matched_attractors = []
    for idx in top_indices:
        score = float(combined[idx])
        if score < threshold:
            break
        matched_attractors.append(
            {
                "source": metadata[idx]["source"],
                "attractor": metadata[idx]["slug"],
                "summary": metadata[idx].get("summary", "")[:100],
                "score": round(score, 4),
            }
        )

    elapsed = time.time() - t0
    _append_trace(
        trace_path,
        "guided_compaction",
        elapsed_s=round(elapsed, 2),
        turns_embedded=len(turn_texts),
        attractors_searched=len(metadata),
        matches_returned=len(matched_attractors),
        top_scores=[a["score"] for a in matched_attractors[:5]],
        top_slugs=[a["attractor"][:50] for a in matched_attractors[:5]],
        threshold=threshold,
    )

    return {
        "status": "ok",
        "mode": "guided",
        "turns_targeted": target,
        "turns_total": len(history),
        "attractor_count": len(metadata),
        "retention_flags": matched_attractors,
        "instruction": (
            "These attractors are semantically related to the conversation being "
            "compacted (ranked by cosine similarity). When you call compact_history "
            "with mode='summarize', preserve any information that connects to these "
            "attractors. Use your conversational judgment for everything else."
        ),
        "turn_preview": turn_preview[:5],
    }


def compact_history(
    client,
    arguments: dict[str, Any],
    *,
    index_path: Path | None = None,
    trace_path: Path | None = None,
    turn_embeddings_loader: Callable[[], tuple[Any, list[str]] | None] | None = None,
) -> dict[str, Any]:
    """Execute the compact_history tool on a command client."""
    history = client._history
    if not history:
        return {"status": "nothing to compact", "turns": 0}

    mode = arguments.get("mode", "drop_tool_results")
    n = arguments.get("n", 0)
    target = len(history) if n == 0 else min(n, len(history))
    trace_path = trace_path or _TRACE_PATH

    if mode == "drop_tool_results":
        compacted = 0
        for i in range(target):
            turn = history[i]
            before = len(turn)
            cleaned = []
            for message in turn:
                if message.get("role") not in ("user", "assistant", "system"):
                    continue
                if message.get("role") == "assistant" and "tool_calls" in message:
                    message = {k: v for k, v in message.items() if k != "tool_calls"}
                cleaned.append(message)
            history[i] = cleaned
            if len(history[i]) < before:
                compacted += 1
        client._save_history()
        _append_trace(
            trace_path,
            "compact_drop_tool_results",
            turns_compacted=compacted,
            turns_total=len(history),
        )
        return {
            "status": "ok",
            "mode": "drop_tool_results",
            "turns_compacted": compacted,
            "turns_total": len(history),
        }

    if mode == "summarize":
        summary = arguments.get("summary", "")
        if not summary:
            return {"error": "summary is required for summarize mode"}
        remaining = history[target:]
        summary_turn = [
            {"role": "user", "content": "[compacted history]"},
            {"role": "assistant", "content": summary},
        ]
        client._history = [summary_turn] + remaining
        client._save_history()
        _append_trace(
            trace_path,
            "compact_summarize",
            turns_replaced=target,
            turns_remaining=len(remaining),
            summary_length=len(summary),
            summary_preview=summary[:200],
        )
        return {
            "status": "ok",
            "mode": "summarize",
            "turns_replaced": target,
            "turns_remaining": len(remaining),
        }

    if mode == "reset_to_summary":
        # Find the most recent summary turn — identified by the
        # "[compacted history]" user message that summarize mode writes.
        summary_idx = None
        for i in range(len(history) - 1, -1, -1):
            turn = history[i]
            if (
                turn
                and isinstance(turn[0], dict)
                and turn[0].get("role") == "user"
                and turn[0].get("content") == "[compacted history]"
            ):
                summary_idx = i
                break
        if summary_idx is None:
            return {"status": "error", "error": "no compaction summary found in history"}
        dropped = len(history) - summary_idx - 1
        client._history = [history[summary_idx]]
        client._save_history()
        _append_trace(
            trace_path,
            "compact_reset_to_summary",
            summary_turn_index=summary_idx,
            turns_dropped=dropped,
        )
        return {
            "status": "ok",
            "mode": "reset_to_summary",
            "turns_dropped": dropped,
            "turns_remaining": 1,
        }

    if mode == "guided":
        try:
            return _guided_compaction(
                history,
                target,
                arguments,
                index_path=index_path or _ATTRACTOR_INDEX_PATH,
                trace_path=trace_path,
                turn_embeddings_loader=turn_embeddings_loader or load_turn_embeddings,
            )
        except RuntimeError as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": f"unknown compact_history mode: {mode}"}


class TurnCarver:
    """Fires attractor carving after each command turn completes."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._base_url = (
            base_url
            or os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8090")
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.environ.get("SPOKE_COMMAND_API_KEY")
            or os.environ.get("OMLX_SERVER_API_KEY", "")
        )
        self._model = (
            model
            or os.environ.get("SPOKE_COMMAND_MODEL", "Qwen3.6-35B-A3B-oQ8")
        )
        self._pending: list[tuple[str, list[dict[str, str]], int]] = []  # (utterance, context_snapshot, current_seq)
        self._embed_pending: list[str] = []  # user utterances not yet embedded
        self._lock = threading.Lock()
        self._embed_io_lock = threading.Lock()  # serialize embed cache read-modify-write
        self._attractor_io_lock = threading.Lock()  # serialize attractor file mutations
        self._thread: threading.Thread | None = None
        self._embed_model_loaded = False
        self._context_buffer: list[dict[str, str]] = []  # rolling window of recent turns
        self._substantive_turn_count = 0  # counts substantive turns for cadence
        self._turn_seq = 0  # monotonic sequence number for context entries
        self._last_carve_seqs: set[int] = set()  # sequence numbers seen by last carve
        self._carve_count = 0  # number of carve events fired (for testing)
        self._anamnesis_io_lock = threading.Lock()  # serialize anamnesis file mutations
        self._topoi_io_lock = threading.Lock()  # serialize topoi file mutations
        self._policy_io_lock = threading.Lock()  # serialize policy file mutations
        self._bearing_io_lock = threading.Lock()  # serialize bearing read-update-write
        _ATTRACTORS_DIR.mkdir(parents=True, exist_ok=True)
        _ANAMNESIS_DIR.mkdir(parents=True, exist_ok=True)
        _TOPOI_DIR.mkdir(parents=True, exist_ok=True)
        _POLICY_DIR.mkdir(parents=True, exist_ok=True)

    def on_turn_complete(self, user_utterance: str, assistant_response: str) -> None:
        """Called after each command turn. Fires async carve + embed."""
        if not user_utterance or not user_utterance.strip():
            return

        with self._lock:
            # Always accumulate context — user turns are never truncated;
            # assistant turns get middle-out truncation for long agent loops
            assistant_text = assistant_response or ""
            assistant_ctx = _middle_out_truncate(
                assistant_text,
                _ASSISTANT_KEEP_HEAD,
                _ASSISTANT_KEEP_TAIL,
            )
            self._turn_seq += 1
            entry = {"user": user_utterance, "assistant": assistant_ctx, "_seq": self._turn_seq}
            self._context_buffer.append(entry)
            if len(self._context_buffer) > _MAX_CONTEXT_BUFFER:
                self._context_buffer = self._context_buffer[-_MAX_CONTEXT_BUFFER:]

            # Always embed (even short turns have semantic content)
            self._embed_pending.append(user_utterance)

            # Only carve substantive turns (>= 10 words)
            if len(user_utterance.split()) >= 10:
                self._substantive_turn_count += 1
                should_carve = False

                if self._substantive_turn_count % _CARVE_CADENCE == 0:
                    should_carve = True
                elif self._last_carve_seqs:
                    # Debounce override: if continued skipping would mean the
                    # next carve sees zero overlap with the last carve's
                    # context, carve now instead of losing coverage.
                    current_seqs = {e["_seq"] for e in self._context_buffer}
                    if not (current_seqs & self._last_carve_seqs):
                        should_carve = True

                if should_carve:
                    context_snapshot = list(self._context_buffer)
                    self._last_carve_seqs = {e["_seq"] for e in self._context_buffer}
                    self._pending.append((user_utterance, context_snapshot, self._turn_seq))

            # Fire background worker if not already running (under lock to
            # prevent concurrent callers from both starting a thread)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._background_loop, daemon=True)
                self._thread.start()

    def _drain_sync(self) -> None:
        """Drain all pending work synchronously. For testing only."""
        self._background_loop()

    def _background_loop(self) -> None:
        """Background thread: dispatch carve and embed work concurrently.

        OMLX batch parallel gives near-linear throughput scaling, so firing
        multiple requests simultaneously is better than serializing them.
        Each pending item gets its own thread; they all batch together on
        the server side.
        """
        while True:
            carve_work: list[tuple[str, list[dict[str, str]], int]] = []
            embed_work: list[str] = []

            with self._lock:
                while self._pending:
                    carve_work.append(self._pending.pop(0))
                while self._embed_pending:
                    embed_work.append(self._embed_pending.pop(0))

            if not carve_work and not embed_work:
                return

            # Fire all work items concurrently
            threads: list[threading.Thread] = []
            for utterance, context, seq in carve_work:
                t = threading.Thread(
                    target=self._safe_call,
                    args=(self._carve_single, utterance, context, seq),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for utterance in embed_work:
                t = threading.Thread(
                    target=self._safe_call,
                    args=(self._embed_single, utterance),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            # Wait for all to complete before checking for more work
            for t in threads:
                t.join(timeout=600)

    @staticmethod
    def _safe_call(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            logger.debug("Converge %s failed", fn.__name__, exc_info=True)

    def _load_existing_entries(self, surface_dir: Path, label: str) -> str:
        """Build a compact summary of existing entries for a carve surface."""
        if not surface_dir.is_dir():
            return ""
        lines = []
        for f in sorted(surface_dir.iterdir()):
            if f.is_file() and f.suffix == ".md":
                try:
                    text = f.read_text(encoding="utf-8")
                    title = f.stem
                    # Try to extract a title line and first content line
                    first_content = ""
                    for line in text.split("\n"):
                        if line.startswith("# "):
                            title = line[2:].strip()
                        elif "Evidence:" in line:
                            first_content = line.split("Evidence:", 1)[1].strip()
                            break
                        elif line.strip() and not line.startswith("-") and not line.startswith("#"):
                            first_content = line.strip()[:120]
                            break
                    if first_content:
                        lines.append(f"- {f.stem}: {title} — {first_content}")
                    else:
                        lines.append(f"- {f.stem}: {title}")
                except OSError:
                    continue
        if not lines:
            return ""
        return f"Existing {label}:\n" + "\n".join(lines)

    def _load_existing_attractors_context(self) -> str:
        """Build a compact summary of existing personal attractors for the prompt."""
        return self._load_existing_entries(_ATTRACTORS_DIR, "personal attractors")

    def _call_model_for_carve(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, float]:
        """Send a carve request to the model and return (response_text, elapsed)."""
        t0 = time.time()
        url = _openai_endpoint(self._base_url, "chat/completions")
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "max_tokens": 4096,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        result_text = body["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        return result_text, elapsed

    def _parse_ops(self, raw: str) -> list[dict] | None:
        """Parse model response into ops list. Returns None on parse failure."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            ops = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if isinstance(ops, dict):
            ops = [ops]
        return ops

    def _build_context_block(
        self, context: list[dict[str, str]] | None, current_seq: int | None
    ) -> str:
        """Build the recent conversation context block for any carve prompt.

        Includes the conversational bearing (if one exists) followed by the
        recent turns. The bearing anchors the origin of the conversational
        vector so the turns read as a continuation of a trajectory.
        """
        parts = []

        # Load bearing if available
        if _BEARING_PATH.exists():
            try:
                bearing = _BEARING_PATH.read_text(encoding="utf-8").strip()
                if bearing:
                    parts.append(
                        f"Conversational bearing (for trajectory context only — do NOT "
                        f"reference bearing content in your output, carve only from the "
                        f"utterance and recent turns below):\n{bearing}\n"
                    )
            except OSError:
                pass

        if not context:
            return "\n".join(parts) + "\n" if parts else ""

        prior = [
            c for c in context
            if c.get("_seq") != current_seq
        ] if current_seq is not None else context
        if prior:
            lines = []
            for c in prior:
                lines.append(f"User: {c['user']}")
                if c.get("assistant"):
                    lines.append(f"Assistant: {c['assistant']}")
            parts.append(
                "Recent conversation context (preceding turns):\n"
                + "\n".join(lines)
            )

        return "\n\n".join(parts) + "\n\n" if parts else ""

    def _recompile_entry(self, existing_path: Path, new_evidence: str) -> str | None:
        """Recompile an existing entry file with new evidence via the model.

        Returns the recompiled content, or None on failure.
        """
        if not existing_path.exists():
            return None
        existing_content = existing_path.read_text(encoding="utf-8")
        user_prompt = (
            f"Here is the existing file:\n\n"
            f"---\n{existing_content}\n---\n\n"
            f"New evidence from a recent conversation:\n\n"
            f"\"{new_evidence}\"\n\n"
            f"Recompile this file, integrating the new evidence."
        )
        try:
            result, _ = self._call_model_for_carve(_RECOMPILE_SYSTEM_PROMPT, user_prompt)
            # Strip markdown fences if present
            content = result.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:markdown|md)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            return content
        except Exception:
            logger.debug("Converge: recompile failed for %s", existing_path.name, exc_info=True)
            return None

    def _update_bearing(self, context: list[dict[str, str]] | None) -> None:
        """Recompile the conversational bearing with the turns the carver just saw."""
        if not context:
            return

        with self._bearing_io_lock:
            # Load current bearing
            current_bearing = ""
            if _BEARING_PATH.exists():
                try:
                    current_bearing = _BEARING_PATH.read_text(encoding="utf-8").strip()
                except OSError:
                    pass

            # Build the recent turns block
            turn_lines = []
            for c in context:
                turn_lines.append(f"User: {c['user']}")
                if c.get("assistant"):
                    turn_lines.append(f"Assistant: {c['assistant']}")

            user_prompt = ""
            if current_bearing:
                user_prompt += f"Current bearing:\n\n{current_bearing}\n\n"
            else:
                user_prompt += "Current bearing: (none — this is the start of the conversation)\n\n"
            user_prompt += (
                "New exchange(s):\n\n"
                + "\n".join(turn_lines)
                + "\n\nUpdate the bearing."
            )

            try:
                result, elapsed = self._call_model_for_carve(_BEARING_SYSTEM_PROMPT, user_prompt)
                content = result.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:markdown|md)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                # Atomic write
                tmp = _BEARING_PATH.with_suffix(".tmp")
                tmp.write_text(content, encoding="utf-8")
                tmp.replace(_BEARING_PATH)
                self._trace("bearing_update", elapsed=round(elapsed, 2), length=len(content))
                logger.debug("Converge: bearing updated (%.1fs, %d chars)", elapsed, len(content))
            except Exception:
                logger.debug("Converge: bearing update failed", exc_info=True)

    def _carve_single(
        self,
        utterance: str,
        context: list[dict[str, str]] | None = None,
        current_seq: int | None = None,
    ) -> None:
        """Run all four carve passes for one utterance, then beast-filter survivors."""
        self._carve_count += 1
        context_block = self._build_context_block(context, current_seq)

        # Collect candidates from all four passes. Each pass appends to this
        # shared list under its own lock section.
        candidates: list[dict] = []  # {"surface": str, "op": dict, "prompt_hash": str}
        candidates_lock = threading.Lock()

        # Stagger pass launches so prefills don't all compete at once.
        # Each pass starts after a brief delay, letting the prior one move
        # from prefill into decode before the next prefill begins.
        pass_fns = [
            self._collect_attractors,
            self._collect_anamnesis,
            self._collect_topoi,
            self._collect_policy,
        ]
        pass_threads: list[threading.Thread] = []
        for i, fn in enumerate(pass_fns):
            t = threading.Thread(
                target=self._safe_call,
                args=(fn, utterance, context_block, candidates, candidates_lock),
                daemon=True,
            )
            if i > 0:
                time.sleep(_PREFILL_STAGGER_S)
            t.start()
            pass_threads.append(t)
        for t in pass_threads:
            t.join(timeout=120)

        if not candidates:
            # Even with no candidates, update the bearing — the conversation
            # moved forward even if nothing was worth carving.
            self._update_bearing(context)
            return

        # Beast pass: species-classify all candidates
        survivors = self._beast_filter(utterance, candidates)

        # Write survivors to their respective surfaces
        self._write_survivors(survivors)

        # Update the bearing with the turns we just processed
        self._update_bearing(context)

    def _collect_pass(
        self,
        surface_name: str,
        system_prompt: str,
        surface_dir: Path,
        utterance: str,
        context_block: str,
        instruction: str,
        candidates: list[dict],
        candidates_lock: threading.Lock,
    ) -> None:
        """Run a carve pass and collect candidates without writing to disk."""
        existing = self._load_existing_entries(surface_dir, surface_name)
        user_prompt = ""
        if context_block:
            user_prompt += context_block
        user_prompt += f"Current user utterance:\n\n\"{utterance}\"\n\n"
        if existing:
            user_prompt += f"{existing}\n\n"
        user_prompt += instruction

        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        try:
            result_text, elapsed = self._call_model_for_carve(system_prompt, user_prompt)
        except Exception:
            logger.debug("Converge: %s pass failed", surface_name, exc_info=True)
            return

        ops = self._parse_ops(result_text)
        if ops is None:
            self._trace("carve_parse_error", surface=surface_name, elapsed=0, raw=result_text[:200], prompt_hash=prompt_hash)
            return
        if not ops:
            self._trace("carve_empty", surface=surface_name, elapsed=elapsed, utterance=utterance[:100], prompt_hash=prompt_hash)
            return

        with candidates_lock:
            for op in ops:
                slug = op.get("slug", "")
                if not slug:
                    continue
                candidates.append({
                    "surface": surface_name,
                    "op": op,
                    "prompt_hash": prompt_hash,
                    "elapsed": elapsed,
                })

        self._trace(
            "carve_candidates",
            surface=surface_name,
            elapsed=elapsed,
            utterance=utterance[:100],
            ops_count=len(ops),
            prompt_hash=prompt_hash,
        )

    def _collect_attractors(
        self, utterance: str, context_block: str,
        candidates: list[dict], candidates_lock: threading.Lock,
    ) -> None:
        self._collect_pass(
            "attractor", _CARVE_SYSTEM_PROMPT, _ATTRACTORS_DIR,
            utterance, context_block,
            "Identify attractor operations for this utterance.",
            candidates, candidates_lock,
        )

    def _collect_anamnesis(
        self, utterance: str, context_block: str,
        candidates: list[dict], candidates_lock: threading.Lock,
    ) -> None:
        self._collect_pass(
            "anamnesis", _ANAMNESIS_SYSTEM_PROMPT, _ANAMNESIS_DIR,
            utterance, context_block,
            "Extract factual observations worth remembering from this utterance.",
            candidates, candidates_lock,
        )

    def _collect_topoi(
        self, utterance: str, context_block: str,
        candidates: list[dict], candidates_lock: threading.Lock,
    ) -> None:
        self._collect_pass(
            "topos", _TOPOS_SYSTEM_PROMPT, _TOPOI_DIR,
            utterance, context_block,
            "Extract changes to the state of ongoing work from this utterance.",
            candidates, candidates_lock,
        )

    def _collect_policy(
        self, utterance: str, context_block: str,
        candidates: list[dict], candidates_lock: threading.Lock,
    ) -> None:
        self._collect_pass(
            "policy", _POLICY_SYSTEM_PROMPT, _POLICY_DIR,
            utterance, context_block,
            "Extract reasoning, rationales, or operational principles from this utterance.",
            candidates, candidates_lock,
        )

    def _beast_filter(self, utterance: str, candidates: list[dict]) -> list[dict]:
        """Beast pass: species-classify candidates and return survivors."""
        # Build the candidate list for the beast prompt
        candidate_lines = []
        for i, c in enumerate(candidates):
            op = c["op"]
            surface = c["surface"]
            op_type = op.get("op", "create")
            slug = op.get("slug", "?")
            content = op.get("content", op.get("evidence", op.get("new_evidence", "")))
            candidate_lines.append(
                f"[{i}] surface={surface} op={op_type} slug={slug}"
                + (f" content=\"{content[:150]}\"" if content else "")
            )

        user_prompt = (
            f"User utterance:\n\"{utterance}\"\n\n"
            f"Candidates ({len(candidates)}):\n"
            + "\n".join(candidate_lines)
            + "\n\nClassify each candidate."
        )

        beast_hash = hashlib.sha256(_BEAST_SPECIES_PROMPT.encode()).hexdigest()[:16]
        try:
            result_text, elapsed = self._call_model_for_carve(_BEAST_SPECIES_PROMPT, user_prompt)
        except Exception:
            logger.debug("Converge: beast pass failed, passing all candidates", exc_info=True)
            # Fail-open: if the beast can't run, pass everything through
            return candidates

        verdicts = self._parse_ops(result_text)
        if verdicts is None:
            logger.debug("Converge: beast parse failed, passing all candidates")
            self._trace("beast_parse_error", raw=result_text[:200], beast_hash=beast_hash)
            return candidates

        # Apply verdicts
        survivors = []
        killed = []
        rerouted = []
        for v in verdicts:
            idx = v.get("index", -1)
            if idx < 0 or idx >= len(candidates):
                continue
            verdict = v.get("verdict", "pass")
            reason = v.get("reason", "")

            if verdict == "pass":
                survivors.append(candidates[idx])
            elif verdict == "kill":
                killed.append({"candidate": candidates[idx], "reason": reason})
            elif verdict.startswith("reroute:"):
                new_surface = verdict.split(":", 1)[1]
                rerouted_candidate = dict(candidates[idx])
                rerouted_candidate["surface"] = new_surface
                survivors.append(rerouted_candidate)
                rerouted.append({
                    "from": candidates[idx]["surface"],
                    "to": new_surface,
                    "slug": candidates[idx]["op"].get("slug", "?"),
                    "reason": reason,
                })

        self._trace(
            "beast_complete",
            elapsed=elapsed,
            candidates=len(candidates),
            survivors=len(survivors),
            killed=len(killed),
            rerouted=len(rerouted),
            kill_details=[{
                "surface": k["candidate"]["surface"],
                "slug": k["candidate"]["op"].get("slug", "?"),
                "reason": k["reason"],
            } for k in killed],
            reroute_details=rerouted,
            beast_hash=beast_hash,
        )

        return survivors

    def _write_survivors(self, survivors: list[dict]) -> None:
        """Write beast-approved candidates to their respective surfaces."""
        today = date.today().isoformat()

        # Group by surface for lock efficiency
        by_surface: dict[str, list[dict]] = {}
        for s in survivors:
            by_surface.setdefault(s["surface"], []).append(s)

        surface_config = {
            "attractor": (_ATTRACTORS_DIR, self._attractor_io_lock),
            "anamnesis": (_ANAMNESIS_DIR, self._anamnesis_io_lock),
            "topos": (_TOPOI_DIR, self._topoi_io_lock),
            "policy": (_POLICY_DIR, self._policy_io_lock),
        }

        for surface_name, items in by_surface.items():
            config = surface_config.get(surface_name)
            if not config:
                logger.debug("Converge: unknown surface %s, skipping", surface_name)
                continue
            surface_dir, io_lock = config

            actions = []
            with io_lock:
                for item in items:
                    op = item["op"]
                    op_type = op.get("op", "create")
                    slug = op.get("slug", "")
                    if not slug:
                        continue

                    path = surface_dir / f"{slug}.md"

                    if surface_name == "attractor":
                        # Attractor-specific ops: reinforce/expand/correct/create
                        if op_type in ("reinforce", "expand"):
                            new_evidence = op.get("evidence", op.get("new_evidence", ""))
                            if path.exists():
                                recompiled = self._recompile_entry(path, new_evidence)
                                if recompiled:
                                    path.write_text(recompiled, encoding="utf-8")
                                    actions.append(f"recompile:{slug}")
                                else:
                                    existing = path.read_text(encoding="utf-8")
                                    existing = existing.rstrip() + f"\n- {op_type.title()}d: {today} — {new_evidence}\n"
                                    path.write_text(existing, encoding="utf-8")
                                    actions.append(f"{op_type}:{slug}")
                            else:
                                logger.debug("Converge: %s target %s not found", op_type, slug)

                        elif op_type == "correct":
                            old_path = _ATTRACTORS_DIR / f"{slug}.md"
                            new_slug = op.get("corrected_slug", slug)
                            new_title = op.get("corrected_title", "")
                            reason = op.get("reason", "")
                            if old_path.exists():
                                existing = old_path.read_text(encoding="utf-8")
                                if new_title:
                                    file_lines = existing.split("\n")
                                    for idx, line in enumerate(file_lines):
                                        if line.startswith("# "):
                                            file_lines[idx] = f"# {new_title}"
                                            break
                                    existing = "\n".join(file_lines)
                                existing = existing.rstrip() + f"\n- Corrected: {today} — {reason}\n"
                                new_path = _ATTRACTORS_DIR / f"{new_slug}.md"
                                new_path.write_text(existing, encoding="utf-8")
                                if new_slug != slug:
                                    old_path.unlink()
                                actions.append(f"correct:{slug}→{new_slug}")

                        elif op_type == "create":
                            title = op.get("title", slug)
                            evidence = op.get("evidence", "")
                            if not evidence and not path.exists():
                                logger.debug("Converge: skipping empty attractor create for %s", slug)
                                actions.append(f"skip_empty:{slug}")
                                continue
                            if path.exists():
                                recompiled = self._recompile_entry(path, evidence)
                                if recompiled:
                                    path.write_text(recompiled, encoding="utf-8")
                                    actions.append(f"recompile:{slug}")
                                else:
                                    existing = path.read_text(encoding="utf-8")
                                    existing = existing.rstrip() + f"\n- Re-observed: {today} — {evidence}\n"
                                    path.write_text(existing, encoding="utf-8")
                                    actions.append(f"reinforce:{slug}")
                            else:
                                content = f"# {title}\n\n{evidence}\n\n- Strength: tentative\n- Last observed: {today}\n"
                                path.write_text(content, encoding="utf-8")
                                actions.append(f"create:{slug}")

                    else:
                        # Generic surfaces: create/update
                        content = op.get("content", "")
                        if not content and op_type == "create":
                            logger.debug("Converge: skipping empty create for %s/%s", surface_name, slug)
                            actions.append(f"skip_empty:{slug}")
                            continue
                        if op_type == "update" and path.exists():
                            recompiled = self._recompile_entry(path, content)
                            if recompiled:
                                path.write_text(recompiled, encoding="utf-8")
                                actions.append(f"recompile:{slug}")
                            else:
                                path.write_text(f"# {slug}\n\n{content}\n\n- Last observed: {today}\n", encoding="utf-8")
                                actions.append(f"update:{slug}")
                        elif op_type == "create":
                            if path.exists():
                                recompiled = self._recompile_entry(path, content)
                                if recompiled:
                                    path.write_text(recompiled, encoding="utf-8")
                                    actions.append(f"recompile:{slug}")
                                else:
                                    actions.append(f"exists:{slug}")
                            else:
                                path.write_text(f"# {slug}\n\n{content}\n\n- Last observed: {today}\n", encoding="utf-8")
                                actions.append(f"create:{slug}")

            if actions:
                self._trace(
                    "write_complete",
                    surface=surface_name,
                    actions=actions,
                )
                for a in actions:
                    logger.info("Converge %s: %s", surface_name, a)

    def _embed_single(self, utterance: str) -> None:
        """Embed a single utterance via OMLX /v1/embeddings and append to cache."""
        t0 = time.time()
        np = _import_numpy("Converge turn embedding cache")

        # Use OMLX's embeddings endpoint — same server, no in-process model load,
        # no Metal race with the command model.
        omlx_url = os.environ.get("SPOKE_OMLX_URL", "http://localhost:8001")
        url = _openai_endpoint(omlx_url, "embeddings")
        payload = json.dumps({
            "model": "Octen-Embedding-8B-mlx",
            "input": utterance[:500],
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        embedding = np.array(body["data"][0]["embedding"], dtype=np.float32)
        elapsed = time.time() - t0

        # Serialize the read-modify-write cycle so concurrent embeds
        # don't clobber each other's appended entries.
        with self._embed_io_lock:
            texts = []
            embeddings = np.empty((0, embedding.shape[0]), dtype=np.float32)

            if _TURN_EMBEDDINGS_PATH.exists():
                try:
                    data = np.load(_TURN_EMBEDDINGS_PATH, allow_pickle=False)
                    embeddings = data["embeddings"]
                    texts = json.loads(str(data["texts"]))
                except Exception:
                    pass

            texts.append(utterance[:500])
            embeddings = np.vstack([embeddings, embedding[np.newaxis, :]])

            # Trim to rolling window
            if len(texts) > _MAX_CACHED_EMBEDDINGS:
                texts = texts[-_MAX_CACHED_EMBEDDINGS:]
                embeddings = embeddings[-_MAX_CACHED_EMBEDDINGS:]

            # Atomic write
            tmp_path = _TURN_EMBEDDINGS_PATH.with_suffix(".tmp.npz")
            np.savez(tmp_path, embeddings=embeddings, texts=json.dumps(texts))
            tmp_path.replace(_TURN_EMBEDDINGS_PATH)

        self._trace(
            "embed_complete",
            elapsed=round(elapsed, 2),
            utterance=utterance[:80],
            cache_size=len(texts),
        )
        logger.debug("Converge: embedded turn in %.1fs (cache: %d)", elapsed, len(texts))

    def _trace(self, event: str, **kwargs) -> None:
        """Append to the trace log."""
        _append_trace(_TRACE_PATH, event, **kwargs)


def load_turn_embeddings() -> tuple[Any, list[str]] | None:
    """Load the pre-computed turn embedding cache.

    Returns (embeddings, texts) or None if no cache exists.
    Used by the guided compaction mode for pure-numpy cosine search.
    """
    if not _TURN_EMBEDDINGS_PATH.exists():
        return None
    try:
        np = _import_numpy("guided compaction turn cache")
        data = np.load(_TURN_EMBEDDINGS_PATH, allow_pickle=False)
        embeddings = data["embeddings"]
        texts = json.loads(str(data["texts"]))
        return embeddings, texts
    except Exception:
        return None
