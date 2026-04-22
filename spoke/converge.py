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
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"
_ATTRACTOR_INDEX_PATH = Path.home() / ".config" / "spoke" / "attractor-index.npz"
_TRACE_PATH = Path.home() / ".config" / "spoke" / "converge-trace.jsonl"
_TURN_EMBEDDINGS_PATH = Path.home() / ".config" / "spoke" / "turn-embeddings.npz"
_MAX_CACHED_EMBEDDINGS = 100  # rolling window of recent turn embeddings

_CARVE_SYSTEM_PROMPT = """\
You are a personal attractor carver. You observe a single turn of voice
interaction between a user and their assistant, and identify any personal
attractors revealed — durable concerns, recurring interests, or persistent
preferences that transcend the immediate task.

The user speaks via voice dictation. Their input contains transcription
errors, false starts, and informal language. Read through them to the
intent. Do NOT create attractors from transcription artifacts — if a word
looks like a garbled version of a technical term, it probably is. "Tractor"
is almost certainly "attractor." "Epístaxis" is correct Greek, not a typo.

Personal attractors are NOT project-specific tasks or bugs. They are patterns
like:
- Aesthetic preferences (visual, auditory, interaction design)
- Technical philosophy (architecture, abstractions, tool choices)
- Workflow patterns (how they prefer to collaborate, communicate)
- Quality standards they consistently enforce
- Recurring interests regardless of current task

You will be given the user's EXISTING personal attractors. Before creating
a new attractor, check if the utterance is evidence for an existing one.

Your response is a JSON array of operations. Each operation is one of:

1. REINFORCE an existing attractor (re-observed evidence):
   {"op": "reinforce", "slug": "<existing-slug>", "evidence": "New evidence observed"}

2. EXPAND an existing attractor (broaden its scope with new detail):
   {"op": "expand", "slug": "<existing-slug>", "new_evidence": "Additional detail", "new_title": "Optional broader title"}

3. CORRECT an existing attractor (fix a transcription error or misattribution):
   {"op": "correct", "slug": "<existing-slug>", "corrected_slug": "fixed-slug", "corrected_title": "Fixed title", "reason": "Why"}

4. CREATE a new attractor (genuinely novel signal not covered by any existing one):
   {"op": "create", "slug": "kebab-case-id", "title": "Short title", "evidence": "One sentence"}

Rules:
- Prefer reinforce/expand over create. New attractors need STRONG signal.
- Be skeptical. Most turns are just task execution and reveal nothing durable.
- If the turn is a test, a short command, or purely about a specific bug, return [].
- Do NOT create attractors about the project's specific features or bugs.
- Output ONLY the JSON array. No markdown, no commentary.
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
            or os.environ.get("SPOKE_COMMAND_MODEL", "Qwen3.6-35B-A3B-bf16")
        )
        self._pending: list[str] = []  # user utterances not yet carved
        self._embed_pending: list[str] = []  # user utterances not yet embedded
        self._lock = threading.Lock()
        self._embed_io_lock = threading.Lock()  # serialize embed cache read-modify-write
        self._attractor_io_lock = threading.Lock()  # serialize attractor file mutations
        self._thread: threading.Thread | None = None
        self._embed_model_loaded = False
        _ATTRACTORS_DIR.mkdir(parents=True, exist_ok=True)

    def on_turn_complete(self, user_utterance: str, assistant_response: str) -> None:
        """Called after each command turn. Fires async carve + embed."""
        if not user_utterance or not user_utterance.strip():
            return

        with self._lock:
            # Always embed (even short turns have semantic content)
            self._embed_pending.append(user_utterance)
            # Only carve substantive turns (>= 10 words)
            if len(user_utterance.split()) >= 10:
                self._pending.append(user_utterance)

        # Fire background worker if not already running
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._background_loop, daemon=True)
            self._thread.start()

    def _background_loop(self) -> None:
        """Background thread: dispatch carve and embed work concurrently.

        OMLX batch parallel gives near-linear throughput scaling, so firing
        multiple requests simultaneously is better than serializing them.
        Each pending item gets its own thread; they all batch together on
        the server side.
        """
        while True:
            work: list[tuple[str, str]] = []  # (kind, utterance)

            with self._lock:
                while self._pending:
                    work.append(("carve", self._pending.pop(0)))
                while self._embed_pending:
                    work.append(("embed", self._embed_pending.pop(0)))

            if not work:
                return

            # Fire all work items concurrently
            threads: list[threading.Thread] = []
            for kind, utterance in work:
                fn = self._carve_single if kind == "carve" else self._embed_single
                t = threading.Thread(target=self._safe_call, args=(fn, utterance), daemon=True)
                t.start()
                threads.append(t)

            # Wait for all to complete before checking for more work
            for t in threads:
                t.join(timeout=120)

    @staticmethod
    def _safe_call(fn, utterance: str) -> None:
        try:
            fn(utterance)
        except Exception:
            logger.debug("Converge %s failed", fn.__name__, exc_info=True)

    def _load_existing_attractors_context(self) -> str:
        """Build a compact summary of existing personal attractors for the prompt."""
        if not _ATTRACTORS_DIR.is_dir():
            return ""
        lines = []
        for f in sorted(_ATTRACTORS_DIR.iterdir()):
            if f.is_file() and f.suffix == ".md":
                try:
                    text = f.read_text(encoding="utf-8")
                    # Extract title and evidence
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
                except OSError:
                    continue
        if not lines:
            return ""
        return "Existing personal attractors:\n" + "\n".join(lines)

    def _carve_single(self, utterance: str) -> None:
        """Send one utterance to the model for attractor carving."""
        t0 = time.time()

        existing_context = self._load_existing_attractors_context()
        user_prompt = (
            f"User utterance from a voice interaction:\n\n"
            f"\"{utterance}\"\n\n"
        )
        if existing_context:
            user_prompt += f"{existing_context}\n\n"
        user_prompt += "Identify attractor operations for this utterance."

        url = _openai_endpoint(self._base_url, "chat/completions")
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _CARVE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": 0.3,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        result_text = body["choices"][0]["message"]["content"]
        elapsed = time.time() - t0

        # Parse response
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            ops = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Converge: failed to parse response: %s", cleaned[:100])
            self._trace("carve_parse_error", elapsed=elapsed, raw=cleaned[:200])
            return

        if not ops:
            self._trace("carve_empty", elapsed=elapsed, utterance=utterance[:100])
            return

        # Process operations — serialize attractor file mutations so
        # concurrent carve threads don't clobber each other's writes.
        today = date.today().isoformat()
        actions = []
        with self._attractor_io_lock:
            for op in ops:
                op_type = op.get("op", "create")
                slug = op.get("slug", "")
                if not slug:
                    continue

                if op_type == "reinforce":
                    path = _ATTRACTORS_DIR / f"{slug}.md"
                    if path.exists():
                        existing = path.read_text(encoding="utf-8")
                        new_evidence = op.get("evidence", "")
                        if "Strength: tentative" in existing:
                            # Re-observed → upgrade to strong
                            existing = existing.replace("Strength: tentative", "Strength: strong")
                            logger.info("Converge: reinforced %s → strong", slug)
                        existing = existing.rstrip() + f"\n- Re-observed: {today} — {new_evidence}\n"
                        path.write_text(existing, encoding="utf-8")
                        actions.append(f"reinforce:{slug}")
                    else:
                        logger.debug("Converge: reinforce target %s not found, skipping", slug)

                elif op_type == "expand":
                    path = _ATTRACTORS_DIR / f"{slug}.md"
                    if path.exists():
                        existing = path.read_text(encoding="utf-8")
                        new_title = op.get("new_title")
                        new_evidence = op.get("new_evidence", "")
                        if new_title:
                            # Replace title line
                            lines = existing.split("\n")
                            for i, line in enumerate(lines):
                                if line.startswith("# "):
                                    lines[i] = f"# {new_title}"
                                    break
                            existing = "\n".join(lines)
                        existing = existing.rstrip() + f"\n- Expanded: {today} — {new_evidence}\n"
                        path.write_text(existing, encoding="utf-8")
                        actions.append(f"expand:{slug}")
                        logger.info("Converge: expanded %s", slug)

                elif op_type == "correct":
                    old_path = _ATTRACTORS_DIR / f"{slug}.md"
                    new_slug = op.get("corrected_slug", slug)
                    new_title = op.get("corrected_title", "")
                    reason = op.get("reason", "")
                    if old_path.exists():
                        existing = old_path.read_text(encoding="utf-8")
                        if new_title:
                            lines = existing.split("\n")
                            for i, line in enumerate(lines):
                                if line.startswith("# "):
                                    lines[i] = f"# {new_title}"
                                    break
                            existing = "\n".join(lines)
                        existing = existing.rstrip() + f"\n- Corrected: {today} — {reason}\n"
                        new_path = _ATTRACTORS_DIR / f"{new_slug}.md"
                        new_path.write_text(existing, encoding="utf-8")
                        if new_slug != slug:
                            old_path.unlink()
                        actions.append(f"correct:{slug}→{new_slug}")
                        logger.info("Converge: corrected %s → %s (%s)", slug, new_slug, reason)

                elif op_type == "create":
                    title = op.get("title", slug)
                    evidence = op.get("evidence", "")
                    path = _ATTRACTORS_DIR / f"{slug}.md"
                    if path.exists():
                        # Already exists — treat as reinforce
                        existing = path.read_text(encoding="utf-8")
                        if "Strength: tentative" in existing:
                            existing = existing.replace("Strength: tentative", "Strength: strong")
                        existing = existing.rstrip() + f"\n- Re-observed: {today} — {evidence}\n"
                        path.write_text(existing, encoding="utf-8")
                        actions.append(f"reinforce:{slug}")
                    else:
                        content = f"# {title}\n\n- Evidence: {evidence}\n- Strength: tentative\n- Observed: {today}\n"
                        path.write_text(content, encoding="utf-8")
                        actions.append(f"create:{slug}")
                        logger.info("Converge: created %s", slug)

        self._trace(
            "carve_complete",
            elapsed=elapsed,
            utterance=utterance[:100],
            ops_received=len(ops),
            actions=actions,
        )

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
