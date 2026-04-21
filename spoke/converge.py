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

import numpy as np

logger = logging.getLogger(__name__)

_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"
_TRACE_PATH = Path.home() / ".config" / "spoke" / "converge-trace.jsonl"
_TURN_EMBEDDINGS_PATH = Path.home() / ".config" / "spoke" / "turn-embeddings.npz"
_MAX_CACHED_EMBEDDINGS = 100  # rolling window of recent turn embeddings

_CARVE_SYSTEM_PROMPT = """\
You are a personal attractor carver. You observe a single turn of voice
interaction between a user and their assistant, and identify any personal
attractors revealed — durable concerns, recurring interests, or persistent
preferences that transcend the immediate task.

Personal attractors are NOT project-specific tasks or bugs. They are patterns
like:
- Aesthetic preferences (visual, auditory, interaction design)
- Technical philosophy (architecture, abstractions, tool choices)
- Workflow patterns (how they prefer to collaborate, communicate)
- Quality standards they consistently enforce
- Recurring interests regardless of current task

If this turn reveals NO personal attractor signal (it's purely task-execution,
a test, or too short), respond with exactly: []

Otherwise, respond with a JSON array of at most 2 attractors:
[{"slug": "kebab-case-id", "title": "Short title", "evidence": "One sentence of what you observed", "strength": "tentative"}]

Rules:
- Only "tentative" strength on first observation. Only the merge process upgrades to "strong".
- Be concrete and specific, not vague.
- Output ONLY the JSON array (or empty array). No markdown, no commentary.
"""


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
        """Background thread: carve and embed pending utterances."""
        while True:
            utterance_to_carve = None
            utterance_to_embed = None

            with self._lock:
                if self._pending:
                    utterance_to_carve = self._pending.pop(0)
                if self._embed_pending:
                    utterance_to_embed = self._embed_pending.pop(0)

            if utterance_to_carve is None and utterance_to_embed is None:
                return

            if utterance_to_carve:
                try:
                    self._carve_single(utterance_to_carve)
                except Exception:
                    logger.debug("Converge carve failed", exc_info=True)

            if utterance_to_embed:
                try:
                    self._embed_single(utterance_to_embed)
                except Exception:
                    logger.debug("Converge embed failed", exc_info=True)

    def _carve_single(self, utterance: str) -> None:
        """Send one utterance to the model for attractor carving."""
        t0 = time.time()

        user_prompt = (
            f"User utterance from a voice interaction:\n\n"
            f"\"{utterance}\"\n\n"
            f"Identify any personal attractors revealed."
        )

        url = f"{self._base_url}/v1/chat/completions"
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
            attractors = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Converge: failed to parse response: %s", cleaned[:100])
            self._trace("carve_parse_error", elapsed=elapsed, raw=cleaned[:200])
            return

        if not attractors:
            self._trace("carve_empty", elapsed=elapsed, utterance=utterance[:100])
            return

        # Write attractors
        written = 0
        for a in attractors:
            slug = a.get("slug", "")
            if not slug:
                continue
            title = a.get("title", slug)
            evidence = a.get("evidence", "")
            strength = a.get("strength", "tentative")

            path = _ATTRACTORS_DIR / f"{slug}.md"
            if path.exists():
                # Already exists — check if we should upgrade strength
                existing = path.read_text(encoding="utf-8")
                if "Strength: strong" in existing:
                    continue  # already strong, don't downgrade
                if "Strength: tentative" in existing and strength == "tentative":
                    # Re-observed tentative → upgrade to strong
                    strength = "strong"
                    logger.info("Converge: upgrading attractor %s to strong", slug)

            today = date.today().isoformat()
            content = f"# {title}\n\n- Evidence: {evidence}\n- Strength: {strength}\n- Observed: {today}\n"
            path.write_text(content, encoding="utf-8")
            written += 1
            logger.info("Converge: wrote attractor %s (%s)", slug, strength)

        self._trace(
            "carve_complete",
            elapsed=elapsed,
            utterance=utterance[:100],
            attractors_found=len(attractors),
            attractors_written=written,
            slugs=[a.get("slug", "") for a in attractors],
        )

    def _embed_single(self, utterance: str) -> None:
        """Embed a single utterance and append to the rolling cache."""
        import sys

        t0 = time.time()

        # Lazy-load the embedding library
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from converge_embed_lib import embed_texts

        embedding = embed_texts([utterance])[0]  # (dim,)
        elapsed = time.time() - t0

        # Load existing cache, append, trim to max, save
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
        try:
            entry = {"timestamp": datetime.now().isoformat(), "event": event, **kwargs}
            with open(_TRACE_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


def load_turn_embeddings() -> tuple[np.ndarray, list[str]] | None:
    """Load the pre-computed turn embedding cache.

    Returns (embeddings, texts) or None if no cache exists.
    Used by the guided compaction mode for pure-numpy cosine search.
    """
    if not _TURN_EMBEDDINGS_PATH.exists():
        return None
    try:
        data = np.load(_TURN_EMBEDDINGS_PATH, allow_pickle=False)
        embeddings = data["embeddings"]
        texts = json.loads(str(data["texts"]))
        return embeddings, texts
    except Exception:
        return None
