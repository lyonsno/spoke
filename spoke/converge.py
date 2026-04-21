"""Converge — per-turn attractor carving via OMLX batch parallel.

After each command response completes, fires an async request to the local
model (same endpoint, same model) asking it to identify personal attractors
from that turn. OMLX's batch parallel scheduling handles contention with
interactive command requests — carve requests simply wait in the queue when
the user is actively talking.

The carver writes identified attractors to ~/.config/spoke/attractors/ and
appends to the trace log at ~/.config/spoke/converge-trace.jsonl.
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

logger = logging.getLogger(__name__)

_ATTRACTORS_DIR = Path.home() / ".config" / "spoke" / "attractors"
_TRACE_PATH = Path.home() / ".config" / "spoke" / "converge-trace.jsonl"

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
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        _ATTRACTORS_DIR.mkdir(parents=True, exist_ok=True)

    def on_turn_complete(self, user_utterance: str, assistant_response: str) -> None:
        """Called after each command turn. Fires async carve."""
        if not user_utterance or not user_utterance.strip():
            return
        # Don't carve trivial turns (< 10 words)
        if len(user_utterance.split()) < 10:
            logger.debug("Converge: skipping short utterance (%d words)", len(user_utterance.split()))
            return

        with self._lock:
            self._pending.append(user_utterance)

        # Fire background carve if not already running
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._carve_loop, daemon=True)
            self._thread.start()

    def _carve_loop(self) -> None:
        """Background thread: carve pending utterances one at a time."""
        while True:
            with self._lock:
                if not self._pending:
                    return
                utterance = self._pending.pop(0)

            try:
                self._carve_single(utterance)
            except Exception:
                logger.debug("Converge carve failed", exc_info=True)

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

    def _trace(self, event: str, **kwargs) -> None:
        """Append to the trace log."""
        try:
            entry = {"timestamp": datetime.now().isoformat(), "event": event, **kwargs}
            with open(_TRACE_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
