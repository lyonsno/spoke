"""Thinking narrator sidecar.

Reads streaming thinking tokens from a reasoning model and produces
short, present-participle status lines via a small local model (or
cloud endpoint).  Architecture A: each summary becomes an assistant
turn in a growing chat history so the narrator naturally *continues*
its own summary stream.

Configuration (env vars):
    SPOKE_NARRATOR_URL      OpenAI-compatible base URL (default: same as command URL)
    SPOKE_NARRATOR_MODEL    Model ID (default: Bonsai-8B-mlx-1bit)
    SPOKE_NARRATOR_API_KEY  Bearer token (falls back to OMLX_SERVER_API_KEY)
    SPOKE_NARRATOR_ENABLED  "1" to enable, "0" to disable (default: "1")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable
from urllib.parse import urlparse

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

_DEFAULT_NARRATOR_MODEL = "Bonsai-8B-mlx-1bit"

_SYSTEM_PROMPT = """\
You narrate an AI's thinking. Read the reasoning excerpt and write a \
short status line — what it is figuring out RIGHT NOW.

Rules:
- One fragment or short sentence. 8–15 words. Never exceed 15 words.
- Start with a present participle: Considering, Evaluating, Comparing, \
Breaking down, Checking, Debugging, Revisiting, Weighing, Testing, etc.
- Be specific: name the concrete thing (algorithm, variable, edge case, \
tradeoff). Never generic ("Thinking about the problem").
- IMPORTANT: Each summary MUST describe something different from your \
previous summaries. Even if the AI is still on the same broad topic, \
zoom in on the specific sub-step, detail, or angle that is NEW in \
this excerpt. What changed? What shifted? What is it drilling into now \
that it wasn't before? If it's reconsidering something, say what and why.
- Say what the AI is doing, not "the user".
- No preamble, no commentary. Output ONLY the status line."""

_LOADING_VAMP_SYSTEM_PROMPT = """\
You write short, fun, varied status lines while a large AI model \
is loading into GPU memory. The user is waiting and can see your \
messages, so be entertaining and informative.

Rules:
- ONE line only. 10–18 words. Must fit on one line of a small overlay.
- NEVER repeat a previous line. Each must be completely different in \
structure, topic, angle, and wording. Read your previous messages — \
if your new line resembles ANY of them, throw it out and try again.
- Vary your approach: alternate between humor, technical commentary, \
encouragement, fun facts, and observations about the wait.
- You can reference: the model name, how long it's been loading, GPU \
memory, what the user asked, the hardware, AI trivia.
- Tone: witty, slightly irreverent, like a clever loading screen.
- No preamble, no quotes. Output ONLY the status line.

Examples (do NOT reuse these, they show the range of styles):
- Waking up 27 billion parameters — neurons need their coffee too
- Tetris but the blocks are 15GB weight matrices
- Your question is in the queue, the GPU just needs a minute
- Fun fact: this model has more parameters than you have neurons
- Almost there — stretching before the big performance"""

_LOADING_VAMP_INTERVAL_S = 8.0  # seconds between subsequent vamp lines

# ── chunking parameters ─────────────────────────────────────────────

_TARGET_CHUNK_TOKENS = 300
_MIN_INTERVAL_S = 5.0  # minimum seconds between narrator calls
_MAX_TOKENS = 30        # generation budget for each summary


def _rough_token_count(text: str) -> int:
    """Approximate token count (words × 1.3)."""
    return int(len(text.split()) * 1.3)


class ThinkingNarrator:
    """Accumulates thinking tokens and periodically summarizes them.

    Thread-safe.  Call ``feed()`` from the streaming thread with each
    thinking token.  Summaries are delivered via the ``on_summary``
    callback on a background thread — the caller is responsible for
    marshalling to the main thread if needed.
    """

    def __init__(
        self,
        on_summary: Callable[[str], None],
        on_thinking_collapsed: Callable[[str], None] | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._on_summary = on_summary
        self._on_thinking_collapsed = on_thinking_collapsed

        # Endpoint config — fall back to command URL, then localhost
        raw_url = (
            base_url
            or os.environ.get("SPOKE_NARRATOR_URL")
            or os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8001")
        ).rstrip("/")
        path = urlparse(raw_url).path.rstrip("/")
        self._url_has_version_prefix = any(
            seg.startswith("v") and seg[1:].replace("beta", "").isdigit()
            for seg in path.split("/") if seg
        )
        self._base_url = raw_url
        self._model = (
            model
            or os.environ.get("SPOKE_NARRATOR_MODEL", _DEFAULT_NARRATOR_MODEL)
        )
        self._api_key = (
            api_key
            or os.environ.get("SPOKE_NARRATOR_API_KEY")
            or os.environ.get("OMLX_SERVER_API_KEY", "")
        )

        # Command URL for polling OMLX status (may differ from narrator URL)
        self._command_url = (
            os.environ.get("SPOKE_COMMAND_URL", "http://localhost:8001")
        ).rstrip("/")
        self._command_api_key = (
            os.environ.get("SPOKE_COMMAND_API_KEY")
            or os.environ.get("OMLX_SERVER_API_KEY", "")
        )

        # State (guarded by _lock)
        self._lock = threading.Lock()
        self._buffer = ""          # accumulated thinking tokens since last dispatch
        self._last_dispatch = 0.0  # monotonic time of last narrator call
        self._messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        self._active = False
        self._dispatch_thread: threading.Thread | None = None
        self._pending_dispatch = False  # a dispatch is in flight

        # Loading vamp state
        self._vamp_active = False
        self._vamp_thread: threading.Thread | None = None
        self._vamp_start_time = 0.0

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("SPOKE_NARRATOR_ENABLED", "1") != "0"

    def start(self) -> None:
        """Begin a new narration session (new thinking phase)."""
        with self._lock:
            self._buffer = ""
            self._thinking_start = 0.0  # set on first thinking token, not here
            self._last_dispatch = time.monotonic()
            self._messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            self._active = True
            self._pending_dispatch = False
        logger.info("Narrator session started")

    def stop(self) -> None:
        """End the narration session without a collapsed summary."""
        with self._lock:
            self._active = False
            self._buffer = ""
        logger.info("Narrator session stopped")

    def stop_and_summarize(self) -> None:
        """End the narration session and produce a collapsed summary.

        Fires one final call to the narrator model asking for a concise
        wrap-up of what was thought about.  The result is delivered via
        on_thinking_collapsed as "Thought for Xs · {summary}".
        Always produces a topic summary for thinking >1s, even if no
        live narrator summaries were dispatched during the session.
        """
        with self._lock:
            self._active = False
            remaining_buffer = self._buffer
            self._buffer = ""
            if self._thinking_start > 0:
                elapsed = time.monotonic() - self._thinking_start
            else:
                elapsed = 0.0  # no thinking tokens were received
            messages = list(self._messages)

        if not self._on_thinking_collapsed:
            return
        if elapsed < 1.0:
            return  # sub-second thinking — not worth reporting

        # Immediately emit the bare duration so it's in the overlay
        # before response tokens start flowing. The topic will be
        # appended when Bonsai returns.
        self._on_thinking_collapsed(f"Thought for {elapsed:.0f}s")

        # Fire async wrap-up — always call Bonsai for the topic summary.
        # If no live summaries were produced, feed the raw thinking buffer
        # directly so the model has something to summarize.
        if len(messages) < 3 and remaining_buffer:
            messages.append({
                "role": "user",
                "content": f"Current reasoning excerpt:\n\n{remaining_buffer}",
            })

        t = threading.Thread(
            target=self._produce_collapsed_summary,
            args=(messages, elapsed),
            daemon=True,
        )
        t.start()

    def _produce_collapsed_summary(self, messages: list[dict], elapsed: float) -> None:
        """Generate and deliver the collapsed thinking summary."""
        try:
            messages = list(messages)
            messages.append({
                "role": "user",
                "content": (
                    "The AI has finished thinking. Write a single short phrase "
                    "(5-10 words, no participle needed) summarizing WHAT it "
                    "thought about overall. Like a section heading. "
                    "Examples: 'palindrome algorithm design and edge cases', "
                    "'debugging the auth token refresh flow', "
                    "'weighing three caching strategies'. "
                    "Output ONLY the phrase."
                ),
            })
            topic = self._chat_completion(messages, max_tokens=25)
            if topic:
                # Append topic to the already-displayed duration line
                logger.info("Thinking topic: %s", topic)
                self._on_thinking_collapsed(f" · {topic}")
        except Exception:
            logger.exception("Failed to produce collapsed summary")

    def feed(self, token: str) -> None:
        """Feed a thinking token.  May trigger an async narrator call."""
        with self._lock:
            if not self._active:
                return
            # Start the thinking clock on the first actual thinking token,
            # not when start() was called (which may include model load time).
            if self._thinking_start == 0.0:
                self._thinking_start = time.monotonic()
            self._buffer += token
            now = time.monotonic()
            elapsed = now - self._last_dispatch
            tokens = _rough_token_count(self._buffer)

            # Dispatch when BOTH conditions met: enough tokens AND enough time
            should_dispatch = (
                tokens >= _TARGET_CHUNK_TOKENS
                and elapsed >= _MIN_INTERVAL_S
                and not self._pending_dispatch
            )
            if not should_dispatch:
                return

            chunk = self._buffer
            self._buffer = ""
            self._last_dispatch = now
            self._pending_dispatch = True

        # Fire async
        t = threading.Thread(target=self._dispatch, args=(chunk,), daemon=True)
        t.start()

    def _dispatch(self, chunk: str) -> None:
        """Call the narrator model and deliver the summary."""
        try:
            # Build user message with the verbatim chunk
            user_content = f"Current reasoning excerpt:\n\n{chunk}"

            with self._lock:
                self._messages.append({"role": "user", "content": user_content})
                messages = list(self._messages)

            summary = self._chat_completion(messages)

            if summary:
                # Add as assistant turn for continuity
                with self._lock:
                    self._messages.append({"role": "assistant", "content": summary})
                    if not self._active:
                        return
                logger.info("Narrator summary: %s", summary)
                self._on_summary(summary)

        except Exception:
            logger.exception("Narrator dispatch failed")
        finally:
            with self._lock:
                self._pending_dispatch = False

    # ── loading vamp mode ─────────────────────────────────────────

    def start_loading_vamp(self, utterance: str = "", model_id: str = "") -> None:
        """Start generating fun vamp lines while a model is loading.

        Call this when you detect the model server is loading. The vamp
        loop polls OMLX /api/status for context and generates lines via
        the narrator sidecar model.
        """
        with self._lock:
            if self._vamp_active:
                return
            self._vamp_active = True
            self._vamp_start_time = time.monotonic()

        logger.info("Loading vamp started for model=%s", model_id)
        self._vamp_thread = threading.Thread(
            target=self._vamp_loop,
            args=(utterance, model_id),
            daemon=True,
        )
        self._vamp_thread.start()

    def stop_loading_vamp(self) -> None:
        """Stop the loading vamp loop."""
        with self._lock:
            self._vamp_active = False
        logger.info("Loading vamp stopped")

    def _vamp_loop(self, utterance: str, model_id: str) -> None:
        """Background loop that generates vamp lines every few seconds."""
        # Wait before starting — don't vamp on fast responses
        _VAMP_GRACE_PERIOD_S = 3.0
        grace_slept = 0.0
        while grace_slept < _VAMP_GRACE_PERIOD_S:
            time.sleep(0.5)
            grace_slept += 0.5
            with self._lock:
                if not self._vamp_active:
                    return

        # Grace period expired — model hasn't responded yet, show loading indicator
        loading_label = f"Loading {model_id}…" if model_id else "Loading model…"
        self._on_summary(loading_label)
        logger.info("Loading indicator shown: %s", loading_label)

        vamp_messages: list[dict] = [
            {"role": "system", "content": _LOADING_VAMP_SYSTEM_PROMPT},
        ]

        while True:
            with self._lock:
                if not self._vamp_active:
                    return
                elapsed = time.monotonic() - self._vamp_start_time

            # Gather context for the vamp prompt
            loading_context = self._poll_loading_status()
            context_parts = []
            if model_id:
                context_parts.append(f"Model being loaded: {model_id}")
            if loading_context:
                context_parts.append(loading_context)
            context_parts.append(f"Time waiting so far: {elapsed:.0f} seconds")
            if utterance:
                context_parts.append(f"The user asked: \"{utterance}\"")

            user_content = "Generate the next loading status line.\n\n" + "\n".join(context_parts)
            vamp_messages.append({"role": "user", "content": user_content})

            try:
                summary = self._chat_completion(
                    vamp_messages, temperature=0.9, max_tokens=40
                )
                if summary:
                    # Deduplicate: skip if too similar to previous line
                    prev_assistants = [
                        m["content"] for m in vamp_messages
                        if m["role"] == "assistant"
                    ]
                    if prev_assistants and self._lines_too_similar(
                        summary, prev_assistants[-1]
                    ):
                        logger.info("Vamp line too similar, skipping: %s", summary)
                        # Remove the user message we just added so history stays clean
                        vamp_messages.pop()
                    else:
                        vamp_messages.append({"role": "assistant", "content": summary})
                        with self._lock:
                            if not self._vamp_active:
                                return
                        logger.info("Vamp line: %s", summary)
                        self._on_summary(summary)
            except Exception:
                logger.exception("Vamp dispatch failed")

            # Wait before next vamp line (check for stop every 500ms)
            waited = 0.0
            while waited < _LOADING_VAMP_INTERVAL_S:
                time.sleep(0.5)
                waited += 0.5
                with self._lock:
                    if not self._vamp_active:
                        return

    @staticmethod
    def _lines_too_similar(a: str, b: str, threshold: float = 0.6) -> bool:
        """Return True if two lines share more than threshold of their words."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        smaller = min(len(words_a), len(words_b))
        return (overlap / smaller) > threshold

    def _poll_loading_status(self) -> str:
        """Poll OMLX /api/status for loading context."""
        try:
            headers = {}
            if self._command_api_key:
                headers["Authorization"] = f"Bearer {self._command_api_key}"
            req = urllib.request.Request(
                f"{self._command_url}/api/status",
                headers=headers,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())

            parts = []
            models_loading = data.get("models_loading", 0)
            if models_loading:
                parts.append(f"{models_loading} model(s) currently loading")
            mem_used = data.get("model_memory_used", 0)
            mem_max = data.get("model_memory_max", 0)
            if mem_max > 0:
                used_gb = mem_used / (1024 ** 3)
                max_gb = mem_max / (1024 ** 3)
                parts.append(f"GPU memory: {used_gb:.1f}GB / {max_gb:.1f}GB")
            loaded = data.get("loaded_models", [])
            if loaded:
                parts.append(f"Models already in memory: {', '.join(loaded[:3])}")
            return ". ".join(parts)
        except Exception:
            logger.debug("Failed to poll OMLX status", exc_info=True)
            return ""

    def _chat_completion(self, messages: list[dict], *, temperature: float = 0.3, max_tokens: int = _MAX_TOKENS) -> str:
        """Synchronous chat completion call."""
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = (
            f"{self._base_url}/chat/completions"
            if self._url_has_version_prefix
            else f"{self._base_url}/v1/chat/completions"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )

        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.monotonic() - t0
        logger.info("Narrator call: %.2fs, %d messages", elapsed, len(messages))

        return result["choices"][0]["message"]["content"].strip()
