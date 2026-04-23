"""Tests for Converge runtime extraction and portability seams."""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_converge():
    sys.modules.pop("spoke.converge", None)
    return importlib.import_module("spoke.converge")


def _load_converge_embed_lib():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "converge_embed_lib.py"
    spec = importlib.util.spec_from_file_location("test_converge_embed_lib", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCommandClient:
    def __init__(self):
        self._history = [
            [
                {"role": "user", "content": "alpha request"},
                {"role": "assistant", "content": "alpha answer"},
            ],
            [
                {"role": "user", "content": "beta request"},
                {"role": "assistant", "content": "beta answer"},
            ],
        ]
        self.save_calls = 0

    def _save_history(self):
        self.save_calls += 1


class TestConvergeService:
    def test_compact_history_drop_tool_results_strips_tool_calls(self):
        mod = _import_converge()
        client = _FakeCommandClient()
        client._history = [
            [
                {"role": "user", "content": "alpha request"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
                {"role": "assistant", "content": "alpha answer"},
            ]
        ]

        result = mod.compact_history(client, {"mode": "drop_tool_results", "n": 0})

        assert result == {
            "status": "ok",
            "mode": "drop_tool_results",
            "turns_compacted": 1,
            "turns_total": 1,
        }
        assert client._history == [
            [
                {"role": "user", "content": "alpha request"},
                {"role": "assistant", "content": "Let me check."},
                {"role": "assistant", "content": "alpha answer"},
            ]
        ]
        assert client.save_calls == 1

    def test_compact_history_guided_mode_lives_in_converge_module(self, tmp_path):
        mod = _import_converge()
        client = _FakeCommandClient()
        index_path = tmp_path / "attractor-index.npz"
        trace_path = tmp_path / "converge-trace.jsonl"

        np.savez(
            index_path,
            full_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            summary_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            metadata=json.dumps(
                [
                    {"source": "personal", "slug": "keep-alpha", "summary": "Alpha summary"},
                    {"source": "personal", "slug": "keep-beta", "summary": "Beta summary"},
                ]
            ),
        )

        result = mod.compact_history(
            client,
            {"mode": "guided", "n": 0, "top_k": 1, "threshold": 0.2},
            index_path=index_path,
            trace_path=trace_path,
            turn_embeddings_loader=lambda: (
                np.array([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32),
                ["alpha request", "beta request"],
            ),
        )

        assert result["status"] == "ok"
        assert result["mode"] == "guided"
        assert result["retention_flags"] == [
            {
                "source": "personal",
                "attractor": "keep-alpha",
                "summary": "Alpha summary",
                "score": 1.0,
            }
        ]
        assert client.save_calls == 0

    def test_converge_module_import_does_not_require_numpy_until_runtime(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = """
import builtins
import sys

repo_root = sys.argv[1]
sys.path.insert(0, repo_root)
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("blocked numpy")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
import spoke.converge
print("imported")
"""
        result = subprocess.run(
            ["python3", "-c", script, str(repo_root)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "imported"

    def test_turn_carver_chat_endpoint_respects_existing_version_prefix(
        self, monkeypatch, tmp_path
    ):
        mod = _import_converge()
        monkeypatch.setattr(mod, "_ATTRACTORS_DIR", tmp_path / "attractors")
        monkeypatch.setattr(mod, "_TRACE_PATH", tmp_path / "trace.jsonl")
        mod._ATTRACTORS_DIR.mkdir(parents=True, exist_ok=True)

        seen = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "[]"}}]}
                ).encode("utf-8")

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            return _Response()

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        carver = mod.TurnCarver(
            base_url="https://example.test/v1",
            api_key="token",
            model="demo-model",
        )

        carver._carve_single("This is a long enough utterance to carve safely.")

        assert seen["url"] == "https://example.test/v1/chat/completions"

    def test_turn_carver_embeddings_endpoint_respects_existing_version_prefix(
        self, monkeypatch, tmp_path
    ):
        mod = _import_converge()
        monkeypatch.setattr(mod, "_TURN_EMBEDDINGS_PATH", tmp_path / "turn-embeddings.npz")
        monkeypatch.setattr(mod, "_TRACE_PATH", tmp_path / "trace.jsonl")
        monkeypatch.setenv("SPOKE_OMLX_URL", "https://example.test/v1")

        seen = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
                ).encode("utf-8")

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            return _Response()

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        carver = mod.TurnCarver(
            base_url="http://localhost:8090",
            api_key="token",
            model="demo-model",
        )

        carver._embed_single("hello world")

        assert seen["url"] == "https://example.test/v1/embeddings"


class TestContextWindowCarver:
    """Tests for the rolling context window and every-other-turn cadence."""

    def _make_carver(self, monkeypatch, tmp_path, mod, urlopen_fn):
        monkeypatch.setattr(mod, "_ATTRACTORS_DIR", tmp_path / "attractors")
        monkeypatch.setattr(mod, "_TRACE_PATH", tmp_path / "trace.jsonl")
        (tmp_path / "attractors").mkdir(exist_ok=True)
        monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen_fn)
        return mod.TurnCarver(
            base_url="http://localhost:8090",
            api_key="token",
            model="demo-model",
        )

    @staticmethod
    def _noop_urlopen():
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "[]"}}]}
                ).encode("utf-8")

        return lambda req, timeout=0: _Resp()

    def test_carver_accumulates_recent_turns_as_context(self, monkeypatch, tmp_path):
        """The carve prompt should contain prior conversational context,
        not just the current utterance in isolation."""
        mod = _import_converge()
        prompts_seen = []

        def capturing_urlopen(req, timeout=0):
            body = json.loads(req.data.decode("utf-8"))
            if "chat/completions" in req.full_url:
                user_msg = [m for m in body["messages"] if m["role"] == "user"][0]
                prompts_seen.append(user_msg["content"])

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return json.dumps(
                        {"choices": [{"message": {"content": "[]"}}]}
                    ).encode("utf-8")

            return _Resp()

        carver = self._make_carver(monkeypatch, tmp_path, mod, capturing_urlopen)
        long = "this message is definitely long enough to exceed the ten word minimum for carving to proceed"

        # Feed two turns so context accumulates, then force a carve
        carver.on_turn_complete("I really prefer dark mode themes for all my editors and tools", "noted")
        carver.on_turn_complete(long, "reply two")
        carver._drain_sync()

        assert len(prompts_seen) >= 1
        last_prompt = prompts_seen[-1]
        assert "dark mode" in last_prompt, (
            "Carve prompt should include prior conversational context"
        )

    def test_carver_fires_every_other_substantive_turn(self, monkeypatch, tmp_path):
        """Carving should fire on every 2nd substantive user message, not every one."""
        mod = _import_converge()
        carve_count = [0]

        def counting_urlopen(req, timeout=0):
            if "chat/completions" in req.full_url:
                carve_count[0] += 1

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return json.dumps(
                        {"choices": [{"message": {"content": "[]"}}]}
                    ).encode("utf-8")

            return _Resp()

        carver = self._make_carver(monkeypatch, tmp_path, mod, counting_urlopen)
        long = "this message is definitely long enough to exceed the ten word minimum for carving to proceed"

        # Feed 4 substantive turns, draining after each
        counts = []
        for i in range(4):
            carver.on_turn_complete(long, f"reply {i}")
            carver._drain_sync()
            counts.append(carve_count[0])

        # Should fire 2 times out of 4, not 4 times
        assert counts[-1] == 2, (
            f"Expected 2 carves out of 4 substantive turns, got {counts[-1]}. "
            f"Per-turn counts: {counts}"
        )

    def test_background_thread_creation_is_serialized(self, monkeypatch, tmp_path):
        """The background worker thread check and creation must happen under
        the lock so concurrent callers don't both start a thread."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        # The thread check+create is inside the lock block in on_turn_complete.
        # Verify structurally: after calling on_turn_complete, at most one
        # thread should be alive.
        import threading as _th
        long = "this message is long enough to exceed the ten word minimum for carving"
        carver.on_turn_complete(long, "reply one")
        carver.on_turn_complete(long + " two", "reply two")
        # Count daemon threads that target _background_loop
        bg_threads = [
            t for t in _th.enumerate()
            if t.daemon and t.is_alive() and getattr(t, "_target", None) is carver._background_loop
        ]
        assert len(bg_threads) <= 1, (
            f"Expected at most 1 background thread, found {len(bg_threads)}"
        )

    def test_context_buffer_is_bounded(self, monkeypatch, tmp_path):
        """The rolling context buffer should not grow unboundedly."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        for i in range(20):
            carver.on_turn_complete(
                f"user message number {i} with enough words to be substantive", f"reply {i}"
            )

        assert hasattr(carver, "_context_buffer"), (
            "TurnCarver should maintain a _context_buffer"
        )
        assert len(carver._context_buffer) <= 4, (
            f"Context buffer should be bounded to recent turns, got {len(carver._context_buffer)}"
        )

    def test_context_buffer_entries_are_user_assistant_pairs(self, monkeypatch, tmp_path):
        """Each entry in the context buffer should be a (user, assistant) pair."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        carver.on_turn_complete("hello world testing one two three", "reply one")
        carver.on_turn_complete("another message for the context buffer test", "reply two")

        assert len(carver._context_buffer) == 2
        entry = carver._context_buffer[0]
        assert isinstance(entry, dict), "Context buffer entries should be dicts"
        assert "user" in entry and "assistant" in entry, (
            f"Context buffer entries should have 'user' and 'assistant' keys, got {entry.keys()}"
        )

    def test_user_turns_are_never_truncated(self, monkeypatch, tmp_path):
        """User utterances must be stored in full, never truncated."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        long_user = "x" * 2000
        carver.on_turn_complete(long_user, "short reply")

        assert carver._context_buffer[0]["user"] == long_user, (
            "User turns must never be truncated in the context buffer"
        )

    def test_assistant_turns_get_middle_out_truncation(self, monkeypatch, tmp_path):
        """Long assistant turns should be truncated by cutting the middle,
        preserving head and tail."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        head = "HEAD_CONTENT " * 30  # ~390 chars
        middle = "MIDDLE_FILLER " * 100  # ~1400 chars
        tail = " TAIL_CONTENT" * 30  # ~390 chars
        long_assistant = head + middle + tail
        carver.on_turn_complete("some user input here for testing", long_assistant)

        stored = carver._context_buffer[0]["assistant"]
        assert len(stored) < len(long_assistant), "Long assistant should be truncated"
        assert "HEAD_CONTENT" in stored, "Head of assistant turn should be preserved"
        assert "TAIL_CONTENT" in stored, "Tail of assistant turn should be preserved"
        assert "[...]" in stored, "Middle-out truncation should leave a marker"

    def test_short_assistant_turns_are_not_truncated(self, monkeypatch, tmp_path):
        """Assistant turns under the threshold should not be truncated."""
        mod = _import_converge()
        carver = self._make_carver(
            monkeypatch, tmp_path, mod, self._noop_urlopen()
        )

        short_assistant = "Here is a short reply."
        carver.on_turn_complete("some user input for testing", short_assistant)

        assert carver._context_buffer[0]["assistant"] == short_assistant

    def test_debounce_override_fires_when_context_fully_stale(self, monkeypatch, tmp_path):
        """If continued debouncing would mean zero overlap with the last
        carve's context, the carver should fire regardless of cadence."""
        mod = _import_converge()
        carve_count = [0]

        def counting_urlopen(req, timeout=0):
            if "chat/completions" in req.full_url:
                carve_count[0] += 1

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return json.dumps(
                        {"choices": [{"message": {"content": "[]"}}]}
                    ).encode("utf-8")

            return _Resp()

        carver = self._make_carver(monkeypatch, tmp_path, mod, counting_urlopen)
        long = "this message is definitely long enough to exceed the ten word minimum for carving to proceed"

        # Turn 1 — not fired (cadence=2, first substantive turn)
        carver.on_turn_complete(long + " turn one", "reply 1")
        carver._drain_sync()
        # Turn 2 — fired (cadence hit)
        carver.on_turn_complete(long + " turn two", "reply 2")
        carver._drain_sync()
        count_after_cadence = carve_count[0]
        assert count_after_cadence == 1, f"Should have 1 carve after 2 turns, got {count_after_cadence}"

        # Now feed more turns through normal cadence, verifying counts.
        # Turn 3 — odd, no cadence
        carver.on_turn_complete(long + " turn three", "reply 3")
        carver._drain_sync()
        assert carve_count[0] == 1, f"Turn 3 (odd) should not carve, got {carve_count[0]}"
        # Turn 4 — even, cadence fires
        carver.on_turn_complete(long + " turn four", "reply 4")
        carver._drain_sync()
        assert carve_count[0] == 2, f"Turn 4 (even) should carve, got {carve_count[0]}"
        # Turn 5 — odd, no cadence
        carver.on_turn_complete(long + " turn five", "reply 5")
        carver._drain_sync()
        assert carve_count[0] == 2, f"Turn 5 (odd) should not carve, got {carve_count[0]}"
        # Turn 6 — even, cadence fires
        carver.on_turn_complete(long + " turn six", "reply 6")
        carver._drain_sync()
        assert carve_count[0] == 3, f"Turn 6 (even) should carve, got {carve_count[0]}"
        # Now skip many cadence hits by using short messages (< 10 words)
        # to rotate the buffer without triggering substantive-turn counting.
        for i in range(6):
            carver.on_turn_complete("short msg", f"reply {7+i}")
        # Buffer is now full of short-msg entries. None of these are substantive
        # so no cadence increments. But the context is fully stale relative to
        # the last carve.
        # Next substantive turn should fire via debounce override even though
        # it's an odd substantive count.
        count_before = carve_count[0]
        carver.on_turn_complete(long + " stale override", "reply 99")
        carver._drain_sync()
        assert carve_count[0] > count_before, (
            "Debounce override should force a carve when context is fully stale"
        )


class TestConvergeEmbedLib:
    def test_embed_model_path_can_come_from_env(self, monkeypatch):
        monkeypatch.setenv("SPOKE_CONVERGE_EMBED_MODEL_PATH", "/tmp/octen-model")
        mod = _load_converge_embed_lib()

        assert mod.resolve_model_path() == Path("/tmp/octen-model")
