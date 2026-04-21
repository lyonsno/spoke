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


class TestConvergeEmbedLib:
    def test_embed_model_path_can_come_from_env(self, monkeypatch):
        monkeypatch.setenv("SPOKE_CONVERGE_EMBED_MODEL_PATH", "/tmp/octen-model")
        mod = _load_converge_embed_lib()

        assert mod.resolve_model_path() == Path("/tmp/octen-model")
