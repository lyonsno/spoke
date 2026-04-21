"""Converge embedding library — importable embedding functions.

This module provides the embed_texts() function for use by both the CLI
script (converge-embed.py) and the in-app guided compaction mode.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_MODEL_PATH = (
    Path.home() / "dev" / "scripts" / "quant" / "models" / "Octen-Embedding-8B-mlx"
)

# Lazy-loaded model state
_model = None
_tokenizer = None
_loaded_model_path: Path | None = None


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the embedding-model path from explicit arg, env, or default."""
    candidate = model_path or os.environ.get("SPOKE_CONVERGE_EMBED_MODEL_PATH")
    if candidate:
        return Path(candidate).expanduser()
    return _DEFAULT_MODEL_PATH


def _load_model(model_path: str | Path | None = None):
    """Load the embedding model (lazy, cached in-process)."""
    global _model, _tokenizer, _loaded_model_path
    resolved_path = resolve_model_path(model_path)
    if _model is not None and _loaded_model_path == resolved_path:
        return _model, _tokenizer

    from mlx_lm import load
    print(f"Loading embedding model from {resolved_path}...", file=sys.stderr)
    t0 = time.time()
    _model, _tokenizer = load(str(resolved_path))
    _loaded_model_path = resolved_path
    print(f"Embedding model loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return _model, _tokenizer


def embed_texts(
    texts: list[str],
    *,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Embed a list of texts, returning (N, dim) float32 numpy array.

    Uses last-token pooling with L2 normalization, matching Octen's
    expected inference protocol.
    """
    import mlx.core as mx

    model, tokenizer = _load_model(model_path=model_path)

    all_embeddings = []
    for text in texts:
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])

        # Get hidden states from the base model (before LM head)
        last_hidden = model.model(input_ids)
        # Last-token pooling
        embedding = last_hidden[:, -1, :]  # (1, hidden_dim)

        # L2 normalize
        norm = mx.sqrt(mx.sum(embedding * embedding, axis=-1, keepdims=True))
        embedding = embedding / mx.maximum(norm, mx.array(1e-12))

        # Eval and convert to numpy
        mx.eval(embedding)
        all_embeddings.append(np.array(embedding[0], dtype=np.float32))

    return np.stack(all_embeddings)
