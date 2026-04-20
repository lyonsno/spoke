"""Converge embedding library — importable embedding functions.

This module provides the embed_texts() function for use by both the CLI
script (converge-embed.py) and the in-app guided compaction mode.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_MODEL_PATH = Path.home() / "dev" / "scripts" / "quant" / "models" / "Octen-Embedding-8B-mlx"

# Lazy-loaded model state
_model = None
_tokenizer = None


def _load_model():
    """Load the embedding model (lazy, cached in-process)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from mlx_lm import load
    print(f"Loading embedding model from {_MODEL_PATH}...", file=sys.stderr)
    t0 = time.time()
    _model, _tokenizer = load(str(_MODEL_PATH))
    print(f"Embedding model loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    return _model, _tokenizer


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning (N, dim) float32 numpy array.

    Uses last-token pooling with L2 normalization, matching Octen's
    expected inference protocol.
    """
    import mlx.core as mx

    model, tokenizer = _load_model()

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
