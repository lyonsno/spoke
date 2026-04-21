#!/usr/bin/env python3
"""Converge embedding engine — build and query the attractor vector index.

Uses Octen-Embedding-8B-mlx (bf16) for semantic embedding of attractors and
conversation turns. Single forward pass (prefill only), last-token pooling,
L2 normalization.

Commands:
    uv run scripts/converge-embed.py build       # build/rebuild the attractor index
    uv run scripts/converge-embed.py query "text" # find matching attractors for text
    uv run scripts/converge-embed.py info        # show index stats

The index is stored at ~/.config/spoke/attractor-index.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_INDEX_PATH = Path.home() / ".config" / "spoke" / "attractor-index.npz"
_ATTRACTOR_SOURCES = [
    ("project", Path.home() / "dev" / "epistaxis" / "attractors"),
    ("personal", Path.home() / ".config" / "spoke" / "attractors"),
]


def embed_texts(
    texts: list[str],
    *,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Embed a list of texts via the shared library."""
    from converge_embed_lib import embed_texts as _embed
    return _embed(texts, model_path=model_path)


def _load_attractors() -> list[dict]:
    """Load all attractors with full text and summaries."""
    entries = []
    for source_label, adir in _ATTRACTOR_SOURCES:
        if not adir.is_dir():
            continue
        for f in sorted(adir.iterdir()):
            if f.is_file() and f.suffix == ".md":
                try:
                    full_text = f.read_text(encoding="utf-8")
                except OSError:
                    continue
                # Extract first meaningful line as summary
                summary = ""
                for line in full_text.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    import re
                    summary = re.sub(
                        r"^-\s*(?:Stimulus|Evidence|Source|Satisfaction|Strength|Observed):\s*",
                        "", line
                    )[:200]
                    break

                entries.append({
                    "source": source_label,
                    "slug": f.stem,
                    "full_text": full_text,
                    "summary": summary,
                })
    return entries


def build_index(model_path: str | Path | None = None):
    """Build the attractor vector index."""
    entries = _load_attractors()
    if not entries:
        print("No attractors found.", file=sys.stderr)
        return

    print(f"Embedding {len(entries)} attractors (full text + summary)...", file=sys.stderr)

    # Embed full texts
    full_texts = [e["full_text"] for e in entries]
    t0 = time.time()
    full_embeddings = embed_texts(full_texts, model_path=model_path)
    t_full = time.time() - t0
    print(f"  Full text embeddings: {t_full:.1f}s ({t_full/len(entries)*1000:.0f}ms/attractor)",
          file=sys.stderr)

    # Embed summaries
    summaries = [e["summary"] for e in entries]
    t0 = time.time()
    summary_embeddings = embed_texts(summaries, model_path=model_path)
    t_summ = time.time() - t0
    print(f"  Summary embeddings: {t_summ:.1f}s ({t_summ/len(entries)*1000:.0f}ms/attractor)",
          file=sys.stderr)

    # Build metadata
    metadata = [{
        "source": e["source"],
        "slug": e["slug"],
        "summary": e["summary"],
    } for e in entries]

    # Save index
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _INDEX_PATH,
        full_embeddings=full_embeddings,
        summary_embeddings=summary_embeddings,
        metadata=json.dumps(metadata),
    )
    print(f"\nIndex saved: {_INDEX_PATH}", file=sys.stderr)
    print(f"  {len(entries)} attractors, {full_embeddings.shape[1]}-dim embeddings", file=sys.stderr)
    print(f"  File size: {_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB", file=sys.stderr)


def load_index() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    """Load the index from disk. Returns (full_emb, summary_emb, metadata) or None."""
    if not _INDEX_PATH.exists():
        return None
    data = np.load(_INDEX_PATH, allow_pickle=False)
    full_emb = data["full_embeddings"]
    summary_emb = data["summary_embeddings"]
    metadata = json.loads(str(data["metadata"]))
    return full_emb, summary_emb, metadata


def query(
    text: str,
    top_k: int = 10,
    threshold: float = 0.3,
    *,
    model_path: str | Path | None = None,
) -> list[dict]:
    """Query the index with a text, return top-k matching attractors above threshold."""
    index = load_index()
    if index is None:
        print("No index found. Run 'build' first.", file=sys.stderr)
        return []

    full_emb, summary_emb, metadata = index

    # Embed the query
    query_emb = embed_texts([text], model_path=model_path)[0]  # (dim,)

    # Full-text embeddings only (summary embeddings have degenerate entries)
    full_scores = full_emb @ query_emb

    # Top-k above threshold
    top_indices = np.argsort(full_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(full_scores[idx])
        if score < threshold:
            break
        results.append({
            **metadata[idx],
            "score": score,
        })

    return results


def query_turns(
    turns: list[str],
    top_k: int = 10,
    threshold: float = 0.3,
    *,
    model_path: str | Path | None = None,
) -> list[dict]:
    """Query with multiple conversation turns, return union of matches."""
    index = load_index()
    if index is None:
        return []

    full_emb, summary_emb, metadata = index

    # Embed all turns at once
    if not turns:
        return []
    turn_embeddings = embed_texts(turns, model_path=model_path)  # (N, dim)

    # Full-text embeddings only (summary embeddings have degenerate entries)
    full_scores = full_emb @ turn_embeddings.T  # (attractors, turns)
    best_full = full_scores.max(axis=1)
    combined = best_full

    # Top-k above threshold
    top_indices = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(combined[idx])
        if score < threshold:
            break
        results.append({
            **metadata[idx],
            "score": round(score, 4),
        })

    return results


def show_info():
    """Show index stats."""
    index = load_index()
    if index is None:
        print("No index found. Run 'build' first.")
        return

    full_emb, summary_emb, metadata = index
    sources = {}
    for m in metadata:
        sources[m["source"]] = sources.get(m["source"], 0) + 1

    print(f"Index: {_INDEX_PATH}")
    print(f"  Attractors: {len(metadata)}")
    print(f"  Dimensions: {full_emb.shape[1]}")
    print(f"  Sources: {sources}")
    print(f"  File size: {_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Converge embedding engine")
    parser.add_argument("command", choices=["build", "query", "info"],
                        help="build: rebuild index, query: search, info: show stats")
    parser.add_argument("text", nargs="?", help="Query text (for 'query' command)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--model-path",
        help=(
            "Optional path to the Octen embedding model. Defaults to "
            "$SPOKE_CONVERGE_EMBED_MODEL_PATH or the repo-local default."
        ),
    )
    args = parser.parse_args()

    if args.command == "build":
        build_index(model_path=args.model_path)
    elif args.command == "query":
        if not args.text:
            print("Usage: converge-embed.py query \"text to search\"", file=sys.stderr)
            sys.exit(1)
        results = query(
            args.text,
            top_k=args.top_k,
            threshold=args.threshold,
            model_path=args.model_path,
        )
        if results:
            for r in results:
                print(f"  {r['score']:.4f}  [{r['source']}] {r['slug']}")
                if r.get("summary"):
                    print(f"           {r['summary'][:80]}")
        else:
            print("No matches above threshold.")
    elif args.command == "info":
        show_info()


if __name__ == "__main__":
    main()
