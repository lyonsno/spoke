"""Seed step with majority vote: 5x top-3 at 50% scale, heatmap output.

Exploits fast warm inference (~0.2s) to run multiple passes and vote
on the best seed cell. Total time ~1.5-3s.
"""

from __future__ import annotations

import base64
import io
import re
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import draw_grid, ROW_LABELS, COL_LABELS

GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"
SEED_ROWS = 5
SEED_COLS = 5
N_RUNS = 5
TOP_K = 3
SCALE = 0.5

SYSTEM_PROMPT = f"""\
You are Spoke's window positioning system.

The screenshot has a grid overlay with yellow reference labels drawn on \
top of it. The grid has rows A-{ROW_LABELS[SEED_ROWS - 1]} (top to bottom), \
columns 1-{SEED_COLS} (left to right). Ignore the yellow labels themselves — \
they are just reference markers. Look THROUGH them at the actual screen \
content underneath.

Your job: find the THREE grid cells that are closest to the center of the \
largest empty regions on the screen. Empty means: no text content, no \
application windows with visible content — just blank space, margins, \
or non-essential UI chrome.

Output ONLY three cell names separated by spaces, ranked from most empty \
to least empty. Nothing else."""


def encode_image(path: str | Path, scale: float = 1.0) -> str:
    img = Image.open(path)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_top3(grid_image_path: str, constraint: str, scale: float = SCALE) -> tuple[list[str], float]:
    b64 = encode_image(grid_image_path, scale=scale)
    t0 = time.time()
    resp = requests.post(
        GRAPHEUS_URL,
        headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
        json={
            "model": "qwen3.6-35b-a3b-oq8",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": (
                        f"Avoid blocking: {constraint}\n"
                        "List your top 3 emptiest cells."
                    )},
                ]},
            ],
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16384,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    pattern = f"[A-{ROW_LABELS[SEED_ROWS - 1]}][1-{SEED_COLS}]"
    cells = re.findall(pattern, raw.upper())
    return cells[:TOP_K], round(elapsed, 3)


def draw_heatmap(grid_path: str, counts: dict[str, int], winner: str, out_path: str) -> None:
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w, cell_h = w / SEED_COLS, h / SEED_ROWS
    max_count = max(counts.values()) if counts else 1

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
        small_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
    except (OSError, IOError):
        font = small_font = ImageFont.load_default()

    for cell_str, count in counts.items():
        row_idx = ord(cell_str[0]) - ord("A")
        col_idx = int(cell_str[1:]) - 1
        if not (0 <= row_idx < SEED_ROWS and 0 <= col_idx < SEED_COLS):
            continue

        x0, y0 = int(col_idx * cell_w), int(row_idx * cell_h)
        x1, y1 = int((col_idx + 1) * cell_w), int((row_idx + 1) * cell_h)

        intensity = count / max_count
        alpha = int(40 + intensity * 120)

        if cell_str == winner:
            draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, alpha))
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 255, 255), width=4)
        else:
            r_val = int(255 * intensity)
            g_val = int(140 * (1 - intensity))
            draw.rectangle([x0, y0, x1, y1], fill=(r_val, g_val, 0, alpha))
            draw.rectangle([x0, y0, x1, y1], outline=(r_val, g_val, 0, 180), width=2)

        label = str(count)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + (x1 - x0 - tw) // 2
        ty = y0 + (y1 - y0 - th) // 2
        text_color = (0, 255, 255, 255) if cell_str == winner else (255, 255, 255, 220)
        draw.text((tx, ty), label, fill=text_color, font=font)

    draw.rectangle([10, h - 90, 600, h - 10], fill=(0, 0, 0, 180))
    draw.text((20, h - 80), f"Winner: {winner} ({counts[winner]} votes)", fill=(0, 255, 255, 255), font=small_font)
    draw.text((20, h - 50), "Orange = other picks, brighter = more votes", fill=(255, 200, 100, 200), font=small_font)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_vote(
    screenshot_path: str,
    constraint: str,
    output_dir: str = "/tmp/spoke-positioning-eval/vote",
    scale: float = SCALE,
    n_runs: int = N_RUNS,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = str(out_dir / "grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=SEED_ROWS, cols=SEED_COLS)

    all_picks = []
    run_details = []
    total_time = 0.0

    for i in range(n_runs):
        cells, elapsed = query_top3(grid_path, constraint, scale=scale)
        total_time += elapsed
        all_picks.extend(cells)
        run_details.append({"run": i + 1, "cells": cells, "time": elapsed})
        print(f"  Run {i + 1}: {' '.join(cells):15s} ({elapsed}s)")

    counts = dict(Counter(all_picks))
    ranked = Counter(all_picks).most_common()
    winner = ranked[0][0]

    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  All picks: {counts}")
    print(f"  Winner: {winner} ({ranked[0][1]}/{len(all_picks)} votes)")

    heatmap_path = str(out_dir / "heatmap.png")
    draw_heatmap(grid_path, counts, winner, heatmap_path)
    print(f"  Heatmap: {heatmap_path}")

    return {
        "winner": winner,
        "winner_votes": ranked[0][1],
        "total_votes": len(all_picks),
        "counts": counts,
        "total_time": round(total_time, 2),
        "runs": run_details,
        "heatmap_path": heatmap_path,
    }


if __name__ == "__main__":
    import sys
    screenshot = sys.argv[1] if len(sys.argv) > 1 else "tools/positioning_eval/sample_screenshot.png"
    constraint = sys.argv[2] if len(sys.argv) > 2 else "the text content in the terminal"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/spoke-positioning-eval/vote"

    result = run_vote(screenshot, constraint, output_dir=output_dir)
    import json
    print(json.dumps(result, indent=2, default=str))
