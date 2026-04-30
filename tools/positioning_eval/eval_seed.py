"""Seed step eval: 5x5 coarse grid, model picks the single cell closest
to the center of the largest empty region.

This is step 1 of the recursive flood-fill positioning approach.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import draw_grid, ROW_LABELS, COL_LABELS


GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"

SEED_ROWS = 5
SEED_COLS = 5

SYSTEM_PROMPT = f"""\
You are Spoke's window positioning system.

The screenshot has a grid overlay with yellow reference labels drawn on \
top of it. The grid has rows A-{ROW_LABELS[SEED_ROWS - 1]} (top to bottom), \
columns 1-{SEED_COLS} (left to right). Ignore the yellow labels themselves — \
they are just reference markers. Look THROUGH them at the actual screen \
content underneath.

Your job: find the single grid cell that is closest to the CENTER of the \
largest empty region on the screen. Empty means: no text content, no \
application windows with visible content — just blank space, margins, \
or non-essential UI chrome.

Output ONLY one cell name (e.g. C4). Nothing else."""


def encode_image(path: str | Path, scale: float = 1.0) -> str:
    img = Image.open(path)
    if scale < 1.0:
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_seed(
    grid_image_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    scale: float = 1.0,
) -> dict:
    b64 = encode_image(grid_image_path, scale=scale)

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": (
            f"Avoid blocking: {constraint}\n\n"
            "Which single cell is closest to the center of the largest "
            "empty region? Output ONLY the cell name."
        )},
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    headers = {"Authorization": "Bearer 1234", "Content-Type": "application/json"}

    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()
    return {"raw": raw, "elapsed_s": round(elapsed, 1), "model": model}


def parse_single_cell(raw: str) -> tuple[int, int] | None:
    cells = re.findall(r"[A-Z]\d{1,2}", raw.upper())
    if not cells:
        return None
    cell = cells[0]
    row_idx = ord(cell[0]) - ord("A")
    try:
        col_idx = int(cell[1:]) - 1
    except ValueError:
        return None
    if 0 <= row_idx < SEED_ROWS and 0 <= col_idx < SEED_COLS:
        return (row_idx, col_idx)
    return None


def visualize_seed(grid_path: str, cell: tuple[int, int], out_path: str) -> None:
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w = w / SEED_COLS
    cell_h = h / SEED_ROWS

    r, c = cell
    x0, y0 = int(c * cell_w), int(r * cell_h)
    x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)

    draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 80), outline=(0, 255, 0, 255), width=4)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36)
    except (OSError, IOError):
        font = ImageFont.load_default()
    label = f"{ROW_LABELS[r]}{COL_LABELS[c]}"
    draw.text((x0 + 20, y0 + 20), f"SEED: {label}", fill=(0, 255, 0, 255), font=font)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_seed(
    screenshot_path: str,
    constraint: str,
    run_id: int = 0,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    output_dir: str = "/tmp/spoke-positioning-eval/seed",
    scale: float = 1.0,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = str(out_dir / "grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=SEED_ROWS, cols=SEED_COLS)

    vlm = query_seed(grid_path, constraint, model=model, url=url, scale=scale)
    raw = vlm["raw"]
    print(f"  Run {run_id}: response ({vlm['elapsed_s']}s): {raw}")

    cell = parse_single_cell(raw)
    if cell is None:
        print(f"  Run {run_id}: PARSE FAIL")
        return {"run": run_id, "status": "parse_fail", "raw": raw, "time": vlm["elapsed_s"]}

    r, c = cell
    label = f"{ROW_LABELS[r]}{COL_LABELS[c]}"
    print(f"  Run {run_id}: seed cell = {label}")

    viz_path = str(out_dir / f"result_run{run_id}.png")
    visualize_seed(grid_path, cell, viz_path)
    print(f"  Run {run_id}: viz saved to {viz_path}")

    return {
        "run": run_id,
        "status": "ok",
        "cell": label,
        "cell_rc": cell,
        "time": vlm["elapsed_s"],
        "raw": raw,
        "viz_path": viz_path,
    }


def main():
    import sys
    screenshot = sys.argv[1] if len(sys.argv) > 1 else "tools/positioning_eval/sample_screenshot.png"
    constraint = "the text content in the terminal — all the readable text on screen"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Running {n_runs} seed evals on 5x5 grid\n")

    results = []
    for i in range(n_runs):
        result = run_seed(screenshot, constraint, run_id=i + 1)
        results.append(result)
        if i < n_runs - 1:
            time.sleep(2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    cells = []
    for r in results:
        if r["status"] == "ok":
            print(f"  Run {r['run']}: {r['cell']}  ({r['time']}s)")
            cells.append(r["cell"])
        else:
            print(f"  Run {r['run']}: {r['status']} — {r.get('raw', '?')[:50]}")

    if cells:
        unique = set(cells)
        print(f"\nUnique cells: {unique}")
        print(f"Convergence: {len(cells) - len(unique) + 1}/{len(cells)} agree")
        if len(unique) == 1:
            print("Perfect convergence — all runs picked the same cell.")
        else:
            print(f"Variance: {len(unique)} different cells across {len(cells)} runs.")


if __name__ == "__main__":
    main()
