"""Inverted grid eval — ask the model to mark OCCUPIED cells, then find
the largest rectangle in the complement.

The hypothesis: marking what's full is a simpler perceptual task than
marking what's empty, and the rectangle extraction is deterministic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from grid_overlay import draw_grid, ROWS, COLS, ROW_LABELS, COL_LABELS
from eval_positioning import (
    encode_image,
    parse_cells,
    largest_inscribed_rectangle,
    rect_to_pixels,
    score_result,
    visualize_result,
    GRAPHEUS_URL,
)

import requests


def make_inverted_prompt(rows: int, cols: int) -> str:
    last_row = ROW_LABELS[rows - 1]
    return f"""\
You are Spoke's window positioning system.

The screenshot has a grid overlay with yellow reference labels drawn on \
top of it. The grid has rows A-{last_row} (top to bottom), columns \
1-{cols} (left to right). Ignore the yellow labels themselves — they are \
just reference markers. Look THROUGH them at the actual screen content \
underneath.

Your job: list every grid cell that CONTAINS the content the user wants \
to keep visible. Only list cells where the underlying screen has actual \
text or material content — not empty space, margins, toolbars, or UI chrome.

Go row by row. For each row A through {last_row}, list every cell that \
contains the specified content. Do not stop early.

Output ONLY cell names separated by spaces. No reasoning, no explanation."""


def query_inverted(
    grid_image_path: str,
    constraint: str,
    rows: int,
    cols: int,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    scale: float = 1.0,
) -> dict:
    import base64
    b64 = encode_image(grid_image_path, scale=scale)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": make_inverted_prompt(rows, cols)},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": (
                    f"Mark every cell that contains: {constraint}\n\n"
                    "List ONLY the cells that have that content. "
                    "Output cell names separated by spaces."
                )},
            ]},
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


def run_eval(
    screenshot_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    output_dir: str = "/tmp/spoke-positioning-eval/inverted",
    scale: float = 1.0,
    grid_rows: int = 9,
    grid_cols: int = 10,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(screenshot_path)
    img_w, img_h = img.size

    # Step 1: Draw grid
    grid_path = str(out_dir / "grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=grid_rows, cols=grid_cols)
    print(f"Grid overlay saved to {grid_path} ({grid_rows}x{grid_cols})")

    # Step 2: Query VLM for OCCUPIED cells
    print(f"Querying {model} for OCCUPIED cells (scale={scale})")
    vlm_result = query_inverted(grid_path, constraint, grid_rows, grid_cols,
                                 model=model, url=url, scale=scale)
    raw = vlm_result["raw"]
    print(f"  Response ({vlm_result['elapsed_s']}s): {raw[:200]}{'...' if len(raw) > 200 else ''}")

    # Step 3: Parse occupied cells
    occupied = parse_cells(raw, rows=grid_rows, cols=grid_cols)
    print(f"  Parsed {len(occupied)} occupied cells")

    total_cells = grid_rows * grid_cols

    # Step 4: Invert — find empty cells
    occupied_set = set(occupied)
    empty_cells = [
        (r, c) for r in range(grid_rows) for c in range(grid_cols)
        if (r, c) not in occupied_set
    ]
    print(f"  Empty cells (complement): {len(empty_cells)} / {total_cells}")

    if not empty_cells:
        print(f"  No empty cells — model thinks everything is occupied")
        return {"status": "no_empty", "occupied": len(occupied), "vlm": vlm_result}

    # Step 5: Find largest inscribed rectangle in the empty cells
    rect = largest_inscribed_rectangle(empty_cells, rows=grid_rows, cols=grid_cols)
    if rect is None:
        print(f"  RECT FAIL")
        return {"status": "rect_fail", "occupied": len(occupied), "empty": len(empty_cells)}

    rect_px = rect_to_pixels(rect, img_w, img_h, rows=grid_rows, cols=grid_cols)
    print(f"  Inscribed rect: {rect_px['grid_tl']}—{rect_px['grid_br']} "
          f"({rect_px['area_fraction'] * 100:.1f}% of screen)")

    # Step 6: Score
    score = score_result(rect_px)
    print(f"  Score: {score['area_pct']}% area")

    # Step 7: Visualize — show occupied in red, empty in green, rect in blue
    from PIL import ImageDraw as ID
    grid_img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", grid_img.size, (0, 0, 0, 0))
    draw = ID.Draw(overlay)

    cell_w = grid_img.width / grid_cols
    cell_h = grid_img.height / grid_rows

    # Red for occupied
    for r, c in occupied:
        x0, y0 = int(c * cell_w), int(r * cell_h)
        x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 50))

    # Green for empty
    for r, c in empty_cells:
        x0, y0 = int(c * cell_w), int(r * cell_h)
        x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 40))

    # Blue for inscribed rect
    x, y = rect_px["x"], rect_px["y"]
    w, h = rect_px["width"], rect_px["height"]
    draw.rectangle([x, y, x + w, y + h], outline=(0, 120, 255, 200), width=3)

    result_img = Image.alpha_composite(grid_img, overlay)
    viz_path = str(out_dir / "result.png")
    result_img.save(viz_path)
    print(f"  Visualization saved to {viz_path}")

    return {
        "status": "ok",
        "constraint": constraint,
        "occupied_cells": len(occupied),
        "empty_cells": len(empty_cells),
        "rect_grid": f"{rect_px['grid_tl']}—{rect_px['grid_br']}",
        "rect_px": rect_px,
        "score": score,
        "vlm": vlm_result,
        "viz_path": viz_path,
    }


if __name__ == "__main__":
    import sys
    screenshot = sys.argv[1] if len(sys.argv) > 1 else "tools/positioning_eval/sample_screenshot.png"
    constraint = "the text content in the terminal — all the readable text on screen"

    result = run_eval(screenshot, constraint)
    print(json.dumps({k: v for k, v in result.items() if k != "vlm"}, indent=2, default=str))
    if result.get("vlm"):
        print(f"Time: {result['vlm']['elapsed_s']}s")
