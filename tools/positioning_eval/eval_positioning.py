"""Eval harness for VLM-driven overlay positioning.

Sends a grid-annotated screenshot to a VLM and asks it to flood-fill the
largest empty region, then extracts the largest inscribed rectangle from
the claimed cells.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from grid_overlay import (
    COLS,
    ROWS,
    ROW_LABELS,
    COL_LABELS,
    draw_grid,
)


GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"

def make_system_prompt(rows: int = ROWS, cols: int = COLS) -> str:
    last_row = ROW_LABELS[rows - 1]
    return f"""\
You are Spoke's window positioning system.

The screenshot has a grid overlay with yellow reference labels (A1, B2, etc.) \
drawn on top of it. The grid has rows A-{last_row} (top to bottom), columns \
1-{cols} (left to right). Ignore the yellow labels themselves — they are just \
reference markers. Look THROUGH them at the actual screen content underneath.

Your job: list every grid cell where the UNDERLYING screen content is empty, \
blank, or contains only non-essential UI chrome (menu bars, toolbars, status \
bars, margins). Skip any cell where the underlying content contains the text \
or material the user wants to keep visible.

Go row by row. For each row A through {last_row}, list every cell whose \
underlying content is empty or unimportant. Do not stop early.

Output ONLY cell names separated by spaces. No reasoning, no explanation."""


def encode_image(path: str | Path, scale: float = 1.0) -> str:
    """Encode image as base64 PNG, optionally downscaling by a factor."""
    import io
    img = Image.open(path)
    if scale < 1.0:
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_vlm(
    grid_image_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    scale: float = 1.0,
    grid_rows: int = ROWS,
    grid_cols: int = COLS,
) -> dict:
    """Send the grid image + constraint to the VLM, return raw response."""
    b64 = encode_image(grid_image_path, scale=scale)

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
        {
            "type": "text",
            "text": (
                f"Avoid blocking: {constraint}\n\n"
                "List every grid cell that does NOT contain that content. "
                "Start at the center of the largest empty region and spiral outward. "
                "Output only cell names separated by spaces."
            ),
        },
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": make_system_prompt(grid_rows, grid_cols)},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    headers = {
        "Authorization": "Bearer 1234",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"].strip()
    return {"raw": raw_text, "elapsed_s": round(elapsed, 1), "model": model}


def parse_cells(raw: str, rows: int = ROWS, cols: int = COLS) -> list[tuple[int, int]]:
    """Extract grid cell references and return as (row_idx, col_idx) pairs."""
    cells = re.findall(r"[A-Z]\d{1,2}", raw.upper())
    result = []
    seen = set()
    for cell in cells:
        row_letter = cell[0]
        try:
            col_num = int(cell[1:])
        except ValueError:
            continue
        if col_num < 1 or col_num > cols:
            continue
        row_idx = ord(row_letter) - ord("A")
        col_idx = col_num - 1
        if row_idx < 0 or row_idx >= rows:
            continue
        key = (row_idx, col_idx)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def largest_inscribed_rectangle(
    cells: list[tuple[int, int]], rows: int = ROWS, cols: int = COLS
) -> dict | None:
    """Find the largest axis-aligned rectangle inscribed in the given cells."""
    if not cells:
        return None

    cell_set = set(cells)

    grid = [[False] * cols for _ in range(rows)]
    for r, c in cell_set:
        grid[r][c] = True

    heights = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        for r in range(rows):
            if grid[r][c]:
                heights[r][c] = (heights[r - 1][c] + 1) if r > 0 else 1
            else:
                heights[r][c] = 0

    best_area = 0
    best_rect = None

    for r in range(rows):
        stack = []
        for c in range(cols + 1):
            h = heights[r][c] if c < cols else 0
            while stack and heights[r][stack[-1]] > h:
                height = heights[r][stack.pop()]
                width = c if not stack else c - stack[-1] - 1
                area = height * width
                if area > best_area:
                    best_area = area
                    left = stack[-1] + 1 if stack else 0
                    best_rect = {
                        "top_row": r - height + 1,
                        "bottom_row": r,
                        "left_col": left,
                        "right_col": left + width - 1,
                    }
            stack.append(c)

    if best_rect is None:
        return None

    return best_rect


def rect_to_pixels(rect: dict, img_w: int, img_h: int, rows: int = ROWS, cols: int = COLS) -> dict:
    """Convert grid-cell rectangle to pixel coordinates."""
    cell_w = img_w / cols
    cell_h = img_h / rows

    tr, br = rect["top_row"], rect["bottom_row"]
    lc, rc = rect["left_col"], rect["right_col"]

    width_cells = rc - lc + 1
    height_cells = br - tr + 1

    return {
        "x": int(lc * cell_w),
        "y": int(tr * cell_h),
        "width": int(width_cells * cell_w),
        "height": int(height_cells * cell_h),
        "area_cells": width_cells * height_cells,
        "area_fraction": (width_cells * height_cells) / (rows * cols),
        "grid_tl": f"{ROW_LABELS[tr]}{COL_LABELS[lc]}",
        "grid_br": f"{ROW_LABELS[br]}{COL_LABELS[rc]}",
    }


def score_result(rect: dict) -> dict:
    """Score a positioning result."""
    return {
        "area_cells": rect["area_cells"],
        "area_fraction": rect["area_fraction"],
        "area_pct": round(rect["area_fraction"] * 100, 1),
        "max_possible": ROWS * COLS,
        "hard_fail": False,
        "fail_reason": None,
    }


def visualize_result(
    grid_path: str,
    cells: list[tuple[int, int]],
    rect_px: dict,
    out_path: str,
) -> None:
    """Draw claimed cells and the inscribed rectangle on the grid image."""
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w = w / COLS
    cell_h = h / ROWS

    # Draw claimed cells in green
    for r, c in cells:
        x0 = int(c * cell_w)
        y0 = int(r * cell_h)
        x1 = int((c + 1) * cell_w)
        y1 = int((r + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 40))

    # Draw inscribed rectangle in blue
    x, y = rect_px["x"], rect_px["y"]
    rw, rh = rect_px["width"], rect_px["height"]
    draw.rectangle([x, y, x + rw, y + rh], fill=(0, 120, 255, 60), outline=(0, 120, 255, 200), width=3)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_eval(
    screenshot_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    output_dir: str = "/tmp/spoke-positioning-eval",
    scale: float = 1.0,
    grid_rows: int = ROWS,
    grid_cols: int = COLS,
) -> dict:
    """Run a single positioning eval."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(screenshot_path)
    img_w, img_h = img.size

    # Step 1: Draw grid
    grid_path = str(out_dir / "grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=grid_rows, cols=grid_cols)
    print(f"Grid overlay saved to {grid_path} ({grid_rows}x{grid_cols})")

    # Step 2: Query VLM
    print(f"Querying {model} with constraint: {constraint!r} (scale={scale})")
    vlm_result = query_vlm(grid_path, constraint, model=model, url=url, scale=scale, grid_rows=grid_rows, grid_cols=grid_cols)
    raw = vlm_result["raw"]
    print(f"  Response ({vlm_result['elapsed_s']}s): {raw[:200]}{'...' if len(raw) > 200 else ''}")

    # Step 3: Parse cells
    cells = parse_cells(raw, rows=grid_rows, cols=grid_cols)
    print(f"  Parsed {len(cells)} unique cells from response (grid {grid_rows}x{grid_cols})")

    if not cells:
        print(f"  PARSE FAIL: no valid grid cells found")
        return {"status": "parse_fail", "raw": raw}

    # Step 4: Find largest inscribed rectangle
    rect = largest_inscribed_rectangle(cells, rows=grid_rows, cols=grid_cols)
    if rect is None:
        print(f"  RECT FAIL: could not find inscribed rectangle")
        return {"status": "rect_fail", "cells": len(cells)}

    rect_px = rect_to_pixels(rect, img_w, img_h, rows=grid_rows, cols=grid_cols)
    print(f"  Inscribed rect: {rect_px['grid_tl']}—{rect_px['grid_br']} "
          f"({rect_px['area_fraction'] * 100:.1f}% of screen, {rect_px['area_cells']} cells)")

    # Step 5: Score
    score = score_result(rect_px)
    print(f"  Score: {score['area_pct']}% area")

    # Step 6: Visualize
    viz_path = str(out_dir / "result.png")
    visualize_result(grid_path, cells, rect_px, viz_path)
    print(f"  Visualization saved to {viz_path}")

    return {
        "status": "ok",
        "constraint": constraint,
        "cells_claimed": len(cells),
        "rect_grid": f"{rect_px['grid_tl']}—{rect_px['grid_br']}",
        "rect_px": rect_px,
        "score": score,
        "vlm": vlm_result,
        "viz_path": viz_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Eval VLM overlay positioning")
    parser.add_argument("screenshot", nargs="?", default="/tmp/spoke-positioning-eval-input.png")
    parser.add_argument("--constraint", "-c", default="the text content in the terminal")
    parser.add_argument("--model", "-m", default="qwen3.6-35b-a3b-oq8")
    parser.add_argument("--url", default=GRAPHEUS_URL)
    parser.add_argument("--output-dir", "-o", default="/tmp/spoke-positioning-eval")
    args = parser.parse_args()

    result = run_eval(
        args.screenshot,
        args.constraint,
        model=args.model,
        url=args.url,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
