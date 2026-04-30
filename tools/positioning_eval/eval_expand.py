"""Expansion step: crop each neighbor of the seed cell, ask yes/no for content.

After the seed vote picks a cell, this crops each of its 8 neighbors
(from the original screenshot, not the grid-annotated version), sends
them in parallel, and gets back binary occupied/empty for each.
"""

from __future__ import annotations

import base64
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import ROW_LABELS, COL_LABELS

GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"

SEED_ROWS = 5
SEED_COLS = 5

EXPAND_SYSTEM = """\
You are checking whether a region of a screen contains content that \
should not be covered by an overlay.

You will see a cropped section of a screenshot. Answer with ONLY one word:

EMPTY — if this region contains no meaningful content (just blank space, \
margins, solid backgrounds, toolbars, menu bars, or UI chrome)

CONTENT — if this region contains text, images, graphs, data, or other \
material that a user would want to keep visible

One word only. EMPTY or CONTENT."""


def encode_image_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def crop_cell(img: Image.Image, row: int, col: int, rows: int = SEED_ROWS, cols: int = SEED_COLS) -> Image.Image:
    """Crop a single grid cell from the image."""
    w, h = img.size
    cell_w, cell_h = w / cols, h / rows
    x0 = int(col * cell_w)
    y0 = int(row * cell_h)
    x1 = int((col + 1) * cell_w)
    y1 = int((row + 1) * cell_h)
    return img.crop((x0, y0, x1, y1))


def query_cell(
    crop_img: Image.Image,
    constraint: str,
    url: str = GRAPHEUS_URL,
) -> tuple[str, float]:
    """Ask the model if a cropped cell contains content. Returns ('EMPTY'|'CONTENT', elapsed)."""
    b64 = encode_image_pil(crop_img)

    t0 = time.time()
    resp = requests.post(
        url,
        headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
        json={
            "model": "qwen3.6-35b-a3b-oq8",
            "messages": [
                {"role": "system", "content": EXPAND_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": (
                        f"Does this region contain: {constraint}\n"
                        "Answer EMPTY or CONTENT."
                    )},
                ]},
            ],
            "temperature": 0.1, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip().upper()

    # Parse — look for EMPTY or CONTENT
    if "EMPTY" in raw:
        return "EMPTY", round(elapsed, 3)
    elif "CONTENT" in raw:
        return "CONTENT", round(elapsed, 3)
    else:
        return f"UNKNOWN({raw[:20]})", round(elapsed, 3)


def get_neighbors(row: int, col: int, rows: int = SEED_ROWS, cols: int = SEED_COLS) -> list[tuple[int, int]]:
    """Get all valid neighbor cells (8-connected)."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


def visualize_expansion(
    screenshot_path: str,
    seed: tuple[int, int],
    results: dict[tuple[int, int], str],
    out_path: str,
) -> None:
    """Draw expansion results: seed in cyan, empty neighbors in green, content in red."""
    img = Image.open(screenshot_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w, cell_h = w / SEED_COLS, h / SEED_ROWS

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
    except (OSError, IOError):
        font = small_font = ImageFont.load_default()

    # Draw seed cell
    sr, sc = seed
    x0, y0 = int(sc * cell_w), int(sr * cell_h)
    x1, y1 = int((sc + 1) * cell_w), int((sr + 1) * cell_h)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, 100), outline=(0, 255, 255, 255), width=4)
    label = f"{ROW_LABELS[sr]}{COL_LABELS[sc]}"
    draw.text((x0 + 10, y0 + 10), f"SEED", fill=(0, 255, 255, 255), font=small_font)

    # Draw neighbor results
    for (nr, nc), verdict in results.items():
        x0, y0 = int(nc * cell_w), int(nr * cell_h)
        x1, y1 = int((nc + 1) * cell_w), int((nr + 1) * cell_h)
        label = f"{ROW_LABELS[nr]}{COL_LABELS[nc]}"

        if verdict == "EMPTY":
            draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 80), outline=(0, 255, 0, 200), width=3)
            draw.text((x0 + 10, y0 + 10), "EMPTY", fill=(0, 255, 0, 255), font=small_font)
        elif verdict == "CONTENT":
            draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 60), outline=(255, 0, 0, 200), width=3)
            draw.text((x0 + 10, y0 + 10), "CONTENT", fill=(255, 100, 100, 255), font=small_font)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 0, 60), outline=(200, 200, 0, 200), width=2)
            draw.text((x0 + 10, y0 + 10), "???", fill=(255, 255, 0, 255), font=small_font)

    # Legend
    draw.rectangle([10, h - 60, 700, h - 10], fill=(0, 0, 0, 180))
    empty_count = sum(1 for v in results.values() if v == "EMPTY")
    content_count = sum(1 for v in results.values() if v == "CONTENT")
    draw.text((20, h - 50),
              f"Seed + {empty_count} EMPTY + {content_count} CONTENT = {1 + empty_count} cells available",
              fill=(255, 255, 255, 220), font=small_font)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_expand(
    screenshot_path: str,
    seed_cell: str,
    constraint: str,
    output_dir: str = "/tmp/spoke-positioning-eval/expand",
    parallel: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(screenshot_path)
    seed_row = ord(seed_cell[0].upper()) - ord("A")
    seed_col = int(seed_cell[1:]) - 1

    neighbors = get_neighbors(seed_row, seed_col)
    print(f"Seed: {seed_cell} ({seed_row},{seed_col}), {len(neighbors)} neighbors")

    # Crop all neighbors
    crops = {}
    for nr, nc in neighbors:
        crops[(nr, nc)] = crop_cell(img, nr, nc)

    # Query all neighbors
    results = {}
    t0 = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=len(neighbors)) as pool:
            futures = {}
            for (nr, nc), crop in crops.items():
                f = pool.submit(query_cell, crop, constraint)
                futures[f] = (nr, nc)

            for f in as_completed(futures):
                nr, nc = futures[f]
                verdict, elapsed = f.result()
                label = f"{ROW_LABELS[nr]}{COL_LABELS[nc]}"
                results[(nr, nc)] = verdict
                print(f"  {label}: {verdict} ({elapsed}s)")
    else:
        for (nr, nc), crop in crops.items():
            verdict, elapsed = query_cell(crop, constraint)
            label = f"{ROW_LABELS[nr]}{COL_LABELS[nc]}"
            results[(nr, nc)] = verdict
            print(f"  {label}: {verdict} ({elapsed}s)")

    total_elapsed = round(time.time() - t0, 2)
    empty_cells = [(nr, nc) for (nr, nc), v in results.items() if v == "EMPTY"]
    content_cells = [(nr, nc) for (nr, nc), v in results.items() if v == "CONTENT"]

    print(f"\n  Total expansion time: {total_elapsed}s ({'parallel' if parallel else 'sequential'})")
    print(f"  Empty: {len(empty_cells)}, Content: {len(content_cells)}")
    print(f"  Available area: {1 + len(empty_cells)} cells (seed + empty neighbors)")

    # Visualize
    viz_path = str(out_dir / "expansion.png")
    visualize_expansion(screenshot_path, (seed_row, seed_col), results, viz_path)
    print(f"  Visualization: {viz_path}")

    return {
        "seed": seed_cell,
        "neighbors": len(neighbors),
        "empty": [f"{ROW_LABELS[r]}{COL_LABELS[c]}" for r, c in empty_cells],
        "content": [f"{ROW_LABELS[r]}{COL_LABELS[c]}" for r, c in content_cells],
        "total_time": total_elapsed,
        "parallel": parallel,
        "viz_path": viz_path,
    }


if __name__ == "__main__":
    import json
    import subprocess
    import sys

    screenshot = sys.argv[1] if len(sys.argv) > 1 else "tools/positioning_eval/sample_screenshot.png"
    seed = sys.argv[2] if len(sys.argv) > 2 else "B4"
    constraint = sys.argv[3] if len(sys.argv) > 3 else "the text content in the terminal — all the readable text on screen"

    result = run_expand(screenshot, seed, constraint)
    print(json.dumps(result, indent=2))
    subprocess.run(["open", result["viz_path"]])
