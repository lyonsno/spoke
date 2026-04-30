"""Draw an A-Z x 1-30 grid overlay on a screenshot for VLM positioning eval."""

from __future__ import annotations

import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Grid dimensions
ROWS = 26  # A-Z
COLS = 30  # 1-30

ROW_LABELS = list(string.ascii_uppercase)
COL_LABELS = [str(i) for i in range(1, COLS + 1)]

# Visual params
GRID_COLOR = (255, 255, 0, 80)  # yellow, semi-transparent
LABEL_COLOR = (255, 255, 0, 200)
LABEL_BG = (0, 0, 0, 140)

# Flood-fill style: big labels filling each cell, semi-transparent
FLOOD_LABEL_COLOR = (255, 255, 0)  # yellow, alpha applied separately
FLOOD_LABEL_ALPHA = 140  # ~55% opacity — must survive scale-down while disrupting text


def draw_grid(
    img_path: str | Path,
    out_path: str | Path,
    style: str = "sparse",
    rows: int = ROWS,
    cols: int = COLS,
) -> tuple[int, int, int, int]:
    """Draw grid on image, return (cell_width, cell_height, rows, cols).

    style: "sparse" = old style (labels at edges + every Nth cell)
           "flood"  = every cell labeled, big chunky text, semi-transparent
    """
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows

    row_labels = ROW_LABELS[:rows]
    col_labels = COL_LABELS[:cols]

    # Draw grid lines
    for c in range(cols + 1):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, h)], fill=GRID_COLOR, width=1)
    for r in range(rows + 1):
        y = int(r * cell_h)
        draw.line([(0, y), (w, y)], fill=GRID_COLOR, width=1)

    if style == "flood":
        _draw_flood_labels(draw, w, h, cell_w, cell_h, row_labels, col_labels, rows, cols)
    else:
        _draw_sparse_labels(draw, cell_w, cell_h, row_labels, col_labels, rows, cols)

    result = Image.alpha_composite(img, overlay)
    result.save(str(out_path))
    return int(cell_w), int(cell_h), rows, cols


def _draw_sparse_labels(draw, cell_w, cell_h, row_labels, col_labels, rows, cols):
    """Original sparse labeling: edges + every Nth cell."""
    try:
        font_size = max(10, min(int(cell_h * 0.35), int(cell_w * 0.4)))
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for row in range(rows):
        for col in range(cols):
            if row == 0 or col == 0 or (row % 3 == 0 and col % 5 == 0):
                label = f"{row_labels[row]}{col_labels[col]}"
                x = int(col * cell_w) + 2
                y = int(row * cell_h) + 1
                bbox = draw.textbbox((x, y), label, font=font)
                draw.rectangle(bbox, fill=LABEL_BG)
                draw.text((x, y), label, fill=LABEL_COLOR, font=font)


def _draw_flood_labels(draw, w, h, cell_w, cell_h, row_labels, col_labels, rows, cols):
    """Every cell gets a big, chunky, semi-transparent label filling it."""
    # Target: labels must be readable at 33% scale
    font_size = max(16, int(min(cell_h * 0.75, cell_w * 0.45)))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

    label_fill = (*FLOOD_LABEL_COLOR, FLOOD_LABEL_ALPHA)

    for row in range(rows):
        for col in range(cols):
            label = f"{row_labels[row]}{col_labels[col]}"
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            tx = cx - tw // 2
            ty = cy - th // 2

            draw.text((tx, ty), label, fill=label_fill, font=font)


def grid_to_pixels(
    cell: str, img_width: int, img_height: int
) -> tuple[int, int]:
    """Convert a grid cell label like 'B5' to pixel center coordinates."""
    row_letter = cell[0].upper()
    col_num = int(cell[1:])

    row_idx = ord(row_letter) - ord("A")
    col_idx = col_num - 1

    cell_w = img_width / COLS
    cell_h = img_height / ROWS

    px = int((col_idx + 0.5) * cell_w)
    py = int((row_idx + 0.5) * cell_h)
    return px, py


def corners_to_rect(
    tl: str, tr: str, bl: str, br: str, img_width: int, img_height: int
) -> dict | str:
    """Convert four corner grid cells to a pixel rect. Returns error string if invalid."""
    try:
        tl_r, tl_c = ord(tl[0].upper()) - ord("A"), int(tl[1:]) - 1
        tr_r, tr_c = ord(tr[0].upper()) - ord("A"), int(tr[1:]) - 1
        bl_r, bl_c = ord(bl[0].upper()) - ord("A"), int(bl[1:]) - 1
        br_r, br_c = ord(br[0].upper()) - ord("A"), int(br[1:]) - 1
    except (IndexError, ValueError) as e:
        return f"parse error: {e}"

    if tl_r != tr_r:
        return f"top row mismatch: {tl} row {tl_r} vs {tr} row {tr_r}"
    if bl_r != br_r:
        return f"bottom row mismatch: {bl} row {bl_r} vs {br} row {br_r}"
    if tl_c != bl_c:
        return f"left col mismatch: {tl} col {tl_c} vs {bl} col {bl_c}"
    if tr_c != br_c:
        return f"right col mismatch: {tr} col {tr_c} vs {br} col {br_c}"
    if tl_r >= bl_r:
        return f"top below bottom: row {tl_r} >= {bl_r}"
    if tl_c >= tr_c:
        return f"left right of right: col {tl_c} >= {tr_c}"

    cell_w = img_width / COLS
    cell_h = img_height / ROWS

    return {
        "x": int(tl_c * cell_w),
        "y": int(tl_r * cell_h),
        "width": int((tr_c - tl_c + 1) * cell_w),
        "height": int((bl_r - tl_r + 1) * cell_h),
        "area_cells": (tr_c - tl_c + 1) * (bl_r - tl_r + 1),
        "area_fraction": ((tr_c - tl_c + 1) * (bl_r - tl_r + 1)) / (ROWS * COLS),
    }


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/spoke-positioning-eval-input.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/spoke-positioning-eval-grid.png"
    style = sys.argv[3] if len(sys.argv) > 3 else "flood"
    cw, ch = draw_grid(inp, out, style=style)
    print(f"Grid drawn ({style}): {ROWS}x{COLS}, cell size {cw}x{ch}px, saved to {out}")
