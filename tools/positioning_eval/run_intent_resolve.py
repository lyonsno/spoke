"""Intent resolve pass: 4x4 grid, model marks which cells contain the user's specified content.

For each cell: does it contain the thing the user wants to avoid? YES or NO.
Forces consideration of every cell. Run 3 times per case, separate heatmaps.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import draw_grid, ROW_LABELS, COL_LABELS
from eval_positioning import largest_inscribed_rectangle

ROWS, COLS = 4, 4
OUT_DIR = "/tmp/spoke-positioning-eval/intent-resolve"

SYSTEM = (
    "You are Spoke's content detection system.\n\n"
    "The screenshot has a 4x4 grid overlay (rows A-D, columns 1-4). "
    "The yellow labels are reference markers — look THROUGH them at the screen.\n\n"
    "The user will describe content they want to keep visible. For EVERY cell "
    "in the grid (A1 through D4), determine whether it contains that content.\n\n"
    "Output exactly 16 lines, one per cell, in order:\n"
    "A1: YES or NO\n"
    "A2: YES or NO\n"
    "...\n"
    "D4: YES or NO\n\n"
    "YES = this cell contains the user's specified content\n"
    "NO = this cell does not contain it (empty, margins, chrome, unrelated UI)\n\n"
    "You MUST output all 16 cells. Do not skip any."
)


def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_resolve(grid_path: str, utterance: str, scale: float = 0.5):
    img = Image.open(grid_path).copy()
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    b64 = encode_pil(img)

    t0 = time.time()
    resp = requests.post(
        "http://localhost:8090/v1/chat/completions",
        headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
        json={
            "model": "qwen3.6-35b-a3b-oq8",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": utterance},
                ]},
            ],
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16384,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return raw, round(elapsed, 1)


def parse_resolve(raw: str) -> dict[str, bool]:
    """Parse 16 YES/NO responses into a dict of cell -> has_content."""
    result = {}
    for row in range(ROWS):
        for col in range(COLS):
            cell = f"{ROW_LABELS[row]}{col + 1}"
            # Look for "A1: YES" or "A1: NO" pattern
            pattern = rf"{cell}\s*:\s*(YES|NO)"
            match = re.search(pattern, raw.upper())
            if match:
                result[cell] = match.group(1) == "YES"
            else:
                # Fallback: try to find the cell name near YES/NO
                result[cell] = None  # couldn't parse
    return result


def draw_resolve_result(grid_path: str, resolve: dict[str, bool], out_path: str,
                         title: str = "", total_time: float = 0.0, rect_info: str = ""):
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cw, ch = w / COLS, h / ROWS

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 42)
        sf = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    except (OSError, IOError):
        font = sf = ImageFont.load_default()

    yes_count = 0
    no_count = 0
    empty_cells = []

    for row in range(ROWS):
        for col in range(COLS):
            cell = f"{ROW_LABELS[row]}{col + 1}"
            x0, y0 = int(col * cw), int(row * ch)
            x1, y1 = int((col + 1) * cw), int((row + 1) * ch)

            has_content = resolve.get(cell)
            if has_content is True:
                draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 70), outline=(255, 0, 0, 200), width=3)
                draw.text((x0 + 10, y0 + 10), "YES", fill=(255, 100, 100, 255), font=font)
                yes_count += 1
            elif has_content is False:
                draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 50), outline=(0, 255, 0, 180), width=3)
                draw.text((x0 + 10, y0 + 10), "NO", fill=(0, 255, 0, 255), font=font)
                no_count += 1
                empty_cells.append((row, col))
            else:
                draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 0, 40), outline=(200, 200, 0, 150), width=2)
                draw.text((x0 + 10, y0 + 10), "???", fill=(200, 200, 0, 255), font=font)

    # Find largest rect in empty cells
    if empty_cells:
        rect = largest_inscribed_rectangle(empty_cells, rows=ROWS, cols=COLS)
        if rect:
            tr, br = rect["top_row"], rect["bottom_row"]
            lc, rc = rect["left_col"], rect["right_col"]
            rx0, ry0 = int(lc * cw), int(tr * ch)
            rx1, ry1 = int((rc + 1) * cw), int((br + 1) * ch)
            draw.rectangle([rx0, ry0, rx1, ry1], outline=(0, 200, 255, 255), width=5)
            rw, rh = rc - lc + 1, br - tr + 1
            rect_info = f"Rect: {ROW_LABELS[tr]}{lc+1}-{ROW_LABELS[br]}{rc+1} ({rw}x{rh} = {round(rw*rh/16*100,1)}%)"

    # Info box
    draw.rectangle([10, h - 100, w - 10, h - 10], fill=(0, 0, 0, 200))
    draw.text((20, h - 95), title, fill=(255, 255, 255, 200), font=sf)
    draw.text((20, h - 72),
              f"YES(content): {yes_count} | NO(empty): {no_count} | {rect_info} | {total_time:.1f}s",
              fill=(0, 255, 255, 255), font=sf)
    draw.text((20, h - 50),
              "RED=contains user content | GREEN=empty/available | CYAN outline=largest rect",
              fill=(200, 200, 200, 180), font=sf)

    Image.alpha_composite(img, overlay).save(out_path)


def run_case(screenshot_path: str, utterance: str, label: str, n_runs: int = 3):
    grid_path = os.path.join(OUT_DIR, f"{label}__grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=ROWS, cols=COLS)

    print(f"\n{'=' * 70}")
    print(f"  {label}: \"{utterance}\"")
    print(f"{'=' * 70}")

    for run_i in range(n_runs):
        raw, elapsed = query_resolve(grid_path, utterance)
        resolve = parse_resolve(raw)

        # Count
        yes_cells = [c for c, v in resolve.items() if v is True]
        no_cells = [c for c, v in resolve.items() if v is False]
        unknown = [c for c, v in resolve.items() if v is None]

        print(f"\n  Run {run_i + 1} ({elapsed}s):")
        print(f"    YES (content): {yes_cells}")
        print(f"    NO  (empty):   {no_cells}")
        if unknown:
            print(f"    ???:           {unknown}")
        print(f"    Raw: {raw[:200]}")

        viz_path = os.path.join(OUT_DIR, f"{label}__run{run_i + 1}.png")
        draw_resolve_result(
            grid_path, resolve, viz_path,
            title=f'{label} run {run_i + 1} | "{utterance}"',
            total_time=elapsed,
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    SD = os.path.expanduser("~/.config/spoke/screenshots")
    cases = [
        (os.path.join(SD, "claude_terminal_smoke_image_1.png"),
         "hey could you just occupy the top third of the screen without covering the menu bar or the tabs",
         "terminal-topthing"),
        (os.path.join(SD, "claude_terminal_smoke_image_1.png"),
         "stop blocking my code",
         "terminal-code"),
        (os.path.join(SD, "github_contributors_smoke_image.png"),
         "stop blocking the graph and the contributor stats",
         "github-graph"),
        (os.path.join(SD, "claude_openai_docs_smoke_image.png"),
         "stop blocking the code examples",
         "openai-code"),
        (os.path.join(SD, "turboquant_cover_upper_right_logo_smoke_image.png"),
         "stop blocking the article",
         "turboquant-article"),
    ]

    for screenshot, utterance, label in cases:
        run_case(screenshot, utterance, label)

    subprocess.run(["open", OUT_DIR])


if __name__ == "__main__":
    main()
