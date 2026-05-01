"""Eval with current overlay position marked as red outline.

Tests whether showing the model its current position improves seed selection.
Each image gets a red rectangle outline indicating where the overlay currently
sits (blocking some important content), and the system prompt references it.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import draw_grid

SEED_ROWS = 5
SEED_COLS = 5
N_RUNS = 5
OUT_DIR = "/tmp/spoke-positioning-eval/positioned-matrix"

SYSTEM = (
    "You are Spoke's overlay positioning mode.\n\n"
    "The screenshot has a grid overlay with yellow reference labels (rows A-E, "
    "columns 1-5). Ignore the yellow labels — look THROUGH them at the screen.\n\n"
    "The RED rectangle outline shows where the overlay is currently positioned. "
    "The user wants you to find a better position.\n\n"
    "The user will tell you what they want to keep visible. Find the THREE "
    "grid cells that have the LEAST amount of whatever the user described. "
    "These are the cells where the user's content is most absent — the emptiest "
    "cells relative to what the user cares about.\n\n"
    "Output ONLY three cell names separated by spaces. Nothing else."
)

# Generic "just move" prompt — no content specification
SYSTEM_GENERIC = (
    "You are Spoke's overlay positioning mode.\n\n"
    "The screenshot has a grid overlay with yellow reference labels (rows A-E, "
    "columns 1-5). Ignore the yellow labels — look THROUGH them at the screen.\n\n"
    "The RED rectangle outline shows where the overlay is currently positioned. "
    "The user wants you to find a better position that blocks less of what's "
    "important on screen.\n\n"
    "Find the THREE grid cells that have the LEAST important content — the "
    "cells where the overlay would block the least.\n\n"
    "Output ONLY three cell names separated by spaces. Nothing else."
)

SD = os.path.expanduser("~/.config/spoke/screenshots")
IMAGES = {
    "terminal":    os.path.join(SD, "claude_terminal_smoke_image_1.png"),
    "github":      os.path.join(SD, "github_contributors_smoke_image.png"),
    "openai-docs": os.path.join(SD, "claude_openai_docs_smoke_image.png"),
    "hf-models":   os.path.join(SD, "hf_model_page_smoke_image.png"),
    "turboquant":  os.path.join(SD, "turboquant_cover_upper_right_logo_smoke_image.png"),
}

# Per-image: current overlay position (x%, y%, w%, h%) and sensible utterances
IMAGE_CONFIGS = {
    "terminal": {
        "overlay_pct": (30, 20, 40, 50),  # center, blocking terminal text
        "utterances": {
            "specific": "stop blocking my code",
            "polite": "hey can you move out of the way of the terminal text please",
            "generic": "can you please move out of the way",
        },
    },
    "github": {
        "overlay_pct": (25, 30, 45, 50),  # blocking the graph
        "utterances": {
            "specific": "stop blocking the graph and the contributor stats",
            "polite": "move so I can see the contributor graph please",
            "generic": "can you please move out of the way",
        },
    },
    "openai-docs": {
        "overlay_pct": (20, 25, 50, 55),  # blocking the code blocks
        "utterances": {
            "specific": "stop blocking the code examples",
            "polite": "hey can you move out of the way of the documentation please",
            "generic": "can you please move out of the way",
        },
    },
    "hf-models": {
        "overlay_pct": (15, 20, 60, 60),  # blocking model cards
        "utterances": {
            "specific": "stop blocking the model list",
            "polite": "move out of the way of the models please",
            "generic": "can you please move out of the way",
        },
    },
    "turboquant": {
        "overlay_pct": (10, 30, 55, 50),  # blocking article text
        "utterances": {
            "specific": "stop blocking the article",
            "polite": "move out of the way of the text please",
            "generic": "can you please move out of the way",
        },
    },
}


def draw_overlay_outline(img: Image.Image, x_pct: int, y_pct: int, w_pct: int, h_pct: int) -> Image.Image:
    """Draw a red rectangle outline indicating current overlay position."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x0 = int(w * x_pct / 100)
    y0 = int(h * y_pct / 100)
    x1 = int(w * (x_pct + w_pct) / 100)
    y1 = int(h * (y_pct + h_pct) / 100)
    # Draw thick red outline
    for offset in range(4):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                       outline=(255, 0, 0, 255))
    return img


def draw_heatmap(grid_path, counts, winner, out_path, title="", total_time=0.0):
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cw, ch = w / SEED_COLS, h / SEED_ROWS
    mx = max(counts.values()) if counts else 1
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
        sf = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    except (OSError, IOError):
        font = sf = ImageFont.load_default()

    for cs, cnt in counts.items():
        ri = ord(cs[0]) - ord("A")
        ci = int(cs[1:]) - 1
        if not (0 <= ri < SEED_ROWS and 0 <= ci < SEED_COLS):
            continue
        x0, y0 = int(ci * cw), int(ri * ch)
        x1, y1 = int((ci + 1) * cw), int((ri + 1) * ch)
        intensity = cnt / mx
        alpha = int(40 + intensity * 120)
        if cs == winner:
            draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, alpha))
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 255, 255), width=4)
        else:
            rv = int(255 * intensity)
            gv = int(140 * (1 - intensity))
            draw.rectangle([x0, y0, x1, y1], fill=(rv, gv, 0, alpha))
            draw.rectangle([x0, y0, x1, y1], outline=(rv, gv, 0, 180), width=2)
        label = str(cnt)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tc = (0, 255, 255, 255) if cs == winner else (255, 255, 255, 220)
        draw.text((x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2), label, fill=tc, font=font)

    # Info box at bottom
    draw.rectangle([10, h - 100, w - 10, h - 10], fill=(0, 0, 0, 200))
    draw.text((20, h - 95), title, fill=(255, 255, 255, 200), font=sf)
    draw.text((20, h - 72),
              f"Winner: {winner} ({counts.get(winner, 0)} votes) | "
              f"Total time: {total_time:.1f}s ({N_RUNS} runs x top-3)",
              fill=(0, 255, 255, 255), font=sf)
    draw.text((20, h - 50),
              f"Distribution: {counts}",
              fill=(200, 200, 200, 180), font=sf)

    Image.alpha_composite(img, overlay).save(out_path)


def run_one(img_key, utt_key, screenshot, utterance, overlay_pct, use_generic_system=False):
    label = f"{img_key}__{utt_key}"

    # Load image, draw red overlay outline, then draw grid on top
    base_img = Image.open(screenshot)
    with_outline = draw_overlay_outline(base_img, *overlay_pct)

    # Save temp image with outline, then draw grid on it
    temp_path = os.path.join(OUT_DIR, f"{label}__outlined.png")
    with_outline.save(temp_path)

    grid_path = os.path.join(OUT_DIR, f"{label}__grid.png")
    draw_grid(temp_path, grid_path, style="flood", rows=SEED_ROWS, cols=SEED_COLS)

    # Downscale to 50%
    full = Image.open(grid_path)
    half = full.resize((full.width // 2, full.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    half.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    system = SYSTEM_GENERIC if use_generic_system else SYSTEM
    picks = []
    t_start = time.time()
    for _ in range(N_RUNS):
        resp = requests.post(
            "http://localhost:8090/v1/chat/completions",
            headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
            json={
                "model": "qwen3.6-35b-a3b-oq8",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": utterance},
                    ]},
                ],
                "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
                "max_tokens": 16384,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
        cells = re.findall(r"[A-E][1-5]", raw)
        picks.extend(cells[:3])
    total_time = round(time.time() - t_start, 1)

    counts = dict(Counter(picks))
    ranked = Counter(picks).most_common()
    winner = ranked[0][0] if ranked else "??"
    wvotes = ranked[0][1] if ranked else 0

    hp = os.path.join(OUT_DIR, f"{label}__heatmap.png")
    draw_heatmap(grid_path, counts, winner, hp,
                 title=f'{img_key} | "{utterance}"',
                 total_time=total_time)

    return label, winner, wvotes, len(picks), total_time, counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    header = f"{'Case':50s} {'Winner':8s} {'Votes':7s} {'Time':7s}  Distribution"
    print(header)
    print("-" * 120)

    for img_key, config in IMAGE_CONFIGS.items():
        screenshot = IMAGES[img_key]
        overlay_pct = config["overlay_pct"]

        for utt_key, utterance in config["utterances"].items():
            use_generic = (utt_key == "generic")
            label, winner, wv, tv, tt, counts = run_one(
                img_key, utt_key, screenshot, utterance, overlay_pct,
                use_generic_system=use_generic,
            )
            print(f"{label:50s} {winner:8s} {wv:2d}/{tv:2d}   {tt:5.1f}s  {counts}")
        print()

    subprocess.run(["open", OUT_DIR])


if __name__ == "__main__":
    main()
