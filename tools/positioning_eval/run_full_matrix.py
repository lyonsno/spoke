"""Full eval matrix: all images x all utterance variants, heatmaps for each."""

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
OUT_DIR = "/tmp/spoke-positioning-eval/full-matrix"

SYSTEM = (
    "You are Spoke's overlay positioning mode.\n\n"
    "The screenshot has a grid overlay with yellow reference labels (rows A-E, "
    "columns 1-5). Ignore the yellow labels — look THROUGH them at the screen.\n\n"
    "The user will tell you what they want to keep visible. Find the THREE "
    "grid cells that have the LEAST amount of whatever the user described. "
    "These are the cells where the user's content is most absent — the emptiest "
    "cells relative to what the user cares about.\n\n"
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

UTTERANCES = {
    "code-terse":  "stop blocking my code",
    "code-polite": "hey can you move out of the way of the code please",
    "text-terse":  "stop blocking the text",
    "text-polite": "hey can you move out of the way of the text please",
}


def draw_heatmap(grid_path, counts, winner, out_path, title=""):
    img = Image.open(grid_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cw, ch = w / SEED_COLS, h / SEED_ROWS
    mx = max(counts.values()) if counts else 1
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 48)
        sf = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
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
    draw.rectangle([10, h - 80, w - 10, h - 10], fill=(0, 0, 0, 180))
    draw.text((20, h - 70), title, fill=(255, 255, 255, 200), font=sf)
    draw.text((20, h - 45), f"Winner: {winner} ({counts.get(winner, 0)} votes)", fill=(0, 255, 255, 255), font=sf)
    Image.alpha_composite(img, overlay).save(out_path)


def run_one(img_key, utt_key, screenshot, utterance):
    label = f"{img_key}__{utt_key}"
    gp = os.path.join(OUT_DIR, f"{label}__grid.png")
    draw_grid(screenshot, gp, style="flood", rows=SEED_ROWS, cols=SEED_COLS)
    full = Image.open(gp)
    half = full.resize((full.width // 2, full.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    half.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    picks = []
    tt = 0.0
    for _ in range(N_RUNS):
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
            timeout=60,
        )
        tt += time.time() - t0
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
        cells = re.findall(r"[A-E][1-5]", raw)
        picks.extend(cells[:3])

    counts = dict(Counter(picks))
    ranked = Counter(picks).most_common()
    winner = ranked[0][0] if ranked else "??"
    wvotes = ranked[0][1] if ranked else 0

    hp = os.path.join(OUT_DIR, f"{label}__heatmap.png")
    draw_heatmap(gp, counts, winner, hp, title=f'{img_key} | "{utterance}"')

    return label, winner, wvotes, len(picks), round(tt, 1), counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    header = f"{'Case':45s} {'Winner':8s} {'Votes':7s} {'Time':7s}  Distribution"
    print(header)
    print("-" * 110)

    for img_key, screenshot in IMAGES.items():
        for utt_key, utterance in UTTERANCES.items():
            label, winner, wv, tv, tt, counts = run_one(img_key, utt_key, screenshot, utterance)
            print(f"{label:45s} {winner:8s} {wv:2d}/{tv:2d}   {tt:5.1f}s  {counts}")
        print()  # blank line between images

    subprocess.run(["open", OUT_DIR])


if __name__ == "__main__":
    main()
