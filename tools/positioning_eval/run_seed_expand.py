"""Seed + expand: vote for seed, then check ALL adjacent neighbors (voted and unvoted)."""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from grid_overlay import draw_grid, ROW_LABELS, COL_LABELS
from eval_positioning import largest_inscribed_rectangle

ROWS, COLS = 4, 4
N_SEED_RUNS = 5
OUT_DIR = "/tmp/spoke-positioning-eval/seed-expand"

SEED_SYSTEM = (
    "You are Spoke's overlay positioning mode.\n\n"
    "The screenshot has a grid overlay with yellow reference labels (rows A-D, "
    "columns 1-4). Ignore the yellow labels — look THROUGH them at the screen.\n\n"
    "The user will tell you what they want to keep visible. Find the THREE "
    "grid cells that have the LEAST amount of whatever the user described. "
    "These are the cells where the user's content is most absent — the emptiest "
    "cells relative to what the user cares about.\n\n"
    "Output ONLY three cell names separated by spaces. Nothing else."
)

CHECK_SYSTEM = (
    "You are checking a cropped region of a screen for content.\n\n"
    "This is a cropped section of a screenshot. Answer with ONLY one word:\n\n"
    "EMPTY — if this region is mostly blank, margins, toolbars, or unimportant UI\n"
    "CONTENT — if this region contains meaningful text, images, graphs, or data "
    "that a user would want to keep visible\n\n"
    "One word only. EMPTY or CONTENT."
)


def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def seed_vote(grid_path: str, utterance: str, scale: float = 0.5):
    img = Image.open(grid_path).copy()
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    b64 = encode_pil(img)

    all_picks = []
    t_start = time.time()
    for _ in range(N_SEED_RUNS):
        resp = requests.post(
            "http://localhost:8090/v1/chat/completions",
            headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
            json={
                "model": "qwen3.6-35b-a3b-oq8",
                "messages": [
                    {"role": "system", "content": SEED_SYSTEM},
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
        cells = re.findall(r"[A-D][1-4]", raw)
        all_picks.extend(cells[:3])

    seed_time = round(time.time() - t_start, 1)
    counts = Counter(all_picks)
    winner = counts.most_common(1)[0][0]
    return winner, counts, seed_time


def check_cell(img: Image.Image, row: int, col: int, utterance: str):
    w, h = img.size
    cw, ch = w / COLS, h / ROWS
    crop = img.crop((int(col * cw), int(row * ch), int((col + 1) * cw), int((row + 1) * ch)))
    b64 = encode_pil(crop)

    t0 = time.time()
    resp = requests.post(
        "http://localhost:8090/v1/chat/completions",
        headers={"Authorization": "Bearer 1234", "Content-Type": "application/json"},
        json={
            "model": "qwen3.6-35b-a3b-oq8",
            "messages": [
                {"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": f"Does this contain: {utterance}"},
                ]},
            ],
            "temperature": 0.1, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
    verdict = "EMPTY" if "EMPTY" in raw else "CONTENT"
    return row, col, verdict, round(elapsed, 2)


def get_all_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                neighbors.append((nr, nc))
    return neighbors


def run_seed_expand(screenshot_path: str, utterance: str, label: str):
    # Load image fully into memory to avoid file locking issues
    img = Image.open(screenshot_path).copy()
    w, h = img.size

    print(f"\n{'=' * 70}")
    print(f"  {label}: \"{utterance}\"")
    print(f"{'=' * 70}")

    # Step 1: Seed vote
    grid_path = os.path.join(OUT_DIR, f"{label}__grid.png")
    draw_grid(screenshot_path, grid_path, style="flood", rows=ROWS, cols=COLS)
    winner, vote_counts, seed_time = seed_vote(grid_path, utterance)
    winner_r = ord(winner[0]) - ord("A")
    winner_c = int(winner[1:]) - 1

    print(f"  Seed: {winner} ({seed_time}s)")
    print(f"  Votes: {dict(vote_counts.most_common())}")

    # Step 2: Get ALL neighbors of the winner (voted and unvoted)
    all_neighbors = get_all_neighbors(winner_r, winner_c)
    voted_set = {c for c, _ in vote_counts.most_common()}
    neighbor_info = []
    for r, c in all_neighbors:
        cell_name = f"{ROW_LABELS[r]}{c + 1}"
        votes = vote_counts.get(cell_name, 0)
        neighbor_info.append((r, c, cell_name, votes))

    print(f"  Checking {len(all_neighbors)} neighbors:")
    for r, c, name, votes in neighbor_info:
        tag = f"({votes} votes)" if votes > 0 else "(0 votes)"
        print(f"    {name} {tag}")

    # Step 3: Check ALL neighbors in parallel
    safe_cells = {(winner_r, winner_c)}
    cell_verdicts = {}
    t_expand_start = time.time()

    with ThreadPoolExecutor(max_workers=len(all_neighbors)) as pool:
        futures = {
            pool.submit(check_cell, img, r, c, utterance): (r, c)
            for r, c in all_neighbors
        }
        for f in as_completed(futures):
            r, c = futures[f]
            row, col, verdict, elapsed = f.result()
            cell_name = f"{ROW_LABELS[r]}{c + 1}"
            votes = vote_counts.get(cell_name, 0)
            cell_verdicts[(r, c)] = verdict
            tag = f"({votes}v)" if votes > 0 else "(0v)"
            print(f"    {cell_name} {tag}: {verdict} ({elapsed}s)")
            if verdict == "EMPTY":
                safe_cells.add((r, c))

    expand_time = round(time.time() - t_expand_start, 1)
    total_time = round(seed_time + expand_time, 1)

    # Step 4: Find largest rectangle
    rect = largest_inscribed_rectangle(list(safe_cells), rows=ROWS, cols=COLS)
    if rect:
        tr, br = rect["top_row"], rect["bottom_row"]
        lc, rc = rect["left_col"], rect["right_col"]
        rw, rh = rc - lc + 1, br - tr + 1
        area_pct = round(rw * rh / (ROWS * COLS) * 100, 1)
    else:
        tr = br = winner_r
        lc = rc = winner_c
        rw = rh = 1
        area_pct = round(1 / (ROWS * COLS) * 100, 1)

    safe_names = sorted(f"{ROW_LABELS[r]}{c + 1}" for r, c in safe_cells)
    print(f"\n  Safe: {safe_names}")
    print(f"  Rect: row {tr}-{br}, col {lc}-{rc} ({rw}x{rh} = {area_pct}%)")
    print(f"  Timing: seed {seed_time}s + expand {expand_time}s = {total_time}s")

    # Visualize
    viz = img.convert("RGBA")
    overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cw, ch = w / COLS, h / ROWS
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 36)
        sf = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    except (OSError, IOError):
        font = sf = ImageFont.load_default()

    # Draw all neighbor verdicts
    for r, c in all_neighbors:
        x0, y0 = int(c * cw), int(r * ch)
        x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
        cell_name = f"{ROW_LABELS[r]}{c + 1}"
        votes = vote_counts.get(cell_name, 0)
        verdict = cell_verdicts.get((r, c), "?")

        if verdict == "EMPTY":
            draw.rectangle([x0, y0, x1, y1], fill=(0, 200, 0, 60), outline=(0, 255, 0, 200), width=3)
            draw.text((x0 + 10, y0 + 10), f"EMPTY ({votes}v)", fill=(0, 255, 0, 255), font=sf)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 40), outline=(255, 0, 0, 150), width=2)
            draw.text((x0 + 10, y0 + 10), f"CONTENT ({votes}v)", fill=(255, 100, 100, 255), font=sf)

    # Seed cell
    x0, y0 = int(winner_c * cw), int(winner_r * ch)
    x1, y1 = int((winner_c + 1) * cw), int((winner_r + 1) * ch)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 255, 80), outline=(0, 255, 255, 255), width=4)
    draw.text((x0 + 10, y0 + 10), "SEED", fill=(0, 255, 255, 255), font=font)

    # Largest rect outline
    if rect:
        rx0, ry0 = int(lc * cw), int(tr * ch)
        rx1, ry1 = int((rc + 1) * cw), int((br + 1) * ch)
        draw.rectangle([rx0, ry0, rx1, ry1], outline=(255, 255, 0, 255), width=5)

    # Info box
    draw.rectangle([10, h - 100, w - 10, h - 10], fill=(0, 0, 0, 200))
    draw.text((20, h - 95), f'{label} | "{utterance}"', fill=(255, 255, 255, 200), font=sf)
    draw.text((20, h - 72),
              f"Seed: {winner} | Safe: {len(safe_cells)} | Rect: {area_pct}% | "
              f"seed {seed_time}s + expand {expand_time}s = {total_time}s",
              fill=(0, 255, 255, 255), font=sf)
    draw.text((20, h - 50),
              "Cyan=seed, Green=EMPTY, Red=CONTENT, Yellow outline=rect | "
              "(Nv)=seed votes",
              fill=(200, 200, 200, 180), font=sf)

    result_path = os.path.join(OUT_DIR, f"{label}__result.png")
    Image.alpha_composite(viz, overlay).save(result_path)

    return {
        "label": label,
        "safe": len(safe_cells),
        "area_pct": area_pct,
        "seed_time": seed_time,
        "expand_time": expand_time,
        "total_time": total_time,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    SD = os.path.expanduser("~/.config/spoke/screenshots")
    cases = [
        (os.path.join(SD, "claude_terminal_smoke_image_1.png"),
         "stop blocking my code", "terminal"),
        (os.path.join(SD, "github_contributors_smoke_image.png"),
         "stop blocking the graph and the contributor stats", "github"),
        (os.path.join(SD, "claude_openai_docs_smoke_image.png"),
         "stop blocking the code examples", "openai"),
        (os.path.join(SD, "turboquant_cover_upper_right_logo_smoke_image.png"),
         "stop blocking the article", "turboquant"),
    ]

    print(f"{'Case':15s} {'Safe':6s} {'Rect':7s} {'Seed':6s} {'Expand':8s} {'Total':7s}")
    print("-" * 55)
    for sp, ut, lb in cases:
        r = run_seed_expand(sp, ut, lb)
        print(f"\n{r['label']:15s} {r['safe']:3d}    {r['area_pct']:5.1f}%  {r['seed_time']:5.1f}s  {r['expand_time']:6.1f}s  {r['total_time']:5.1f}s")

    subprocess.run(["open", OUT_DIR])


if __name__ == "__main__":
    main()
