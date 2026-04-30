"""Bare VLM overlay positioning eval — no grid, no markers.

Tests three output formats:
1. Percentages (distance-from-edges)
2. Pixels (origin + width/height)
3. Pixels (four corners)

Each with thinking on and off.
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


GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"


def encode_image(path: str | Path, scale: float = 1.0) -> str:
    img = Image.open(path)
    if scale < 1.0:
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


PROMPTS = {
    "pct": {
        "system": """\
You are Spoke's window positioning system.

Your job: find the largest rectangular area of the screen that does NOT \
cover the content the user wants to keep visible. The rectangle can cover \
toolbars, menu bars, status bars, margins, or any UI chrome that is not \
the specified content.

Output exactly four numbers on one line, space-separated:
LEFT_START% LEFT_END% TOP_START% BOTTOM_END%

LEFT_START% = distance from the left edge where the rectangle starts (0 = left edge)
LEFT_END% = distance from the left edge where the rectangle ends (100 = right edge)
TOP_START% = distance from the top where the rectangle starts (0 = top edge)
BOTTOM_END% = distance from the top where the rectangle ends (100 = bottom edge)

Example: 60 100 0 75
means a rectangle starting 60% from the left to the right edge, from the top to 75% down.

Output ONLY the four numbers. No reasoning, no explanation.""",
        "parse": "pct",
    },
    "px_origin": {
        "system": """\
You are Spoke's window positioning system.

The screen is {w}x{h} pixels.

Your job: find the largest rectangular area that does NOT cover the content \
the user wants to keep visible.

Output exactly four numbers on one line, space-separated:
X Y WIDTH HEIGHT

X and Y are the top-left corner in pixels. WIDTH and HEIGHT are the size in pixels.

Output ONLY the four numbers. No reasoning, no explanation.""",
        "parse": "px_origin",
    },
    "px_corners": {
        "system": """\
You are Spoke's window positioning system.

The screen is {w}x{h} pixels.

Your job: find the largest rectangular area that does NOT cover the content \
the user wants to keep visible.

Output exactly four numbers on one line, space-separated:
LEFT TOP RIGHT BOTTOM

LEFT and TOP are the top-left corner in pixels. RIGHT and BOTTOM are the \
bottom-right corner in pixels.

Output ONLY the four numbers. No reasoning, no explanation.""",
        "parse": "px_corners",
    },
}


def query_vlm(
    image_path: str,
    constraint: str,
    prompt_key: str,
    img_w: int,
    img_h: int,
    thinking: bool = False,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    scale: float = 1.0,
) -> dict:
    b64 = encode_image(image_path, scale=scale)
    prompt_def = PROMPTS[prompt_key]
    system = prompt_def["system"].format(w=img_w, h=img_h)

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": (
            f"Avoid blocking: {constraint}\n\n"
            "Return the largest rectangle that avoids that content."
        )},
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
        "max_tokens": 16384,
    }
    if not thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Authorization": "Bearer 1234", "Content-Type": "application/json"}

    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()
    return {"raw": raw, "elapsed_s": round(elapsed, 1), "model": model}


def parse_result(raw: str, parse_type: str, img_w: int, img_h: int) -> dict | None:
    numbers = re.findall(r"\d+", raw)
    if len(numbers) < 4:
        return None
    vals = [int(n) for n in numbers[:4]]

    if parse_type == "pct":
        left_start, left_end, top_start, bottom_end = vals
        if not (0 <= left_start < left_end <= 100 and 0 <= top_start < bottom_end <= 100):
            return None
        x = int(img_w * left_start / 100)
        y = int(img_h * top_start / 100)
        w = int(img_w * left_end / 100) - x
        h = int(img_h * bottom_end / 100) - y
    elif parse_type == "px_origin":
        x, y, w, h = vals
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return None
    elif parse_type == "px_corners":
        left, top, right, bottom = vals
        if not (0 <= left < right <= img_w and 0 <= top < bottom <= img_h):
            return None
        x, y, w, h = left, top, right - left, bottom - top
    else:
        return None

    area_frac = (w * h) / (img_w * img_h)
    return {
        "x": x, "y": y, "width": w, "height": h,
        "area_fraction": area_frac,
        "area_pct": round(area_frac * 100, 1),
    }


def visualize(image_path: str, rect: dict, out_path: str, label: str = "") -> None:
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    draw.rectangle([x, y, x + w, y + h], fill=(0, 120, 255, 60), outline=(0, 120, 255, 200), width=3)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    if label:
        draw.text((x + 10, y + 10), label, fill=(255, 255, 255, 220), font=font)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_all(screenshot_path: str, constraint: str, output_dir: str = "/tmp/spoke-positioning-eval/bare"):
    img = Image.open(screenshot_path)
    img_w, img_h = img.size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for prompt_key in ["pct", "px_origin", "px_corners"]:
        for thinking in [False, True]:
            think_label = "think" if thinking else "nothink"
            run_label = f"{prompt_key}_{think_label}"
            print(f"\n{'='*60}")
            print(f"  {run_label}")
            print(f"{'='*60}")

            try:
                vlm = query_vlm(
                    screenshot_path, constraint, prompt_key,
                    img_w, img_h, thinking=thinking,
                )
                raw = vlm["raw"]
                elapsed = vlm["elapsed_s"]
                print(f"  Response ({elapsed}s): {raw[:100]}")

                parse_type = PROMPTS[prompt_key]["parse"]
                rect = parse_result(raw, parse_type, img_w, img_h)

                if rect is None:
                    print(f"  PARSE FAIL")
                    results.append({"run": run_label, "status": "parse_fail", "raw": raw, "time": elapsed})
                else:
                    print(f"  Rect: {rect['x']},{rect['y']} {rect['width']}x{rect['height']} ({rect['area_pct']}%)")
                    viz_path = str(out_dir / f"{run_label}.png")
                    visualize(screenshot_path, rect, viz_path, f"{run_label}: {rect['area_pct']}%")
                    print(f"  Viz: {viz_path}")
                    results.append({"run": run_label, "status": "ok", "rect": rect, "time": elapsed})

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"run": run_label, "status": "error", "error": str(e)})

            time.sleep(3)

    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}\n")
    for r in results:
        if r["status"] == "ok":
            rect = r["rect"]
            print(f"  {r['run']:25s} {rect['area_pct']:5.1f}% area  {r['time']:6.1f}s")
        else:
            print(f"  {r['run']:25s} {r['status']:10s}  {r.get('time', '?')}s")


if __name__ == "__main__":
    import sys
    screenshot = sys.argv[1] if len(sys.argv) > 1 else "tools/positioning_eval/sample_screenshot.png"
    constraint = "the text content in the terminal — all the readable text on screen"
    run_all(screenshot, constraint)
