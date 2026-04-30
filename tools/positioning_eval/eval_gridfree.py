"""Grid-free VLM overlay positioning eval.

Draws quartile marker lines (25%, 50%, 75%) on the screenshot and asks
the model to pick horizontal and vertical start/end positions for the
overlay rectangle. No grid cells — just continuous coordinates.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


GRAPHEUS_URL = "http://localhost:8090/v1/chat/completions"

MARKER_COLOR = (255, 255, 0, 120)  # yellow, semi-transparent
LABEL_COLOR = (255, 255, 0, 200)
LABEL_BG = (0, 0, 0, 160)

MARKERS = [0, 25, 50, 75, 100]  # percentage positions


def draw_markers(img_path: str | Path, out_path: str | Path) -> None:
    """Draw quartile marker lines with percentage labels on the image."""
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw vertical marker lines (horizontal position)
    for pct in MARKERS:
        x = int(w * pct / 100)
        if 0 < x < w:
            draw.line([(x, 0), (x, h)], fill=MARKER_COLOR, width=2)
        # Label at top
        label = f"{pct}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        lx = max(2, min(x - tw // 2, w - tw - 2))
        draw.rectangle([lx - 2, 2, lx + tw + 2, bbox[3] - bbox[1] + 6], fill=LABEL_BG)
        draw.text((lx, 4), label, fill=LABEL_COLOR, font=font)

    # Draw horizontal marker lines (vertical position)
    for pct in MARKERS:
        y = int(h * pct / 100)
        if 0 < y < h:
            draw.line([(0, y), (w, y)], fill=MARKER_COLOR, width=2)
        # Label at left
        label = f"{pct}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        ly = max(2, min(y - th // 2, h - th - 2))
        draw.rectangle([2, ly - 2, bbox[2] - bbox[0] + 6, ly + th + 2], fill=LABEL_BG)
        draw.text((4, ly), label, fill=LABEL_COLOR, font=font)

    result = Image.alpha_composite(img, overlay)
    result.save(str(out_path))


def encode_image(path: str | Path, scale: float = 1.0) -> str:
    """Encode image as base64 PNG, optionally downscaling."""
    img = Image.open(path)
    if scale < 1.0:
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


SYSTEM_PROMPT = """\
You are Spoke's window positioning system.

The screenshot has yellow percentage markers at 0%, 25%, 50%, 75%, 100% \
along both axes. Use these as reference points.

Your job: find the largest rectangular area that does NOT cover the content \
the user wants to keep visible. The rectangle can cover toolbars, menu bars, \
status bars, margins, or any UI chrome that is not the specified content.

Output exactly four numbers on one line, space-separated:
LEFT% TOP% RIGHT% BOTTOM%

These are percentages of the screen dimensions. For example:
60 0 100 70
means a rectangle from 60% from the left edge to the right edge, and from \
the top to 70% down.

Use the yellow marker lines as visual reference points to estimate positions. \
You can use any percentage value 0-100, not just the marked ones.

Output ONLY the four numbers. No reasoning, no explanation."""


def query_vlm(
    marked_image_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    scale: float = 1.0,
) -> dict:
    """Send the marked image + constraint to the VLM."""
    b64 = encode_image(marked_image_path, scale=scale)

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
        {
            "type": "text",
            "text": (
                f"Avoid blocking: {constraint}\n\n"
                "Return the largest rectangle that avoids that content. "
                "Format: LEFT% TOP% RIGHT% BOTTOM%"
            ),
        },
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def parse_percentages(raw: str) -> tuple[int, int, int, int] | None:
    """Extract four percentage values from model response."""
    numbers = re.findall(r"\d+", raw)
    if len(numbers) >= 4:
        vals = [int(n) for n in numbers[:4]]
        # Validate ranges
        if all(0 <= v <= 100 for v in vals):
            left, top, right, bottom = vals
            if left < right and top < bottom:
                return left, top, right, bottom
    return None


def pct_to_pixels(left: int, top: int, right: int, bottom: int, img_w: int, img_h: int) -> dict:
    """Convert percentage rect to pixel coordinates."""
    x = int(img_w * left / 100)
    y = int(img_h * top / 100)
    w = int(img_w * right / 100) - x
    h = int(img_h * bottom / 100) - y
    area_fraction = (w * h) / (img_w * img_h)
    return {
        "x": x, "y": y, "width": w, "height": h,
        "area_fraction": area_fraction,
        "area_pct": round(area_fraction * 100, 1),
        "pct": f"{left}%-{right}% x {top}%-{bottom}%",
    }


def visualize_result(
    marked_path: str,
    rect_px: dict,
    out_path: str,
) -> None:
    """Draw the chosen rectangle on the marked image."""
    img = Image.open(marked_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = rect_px["x"], rect_px["y"]
    w, h = rect_px["width"], rect_px["height"]
    draw.rectangle([x, y, x + w, y + h], fill=(0, 120, 255, 60), outline=(0, 120, 255, 200), width=3)

    # Label the rectangle
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    label = f"{rect_px['pct']} ({rect_px['area_pct']}%)"
    draw.text((x + 10, y + 10), label, fill=(255, 255, 255, 220), font=font)

    result = Image.alpha_composite(img, overlay)
    result.save(out_path)


def run_eval(
    screenshot_path: str,
    constraint: str,
    model: str = "qwen3.6-35b-a3b-oq8",
    url: str = GRAPHEUS_URL,
    output_dir: str = "/tmp/spoke-positioning-eval/gridfree",
    scale: float = 1.0,
) -> dict:
    """Run a single grid-free positioning eval."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(screenshot_path)
    img_w, img_h = img.size

    # Step 1: Draw markers
    marked_path = str(out_dir / "marked.png")
    draw_markers(screenshot_path, marked_path)
    print(f"Marked image saved to {marked_path}")

    # Step 2: Query VLM
    print(f"Querying {model} with constraint: {constraint!r} (scale={scale})")
    vlm_result = query_vlm(marked_path, constraint, model=model, url=url, scale=scale)
    raw = vlm_result["raw"]
    print(f"  Response ({vlm_result['elapsed_s']}s): {raw}")

    # Step 3: Parse
    pcts = parse_percentages(raw)
    if pcts is None:
        print(f"  PARSE FAIL: could not extract valid percentages")
        return {"status": "parse_fail", "raw": raw, "vlm": vlm_result}

    left, top, right, bottom = pcts
    print(f"  Parsed: left={left}% top={top}% right={right}% bottom={bottom}%")

    # Step 4: Convert to pixels
    rect_px = pct_to_pixels(left, top, right, bottom, img_w, img_h)
    print(f"  Rectangle: {rect_px['x']},{rect_px['y']} {rect_px['width']}x{rect_px['height']} "
          f"({rect_px['area_pct']}% of screen)")

    # Step 5: Visualize
    viz_path = str(out_dir / "result.png")
    visualize_result(marked_path, rect_px, viz_path)
    print(f"  Visualization saved to {viz_path}")

    return {
        "status": "ok",
        "constraint": constraint,
        "percentages": {"left": left, "top": top, "right": right, "bottom": bottom},
        "rect_px": rect_px,
        "vlm": vlm_result,
        "viz_path": viz_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid-free VLM overlay positioning eval")
    parser.add_argument("screenshot", nargs="?",
                        default="tools/positioning_eval/sample_screenshot.png")
    parser.add_argument("--constraint", "-c",
                        default="the text content in the terminal — all the readable text on screen")
    parser.add_argument("--model", "-m", default="qwen3.6-35b-a3b-oq8")
    parser.add_argument("--url", default=GRAPHEUS_URL)
    parser.add_argument("--output-dir", "-o", default="/tmp/spoke-positioning-eval/gridfree")
    parser.add_argument("--scale", "-s", type=float, default=1.0)
    args = parser.parse_args()

    result = run_eval(
        args.screenshot,
        args.constraint,
        model=args.model,
        url=args.url,
        output_dir=args.output_dir,
        scale=args.scale,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
