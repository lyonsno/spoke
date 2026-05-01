"""Semantic overlay repositioning pipeline.

Two layers:
1. Intent resolve: user utterance → content description noun phrase
2. Content detection: 4x4 grid binary classification → content map
3. (Deterministic) Largest inscribed rectangle from empty cells

Returns a CGRect-compatible dict for the overlay's new position.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from .grid_overlay import draw_grid, ROW_LABELS, COLS, ROWS

# Default grid for positioning
POS_ROWS = 4
POS_COLS = 4

INTENT_SYSTEM = (
    "You convert user requests about overlay positioning into structured intents.\n\n"
    "The user wants to reposition an overlay on their screen. There are two modes:\n\n"
    "1. AVOID mode: The user wants the overlay to avoid certain content.\n"
    "   Output: AVOID: <content noun phrase>\n\n"
    "2. TARGET mode: The user wants the overlay to go to a specific screen region.\n"
    "   Output: TARGET: <region description>\n"
    "   Valid regions: top-left, top, top-right, left, center, right, "
    "bottom-left, bottom, bottom-right, top-half, bottom-half, left-half, right-half\n\n"
    "Examples:\n"
    "- 'stop blocking my code' → AVOID: code\n"
    "- 'move out of the way of the terminal text' → AVOID: terminal text\n"
    "- 'get off the graph' → AVOID: graph\n"
    "- 'avoid the regions with blue or purple text' → AVOID: blue, purple, or lavender text\n"
    "- 'can you please move out of the way' → AVOID: important content\n"
    "- 'just move' → AVOID: important content\n"
    "- 'go to the top right' → TARGET: top-right\n"
    "- 'occupy the middle of the screen' → TARGET: center\n"
    "- 'position yourself in the middle' → TARGET: center\n"
    "- 'move to the bottom' → TARGET: bottom\n"
    "- 'can you go to the left side' → TARGET: left-half\n"
    "- 'stay in the top half' → TARGET: top-half\n\n"
    "Output ONLY the mode and value. Nothing else."
)

DETECT_SYSTEM = (
    "You are a content detection system.\n\n"
    "You will be shown an image of a screen divided into a 4x4 grid "
    "(rows A-D, columns 1-4).\n\n"
    "IMPORTANT: The yellow text labels (A1, A2, B1, etc.) and yellow grid lines "
    "are reference markers drawn ON TOP of the screen image. They are NOT part of "
    "the screen content. Ignore them completely when deciding YES or NO. Only look "
    "at what is underneath them.\n\n"
    "You will also receive a description of a type of content. For each cell "
    "in the grid, determine whether the UNDERLYING screen content (not the yellow "
    "labels) contains that type of content.\n\n"
    "Output exactly 16 lines, one per cell, in order:\n"
    "A1: YES or NO\n"
    "A2: YES or NO\n"
    "...\n"
    "D4: YES or NO\n\n"
    "YES = this cell contains that type of content\n"
    "NO = this cell does not\n\n"
    "You MUST output all 16 cells."
)


def _get_api_url():
    """Get the OMLX/Grapheus endpoint URL."""
    return os.environ.get("SPOKE_VLM_URL", "http://localhost:8090/v1/chat/completions")


def _get_api_key():
    """Get the API key."""
    return os.environ.get("OMLX_SERVER_API_KEY", "1234")


def _encode_image(img: Image.Image, scale: float = 0.5) -> str:
    """Encode PIL image as base64 PNG, optionally downscaling."""
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_TARGET_REGIONS = {
    "top-left":     {"x": 0.0,  "y": 0.0,  "width": 0.5,  "height": 0.5},
    "top":          {"x": 0.25, "y": 0.0,  "width": 0.5,  "height": 0.5},
    "top-right":    {"x": 0.5,  "y": 0.0,  "width": 0.5,  "height": 0.5},
    "left":         {"x": 0.0,  "y": 0.25, "width": 0.5,  "height": 0.5},
    "center":       {"x": 0.25, "y": 0.25, "width": 0.5,  "height": 0.5},
    "right":        {"x": 0.5,  "y": 0.25, "width": 0.5,  "height": 0.5},
    "bottom-left":  {"x": 0.0,  "y": 0.5,  "width": 0.5,  "height": 0.5},
    "bottom":       {"x": 0.25, "y": 0.5,  "width": 0.5,  "height": 0.5},
    "bottom-right": {"x": 0.5,  "y": 0.5,  "width": 0.5,  "height": 0.5},
    "top-half":     {"x": 0.0,  "y": 0.0,  "width": 1.0,  "height": 0.5},
    "bottom-half":  {"x": 0.0,  "y": 0.5,  "width": 1.0,  "height": 0.5},
    "left-half":    {"x": 0.0,  "y": 0.0,  "width": 0.5,  "height": 1.0},
    "right-half":   {"x": 0.5,  "y": 0.0,  "width": 0.5,  "height": 1.0},
}


_TARGET_ALIASES = {
    "middle": "center",
    "centre": "center",
    "upper-left": "top-left",
    "upper-right": "top-right",
    "upper left": "top-left",
    "upper right": "top-right",
    "lower-left": "bottom-left",
    "lower-right": "bottom-right",
    "lower left": "bottom-left",
    "lower right": "bottom-right",
    "top left": "top-left",
    "top right": "top-right",
    "bottom left": "bottom-left",
    "bottom right": "bottom-right",
    "top-center": "top",
    "bottom-center": "bottom",
    "center-left": "left",
    "center-right": "right",
    "upper-half": "top-half",
    "lower-half": "bottom-half",
    "upper half": "top-half",
    "lower half": "bottom-half",
    "top half": "top-half",
    "bottom half": "bottom-half",
    "left half": "left-half",
    "right half": "right-half",
    "left side": "left-half",
    "right side": "right-half",
}


def _resolve_target_region(region: str) -> dict | None:
    """Resolve a target region string to a rect, with alias support."""
    region = region.strip().lower().rstrip(".")
    # Direct match
    rect = _TARGET_REGIONS.get(region)
    if rect is not None:
        return rect
    # Alias match
    canonical = _TARGET_ALIASES.get(region)
    if canonical is not None:
        return _TARGET_REGIONS.get(canonical)
    # Substring match — prefer longest canonical key contained in input
    best_key, best_rect = None, None
    for key, rect in _TARGET_REGIONS.items():
        if key in region and (best_key is None or len(key) > len(best_key)):
            best_key, best_rect = key, rect
    return best_rect


def resolve_intent(utterance: str) -> dict:
    """Layer 1: Convert user utterance to structured positioning intent.

    Returns {"mode": "avoid", "content_desc": "..."} or
            {"mode": "target", "region": "...", "rect": {...}}.
    """
    resp = requests.post(
        _get_api_url(),
        headers={"Authorization": f"Bearer {_get_api_key()}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": utterance},
            ],
            "temperature": 0.3, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip().strip("\"'")

    if raw.upper().startswith("TARGET:"):
        region = raw.split(":", 1)[1].strip().lower()
        rect = _resolve_target_region(region)
        if rect is not None:
            return {"mode": "target", "region": region, "rect": dict(rect)}
    if raw.upper().startswith("AVOID:"):
        content_desc = raw.split(":", 1)[1].strip()
        return {"mode": "avoid", "content_desc": content_desc}

    # Legacy format (bare noun phrase) — treat as avoid
    return {"mode": "avoid", "content_desc": raw}


def detect_content(screenshot: Image.Image, content_desc: str) -> dict[str, bool]:
    """Layer 2: Binary content detection per cell on a 4x4 grid.

    Returns dict mapping cell name (e.g. 'A1') to whether it contains the content.
    """
    import tempfile

    # Draw grid on screenshot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        screenshot.save(f, format="PNG")
        screenshot_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        grid_path = f.name

    try:
        draw_grid(screenshot_path, grid_path, style="flood", rows=POS_ROWS, cols=POS_COLS)
        grid_img = Image.open(grid_path).copy()
    finally:
        os.unlink(screenshot_path)
        os.unlink(grid_path)

    b64 = _encode_image(grid_img, scale=0.5)

    resp = requests.post(
        _get_api_url(),
        headers={"Authorization": f"Bearer {_get_api_key()}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": DETECT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": content_desc},
                ]},
            ],
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16384,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    result = {}
    for row in range(POS_ROWS):
        for col in range(POS_COLS):
            cell = f"{ROW_LABELS[row]}{col + 1}"
            match = re.search(rf"{cell}\s*:\s*(YES|NO)", raw.upper())
            result[cell] = match.group(1) == "YES" if match else False
    return result


def largest_rectangle(content_map: dict[str, bool]) -> dict | None:
    """Find the largest rectangle in cells marked as NOT containing content.

    Returns dict with keys: x, y, width, height (as fractions 0-1 of screen).
    """
    # Build grid of empty cells
    grid = [[False] * POS_COLS for _ in range(POS_ROWS)]
    for cell, has_content in content_map.items():
        if not has_content:
            row = ord(cell[0]) - ord("A")
            col = int(cell[1:]) - 1
            if 0 <= row < POS_ROWS and 0 <= col < POS_COLS:
                grid[row][col] = True

    # Histogram method for largest rectangle
    heights = [[0] * POS_COLS for _ in range(POS_ROWS)]
    for c in range(POS_COLS):
        for r in range(POS_ROWS):
            if grid[r][c]:
                heights[r][c] = (heights[r - 1][c] + 1) if r > 0 else 1

    best_area = 0
    best_rect = None

    for r in range(POS_ROWS):
        stack = []
        for c in range(POS_COLS + 1):
            h = heights[r][c] if c < POS_COLS else 0
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

    tr, br = best_rect["top_row"], best_rect["bottom_row"]
    lc, rc = best_rect["left_col"], best_rect["right_col"]

    return {
        "x": lc / POS_COLS,
        "y": tr / POS_ROWS,
        "width": (rc - lc + 1) / POS_COLS,
        "height": (br - tr + 1) / POS_ROWS,
    }


def reposition(utterance: str, screenshot: Image.Image) -> dict | None:
    """Full pipeline: utterance + screenshot → new overlay position.

    Returns dict with x, y, width, height as fractions of screen (0-1),
    or None if no viable position found.
    """
    t0 = time.time()

    # Layer 1: resolve intent
    intent = resolve_intent(utterance)

    if intent["mode"] == "target":
        # Direct targeting — no content detection needed
        rect = dict(intent["rect"])
        rect["content_desc"] = f"targeting: {intent['region']}"
        rect["utterance"] = utterance
        rect["elapsed_s"] = round(time.time() - t0, 2)
        return rect

    # Avoid mode — detect content and find empty space
    content_desc = intent["content_desc"]

    # Layer 2: detect content
    content_map = detect_content(screenshot, content_desc)

    # Layer 3: find largest empty rectangle
    rect = largest_rectangle(content_map)

    elapsed = round(time.time() - t0, 2)

    if rect:
        rect["content_desc"] = content_desc
        rect["content_map"] = content_map
        rect["utterance"] = utterance
        rect["elapsed_s"] = elapsed

    return rect
