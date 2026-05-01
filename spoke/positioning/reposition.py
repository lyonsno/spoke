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
    "You convert user requests about overlay positioning into content descriptions.\n\n"
    "The user wants to reposition an overlay on their screen. They will describe "
    "what they want to keep visible, or what they don't want blocked, or where "
    "they want the overlay to go.\n\n"
    "Extract the type of content they want to keep visible and output it as a "
    "short noun phrase. If the user does not specify any particular content, "
    "output 'important content' as the default.\n\n"
    "Examples:\n"
    "- 'stop blocking my code' → 'code'\n"
    "- 'move out of the way of the terminal text' → 'terminal text'\n"
    "- 'get off the graph' → 'graph'\n"
    "- 'can you occupy the top without covering the article' → 'article text'\n"
    "- 'move so I can see the contributor stats' → 'contributor statistics'\n"
    "- 'can you please move out of the way' → 'important content'\n"
    "- 'just move' → 'important content'\n\n"
    "Output ONLY the noun phrase. Nothing else."
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


def resolve_intent(utterance: str) -> str:
    """Layer 1: Convert user utterance to content description noun phrase."""
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
    content_desc = resp.json()["choices"][0]["message"]["content"].strip().strip("\"'")
    return content_desc


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
    content_desc = resolve_intent(utterance)

    # Layer 2: detect content
    content_map = detect_content(screenshot, content_desc)

    # Layer 3: find largest empty rectangle
    rect = largest_rectangle(content_map)

    elapsed = round(time.time() - t0, 2)

    if rect:
        rect["content_desc"] = content_desc
        rect["content_map"] = content_map
        rect["elapsed_s"] = elapsed

    return rect
