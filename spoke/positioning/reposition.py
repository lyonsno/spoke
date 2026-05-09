"""Semantic overlay repositioning pipeline.

Two layers:
1. Intent resolve: user utterance → content description noun phrase
2. Content detection: 4x4 grid binary classification → content map
3. (Deterministic) Largest inscribed rectangle from empty cells

Returns a CGRect-compatible dict for the overlay's new position.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import concurrent.futures
import functools
import io
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from .grid_overlay import draw_grid, ROW_LABELS, COLS, ROWS

# Default grid for positioning
POS_ROWS = 6
POS_COLS = 6
_POSITIONING_UTTERANCE_ID = contextvars.ContextVar(
    "spoke_positioning_utterance_id",
    default=None,
)

INTENT_SYSTEM = (
    "Your only job is to classify overlay positioning requests.\n\n"
    "The user wants to reposition an overlay on their screen. There are only three modes:\n\n"
    "1. AVOID mode: The user wants the overlay to move AWAY from certain content.\n"
    "   Output only: AVOID: <content noun phrase>\n\n"
    "2. TARGET mode: The user wants the overlay to go to a specific SPATIAL region\n"
    "   (no screen content needs to be identified — just a direction or area).\n"
    "   Output only: TARGET: <short region description>\n\n"
    "3. FIND mode: The user wants the overlay to go TO where certain content IS.\n"
    "   The content must be identified on screen visually.\n"
    "   Output only: FIND: <content noun phrase>\n\n"
    "AVOID = move away from content. FIND = move toward content. TARGET = go to a region.\n\n"
    "Examples:\n"
    "- 'stop blocking my code' → AVOID: code\n"
    "- 'move out of the way of the terminal text' → AVOID: terminal text\n"
    "- 'get off the graph' → AVOID: graph\n"
    "- 'avoid the regions with blue or purple text' → AVOID: blue, purple, or lavender text\n"
    "- 'can you please move out of the way' → AVOID: important content\n"
    "- 'just move' → AVOID: important content\n"
    "- 'go to the top right' → TARGET: top right\n"
    "- 'occupy the middle of the screen' → TARGET: center\n"
    "- 'position yourself in the middle' → TARGET: center\n"
    "- 'move to the bottom' → TARGET: bottom\n"
    "- 'can you go to the left side' → TARGET: left side\n"
    "- 'stay in the top half' → TARGET: top half\n"
    "- 'go where the code is' → FIND: code\n"
    "- 'position yourself over the terminal' → FIND: terminal\n"
    "- 'move onto the graph' → FIND: graph\n"
    "- 'cover the sidebar' → FIND: sidebar\n"
    "- 'sit on top of the dark pane' → FIND: dark pane\n\n"
    "Output ONLY the mode prefix and value. Nothing else. No explanation."
)

TARGET_SYSTEM = (
    "You are a screen layout system.\n\n"
    "The screen is divided into a 6×6 grid:\n"
    "  Rows: A (top) to F (bottom)\n"
    "  Columns: 1 (left) to 6 (right)\n\n"
    "  A1 A2 A3 A4 A5 A6   ← top of screen\n"
    "  B1 B2 B3 B4 B5 B6\n"
    "  C1 C2 C3 C4 C5 C6\n"
    "  D1 D2 D3 D4 D5 D6\n"
    "  E1 E2 E3 E4 E5 E6\n"
    "  F1 F2 F3 F4 F5 F6   ← bottom of screen\n\n"
    "You will receive a description of where an overlay should be placed. "
    "Mark each cell YES if the overlay should occupy it, NO if not.\n\n"
    "Output exactly 36 lines:\n"
    "A1: YES or NO\nA2: YES or NO\n...\nF6: YES or NO\n\n"
    "Examples:\n"
    "- 'center' → C3:YES C4:YES D3:YES D4:YES, rest NO\n"
    "- 'top right corner' → A5:YES A6:YES B5:YES B6:YES, rest NO\n"
    "- 'bottom third' → E1-E6:YES F1-F6:YES, rest NO\n"
    "- 'middle third horizontally' → all rows columns 3-4 YES, rest NO\n"
    "- 'left side' → columns 1-3 YES for all rows, rest NO\n\n"
    "You MUST output all 36 cells."
)

DETECT_SYSTEM = (
    "You are a content detection system.\n\n"
    "You will be shown an image of a screen divided into a 6x6 grid "
    "(rows A-F, columns 1-6).\n\n"
    "IMPORTANT: The gray text labels (A1, A2, B1, etc.) and gray grid lines "
    "are reference markers drawn ON TOP of the screen image. They are NOT part of "
    "the screen content. Ignore them completely when deciding YES or NO. Only look "
    "at what is underneath them.\n\n"
    "You will also receive a description of a type of content. For each cell "
    "in the grid, determine whether the UNDERLYING screen content (not the gray "
    "labels) contains that type of content.\n\n"
    "Output exactly 36 lines, one per cell, in order:\n"
    "A1: YES or NO\n"
    "A2: YES or NO\n"
    "...\n"
    "F6: YES or NO\n\n"
    "YES = this cell contains that type of content\n"
    "NO = this cell does not\n\n"
    "You MUST output all 36 cells."
)


def _get_api_url():
    """Get the OMLX/Grapheus endpoint URL."""
    return os.environ.get("SPOKE_VLM_URL", "http://localhost:8090/v1/chat/completions")


def _get_api_key():
    """Get the API key."""
    return os.environ.get("OMLX_SERVER_API_KEY", "1234")


def _new_positioning_utterance_id() -> str:
    return f"positioning-{uuid.uuid4().hex[:12]}"


@contextlib.contextmanager
def positioning_utterance_scope(utterance_id: str | int | None = None):
    """Group all positioning model calls from one user action for Grapheus."""
    value = str(utterance_id) if utterance_id is not None else _new_positioning_utterance_id()
    token = _POSITIONING_UTTERANCE_ID.set(value)
    try:
        yield value
    finally:
        _POSITIONING_UTTERANCE_ID.reset(token)


def _current_positioning_utterance_id(utterance_id: str | int | None = None) -> str:
    if utterance_id is not None:
        return str(utterance_id)
    current = _POSITIONING_UTTERANCE_ID.get()
    if current:
        return str(current)
    return _new_positioning_utterance_id()


def _api_headers(
    step: str = "positioning",
    *,
    mode: str | None = None,
    iteration: int | None = None,
    utterance_id: str | int | None = None,
) -> dict[str, str]:
    """Standard headers for positioning API calls, including Grapheus X-Spoke-* metadata."""
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "X-Spoke-Pathway": "positioning",
        "X-Spoke-Utterance-ID": _current_positioning_utterance_id(utterance_id),
        "X-Spoke-Step": step,
    }
    if mode:
        headers["X-Spoke-Positioning-Mode"] = mode
    if iteration is not None:
        headers["X-Spoke-Positioning-Iteration"] = str(iteration)
    return headers


def _with_positioning_utterance_scope(fn):
    """Ensure top-level positioning pipelines share one Grapheus utterance id."""
    @functools.wraps(fn)
    def wrapped(utterance, *args, **kwargs):
        active_id = _POSITIONING_UTTERANCE_ID.get()
        if active_id:
            result = fn(utterance, *args, **kwargs)
            utterance_id = str(active_id)
        else:
            with positioning_utterance_scope() as utterance_id:
                result = fn(utterance, *args, **kwargs)
        if isinstance(result, dict):
            result.setdefault("_positioning_utterance_id", utterance_id)
        return result

    wrapped._last_debug = getattr(fn, "_last_debug", [])
    return wrapped


def _detect_thinking_enabled() -> bool:
    """Whether to enable thinking for content detection calls."""
    return os.environ.get("SPOKE_POSITIONING_THINKING", "0") == "1"


def _detect_max_tokens() -> int:
    """Max tokens for content detection: 256 without thinking, 16384 with."""
    if _detect_thinking_enabled():
        return 16384
    # 6×6 grid × 4 tokens per cell worst case = 144, rounded up with headroom
    return 256


def _sampling_params(max_tokens: int | None = None) -> dict:
    """Centralized sampling parameters following Qwen 3.6 official presets.

    Thinking mode (general): temp=1.0, top_p=0.95, presence_penalty=1.5
    Non-thinking mode:       temp=0.7, top_p=0.80, presence_penalty=1.5
    """
    thinking = _detect_thinking_enabled()
    params = {
        "temperature": 1.0 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.80,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    else:
        params["max_tokens"] = 16384 if thinking else 256
    return params


def _image_scale() -> float:
    """Image scale factor from env, default 0.5."""
    return float(os.environ.get("SPOKE_POSITIONING_IMAGE_SCALE", "0.5"))


def _encode_image(img: Image.Image, scale: float | None = None) -> str:
    """Encode PIL image as base64 PNG, optionally downscaling."""
    if scale is None:
        scale = _image_scale()
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def resolve_intent(utterance: str) -> dict:
    """Layer 1: Convert user utterance to structured positioning intent.

    Returns {"mode": "avoid", "content_desc": "..."} or
            {"mode": "target", "region": "...", "rect": {...}}.
    """
    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("intent"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": utterance},
            ],
            **_sampling_params(max_tokens=64),
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip().strip("\"'")

    if raw.upper().startswith("TARGET:"):
        target_desc = raw.split(":", 1)[1].strip()
        return {"mode": "target", "target_desc": target_desc}
    if raw.upper().startswith("FIND:"):
        content_desc = raw.split(":", 1)[1].strip()
        return {"mode": "find", "content_desc": content_desc}
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
        headers=_api_headers("detect"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": DETECT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": content_desc},
                ]},
            ],
            **_sampling_params(),
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
            # Default to True (content present) on parse failure — safe for AVOID
            # mode (won't place overlay on unparseable cells) and neutral for FIND
            # mode (caller inverts, so unparseable cells won't attract the overlay).
            result[cell] = match.group(1) == "YES" if match else True
    return result


def target_cells(target_desc: str) -> dict[str, bool]:
    """Layer 2b: For TARGET mode — model outputs which cells the overlay should occupy.

    Same YES/NO contract as detect_content, but text-only (no image).
    YES = overlay should occupy this cell.
    Returns dict mapping cell name to bool.
    """
    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("target"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": TARGET_SYSTEM},
                {"role": "user", "content": target_desc},
            ],
            **_sampling_params(),
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


def largest_rectangle_target(target_map: dict[str, bool]) -> dict | None:
    """Find the largest rectangle in cells marked YES (overlay should occupy).

    largest_rectangle treats NO (False) cells as available space.
    For targeting, YES (True) cells are where we want to be.
    Invert so YES→False (="no content here, available") and the
    rectangle finder picks them.
    """
    inverted = {k: not v for k, v in target_map.items()}
    return largest_rectangle(inverted)


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


# ── Two-step center+size pipeline ──

COARSE_REGIONS = {
    "TL": (1/6, 1/6),   "TC": (1/2, 1/6),   "TR": (5/6, 1/6),
    "ML": (1/6, 1/2),   "MC": (1/2, 1/2),   "MR": (5/6, 1/2),
    "BL": (1/6, 5/6),   "BC": (1/2, 5/6),   "BR": (5/6, 5/6),
}

CENTER_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The screen is divided into a 3×3 grid of regions:\n"
    "  TL  TC  TR   ← top\n"
    "  ML  MC  MR   ← middle\n"
    "  BL  BC  BR   ← bottom\n\n"
    "The image shows the current screen. A red dashed outline shows where the "
    "overlay is currently positioned.\n\n"
    "Based on the user's request and what you see on screen, pick the ONE region "
    "where the CENTER of the overlay should move to.\n\n"
    "Output ONLY the region code (e.g. TR). Nothing else."
)

SIZE_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The image shows the current screen. A red dashed outline shows the overlay's "
    "current position and size.\n\n"
    "The overlay center has been decided. Now decide how wide and tall the overlay "
    "should be, as a percentage of the full screen.\n\n"
    "Consider the user's request and what's visible on screen. If they want to "
    "cover a region, make it big enough to cover it. If they want to avoid content, "
    "keep it compact. 'Full height' means 100. 'Full width' means 100.\n\n"
    "Output ONLY two numbers on one line: width_pct height_pct\n"
    "Each is a percentage of the full screen dimension (1-100).\n"
    "Examples:\n"
    "  40 15   (40% of screen width, 15% of screen height — compact)\n"
    "  30 100  (narrow column, full height)\n"
    "  100 30  (full width, short strip)\n"
    "  50 50   (half the screen in each dimension)\n"
    "  20 20   (small overlay)\n\n"
    "Output ONLY the two numbers. Nothing else."
)

BBOX_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The image shows the current screen. A red dashed outline shows the overlay's "
    "current position and size.\n\n"
    "Based on the user's request, output the pixel coordinates where the overlay "
    "should be placed.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors (e.g. 'write' for 'right', 'senner' for 'center'). Interpret "
    "phonetically when a literal reading doesn't make sense.\n\n"
    "SPEED IS CRITICAL. The user is waiting for the overlay to move.\n"
    "- For general spatial requests ('top right', 'left side', 'center'), estimate "
    "quickly from the region name. Do not analyze screen contents.\n"
    "- Only examine screen contents when the user explicitly references something "
    "visible ('cover the terminal', 'avoid the graph', 'over the blue text').\n"
    "- Even then, estimate roughly. A fast approximate answer is better than a "
    "slow precise one.\n"
    "- If the request is ambiguous, pick the most likely interpretation and go. "
    "Do not deliberate.\n\n"
    "Output ONLY four numbers on one line: center_x center_y width height\n"
    "All values are in pixels. (0,0) is the top-left corner of the screen.\n"
    "center_x, center_y = center point of the overlay.\n"
    "width, height = overlay size.\n\n"
    "Examples for a 1920×1080 screen:\n"
    "  960 540 1920 1080  (full screen, centered)\n"
    "  1440 540 960 1080  (right half, full height)\n"
    "  650 350 500 400    (a box in the upper-center area)\n"
    "  150 540 300 1080   (narrow column on the left, full height)\n"
    "  960 25 1920 50     (full-width thin strip at the very top)\n\n"
    "Output ONLY the four numbers. Nothing else."
)


CENTER_ONLY_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The image shows the current screen. A red dashed outline shows the overlay's "
    "current position and size.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically when a literal reading doesn't make sense.\n\n"
    "SPEED IS CRITICAL.\n"
    "- For general spatial requests, estimate quickly. Do not analyze screen contents.\n"
    "- Only examine screen contents when the user references something visible.\n"
    "- If the request is ambiguous, pick the most likely interpretation and go.\n\n"
    "Does this request require moving the overlay's center?\n"
    "- If YES: output the new center as two numbers: center_x center_y\n"
    "- If NO (the user only wants to resize): output NONE\n\n"
    "All values are in pixels. (0,0) is the top-left corner of the screen.\n\n"
    "Output ONLY 'center_x center_y' or 'NONE'. Nothing else."
)

RESIZE_SYSTEM = (
    "You are a screen layout system. The user just repositioned an overlay.\n\n"
    "The overlay's center has been placed. Now decide if the size should change.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically when a literal reading doesn't make sense.\n\n"
    "Does this request require changing the overlay's size?\n"
    "- If YES: output the new size as two numbers: width height (in pixels)\n"
    "- If NO (the user only wanted to move it): output NONE\n\n"
    "Consider: if the user said 'cover X' or 'fill the right side', the size "
    "should match that region. If they said 'move to the top right' or 'go left', "
    "keep the current size.\n\n"
    "Output ONLY 'width height' or 'NONE'. Nothing else."
)


def _pick_center_only(screenshot_b64: str, utterance: str,
                      screen_w: int, screen_h: int,
                      current_overlay: dict | None = None,
                      bearing: str | None = None) -> tuple[int, int] | None:
    """Step 1: Pick center point only. Returns (cx, cy) or None if no move needed."""
    cur_x = int((current_overlay or {}).get("x", 0.3) * screen_w)
    cur_y = int((current_overlay or {}).get("y", 0.3) * screen_h)
    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)
    cur_cx = cur_x + cur_w // 2
    cur_cy = cur_y + cur_h // 2

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Current overlay: center=({cur_cx}, {cur_cy}) width={cur_w} height={cur_h}"
    )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("center"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": CENTER_ONLY_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()

    if "NONE" in raw.upper():
        return None
    parts = raw.split()
    try:
        cx = max(0, min(screen_w, int(float(parts[0]))))
        cy = max(0, min(screen_h, int(float(parts[1]))))
        return cx, cy
    except (ValueError, IndexError):
        return None


def _pick_resize(screenshot_b64: str, utterance: str,
                 screen_w: int, screen_h: int,
                 new_center: tuple[int, int],
                 current_overlay: dict | None = None,
                 bearing: str | None = None) -> tuple[int, int] | None:
    """Step 2: Pick new size. Returns (w, h) or None if no resize needed."""
    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Overlay center just moved to: ({new_center[0]}, {new_center[1]})\n"
        f"Current overlay size: width={cur_w} height={cur_h}"
    )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("resize"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": RESIZE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()

    if "NONE" in raw.upper():
        return None
    parts = raw.split()
    try:
        w = max(1, min(screen_w, int(float(parts[0]))))
        h = max(1, min(screen_h, int(float(parts[1]))))
        return w, h
    except (ValueError, IndexError):
        return None


def reposition_centersize(
    utterance: str,
    screenshot: Image.Image,
    current_overlay: dict | None = None,
    screen_w: int = 1920,
    screen_h: int = 1080,
    on_step: "callable | None" = None,
    bearing: str | None = None,
) -> dict | None:
    """Two-phase pipeline: center first (overlay moves), then resize.

    Returns dict with x, y, width, height as fractions of screen (0-1),
    plus _screenshot_b64 for bearing update. Calls on_step after center
    resolves so the overlay can move before resize completes.
    """
    reposition_centersize._last_debug = []
    t0 = time.time()

    def _report():
        if on_step:
            on_step(list(reposition_centersize._last_debug))

    if current_overlay is None:
        current_overlay = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}

    if bearing:
        reposition_centersize._last_debug.append(f"Bearing: {bearing[:200]}")
        _report()

    # Prepare image with overlay outline
    annotated = _draw_overlay_outline(screenshot, current_overlay)
    screenshot_b64 = _encode_image(annotated)

    cur_w = int(current_overlay["width"] * screen_w)
    cur_h = int(current_overlay["height"] * screen_h)
    cur_cx = int(current_overlay["x"] * screen_w) + cur_w // 2
    cur_cy = int(current_overlay["y"] * screen_h) + cur_h // 2

    # Step 1: center
    center_result = _pick_center_only(
        screenshot_b64, utterance, screen_w, screen_h,
        current_overlay, bearing,
    )
    t_center = time.time()

    if center_result is not None:
        cx, cy = center_result
        reposition_centersize._last_debug.append(
            f"Center: ({cx}, {cy}) ({t_center - t0:.1f}s)"
        )
    else:
        cx, cy = cur_cx, cur_cy
        reposition_centersize._last_debug.append(
            f"Center: unchanged ({t_center - t0:.1f}s)"
        )

    # Emit intermediate result so overlay can move NOW
    bx = max(0, min(screen_w - cur_w, cx - cur_w // 2))
    by = max(0, min(screen_h - cur_h, cy - cur_h // 2))
    intermediate = {
        "x": bx / screen_w,
        "y": by / screen_h,
        "width": cur_w / screen_w,
        "height": cur_h / screen_h,
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": round(t_center - t0, 2),
        "_intermediate": True,
    }
    _report()

    # Signal the caller to move the overlay now
    if on_step:
        on_step(list(reposition_centersize._last_debug), intermediate)

    # Step 2: resize
    resize_result = _pick_resize(
        screenshot_b64, utterance, screen_w, screen_h,
        (cx, cy), current_overlay, bearing,
    )
    t_resize = time.time()

    if resize_result is not None:
        new_w, new_h = resize_result
        reposition_centersize._last_debug.append(
            f"Resize: {new_w}×{new_h}px ({t_resize - t_center:.1f}s)"
        )
    else:
        new_w, new_h = cur_w, cur_h
        reposition_centersize._last_debug.append(
            f"Resize: unchanged ({t_resize - t_center:.1f}s)"
        )

    # Final position with new size
    bx = max(0, min(screen_w - new_w, cx - new_w // 2))
    by = max(0, min(screen_h - new_h, cy - new_h // 2))

    elapsed = round(time.time() - t0, 2)
    reposition_centersize._last_debug.append(f"Total: {elapsed:.1f}s")
    _report()

    return {
        "x": bx / screen_w,
        "y": by / screen_h,
        "width": new_w / screen_w,
        "height": new_h / screen_h,
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": elapsed,
        "_screenshot_b64": screenshot_b64,
    }


BEARING_SYSTEM = (
    "You are an operator context system. You just repositioned an overlay on the "
    "user's screen. Based on the screenshot, the user's request, and the previous "
    "bearing (if any), update the bearing.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically.\n\n"
    "The bearing is a compressed snapshot of what the user is doing and what the "
    "overlay means to them right now. It helps future repositioning calls make "
    "better decisions faster.\n\n"
    "Output EXACTLY this structure, one short phrase per field:\n"
    "screen_layout: <what's on screen — panes, apps, content areas>\n"
    "overlay_role: <what the overlay is being used for right now>\n"
    "default_size: <WxH pixels — the 'normal' size when not covering something specific>\n"
    "default_position: <where the overlay naturally sits when not directed elsewhere>\n"
    "user_tendency: <how the user tends to use repositioning — what they avoid, prefer>\n"
    "last_action: <what the user just asked for and what happened>\n\n"
    "Keep each field to one short phrase. Do not elaborate. Do not add fields."
)


def update_bearing(
    screenshot_b64: str,
    utterance: str,
    bbox_result: dict,
    previous_bearing: str | None,
    screen_w: int,
    screen_h: int,
    *,
    utterance_id: str | int | None = None,
) -> str:
    """Background call: update the operator bearing after repositioning.

    Runs with thinking on for richness. Returns the new bearing as a string.
    """
    bx = int(bbox_result["x"] * screen_w)
    by = int(bbox_result["y"] * screen_h)
    bw = int(bbox_result["width"] * screen_w)
    bh = int(bbox_result["height"] * screen_h)

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Overlay was placed at: x={bx} y={by} width={bw} height={bh}\n"
    )
    if previous_bearing:
        user_text += f"\nPrevious bearing:\n{previous_bearing}\n"
    else:
        user_text += "\nNo previous bearing (first positioning run).\n"

    # Always use thinking preset for bearing updates
    params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("bearing", utterance_id=utterance_id),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": BEARING_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **params,
        },
        timeout=120,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content", "").strip()

    # The bearing is the content — structured fields
    return content


# ── Grid-point pipeline ──
# 4×4 grid → 3×3 interior intersections = 9 candidate center points

GRIDPOINT_ROWS = 3
GRIDPOINT_COLS = 3
GRIDPOINT_LABELS = [
    f"{r}{c}"
    for r in ["A", "B", "C"]
    for c in ["1", "2", "3"]
]

GRIDPOINT_LATTICES = {
    "A": (0.25, 0.50, 0.75),
    "B": (0.3125, 0.5625, 0.8125),
}


def _gridpoint_coords(label: str, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Convert a grid-point label (e.g. 'B2') to pixel center coordinates."""
    return _gridpoint_lattice_coords("A", label, screen_w, screen_h)


def _gridpoint_lattice_coords(
    lattice: str,
    label: str,
    screen_w: int,
    screen_h: int,
) -> tuple[int, int]:
    """Convert a grid-point label through a named offset lattice."""

    row = ord(label[0]) - ord("A")  # 0-2
    col = int(label[1]) - 1          # 0-2
    positions = GRIDPOINT_LATTICES.get(lattice, GRIDPOINT_LATTICES["A"])
    cx = int(screen_w * positions[col])
    cy = int(screen_h * positions[row])
    return cx, cy


def _draw_grid_points(img: Image.Image, lattice: str = "A") -> Image.Image:
    """Draw 3×3 lattice points with labels on the image."""

    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    marker_r = max(4, min(w, h) // 150)

    try:
        from PIL import ImageFont
        font_size = max(10, min(w, h) // 80)
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except (OSError, IOError):
        font = None

    for label in GRIDPOINT_LABELS:
        cx, cy = _gridpoint_lattice_coords(lattice, label, w, h)
        # Crosshair
        cross_len = marker_r * 2
        draw.line([(cx - cross_len, cy), (cx + cross_len, cy)],
                  fill=(200, 200, 200), width=2)
        draw.line([(cx, cy - cross_len), (cx, cy + cross_len)],
                  fill=(200, 200, 200), width=2)
        # Label
        draw.text((cx + marker_r + 2, cy - marker_r - 2), label,
                  fill=(200, 200, 200), font=font)

    return img


def _positioning_gridpoint_ensemble_enabled() -> bool:
    return os.environ.get("SPOKE_POSITIONING_GRIDPOINT_ENSEMBLE", "0") == "1"


def _gridpoint_ensemble_workers() -> int:
    raw = os.environ.get("SPOKE_POSITIONING_GRIDPOINT_ENSEMBLE_WORKERS", "4")
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 4


def _gridpoint_ensemble_lattice_sequence(workers: int) -> list[str]:
    names = list(GRIDPOINT_LATTICES.keys())
    return [names[index % len(names)] for index in range(max(1, workers))]


def _parse_gridpoint_label(raw: str) -> str | None:
    normalized = raw.strip().upper()
    for label in GRIDPOINT_LABELS:
        if label in normalized:
            return label
    return None


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _aggregate_gridpoint_ensemble_samples(
    samples: list[dict],
    *,
    screen_w: int,
    screen_h: int,
) -> dict | None:
    """Aggregate lattice-tagged grid-point labels into a coordinate estimate."""

    successes = []
    for sample in samples:
        label = sample.get("label")
        lattice = str(sample.get("lattice") or "A")
        if label not in GRIDPOINT_LABELS:
            continue
        cx, cy = _gridpoint_lattice_coords(lattice, label, screen_w, screen_h)
        successes.append({
            "lattice": lattice,
            "label": label,
            "center_x": cx,
            "center_y": cy,
        })
    if not successes:
        return None

    xs = [float(sample["center_x"]) for sample in successes]
    ys = [float(sample["center_y"]) for sample in successes]
    labels = [str(sample["label"]) for sample in successes]
    lattices = [str(sample["lattice"]) for sample in successes]
    winner_count = max(labels.count(label) for label in set(labels))
    total = max(1, len(samples))
    schema_success_fraction = len(successes) / total
    winner_fraction = winner_count / max(1, len(successes))
    spread_x = (max(xs) - min(xs)) / max(1, screen_w)
    spread_y = (max(ys) - min(ys)) / max(1, screen_h)
    spread_confidence = max(0.0, 1.0 - max(spread_x, spread_y))
    confidence = min(schema_success_fraction, winner_fraction, spread_confidence)

    return {
        "center_x": _median(xs),
        "center_y": _median(ys),
        "confidence": confidence,
        "schema_success_fraction": schema_success_fraction,
        "winner_fraction": winner_fraction,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "labels": labels,
        "lattices": lattices,
        "samples": successes,
    }


GRIDPOINT_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The image shows the current screen with 9 labeled intersection points "
    "(gray crosshairs with labels A1-C3). A red dashed outline shows the "
    "overlay's current position.\n\n"
    "The grid layout:\n"
    "  A1  A2  A3   ← upper row (25% from top)\n"
    "  B1  B2  B3   ← middle row (50%)\n"
    "  C1  C2  C3   ← lower row (75% from top)\n"
    "Columns: 1=25% from left, 2=center, 3=75% from left.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically.\n\n"
    "Pick the ONE point where the overlay's center should move to.\n\n"
    "Output ONLY the point label (e.g. A3). Nothing else."
)

GRIDPOINT_ENSEMBLE_SYSTEM = (
    "You are a screen layout system. The user wants to reposition an overlay.\n\n"
    "The image shows the current screen with 9 labeled candidate center points "
    "(gray crosshairs with labels A1-C3). A red dashed outline shows the "
    "overlay's current position.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically.\n\n"
    "Pick the ONE visible point where the overlay's center should move to. "
    "Use the labels drawn in this image; do not assume every image uses the "
    "same exact point positions.\n\n"
    "Output ONLY the point label (e.g. A3). Nothing else."
)

GRIDPOINT_RESIZE_SYSTEM = (
    "You are a screen layout system. An overlay was just repositioned.\n\n"
    "The image shows the current screen. The overlay's center has been placed "
    "at the position shown.\n\n"
    "The user's request was transcribed from speech and may contain phonetic "
    "errors. Interpret phonetically.\n\n"
    "Based on the user's request and what's visible on screen, decide the "
    "overlay's size in pixels.\n\n"
    "If the user only asked to move (not resize), output NONE.\n"
    "Otherwise output: width height (in pixels)\n\n"
    "Output ONLY 'width height' or 'NONE'. Nothing else."
)

SUITABILITY_AUDIT_SYSTEM = (
    "You are auditing a proposed overlay rectangle.\n\n"
    "The red dashed outline is the current proposed overlay. The user request "
    "is the desired final result.\n\n"
    "Decide whether this outline already satisfies the request in the visible "
    "screen context. If it does not, mark whether the failure is about position, "
    "size, or both.\n\n"
    "Output a valid JSON object and nothing else:\n"
    '{"done":true_or_false,'
    '"needs_position":true_or_false,'
    '"needs_size":true_or_false,'
    '"reason":"short reason"}'
)

CENTER_AUDIT_SYSTEM = (
    "You are correcting the center of a proposed overlay rectangle.\n\n"
    "The red dashed outline is the current proposed overlay. The user request "
    "is the desired final result.\n\n"
    "If the user text includes a suitability diagnosis, treat that diagnosis "
    "as the concrete problem you are repairing. Do not switch to a different "
    "target or return KEEP merely because the outline is directionally close.\n\n"
    "Judge only the overlay center. Choose the center that best serves the user "
    "request, assuming width and height will be adjusted optimally afterward to "
    "satisfy the same request. Do not preserve the current size if a different "
    "eventual size would change the best center.\n\n"
    "Output a valid JSON object and nothing else:\n"
    '{"center_x":"KEEP or <center_x_px>",'
    '"center_y":"KEEP or <center_y_px>",'
    '"reason":"short reason"}'
)

SIZE_AUDIT_SYSTEM = (
    "You are correcting the size of a proposed overlay rectangle.\n\n"
    "The red dashed outline is the current proposed overlay. The user request "
    "is the desired final result.\n\n"
    "If the user text includes a suitability diagnosis, treat that diagnosis "
    "as the concrete problem you are repairing. Do not switch to a different "
    "target or return KEEP merely because the outline is directionally close.\n\n"
    "Judge only the overlay size. Choose the width and height the overlay should "
    "have in the final placement, assuming the center will be adjusted optimally "
    "to satisfy the same request. Do not preserve the current size if the final "
    "rectangle should be larger, smaller, wider, narrower, taller, or shorter.\n\n"
    "Output a valid JSON object and nothing else:\n"
    '{"width":"KEEP or <width_px>",'
    '"height":"KEEP or <height_px>",'
    '"reason":"short reason"}'
)


def _pick_gridpoint(screenshot_b64: str, utterance: str,
                    screen_w: int, screen_h: int,
                    current_overlay: dict | None = None,
                    bearing: str | None = None) -> str | None:
    """Pick a grid intersection point. Returns label like 'B2' or None on parse failure."""
    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)
    cur_cx = int((current_overlay or {}).get("x", 0.3) * screen_w) + cur_w // 2
    cur_cy = int((current_overlay or {}).get("y", 0.3) * screen_h) + cur_h // 2

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Current overlay: center=({cur_cx}, {cur_cy}) width={cur_w} height={cur_h}"
    )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("gridpoint"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": GRIDPOINT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=8),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip().upper()

    for label in GRIDPOINT_LABELS:
        if label in raw:
            return label
    # Store raw for diagnostics
    _pick_gridpoint._last_raw = raw
    return None


def _pick_gridpoint_on_lattice(
    screenshot_b64: str,
    utterance: str,
    screen_w: int,
    screen_h: int,
    current_overlay: dict | None = None,
    bearing: str | None = None,
    *,
    lattice: str,
    worker_index: int,
) -> dict:
    """Pick one labeled point on a specific lattice image."""

    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)
    cur_cx = int((current_overlay or {}).get("x", 0.3) * screen_w) + cur_w // 2
    cur_cy = int((current_overlay or {}).get("y", 0.3) * screen_h) + cur_h // 2
    positions = GRIDPOINT_LATTICES.get(lattice, GRIDPOINT_LATTICES["A"])

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Current overlay: center=({cur_cx}, {cur_cy}) width={cur_w} height={cur_h}\n"
        f"Grid lattice: {lattice}\n"
        f"Rows and columns are at normalized positions: {positions}\n"
        "Choose one visible label from this image."
    )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers(
            f"gridpoint-ensemble-{lattice}",
            mode="gridpoint-ensemble",
            iteration=worker_index,
        ),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": GRIDPOINT_ENSEMBLE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=8),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()
    label = _parse_gridpoint_label(raw)
    return {"lattice": lattice, "label": label, "raw": raw, "worker": worker_index}


def _pick_gridpoint_ensemble(
    screenshot: Image.Image,
    utterance: str,
    screen_w: int,
    screen_h: int,
    current_overlay: dict | None = None,
    bearing: str | None = None,
) -> dict | None:
    """Run offset-grid point pickers in parallel and aggregate coordinates."""

    worker_count = _gridpoint_ensemble_workers()
    lattices = _gridpoint_ensemble_lattice_sequence(worker_count)
    encoded_by_lattice: dict[str, str] = {}
    for lattice in sorted(set(lattices)):
        annotated = _draw_overlay_outline(screenshot, current_overlay)
        annotated = _draw_grid_points(annotated, lattice=lattice)
        encoded_by_lattice[lattice] = _encode_image(annotated)

    samples: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for index, lattice in enumerate(lattices, start=1):
            futures.append(
                executor.submit(
                    _pick_gridpoint_on_lattice,
                    encoded_by_lattice[lattice],
                    utterance,
                    screen_w,
                    screen_h,
                    current_overlay,
                    bearing,
                    lattice=lattice,
                    worker_index=index,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                samples.append(future.result())
            except Exception as exc:
                samples.append({"lattice": "?", "label": None, "raw": repr(exc), "worker": -1})

    _pick_gridpoint_ensemble._last_samples = samples
    return _aggregate_gridpoint_ensemble_samples(samples, screen_w=screen_w, screen_h=screen_h)


def _pick_gridpoint_resize(screenshot_b64: str, utterance: str,
                           screen_w: int, screen_h: int,
                           new_center: tuple[int, int],
                           current_overlay: dict | None = None,
                           bearing: str | None = None) -> tuple[int, int] | None:
    """Pick new size after grid-point center. Returns (w, h) or None."""
    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Overlay center placed at: ({new_center[0]}, {new_center[1]})\n"
        f"Current overlay size: width={cur_w} height={cur_h}"
    )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"

    # Resize always uses thinking for precision
    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("gridresize"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": GRIDPOINT_RESIZE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()

    if "NONE" in raw.upper():
        return None
    parts = raw.split()
    try:
        w = max(1, min(screen_w, int(float(parts[0]))))
        h = max(1, min(screen_h, int(float(parts[1]))))
        return w, h
    except (ValueError, IndexError):
        return None


def _extract_json_object(raw: str) -> dict | None:
    json_text = raw.strip()
    if json_text.startswith("```"):
        json_text = re.sub(r"^```(?:json)?\s*", "", json_text, flags=re.IGNORECASE)
        json_text = re.sub(r"\s*```$", "", json_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", json_text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _coerce_bool(value, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return default


def _coerce_pixel_or_keep(value, *, limit: int, minimum: int) -> int | str:
    if isinstance(value, str) and value.strip().upper().startswith("KEEP"):
        return "KEEP"
    if isinstance(value, (int, float)):
        parsed = int(float(value))
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return "KEEP"
        parsed = int(float(match.group(0)))
    else:
        return "KEEP"
    return max(minimum, min(limit, parsed))


def _parse_suitability_audit_response(raw: str) -> dict[str, bool | str]:
    """Parse a suitability response into a bounded correction decision."""

    result: dict[str, bool | str] = {
        "done": False,
        "needs_position": True,
        "needs_size": True,
        "reason": "",
    }
    payload = _extract_json_object(raw)
    if not payload:
        return result
    result["done"] = _coerce_bool(payload.get("done"), default=False)
    result["needs_position"] = _coerce_bool(payload.get("needs_position"), default=True)
    result["needs_size"] = _coerce_bool(payload.get("needs_size"), default=True)
    reason = payload.get("reason", "")
    if reason is not None:
        result["reason"] = str(reason).strip()
    return result


def _parse_center_audit_response(
    raw: str,
    *,
    screen_w: int,
    screen_h: int,
) -> dict[str, int | str]:
    """Parse a center repair response into clamped pixel coordinates."""

    result: dict[str, int | str] = {"center_x": "KEEP", "center_y": "KEEP", "reason": ""}
    payload = _extract_json_object(raw)
    if not payload:
        return result
    result["center_x"] = _coerce_pixel_or_keep(
        payload.get("center_x"),
        limit=screen_w,
        minimum=0,
    )
    result["center_y"] = _coerce_pixel_or_keep(
        payload.get("center_y"),
        limit=screen_h,
        minimum=0,
    )
    reason = payload.get("reason", "")
    if reason is not None:
        result["reason"] = str(reason).strip()
    return result


def _parse_size_audit_response(
    raw: str,
    *,
    screen_w: int,
    screen_h: int,
) -> dict[str, int | str]:
    """Parse a size repair response into clamped pixel dimensions."""

    result: dict[str, int | str] = {"width": "KEEP", "height": "KEEP", "reason": ""}
    payload = _extract_json_object(raw)
    if not payload:
        return result
    result["width"] = _coerce_pixel_or_keep(payload.get("width"), limit=screen_w, minimum=1)
    result["height"] = _coerce_pixel_or_keep(payload.get("height"), limit=screen_h, minimum=1)
    reason = payload.get("reason", "")
    if reason is not None:
        result["reason"] = str(reason).strip()
    return result


def _suitability_context_for_actuator(suitability: dict[str, int | str | bool]) -> str:
    """Format the suitability diagnosis for center/size repair prompts."""

    reason = str(suitability.get("reason") or "").strip()
    return (
        f"done={suitability.get('done')} "
        f"needs_position={suitability.get('needs_position')} "
        f"needs_size={suitability.get('needs_size')}\n"
        f"reason: {reason}"
    )


def _candidate_pixels(candidate_overlay: dict, *, screen_w: int, screen_h: int) -> tuple[int, int, int, int, int, int]:
    bx = int(candidate_overlay["x"] * screen_w)
    by = int(candidate_overlay["y"] * screen_h)
    bw = max(1, int(candidate_overlay["width"] * screen_w))
    bh = max(1, int(candidate_overlay["height"] * screen_h))
    cx = bx + bw // 2
    cy = by + bh // 2
    return bx, by, bw, bh, cx, cy


def _candidate_audit_user_text(
    *,
    utterance: str,
    screen_w: int,
    screen_h: int,
    candidate_overlay: dict,
    iteration: int,
    bearing: str | None,
    suitability_context: str | None = None,
) -> str:
    _bx, _by, bw, bh, cx, cy = _candidate_pixels(
        candidate_overlay,
        screen_w=screen_w,
        screen_h=screen_h,
    )
    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Candidate overlay: center=({cx}, {cy}) width={bw} height={bh}\n"
        f"Audit iteration: {iteration}"
    )
    if suitability_context:
        user_text += (
            "\n\nSuitability diagnosis for this same candidate:\n"
            f"{suitability_context}\n"
            "Repair that diagnosis with this actuator. If you output KEEP, "
            "your reason must explain why this actuator cannot improve the "
            "specific diagnosis."
        )
    if bearing:
        user_text += f"\n\nOperator bearing:\n{bearing}"
    return user_text


def _pick_suitability_audit(
    screenshot_b64: str,
    utterance: str,
    screen_w: int,
    screen_h: int,
    candidate_overlay: dict,
    *,
    iteration: int,
    bearing: str | None = None,
) -> dict[str, int | str | bool]:
    """Ask the VLM whether the current candidate is suitable."""

    user_text = _candidate_audit_user_text(
        utterance=utterance,
        screen_w=screen_w,
        screen_h=screen_h,
        candidate_overlay=candidate_overlay,
        iteration=iteration,
        bearing=bearing,
    )

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers(
            "suitability-audit",
            mode="gridpoint-iterative",
            iteration=iteration,
        ),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": SUITABILITY_AUDIT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=96),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()
    _pick_suitability_audit._last_raw = raw
    return _parse_suitability_audit_response(raw)


def _pick_center_audit(
    screenshot_b64: str,
    utterance: str,
    screen_w: int,
    screen_h: int,
    candidate_overlay: dict,
    *,
    iteration: int,
    bearing: str | None = None,
    suitability_context: str | None = None,
) -> dict[str, int | str]:
    """Ask the VLM to repair only the candidate center."""

    user_text = _candidate_audit_user_text(
        utterance=utterance,
        screen_w=screen_w,
        screen_h=screen_h,
        candidate_overlay=candidate_overlay,
        iteration=iteration,
        bearing=bearing,
        suitability_context=suitability_context,
    )
    resp = requests.post(
        _get_api_url(),
        headers=_api_headers(
            "center-audit",
            mode="gridpoint-iterative",
            iteration=iteration,
        ),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": CENTER_AUDIT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=96),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()
    _pick_center_audit._last_raw = raw
    return _parse_center_audit_response(raw, screen_w=screen_w, screen_h=screen_h)


def _pick_size_audit(
    screenshot_b64: str,
    utterance: str,
    screen_w: int,
    screen_h: int,
    candidate_overlay: dict,
    *,
    iteration: int,
    bearing: str | None = None,
    suitability_context: str | None = None,
) -> dict[str, int | str]:
    """Ask the VLM to repair only the candidate size."""

    user_text = _candidate_audit_user_text(
        utterance=utterance,
        screen_w=screen_w,
        screen_h=screen_h,
        candidate_overlay=candidate_overlay,
        iteration=iteration,
        bearing=bearing,
        suitability_context=suitability_context,
    )
    resp = requests.post(
        _get_api_url(),
        headers=_api_headers(
            "size-audit",
            mode="gridpoint-iterative",
            iteration=iteration,
        ),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": SIZE_AUDIT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=96),
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"].get("content", "").strip()
    _pick_size_audit._last_raw = raw
    return _parse_size_audit_response(raw, screen_w=screen_w, screen_h=screen_h)


def _candidate_from_center(
    cx: int,
    cy: int,
    width: int,
    height: int,
    *,
    screen_w: int,
    screen_h: int,
) -> dict[str, float]:
    width = max(1, min(screen_w, int(width)))
    height = max(1, min(screen_h, int(height)))
    bx = max(0, min(screen_w - width, int(cx) - width // 2))
    by = max(0, min(screen_h - height, int(cy) - height // 2))
    return {
        "x": bx / screen_w,
        "y": by / screen_h,
        "width": width / screen_w,
        "height": height / screen_h,
    }


def _field_edge_margin_px(*, screen_w: int, screen_h: int) -> int:
    """Small renderer hygiene inset for the optical field's screen-edge rim."""

    raw = os.environ.get("SPOKE_POSITIONING_FIELD_EDGE_MARGIN_PX")
    if raw is None:
        margin = 16 if min(screen_w, screen_h) >= 400 else 0
    else:
        try:
            margin = int(float(raw))
        except (TypeError, ValueError):
            margin = 16 if min(screen_w, screen_h) >= 400 else 0
    max_margin = max(0, min(screen_w, screen_h) // 2 - 1)
    return max(0, min(max_margin, margin))


def _sanitize_candidate_for_field_margin(
    candidate_overlay: dict,
    *,
    screen_w: int,
    screen_h: int,
) -> dict:
    """Crop presented positioning geometry against an inset screen-edge band."""

    margin = _field_edge_margin_px(screen_w=screen_w, screen_h=screen_h)
    if margin <= 0:
        return dict(candidate_overlay)

    x_px = int(candidate_overlay["x"] * screen_w)
    y_px = int(candidate_overlay["y"] * screen_h)
    w_px = max(1, int(candidate_overlay["width"] * screen_w))
    h_px = max(1, int(candidate_overlay["height"] * screen_h))

    left = max(margin, x_px)
    top = max(margin, y_px)
    right = min(screen_w - margin, x_px + w_px)
    bottom = min(screen_h - margin, y_px + h_px)

    if right <= left:
        left = max(margin, min(screen_w - margin - 1, x_px))
        right = min(screen_w - margin, left + 1)
    if bottom <= top:
        top = max(margin, min(screen_h - margin - 1, y_px))
        bottom = min(screen_h - margin, top + 1)

    sanitized = dict(candidate_overlay)
    sanitized.update({
        "x": left / screen_w,
        "y": top / screen_h,
        "width": max(1, right - left) / screen_w,
        "height": max(1, bottom - top) / screen_h,
    })
    return sanitized


def _apply_split_audit(
    candidate_overlay: dict,
    center_audit: dict[str, int | str],
    size_audit: dict[str, int | str],
    *,
    screen_w: int,
    screen_h: int,
) -> dict[str, float]:
    _bx, _by, bw, bh, cx, cy = _candidate_pixels(
        candidate_overlay,
        screen_w=screen_w,
        screen_h=screen_h,
    )

    if isinstance(center_audit.get("center_x"), int):
        cx = int(center_audit["center_x"])
    if isinstance(center_audit.get("center_y"), int):
        cy = int(center_audit["center_y"])
    if isinstance(size_audit.get("width"), int):
        bw = int(size_audit["width"])
    if isinstance(size_audit.get("height"), int):
        bh = int(size_audit["height"])

    return _candidate_from_center(cx, cy, bw, bh, screen_w=screen_w, screen_h=screen_h)


def reposition_gridpoint(
    utterance: str,
    screenshot: Image.Image,
    current_overlay: dict | None = None,
    screen_w: int = 1920,
    screen_h: int = 1080,
    on_step: "callable | None" = None,
    bearing: str | None = None,
) -> dict | None:
    """Grid-point pipeline: pick intersection → move → resize.

    Up to 5 retries on grid-point parse failure, then falls back to
    bbox with reasoning for position. Resize uses reasoning and a
    fresh image with the new position inscribed.
    """
    reposition_gridpoint._last_debug = []
    t0 = time.time()

    def _report(intermediate=None):
        if on_step:
            on_step(list(reposition_gridpoint._last_debug), intermediate)

    if current_overlay is None:
        current_overlay = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}

    if bearing:
        reposition_gridpoint._last_debug.append(f"Bearing: {bearing[:200]}")
        _report()

    # Draw grid points and overlay outline on the screenshot
    annotated = _draw_overlay_outline(screenshot, current_overlay)
    annotated = _draw_grid_points(annotated)
    screenshot_b64 = _encode_image(annotated)

    cur_w = int(current_overlay["width"] * screen_w)
    cur_h = int(current_overlay["height"] * screen_h)

    # Step 1: Grid-point center pick (up to 5 retries)
    cx, cy = None, None
    for attempt in range(5):
        label = _pick_gridpoint(
            screenshot_b64, utterance, screen_w, screen_h,
            current_overlay, bearing,
        )
        t_attempt = time.time()
        if label is not None:
            cx, cy = _gridpoint_coords(label, screen_w, screen_h)
            reposition_gridpoint._last_debug.append(
                f"GridPoint: {label} → ({cx}, {cy}) (attempt {attempt+1}, {t_attempt - t0:.1f}s)"
            )
            break
        else:
            last_raw = getattr(_pick_gridpoint, '_last_raw', '?')
            reposition_gridpoint._last_debug.append(
                f"GridPoint: parse fail (attempt {attempt+1}, {t_attempt - t0:.1f}s) raw={last_raw!r}"
            )
        _report()

    # Fallback: bbox with reasoning if all grid attempts failed
    if cx is None:
        reposition_gridpoint._last_debug.append("GridPoint: all 5 attempts failed, falling back to bbox reasoning")
        _report()

        # Use bbox with thinking forced on
        old_thinking = os.environ.get("SPOKE_POSITIONING_THINKING")
        os.environ["SPOKE_POSITIONING_THINKING"] = "1"
        try:
            bx, by, bw, bh = _pick_bbox(
                screenshot_b64, utterance, utterance, "direct",
                screen_w, screen_h, current_overlay, bearing=bearing,
            )
            cx = bx + bw // 2
            cy = by + bh // 2
            # Also use the bbox size as a hint
            cur_w, cur_h = bw, bh
            t_fallback = time.time()
            reposition_gridpoint._last_debug.append(
                f"BBox fallback: center=({cx}, {cy}) size={bw}×{bh}px ({t_fallback - t0:.1f}s)"
            )
        finally:
            if old_thinking is not None:
                os.environ["SPOKE_POSITIONING_THINKING"] = old_thinking
            else:
                os.environ.pop("SPOKE_POSITIONING_THINKING", None)
        _report()

    # Emit intermediate — overlay moves NOW at current size
    bx = max(0, min(screen_w - cur_w, cx - cur_w // 2))
    by = max(0, min(screen_h - cur_h, cy - cur_h // 2))
    intermediate = {
        **_sanitize_candidate_for_field_margin(
            {
                "x": bx / screen_w,
                "y": by / screen_h,
                "width": cur_w / screen_w,
                "height": cur_h / screen_h,
            },
            screen_w=screen_w,
            screen_h=screen_h,
        ),
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": round(time.time() - t0, 2),
        "_intermediate": True,
    }
    _report(intermediate)

    # Step 2: Resize with fresh image showing new position
    new_overlay = {"x": bx / screen_w, "y": by / screen_h,
                   "width": cur_w / screen_w, "height": cur_h / screen_h}
    resized_annotated = _draw_overlay_outline(screenshot, new_overlay)
    resize_b64 = _encode_image(resized_annotated)

    resize_result = _pick_gridpoint_resize(
        resize_b64, utterance, screen_w, screen_h,
        (cx, cy), current_overlay, bearing,
    )
    t_resize = time.time()

    if resize_result is not None:
        new_w, new_h = resize_result
        reposition_gridpoint._last_debug.append(
            f"Resize: {new_w}×{new_h}px ({t_resize - t0:.1f}s)"
        )
    else:
        new_w, new_h = cur_w, cur_h
        reposition_gridpoint._last_debug.append(
            f"Resize: unchanged ({t_resize - t0:.1f}s)"
        )

    # Final position with new size
    bx = max(0, min(screen_w - new_w, cx - new_w // 2))
    by = max(0, min(screen_h - new_h, cy - new_h // 2))

    elapsed = round(time.time() - t0, 2)
    reposition_gridpoint._last_debug.append(f"Total: {elapsed:.1f}s")
    _report()

    return {
        **_sanitize_candidate_for_field_margin(
            {
                "x": bx / screen_w,
                "y": by / screen_h,
                "width": new_w / screen_w,
                "height": new_h / screen_h,
            },
            screen_w=screen_w,
            screen_h=screen_h,
        ),
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": elapsed,
        "_screenshot_b64": screenshot_b64,
    }


def reposition_gridpoint_iterative(
    utterance: str,
    screenshot: Image.Image,
    current_overlay: dict | None = None,
    screen_w: int = 1920,
    screen_h: int = 1080,
    on_step: "callable | None" = None,
    bearing: str | None = None,
) -> dict | None:
    """Grid-point seed followed by bounded cheap suitability/center/size rounds."""

    reposition_gridpoint_iterative._last_debug = []
    t0 = time.time()

    def _report(intermediate=None):
        if on_step:
            on_step(list(reposition_gridpoint_iterative._last_debug), intermediate)

    if current_overlay is None:
        current_overlay = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}

    if bearing:
        reposition_gridpoint_iterative._last_debug.append(f"Bearing: {bearing[:200]}")
        _report()

    annotated = _draw_overlay_outline(screenshot, current_overlay)
    annotated = _draw_grid_points(annotated)
    screenshot_b64 = _encode_image(annotated)

    cur_w = max(1, int(current_overlay["width"] * screen_w))
    cur_h = max(1, int(current_overlay["height"] * screen_h))
    cur_cx = int(current_overlay["x"] * screen_w) + cur_w // 2
    cur_cy = int(current_overlay["y"] * screen_h) + cur_h // 2

    cx, cy = cur_cx, cur_cy
    label = None
    ensemble = None
    if _positioning_gridpoint_ensemble_enabled():
        ensemble = _pick_gridpoint_ensemble(
            screenshot,
            utterance,
            screen_w,
            screen_h,
            current_overlay,
            bearing,
        )
        t_ensemble = time.time()
        if ensemble is not None:
            cx = int(ensemble["center_x"])
            cy = int(ensemble["center_y"])
            label = "ENSEMBLE"
            labels = ",".join(str(value) for value in ensemble.get("labels", []))
            lattices = ",".join(str(value) for value in ensemble.get("lattices", []))
            reposition_gridpoint_iterative._last_debug.append(
                "GridEnsemble: labels=%s lattices=%s center=(%d, %d) "
                "confidence=%.2f spread=(%.3f, %.3f) (%.1fs)"
                % (
                    labels,
                    lattices,
                    cx,
                    cy,
                    float(ensemble.get("confidence", 0.0)),
                    float(ensemble.get("spread_x", 0.0)),
                    float(ensemble.get("spread_y", 0.0)),
                    t_ensemble - t0,
                )
            )
        else:
            samples = getattr(_pick_gridpoint_ensemble, "_last_samples", [])
            reposition_gridpoint_iterative._last_debug.append(
                f"GridEnsemble: no parseable samples, falling back to single grid raw={samples!r}"
            )
            _report()

    if label is None:
        for attempt in range(3):
            label = _pick_gridpoint(
                screenshot_b64,
                utterance,
                screen_w,
                screen_h,
                current_overlay,
                bearing,
            )
            t_attempt = time.time()
            if label is not None:
                cx, cy = _gridpoint_coords(label, screen_w, screen_h)
                reposition_gridpoint_iterative._last_debug.append(
                    f"GridPoint: {label} → ({cx}, {cy}) "
                    f"(attempt {attempt + 1}, {t_attempt - t0:.1f}s)"
                )
                break
            raw = getattr(_pick_gridpoint, "_last_raw", "?")
            reposition_gridpoint_iterative._last_debug.append(
                f"GridPoint: parse fail (attempt {attempt + 1}, {t_attempt - t0:.1f}s) raw={raw!r}"
            )
            _report()

    if label is None:
        reposition_gridpoint_iterative._last_debug.append(
            "GridPoint: no parse after 3 attempts, auditing current center"
        )

    candidate = _candidate_from_center(cx, cy, cur_w, cur_h, screen_w=screen_w, screen_h=screen_h)
    max_rounds = max(1, int(os.environ.get("SPOKE_POSITIONING_AUDIT_ROUNDS", "3")))

    for iteration in range(1, max_rounds + 1):
        annotated = _draw_overlay_outline(screenshot, candidate)
        candidate_b64 = _encode_image(annotated)
        suitability = _pick_suitability_audit(
            candidate_b64,
            utterance,
            screen_w,
            screen_h,
            candidate,
            iteration=iteration,
            bearing=bearing,
        )
        raw = getattr(_pick_suitability_audit, "_last_raw", None)
        reposition_gridpoint_iterative._last_debug.append(
            "Audit %d suitability: done=%s position=%s size=%s"
            % (
                iteration,
                suitability.get("done"),
                suitability.get("needs_position"),
                suitability.get("needs_size"),
            )
        )
        if raw:
            reposition_gridpoint_iterative._last_debug.append(
                f"Audit {iteration} suitability raw: {raw}"
            )
        reason = suitability.get("reason")
        if reason:
            reposition_gridpoint_iterative._last_debug.append(
                f"Audit {iteration} suitability reason: {reason}"
            )
        if suitability.get("done") is True:
            break

        center_audit: dict[str, int | str] = {
            "center_x": "KEEP",
            "center_y": "KEEP",
            "reason": "",
        }
        size_audit: dict[str, int | str] = {"width": "KEEP", "height": "KEEP", "reason": ""}
        run_center = True
        run_size = True
        suitability_context = _suitability_context_for_actuator(suitability)

        if run_center:
            center_audit = _pick_center_audit(
                candidate_b64,
                utterance,
                screen_w,
                screen_h,
                candidate,
                iteration=iteration,
                bearing=bearing,
                suitability_context=suitability_context,
            )
            center_raw = getattr(_pick_center_audit, "_last_raw", None)
            reposition_gridpoint_iterative._last_debug.append(
                "Audit %d center: center_x=%s center_y=%s"
                % (
                    iteration,
                    center_audit.get("center_x"),
                    center_audit.get("center_y"),
                )
            )
            if center_raw:
                reposition_gridpoint_iterative._last_debug.append(
                    f"Audit {iteration} center raw: {center_raw}"
                )
            center_reason = center_audit.get("reason")
            if center_reason:
                reposition_gridpoint_iterative._last_debug.append(
                    f"Audit {iteration} center reason: {center_reason}"
                )

        if run_size:
            size_audit = _pick_size_audit(
                candidate_b64,
                utterance,
                screen_w,
                screen_h,
                candidate,
                iteration=iteration,
                bearing=bearing,
                suitability_context=suitability_context,
            )
            size_raw = getattr(_pick_size_audit, "_last_raw", None)
            reposition_gridpoint_iterative._last_debug.append(
                "Audit %d size: width=%s height=%s"
                % (
                    iteration,
                    size_audit.get("width"),
                    size_audit.get("height"),
                )
            )
            if size_raw:
                reposition_gridpoint_iterative._last_debug.append(
                    f"Audit {iteration} size raw: {size_raw}"
                )
            size_reason = size_audit.get("reason")
            if size_reason:
                reposition_gridpoint_iterative._last_debug.append(
                    f"Audit {iteration} size reason: {size_reason}"
                )

        updated = _apply_split_audit(
            candidate,
            center_audit,
            size_audit,
            screen_w=screen_w,
            screen_h=screen_h,
        )
        if updated == candidate:
            reposition_gridpoint_iterative._last_debug.append(
                f"Audit {iteration}: no material candidate change"
            )
            break
        candidate = updated
        intermediate = {
            **_sanitize_candidate_for_field_margin(
                candidate,
                screen_w=screen_w,
                screen_h=screen_h,
            ),
            "content_desc": utterance,
            "utterance": utterance,
            "elapsed_s": round(time.time() - t0, 2),
            "_intermediate": True,
        }
        _report(intermediate)

    elapsed = round(time.time() - t0, 2)
    reposition_gridpoint_iterative._last_debug.append(f"Total: {elapsed:.1f}s")
    _report()

    return {
        **_sanitize_candidate_for_field_margin(
            candidate,
            screen_w=screen_w,
            screen_h=screen_h,
        ),
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": elapsed,
        "_screenshot_b64": screenshot_b64,
        "_debug_lines": list(reposition_gridpoint_iterative._last_debug),
    }


def _draw_overlay_outline(screenshot: Image.Image, overlay_rect: dict | None) -> Image.Image:
    """Draw a red dashed outline on the screenshot showing the current overlay position."""
    img = screenshot.copy()
    if overlay_rect is None:
        return img

    draw = ImageDraw.Draw(img)
    sw, sh = img.size
    x = int(overlay_rect.get("x", 0.3) * sw)
    y = int(overlay_rect.get("y", 0.3) * sh)
    w = int(overlay_rect.get("width", 0.4) * sw)
    h = int(overlay_rect.get("height", 0.4) * sh)

    # Draw dashed outline in red
    outline_color = (255, 60, 60)
    thickness = max(2, min(sw, sh) // 300)
    # Top, bottom, left, right edges
    for edge_start, edge_end in [
        ((x, y), (x + w, y)),           # top
        ((x, y + h), (x + w, y + h)),   # bottom
        ((x, y), (x, y + h)),           # left
        ((x + w, y), (x + w, y + h)),   # right
    ]:
        dash_len = 12
        sx, sy = edge_start
        ex, ey = edge_end
        if sx == ex:  # vertical
            for dy in range(0, ey - sy, dash_len * 2):
                draw.line([(sx, sy + dy), (sx, min(sy + dy + dash_len, ey))],
                          fill=outline_color, width=thickness)
        else:  # horizontal
            for dx in range(0, ex - sx, dash_len * 2):
                draw.line([(sx + dx, sy), (min(sx + dx + dash_len, ex), sy)],
                          fill=outline_color, width=thickness)

    return img


def _pick_center(screenshot_b64: str, utterance: str, content_desc: str, mode: str) -> str:
    """Step 1: Pick a coarse center region from the 3×3 grid."""
    user_text = f"User request: {utterance}\nMode: {mode}\nContent: {content_desc}"
    thinking = _detect_thinking_enabled()

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("center"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": CENTER_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16384 if thinking else 8),
        },
        timeout=120,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    raw = msg.get("content", "").strip().upper()

    # Capture thinking for diagnostics
    thinking_text = msg.get("reasoning_content") or msg.get("thinking", "")
    if thinking_text:
        _pick_center._last_thinking = thinking_text

    # Extract region code
    for region in COARSE_REGIONS:
        if region in raw:
            return region
    import logging
    logging.getLogger(__name__).warning("Could not parse center region from %r, falling back to MC", raw)
    return "MC"


def _pick_size(screenshot_b64: str, utterance: str, content_desc: str, mode: str,
               center_region: str,
               current_overlay: dict | None = None) -> tuple[float, float]:
    """Step 2: Pick width/height as screen fractions given the center and screen image.

    Returns (width_frac, height_frac) as fractions of screen (0-1).
    """
    thinking = _detect_thinking_enabled()
    cur_w_pct = int((current_overlay or {}).get("width", 0.4) * 100)
    cur_h_pct = int((current_overlay or {}).get("height", 0.4) * 100)

    user_text = (
        f"User request: {utterance}\n"
        f"Mode: {mode}\n"
        f"Content: {content_desc}\n"
        f"Overlay center: {center_region}\n"
        f"Current overlay size: {cur_w_pct}% of screen width, {cur_h_pct}% of screen height"
    )

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("size"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": SIZE_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16384 if thinking else 16),
        },
        timeout=120,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    raw = msg.get("content", "").strip()

    # Capture thinking for diagnostics
    thinking_text = msg.get("reasoning_content") or msg.get("thinking", "")
    if thinking_text:
        _pick_size._last_thinking = thinking_text

    # Parse "width_pct height_pct"
    parts = raw.split()
    try:
        w_pct = max(5, min(100, float(parts[0])))
        h_pct = max(5, min(100, float(parts[1]))) if len(parts) > 1 else w_pct
    except (ValueError, IndexError):
        w_pct, h_pct = 40.0, 40.0
    return w_pct / 100.0, h_pct / 100.0


def _pick_bbox(screenshot_b64: str, utterance: str, content_desc: str, mode: str,
               screen_w: int, screen_h: int,
               current_overlay: dict | None = None,
               bearing: str | None = None) -> tuple[int, int, int, int]:
    """Single-shot: pick overlay bounding box in pixel coordinates."""
    thinking = _detect_thinking_enabled()

    cur_x = int((current_overlay or {}).get("x", 0.3) * screen_w)
    cur_y = int((current_overlay or {}).get("y", 0.3) * screen_h)
    cur_w = int((current_overlay or {}).get("width", 0.4) * screen_w)
    cur_h = int((current_overlay or {}).get("height", 0.4) * screen_h)
    cur_cx = cur_x + cur_w // 2
    cur_cy = cur_y + cur_h // 2

    user_text = (
        f"User request: {utterance}\n"
        f"Screen resolution: {screen_w}×{screen_h} pixels\n"
        f"Current overlay: center=({cur_cx}, {cur_cy}) width={cur_w} height={cur_h}"
    )
    if bearing:
        user_text += (
            f"\n\nOperator bearing (from recent context — use if coherent with "
            f"what you see, ignore if stale):\n{bearing}"
        )

    resp = requests.post(
        _get_api_url(),
        headers=_api_headers("bbox"),
        json={
            "model": os.environ.get("SPOKE_VLM_MODEL", "qwen3.6-35b-a3b-oq8"),
            "messages": [
                {"role": "system", "content": BBOX_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
            **_sampling_params(max_tokens=16384 if thinking else 32),
        },
        timeout=120,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    raw = msg.get("content", "").strip()

    thinking_text = msg.get("reasoning_content") or msg.get("thinking", "")
    if thinking_text:
        _pick_bbox._last_thinking = thinking_text

    # Parse "center_x center_y width height" → convert to corner coords
    parts = raw.split()
    try:
        cx = int(float(parts[0]))
        cy = int(float(parts[1]))
        bw = max(1, int(float(parts[2])))
        bh = max(1, int(float(parts[3])))
        # Convert center to top-left corner, clamp to screen
        bx = max(0, min(screen_w - bw, cx - bw // 2))
        by = max(0, min(screen_h - bh, cy - bh // 2))
        bw = min(bw, screen_w - bx)
        bh = min(bh, screen_h - by)
    except (ValueError, IndexError):
        import logging
        logging.getLogger(__name__).warning("Could not parse bbox from %r, using center default", raw)
        bw, bh = screen_w // 3, screen_h // 3
        bx, by = (screen_w - bw) // 2, (screen_h - bh) // 2
    return bx, by, bw, bh


def reposition_bbox(
    utterance: str,
    screenshot: Image.Image,
    current_overlay: dict | None = None,
    screen_w: int = 1920,
    screen_h: int = 1080,
    on_step: "callable | None" = None,
    bearing: str | None = None,
) -> dict | None:
    """Single-shot bbox pipeline: one VLM call → pixel bounding box.

    Returns dict with x, y, width, height as fractions of screen (0-1),
    plus _screenshot_b64 for the bearing update call.
    """
    reposition_bbox._last_debug = []
    t0 = time.time()

    def _report():
        if on_step:
            on_step(list(reposition_bbox._last_debug))

    if current_overlay is None:
        current_overlay = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}

    if bearing:
        reposition_bbox._last_debug.append(f"Bearing: {bearing[:200]}")
        _report()

    # Prepare image with overlay outline
    annotated = _draw_overlay_outline(screenshot, current_overlay)
    screenshot_b64 = _encode_image(annotated)

    # Single VLM call: raw utterance straight to bbox
    bx, by, bw, bh = _pick_bbox(
        screenshot_b64, utterance, utterance, "direct",
        screen_w, screen_h, current_overlay, bearing=bearing,
    )
    t_bbox = time.time()
    reposition_bbox._last_debug.append(
        f"BBox: ({bx}, {by}) {bw}×{bh}px ({t_bbox - t0:.1f}s)"
    )
    bbox_thinking = getattr(_pick_bbox, '_last_thinking', None)
    if bbox_thinking:
        reposition_bbox._last_debug.append(f"BBox thinking: {bbox_thinking}")
        _pick_bbox._last_thinking = None
    _report()

    elapsed = round(time.time() - t0, 2)
    reposition_bbox._last_debug.append(f"Total: {elapsed:.1f}s")
    _report()

    result = {
        "x": bx / screen_w,
        "y": by / screen_h,
        "width": bw / screen_w,
        "height": bh / screen_h,
        "content_desc": utterance,
        "utterance": utterance,
        "elapsed_s": elapsed,
        "_screenshot_b64": screenshot_b64,
    }
    return result


def reposition_twostep(
    utterance: str,
    screenshot: Image.Image,
    current_overlay: dict | None = None,
    on_step: "callable | None" = None,
) -> dict | None:
    """Two-step pipeline: center pick + size as screen percentages.

    current_overlay: dict with x, y, width, height as fractions (0-1),
    or None for default 40%×40% centered overlay.
    on_step: optional callback(debug_lines: list[str]) called after each
    pipeline step completes, for incremental progress display.

    Returns dict with x, y, width, height as fractions of screen (0-1).
    """
    reposition_twostep._last_debug = []
    t0 = time.time()

    def _report():
        if on_step:
            on_step(list(reposition_twostep._last_debug))

    if current_overlay is None:
        current_overlay = {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4}

    # Resolve intent (text-only, fast)
    intent = resolve_intent(utterance)
    t_intent = time.time()
    reposition_twostep._last_debug.append(
        f"Intent: {intent} ({t_intent - t0:.1f}s)"
    )
    _report()

    mode = intent["mode"]
    if mode == "target":
        content_desc = intent.get("target_desc", utterance)
    else:
        content_desc = intent.get("content_desc", utterance)

    # Prepare image with overlay outline
    annotated = _draw_overlay_outline(screenshot, current_overlay)
    screenshot_b64 = _encode_image(annotated, scale=0.5)

    # Step 1: pick center (VLM, image prefill happens here)
    center_region = _pick_center(screenshot_b64, utterance, content_desc, mode)
    t_center = time.time()
    center_x, center_y = COARSE_REGIONS.get(center_region, (0.5, 0.5))
    reposition_twostep._last_debug.append(
        f"Center: {center_region} → ({center_x:.2f}, {center_y:.2f}) ({t_center - t_intent:.1f}s)"
    )
    center_thinking = getattr(_pick_center, '_last_thinking', None)
    if center_thinking:
        reposition_twostep._last_debug.append(f"Center thinking: {center_thinking}")
        _pick_center._last_thinking = None
    _report()

    # Step 2: pick size (VLM, same image — returns screen fractions directly)
    new_w, new_h = _pick_size(screenshot_b64, utterance, content_desc, mode,
                              center_region, current_overlay)
    t_size = time.time()
    reposition_twostep._last_debug.append(
        f"Size: {new_w:.0%} width, {new_h:.0%} height ({t_size - t_center:.1f}s)"
    )
    size_thinking = getattr(_pick_size, '_last_thinking', None)
    if size_thinking:
        reposition_twostep._last_debug.append(f"Size thinking: {size_thinking}")
        _pick_size._last_thinking = None
    _report()

    # Clamp so overlay stays on screen
    new_x = max(0.0, min(1.0 - new_w, center_x - new_w / 2))
    new_y = max(0.0, min(1.0 - new_h, center_y - new_h / 2))

    elapsed = round(time.time() - t0, 2)
    reposition_twostep._last_debug.append(f"Total: {elapsed:.1f}s")

    return {
        "x": new_x,
        "y": new_y,
        "width": new_w,
        "height": new_h,
        "content_desc": f"{mode}: {content_desc}",
        "utterance": utterance,
        "elapsed_s": elapsed,
        "center_region": center_region,
    }


def reposition(utterance: str, screenshot: Image.Image) -> dict | None:
    """Full pipeline: utterance + screenshot → new overlay position.

    Returns dict with x, y, width, height as fractions of screen (0-1),
    or None if no viable position found.

    Sets reposition._last_debug with raw model outputs for fail-loud diagnostics.
    """
    reposition._last_debug = []
    t0 = time.time()

    # Layer 1: resolve intent
    intent = resolve_intent(utterance)
    reposition._last_debug.append(f"Intent: {intent}")

    if intent["mode"] == "target":
        # Target mode — model picks which cells the overlay should occupy
        target_desc = intent["target_desc"]
        target_map = target_cells(target_desc)
        yes_cells = [k for k, v in target_map.items() if v]
        no_cells = [k for k, v in target_map.items() if not v]
        reposition._last_debug.append(f"Target cells YES: {yes_cells}")
        reposition._last_debug.append(f"Target cells NO: {no_cells}")
        rect = largest_rectangle_target(target_map)
        elapsed = round(time.time() - t0, 2)
        if rect:
            rect["content_desc"] = f"targeting: {target_desc}"
            rect["content_map"] = target_map
            rect["utterance"] = utterance
            rect["elapsed_s"] = elapsed
        else:
            reposition._last_debug.append("No YES cells → no viable rectangle")
        return rect

    # AVOID or FIND mode — both need VLM content detection
    is_find = intent["mode"] == "find"
    content_desc = intent["content_desc"]
    reposition._last_debug.append(f"{'FIND' if is_find else 'AVOID'} content: {content_desc}")

    # Layer 2: detect content on screen
    content_map = detect_content(screenshot, content_desc)
    yes_cells = [k for k, v in content_map.items() if v]
    no_cells = [k for k, v in content_map.items() if not v]
    reposition._last_debug.append(f"Detect YES: {yes_cells}")
    reposition._last_debug.append(f"Detect NO: {no_cells}")

    # Layer 3: find rectangle
    # AVOID → largest rect in NO cells (empty space)
    # FIND  → largest rect in YES cells (where content is)
    if is_find:
        rect = largest_rectangle_target(content_map)
    else:
        rect = largest_rectangle(content_map)

    elapsed = round(time.time() - t0, 2)

    if rect:
        prefix = "finding" if is_find else "avoiding"
        rect["content_desc"] = f"{prefix}: {content_desc}"
        rect["content_map"] = content_map
        rect["utterance"] = utterance
        rect["elapsed_s"] = elapsed
    else:
        if is_find:
            reposition._last_debug.append("No YES cells → content not found on screen")
        else:
            reposition._last_debug.append("All cells YES → no empty space for overlay")

    return rect


reposition_centersize = _with_positioning_utterance_scope(reposition_centersize)
reposition_gridpoint = _with_positioning_utterance_scope(reposition_gridpoint)
reposition_gridpoint_iterative = _with_positioning_utterance_scope(
    reposition_gridpoint_iterative
)
reposition_bbox = _with_positioning_utterance_scope(reposition_bbox)
reposition_twostep = _with_positioning_utterance_scope(reposition_twostep)
reposition = _with_positioning_utterance_scope(reposition)
