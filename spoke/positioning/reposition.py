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
POS_ROWS = 6
POS_COLS = 6

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


def _api_headers(step: str = "positioning") -> dict[str, str]:
    """Standard headers for positioning API calls, including Grapheus X-Spoke-* metadata."""
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "X-Spoke-Pathway": "positioning",
        "X-Spoke-Step": step,
    }


def _detect_thinking_enabled() -> bool:
    """Whether to enable thinking for content detection calls."""
    return os.environ.get("SPOKE_POSITIONING_THINKING", "0") == "1"


def _detect_max_tokens() -> int:
    """Max tokens for content detection: 256 without thinking, 16384 with."""
    if _detect_thinking_enabled():
        return 16384
    # 6×6 grid × 4 tokens per cell worst case = 144, rounded up with headroom
    return 256


def _encode_image(img: Image.Image, scale: float = 0.5) -> str:
    """Encode PIL image as base64 PNG, optionally downscaling."""
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
            "temperature": 0.3, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
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
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": _detect_max_tokens(),
            "chat_template_kwargs": {"enable_thinking": _detect_thinking_enabled()},
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
            "temperature": 0.3, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": _detect_max_tokens(),
            "chat_template_kwargs": {"enable_thinking": _detect_thinking_enabled()},
        },
        timeout=30,
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
            "temperature": 0.6 if thinking else 0.3,
            "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16384 if thinking else 8,
            "chat_template_kwargs": {"enable_thinking": thinking},
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
            "temperature": 0.6 if thinking else 0.3,
            "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
            "max_tokens": 16384 if thinking else 16,
            "chat_template_kwargs": {"enable_thinking": thinking},
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
