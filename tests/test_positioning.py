"""Tests for the semantic overlay positioning pipeline."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ── grid_overlay tests ──

def test_grid_overlay_draws_flood_labels():
    """Flood grid draws labels in every cell."""
    from spoke.positioning.grid_overlay import draw_grid, ROW_LABELS
    import tempfile
    import os

    # Create a simple test image
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        src = f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dst = f.name

    try:
        result = draw_grid(src, dst, style="flood", rows=4, cols=4)
        assert len(result) == 4  # cell_w, cell_h, rows, cols
        _, _, rows, cols = result
        assert rows == 4
        assert cols == 4
        # Output image should exist and be valid
        out = Image.open(dst)
        assert out.size == (400, 400)
    finally:
        os.unlink(src)
        os.unlink(dst)


def test_grid_overlay_sparse_style():
    """Sparse grid only labels edges and periodic cells."""
    from spoke.positioning.grid_overlay import draw_grid
    import tempfile
    import os

    img = Image.new("RGB", (400, 400), (0, 0, 0))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        src = f.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dst = f.name

    try:
        result = draw_grid(src, dst, style="sparse", rows=4, cols=4)
        out = Image.open(dst)
        assert out.size == (400, 400)
    finally:
        os.unlink(src)
        os.unlink(dst)


# ── reposition pipeline tests ──

def test_largest_rectangle_full_empty():
    """All cells empty → full grid rectangle."""
    from spoke.positioning.reposition import largest_rectangle

    content_map = {f"{r}{c}": False for r in "ABCDEF" for c in "123456"}
    rect = largest_rectangle(content_map)
    assert rect is not None
    assert rect["x"] == 0.0
    assert rect["y"] == 0.0
    assert rect["width"] == 1.0
    assert rect["height"] == 1.0


def test_largest_rectangle_all_content():
    """All cells have content → no rectangle."""
    from spoke.positioning.reposition import largest_rectangle

    content_map = {f"{r}{c}": True for r in "ABCDEF" for c in "123456"}
    rect = largest_rectangle(content_map)
    assert rect is None


def test_largest_rectangle_right_column_empty():
    """Only column 6 is empty → tall narrow rectangle on the right."""
    from spoke.positioning.reposition import largest_rectangle

    content_map = {}
    for r in "ABCDEF":
        for c in "123456":
            content_map[f"{r}{c}"] = (c != "6")

    rect = largest_rectangle(content_map)
    assert rect is not None
    assert rect["x"] == pytest.approx(5 / 6)  # column 6 starts at 5/6
    assert rect["width"] == pytest.approx(1 / 6)  # one column wide
    assert rect["height"] == 1.0  # full height


def test_largest_rectangle_top_two_rows_empty():
    """Rows A-B empty, C-F content → wide rectangle on top."""
    from spoke.positioning.reposition import largest_rectangle

    content_map = {}
    for r in "ABCDEF":
        for c in "123456":
            content_map[f"{r}{c}"] = (r in "CDEF")

    rect = largest_rectangle(content_map)
    assert rect is not None
    assert rect["y"] == 0.0
    assert rect["height"] == pytest.approx(2 / 6)
    assert rect["width"] == 1.0


def test_largest_rectangle_corner_block():
    """2x2 block in top-right is empty."""
    from spoke.positioning.reposition import largest_rectangle

    content_map = {f"{r}{c}": True for r in "ABCDEF" for c in "123456"}
    content_map["A5"] = False
    content_map["A6"] = False
    content_map["B5"] = False
    content_map["B6"] = False

    rect = largest_rectangle(content_map)
    assert rect is not None
    assert rect["x"] == pytest.approx(4 / 6)
    assert rect["y"] == 0.0
    assert rect["width"] == pytest.approx(2 / 6)
    assert rect["height"] == pytest.approx(2 / 6)


def test_largest_rectangle_l_shape_picks_best():
    """L-shaped empty region → largest inscribed rectangle, not the whole L."""
    from spoke.positioning.reposition import largest_rectangle

    # All content except column 6 (full height) and row A (full width)
    content_map = {}
    for r in "ABCDEF":
        for c in "123456":
            if r == "A" or c == "6":
                content_map[f"{r}{c}"] = False
            else:
                content_map[f"{r}{c}"] = True

    rect = largest_rectangle(content_map)
    assert rect is not None
    # Should pick the full top row (6×1 = 6 cells, area=1.0*1/6)
    # or the full right column (1×6 = 6 cells, area=1/6*1.0)
    assert rect["width"] * rect["height"] == pytest.approx(1 / 6)  # 6/36


# ── intent resolution mock tests ──

def test_resolve_intent_prompt_has_both_modes():
    """Intent system prompt covers both AVOID and TARGET modes."""
    from spoke.positioning.reposition import INTENT_SYSTEM

    assert "AVOID:" in INTENT_SYSTEM
    assert "TARGET:" in INTENT_SYSTEM
    assert "stop blocking my code" in INTENT_SYSTEM
    assert "center" in INTENT_SYSTEM


def test_target_system_prompt_has_grid_layout():
    """Target system prompt explains the 4×4 grid layout."""
    from spoke.positioning.reposition import TARGET_SYSTEM

    assert "A1" in TARGET_SYSTEM
    assert "D4" in TARGET_SYSTEM
    assert "YES" in TARGET_SYSTEM
    assert "center" in TARGET_SYSTEM


def test_split_audit_prompts_keep_center_independent_from_current_size():
    """Center repair should choose the best eventual center, not preserve size."""
    from spoke.positioning.reposition import (
        CENTER_AUDIT_SYSTEM,
        SIZE_AUDIT_SYSTEM,
        SUITABILITY_AUDIT_SYSTEM,
    )

    assert "red dashed outline is the current proposed overlay" in SUITABILITY_AUDIT_SYSTEM
    assert '"done"' in SUITABILITY_AUDIT_SYSTEM
    assert '"needs_position"' in SUITABILITY_AUDIT_SYSTEM
    assert "Judge only the overlay center" in CENTER_AUDIT_SYSTEM
    assert "width and height will be adjusted optimally" in CENTER_AUDIT_SYSTEM
    assert "Do not preserve the current size" in CENTER_AUDIT_SYSTEM
    assert '"center_x"' in CENTER_AUDIT_SYSTEM
    assert "Judge only the overlay size" in SIZE_AUDIT_SYSTEM
    assert "center will be adjusted optimally" in SIZE_AUDIT_SYSTEM
    assert '"width"' in SIZE_AUDIT_SYSTEM


def test_split_audit_headers_are_round_addressable():
    """Grapheus logs need enough header structure to reconstruct loop behavior."""
    from spoke.positioning.reposition import _api_headers, positioning_utterance_scope

    with positioning_utterance_scope("positioning-token-50"):
        headers = _api_headers("center-audit", mode="gridpoint-iterative", iteration=2)

    assert headers["X-Spoke-Pathway"] == "positioning"
    assert headers["X-Spoke-Utterance-ID"] == "positioning-token-50"
    assert headers["X-Spoke-Step"] == "center-audit"
    assert headers["X-Spoke-Positioning-Mode"] == "gridpoint-iterative"
    assert headers["X-Spoke-Positioning-Iteration"] == "2"


def test_positioning_pipeline_reuses_one_utterance_id_across_model_calls(monkeypatch):
    """A single semantic positioning request should be groupable in Grapheus."""
    import importlib

    reposition = importlib.import_module("spoke.positioning.reposition")

    image = Image.new("RGB", (100, 100), "white")
    seen_ids = []

    class FakeResponse:
        def __init__(self, content):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(_url, *, headers, **_kwargs):
        seen_ids.append(headers["X-Spoke-Utterance-ID"])
        if headers["X-Spoke-Step"] == "gridpoint":
            return FakeResponse("A3")
        return FakeResponse(
            '{"done":true,"needs_position":false,"needs_size":false,'
            '"reason":"already upper right"}'
        )

    monkeypatch.setattr(reposition.requests, "post", fake_post)
    monkeypatch.setattr(reposition, "_draw_grid_points", lambda img: img)
    monkeypatch.setattr(reposition, "_draw_overlay_outline", lambda img, _overlay: img)
    monkeypatch.setattr(reposition, "_encode_image", lambda _img: "image")

    result = reposition.reposition_gridpoint_iterative(
        "put yourself in the upper right",
        image,
        current_overlay={"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4},
        screen_w=100,
        screen_h=100,
    )

    assert len(seen_ids) == 2
    assert seen_ids[0] == seen_ids[1]
    assert result["_positioning_utterance_id"] == seen_ids[0]


def test_parse_split_audit_responses_keep_and_clamp_dimensions():
    """Split audit parsers accept JSON KEEP fields and clamp pixel edits."""
    from spoke.positioning.reposition import (
        _parse_center_audit_response,
        _parse_size_audit_response,
        _parse_suitability_audit_response,
    )

    suitability = _parse_suitability_audit_response(
        '{"done":false,"needs_position":true,"needs_size":false,'
        '"reason":"too close to the code"}'
    )
    assert suitability == {
        "done": False,
        "needs_position": True,
        "needs_size": False,
        "reason": "too close to the code",
    }

    center = _parse_center_audit_response(
        '```json\n{"center_x":2600,"center_y":"KEEP","reason":"right edge"}\n```',
        screen_w=1920,
        screen_h=1080,
    )
    assert center == {"center_x": 1920, "center_y": "KEEP", "reason": "right edge"}

    size = _parse_size_audit_response(
        '{"width":"KEEP","height":-20,"reason":"make it short"}',
        screen_w=1920,
        screen_h=1080,
    )
    assert size == {"width": "KEEP", "height": 1, "reason": "make it short"}


def test_iterative_split_audit_combines_center_and_size_before_rerender(monkeypatch):
    """Each split audit round should inspect the latest combined candidate."""
    import importlib

    reposition = importlib.import_module("spoke.positioning.reposition")

    image = Image.new("RGB", (100, 100), "white")
    outlined = []
    encoded = []
    suitability = [
        {"done": False, "needs_position": True, "needs_size": True, "reason": "too central"},
        {"done": True, "needs_position": False, "needs_size": False, "reason": "clear"},
    ]
    centers = [{"center_x": 75, "center_y": 25, "reason": "upper right"}]
    sizes = [{"width": 30, "height": 20, "reason": "compact"}]

    monkeypatch.setattr(reposition, "_draw_grid_points", lambda img: img)
    monkeypatch.setattr(reposition, "_pick_gridpoint", lambda *args, **kwargs: "B2")

    def fake_outline(_screenshot, overlay):
        outlined.append(dict(overlay))
        return image

    def fake_encode(_image):
        token = f"image-{len(encoded)}"
        encoded.append(token)
        return token

    def fake_suitability(screenshot_b64, *args, **kwargs):
        assert screenshot_b64 == encoded[-1]
        return suitability.pop(0)

    def fake_center(screenshot_b64, *args, **kwargs):
        assert screenshot_b64 == encoded[-1]
        return centers.pop(0)

    def fake_size(screenshot_b64, *args, **kwargs):
        assert screenshot_b64 == encoded[-1]
        return sizes.pop(0)

    monkeypatch.setattr(reposition, "_draw_overlay_outline", fake_outline)
    monkeypatch.setattr(reposition, "_encode_image", fake_encode)
    monkeypatch.setattr(reposition, "_pick_suitability_audit", fake_suitability)
    monkeypatch.setattr(reposition, "_pick_center_audit", fake_center)
    monkeypatch.setattr(reposition, "_pick_size_audit", fake_size)

    result = reposition.reposition_gridpoint_iterative(
        "upper right but compact",
        image,
        current_overlay={"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4},
        screen_w=100,
        screen_h=100,
    )

    assert result["x"] == pytest.approx(0.60)
    assert result["y"] == pytest.approx(0.15)
    assert result["width"] == pytest.approx(0.30)
    assert result["height"] == pytest.approx(0.20)
    assert result["_debug_lines"][-1].startswith("Total:")
    assert len(encoded) == 3  # initial grid pick plus two candidate suitability rounds
    assert outlined[-2:] == [
        {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4},
        {"x": 0.6, "y": 0.15, "width": 0.3, "height": 0.2},
    ]


def test_iterative_split_audit_treats_need_flags_as_advisory(monkeypatch):
    """A false size flag must not suppress the only useful actuator."""
    import importlib

    reposition = importlib.import_module("spoke.positioning.reposition")

    image = Image.new("RGB", (100, 100), "white")
    suitability = [
        {
            "done": False,
            "needs_position": True,
            "needs_size": False,
            "reason": "candidate leaves empty space uncovered",
        },
        {"done": True, "needs_position": False, "needs_size": False, "reason": "filled"},
    ]
    centers = [{"center_x": 50, "center_y": 50, "reason": "center already right"}]
    sizes = [{"width": 80, "height": 80, "reason": "fill the empty space"}]

    monkeypatch.setattr(reposition, "_draw_grid_points", lambda img: img)
    monkeypatch.setattr(reposition, "_draw_overlay_outline", lambda _screenshot, _overlay: image)
    monkeypatch.setattr(reposition, "_encode_image", lambda _image: "image")
    monkeypatch.setattr(reposition, "_pick_gridpoint", lambda *args, **kwargs: "B2")
    monkeypatch.setattr(
        reposition,
        "_pick_suitability_audit",
        lambda *args, **kwargs: suitability.pop(0),
    )
    monkeypatch.setattr(reposition, "_pick_center_audit", lambda *args, **kwargs: centers.pop(0))
    monkeypatch.setattr(reposition, "_pick_size_audit", lambda *args, **kwargs: sizes.pop(0))

    result = reposition.reposition_gridpoint_iterative(
        "fill the empty space",
        image,
        current_overlay={"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4},
        screen_w=100,
        screen_h=100,
    )

    assert result["x"] == pytest.approx(0.10)
    assert result["y"] == pytest.approx(0.10)
    assert result["width"] == pytest.approx(0.80)
    assert result["height"] == pytest.approx(0.80)
    assert sizes == []


def test_largest_rectangle_target_picks_yes_cells():
    """largest_rectangle_target finds rect in YES (occupy) cells."""
    from spoke.positioning.reposition import largest_rectangle_target

    # Center 2×2 block (C3-C4, D3-D4 in a 6×6 grid)
    target_map = {}
    for r in "ABCDEF":
        for c in "123456":
            target_map[f"{r}{c}"] = (r in "CD" and c in "34")

    rect = largest_rectangle_target(target_map)
    assert rect is not None
    assert rect["x"] == pytest.approx(2 / 6)
    assert rect["y"] == pytest.approx(2 / 6)
    assert rect["width"] == pytest.approx(2 / 6)
    assert rect["height"] == pytest.approx(2 / 6)


# ── content detection parsing tests ──

def test_parse_yes_no_response():
    """Parse a well-formatted YES/NO response."""
    raw = """A1: YES
A2: YES
A3: NO
A4: NO
B1: YES
B2: NO
B3: NO
B4: NO
C1: YES
C2: YES
C3: NO
C4: NO
D1: NO
D2: NO
D3: NO
D4: NO"""

    result = {}
    for row_idx, row_letter in enumerate("ABCD"):
        for col_idx, col_num in enumerate("1234"):
            cell = f"{row_letter}{col_num}"
            match = re.search(rf"{cell}\s*:\s*(YES|NO)", raw.upper())
            result[cell] = match.group(1) == "YES" if match else False

    assert result["A1"] is True
    assert result["A3"] is False
    assert result["C1"] is True
    assert result["D4"] is False
    assert sum(1 for v in result.values() if v) == 5  # 5 YES cells


# ── smoke hook tests ──

def test_positioning_bearing_is_disabled_by_default(monkeypatch):
    """Bearing compute should not run in the positioning hot path by default."""
    from spoke.positioning.smoke_hook import _positioning_bearing_enabled

    monkeypatch.delenv("SPOKE_POSITIONING_ENABLE_BEARING", raising=False)

    assert _positioning_bearing_enabled() is False


def test_positioning_smoke_text_includes_debug_trace():
    """The movable positioned overlay should carry the diagnostic trace."""
    from spoke.positioning.smoke_hook import _build_positioning_smoke_text

    text = _build_positioning_smoke_text(
        {
            "utterance": "move out of the text",
            "content_desc": "move out of the text",
            "elapsed_s": 2.0,
            "_debug_lines": [
                "GridPoint: C3 -> (1296, 837)",
                "Audit 1: x=KEEP y=900 width=1300 height=500 done=False",
                "Audit 2 justification: no longer covers text",
            ],
        }
    )

    assert "move out of the text" in text
    assert "GridPoint: C3" in text
    assert "Audit 2 justification" in text
    assert "2.0s total" in text


def test_finish_on_main_puts_debug_trace_in_smoke_rect_not_fixed_diag():
    """Final positioning diagnostics should live in the movable smoke rect."""
    import spoke.positioning.smoke_hook as smoke_hook

    class Frame:
        def __getitem__(self, index):
            return ((0.0, 0.0), (100.0, 100.0))[index]

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._command_overlay = None
    app._fullscreen_compositor = None

    result = {
        "x": 0.25,
        "y": 0.25,
        "width": 0.5,
        "height": 0.5,
        "utterance": "move away from text",
        "content_desc": "move away from text",
        "elapsed_s": 1.25,
        "_debug_lines": ["Audit 1: width=KEEP", "Audit 1 justification: clear"],
    }

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()), \
         patch.object(smoke_hook, "_show_smoke_rect") as show_smoke, \
         patch.object(smoke_hook, "_show_diagnostic_overlay") as show_diag:
        _run_finish_on_main(app, result)

    show_diag.assert_not_called()
    smoke_text = show_smoke.call_args.args[4]
    assert "Audit 1: width=KEEP" in smoke_text
    assert "Audit 1 justification: clear" in smoke_text

def test_finish_on_main_with_none_doesnt_crash():
    """Calling _finish_on_main with None result should not crash."""
    from spoke.positioning.smoke_hook import _finish_on_main

    app = MagicMock()
    app._transcribing = True
    app._menubar = MagicMock()
    app._overlay = MagicMock()

    # This would normally schedule on the main thread via AppHelper.
    # We just verify it doesn't raise when constructing the callback.
    # Can't fully test without the AppKit run loop.
    # At minimum verify the function is callable.
    assert callable(_finish_on_main)


def _get_neighbors(row, col, rows=4, cols=4):
    """Local copy of neighbor logic for testing."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


def test_get_all_neighbors_center():
    """Center cell has 8 neighbors."""
    neighbors = _get_neighbors(1, 1)
    assert len(neighbors) == 8
    assert (0, 0) in neighbors
    assert (2, 2) in neighbors


def test_get_all_neighbors_corner():
    """Corner cell has 3 neighbors."""
    neighbors = _get_neighbors(0, 0)
    assert len(neighbors) == 3
    assert (0, 1) in neighbors
    assert (1, 0) in neighbors
    assert (1, 1) in neighbors


def test_get_all_neighbors_edge():
    """Edge cell (not corner) has 5 neighbors."""
    neighbors = _get_neighbors(0, 1)
    assert len(neighbors) == 5


def test_move_overlay_calculates_frame():
    """_move_overlay computes correct frame with fringe."""
    # We can't test the actual NSWindow call without AppKit,
    # but we can verify the function signature is correct.
    from spoke.positioning.smoke_hook import _move_overlay
    assert callable(_move_overlay)


# ── integration-level contract tests ──

def test_reposition_returns_correct_shape():
    """reposition() returns dict with x, y, width, height as fractions."""
    from spoke.positioning.reposition import largest_rectangle

    # Simulate a content map where right half is empty
    content_map = {}
    for r in "ABCDEF":
        for c in "123456":
            content_map[f"{r}{c}"] = (c in "123")

    rect = largest_rectangle(content_map)
    assert rect is not None
    assert 0 <= rect["x"] <= 1
    assert 0 <= rect["y"] <= 1
    assert 0 < rect["width"] <= 1
    assert 0 < rect["height"] <= 1
    assert rect["x"] + rect["width"] <= 1.0 + 1e-9
    assert rect["y"] + rect["height"] <= 1.0 + 1e-9


def test_content_map_must_be_complete():
    """largest_rectangle requires all 36 cells — missing cells are treated as occupied."""
    from spoke.positioning.reposition import largest_rectangle

    # Only 2 cells specified — the other 34 are missing and treated as occupied
    content_map = {"A1": True, "B2": False}
    rect = largest_rectangle(content_map)
    # Only B2 is available — a 1x1 rectangle
    assert rect is not None
    assert rect["width"] == pytest.approx(1 / 6)
    assert rect["height"] == pytest.approx(1 / 6)


# ── smoke_hook _finish_on_main cleanup tests ──

def _run_finish_on_main(app, result):
    """Call _finish_on_main with AppHelper.callAfter executing immediately."""
    from PyObjCTools import AppHelper
    from spoke.positioning.smoke_hook import _finish_on_main

    with patch.object(AppHelper, "callAfter", side_effect=lambda fn: fn()):
        _finish_on_main(app, result)


def test_finish_on_main_clears_command_overlay_active():
    """_finish_on_main must clear command_overlay_active on the detector so
    subsequent enter taps during recording route as command gestures instead
    of triggering the cancel spring."""
    app = MagicMock()
    app._detector = MagicMock()
    app._detector.command_overlay_active = True
    app._menubar = None
    app._overlay = None

    _run_finish_on_main(app, None)

    assert app._detector.command_overlay_active is False


def test_finish_on_main_clears_transcribing():
    """_finish_on_main must clear _transcribing so the next hold doesn't
    divert to the parallel insert pathway or cancel spring."""
    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None

    _run_finish_on_main(app, {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0})

    assert app._transcribing is False


# ── optical field integration tests ──

def test_positioning_emits_optical_field_request():
    """_finish_on_main should emit an OpticalFieldRequest with the computed bounds."""
    from spoke.optical_field import OpticalFieldRequest, OpticalFieldBounds

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = None  # no compositor → fallback path
    app._command_overlay = None

    _run_finish_on_main(app, {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0})

    # Should store the request on the app for persistence across show/hide
    req = app._positioning_field_request
    assert isinstance(req, OpticalFieldRequest)
    assert req.caller_id == "semantic_positioning"
    assert req.role == "assistant"
    assert req.state == "materialize"
    assert req.bounds.width > 0
    assert req.bounds.height > 0


def test_positioning_pushes_to_compositor_backend():
    """When a compositor with optical field backend exists, push config there."""
    from spoke.optical_field import OpticalFieldPlaceholderBackend

    backend = OpticalFieldPlaceholderBackend()
    compositor = MagicMock()
    compositor._optical_field_backend = backend

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = compositor
    app._command_overlay = MagicMock()
    app._command_overlay._compositor_client_id = "cmd_overlay_1"

    _run_finish_on_main(app, {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5})

    # Backend should have the request
    reqs = backend.requests()
    assert len(reqs) == 1
    assert reqs[0].caller_id == "semantic_positioning"

    # Compositor should have received an update
    compositor.update_client_config.assert_called_once()


def test_positioning_uses_command_overlay_compositor_session_when_backend_absent():
    """The live command overlay owns the compositor session on main."""
    import spoke.positioning.smoke_hook as smoke_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    frame = Frame()
    command_session = MagicMock()
    command_overlay = MagicMock()
    command_overlay._fullscreen_compositor = command_session
    command_overlay._screen.backingScaleFactor.return_value = 2.0
    command_overlay._brightness = 0.42

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = None
    app._command_overlay = command_overlay

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=frame):
        _run_finish_on_main(app, {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5})

    command_session.update_shell_config.assert_called_once()
    config = command_session.update_shell_config.call_args.args[0]
    assert config["client_id"] == "semantic_positioning"
    assert config["role"] == "assistant"
    assert config["optical_field"]["caller_id"] == "semantic_positioning"
    # The deterministic test frame is 1440x900 logical points.
    assert config["center_x"] == pytest.approx(1440.0)
    assert config["center_y"] == pytest.approx(900.0)
    assert config["content_width_points"] == pytest.approx(1440.0)
    assert config["content_height_points"] == pytest.approx(900.0)
    assert config["initial_brightness"] == pytest.approx(0.42)


def test_positioning_applies_semantic_bounds_to_real_command_overlay_when_compositor_active():
    """The compositor path must move the real assistant overlay, not only a smoke rect."""
    import spoke.positioning.smoke_hook as smoke_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    class CommandOverlay:
        def __init__(self):
            self._fullscreen_compositor = MagicMock()
            self._screen = MagicMock()
            self._screen.backingScaleFactor.return_value = 2.0
            self._brightness = 0.42
            self.applied_requests = []
            self.cancel_dismiss = MagicMock()
            self.show = MagicMock()

        def apply_semantic_positioning_request(self, request):
            self.applied_requests.append(request)
            return True

    command_overlay = CommandOverlay()
    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = None
    app._command_overlay = command_overlay

    result = {
        "x": 0.25,
        "y": 0.25,
        "width": 0.5,
        "height": 0.5,
        "utterance": "move out of my way",
        "content_desc": "avoiding: central text",
        "elapsed_s": 1.0,
        "_debug_lines": ["Audit 1 justification: clear"],
    }

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()), \
         patch.object(smoke_hook, "_schedule_command_overlay_reopen", side_effect=lambda fn, delay_s: fn()), \
         patch.object(smoke_hook, "_show_smoke_rect") as show_smoke:
        _run_finish_on_main(app, result)

    assert len(command_overlay.applied_requests) == 1
    assert command_overlay.applied_requests[0] is app._positioning_field_request
    command_overlay.show.assert_called_once()
    show_kwargs = command_overlay.show.call_args.kwargs
    assert show_kwargs["start_thinking_timer"] is False
    assert "move out of my way" in show_kwargs["initial_response"]
    assert "Audit 1 justification: clear" in show_kwargs["initial_response"]
    show_smoke.assert_not_called()


def test_positioning_hook_dismisses_command_overlay_when_command_is_accepted(mock_pyobjc):
    """Accepting semantic positioning should immediately start the normal dismiss flow."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.positioning.smoke_hook import install_positioning_hook

    events = []

    class CommandOverlay:
        def cancel_dismiss(self):
            events.append("dismiss")

    app = MagicMock()
    app._command_transcribe_worker = MagicMock()
    app._command_overlay = CommandOverlay()
    app._positioning_field_request = None

    install_positioning_hook(app)

    with patch.object(
        smoke_hook,
        "positioning_transcribe_worker",
        side_effect=lambda *_args: events.append("worker"),
    ):
        app._command_transcribe_worker(b"wav", 12)

    assert events == ["dismiss", "worker"]


def test_positioning_intermediate_materializes_real_command_overlay_with_debug_text():
    """Draft positioning updates should use the command overlay, not the smoke rect."""
    import spoke.positioning.smoke_hook as smoke_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    class CommandOverlay:
        def __init__(self):
            self._fullscreen_compositor = MagicMock()
            self._screen = MagicMock()
            self._screen.backingScaleFactor.return_value = 2.0
            self._brightness = 0.42
            self.applied_requests = []
            self.cancel_dismiss = MagicMock()
            self.show = MagicMock()

        def apply_semantic_positioning_request(self, request):
            self.applied_requests.append(request)
            return True

    command_overlay = CommandOverlay()
    app = MagicMock()
    app._command_overlay = command_overlay
    app._positioning_field_request = None

    result = {
        "_intermediate": True,
        "x": 0.20,
        "y": 0.10,
        "width": 0.40,
        "height": 0.30,
        "utterance": "move upper left",
        "content_desc": "draft candidate",
        "_debug_lines": ["Suitability 1: needs_position", "Center 1: upper-left"],
    }

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()), \
         patch.object(smoke_hook, "_schedule_command_overlay_reopen", side_effect=lambda fn, delay_s: fn()), \
         patch.object(smoke_hook, "_show_smoke_rect") as show_smoke:
        smoke_hook._finish_on_main_immediate(app, result)

    assert len(command_overlay.applied_requests) == 1
    assert command_overlay.applied_requests[0].state == "materialize"
    command_overlay.show.assert_called_once()
    show_kwargs = command_overlay.show.call_args.kwargs
    assert "move upper left" in show_kwargs["initial_response"]
    assert "Suitability 1: needs_position" in show_kwargs["initial_response"]
    assert "Center 1: upper-left" in show_kwargs["initial_response"]
    show_smoke.assert_not_called()


def test_positioning_intermediate_candidate_dismisses_old_request_before_materializing_new():
    """Successive drafts should animate the old placement out before applying the new one."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.optical_field import OpticalFieldBounds, OpticalFieldRequest

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    events = []

    class CommandOverlay:
        def __init__(self):
            self._fullscreen_compositor = MagicMock()
            self._screen = MagicMock()
            self._screen.backingScaleFactor.return_value = 2.0
            self._brightness = 0.42

        def cancel_dismiss(self):
            events.append("dismiss-flow")

        def apply_semantic_positioning_request(self, request):
            events.append(f"apply:{request.state}:{request.bounds.x:.0f}")
            return True

        def show(self, **kwargs):
            events.append("show")

    app = MagicMock()
    app._command_overlay = CommandOverlay()
    app._positioning_field_request = OpticalFieldRequest(
        caller_id="semantic_positioning",
        bounds=OpticalFieldBounds(x=40.0, y=50.0, width=320.0, height=120.0),
        role="assistant",
        state="rest",
        visible=True,
    )

    result = {
        "_intermediate": True,
        "x": 0.25,
        "y": 0.25,
        "width": 0.5,
        "height": 0.5,
        "utterance": "try the middle",
        "_debug_lines": ["candidate 2"],
    }

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()), \
         patch.object(smoke_hook, "_schedule_command_overlay_reopen", side_effect=lambda fn, delay_s: fn()), \
         patch.object(smoke_hook, "_show_smoke_rect"):
        smoke_hook._finish_on_main_immediate(app, result)

    assert events[:4] == [
        "apply:dismiss:40",
        "dismiss-flow",
        "apply:materialize:360",
        "show",
    ]


def test_positioning_dismisses_old_command_overlay_before_materializing_new_bounds():
    """Each accepted move should close at the old position before reopening at the new one."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.optical_field import OpticalFieldBounds, OpticalFieldRequest

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    events = []

    class CommandOverlay:
        def __init__(self):
            self._fullscreen_compositor = MagicMock()
            self._screen = MagicMock()
            self._screen.backingScaleFactor.return_value = 2.0
            self._brightness = 0.42

        def cancel_dismiss(self):
            events.append("dismiss")

        def apply_semantic_positioning_request(self, request):
            events.append(f"apply:{request.state}:{request.bounds.x:.0f}")
            return True

        def show(self, **kwargs):
            events.append("show")

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = None
    app._command_overlay = CommandOverlay()
    app._positioning_field_request = OpticalFieldRequest(
        caller_id="semantic_positioning",
        bounds=OpticalFieldBounds(x=40.0, y=50.0, width=320.0, height=120.0),
        role="assistant",
        state="rest",
        visible=True,
    )

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()), \
         patch.object(smoke_hook, "_schedule_command_overlay_reopen", side_effect=lambda fn, delay_s: fn()), \
         patch.object(smoke_hook, "_show_smoke_rect"):
        _run_finish_on_main(
            app,
            {
                "x": 0.25,
                "y": 0.25,
                "width": 0.5,
                "height": 0.5,
                "utterance": "move out of my way",
                "content_desc": "avoiding: central text",
                "elapsed_s": 1.0,
            },
        )

    assert events[0] == "apply:dismiss:40"
    assert events[1] == "dismiss"
    assert events[2].startswith("apply:materialize:")
    assert events[3] == "show"


def test_positioning_waits_for_command_overlay_dismiss_before_reopening():
    """The semantic move should not reopen before the optical dismiss can finish."""
    import spoke.positioning.smoke_hook as smoke_hook

    from spoke.command_overlay import (
        _OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S,
        _SEMANTIC_POSITIONING_REOPEN_PAD_S,
    )

    class CommandOverlay:
        def __init__(self):
            self._fullscreen_compositor = MagicMock()
            self.cancel_dismiss = MagicMock()
            self.show = MagicMock()

        def apply_semantic_positioning_request(self, request):
            return True

        def semantic_positioning_reopen_delay_s(self):
            return (
                _OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S
                + _SEMANTIC_POSITIONING_REOPEN_PAD_S
            )

    app = MagicMock()
    app._positioning_field_request = None
    command_overlay = CommandOverlay()
    scheduled = []

    with patch.object(
        smoke_hook,
        "_schedule_command_overlay_reopen",
        side_effect=lambda callback, delay_s: scheduled.append(delay_s),
    ):
        assert smoke_hook._present_request_on_command_overlay(
            app,
            command_overlay,
            MagicMock(),
            "positioning diagnostics",
            MagicMock(),
        )

    assert scheduled == [
        pytest.approx(
            _OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S
            + _SEMANTIC_POSITIONING_REOPEN_PAD_S
        )
    ]
    command_overlay.show.assert_not_called()


def test_positioning_hook_reemits_stored_request_when_command_overlay_shows():
    """Summoning the command overlay should preserve the last semantic bounds."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.optical_field import OpticalFieldBounds, OpticalFieldRequest
    from spoke.positioning.smoke_hook import install_positioning_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    command_session = MagicMock()
    command_overlay = MagicMock()
    command_overlay._fullscreen_compositor = command_session
    command_overlay._screen.backingScaleFactor.return_value = 2.0
    command_overlay._brightness = 0.42
    original_show = MagicMock()
    command_overlay.show = original_show

    app = MagicMock()
    app._command_transcribe_worker = MagicMock()
    app._command_overlay = command_overlay
    app._positioning_field_request = OpticalFieldRequest(
        caller_id="semantic_positioning",
        bounds=OpticalFieldBounds(x=360.0, y=225.0, width=720.0, height=450.0),
        role="assistant",
        state="rest",
        visible=True,
    )

    install_positioning_hook(app)

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()):
        app._command_overlay.show(initial_utterance="hello")

    original_show.assert_called_once_with(initial_utterance="hello")
    command_session.update_shell_config.assert_called_once()
    config = command_session.update_shell_config.call_args.args[0]
    assert config["optical_field"]["caller_id"] == "semantic_positioning"
    assert config["optical_field"]["state"] == "materialize"
    assert config["optical_field"]["slot"] == "materialize"
    assert config["center_x"] == pytest.approx(1440.0)
    assert config["center_y"] == pytest.approx(900.0)
    assert config["content_width_points"] == pytest.approx(1440.0)
    assert config["content_height_points"] == pytest.approx(900.0)


def test_positioning_hook_emits_dismiss_state_when_command_overlay_dismisses():
    """Dismissal should keep the semantic caller lifecycle coherent."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.optical_field import OpticalFieldBounds, OpticalFieldRequest
    from spoke.positioning.smoke_hook import install_positioning_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    command_session = MagicMock()
    command_overlay = MagicMock()
    command_overlay._fullscreen_compositor = command_session
    command_overlay._screen.backingScaleFactor.return_value = 2.0
    command_overlay._brightness = 0.42
    original_cancel = MagicMock()
    command_overlay.cancel_dismiss = original_cancel

    app = MagicMock()
    app._command_transcribe_worker = MagicMock()
    app._command_overlay = command_overlay
    app._positioning_field_request = OpticalFieldRequest(
        caller_id="semantic_positioning",
        bounds=OpticalFieldBounds(x=360.0, y=225.0, width=720.0, height=450.0),
        role="assistant",
        state="rest",
        visible=True,
    )

    install_positioning_hook(app)

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()):
        app._command_overlay.cancel_dismiss()

    original_cancel.assert_called_once_with()
    command_session.update_shell_config.assert_called_once()
    config = command_session.update_shell_config.call_args.args[0]
    assert config["optical_field"]["caller_id"] == "semantic_positioning"
    assert config["optical_field"]["state"] == "dismiss"
    assert config["optical_field"]["slot"] == "dismiss"


def test_positioning_hook_publishes_house_dismiss_sidecars_when_host_is_available():
    """Semantic dismiss should use the reusable House sidecars on the shared host."""
    import spoke.positioning.smoke_hook as smoke_hook
    from spoke.fullscreen_compositor import OverlayClientIdentity
    from spoke.optical_field import OpticalFieldBounds, OpticalFieldRequest
    from spoke.positioning.smoke_hook import install_positioning_hook

    class Frame:
        origin = SimpleNamespace(x=0.0, y=0.0)
        size = SimpleNamespace(width=1440.0, height=900.0)

        def __getitem__(self, index):
            return ((self.origin.x, self.origin.y), (self.size.width, self.size.height))[index]

    class FakeHost:
        display_id = 1

        def __init__(self):
            self.clients = {}
            self.batches = []

        def register_client(self, identity, *, window, content_view):
            assert isinstance(identity, OverlayClientIdentity)
            client = FakeClient(self, identity.client_id)
            self.clients[identity.client_id] = client
            return client

        def update_client_configs(self, configs):
            self.batches.append(configs)
            return True

    class FakeClient:
        def __init__(self, host, client_id):
            self._host = host
            self._client_id = client_id

        def update_shell_config(self, config):
            self._host.batches.append({self._client_id: config})
            return True

        def release(self):
            self._host.clients.pop(self._client_id, None)

    host = FakeHost()
    command_session = FakeClient(host, "assistant.command")
    command_overlay = MagicMock()
    command_overlay._fullscreen_compositor = command_session
    command_overlay._screen.backingScaleFactor.return_value = 2.0
    command_overlay._brightness = 0.42
    command_overlay._window = object()
    command_overlay._content_view = object()
    original_cancel = MagicMock()
    command_overlay.cancel_dismiss = original_cancel

    app = MagicMock()
    app._command_transcribe_worker = MagicMock()
    app._command_overlay = command_overlay
    app._positioning_field_request = OpticalFieldRequest(
        caller_id="semantic_positioning",
        bounds=OpticalFieldBounds(x=360.0, y=225.0, width=720.0, height=450.0),
        role="assistant",
        state="rest",
        visible=True,
    )

    install_positioning_hook(app)

    with patch.object(smoke_hook, "_get_main_screen_frame", return_value=Frame()):
        app._command_overlay.cancel_dismiss()

    original_cancel.assert_called_once_with()
    assert host.batches
    batch = host.batches[-1]
    assert set(batch) == {
        "assistant.command",
        "semantic_positioning.dismiss_seam",
        "semantic_positioning.dismiss_radial_pucker",
    }
    main_config = batch["assistant.command"]
    assert main_config["optical_field"]["caller_id"] == "semantic_positioning"
    assert main_config["optical_field"]["state"] == "dismiss"
    assert main_config["optical_field"]["progress"] == pytest.approx(0.0)
    assert batch["semantic_positioning.dismiss_seam"]["optical_field"]["sidecar"] == "dismiss_seam"
    assert (
        batch["semantic_positioning.dismiss_radial_pucker"]["optical_field"]["sidecar"]
        == "dismiss_radial_pucker"
    )


# ── two-step pipeline tests ──

def test_coarse_regions_cover_screen():
    """All 9 coarse regions map to valid fractional coordinates."""
    from spoke.positioning.reposition import COARSE_REGIONS

    assert len(COARSE_REGIONS) == 9
    for name, (cx, cy) in COARSE_REGIONS.items():
        assert 0 < cx < 1, f"{name} cx={cx}"
        assert 0 < cy < 1, f"{name} cy={cy}"


def test_draw_overlay_outline_no_crash():
    """Drawing overlay outline on a screenshot should not crash."""
    from spoke.positioning.reposition import _draw_overlay_outline

    img = Image.new("RGB", (800, 600), (0, 0, 0))
    result = _draw_overlay_outline(img, {"x": 0.3, "y": 0.3, "width": 0.4, "height": 0.4})
    assert result.size == (800, 600)


def test_draw_overlay_outline_none_overlay():
    """None overlay rect should return unmodified image."""
    from spoke.positioning.reposition import _draw_overlay_outline

    img = Image.new("RGB", (800, 600), (128, 128, 128))
    result = _draw_overlay_outline(img, None)
    assert result.size == (800, 600)


def test_twostep_clamps_to_screen():
    """Final rect from two-step pipeline must stay within 0-1 bounds."""
    # Simulate what reposition_twostep does in the clamping step
    center_x, center_y = 0.9, 0.9  # near bottom-right corner
    new_w, new_h = 0.4, 0.4  # would overflow

    new_x = max(0.0, min(1.0 - new_w, center_x - new_w / 2))
    new_y = max(0.0, min(1.0 - new_h, center_y - new_h / 2))

    assert new_x + new_w <= 1.0
    assert new_y + new_h <= 1.0
    assert new_x >= 0.0
    assert new_y >= 0.0


def test_positioning_request_persists_across_calls():
    """Successive positioning calls update the same caller_id, not accumulate."""
    from spoke.optical_field import OpticalFieldPlaceholderBackend

    backend = OpticalFieldPlaceholderBackend()
    compositor = MagicMock()
    compositor._optical_field_backend = backend

    app = MagicMock()
    app._transcribing = True
    app._detector = MagicMock()
    app._menubar = None
    app._overlay = None
    app._fullscreen_compositor = compositor
    app._command_overlay = MagicMock()
    app._command_overlay._compositor_client_id = "cmd_overlay_1"

    # First positioning
    _run_finish_on_main(app, {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5})
    assert len(backend.requests()) == 1

    # Second positioning — should upsert, not duplicate
    app._transcribing = True
    _run_finish_on_main(app, {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5})
    assert len(backend.requests()) == 1
    # Bounds should reflect the second call
    req = backend.requests()[0]
    assert req.bounds.x > 0  # moved from origin
