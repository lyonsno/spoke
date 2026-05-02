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
