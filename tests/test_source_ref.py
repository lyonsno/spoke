"""Tests for source ref resolution.

Tests the SourceRef dataclass and the resolve() function that maps
refs to exact text from scene captures, clipboard, selection, etc.
"""

from __future__ import annotations

import importlib
import sys
import time
from unittest.mock import MagicMock, patch

import pytest


def _import_scene():
    sys.modules.pop("spoke.scene_capture", None)
    return importlib.import_module("spoke.scene_capture")


def _import_module():
    sys.modules.pop("spoke.source_ref", None)
    return importlib.import_module("spoke.source_ref")


# ── SourceRef dataclass ──────────────────────────────────────────


class TestSourceRef:
    def test_construction(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="literal", value="hello world")
        assert ref.kind == "literal"
        assert ref.value == "hello world"

    def test_all_kinds_are_valid(self):
        mod = _import_module()
        for kind in ("last_response", "selection", "clipboard", "scene_block", "ax_hint", "literal"):
            ref = mod.SourceRef(kind=kind, value="test")
            assert ref.kind == kind


# ── parse_ref ────────────────────────────────────────────────────


class TestParseRef:
    def test_literal(self):
        mod = _import_module()
        ref = mod.parse_ref("literal:hello world")
        assert ref.kind == "literal"
        assert ref.value == "hello world"

    def test_clipboard(self):
        mod = _import_module()
        ref = mod.parse_ref("clipboard:current")
        assert ref.kind == "clipboard"
        assert ref.value == "current"

    def test_selection(self):
        mod = _import_module()
        ref = mod.parse_ref("selection:frontmost")
        assert ref.kind == "selection"
        assert ref.value == "frontmost"

    def test_last_response(self):
        mod = _import_module()
        ref = mod.parse_ref("last_response:current")
        assert ref.kind == "last_response"
        assert ref.value == "current"

    def test_scene_block(self):
        mod = _import_module()
        ref = mod.parse_ref("scene_block:scene-abc:block-1")
        assert ref.kind == "scene_block"
        assert ref.value == "scene-abc:block-1"

    def test_ax_hint(self):
        mod = _import_module()
        ref = mod.parse_ref("ax_hint:scene-abc:focus")
        assert ref.kind == "ax_hint"
        assert ref.value == "scene-abc:focus"

    def test_bare_scene_block_ref(self):
        """Models send bare refs from capture_context without kind prefix."""
        mod = _import_module()
        ref = mod.parse_ref("scene-abc123:block-5")
        assert ref.kind == "scene_block"
        assert ref.value == "scene-abc123:block-5"

    def test_bare_ax_hint_ref(self):
        """Bare AX hint ref inferred from scene-*:focus pattern."""
        mod = _import_module()
        ref = mod.parse_ref("scene-abc123:focus")
        assert ref.kind == "ax_hint"
        assert ref.value == "scene-abc123:focus"

    def test_prefixed_scene_block_still_works(self):
        """Explicit scene_block: prefix should still parse correctly."""
        mod = _import_module()
        ref = mod.parse_ref("scene_block:scene-abc:block-1")
        assert ref.kind == "scene_block"
        assert ref.value == "scene-abc:block-1"

    def test_invalid_ref_raises(self):
        mod = _import_module()
        with pytest.raises(ValueError, match="Unknown source ref kind"):
            mod.parse_ref("unknown_kind:value")

    def test_no_colon_raises(self):
        mod = _import_module()
        with pytest.raises(ValueError, match="Invalid source ref"):
            mod.parse_ref("justakind")


# ── resolve ──────────────────────────────────────────────────────


class TestResolve:
    def _make_cache_with_capture(self):
        """Build a SceneCaptureCache with one capture for testing."""
        sc = _import_scene()
        cache = sc.SceneCaptureCache(max_captures=5)
        cap = sc.SceneCapture(
            scene_ref="scene-abc",
            created_at=time.time(),
            scope="active_window",
            app_name="Safari",
            bundle_id=None,
            window_title="Test Page",
            image_path="/tmp/test.png",
            image_size=(2560, 1440),
            model_image_size=(1280, 720),
            ocr_text="Hello World",
            ocr_blocks=[
                sc.OCRBlock(ref="scene-abc:block-0", text="Hello", bbox=(0, 0, 50, 20), confidence=0.99),
                sc.OCRBlock(ref="scene-abc:block-1", text="World", bbox=(60, 0, 50, 20), confidence=0.95),
            ],
            ax_hints=[
                sc.AXHint(ref="scene-abc:focus", role="AXTextField", label="Search", value="query text"),
            ],
        )
        cache.store(cap)
        return cache

    def test_resolve_literal(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="literal", value="hello world")
        text = mod.resolve(ref)
        assert text == "hello world"

    def test_resolve_scene_block(self):
        mod = _import_module()
        cache = self._make_cache_with_capture()
        ref = mod.SourceRef(kind="scene_block", value="scene-abc:block-0")
        text = mod.resolve(ref, scene_cache=cache)
        assert text == "Hello"

    def test_resolve_scene_block_not_found(self):
        mod = _import_module()
        cache = self._make_cache_with_capture()
        ref = mod.SourceRef(kind="scene_block", value="scene-zzz:block-0")
        with pytest.raises(mod.RefResolutionError, match="not found"):
            mod.resolve(ref, scene_cache=cache)

    def test_resolve_scene_block_no_cache(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="scene_block", value="scene-abc:block-0")
        with pytest.raises(mod.RefResolutionError, match="No scene cache"):
            mod.resolve(ref)

    def test_resolve_ax_hint(self):
        mod = _import_module()
        cache = self._make_cache_with_capture()
        ref = mod.SourceRef(kind="ax_hint", value="scene-abc:focus")
        text = mod.resolve(ref, scene_cache=cache)
        assert text == "query text"

    def test_resolve_ax_hint_not_found(self):
        mod = _import_module()
        cache = self._make_cache_with_capture()
        ref = mod.SourceRef(kind="ax_hint", value="scene-abc:nonexistent")
        with pytest.raises(mod.RefResolutionError, match="not found"):
            mod.resolve(ref, scene_cache=cache)

    def test_resolve_clipboard(self):
        mod = _import_module()
        with patch.object(mod, "_get_clipboard_text", return_value="clipboard content"):
            ref = mod.SourceRef(kind="clipboard", value="current")
            text = mod.resolve(ref)
            assert text == "clipboard content"

    def test_resolve_clipboard_empty(self):
        mod = _import_module()
        with patch.object(mod, "_get_clipboard_text", return_value=""):
            ref = mod.SourceRef(kind="clipboard", value="current")
            with pytest.raises(mod.RefResolutionError, match="empty"):
                mod.resolve(ref)

    def test_resolve_selection(self):
        mod = _import_module()
        with patch.object(mod, "_get_selection_text", return_value="selected text"):
            ref = mod.SourceRef(kind="selection", value="frontmost")
            text = mod.resolve(ref)
            assert text == "selected text"

    def test_resolve_selection_empty(self):
        mod = _import_module()
        with patch.object(mod, "_get_selection_text", return_value=""):
            ref = mod.SourceRef(kind="selection", value="frontmost")
            with pytest.raises(mod.RefResolutionError, match="empty"):
                mod.resolve(ref)

    def test_resolve_last_response(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="last_response", value="current")
        text = mod.resolve(ref, last_response="The assistant said this.")
        assert text == "The assistant said this."

    def test_resolve_last_response_no_response(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="last_response", value="current")
        with pytest.raises(mod.RefResolutionError, match="No last response"):
            mod.resolve(ref)

    def test_resolve_empty_literal_raises(self):
        mod = _import_module()
        ref = mod.SourceRef(kind="literal", value="")
        with pytest.raises(mod.RefResolutionError, match="empty"):
            mod.resolve(ref)
