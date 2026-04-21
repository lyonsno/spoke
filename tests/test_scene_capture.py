"""Tests for scene capture module.

Tests the SceneCapture dataclasses, capture logic, OCR block extraction,
AX hint collection, downsampling, and artifact caching.

Mocks macOS framework calls (Quartz, Vision, AX) since tests run without
a GUI runtime.
"""

from __future__ import annotations

import importlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


def _import_module():
    """Import spoke.scene_capture, clearing cache first."""
    sys.modules.pop("spoke.scene_capture", None)
    return importlib.import_module("spoke.scene_capture")


# ── Dataclass construction ───────────────────────────────────────


class TestOCRBlock:
    def test_basic_construction(self):
        mod = _import_module()
        block = mod.OCRBlock(
            ref="scene-abc:block-1",
            text="Hello world",
            bbox=(10.0, 20.0, 100.0, 30.0),
            confidence=0.95,
        )
        assert block.ref == "scene-abc:block-1"
        assert block.text == "Hello world"
        assert block.bbox == (10.0, 20.0, 100.0, 30.0)
        assert block.confidence == 0.95

    def test_confidence_defaults_to_none(self):
        mod = _import_module()
        block = mod.OCRBlock(ref="r", text="t", bbox=(0, 0, 0, 0))
        assert block.confidence is None


class TestAXHint:
    def test_basic_construction(self):
        mod = _import_module()
        hint = mod.AXHint(
            ref="scene-abc:focus",
            role="AXWebArea",
            label="Main content",
            value="some text",
        )
        assert hint.role == "AXWebArea"
        assert hint.label == "Main content"

    def test_label_and_value_default_to_none(self):
        mod = _import_module()
        hint = mod.AXHint(ref="r", role="AXButton")
        assert hint.label is None
        assert hint.value is None


class TestSceneCapture:
    def test_basic_construction(self):
        mod = _import_module()
        cap = mod.SceneCapture(
            scene_ref="scene-abc",
            created_at=1000.0,
            scope="active_window",
            app_name="Safari",
            bundle_id="com.apple.Safari",
            window_title="GitHub",
            image_path="/tmp/scene-abc.png",
            image_size=(2560, 1440),
            model_image_size=(1280, 720),
            ocr_text="Hello world",
            ocr_blocks=[
                mod.OCRBlock(
                    ref="scene-abc:block-0",
                    text="Hello world",
                    bbox=(10.0, 20.0, 100.0, 30.0),
                    confidence=0.98,
                )
            ],
            ax_hints=[],
        )
        assert cap.scene_ref == "scene-abc"
        assert cap.scope == "active_window"
        assert len(cap.ocr_blocks) == 1


# ── Scene ref generation ─────────────────────────────────────────


class TestSceneRefGeneration:
    def test_refs_are_unique(self):
        mod = _import_module()
        refs = {mod._generate_scene_ref() for _ in range(100)}
        assert len(refs) == 100

    def test_ref_has_prefix(self):
        mod = _import_module()
        ref = mod._generate_scene_ref()
        assert ref.startswith("scene-")


# ── OCR extraction ───────────────────────────────────────────────


class TestExtractOCRBlocks:
    """Test _extract_ocr_blocks against mocked Vision results."""

    def test_extracts_blocks_from_observations(self):
        mod = _import_module()

        # Mock Vision observation objects
        obs1 = MagicMock()
        candidate1 = MagicMock()
        candidate1.string.return_value = "Hello"
        candidate1.confidence.return_value = 0.98
        obs1.topCandidates_.return_value = [candidate1]
        # boundingBox returns a normalized CGRect (origin + size)
        bbox1 = MagicMock()
        bbox1.origin.x = 0.1
        bbox1.origin.y = 0.2
        bbox1.size.width = 0.3
        bbox1.size.height = 0.05
        obs1.boundingBox.return_value = bbox1

        obs2 = MagicMock()
        candidate2 = MagicMock()
        candidate2.string.return_value = "World"
        candidate2.confidence.return_value = 0.91
        obs2.topCandidates_.return_value = [candidate2]
        bbox2 = MagicMock()
        bbox2.origin.x = 0.5
        bbox2.origin.y = 0.2
        bbox2.size.width = 0.3
        bbox2.size.height = 0.05
        obs2.boundingBox.return_value = bbox2

        blocks = mod._extract_ocr_blocks(
            observations=[obs1, obs2],
            scene_ref="scene-test",
            image_width=2560,
            image_height=1440,
        )

        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].ref == "scene-test:block-0"
        assert blocks[0].confidence == 0.98
        # bbox should be converted from normalized to pixel coordinates
        assert blocks[0].bbox[0] == pytest.approx(0.1 * 2560, abs=1)
        assert blocks[1].text == "World"
        assert blocks[1].ref == "scene-test:block-1"

    def test_empty_observations_returns_empty(self):
        mod = _import_module()
        blocks = mod._extract_ocr_blocks(
            observations=[], scene_ref="scene-x", image_width=100, image_height=100
        )
        assert blocks == []

    def test_skips_observation_without_candidates(self):
        mod = _import_module()
        obs = MagicMock()
        obs.topCandidates_.return_value = []
        blocks = mod._extract_ocr_blocks(
            observations=[obs], scene_ref="scene-x", image_width=100, image_height=100
        )
        assert blocks == []


# ── AX hints ─────────────────────────────────────────────────────


class TestCollectAXHints:
    """Test _collect_ax_hints with mocked AX API."""

    def test_returns_focus_hint_when_available(self):
        mod = _import_module()
        with patch.object(
            mod, "_get_focused_ax_info",
            return_value=("AXTextField", "Search", "hello"),
        ):
            hints = mod._collect_ax_hints("scene-test", timeout=1.0)
        assert len(hints) == 1
        assert hints[0].ref == "scene-test:focus"
        assert hints[0].role == "AXTextField"
        assert hints[0].label == "Search"
        assert hints[0].value == "hello"

    def test_returns_empty_on_no_focus(self):
        mod = _import_module()
        with patch.object(
            mod, "_get_focused_ax_info",
            return_value=None,
        ):
            hints = mod._collect_ax_hints("scene-test", timeout=1.0)
        assert hints == []

    def test_returns_empty_on_exception(self):
        mod = _import_module()
        with patch.object(
            mod, "_get_focused_ax_info",
            side_effect=OSError("AX unavailable"),
        ):
            hints = mod._collect_ax_hints("scene-test", timeout=1.0)
        assert hints == []

    def test_returns_empty_on_timeout(self):
        mod = _import_module()

        def slow_ax(*args, **kwargs):
            time.sleep(2.0)
            return ("AXTextField", "Search", "hello")

        with patch.object(mod, "_get_focused_ax_info", side_effect=slow_ax):
            hints = mod._collect_ax_hints("scene-test", timeout=0.1)
        assert hints == []


# ── Downsampling ─────────────────────────────────────────────────


class TestDownsampleSize:
    def test_defaults_to_one_third_resolution(self):
        mod = _import_module()
        result = mod._downsample_size(2560, 1440)
        assert result == (853, 480)

    def test_skips_if_already_small(self):
        """Small windows should not be downsampled below a usable size."""
        mod = _import_module()
        result = mod._downsample_size(400, 300)
        # Should return original size since it's already small
        assert result == (400, 300)

    def test_custom_scale(self):
        mod = _import_module()
        result = mod._downsample_size(2560, 1440, scale=0.25)
        assert result == (640, 360)


class TestPickTargetWindow:
    def test_prefers_ax_focused_pid_over_workspace_pid(self):
        mod = _import_module()
        windows = [
            {
                "kCGWindowOwnerPID": 101,
                "kCGWindowOwnerName": "WezTerm",
                "kCGWindowName": "terminal",
                "kCGWindowBounds": {"Width": 1200, "Height": 800},
            },
            {
                "kCGWindowOwnerPID": 202,
                "kCGWindowOwnerName": "Safari",
                "kCGWindowName": "GitHub - Avatar",
                "kCGWindowBounds": {"Width": 1400, "Height": 900},
            },
        ]

        picked = mod._pick_target_window(
            windows,
            preferred_pid=202,
            preferred_title="GitHub - Avatar",
            fallback_pid=101,
            my_pid=999,
        )

        assert picked is windows[1]

    def test_prefers_title_match_within_ax_pid_candidates(self):
        mod = _import_module()
        windows = [
            {
                "kCGWindowOwnerPID": 202,
                "kCGWindowOwnerName": "Safari",
                "kCGWindowName": "Other Tab",
                "kCGWindowBounds": {"Width": 1400, "Height": 900},
            },
            {
                "kCGWindowOwnerPID": 202,
                "kCGWindowOwnerName": "Safari",
                "kCGWindowName": "GitHub - Avatar",
                "kCGWindowBounds": {"Width": 1400, "Height": 900},
            },
        ]

        picked = mod._pick_target_window(
            windows,
            preferred_pid=202,
            preferred_title="GitHub - Avatar",
            fallback_pid=None,
            my_pid=999,
        )

        assert picked is windows[1]


class TestCaptureContext:
    def test_capture_context_saves_model_image_artifact(self, tmp_path):
        mod = _import_module()
        raw_image = object()
        model_image = object()

        with (
            patch.object(mod, "_generate_scene_ref", return_value="scene-test"),
            patch.object(
                mod,
                "_capture_active_window",
                return_value=(
                    raw_image,
                    "Safari",
                    "com.apple.Safari",
                    "Test Page",
                ),
            ),
            patch.object(mod, "_image_dimensions", return_value=(2560, 1440)),
            patch.object(mod, "_downsample_image", return_value=model_image) as downsample,
            patch.object(mod, "_save_image", return_value=True) as save_image,
            patch.object(mod, "_run_ocr", return_value=("Hello", [])),
            patch.object(mod, "_collect_ax_hints", return_value=[]),
        ):
            capture = mod.capture_context(cache_dir=str(tmp_path))

        assert capture is not None
        assert capture.image_path == str(tmp_path / "scene-test.png")
        assert capture.model_image_path == str(tmp_path / "scene-test-model.png")
        assert capture.model_image_size == (853, 480)
        downsample.assert_called_once_with(raw_image)
        assert save_image.call_args_list == [
            call(raw_image, str(tmp_path / "scene-test.png")),
            call(model_image, str(tmp_path / "scene-test-model.png")),
        ]

    def test_capture_context_skips_ocr_when_env_enabled(self, tmp_path, monkeypatch):
        mod = _import_module()
        raw_image = object()
        model_image = object()
        monkeypatch.setenv("SPOKE_SKIP_OCR", "1")

        with (
            patch.object(mod, "_generate_scene_ref", return_value="scene-test"),
            patch.object(
                mod,
                "_capture_active_window",
                return_value=(
                    raw_image,
                    "Safari",
                    "com.apple.Safari",
                    "Test Page",
                ),
            ),
            patch.object(mod, "_image_dimensions", return_value=(2560, 1440)),
            patch.object(mod, "_downsample_image", return_value=model_image),
            patch.object(mod, "_save_image", return_value=True),
            patch.object(mod, "_run_ocr") as run_ocr,
            patch.object(mod, "_collect_ax_hints", return_value=[]),
        ):
            capture = mod.capture_context(cache_dir=str(tmp_path))

        assert capture is not None
        assert capture.ocr_text == ""
        assert capture.ocr_blocks == []
        run_ocr.assert_not_called()


class TestCaptureActiveWindow:
    def test_ax_selected_window_refreshes_app_metadata(self):
        mod = _import_module()
        image = object()
        workspace_app = MagicMock()
        workspace_app.processIdentifier.return_value = 101
        workspace_app.localizedName.return_value = "WezTerm"
        workspace_app.bundleIdentifier.return_value = "com.github.wez.wezterm"

        target_app = MagicMock()
        target_app.localizedName.return_value = "Safari"
        target_app.bundleIdentifier.return_value = "com.apple.Safari"

        fake_workspace = MagicMock()
        fake_workspace.frontmostApplication.return_value = workspace_app

        fake_appkit = SimpleNamespace(
            NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: fake_workspace),
            NSRunningApplication=SimpleNamespace(
                runningApplicationWithProcessIdentifier_=lambda pid: target_app if pid == 202 else None
            ),
        )
        fake_quartz = SimpleNamespace(
            CGRectNull=None,
            CGWindowListCopyWindowInfo=lambda options, window_id: [
                {
                    "kCGWindowNumber": 41,
                    "kCGWindowOwnerPID": 202,
                    "kCGWindowOwnerName": "Safari",
                    "kCGWindowName": "GitHub - Avatar",
                    "kCGWindowLayer": 0,
                    "kCGWindowBounds": {"Width": 1400, "Height": 900},
                },
                {
                    "kCGWindowNumber": 42,
                    "kCGWindowOwnerPID": 101,
                    "kCGWindowOwnerName": "WezTerm",
                    "kCGWindowName": "terminal",
                    "kCGWindowLayer": 0,
                    "kCGWindowBounds": {"Width": 1200, "Height": 800},
                },
            ],
            CGWindowListCreateImage=lambda *args: image,
            kCGWindowImageBoundsIgnoreFraming=1,
            kCGWindowListExcludeDesktopElements=2,
            kCGWindowListOptionIncludingWindow=4,
            kCGWindowListOptionOnScreenOnly=8,
        )

        with (
            patch.dict(sys.modules, {"AppKit": fake_appkit, "Quartz": fake_quartz}),
            patch.object(mod, "_get_focused_window_hint", return_value=(202, "GitHub - Avatar")),
            patch.object(mod.os, "getpid", return_value=999),
        ):
            result = mod._capture_active_window()

        assert result == (
            image,
            "Safari",
            "com.apple.Safari",
            "GitHub - Avatar",
        )


# ── Artifact cache ───────────────────────────────────────────────


class TestSceneCaptureCache:
    def test_store_and_retrieve(self):
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=5)

        cap = mod.SceneCapture(
            scene_ref="scene-abc",
            created_at=time.time(),
            scope="active_window",
            app_name="Safari",
            bundle_id=None,
            window_title="Test",
            image_path="/tmp/test.png",
            image_size=(2560, 1440),
            model_image_size=(1280, 720),
            ocr_text="Hello",
            ocr_blocks=[],
            ax_hints=[],
        )
        cache.store(cap)

        assert cache.get("scene-abc") is cap
        assert cache.get("nonexistent") is None

    def test_evicts_oldest_when_full(self):
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=3)

        for i in range(5):
            cap = mod.SceneCapture(
                scene_ref=f"scene-{i}",
                created_at=float(i),
                scope="active_window",
                app_name=None,
                bundle_id=None,
                window_title=None,
                image_path=f"/tmp/{i}.png",
                image_size=(100, 100),
                model_image_size=(50, 50),
                ocr_text="",
                ocr_blocks=[],
                ax_hints=[],
            )
            cache.store(cap)

        # Oldest two should be evicted
        assert cache.get("scene-0") is None
        assert cache.get("scene-1") is None
        # Newest three should remain
        assert cache.get("scene-2") is not None
        assert cache.get("scene-3") is not None
        assert cache.get("scene-4") is not None

    def test_list_refs(self):
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=10)

        for i in range(3):
            cap = mod.SceneCapture(
                scene_ref=f"scene-{i}",
                created_at=float(i),
                scope="active_window",
                app_name=None,
                bundle_id=None,
                window_title=None,
                image_path=f"/tmp/{i}.png",
                image_size=(100, 100),
                model_image_size=(50, 50),
                ocr_text="",
                ocr_blocks=[],
                ax_hints=[],
            )
            cache.store(cap)

        refs = cache.list_refs()
        assert refs == ["scene-0", "scene-1", "scene-2"]

    def test_resolve_block_ref(self):
        """Cache can resolve a scene_block ref to its text."""
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=5)

        cap = mod.SceneCapture(
            scene_ref="scene-abc",
            created_at=time.time(),
            scope="active_window",
            app_name=None,
            bundle_id=None,
            window_title=None,
            image_path="/tmp/test.png",
            image_size=(100, 100),
            model_image_size=(50, 50),
            ocr_text="Hello World",
            ocr_blocks=[
                mod.OCRBlock(ref="scene-abc:block-0", text="Hello", bbox=(0, 0, 50, 20)),
                mod.OCRBlock(ref="scene-abc:block-1", text="World", bbox=(60, 0, 50, 20)),
            ],
            ax_hints=[],
        )
        cache.store(cap)

        assert cache.resolve_block("scene-abc:block-0") == "Hello"
        assert cache.resolve_block("scene-abc:block-1") == "World"
        assert cache.resolve_block("scene-abc:block-99") is None
        assert cache.resolve_block("scene-zzz:block-0") is None

    def test_resolve_ax_hint(self):
        """Cache can resolve an ax_hint ref to its label/value text."""
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=5)

        cap = mod.SceneCapture(
            scene_ref="scene-abc",
            created_at=time.time(),
            scope="active_window",
            app_name=None,
            bundle_id=None,
            window_title=None,
            image_path="/tmp/test.png",
            image_size=(100, 100),
            model_image_size=(50, 50),
            ocr_text="",
            ocr_blocks=[],
            ax_hints=[
                mod.AXHint(ref="scene-abc:focus", role="AXTextField", label="Search", value="query text"),
            ],
        )
        cache.store(cap)

        # Should return value when available, fall back to label
        assert cache.resolve_ax_hint("scene-abc:focus") == "query text"
        assert cache.resolve_ax_hint("scene-abc:nonexistent") is None

    def test_resolve_ax_hint_falls_back_to_label(self):
        mod = _import_module()
        cache = mod.SceneCaptureCache(max_captures=5)

        cap = mod.SceneCapture(
            scene_ref="scene-abc",
            created_at=time.time(),
            scope="active_window",
            app_name=None,
            bundle_id=None,
            window_title=None,
            image_path="/tmp/test.png",
            image_size=(100, 100),
            model_image_size=(50, 50),
            ocr_text="",
            ocr_blocks=[],
            ax_hints=[
                mod.AXHint(ref="scene-abc:focus", role="AXStaticText", label="Status: OK"),
            ],
        )
        cache.store(cap)

        assert cache.resolve_ax_hint("scene-abc:focus") == "Status: OK"
