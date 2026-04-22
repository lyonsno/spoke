"""Black-box fixture tests for the targeted file edit contract."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "targeted_file_edit"
FIXTURE_SUITES = {
    "uniqueness": FIXTURE_ROOT / "uniqueness",
    "normalization": FIXTURE_ROOT / "normalization",
    "return_shape": FIXTURE_ROOT / "return_shape",
}
REQUIRED_PARAMETERS = {"file", "old_string", "new_string"}


def _import_tools():
    sys.modules.pop("spoke.tool_dispatch", None)
    return importlib.import_module("spoke.tool_dispatch")


def _discover_targeted_edit_tool_name(schemas: list[dict]) -> str:
    matches: list[str] = []
    for schema in schemas:
        function = schema.get("function", {})
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        if REQUIRED_PARAMETERS.issubset(properties):
            matches.append(function["name"])
    if not matches:
        pytest.fail(
            "No targeted edit tool schema exposes the file/old_string/new_string contract yet."
        )
    if len(matches) != 1:
        pytest.fail(
            "Expected exactly one targeted edit tool schema exposing the "
            f"file/old_string/new_string contract, found {matches!r}."
        )
    return matches[0]


def _fixture_dirs(fixture_root: Path) -> list[Path]:
    return sorted(path for path in fixture_root.iterdir() if path.is_dir())


def _all_fixture_cases() -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for suite_name, fixture_root in FIXTURE_SUITES.items():
        for fixture_dir in _fixture_dirs(fixture_root):
            cases.append((suite_name, fixture_dir))
    return cases


def _fixture_content_path(fixture_dir: Path, stem: str) -> Path:
    text_path = fixture_dir / f"{stem}.txt"
    hex_path = fixture_dir / f"{stem}.hex.json"
    candidates = [path for path in (text_path, hex_path) if path.exists()]
    assert len(candidates) == 1, (
        f"Expected exactly one {stem} payload in {fixture_dir}, "
        f"found {[path.name for path in candidates]!r}."
    )
    return candidates[0]


def _load_fixture_bytes(fixture_dir: Path, stem: str) -> bytes:
    payload_path = _fixture_content_path(fixture_dir, stem)
    if payload_path.suffix == ".txt":
        return payload_path.read_bytes()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return bytes.fromhex(payload["hex"])


def _load_fixture(fixture_dir: Path) -> tuple[dict, dict, bytes, bytes]:
    request = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    source_bytes = _load_fixture_bytes(fixture_dir, "source")
    expected_bytes = _load_fixture_bytes(fixture_dir, "expected_file")
    return request, expected, source_bytes, expected_bytes


def _resolve_expected_value(value, *, target_path: Path):
    if value == "__TARGET_FILE__":
        return str(target_path)
    if isinstance(value, dict):
        return {
            key: _resolve_expected_value(item, target_path=target_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_expected_value(item, target_path=target_path) for item in value]
    return value


def _assert_result_subset(actual, expected, *, target_path: Path):
    expected = _resolve_expected_value(expected, target_path=target_path)
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict result fragment, got {type(actual)!r}"
        for key, value in expected.items():
            assert key in actual, f"Missing expected result key {key!r}."
            _assert_result_subset(actual[key], value, target_path=target_path)
        return
    assert actual == expected


class TestTargetedFileEditFixtures:
    @pytest.mark.parametrize(("suite_name", "fixture_root"), FIXTURE_SUITES.items())
    def test_fixture_corpus_exists(self, suite_name, fixture_root):
        fixture_dirs = _fixture_dirs(fixture_root)
        assert fixture_dirs, f"Expected at least one {suite_name} fixture directory."
        for fixture_dir in fixture_dirs:
            assert (fixture_dir / "README.md").exists()
            assert (fixture_dir / "request.json").exists()
            assert (fixture_dir / "expected.json").exists()
            assert _fixture_content_path(fixture_dir, "source").exists()
            assert _fixture_content_path(fixture_dir, "expected_file").exists()

    @pytest.mark.parametrize(("suite_name", "fixture_dir"), _all_fixture_cases())
    def test_fixtures_run_black_box(self, suite_name, fixture_dir, tmp_path):
        mod = _import_tools()
        tool_name = _discover_targeted_edit_tool_name(mod.get_tool_schemas())
        request, expected, source_bytes, expected_bytes = _load_fixture(fixture_dir)
        assert expected["outcome"] in {"success", "failure"}

        target_path = tmp_path / request["file"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_bytes)

        result = json.loads(
            mod.execute_tool(
                tool_name,
                {
                    "file": str(target_path),
                    "old_string": request["old_string"],
                    "new_string": request["new_string"],
                },
            )
        )

        if "result" in expected:
            _assert_result_subset(result, expected["result"], target_path=target_path)

        if expected["outcome"] == "success":
            if "failure_reason" in result:
                assert result["failure_reason"] is None
            if "error" in result and result["error"] is not None:
                pytest.fail(
                    f"Targeted edit fixture {fixture_dir.name} returned error: {result['error']}"
                )
        elif expected["outcome"] == "failure":
            assert result.get("status") == "error"
        else:
            pytest.fail(
                f"Unknown fixture outcome {expected['outcome']!r} in {fixture_dir.name}."
            )
        assert target_path.read_bytes() == expected_bytes, (
            f"{suite_name} fixture {fixture_dir.name} wrote unexpected bytes."
        )
