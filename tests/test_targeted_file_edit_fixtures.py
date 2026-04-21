"""Black-box fixture tests for the targeted file edit contract."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "targeted_file_edit"
UNIQUENESS_FIXTURE_ROOT = FIXTURE_ROOT / "uniqueness"
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


def _load_fixture(fixture_dir: Path) -> tuple[dict, dict, str, str]:
    request = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    source_text = (fixture_dir / "source.txt").read_text(encoding="utf-8")
    expected_text = (fixture_dir / "expected_file.txt").read_text(encoding="utf-8")
    return request, expected, source_text, expected_text


class TestTargetedFileEditFixtures:
    def test_uniqueness_fixture_corpus_exists(self):
        fixture_dirs = sorted(path for path in UNIQUENESS_FIXTURE_ROOT.iterdir() if path.is_dir())
        assert fixture_dirs, "Expected at least one uniqueness fixture directory."
        for fixture_dir in fixture_dirs:
            assert (fixture_dir / "README.md").exists()
            assert (fixture_dir / "request.json").exists()
            assert (fixture_dir / "expected.json").exists()
            assert (fixture_dir / "source.txt").exists()
            assert (fixture_dir / "expected_file.txt").exists()

    def test_uniqueness_fixtures_run_black_box(self, tmp_path):
        mod = _import_tools()
        tool_name = _discover_targeted_edit_tool_name(mod.get_tool_schemas())
        fixture_dirs = sorted(path for path in UNIQUENESS_FIXTURE_ROOT.iterdir() if path.is_dir())

        for fixture_dir in fixture_dirs:
            request, expected, source_text, expected_text = _load_fixture(fixture_dir)
            assert expected["outcome"] == "success"

            target_path = tmp_path / request["file"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(source_text, encoding="utf-8")

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

            if "failure_reason" in result:
                assert result["failure_reason"] is None
            if "error" in result and result["error"] is not None:
                pytest.fail(
                    f"Targeted edit fixture {fixture_dir.name} returned error: {result['error']}"
                )
            assert target_path.read_text(encoding="utf-8") == expected_text

