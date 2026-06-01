from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spoke import visual_witness


def _instant(second: int) -> datetime:
    return datetime(2026, 6, 1, 0, 0, second, tzinfo=timezone.utc)


def test_visual_witness_plan_loads_smoke_env_and_routes_assistant_overlay(tmp_path):
    repo_root = tmp_path / "spoke"
    repo_root.mkdir()
    control_path = tmp_path / "control.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    repo_root.joinpath(".spoke-smoke-env").write_text(
        "\n".join(
            [
                f"SPOKE_WITNESS_CONTROL_PATH={control_path}",
                f"SPOKE_COMMAND_OVERLAY_TRACE_PATH={trace_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan = visual_witness.build_visual_witness_plan(
        "assistant-overlay-live",
        repo_root=repo_root,
        output_root=tmp_path / "out",
        capture_command="/usr/local/bin/global-witness-capture",
    )

    assert plan.recipe.name == "assistant-overlay-live"
    assert plan.control_path == control_path
    assert plan.trace_path == trace_path
    assert plan.show_action == "show_command_overlay"
    assert plan.hide_action == "hide_command_overlay"
    assert plan.command[0] == "/usr/local/bin/global-witness-capture"
    assert plan.command[plan.command.index("--source-window") + 1] == "Command Overlay"
    assert plan.output_dir.parent == tmp_path / "out"


def test_run_visual_witness_recipe_records_stimulus_and_capture_identity(tmp_path):
    repo_root = tmp_path / "spoke"
    repo_root.mkdir()
    control_path = tmp_path / "control.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    repo_root.joinpath(".spoke-smoke-env").write_text(
        "\n".join(
            [
                f"SPOKE_WITNESS_CONTROL_PATH={control_path}",
                f"SPOKE_COMMAND_OVERLAY_TRACE_PATH={trace_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-01T00:00:01Z",
                "event": "witness.control.received",
                "action": "show_command_overlay",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    class FakeCapture:
        def __init__(self, command, cwd=None):
            calls.append({"command": list(command), "cwd": cwd})
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath("manifest.json").write_text(
                json.dumps({"frames": [{"path": "frame-000.png"}]}),
                encoding="utf-8",
            )

        def wait(self):
            return 0

        def terminate(self):
            calls.append({"terminated": True})

    times = iter([_instant(0), _instant(2), _instant(3), _instant(4)])

    index = visual_witness.run_visual_witness_recipe(
        "assistant-overlay-live",
        repo_root=repo_root,
        output_root=tmp_path / "out",
        duration_seconds=1.0,
        fps=1.0,
        pre_stimulus_delay_seconds=0.0,
        settle_seconds=0.0,
        capture_command="/usr/local/bin/global-witness-capture",
        popen=FakeCapture,
        sleep=lambda _seconds: None,
        now=lambda: next(times),
    )

    payload = json.loads(index.read_text(encoding="utf-8"))
    control_lines = [json.loads(line) for line in control_path.read_text(encoding="utf-8").splitlines()]

    assert payload["schema"] == "spoke.visual_witness.v1"
    assert payload["recipe"] == "assistant-overlay-live"
    assert payload["retina_lasso_manifest_loaded"] is True
    assert payload["frame_count"] == 1
    assert payload["stimulus"]["show"]["action"] == "show_command_overlay"
    assert payload["stimulus"]["hide"]["action"] == "hide_command_overlay"
    assert payload["effective_route"]["control_path"] == str(control_path)
    assert payload["effective_route"]["capture_command"] == "/usr/local/bin/global-witness-capture"
    assert [line["action"] for line in control_lines] == [
        "show_command_overlay",
        "hide_command_overlay",
    ]
    assert calls[0]["cwd"] is None


def test_run_visual_witness_recipe_fails_loud_when_capture_writes_no_frames(tmp_path):
    repo_root = tmp_path / "spoke"
    repo_root.mkdir()
    control_path = tmp_path / "control.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    output_dir = tmp_path / "out" / "empty-capture"
    repo_root.joinpath(".spoke-smoke-env").write_text(
        "\n".join(
            [
                f"SPOKE_WITNESS_CONTROL_PATH={control_path}",
                f"SPOKE_COMMAND_OVERLAY_TRACE_PATH={trace_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEmptyCapture:
        def __init__(self, command, cwd=None):
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)

        def wait(self):
            return 0

        def terminate(self):
            pass

    times = iter([_instant(0), _instant(1), _instant(2), _instant(3)])

    with pytest.raises(visual_witness.VisualWitnessError, match="no captured frames"):
        visual_witness.run_visual_witness_recipe(
            "assistant-overlay-live",
            repo_root=repo_root,
            output_dir=output_dir,
            duration_seconds=1.0,
            fps=1.0,
            pre_stimulus_delay_seconds=0.0,
            settle_seconds=0.0,
            capture_command="/usr/local/bin/global-witness-capture",
            popen=FakeEmptyCapture,
            sleep=lambda _seconds: None,
            now=lambda: next(times),
        )

    payload = json.loads(output_dir.joinpath("witness-index.json").read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "capture_frames"
    assert payload["frame_count"] == 0
