"""Tests for the shared optical-shell metrics surface."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from spoke.optical_shell_metrics import OpticalShellMetrics
from spoke.optical_shell_baseline import (
    OPTICAL_SHELL_BASELINE_SCENARIOS,
    REQUIRED_BUDGET_METRICS,
    OpticalShellBaselineSample,
    build_optical_shell_budget,
)


def test_metrics_snapshot_reports_counts_and_cadence():
    metrics = OpticalShellMetrics()

    metrics.record_capture_tick(now=1.0)
    metrics.record_capture_tick(now=1.1, elapsed_ms=2.5)
    metrics.record_capture_poll(0, now=2.0, elapsed_ms=1.0)
    metrics.record_capture_poll(3, now=2.2, elapsed_ms=2.0)
    metrics.record_display_tick(now=3.0, elapsed_ms=4.0)
    metrics.record_display_tick(now=3.05, elapsed_ms=5.0)
    metrics.record_presented_frame(now=4.0, elapsed_ms=6.0)
    metrics.record_presented_frame(now=4.2, elapsed_ms=7.0)
    metrics.record_brightness_sample(now=5.0, elapsed_ms=8.0)
    metrics.record_brightness_sample(now=5.5, elapsed_ms=9.0)

    snapshot = metrics.snapshot()

    assert snapshot["capture_ticks"] == 2
    assert snapshot["capture_polls"] == 2
    assert snapshot["duplicate_frames"] == 1
    assert snapshot["skipped_frames"] == 2
    assert snapshot["display_link_ticks"] == 2
    assert snapshot["presented_frames"] == 2
    assert snapshot["brightness_samples"] == 2
    assert snapshot["avg_capture_tick_ms"] == 1.25
    assert snapshot["avg_capture_poll_ms"] == 1.5
    assert snapshot["avg_display_tick_ms"] == 4.5
    assert snapshot["avg_presented_frame_ms"] == 6.5
    assert snapshot["avg_brightness_sample_ms"] == 8.5
    assert snapshot["avg_capture_tick_interval_ms"] == pytest.approx(100.0)
    assert snapshot["avg_capture_poll_interval_ms"] == pytest.approx(200.0)
    assert snapshot["avg_display_interval_ms"] == pytest.approx(50.0)
    assert snapshot["avg_presented_interval_ms"] == pytest.approx(200.0)
    assert snapshot["avg_brightness_interval_ms"] == pytest.approx(500.0)


def test_optical_shell_budget_contract_requires_packet_scenarios():
    assert OPTICAL_SHELL_BASELINE_SCENARIOS == (
        "static_background",
        "scrolling_background",
        "assistant_only_shell",
        "preview_only_shell",
        "combined_shells",
    )
    assert REQUIRED_BUDGET_METRICS == (
        "sample_duration_s",
        "spoke_cpu_percent",
        "windowserver_cpu_percent",
        "resident_memory_mb",
        "display_link_fps",
        "capture_fps",
        "presented_fps",
        "presentation_ratio",
        "duplicate_frame_ratio",
        "skipped_frame_ratio",
        "brightness_sample_hz",
        "avg_compositor_tick_ms",
        "avg_warp_to_drawable_ms",
        "avg_presented_frame_ms",
        "avg_brightness_sample_ms",
    )


def test_build_optical_shell_budget_derives_stable_per_scenario_metrics():
    sample = OpticalShellBaselineSample(
        scenario="combined_shells",
        duration_s=20.0,
        diagnostics={
            "capture_frames": 600,
            "display_link_ticks": 2400,
            "presented_frames": 2350,
            "duplicate_frames": 24,
            "skipped_frames": 6,
            "brightness_samples": 80,
            "avg_compositor_tick_ms": 0.52,
            "avg_warp_to_drawable_ms": 0.31,
            "avg_presented_frame_ms": 0.44,
            "avg_brightness_sample_ms": 1.7,
        },
        spoke_cpu_percent=18.5,
        windowserver_cpu_percent=9.25,
        resident_memory_mb=312.0,
    )

    budget = build_optical_shell_budget(
        [sample],
        machine="MacBook-Pro-2.local",
        display_hz=120.0,
        branch="cc/thermal-baseline-profiler",
        commit="abc1234",
        captured_at="2026-04-29T05:30:00-0400",
    )

    assert budget["schema"] == "spoke.optical_shell.baseline.v1"
    assert budget["machine"] == "MacBook-Pro-2.local"
    assert budget["branch"] == "cc/thermal-baseline-profiler"
    assert budget["commit"] == "abc1234"
    assert budget["required_scenarios"] == list(OPTICAL_SHELL_BASELINE_SCENARIOS)
    assert budget["required_metrics"] == list(REQUIRED_BUDGET_METRICS)
    combined = budget["scenarios"]["combined_shells"]
    assert combined["sample_duration_s"] == pytest.approx(20.0)
    assert combined["capture_fps"] == pytest.approx(30.0)
    assert combined["display_link_fps"] == pytest.approx(120.0)
    assert combined["presented_fps"] == pytest.approx(117.5)
    assert combined["presentation_ratio"] == pytest.approx(2350 / 2400)
    assert combined["duplicate_frame_ratio"] == pytest.approx(24 / 2400)
    assert combined["skipped_frame_ratio"] == pytest.approx(6 / 600)
    assert combined["brightness_sample_hz"] == pytest.approx(4.0)
    assert combined["spoke_cpu_percent"] == pytest.approx(18.5)
    assert combined["windowserver_cpu_percent"] == pytest.approx(9.25)
    assert combined["resident_memory_mb"] == pytest.approx(312.0)
    assert combined["avg_compositor_tick_ms"] == pytest.approx(0.52)
    assert combined["avg_warp_to_drawable_ms"] == pytest.approx(0.31)
    assert combined["avg_presented_frame_ms"] == pytest.approx(0.44)
    assert combined["avg_brightness_sample_ms"] == pytest.approx(1.7)
    assert budget["missing_scenarios"] == [
        "static_background",
        "scrolling_background",
        "assistant_only_shell",
        "preview_only_shell",
    ]


def test_build_optical_shell_budget_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown optical shell baseline scenario"):
        build_optical_shell_budget(
            [
                OpticalShellBaselineSample(
                    scenario="assistant_shellish",
                    duration_s=10.0,
                    diagnostics={},
                )
            ],
            machine="MacBook-Pro-2.local",
        )


def test_optical_shell_baseline_script_writes_budget_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "optical-shell-baseline.py"
    spec = importlib.util.spec_from_file_location("optical_shell_baseline_script", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    sample_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "budget.json"
    sample_path.write_text(
        json.dumps(
            {
                "scenario": "assistant_only_shell",
                "duration_s": 10.0,
                "diagnostics": {
                    "display_link_ticks": 1200,
                    "capture_frames": 300,
                    "presented_frames": 1190,
                    "duplicate_frames": 9,
                    "brightness_samples": 40,
                },
                "process": {
                    "spoke_cpu_percent": 14.0,
                    "windowserver_cpu_percent": 7.0,
                    "resident_memory_mb": 256.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--input",
            str(sample_path),
            "--output",
            str(output_path),
            "--machine",
            "MacBook-Pro-2.local",
            "--branch",
            "cc/thermal-baseline-profiler",
            "--commit",
            "abc1234",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assistant = payload["scenarios"]["assistant_only_shell"]
    assert assistant["display_link_fps"] == pytest.approx(120.0)
    assert assistant["capture_fps"] == pytest.approx(30.0)
    assert assistant["brightness_sample_hz"] == pytest.approx(4.0)
    assert assistant["spoke_cpu_percent"] == pytest.approx(14.0)
