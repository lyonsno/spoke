"""Tests for the shared optical-shell metrics surface."""

from __future__ import annotations

import pytest

from spoke.optical_shell_metrics import OpticalShellMetrics


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
