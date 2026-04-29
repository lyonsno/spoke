"""Baseline budget contract for optical shell residency measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


OPTICAL_SHELL_BASELINE_SCENARIOS = (
    "static_background",
    "scrolling_background",
    "assistant_only_shell",
    "preview_only_shell",
    "combined_shells",
)

REQUIRED_BUDGET_METRICS = (
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


@dataclass(frozen=True)
class OpticalShellBaselineSample:
    """One measurement sample for one packet-owned baseline scenario."""

    scenario: str
    duration_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    spoke_cpu_percent: float | None = None
    windowserver_cpu_percent: float | None = None
    resident_memory_mb: float | None = None
    notes: str | None = None


def build_optical_shell_budget(
    samples: Iterable[OpticalShellBaselineSample],
    *,
    machine: str,
    display_hz: float = 120.0,
    branch: str | None = None,
    commit: str | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Aggregate samples into the sibling-consumable baseline budget shape."""

    grouped: dict[str, list[OpticalShellBaselineSample]] = {
        scenario: [] for scenario in OPTICAL_SHELL_BASELINE_SCENARIOS
    }
    for sample in samples:
        if sample.scenario not in grouped:
            raise ValueError(f"unknown optical shell baseline scenario: {sample.scenario}")
        if sample.duration_s <= 0:
            raise ValueError("optical shell baseline samples require positive duration_s")
        grouped[sample.scenario].append(sample)

    scenarios = {
        scenario: _aggregate_scenario(rows)
        for scenario, rows in grouped.items()
        if rows
    }
    missing = [scenario for scenario in OPTICAL_SHELL_BASELINE_SCENARIOS if scenario not in scenarios]

    return {
        "schema": "spoke.optical_shell.baseline.v1",
        "machine": machine,
        "display_hz": float(display_hz),
        "branch": branch,
        "commit": commit,
        "captured_at": captured_at,
        "required_scenarios": list(OPTICAL_SHELL_BASELINE_SCENARIOS),
        "required_metrics": list(REQUIRED_BUDGET_METRICS),
        "missing_scenarios": missing,
        "scenarios": scenarios,
    }


def sample_from_mapping(payload: dict[str, Any]) -> OpticalShellBaselineSample:
    """Parse a JSON-friendly sample mapping used by the baseline harness."""

    process = payload.get("process") if isinstance(payload.get("process"), dict) else {}
    return OpticalShellBaselineSample(
        scenario=str(payload["scenario"]),
        duration_s=float(payload["duration_s"]),
        diagnostics=dict(payload.get("diagnostics") or {}),
        spoke_cpu_percent=_optional_float(
            payload.get("spoke_cpu_percent", process.get("spoke_cpu_percent"))
        ),
        windowserver_cpu_percent=_optional_float(
            payload.get("windowserver_cpu_percent", process.get("windowserver_cpu_percent"))
        ),
        resident_memory_mb=_optional_float(
            payload.get("resident_memory_mb", process.get("resident_memory_mb"))
        ),
        notes=str(payload["notes"]) if payload.get("notes") is not None else None,
    )


def _aggregate_scenario(samples: list[OpticalShellBaselineSample]) -> dict[str, Any]:
    duration_s = sum(sample.duration_s for sample in samples)
    capture_frames = _sum_diagnostics(samples, "capture_frames")
    display_link_ticks = _sum_diagnostics(samples, "display_link_ticks")
    presented_frames = _sum_diagnostics(samples, "presented_frames")
    duplicate_frames = _sum_diagnostics(samples, "duplicate_frames")
    skipped_frames = _sum_diagnostics(samples, "skipped_frames")
    brightness_samples = _sum_diagnostics(samples, "brightness_samples")
    warp_calls = _sum_diagnostics(samples, "warp_to_drawable_calls")

    return {
        "sample_count": len(samples),
        "sample_duration_s": duration_s,
        "spoke_cpu_percent": _weighted_sample_average(samples, "spoke_cpu_percent"),
        "windowserver_cpu_percent": _weighted_sample_average(samples, "windowserver_cpu_percent"),
        "resident_memory_mb": _weighted_sample_average(samples, "resident_memory_mb"),
        "display_link_fps": _rate_or_weighted_average(
            samples,
            total_count=display_link_ticks,
            duration_s=duration_s,
            diagnostics_key="display_link_fps",
        ),
        "capture_fps": _rate_or_weighted_average(
            samples,
            total_count=capture_frames,
            duration_s=duration_s,
            diagnostics_key="capture_fps",
        ),
        "presented_fps": _rate_or_weighted_average(
            samples,
            total_count=presented_frames,
            duration_s=duration_s,
            diagnostics_key="presented_fps",
        ),
        "presentation_ratio": _ratio(presented_frames, display_link_ticks),
        "duplicate_frame_ratio": _ratio(duplicate_frames, display_link_ticks),
        "skipped_frame_ratio": _ratio(skipped_frames, capture_frames),
        "brightness_sample_hz": _ratio(brightness_samples, duration_s),
        "avg_compositor_tick_ms": _weighted_diagnostic_average(
            samples, "avg_compositor_tick_ms", "display_link_ticks"
        ),
        "avg_warp_to_drawable_ms": _weighted_diagnostic_average(
            samples,
            "avg_warp_to_drawable_ms",
            "warp_to_drawable_calls",
            fallback_weight_key="presented_frames",
            fallback_weight=warp_calls or presented_frames,
        ),
        "avg_presented_frame_ms": _weighted_diagnostic_average(
            samples, "avg_presented_frame_ms", "presented_frames"
        ),
        "avg_brightness_sample_ms": _weighted_diagnostic_average(
            samples, "avg_brightness_sample_ms", "brightness_samples"
        ),
    }


def _sum_diagnostics(samples: list[OpticalShellBaselineSample], key: str) -> float:
    total = 0.0
    for sample in samples:
        total += max(_optional_float(sample.diagnostics.get(key)) or 0.0, 0.0)
    return total


def _rate_or_weighted_average(
    samples: list[OpticalShellBaselineSample],
    *,
    total_count: float,
    duration_s: float,
    diagnostics_key: str,
) -> float | None:
    if total_count > 0 and duration_s > 0:
        return total_count / duration_s
    return _weighted_diagnostic_average(samples, diagnostics_key, None)


def _weighted_sample_average(samples: list[OpticalShellBaselineSample], attr: str) -> float | None:
    total = 0.0
    weight = 0.0
    for sample in samples:
        value = getattr(sample, attr)
        if value is None:
            continue
        total += float(value) * sample.duration_s
        weight += sample.duration_s
    return _ratio(total, weight)


def _weighted_diagnostic_average(
    samples: list[OpticalShellBaselineSample],
    value_key: str,
    weight_key: str | None,
    *,
    fallback_weight_key: str | None = None,
    fallback_weight: float | None = None,
) -> float | None:
    total = 0.0
    weight = 0.0
    for sample in samples:
        value = _optional_float(sample.diagnostics.get(value_key))
        if value is None:
            continue
        sample_weight = 0.0
        if weight_key is not None:
            sample_weight = _optional_float(sample.diagnostics.get(weight_key)) or 0.0
        if sample_weight <= 0 and fallback_weight_key is not None:
            sample_weight = _optional_float(sample.diagnostics.get(fallback_weight_key)) or 0.0
        if sample_weight <= 0:
            sample_weight = sample.duration_s
        total += value * sample_weight
        weight += sample_weight
    if weight <= 0 and fallback_weight is not None and fallback_weight <= 0:
        return None
    return _ratio(total, weight)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
