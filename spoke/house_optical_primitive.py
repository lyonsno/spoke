"""Shared House optical primitive compiler.

This module owns the assistant-overlay optical shell contract so sibling
consumers can reuse the same primitive instead of copying its visual law.
"""

from __future__ import annotations

import math
import os

from .optical_lifecycle import OPTICAL_BODY_READY_PROGRESS, OPTICAL_MAG_SEED_PROGRESS


def _env(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in {"0", "false", "False", "no", "off"}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lerp(start: float, end: float, t: float) -> float:
    return float(start) + (float(end) - float(start)) * float(t)


def smoothstep(progress: float) -> float:
    t = clamp01(progress)
    return t * t * (3.0 - 2.0 * t)


def snap_ease_in(progress: float) -> float:
    t = clamp01(progress)
    return t * t * t


POINTS_PER_CM = 72.0 / 2.54
PRESSURE_SLIT_SMOKE_TIME_SCALE = 1.0 / 3.0
OPTICAL_SHELL_FEATHER = 140.0

OPTICAL_MATERIALIZATION_BASE_S = 1.36 * PRESSURE_SLIT_SMOKE_TIME_SCALE
OPTICAL_MATERIALIZATION_BASE_SPREAD_END = 0.77
OPTICAL_MATERIALIZATION_SEAM_OPEN_SPEEDUP = 2.0
OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE = 2.0
OPTICAL_MATERIALIZATION_SEAM_OPEN_S = (
    OPTICAL_MATERIALIZATION_BASE_S
    * OPTICAL_MATERIALIZATION_BASE_SPREAD_END
    / OPTICAL_MATERIALIZATION_SEAM_OPEN_SPEEDUP
)
OPTICAL_MATERIALIZATION_S = (
    OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        OPTICAL_MATERIALIZATION_BASE_S
        - OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
)
OPTICAL_MATERIALIZATION_DISMISS_S = OPTICAL_MATERIALIZATION_BASE_S
OPTICAL_MATERIALIZATION_PUCKER_TAIL_S = 1.5 * PRESSURE_SLIT_SMOKE_TIME_SCALE
OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS = 0.42
OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS = 0.12
OPTICAL_MATERIALIZATION_SEAM_LATCH_START = 0.0
OPTICAL_MATERIALIZATION_SEAM_LATCH_INTENSITY = 2.0
OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC = 0.8
OPTICAL_MATERIALIZATION_SEAM_LENGTH_CLOSED_FRAC = 0.0
OPTICAL_MATERIALIZATION_SEAM_THICKNESS_FRAC = 0.15
OPTICAL_MATERIALIZATION_SEAM_FOCUS_FRAC = 1.0
OPTICAL_MATERIALIZATION_SEAM_VERTICAL_GRIP = 1.0
OPTICAL_MATERIALIZATION_SEAM_HORIZONTAL_GRIP = 0.60
OPTICAL_MATERIALIZATION_SEAM_AXIS_ROTATION = 0.0
OPTICAL_MATERIALIZATION_SEAM_MIRRORED_LIP = 0.0
OPTICAL_MATERIALIZATION_SEAM_FIELD_HEIGHT_FRAC = 0.72
OPTICAL_MATERIALIZATION_SEAM_FIELD_MIN_HEIGHT_POINTS = 96.0
DISMISS_SEAM_CLIENT_ID = "assistant.command.dismiss_seam"
DISMISS_RADIAL_PUCKER_CLIENT_ID = "assistant.command.dismiss_radial_pucker"
OPTICAL_MATERIALIZATION_RADIAL_PUCKER_INTENSITY = 0.25
OPTICAL_MATERIALIZATION_RADIAL_AREA_MULTIPLIER = 1.0
OPTICAL_MATERIALIZATION_RADIAL_DIAMETER_HEIGHT_FRAC = 0.72
OPTICAL_MATERIALIZATION_RADIAL_MAX_HEIGHT_FRAC = 0.85
OPTICAL_MATERIALIZATION_RADIAL_MAX_WIDTH_FRAC = 0.20
OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN = 5.0
OPTICAL_MATERIALIZATION_PUCKER_GAIN_PEAK_AT = 0.30
OPTICAL_MATERIALIZATION_RADIAL_CYCLES = 2.35
OPTICAL_MATERIALIZATION_RADIAL_DAMPING = 4.4
OPTICAL_MATERIALIZATION_DISMISS_TOTAL_S = (
    OPTICAL_MATERIALIZATION_DISMISS_S + OPTICAL_MATERIALIZATION_PUCKER_TAIL_S
)
OPTICAL_MATERIALIZATION_BODY_READY = OPTICAL_BODY_READY_PROGRESS
OPTICAL_MATERIALIZATION_SEED_WIDTH_FRAC = 0.06
OPTICAL_MATERIALIZATION_SEED_HEIGHT_FRAC = 0.028
OPTICAL_MATERIALIZATION_SPREAD_END = (
    OPTICAL_MATERIALIZATION_SEAM_OPEN_S / OPTICAL_MATERIALIZATION_S
)
OPTICAL_MATERIALIZATION_BLOOM_START = OPTICAL_MATERIALIZATION_SPREAD_END
OPTICAL_MATERIALIZATION_MAG_SEED_FRAC = OPTICAL_MAG_SEED_PROGRESS
OPTICAL_MATERIALIZATION_MAG_ACCEL_END = 0.42
OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT = 0.72
OPTICAL_MATERIALIZATION_MAG_OVERSHOOT = 1.20
OPTICAL_MATERIAL_FILL_START = OPTICAL_MATERIALIZATION_SPREAD_END
OPTICAL_MATERIAL_FILL_SOLID_AT = (
    OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        OPTICAL_MATERIALIZATION_BASE_S * 0.84
        - OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
) / OPTICAL_MATERIALIZATION_S
OPTICAL_MATERIAL_FILL_FULL_AT = (
    OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    + (
        OPTICAL_MATERIALIZATION_BASE_S * 0.96
        - OPTICAL_MATERIALIZATION_SEAM_OPEN_S
    )
    * OPTICAL_MATERIALIZATION_POST_SPREAD_TIME_SCALE
) / OPTICAL_MATERIALIZATION_S
OPTICAL_COMPOSITOR_PUBLICATION_MIN_PROGRESS = (
    OPTICAL_MATERIALIZATION_SPREAD_END * 0.75
)
OPTICAL_APPKIT_ENTRANCE_MIN_PROGRESS = 0.999
OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC = 0.011
OPTICAL_TEXT_RELEASE_MIN_HEIGHT_FRAC = 1.0 / 3.0
OPTICAL_DISMISS_TEXT_BLOB_FRAC = 0.025
OPTICAL_DISMISS_TEXT_COLLAPSE_START_PROGRESS = min(
    1.0,
    OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS + 0.30,
)
OPTICAL_DISMISS_TEXT_BLOB_AT_PROGRESS = (
    OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS + 0.05
)
OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS = (
    OPTICAL_MATERIAL_FILL_SOLID_AT
    + (OPTICAL_MATERIAL_FILL_FULL_AT - OPTICAL_MATERIAL_FILL_SOLID_AT)
    * (
        (1.0 / 3.0 - OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC)
        / (1.0 - OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC)
    )
    ** (1.0 / 3.0)
)
OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS = (
    OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS
)
OPTICAL_MATERIALIZATION_SEAM_PEAK_PROGRESS = OPTICAL_MATERIAL_FILL_SOLID_AT


def cm_to_points(cm: float) -> float:
    return float(cm) * POINTS_PER_CM


def default_content_width_points() -> float:
    return _env("SPOKE_COMMAND_OVERLAY_WIDTH", 600.0)


def default_content_height_points() -> float:
    return _env("SPOKE_COMMAND_OVERLAY_HEIGHT", 80.0)


def material_fill_overscan_points() -> float:
    return _env("SPOKE_COMMAND_MATERIAL_FILL_OVERSCAN_MM", 1.5) / 10.0 * POINTS_PER_CM


def gpu_material_enabled() -> bool:
    return _env_bool("SPOKE_COMMAND_GPU_MATERIAL_ENABLED", False)


def optical_shell_core_magnification() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CORE_MAGNIFICATION", 1.55)


def optical_shell_band_mm() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_BAND_MM", 4.0)


def optical_shell_tail_mm() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_MM", 3.0)


def optical_shell_ring_amplitude_points() -> float:
    band_mm = optical_shell_band_mm()
    refraction = _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_REFRACTION", 2.6)
    return _env(
        "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_RING_AMPLITUDE_POINTS",
        (band_mm / 10.0) * POINTS_PER_CM * refraction,
    )


def optical_shell_tail_amplitude_points() -> float:
    tail_mm = optical_shell_tail_mm()
    refraction = _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_REFRACTION", 0.75)
    return _env(
        "SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_TAIL_AMPLITUDE_POINTS",
        (tail_mm / 10.0) * POINTS_PER_CM * refraction,
    )


def optical_shell_cleanup_blur_radius() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_CLEANUP_BLUR_RADIUS", 0.75)


def optical_shell_inflation_x_radii() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_X_RADII", 1.0)


def optical_shell_inflation_y_radii() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_INFLATION_Y_RADII", 1.0)


def optical_shell_debug_visualize() -> bool:
    return _env_bool("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_VISUALIZE", False)


def optical_shell_debug_grid_spacing_points() -> float:
    return _env("SPOKE_COMMAND_BACKDROP_OPTICAL_SHELL_DEBUG_GRID_SPACING_POINTS", 18.0)


def optical_shell_body_corner_radius(body_height_points: float) -> float:
    body_height = max(float(body_height_points), 1.0)
    return min(default_content_height_points() * 0.25, body_height * 0.5)


def gpu_material_shell_fields(
    brightness: float,
    scale: float,
    *,
    text_contrast_bias: float = 0.5,
    ridge_emphasis: float = 0.5,
) -> dict[str, float]:
    if not gpu_material_enabled():
        return {}
    return {
        "gpu_material_enabled": 1.0,
        "gpu_material_brightness": clamp01(brightness),
        "gpu_material_opacity": 1.0,
        "gpu_material_feather_points": OPTICAL_SHELL_FEATHER * max(float(scale), 1.0),
        "gpu_material_fill_overscan_points": (
            material_fill_overscan_points() * max(float(scale), 1.0)
        ),
        "gpu_material_text_contrast_bias": clamp01(text_contrast_bias),
        "gpu_material_ridge_emphasis": clamp01(ridge_emphasis),
    }


def with_gpu_material_basis(
    config: dict,
    *,
    width: float,
    height: float,
    corner_radius: float,
    progress: float = 1.0,
) -> None:
    if float(config.get("gpu_material_enabled", 0.0)) < 0.5:
        return
    state = house_materialization_fill_state(progress)
    config["gpu_material_base_width_points"] = max(float(width), 1.0)
    config["gpu_material_base_height_points"] = max(float(height), 1.0)
    config["gpu_material_base_corner_radius_points"] = max(float(corner_radius), 1.0)
    config["gpu_material_height_frac"] = state["height_frac"]
    config["gpu_material_opacity"] = min(
        float(config.get("gpu_material_opacity", 1.0)),
        state["opacity"] if progress < 1.0 else float(config.get("gpu_material_opacity", 1.0)),
    )


def compile_house_optical_shell_config(
    content_width_points: float | None = None,
    content_height_points: float | None = None,
) -> dict[str, float | bool]:
    width_points = (
        default_content_width_points()
        if content_width_points is None
        else max(float(content_width_points), 1.0)
    )
    height_points = (
        default_content_height_points()
        if content_height_points is None
        else max(float(content_height_points), 1.0)
    )
    shell_body_corner_r = optical_shell_body_corner_radius(height_points)
    width_points += optical_shell_inflation_x_radii() * shell_body_corner_r
    height_points += optical_shell_inflation_y_radii() * shell_body_corner_r
    return {
        "enabled": True,
        "content_width_points": width_points,
        "content_height_points": height_points,
        "corner_radius_points": shell_body_corner_r,
        "core_magnification": optical_shell_core_magnification(),
        "band_width_points": cm_to_points(optical_shell_band_mm() / 10.0),
        "tail_width_points": cm_to_points(optical_shell_tail_mm() / 10.0),
        "ring_amplitude_points": optical_shell_ring_amplitude_points(),
        "tail_amplitude_points": optical_shell_tail_amplitude_points(),
        "debug_visualize": optical_shell_debug_visualize(),
        "debug_grid_spacing_points": optical_shell_debug_grid_spacing_points(),
        "cleanup_blur_radius_points": optical_shell_cleanup_blur_radius(),
    }


def materialized_house_optical_shell_config(shell_config: dict, progress: float) -> dict:
    config = dict(shell_config)
    p = clamp01(progress)
    if p >= 1.0:
        return config

    base_w = max(float(config.get("content_width_points", 1.0)), 1.0)
    base_h = max(float(config.get("content_height_points", 1.0)), 1.0)
    base_radius = max(float(config.get("corner_radius_points", 1.0)), 1.0)
    config["_materialization_base_width_points"] = base_w
    config["_materialization_base_height_points"] = base_h
    config["_materialization_base_corner_radius_points"] = base_radius
    with_gpu_material_basis(
        config,
        width=base_w,
        height=base_h,
        corner_radius=base_radius,
        progress=p,
    )

    spread_t = snap_ease_in(p / OPTICAL_MATERIALIZATION_SPREAD_END)
    bloom_t = snap_ease_in(
        (p - OPTICAL_MATERIALIZATION_BLOOM_START)
        / max(1.0 - OPTICAL_MATERIALIZATION_BLOOM_START, 1e-6)
    )
    seed_w = max(24.0, min(base_w * OPTICAL_MATERIALIZATION_SEED_WIDTH_FRAC, 72.0))
    seed_h = max(2.5, min(base_h * OPTICAL_MATERIALIZATION_SEED_HEIGHT_FRAC, 7.0))
    width = lerp(seed_w, base_w, spread_t)
    height = lerp(seed_h, base_h, bloom_t)

    config["content_width_points"] = width
    config["content_height_points"] = height
    config["corner_radius_points"] = min(base_radius, height * 0.5)
    if "core_magnification" in config:
        base_mag = max(float(config.get("core_magnification", 1.0)), 0.0)
        seed_mag = base_mag * OPTICAL_MATERIALIZATION_MAG_SEED_FRAC
        if p <= OPTICAL_MATERIALIZATION_MAG_ACCEL_END:
            t = clamp01(p / OPTICAL_MATERIALIZATION_MAG_ACCEL_END)
            config["core_magnification"] = lerp(
                seed_mag,
                base_mag * 0.82,
                snap_ease_in(t),
            )
        elif p <= OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT:
            t = clamp01(
                (p - OPTICAL_MATERIALIZATION_MAG_ACCEL_END)
                / (
                    OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT
                    - OPTICAL_MATERIALIZATION_MAG_ACCEL_END
                )
            )
            config["core_magnification"] = lerp(
                base_mag * 0.82,
                base_mag * OPTICAL_MATERIALIZATION_MAG_OVERSHOOT,
                snap_ease_in(t),
            )
        else:
            t = clamp01(
                (p - OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT)
                / max(1.0 - OPTICAL_MATERIALIZATION_MAG_OVERSHOOT_AT, 1e-6)
            )
            config["core_magnification"] = lerp(
                base_mag * OPTICAL_MATERIALIZATION_MAG_OVERSHOOT,
                base_mag,
                snap_ease_in(t),
            )
    for key in ("band_width_points", "tail_width_points"):
        if key in config:
            config[key] = max(1.0, float(config[key]) * lerp(0.25, 1.0, p))
    for key in ("ring_amplitude_points", "tail_amplitude_points"):
        if key in config:
            config[key] = float(config[key]) * lerp(0.35, 1.0, p)
    config["continuous_present"] = True
    return config


def house_materialization_fill_state(progress: float) -> dict[str, float]:
    p = clamp01(progress)
    if p <= OPTICAL_MATERIAL_FILL_START:
        opacity = 0.0
    else:
        opacity = smoothstep(
            (p - OPTICAL_MATERIAL_FILL_START)
            / max(OPTICAL_MATERIAL_FILL_SOLID_AT - OPTICAL_MATERIAL_FILL_START, 1e-6)
        )
    height = lerp(
        OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC,
        1.0,
        clamp01(
            (p - OPTICAL_MATERIAL_FILL_SOLID_AT)
            / max(OPTICAL_MATERIAL_FILL_FULL_AT - OPTICAL_MATERIAL_FILL_SOLID_AT, 1e-6)
        )
        ** 3.0,
    )
    warp_bloom = snap_ease_in(
        (p - OPTICAL_MATERIALIZATION_BLOOM_START)
        / max(1.0 - OPTICAL_MATERIALIZATION_BLOOM_START, 1e-6)
    )
    height = min(height, max(OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC, warp_bloom))
    return {
        "opacity": clamp01(opacity),
        "height_frac": clamp01(height),
    }


def material_fill_progress_for_height(height_frac: float) -> float:
    hf = clamp01(height_frac)
    if hf <= OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC:
        return OPTICAL_MATERIAL_FILL_SOLID_AT
    fill_span = max(
        OPTICAL_MATERIAL_FILL_FULL_AT - OPTICAL_MATERIAL_FILL_SOLID_AT,
        1e-6,
    )
    normalized = (
        (hf - OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC)
        / max(1.0 - OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC, 1e-6)
    )
    return OPTICAL_MATERIAL_FILL_SOLID_AT + fill_span * (normalized ** (1.0 / 3.0))


def optical_text_release_progress() -> float:
    return max(
        OPTICAL_MATERIALIZATION_BODY_READY,
        material_fill_progress_for_height(OPTICAL_TEXT_RELEASE_MIN_HEIGHT_FRAC),
    )


def optical_entrance_text_ready(
    progress: float | None,
    height_frac: float | None = None,
) -> bool:
    if progress is None:
        return (
            height_frac is not None
            and clamp01(height_frac) >= OPTICAL_TEXT_RELEASE_MIN_HEIGHT_FRAC
        )
    p = clamp01(progress)
    hf = (
        house_materialization_fill_state(p)["height_frac"]
        if height_frac is None
        else clamp01(height_frac)
    )
    return (
        p >= OPTICAL_MATERIALIZATION_BODY_READY
        and hf >= OPTICAL_TEXT_RELEASE_MIN_HEIGHT_FRAC
    )


def house_dismiss_materialization_fill_state(progress: float) -> dict[str, float]:
    p = clamp01(progress)
    if p <= OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS:
        return {
            "opacity": 0.0,
            "height_frac": OPTICAL_MATERIAL_FILL_MIN_HEIGHT_FRAC,
        }
    state = house_materialization_fill_state(p)
    return {
        "opacity": state["opacity"],
        "height_frac": state["height_frac"],
    }


def house_dismiss_text_collapse_state(progress: float) -> dict[str, float]:
    p = clamp01(progress)
    gone_at = OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS
    blob_at = OPTICAL_DISMISS_TEXT_BLOB_AT_PROGRESS
    collapse_start = OPTICAL_DISMISS_TEXT_COLLAPSE_START_PROGRESS
    blob = OPTICAL_DISMISS_TEXT_BLOB_FRAC
    if p <= gone_at:
        return {
            "width_frac": blob,
            "height_frac": blob,
            "alpha": 0.0,
        }
    if p <= blob_at:
        return {
            "width_frac": blob,
            "height_frac": blob,
            "alpha": 0.0,
        }
    t = smoothstep((p - blob_at) / max(collapse_start - blob_at, 1e-6))
    frac = lerp(blob, 1.0, t)
    return {
        "width_frac": clamp01(frac),
        "height_frac": clamp01(frac),
        "alpha": clamp01(t * t),
    }


def dismiss_text_collapse_progress_for_body_height(
    progress: float,
    body_height_frac: float,
) -> float:
    _ = body_height_frac
    return clamp01(progress)


def dismiss_pucker_amount(progress: float) -> float:
    p = clamp01(progress)
    return math.exp(-OPTICAL_MATERIALIZATION_RADIAL_DAMPING * p) * math.cos(
        2.0 * math.pi * OPTICAL_MATERIALIZATION_RADIAL_CYCLES * p
    )


def dismiss_pucker_tail_progress_for_close_progress(close_progress: float) -> float:
    start = max(OPTICAL_MATERIALIZATION_PUCKER_PREARM_START_PROGRESS, 1e-6)
    phase = clamp01((start - clamp01(close_progress)) / start)
    return lerp(
        0.0,
        OPTICAL_MATERIALIZATION_PUCKER_PREARM_TAIL_PROGRESS,
        phase,
    )


def dismiss_seam_latch_amount(progress: float) -> float:
    p = clamp01(progress)
    start = max(OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS, 1e-6)
    t = clamp01((start - p) / start)
    return lerp(
        OPTICAL_MATERIALIZATION_SEAM_LATCH_START,
        1.0,
        1.0 - (1.0 - t) ** 3.0,
    )


def seam_pucker_tuning_defaults() -> dict[str, float]:
    return {
        "preview_progress": OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS * 0.45,
        "seam_latch_start": OPTICAL_MATERIALIZATION_SEAM_LATCH_START,
        "seam_latch_intensity": OPTICAL_MATERIALIZATION_SEAM_LATCH_INTENSITY,
        "scar_seam_length_frac": OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC,
        "scar_seam_thickness_frac": OPTICAL_MATERIALIZATION_SEAM_THICKNESS_FRAC,
        "scar_seam_focus_frac": OPTICAL_MATERIALIZATION_SEAM_FOCUS_FRAC,
        "scar_vertical_grip": OPTICAL_MATERIALIZATION_SEAM_VERTICAL_GRIP,
        "scar_horizontal_grip": OPTICAL_MATERIALIZATION_SEAM_HORIZONTAL_GRIP,
        "scar_axis_rotation": OPTICAL_MATERIALIZATION_SEAM_AXIS_ROTATION,
        "scar_mirrored_lip": OPTICAL_MATERIALIZATION_SEAM_MIRRORED_LIP,
    }


def dismiss_pucker_amplitude_multiplier(progress: float) -> float:
    p = clamp01(progress)
    peak_at = max(OPTICAL_MATERIALIZATION_PUCKER_GAIN_PEAK_AT, 1e-6)
    if p <= peak_at:
        t = p / peak_at
        return lerp(
            1.0,
            OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN,
            1.0 - (1.0 - t) ** 3.0,
        )
    t = (p - peak_at) / max(1.0 - peak_at, 1e-6)
    return OPTICAL_MATERIALIZATION_PUCKER_DIAGNOSTIC_GAIN * math.exp(-5.0 * t)


def apply_dismiss_seam_latch_fields(
    config: dict,
    progress: float,
    tuning: dict[str, float] | None = None,
) -> dict:
    updated = dict(config)
    settings = seam_pucker_tuning_defaults()
    if tuning:
        for key, value in tuning.items():
            if key in settings:
                settings[key] = float(value)
    p = clamp01(progress)
    overlap_start = max(OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS, 1e-6)
    t = clamp01((overlap_start - p) / overlap_start)
    base_h = max(
        float(
            updated.get(
                "_materialization_base_height_points",
                updated.get("content_height_points", 1.0),
            )
        ),
        1.0,
    )
    current_h = max(float(updated.get("content_height_points", 1.0)), 1.0)
    seam_field_h = max(
        current_h,
        min(
            base_h,
            max(
                OPTICAL_MATERIALIZATION_SEAM_FIELD_MIN_HEIGHT_POINTS,
                base_h * OPTICAL_MATERIALIZATION_SEAM_FIELD_HEIGHT_FRAC,
            ),
        ),
    )
    amount = lerp(
        settings["seam_latch_start"],
        1.0,
        1.0 - (1.0 - t) ** 3.0,
    ) * settings["seam_latch_intensity"]
    updated["content_height_points"] = seam_field_h
    updated["corner_radius_points"] = min(
        max(float(updated.get("corner_radius_points", 1.0)), 1.0),
        seam_field_h * 0.5,
    )
    updated["cleanup_blur_radius_points"] = 0.0
    updated["mip_blur_strength"] = 0.0
    updated["warp_mode"] = 3.0 if settings["scar_mirrored_lip"] >= 0.5 else 1.0
    updated["scar_amount"] = amount
    updated["scar_seam_length_frac"] = settings["scar_seam_length_frac"]
    updated["scar_seam_thickness_frac"] = settings["scar_seam_thickness_frac"]
    updated["scar_seam_focus_frac"] = settings["scar_seam_focus_frac"]
    updated["scar_vertical_grip"] = settings["scar_vertical_grip"]
    updated["scar_horizontal_grip"] = settings["scar_horizontal_grip"]
    updated["scar_axis_rotation"] = settings["scar_axis_rotation"]
    updated["scar_mirrored_lip"] = settings["scar_mirrored_lip"]
    updated["x_squeeze"] = 1.0
    updated["y_squeeze"] = 1.0
    updated["ring_amplitude_points"] = 0.0
    updated["tail_amplitude_points"] = 0.0
    updated["continuous_present"] = True
    return updated


def dismiss_seam_tuning_for_close_progress(
    close_progress: float,
    tuning: dict[str, float] | None = None,
) -> dict[str, float]:
    settings = seam_pucker_tuning_defaults()
    if tuning:
        for key, value in tuning.items():
            if key in settings:
                settings[key] = float(value)
    p = clamp01(close_progress)
    arm_start = max(OPTICAL_MATERIALIZATION_SEAM_OVERLAP_START_PROGRESS, 1e-6)
    peak = max(OPTICAL_MATERIALIZATION_SEAM_PEAK_PROGRESS, 1e-6)
    if p >= peak:
        arm_phase = smoothstep((arm_start - p) / max(arm_start - peak, 1e-6))
        settings["preview_progress"] = 0.0
        settings["seam_latch_intensity"] *= arm_phase
        settings["scar_seam_length_frac"] = OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC
        return settings

    phase = clamp01((peak - p) / peak)
    settings["preview_progress"] = lerp(
        0.0,
        OPTICAL_MATERIALIZATION_PUCKER_OVERLAP_START_PROGRESS,
        phase,
    )
    settings["scar_seam_length_frac"] = lerp(
        OPTICAL_MATERIALIZATION_SEAM_LENGTH_FRAC,
        OPTICAL_MATERIALIZATION_SEAM_LENGTH_CLOSED_FRAC,
        phase,
    )
    return settings


def dismiss_seam_latch_house_shell_config(
    final_shell_config: dict,
    progress: float,
    tuning: dict[str, float] | None = None,
) -> dict:
    config = dict(final_shell_config)
    config["client_id"] = DISMISS_SEAM_CLIENT_ID
    config["role"] = "assistant"
    config["visible"] = True
    config["z_index"] = 10
    seam_tuning = dismiss_seam_tuning_for_close_progress(progress, tuning)
    return apply_dismiss_seam_latch_fields(
        config,
        seam_tuning["preview_progress"],
        seam_tuning,
    )


def apply_dismiss_radial_pucker_fields(config: dict, progress: float) -> dict:
    updated = dict(config)
    amount = (
        dismiss_pucker_amount(progress)
        * OPTICAL_MATERIALIZATION_RADIAL_PUCKER_INTENSITY
        * dismiss_pucker_amplitude_multiplier(progress)
    )
    updated["cleanup_blur_radius_points"] = 0.0
    updated["mip_blur_strength"] = 0.0
    updated["warp_mode"] = 2.0
    updated["scar_amount"] = amount
    updated["x_squeeze"] = 1.0
    updated["y_squeeze"] = 1.0
    updated["ring_amplitude_points"] = 0.0
    updated["tail_amplitude_points"] = 0.0
    updated["continuous_present"] = True
    return updated


def dismiss_pucker_house_shell_config(shell_config: dict, progress: float) -> dict:
    base_w = max(float(shell_config.get("content_width_points", 1.0)), 1.0)
    base_h = max(float(shell_config.get("content_height_points", 1.0)), 1.0)
    config = materialized_house_optical_shell_config(shell_config, 0.0)
    base_diameter = min(
        base_h * OPTICAL_MATERIALIZATION_RADIAL_DIAMETER_HEIGHT_FRAC,
        base_h * OPTICAL_MATERIALIZATION_RADIAL_MAX_HEIGHT_FRAC,
        base_w * OPTICAL_MATERIALIZATION_RADIAL_MAX_WIDTH_FRAC,
    )
    diameter = max(
        1.0,
        base_diameter * math.sqrt(OPTICAL_MATERIALIZATION_RADIAL_AREA_MULTIPLIER),
    )
    config["content_width_points"] = diameter
    config["content_height_points"] = diameter
    config["corner_radius_points"] = diameter * 0.5
    config["core_magnification"] = 1.0
    return apply_dismiss_radial_pucker_fields(config, progress)


def dismiss_radial_pucker_house_shell_config(shell_config: dict, progress: float) -> dict:
    config = dismiss_pucker_house_shell_config(shell_config, progress)
    config["client_id"] = DISMISS_RADIAL_PUCKER_CLIENT_ID
    config["role"] = "assistant"
    config["visible"] = True
    config["z_index"] = 9
    return config


def hidden_dismiss_main_house_shell_config(shell_config: dict) -> dict:
    config = materialized_house_optical_shell_config(shell_config, 0.0)
    config["visible"] = False
    config["continuous_present"] = True
    config["mip_blur_strength"] = 0.0
    config["cleanup_blur_radius_points"] = 0.0
    return config
