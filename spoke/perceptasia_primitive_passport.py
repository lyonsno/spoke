"""Perceptasia consumer passport for the public optical primitive.

The passport is intentionally pure. It names the optical field Perceptasia may
request and the carrier constraints a live viewer must satisfy, without
importing AppKit/WebKit or mutating assistant-overlay lifecycle state.
"""

from __future__ import annotations

from typing import Mapping

from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldMotionIntent,
    OpticalFieldPresentation,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldState,
    OpticalFieldSlotOverride,
    compile_placeholder_shell_config,
)


PERCEPTASIA_PRIMITIVE_CLIENT_ID = "perceptasia.throughglass"
PERCEPTASIA_PRIMITIVE_LAYOUT_RECIPE = "perceptasia-primitive-passport"
PERCEPTASIA_PRIMITIVE_PROVIDER_ENV = "SPOKE_PERCEPTASIA_PRIMITIVE_URL"
PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_ENV = (
    "SPOKE_PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_REQUIRED"
)
PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL_ENV = "SPOKE_PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL"
_PRIMITIVE_MATERIALIZATION_SPREAD_END = 0.24
_PRIMITIVE_MATERIALIZATION_BLOOM_START = _PRIMITIVE_MATERIALIZATION_SPREAD_END
_PRIMITIVE_MATERIALIZATION_SEED_WIDTH_FRAC = 0.052
_PRIMITIVE_MATERIALIZATION_SEED_HEIGHT_FRAC = 0.028
_PRIMITIVE_MATERIAL_FILL_START = _PRIMITIVE_MATERIALIZATION_SPREAD_END
_PRIMITIVE_MATERIAL_FILL_SOLID_AT = 0.80
_PRIMITIVE_MATERIAL_FILL_FULL_AT = 0.95
_PRIMITIVE_MATERIAL_FILL_MIN_HEIGHT_FRAC = 0.011


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * _clamp01(t)


def _smoothstep(value: float) -> float:
    t = _clamp01(value)
    return t * t * (3.0 - 2.0 * t)


def _snap_ease_in(value: float) -> float:
    t = _clamp01(value)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _materialization_fill_state(progress: float) -> dict[str, float]:
    p = _clamp01(progress)
    if p <= _PRIMITIVE_MATERIAL_FILL_START:
        opacity = 0.0
    else:
        opacity = _smoothstep(
            (p - _PRIMITIVE_MATERIAL_FILL_START)
            / max(
                _PRIMITIVE_MATERIAL_FILL_SOLID_AT - _PRIMITIVE_MATERIAL_FILL_START,
                1e-6,
            )
        )
    height = _lerp(
        _PRIMITIVE_MATERIAL_FILL_MIN_HEIGHT_FRAC,
        1.0,
        _clamp01(
            (p - _PRIMITIVE_MATERIAL_FILL_SOLID_AT)
            / max(
                _PRIMITIVE_MATERIAL_FILL_FULL_AT - _PRIMITIVE_MATERIAL_FILL_SOLID_AT,
                1e-6,
            )
        )
        ** 3.0,
    )
    warp_bloom = _snap_ease_in(
        (p - _PRIMITIVE_MATERIALIZATION_BLOOM_START)
        / max(1.0 - _PRIMITIVE_MATERIALIZATION_BLOOM_START, 1e-6)
    )
    return {
        "opacity": _clamp01(opacity),
        "height_frac": _clamp01(
            min(height, max(_PRIMITIVE_MATERIAL_FILL_MIN_HEIGHT_FRAC, warp_bloom))
        ),
    }


def _apply_materialization_progress(config: dict[str, object], progress: float) -> None:
    p = _clamp01(progress)
    config["continuous_present"] = True
    if p >= 1.0:
        return

    base_w = max(float(config.get("content_width_points", 1.0)), 1.0)
    base_h = max(float(config.get("content_height_points", 1.0)), 1.0)
    base_radius = max(float(config.get("corner_radius_points", 1.0)), 1.0)
    base_band = max(float(config.get("band_width_points", 0.0)), 0.0)
    base_tail = max(float(config.get("tail_width_points", 0.0)), 0.0)
    base_ring = max(float(config.get("ring_amplitude_points", 0.0)), 0.0)
    base_tail_amp = max(float(config.get("tail_amplitude_points", 0.0)), 0.0)

    spread_t = _snap_ease_in(p / _PRIMITIVE_MATERIALIZATION_SPREAD_END)
    bloom_t = _snap_ease_in(
        (p - _PRIMITIVE_MATERIALIZATION_BLOOM_START)
        / max(1.0 - _PRIMITIVE_MATERIALIZATION_BLOOM_START, 1e-6)
    )
    seed_w = max(24.0, min(base_w * _PRIMITIVE_MATERIALIZATION_SEED_WIDTH_FRAC, 72.0))
    seed_h = max(2.5, min(base_h * _PRIMITIVE_MATERIALIZATION_SEED_HEIGHT_FRAC, 7.0))
    width = _lerp(seed_w, base_w, spread_t)
    height = _lerp(seed_h, base_h, bloom_t)
    edge_t = max(0.18, _smoothstep(max(spread_t, bloom_t)))
    fill_state = _materialization_fill_state(p)

    config["_materialization_base_width_points"] = base_w
    config["_materialization_base_height_points"] = base_h
    config["_materialization_base_corner_radius_points"] = base_radius
    config["content_width_points"] = width
    config["content_height_points"] = height
    config["corner_radius_points"] = min(base_radius, height * 0.5)
    config["cut_radius_points"] = config["corner_radius_points"]
    config["band_width_points"] = _lerp(max(1.5, base_band * 0.18), base_band, edge_t)
    config["tail_width_points"] = _lerp(max(1.0, base_tail * 0.16), base_tail, edge_t)
    config["ring_amplitude_points"] = _lerp(max(0.7, base_ring * 0.16), base_ring, edge_t)
    config["tail_amplitude_points"] = _lerp(max(0.3, base_tail_amp * 0.14), base_tail_amp, edge_t)
    config["gpu_material_base_width_points"] = base_w
    config["gpu_material_base_height_points"] = base_h
    config["gpu_material_base_corner_radius_points"] = base_radius
    config["gpu_material_height_frac"] = fill_state["height_frac"]
    config["gpu_material_opacity"] = min(
        float(config.get("gpu_material_opacity", 1.0)),
        fill_state["opacity"],
    )
    if isinstance(config.get("optical_field"), dict):
        optical_field = dict(config["optical_field"])
        optical_field["cut_radius_points"] = config["cut_radius_points"]
        config["optical_field"] = optical_field


def build_perceptasia_primitive_request(
    bounds: OpticalFieldBounds,
    *,
    state: OpticalFieldState = "rest",
    visible: bool = True,
    continuity_key: str = PERCEPTASIA_PRIMITIVE_CLIENT_ID,
) -> OpticalFieldRequest:
    """Build Perceptasia's sibling optical request without animation custody."""

    return OpticalFieldRequest(
        caller_id=PERCEPTASIA_PRIMITIVE_CLIENT_ID,
        continuity_key=continuity_key,
        bounds=bounds,
        role="hud",
        state=state,
        visible=visible,
        presentation=OpticalFieldPresentation(layer="hud", order=42),
        presentation_layer="hud",
        layout_recipe=PERCEPTASIA_PRIMITIVE_LAYOUT_RECIPE,
        motion=OpticalFieldMotionIntent(
            strategy="continuous",
            continuity="preserve_identity",
            urgency="normal",
            latency_mask="content_proof",
        ),
        profile=OpticalFieldProfileRef(
            base="captured_scene",
            params={
                "mip_blur_strength": 0.0,
                "band_width_frac": 0.031,
                "tail_width_frac": 0.021,
                "ring_amplitude_frac": 0.024,
                "tail_amplitude_frac": 0.008,
                "exterior_mix_frac": 0.069,
            },
            slots={
                "materialize": OpticalFieldSlotOverride(
                    params={
                        "mip_blur_strength": 0.0,
                        "band_width_frac": 0.038,
                        "tail_width_frac": 0.026,
                        "ring_amplitude_frac": 0.032,
                        "tail_amplitude_frac": 0.012,
                        "exterior_mix_frac": 0.085,
                    }
                ),
                "dismiss": OpticalFieldSlotOverride(
                    params={
                        "mip_blur_strength": 0.0,
                        "band_width_frac": 0.034,
                        "tail_width_frac": 0.024,
                        "ring_amplitude_frac": 0.028,
                        "tail_amplitude_frac": 0.010,
                        "exterior_mix_frac": 0.075,
                    }
                ),
            },
        ),
        visibility_scope="independent",
        z_index=42,
    )


def compile_perceptasia_primitive_carrier_config(
    bounds: OpticalFieldBounds,
    *,
    state: OpticalFieldState = "rest",
    visible: bool = True,
    content_proof_required: bool = True,
    materialization_progress: float | None = None,
) -> dict[str, object]:
    """Compile the optical shell envelope for the captured Perceptasia payload.

    The WebView is the primitive payload, not an unrelated sibling plate. The
    compositor captures it, preserves interior pixels, and clips rectangular
    carrier corners outside the optical body.
    """

    request = build_perceptasia_primitive_request(bounds, state=state, visible=visible)
    config = compile_placeholder_shell_config(request)
    material_opacity_by_state = {
        "materialize": 0.34,
        "rest": 0.22,
        "dismiss": 0.30,
        "hidden": 0.0,
    }
    material_ridge_by_state = {
        "materialize": 0.68,
        "rest": 0.54,
        "dismiss": 0.62,
        "hidden": 0.0,
    }
    capture_live_payload = bool(visible and state == "rest")
    config.update(
        {
            "visible": bool(visible and state != "hidden"),
            "gpu_material_enabled": 1.0,
            "gpu_material_opacity": material_opacity_by_state.get(state, 0.22),
            "gpu_material_feather_points": 118.0,
            "gpu_material_fill_overscan_points": 4.0,
            "gpu_material_base_width_points": float(bounds.width),
            "gpu_material_base_height_points": float(bounds.height),
            "gpu_material_base_corner_radius_points": float(
                config["corner_radius_points"]
            ),
            "gpu_material_height_frac": 1.0,
            "gpu_material_text_contrast_bias": 0.55,
            "gpu_material_ridge_emphasis": material_ridge_by_state.get(state, 0.54),
            "mip_blur_strength": 0.0,
            "throughglass_content_carrier": (
                "captured_webview_payload" if capture_live_payload else "shell_transition_only"
            ),
            "include_carrier_window_in_capture": capture_live_payload,
            "clip_captured_carrier_to_shell": capture_live_payload,
            "content_proof_required": bool(content_proof_required),
        }
    )
    if materialization_progress is not None:
        _apply_materialization_progress(config, materialization_progress)
    return config


def build_perceptasia_primitive_env(
    *,
    provider_url: str,
    content_proof_required: bool = True,
    publish_shell: bool = False,
) -> Mapping[str, str]:
    """Return Perceptasia-only env overrides for a primitive consumer surface."""

    normalized_provider_url = provider_url.strip().rstrip("/")
    if not normalized_provider_url:
        raise ValueError("provider_url must be non-empty")
    return {
        PERCEPTASIA_PRIMITIVE_PROVIDER_ENV: normalized_provider_url,
        PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_ENV: "1"
        if content_proof_required
        else "0",
        PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL_ENV: "1" if publish_shell else "0",
    }
