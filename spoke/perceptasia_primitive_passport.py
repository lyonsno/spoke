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
) -> dict[str, object]:
    """Compile the optical shell envelope for a live WebView captured by the primitive.

    The optical primitive places, animates, and rims the surface by capturing
    the live Perceptasia WebView into the compositor. The Perceptasia renderer
    remains the content authority; the shader material is only the perimeter
    pressure field around that captured payload, never a substitute fill.
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
            "throughglass_content_carrier": "captured_webview",
            "include_carrier_window_in_capture": True,
            "clip_captured_carrier_to_shell": True,
            "content_proof_required": bool(content_proof_required),
        }
    )
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
