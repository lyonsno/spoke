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
                        "ring_amplitude_frac": 0.018,
                    }
                ),
                "dismiss": OpticalFieldSlotOverride(
                    params={
                        "mip_blur_strength": 0.0,
                        "ring_amplitude_frac": 0.012,
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
    remains the content authority, so shell material is disabled and cannot
    visually substitute for missing WebKit/WebGL pixels.
    """

    request = build_perceptasia_primitive_request(bounds, state=state, visible=visible)
    config = compile_placeholder_shell_config(request)
    config.update(
        {
            "visible": bool(visible and state != "hidden"),
            "gpu_material_enabled": 0.0,
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
