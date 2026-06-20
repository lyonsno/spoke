"""Perceptasia consumer passport for the public optical primitive.

The passport is intentionally pure. It names the optical field Perceptasia may
request and the carrier constraints a live viewer must satisfy, without
importing AppKit/WebKit or mutating assistant-overlay lifecycle state.
"""

from __future__ import annotations

from typing import Mapping

from spoke.house_optical_primitive import (
    OPTICAL_SHELL_FEATHER,
    compile_house_optical_shell_config,
    material_fill_overscan_points,
    materialized_house_optical_shell_config,
    with_gpu_material_basis,
)
from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldMotionIntent,
    OpticalFieldPresentation,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldState,
    compile_placeholder_shell_config,
)


PERCEPTASIA_PRIMITIVE_CLIENT_ID = "perceptasia.throughglass"
PERCEPTASIA_PRIMITIVE_LAYOUT_RECIPE = "perceptasia-primitive-passport"
PERCEPTASIA_PRIMITIVE_PROVIDER_ENV = "SPOKE_PERCEPTASIA_PRIMITIVE_URL"
PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_ENV = (
    "SPOKE_PERCEPTASIA_PRIMITIVE_CONTENT_PROOF_REQUIRED"
)
PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL_ENV = "SPOKE_PERCEPTASIA_PRIMITIVE_PUBLISH_SHELL"


def _apply_materialization_progress(config: dict[str, object], progress: float) -> None:
    config.update(materialized_house_optical_shell_config(config, progress))
    _sync_cut_radius(config)


def _sync_cut_radius(config: dict[str, object]) -> None:
    config["cut_radius_points"] = float(config["corner_radius_points"])
    if isinstance(config.get("optical_field"), dict):
        optical_field = dict(config["optical_field"])
        optical_field["cut_radius_points"] = float(config["corner_radius_points"])
        config["optical_field"] = optical_field


def _apply_house_shell_contract(
    config: dict[str, object],
    bounds: OpticalFieldBounds,
) -> None:
    house_config = compile_house_optical_shell_config(bounds.width, bounds.height)
    config.update(house_config)
    config["mip_blur_strength"] = 0.0
    config.pop("bleed_zone_frac", None)
    config.pop("exterior_mix_width_points", None)
    _sync_cut_radius(config)


def build_perceptasia_primitive_request(
    bounds: OpticalFieldBounds,
    *,
    state: OpticalFieldState = "rest",
    visible: bool = True,
    continuity_key: str = PERCEPTASIA_PRIMITIVE_CLIENT_ID,
) -> OpticalFieldRequest:
    """Build Perceptasia's independent request for the shared House primitive."""

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
            base="assistant_shell",
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
    """Compile the optical shell envelope for the live Perceptasia carrier.

    The WebView remains the live external carrier. The compositor owns the
    shell/perimeter field around that carrier, not the carrier pixels.
    """

    request = build_perceptasia_primitive_request(bounds, state=state, visible=visible)
    config = compile_placeholder_shell_config(request)
    _apply_house_shell_contract(config, bounds)
    material_opacity_by_state = {
        "materialize": 0.34,
        "rest": 0.0,
        "dismiss": 0.30,
        "hidden": 0.0,
    }
    material_ridge_by_state = {
        "materialize": 0.68,
        "rest": 0.54,
        "dismiss": 0.62,
        "hidden": 0.0,
    }
    use_live_carrier = bool(visible and state == "rest")
    config.update(
        {
            "visible": bool(visible and state != "hidden"),
            "gpu_material_enabled": 1.0,
            "gpu_material_opacity": material_opacity_by_state.get(state, 0.22),
            "gpu_material_feather_points": OPTICAL_SHELL_FEATHER,
            "gpu_material_fill_overscan_points": material_fill_overscan_points(),
            "gpu_material_height_frac": 1.0,
            "gpu_material_text_contrast_bias": 0.55,
            "gpu_material_ridge_emphasis": material_ridge_by_state.get(state, 0.54),
            "mip_blur_strength": 0.0,
            "throughglass_content_carrier": (
                "external_webview" if use_live_carrier else "shell_transition_only"
            ),
            "include_carrier_window_in_capture": False,
            "clip_captured_carrier_to_shell": False,
            "content_proof_required": bool(content_proof_required),
        }
    )
    with_gpu_material_basis(
        config,
        width=float(config["content_width_points"]),
        height=float(config["content_height_points"]),
        corner_radius=float(config["corner_radius_points"]),
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
