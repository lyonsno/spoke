"""Gutterglass consumer passport for the shared House optical primitive."""

from __future__ import annotations

from spoke.house_optical_primitive import (
    OPTICAL_SHELL_FEATHER,
    apply_dismiss_radial_pucker_fields,
    compile_house_optical_shell_config,
    dismiss_pucker_house_shell_config,
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


GUTTERGLASS_PRIMITIVE_CLIENT_ID = "gutterglass.smoke_stage"
GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID = "gutterglass.smoke_stage.radial_pucker"
GUTTERGLASS_PRIMITIVE_LAYOUT_RECIPE = "gutterglass-smoke-stage"


def _sync_cut_radius(config: dict[str, object]) -> None:
    config["cut_radius_points"] = float(config["corner_radius_points"])
    if isinstance(config.get("optical_field"), dict):
        optical_field = dict(config["optical_field"])
        optical_field["cut_radius_points"] = float(config["corner_radius_points"])
        config["optical_field"] = optical_field


def _apply_materialization_progress(config: dict[str, object], progress: float) -> None:
    config.update(materialized_house_optical_shell_config(config, progress))
    _sync_cut_radius(config)


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


def build_gutterglass_primitive_request(
    bounds: OpticalFieldBounds,
    *,
    state: OpticalFieldState = "rest",
    visible: bool = True,
    continuity_key: str = GUTTERGLASS_PRIMITIVE_CLIENT_ID,
) -> OpticalFieldRequest:
    """Build Gutterglass's request for the shared House primitive."""

    return OpticalFieldRequest(
        caller_id=GUTTERGLASS_PRIMITIVE_CLIENT_ID,
        continuity_key=continuity_key,
        bounds=bounds,
        role="hud",
        state=state,
        visible=visible,
        presentation=OpticalFieldPresentation(layer="hud", order=36),
        presentation_layer="hud",
        layout_recipe=GUTTERGLASS_PRIMITIVE_LAYOUT_RECIPE,
        motion=OpticalFieldMotionIntent(
            strategy="continuous",
            continuity="preserve_identity",
            urgency="normal",
            latency_mask="source_proof",
        ),
        profile=OpticalFieldProfileRef(base="assistant_shell"),
        visibility_scope="independent",
        z_index=36,
    )


def compile_gutterglass_primitive_stage_config(
    bounds: OpticalFieldBounds,
    *,
    state: OpticalFieldState = "rest",
    visible: bool = True,
    materialization_progress: float | None = None,
) -> dict[str, object]:
    """Compile the optical shell for the Gutterglass smoke-stage panel."""

    request = build_gutterglass_primitive_request(bounds, state=state, visible=visible)
    config = compile_placeholder_shell_config(request)
    _apply_house_shell_contract(config, bounds)
    material_opacity_by_state = {
        "materialize": 0.32,
        "rest": 0.0,
        "dismiss": 0.28,
        "hidden": 0.0,
    }
    config.update(
        {
            "visible": bool(visible and state != "hidden"),
            "gpu_material_enabled": 1.0,
            "gpu_material_opacity": material_opacity_by_state.get(state, 0.22),
            "gpu_material_feather_points": OPTICAL_SHELL_FEATHER,
            "gpu_material_fill_overscan_points": material_fill_overscan_points(),
            "gpu_material_height_frac": 1.0,
            "gpu_material_text_contrast_bias": 0.55,
            "gpu_material_ridge_emphasis": 0.52,
            "mip_blur_strength": 0.0,
            "content_carrier": "external_stage_panel" if state == "rest" else "shell_transition_only",
            "include_carrier_window_in_capture": False,
            "clip_captured_carrier_to_shell": False,
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


def compile_gutterglass_dismiss_pucker_config(
    shell_config: dict[str, object],
    progress: float,
) -> dict[str, object]:
    """Compile Gutterglass's post-dismiss radial damp oscillator."""

    config = dismiss_pucker_house_shell_config(shell_config, progress)
    config = apply_dismiss_radial_pucker_fields(config, progress)
    config["client_id"] = GUTTERGLASS_RADIAL_PUCKER_CLIENT_ID
    config["role"] = "hud"
    config["visible"] = True
    config["z_index"] = 35
    config["presentation_layer"] = "hud"
    return config
