from __future__ import annotations

from dataclasses import fields

import pytest

from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldCoordinateContext,
    OpticalFieldDisturbance,
    OpticalFieldMotionIntent,
    OpticalFieldPlaceholderBackend,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldSignal,
    OpticalFieldSlotOverride,
    compile_placeholder_shell_config,
    normalize_optical_field_bounds,
    optical_field_overlap_ratio,
    resolve_optical_field_motion,
)


def test_placeholder_backend_preserves_contract_identity_and_profile_slot_metadata():
    backend = OpticalFieldPlaceholderBackend()
    request = OpticalFieldRequest(
        caller_id="agent.card.codex-1",
        bounds=OpticalFieldBounds(x=40.0, y=80.0, width=320.0, height=96.0),
        role="agent_card",
        state="rest",
        profile=OpticalFieldProfileRef(
            base="agent_card",
            slots={
                "rest": OpticalFieldSlotOverride(params={"core_magnification": 1.08}),
                "dismiss": OpticalFieldSlotOverride(params={"mip_blur_strength": 0.0}),
            },
        ),
        disturbances=(
            OpticalFieldDisturbance(
                disturbance_id="ready-pulse",
                kind="readiness_pulse",
                mode="ephemeral",
                strength=0.35,
            ),
        ),
        z_index=7,
    )

    backend.upsert(request)

    (shell_config,) = backend.compile_shell_configs()
    assert shell_config["client_id"] == "agent.card.codex-1"
    assert shell_config["role"] == "agent_card"
    assert shell_config["z_index"] == 7
    assert shell_config["content_width_points"] == pytest.approx(320.0)
    assert shell_config["content_height_points"] == pytest.approx(96.0)
    assert shell_config["center_x"] == pytest.approx(200.0)
    assert shell_config["center_y"] == pytest.approx(128.0)
    assert shell_config["core_magnification"] == pytest.approx(1.08)
    assert shell_config["optical_field"]["caller_id"] == "agent.card.codex-1"
    assert shell_config["optical_field"]["role"] == "agent_card"
    assert shell_config["optical_field"]["resolved_presentation_layer"] == "agent_card"
    assert shell_config["optical_field"]["presentation_order"] == 20
    assert shell_config["optical_field"]["visibility_scope"] == "independent"
    assert shell_config["optical_field"]["profile"] == "agent_card"
    assert shell_config["optical_field"]["state"] == "rest"
    assert shell_config["optical_field"]["slot"] == "rest"
    assert shell_config["optical_field"]["disturbances"] == ("ready-pulse",)
    assert "phase" not in shell_config["optical_field"]
    assert "progress" not in shell_config["optical_field"]


def test_profile_slots_override_independently_without_leaking_between_lifecycle_states():
    backend = OpticalFieldPlaceholderBackend()
    profile = OpticalFieldProfileRef(
        base="assistant_shell",
        slots={
            "materialize": OpticalFieldSlotOverride(
                params={"ring_amplitude_frac": 0.20, "mip_blur_strength": 0.25}
            ),
            "dismiss": OpticalFieldSlotOverride(
                params={"ring_amplitude_frac": 0.04, "mip_blur_strength": 0.0}
            ),
        },
    )
    bounds = OpticalFieldBounds(x=10.0, y=20.0, width=600.0, height=180.0)

    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=bounds,
            role="assistant",
            state="materialize",
            profile=profile,
        )
    )
    materialize = backend.compile_shell_configs()[0]

    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=bounds,
            role="assistant",
            state="dismiss",
            profile=profile,
        )
    )
    dismiss = backend.compile_shell_configs()[0]

    assert materialize["ring_amplitude_points"] == pytest.approx(36.0)
    assert materialize["mip_blur_strength"] == pytest.approx(0.25)
    assert dismiss["ring_amplitude_points"] == pytest.approx(7.2)
    assert dismiss["mip_blur_strength"] == pytest.approx(0.0)


def test_normalized_profile_values_scale_with_geometry_not_raw_preview_tuning():
    backend = OpticalFieldPlaceholderBackend()
    profile = OpticalFieldProfileRef(base="agent_card")

    backend.upsert(
        OpticalFieldRequest(
            caller_id="small",
            bounds=OpticalFieldBounds(x=0.0, y=0.0, width=240.0, height=80.0),
            role="agent_card",
            state="rest",
            profile=profile,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="large",
            bounds=OpticalFieldBounds(x=0.0, y=0.0, width=960.0, height=320.0),
            role="agent_card",
            state="rest",
            profile=profile,
        )
    )

    small, large = backend.compile_shell_configs()
    assert large["ring_amplitude_points"] / small["ring_amplitude_points"] == pytest.approx(4.0)
    assert large["band_width_points"] / small["band_width_points"] == pytest.approx(4.0)
    assert large["corner_radius_points"] / small["corner_radius_points"] == pytest.approx(4.0)


def test_backend_upsert_and_remove_are_stable_by_caller_id():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="preview",
            bounds=OpticalFieldBounds(x=0.0, y=0.0, width=180.0, height=42.0),
            role="preview",
            state="rest",
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="preview",
            bounds=OpticalFieldBounds(x=10.0, y=20.0, width=240.0, height=64.0),
            role="preview",
            state="rest",
        )
    )

    (updated,) = backend.compile_shell_configs()
    assert updated["center_x"] == pytest.approx(130.0)
    assert updated["center_y"] == pytest.approx(52.0)

    assert backend.remove("preview") is True
    assert backend.compile_shell_configs() == ()
    assert backend.remove("preview") is False


def test_placeholder_configs_survive_fullscreen_compositor_snapshot_round_trip():
    from spoke.fullscreen_compositor import (
        OverlayClientIdentity,
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.codex-1",
            bounds=OpticalFieldBounds(x=40.0, y=80.0, width=320.0, height=96.0),
            role="agent_card",
            state="rest",
            profile=OpticalFieldProfileRef(base="agent_card"),
        )
    )
    (shell_config,) = backend.compile_shell_configs()

    snapshot = _snapshot_from_shell_config(
        OverlayClientIdentity(
            client_id="agent.card.codex-1",
            display_id=1,
            role="preview",
        ),
        shell_config,
        generation=3,
    )
    round_tripped = _snapshot_to_shell_config(snapshot)

    assert round_tripped["optical_field"] == shell_config["optical_field"]


def test_public_consumer_contract_round_trips_coordinate_content_motion_and_freshness():
    request = OpticalFieldRequest(
        caller_id="semantic.assistant",
        continuity_key="thread.codex-123",
        bounds=OpticalFieldBounds(x=100.0, y=140.0, width=640.0, height=220.0),
        content_frame=OpticalFieldBounds(x=124.0, y=164.0, width=592.0, height=172.0),
        coordinate_space="screen_points",
        display_epoch="display-main:42",
        source_epoch="capture:vlm:17",
        freshness_epoch="request:99",
        role="assistant_shell",
        state="retarget",
        presentation_layer="assistant",
        layout_recipe="semantic-placement-candidate",
        motion=OpticalFieldMotionIntent(
            strategy="auto",
            urgency="normal",
            latency_mask="pending_resolution",
            params={"overlap_threshold": 0.50},
        ),
        continuity="preserve_identity",
        signals=(
            OpticalFieldSignal(
                name="audio_rms",
                value=0.42,
                freshness_epoch="audio:5",
                params={"window_ms": 80.0},
            ),
            OpticalFieldSignal(
                name="pending_resolution",
                value=True,
                freshness_epoch="vlm:pending",
            ),
        ),
        provisional=True,
        confidence=0.76,
        z_index=9,
    )

    shell_config = compile_placeholder_shell_config(request)
    optical_field = shell_config["optical_field"]

    assert optical_field["caller_id"] == "semantic.assistant"
    assert optical_field["continuity_key"] == "thread.codex-123"
    assert optical_field["lifecycle"] == "retarget"
    assert optical_field["bounds"] == {
        "x": 100.0,
        "y": 140.0,
        "width": 640.0,
        "height": 220.0,
    }
    assert optical_field["content_frame"] == {
        "x": 124.0,
        "y": 164.0,
        "width": 592.0,
        "height": 172.0,
    }
    assert optical_field["coordinate_space"] == "screen_points"
    assert optical_field["display_epoch"] == "display-main:42"
    assert optical_field["source_epoch"] == "capture:vlm:17"
    assert optical_field["freshness_epoch"] == "request:99"
    assert optical_field["presentation_layer"] == "assistant"
    assert optical_field["layout_recipe"] == "semantic-placement-candidate"
    assert optical_field["motion"] == {
        "strategy": "auto",
        "continuity": "preserve_identity",
        "overlap_threshold": 0.50,
        "same_presence_strategy": "squirt",
        "urgency": "normal",
        "latency_mask": "pending_resolution",
        "params": {"overlap_threshold": 0.50},
    }
    assert optical_field["continuity"] == "preserve_identity"
    assert optical_field["signals"] == (
        {
            "name": "audio_rms",
            "value": 0.42,
            "freshness_epoch": "audio:5",
            "params": {"window_ms": 80.0},
        },
        {
            "name": "pending_resolution",
            "value": True,
            "freshness_epoch": "vlm:pending",
            "params": {},
        },
    )
    assert optical_field["provisional"] is True
    assert optical_field["final"] is False
    assert optical_field["confidence"] == pytest.approx(0.76)


def test_production_consumer_request_schema_excludes_progress_and_phase_custody():
    request_field_names = {field.name for field in fields(OpticalFieldRequest)}
    disturbance_field_names = {field.name for field in fields(OpticalFieldDisturbance)}
    signal_field_names = {field.name for field in fields(OpticalFieldSignal)}

    assert "progress" not in request_field_names
    assert "phase" not in request_field_names
    assert "progress" not in disturbance_field_names
    assert "phase" not in disturbance_field_names
    assert "progress" not in signal_field_names
    assert "phase" not in signal_field_names

    kwargs = {
        "caller_id": "preview",
        "bounds": OpticalFieldBounds(x=0.0, y=0.0, width=180.0, height=42.0),
        "role": "preview",
    }
    with pytest.raises(TypeError):
        OpticalFieldRequest(**kwargs, progress=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpticalFieldRequest(**kwargs, phase=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumer-authored progress/phase"):
        OpticalFieldSignal(name="progress", value=0.5)
    with pytest.raises(ValueError, match="consumer-authored progress/phase"):
        OpticalFieldSignal(name="transition.phase", value=0.5)


def test_legacy_minimal_requests_compile_with_explicit_public_contract_defaults():
    request = OpticalFieldRequest(
        caller_id="assistant",
        bounds=OpticalFieldBounds(x=16.0, y=24.0, width=640.0, height=180.0),
        role="assistant_shell",
    )

    optical_field = compile_placeholder_shell_config(request)["optical_field"]

    assert optical_field["continuity_key"] == "assistant"
    assert optical_field["lifecycle"] == "rest"
    assert optical_field["bounds"] == optical_field["content_frame"]
    assert optical_field["coordinate_space"] == "display_local"
    assert optical_field["display_epoch"] is None
    assert optical_field["source_epoch"] is None
    assert optical_field["freshness_epoch"] is None
    assert optical_field["presentation_layer"] == "default"
    assert optical_field["layout_recipe"] == "direct_positioned"
    assert optical_field["motion"]["strategy"] == "auto"
    assert optical_field["continuity"] == "preserve_identity"
    assert optical_field["signals"] == ()
    assert optical_field["provisional"] is False
    assert optical_field["final"] is True
    assert optical_field["confidence"] is None


def test_backing_pixel_bounds_normalize_to_display_points_with_scale_and_epoch_metadata():
    bounds = OpticalFieldBounds(x=200.0, y=100.0, width=640.0, height=192.0)
    context = OpticalFieldCoordinateContext(
        coordinate_space="backing_pixels",
        display_id="main-display",
        display_epoch="display-7",
        source_epoch="capture-3",
        backing_scale=2.0,
    )

    normalized = normalize_optical_field_bounds(bounds, context)

    assert normalized == OpticalFieldBounds(x=100.0, y=50.0, width=320.0, height=96.0)

    backend = OpticalFieldPlaceholderBackend(display_epochs={"main-display": "display-7"})
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.pixel-space",
            bounds=bounds,
            role="agent_card",
            coordinate_context=context,
        )
    )
    (shell_config,) = backend.compile_shell_configs()

    assert shell_config["center_x"] == pytest.approx(260.0)
    assert shell_config["center_y"] == pytest.approx(98.0)
    assert shell_config["optical_field"]["coordinate_space"] == "display_points"
    assert shell_config["optical_field"]["source_coordinate_space"] == "backing_pixels"
    assert shell_config["optical_field"]["display_id"] == "main-display"
    assert shell_config["optical_field"]["display_epoch"] == "display-7"
    assert shell_config["optical_field"]["source_epoch"] == "capture-3"
    assert shell_config["optical_field"]["backing_scale"] == pytest.approx(2.0)


def test_backing_pixel_bounds_without_scale_fail_loudly_instead_of_impersonating_points():
    context = OpticalFieldCoordinateContext(
        coordinate_space="backing_pixels",
        display_id="main-display",
        display_epoch="display-7",
    )

    with pytest.raises(ValueError, match="backing_scale"):
        normalize_optical_field_bounds(
            OpticalFieldBounds(x=200.0, y=100.0, width=640.0, height=192.0),
            context,
        )


def test_parent_local_optical_bounds_and_content_frame_compile_as_distinct_display_rects():
    backend = OpticalFieldPlaceholderBackend(display_epochs={"main-display": "display-7"})
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.parent-local",
            bounds=OpticalFieldBounds(x=20.0, y=10.0, width=320.0, height=96.0),
            content_frame=OpticalFieldBounds(x=36.0, y=22.0, width=260.0, height=52.0),
            role="agent_card",
            coordinate_context=OpticalFieldCoordinateContext(
                coordinate_space="parent_points",
                display_id="main-display",
                display_epoch="display-7",
                parent_origin=(100.0, 200.0),
            ),
            content_coordinate_context=OpticalFieldCoordinateContext(
                coordinate_space="content_points",
                display_id="main-display",
                display_epoch="display-7",
                content_origin=(120.0, 214.0),
            ),
        )
    )

    (shell_config,) = backend.compile_shell_configs()

    assert shell_config["center_x"] == pytest.approx(280.0)
    assert shell_config["center_y"] == pytest.approx(258.0)
    assert shell_config["content_width_points"] == pytest.approx(320.0)
    assert shell_config["content_height_points"] == pytest.approx(96.0)
    assert shell_config["optical_field"]["bounds"] == {
        "x": 120.0,
        "y": 210.0,
        "width": 320.0,
        "height": 96.0,
    }
    assert shell_config["optical_field"]["content_frame"] == {
        "x": 156.0,
        "y": 236.0,
        "width": 260.0,
        "height": 52.0,
    }


def test_backend_rejects_stale_display_and_capture_epochs_before_compiling_geometry():
    backend = OpticalFieldPlaceholderBackend(
        display_epochs={"main-display": "display-7"},
        source_epochs={"main-display": "capture-3"},
    )

    with pytest.raises(ValueError, match="stale display_epoch"):
        backend.upsert(
            OpticalFieldRequest(
                caller_id="agent.card.old-display",
                bounds=OpticalFieldBounds(x=0.0, y=0.0, width=100.0, height=40.0),
                role="agent_card",
                coordinate_context=OpticalFieldCoordinateContext(
                    coordinate_space="display_points",
                    display_id="main-display",
                    display_epoch="display-6",
                    source_epoch="capture-3",
                ),
            )
        )

    with pytest.raises(ValueError, match="stale source_epoch"):
        backend.upsert(
            OpticalFieldRequest(
                caller_id="agent.card.old-capture",
                bounds=OpticalFieldBounds(x=0.0, y=0.0, width=100.0, height=40.0),
                role="agent_card",
                coordinate_context=OpticalFieldCoordinateContext(
                    coordinate_space="display_points",
                    display_id="main-display",
                    display_epoch="display-7",
                    source_epoch="capture-2",
                ),
            )
        )


def test_material_signals_compile_as_finite_material_basis_not_shader_knobs():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.command",
            bounds=OpticalFieldBounds(x=10.0, y=20.0, width=600.0, height=160.0),
            role="assistant_shell",
            state="rest",
            profile=OpticalFieldProfileRef(base="assistant_shell"),
            signals=(
                OpticalFieldSignal(name="background_luminance", value=0.78),
                OpticalFieldSignal(name="text_contrast_bias", value=0.64),
                OpticalFieldSignal(name="ridge_emphasis", value=0.35),
            ),
        )
    )

    (shell_config,) = backend.compile_shell_configs()

    assert shell_config["gpu_material_enabled"] == pytest.approx(1.0)
    assert shell_config["gpu_material_brightness"] == pytest.approx(0.78)
    assert shell_config["gpu_material_text_contrast_bias"] == pytest.approx(0.64)
    assert shell_config["gpu_material_ridge_emphasis"] == pytest.approx(0.35)
    assert shell_config["optical_field"]["signals"] == (
        {"name": "background_luminance", "value": 0.78, "freshness_epoch": None, "params": {}},
        {"name": "text_contrast_bias", "value": 0.64, "freshness_epoch": None, "params": {}},
        {"name": "ridge_emphasis", "value": 0.35, "freshness_epoch": None, "params": {}},
    )
    assert "shader_params" not in shell_config["optical_field"]


def test_unknown_signals_do_not_compile_into_gpu_material_knobs():
    shell_config = compile_placeholder_shell_config(
        OpticalFieldRequest(
            caller_id="assistant.command",
            bounds=OpticalFieldBounds(x=10.0, y=20.0, width=600.0, height=160.0),
            role="assistant_shell",
            signals=(OpticalFieldSignal(name="fragment_shader_uniform", value=0.5),),
        )
    )

    assert "gpu_material_enabled" not in shell_config
    assert shell_config["optical_field"]["signals"] == (
        {"name": "fragment_shader_uniform", "value": 0.5, "freshness_epoch": None, "params": {}},
    )

def _field_request(
    caller_id: str,
    *,
    bounds: OpticalFieldBounds,
    state: str = "rest",
    display_epoch: int = 1,
    source_epoch: int | None = 1,
    provisional: bool = False,
    visible: bool = True,
) -> OpticalFieldRequest:
    return OpticalFieldRequest(
        caller_id=caller_id,
        bounds=bounds,
        role="agent_card",
        state=state,
        profile=OpticalFieldProfileRef(base="agent_card"),
        display_epoch=display_epoch,
        source_epoch=source_epoch,
        provisional=provisional,
        visible=visible,
    )


def _bounds_tuple(bounds: OpticalFieldBounds) -> tuple[float, float, float, float]:
    return (bounds.x, bounds.y, bounds.width, bounds.height)


def test_transition_mailbox_coalesces_same_caller_provisional_geometry_to_latest_target():
    backend = OpticalFieldPlaceholderBackend()
    initial = OpticalFieldBounds(x=20.0, y=30.0, width=240.0, height=80.0)
    older_target = OpticalFieldBounds(x=40.0, y=50.0, width=300.0, height=96.0)
    latest_target = OpticalFieldBounds(x=60.0, y=70.0, width=360.0, height=112.0)

    backend.upsert(_field_request("agent.card.codex-1", bounds=initial, display_epoch=1))
    first_resize = backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=older_target,
            state="resize",
            display_epoch=2,
            source_epoch=2,
            provisional=True,
        )
    )
    second_resize = backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=latest_target,
            state="resize",
            display_epoch=3,
            source_epoch=3,
            provisional=True,
        )
    )

    assert first_resize.accepted is True
    assert second_resize.accepted is True
    (request,) = backend.requests()
    assert request.bounds == latest_target
    transition = backend.transition_for("agent.card.codex-1")
    assert transition is not None
    assert transition.previous_bounds == initial
    assert transition.presented_bounds == initial
    assert transition.target_bounds == latest_target
    assert transition.pending_request is None


def test_transition_mailbox_rejects_stale_display_and_source_epochs_without_replay():
    backend = OpticalFieldPlaceholderBackend()
    accepted_bounds = OpticalFieldBounds(x=10.0, y=10.0, width=200.0, height=60.0)
    stale_display_bounds = OpticalFieldBounds(x=300.0, y=10.0, width=200.0, height=60.0)
    stale_source_bounds = OpticalFieldBounds(x=500.0, y=10.0, width=200.0, height=60.0)

    accepted = backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=accepted_bounds,
            state="resize",
            display_epoch=5,
            source_epoch=10,
        )
    )
    stale_display = backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=stale_display_bounds,
            state="resize",
            display_epoch=4,
            source_epoch=11,
        )
    )
    stale_source = backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=stale_source_bounds,
            state="resize",
            display_epoch=5,
            source_epoch=9,
        )
    )

    assert accepted.accepted is True
    assert stale_display.accepted is False
    assert stale_display.reason == "stale_display_epoch"
    assert stale_source.accepted is False
    assert stale_source.reason == "stale_source_epoch"
    (request,) = backend.requests()
    assert request.bounds == accepted_bounds


def test_transition_mailbox_dismiss_and_hidden_clear_pending_geometry_for_caller():
    backend = OpticalFieldPlaceholderBackend()
    materialize_bounds = OpticalFieldBounds(x=100.0, y=100.0, width=320.0, height=120.0)
    pending_bounds = OpticalFieldBounds(x=180.0, y=140.0, width=420.0, height=140.0)
    dismiss_bounds = OpticalFieldBounds(x=180.0, y=140.0, width=420.0, height=140.0)

    backend.upsert(
        _field_request(
            "preview",
            bounds=materialize_bounds,
            state="materialize",
            display_epoch=1,
            provisional=True,
        )
    )
    backend.upsert(
        _field_request(
            "preview",
            bounds=pending_bounds,
            state="resize",
            display_epoch=2,
            provisional=True,
        )
    )
    dismiss = backend.upsert(
        _field_request(
            "preview",
            bounds=dismiss_bounds,
            state="dismiss",
            display_epoch=3,
            provisional=False,
        )
    )

    assert dismiss.accepted is True
    transition = backend.transition_for("preview")
    assert transition is not None
    assert transition.target_request.state == "dismiss"
    assert transition.target_bounds == dismiss_bounds
    assert transition.pending_request is None

    hidden = backend.upsert(
        _field_request(
            "preview",
            bounds=dismiss_bounds,
            state="hidden",
            display_epoch=4,
            provisional=False,
            visible=False,
        )
    )
    assert hidden.accepted is True
    transition = backend.transition_for("preview")
    assert transition is not None
    assert transition.target_request.state == "hidden"
    assert transition.pending_request is None
    assert backend.compile_shell_configs() == ()


def test_transition_mailbox_interrupts_from_sampled_presented_bounds_not_stale_from_bounds():
    backend = OpticalFieldPlaceholderBackend()
    initial = OpticalFieldBounds(x=0.0, y=0.0, width=240.0, height=80.0)
    first_target = OpticalFieldBounds(x=400.0, y=0.0, width=320.0, height=100.0)
    sampled_presented = OpticalFieldBounds(x=160.0, y=0.0, width=272.0, height=88.0)
    interrupted_target = OpticalFieldBounds(x=160.0, y=220.0, width=300.0, height=96.0)

    backend.upsert(_field_request("assistant", bounds=initial, display_epoch=1))
    backend.upsert(
        _field_request(
            "assistant",
            bounds=first_target,
            state="resize",
            display_epoch=2,
            source_epoch=2,
        )
    )
    backend.sample_presented_bounds("assistant", sampled_presented)
    interrupt = backend.upsert(
        _field_request(
            "assistant",
            bounds=interrupted_target,
            state="recenter",
            display_epoch=3,
            source_epoch=3,
        )
    )

    assert interrupt.accepted is True
    transition = backend.transition_for("assistant")
    assert transition is not None
    assert transition.previous_bounds == sampled_presented
    assert transition.presented_bounds == sampled_presented
    assert transition.target_bounds == interrupted_target

    (shell_config,) = backend.compile_shell_configs()
    metadata = shell_config["optical_field"]["transition"]
    assert metadata["from_bounds"] == pytest.approx(_bounds_tuple(sampled_presented))
    assert metadata["target_bounds"] == pytest.approx(_bounds_tuple(interrupted_target))


def test_transition_mailbox_payload_survives_fullscreen_compositor_snapshot_round_trip():
    from spoke.fullscreen_compositor import (
        OverlayClientIdentity,
        _snapshot_from_shell_config,
        _snapshot_to_shell_config,
    )

    backend = OpticalFieldPlaceholderBackend()
    initial = OpticalFieldBounds(x=20.0, y=30.0, width=240.0, height=80.0)
    sampled_presented = OpticalFieldBounds(x=50.0, y=60.0, width=260.0, height=88.0)
    target = OpticalFieldBounds(x=80.0, y=90.0, width=320.0, height=96.0)

    backend.upsert(_field_request("agent.card.codex-1", bounds=initial, display_epoch=1))
    backend.sample_presented_bounds("agent.card.codex-1", sampled_presented)
    backend.upsert(
        _field_request(
            "agent.card.codex-1",
            bounds=target,
            state="resize",
            display_epoch=2,
            source_epoch=7,
        )
    )
    (shell_config,) = backend.compile_shell_configs()

    snapshot = _snapshot_from_shell_config(
        OverlayClientIdentity(
            client_id="agent.card.codex-1",
            display_id=1,
            role="preview",
        ),
        shell_config,
        generation=4,
    )
    round_tripped = _snapshot_to_shell_config(snapshot)

    assert round_tripped["optical_field"]["transition"]["from_bounds"] == pytest.approx(
        _bounds_tuple(sampled_presented)
    )
    assert round_tripped["optical_field"]["transition"]["target_bounds"] == pytest.approx(
        _bounds_tuple(target)
    )
    assert round_tripped["optical_field"]["transition"]["display_epoch"] == 2
    assert round_tripped["optical_field"]["transition"]["source_epoch"] == 7


def test_auto_motion_overlap_metric_uses_intersection_over_smaller_area():
    current = OpticalFieldBounds(x=0.0, y=0.0, width=100.0, height=100.0)

    assert optical_field_overlap_ratio(
        current,
        OpticalFieldBounds(x=25.0, y=0.0, width=100.0, height=100.0),
    ) == pytest.approx(0.75)
    assert optical_field_overlap_ratio(
        current,
        OpticalFieldBounds(x=50.0, y=0.0, width=100.0, height=100.0),
    ) == pytest.approx(0.50)
    assert optical_field_overlap_ratio(
        current,
        OpticalFieldBounds(x=10.0, y=10.0, width=50.0, height=50.0),
    ) == pytest.approx(1.0)
    assert optical_field_overlap_ratio(
        current,
        OpticalFieldBounds(x=160.0, y=0.0, width=100.0, height=100.0),
    ) == pytest.approx(0.0)


def test_auto_motion_policy_prefers_same_presence_only_above_threshold():
    current = OpticalFieldBounds(x=0.0, y=0.0, width=100.0, height=100.0)
    intent = OpticalFieldMotionIntent(strategy="auto")

    near = resolve_optical_field_motion(
        current,
        OpticalFieldBounds(x=24.0, y=0.0, width=100.0, height=100.0),
        intent,
    )
    boundary = resolve_optical_field_motion(
        current,
        OpticalFieldBounds(x=50.0, y=0.0, width=100.0, height=100.0),
        intent,
    )
    forced_new = resolve_optical_field_motion(
        current,
        OpticalFieldBounds(x=10.0, y=0.0, width=100.0, height=100.0),
        OpticalFieldMotionIntent(strategy="auto", continuity="new_presence"),
    )

    assert near.resolved_strategy == "squirt"
    assert near.same_presence is True
    assert near.overlap_ratio == pytest.approx(0.76)
    assert boundary.resolved_strategy == "dematerialize_rematerialize"
    assert boundary.same_presence is False
    assert boundary.overlap_ratio == pytest.approx(0.50)
    assert forced_new.resolved_strategy == "dematerialize_rematerialize"
    assert forced_new.reason == "new_presence_continuity"


def test_auto_motion_metadata_uses_mailbox_presented_bounds_for_retarget_decision():
    backend = OpticalFieldPlaceholderBackend()
    initial = OpticalFieldBounds(x=0.0, y=0.0, width=240.0, height=80.0)
    stale_target = OpticalFieldBounds(x=500.0, y=0.0, width=240.0, height=80.0)
    sampled_presented = OpticalFieldBounds(x=32.0, y=0.0, width=240.0, height=80.0)
    final_target = OpticalFieldBounds(x=60.0, y=0.0, width=240.0, height=80.0)

    backend.upsert(_field_request("assistant", bounds=initial, display_epoch=1))
    backend.upsert(
        _field_request(
            "assistant",
            bounds=stale_target,
            state="recenter",
            display_epoch=2,
            source_epoch=2,
            provisional=True,
        )
    )
    backend.sample_presented_bounds("assistant", sampled_presented)
    final = backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=final_target,
            role="agent_card",
            state="recenter",
            profile=OpticalFieldProfileRef(base="agent_card"),
            display_epoch=3,
            source_epoch=3,
            provisional=False,
            motion=OpticalFieldMotionIntent(strategy="auto"),
        )
    )

    assert final.accepted is True
    (shell_config,) = backend.compile_shell_configs()
    motion = shell_config["optical_field"]["resolved_motion"]
    assert motion["requested_strategy"] == "auto"
    assert motion["resolved_strategy"] == "squirt"
    assert motion["same_presence"] is True
    assert motion["overlap_ratio"] == pytest.approx(0.8833333333)
    assert shell_config["optical_field"]["transition"]["from_bounds"] == pytest.approx(
        _bounds_tuple(sampled_presented)
    )


def test_final_auto_motion_interrupts_provisional_and_rejects_obsolete_provisional_fifo():
    backend = OpticalFieldPlaceholderBackend()
    initial = OpticalFieldBounds(x=0.0, y=0.0, width=200.0, height=80.0)
    provisional_target = OpticalFieldBounds(x=24.0, y=0.0, width=200.0, height=80.0)
    final_target = OpticalFieldBounds(x=48.0, y=0.0, width=200.0, height=80.0)
    obsolete_target = OpticalFieldBounds(x=72.0, y=0.0, width=200.0, height=80.0)
    motion = OpticalFieldMotionIntent(strategy="auto")

    backend.upsert(_field_request("assistant", bounds=initial, display_epoch=1))
    provisional = backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=provisional_target,
            role="agent_card",
            state="recenter",
            profile=OpticalFieldProfileRef(base="agent_card"),
            display_epoch=2,
            source_epoch=2,
            provisional=True,
            motion=motion,
        )
    )
    final = backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=final_target,
            role="agent_card",
            state="recenter",
            profile=OpticalFieldProfileRef(base="agent_card"),
            display_epoch=3,
            source_epoch=3,
            provisional=False,
            motion=motion,
        )
    )
    obsolete = backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant",
            bounds=obsolete_target,
            role="agent_card",
            state="recenter",
            profile=OpticalFieldProfileRef(base="agent_card"),
            display_epoch=3,
            source_epoch=3,
            provisional=True,
            motion=motion,
        )
    )

    assert provisional.accepted is True
    assert final.accepted is True
    assert obsolete.accepted is False
    assert obsolete.reason == "stale_provisional_after_final"
    (request,) = backend.requests()
    assert request.bounds == final_target
    assert request.provisional is False
