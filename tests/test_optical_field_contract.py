from __future__ import annotations

import pytest

from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldDisturbance,
    OpticalFieldPlaceholderBackend,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldSlotOverride,
    OpticalFieldTransitionTiming,
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
                phase=0.25,
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
    assert shell_config["optical_field"] == {
        "caller_id": "agent.card.codex-1",
        "profile": "agent_card",
        "state": "rest",
        "slot": "rest",
        "bounds": {
            "x": 40.0,
            "y": 80.0,
            "width": 320.0,
            "height": 96.0,
        },
        "previous_bounds": None,
        "transition_timing": None,
        "disturbances": ("ready-pulse",),
    }


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


def test_resize_is_first_class_lifecycle_with_profile_timing_and_bounds_metadata():
    backend = OpticalFieldPlaceholderBackend()
    timing = OpticalFieldTransitionTiming(
        duration_s=0.24,
        attack_curve="ease_out_cubic",
        release_curve="critically_damped",
        params={"settle_epsilon": 0.01},
    )
    profile = OpticalFieldProfileRef(
        base="agent_card",
        slots={
            "resize": OpticalFieldSlotOverride(
                params={"ring_amplitude_frac": 0.12, "mip_blur_strength": 0.15}
            )
        },
    )
    original_bounds = OpticalFieldBounds(x=20.0, y=30.0, width=240.0, height=80.0)
    next_bounds = OpticalFieldBounds(x=40.0, y=50.0, width=360.0, height=120.0)
    request = OpticalFieldRequest(
        caller_id="agent.card.codex-1",
        bounds=original_bounds,
        role="agent_card",
        profile=profile,
    ).resize_to(next_bounds, timing=timing)

    backend.upsert(request)

    (shell_config,) = backend.compile_shell_configs()
    assert request.state == "resize"
    assert request.previous_bounds == original_bounds
    assert request.transition_timing == timing
    assert shell_config["content_width_points"] == pytest.approx(360.0)
    assert shell_config["content_height_points"] == pytest.approx(120.0)
    assert shell_config["ring_amplitude_points"] == pytest.approx(14.4)
    assert shell_config["mip_blur_strength"] == pytest.approx(0.15)
    assert shell_config["optical_field"]["slot"] == "resize"
    assert shell_config["optical_field"]["bounds"] == {
        "x": 40.0,
        "y": 50.0,
        "width": 360.0,
        "height": 120.0,
    }
    assert shell_config["optical_field"]["previous_bounds"] == {
        "x": 20.0,
        "y": 30.0,
        "width": 240.0,
        "height": 80.0,
    }
    assert shell_config["optical_field"]["transition_timing"] == {
        "duration_s": 0.24,
        "attack_curve": "ease_out_cubic",
        "release_curve": "critically_damped",
        "params": {"settle_epsilon": 0.01},
    }


def test_recenter_preserves_size_while_center_changes_as_primitive_lifecycle():
    backend = OpticalFieldPlaceholderBackend()
    timing = OpticalFieldTransitionTiming(
        duration_s=0.18,
        attack_curve="snap",
        release_curve="soft_land",
    )
    profile = OpticalFieldProfileRef(
        base="assistant_shell",
        slots={
            "recenter": OpticalFieldSlotOverride(
                params={"band_width_frac": 0.11, "tail_width_frac": 0.02}
            )
        },
    )
    original_bounds = OpticalFieldBounds(x=100.0, y=120.0, width=300.0, height=90.0)
    request = OpticalFieldRequest(
        caller_id="assistant",
        bounds=original_bounds,
        role="assistant",
        profile=profile,
    ).recenter_to(center_x=420.0, center_y=260.0, timing=timing)

    backend.upsert(request)

    (shell_config,) = backend.compile_shell_configs()
    assert request.state == "recenter"
    assert request.bounds.width == pytest.approx(300.0)
    assert request.bounds.height == pytest.approx(90.0)
    assert request.bounds.x == pytest.approx(270.0)
    assert request.bounds.y == pytest.approx(215.0)
    assert request.previous_bounds == original_bounds
    assert shell_config["center_x"] == pytest.approx(420.0)
    assert shell_config["center_y"] == pytest.approx(260.0)
    assert shell_config["content_width_points"] == pytest.approx(300.0)
    assert shell_config["content_height_points"] == pytest.approx(90.0)
    assert shell_config["band_width_points"] == pytest.approx(9.9)
    assert shell_config["tail_width_points"] == pytest.approx(1.8)
    assert shell_config["optical_field"]["state"] == "recenter"
    assert shell_config["optical_field"]["slot"] == "recenter"
    assert shell_config["optical_field"]["transition_timing"]["attack_curve"] == "snap"


def test_resize_and_recenter_replace_in_flight_lifecycle_requests_for_same_caller():
    backend = OpticalFieldPlaceholderBackend()
    original_bounds = OpticalFieldBounds(x=0.0, y=0.0, width=200.0, height=80.0)
    base = OpticalFieldRequest(
        caller_id="agent.card.codex-1",
        bounds=original_bounds,
        role="agent_card",
    )

    backend.upsert(base.as_materializing())
    backend.upsert(base.resize_to(OpticalFieldBounds(x=20.0, y=20.0, width=280.0, height=120.0)))
    backend.upsert(base.as_dismissing())
    backend.upsert(base.recenter_to(center_x=500.0, center_y=220.0))

    (shell_config,) = backend.compile_shell_configs()
    assert shell_config["optical_field"]["state"] == "recenter"
    assert shell_config["center_x"] == pytest.approx(500.0)
    assert shell_config["center_y"] == pytest.approx(220.0)
    assert "queued_transitions" not in shell_config["optical_field"]


def test_lifecycle_helpers_keep_consumers_out_of_progress_and_shader_ownership():
    backend = OpticalFieldPlaceholderBackend()
    request = OpticalFieldRequest(
        caller_id="preview",
        bounds=OpticalFieldBounds(x=0.0, y=0.0, width=180.0, height=42.0),
        role="preview",
        profile=OpticalFieldProfileRef(base="preview_pill"),
    )

    materializing = request.as_materializing()
    resting = materializing.as_resting()
    dismissing = resting.as_dismissing()
    hidden = dismissing.as_hidden()

    assert materializing.state == "materialize"
    assert resting.state == "rest"
    assert dismissing.state == "dismiss"
    assert hidden.state == "hidden"
    assert hidden.visible is False
    assert materializing.profile is request.profile
    assert not hasattr(materializing, "progress")

    backend.upsert(hidden)
    assert backend.compile_shell_configs() == ()


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
