from __future__ import annotations

import pytest

from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldDisturbance,
    OpticalFieldPlaceholderBackend,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldSlotOverride,
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
        "progress": 1.0,
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


def test_materialization_progress_is_part_of_the_reusable_contract():
    backend = OpticalFieldPlaceholderBackend()
    bounds = OpticalFieldBounds(x=100.0, y=200.0, width=1200.0, height=220.0)

    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.shell",
            bounds=bounds,
            role="assistant",
            state="materialize",
            progress=0.0,
            profile=OpticalFieldProfileRef(base="assistant_shell"),
        )
    )
    start = backend.compile_shell_configs()[0]

    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.shell",
            bounds=bounds,
            role="assistant",
            state="materialize",
            progress=1.0,
            profile=OpticalFieldProfileRef(base="assistant_shell"),
        )
    )
    finished = backend.compile_shell_configs()[0]

    assert start["client_id"] == "assistant.shell"
    assert start["content_width_points"] < finished["content_width_points"]
    assert start["content_height_points"] < finished["content_height_points"]
    assert start["core_magnification"] < finished["core_magnification"]
    assert finished["content_width_points"] == pytest.approx(bounds.width)
    assert finished["content_height_points"] == pytest.approx(bounds.height)
    assert finished["optical_field"]["progress"] == pytest.approx(1.0)


def test_dismiss_compiles_generic_seam_and_radial_sidecars_without_private_ids():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="agent.card.codex-1",
            bounds=OpticalFieldBounds(x=40.0, y=80.0, width=640.0, height=160.0),
            role="agent_card",
            state="dismiss",
            progress=0.35,
            profile=OpticalFieldProfileRef(base="assistant_shell"),
            z_index=4,
        )
    )

    configs = backend.compile_shell_configs()
    by_id = {config["client_id"]: config for config in configs}

    assert set(by_id) == {
        "agent.card.codex-1",
        "agent.card.codex-1.dismiss_seam",
        "agent.card.codex-1.dismiss_radial_pucker",
    }
    assert all(not client_id.startswith("assistant.command") for client_id in by_id)

    seam = by_id["agent.card.codex-1.dismiss_seam"]
    assert seam["z_index"] > by_id["agent.card.codex-1"]["z_index"]
    assert seam["warp_mode"] in {1, 3}
    assert seam["mip_blur_strength"] == pytest.approx(0.0)
    assert seam["scar_amount"] > 0.0
    assert seam["optical_field"]["sidecar"] == "dismiss_seam"

    radial = by_id["agent.card.codex-1.dismiss_radial_pucker"]
    assert radial["warp_mode"] == 2
    assert radial["mip_blur_strength"] == pytest.approx(0.0)
    assert radial["optical_field"]["sidecar"] == "dismiss_radial_pucker"


def test_reusable_backend_allows_distinct_consumers_to_share_pressure_lifecycle():
    backend = OpticalFieldPlaceholderBackend()
    backend.upsert(
        OpticalFieldRequest(
            caller_id="assistant.shell",
            bounds=OpticalFieldBounds(x=120.0, y=600.0, width=1400.0, height=260.0),
            role="assistant",
            state="materialize",
            progress=0.45,
            profile=OpticalFieldProfileRef(base="assistant_shell"),
            z_index=10,
        )
    )
    backend.upsert(
        OpticalFieldRequest(
            caller_id="semantic.card.move-bitch",
            bounds=OpticalFieldBounds(x=80.0, y=120.0, width=380.0, height=110.0),
            role="agent_card",
            state="dismiss",
            progress=0.30,
            profile=OpticalFieldProfileRef(
                base="assistant_shell",
                params={"core_magnification": 1.05, "mip_blur_strength": 0.35},
            ),
            z_index=3,
        )
    )

    configs = backend.compile_shell_configs()
    ids = {config["client_id"] for config in configs}

    assert "assistant.shell" in ids
    assert "semantic.card.move-bitch" in ids
    assert "semantic.card.move-bitch.dismiss_seam" in ids
    assert "semantic.card.move-bitch.dismiss_radial_pucker" in ids
    assert all(config["optical_field"]["profile"] == "assistant_shell" for config in configs)
