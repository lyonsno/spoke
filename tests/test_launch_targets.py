import json

import pytest

import spoke.launch_targets as launch_targets

from spoke.launch_targets import (
    current_launch_target,
    current_launch_target_id,
    parse_env_overrides,
    resolve_launch_target,
    save_selected_launch_target,
)


def test_require_selected_launch_target_rejects_missing_selected_path(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    missing_checkout = tmp_path / "reboot-erased-checkout"
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "label": "Reviewed build",
                        "path": str(missing_checkout),
                    }
                ],
            }
        )
    )

    require_selected = getattr(launch_targets, "require_selected_launch_target", None)
    unavailable_error = getattr(launch_targets, "LaunchTargetUnavailable", None)
    assert callable(require_selected), "launcher needs a required-target resolver"
    assert unavailable_error is not None, "launcher needs an explicit unavailable-target error"

    with pytest.raises(unavailable_error, match="reviewed.*unavailable"):
        require_selected(registry_path)


def test_require_selected_launch_target_rejects_unselected_registry(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    registry_path.write_text(json.dumps({"selected": None, "targets": []}))

    require_selected = getattr(launch_targets, "require_selected_launch_target", None)
    unavailable_error = getattr(launch_targets, "LaunchTargetUnavailable", None)
    assert callable(require_selected), "launcher needs a required-target resolver"
    assert unavailable_error is not None, "launcher needs an explicit unavailable-target error"

    with pytest.raises(unavailable_error, match="No Spoke launch target is selected"):
        require_selected(registry_path)


def test_require_selected_launch_target_uses_one_registry_snapshot(tmp_path, monkeypatch):
    registry_path = tmp_path / "launch_targets.json"
    available_checkout = tmp_path / "available"
    available_checkout.mkdir()
    snapshots = iter(
        [
            json.dumps(
                {
                    "selected": "reviewed",
                    "targets": [
                        {
                            "id": "reviewed",
                            "path": str(tmp_path / "missing-in-first-snapshot"),
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "selected": "other",
                    "targets": [
                        {"id": "reviewed", "path": str(available_checkout)},
                        {"id": "other", "path": str(tmp_path / "missing-other")},
                    ],
                }
            ),
        ]
    )
    reads = []

    def read_snapshot(path, *args, **kwargs):
        assert path == registry_path
        reads.append(path)
        return next(snapshots)

    monkeypatch.setattr(type(registry_path), "read_text", read_snapshot)

    with pytest.raises(launch_targets.LaunchTargetUnavailable, match="reviewed.*unavailable"):
        launch_targets.require_selected_launch_target(registry_path)

    assert reads == [registry_path]


@pytest.mark.parametrize(
    "selected, targets",
    [
        (7, [{"id": 7, "path": "/tmp"}]),
        ("relative", [{"id": "relative", "path": "."}]),
        (
            "duplicate",
            [
                {"id": "duplicate", "path": "/tmp"},
                {"id": "duplicate", "path": "/tmp"},
            ],
        ),
        ("bad-env", [{"id": "bad-env", "path": "/tmp", "env": {"GOOD": "yes", "BAD": 7}}]),
        ("bad-env-shape", [{"id": "bad-env-shape", "path": "/tmp", "env": ["NOPE"]}]),
        ("bad-env-key", [{"id": "bad-env-key", "path": "/tmp", "env": {" ROUTE": "wrong"}}]),
        ("bad-env-equals", [{"id": "bad-env-equals", "path": "/tmp", "env": {"A=B": "wrong"}}]),
        ("bad-env-nul-key", [{"id": "bad-env-nul-key", "path": "/tmp", "env": {"A\x00B": "wrong"}}]),
        ("bad-env-nul-value", [{"id": "bad-env-nul-value", "path": "/tmp", "env": {"A": "x\x00y"}}]),
        ("bad-label-empty", [{"id": "bad-label-empty", "label": "", "path": "/tmp"}]),
        ("bad-label-blank", [{"id": "bad-label-blank", "label": "   ", "path": "/tmp"}]),
        ("bad-label-padded", [{"id": "bad-label-padded", "label": " reviewed ", "path": "/tmp"}]),
        ("bad-label-control", [{"id": "bad-label-control", "label": "reviewed\n", "path": "/tmp"}]),
        ("bad-id\x00", [{"id": "bad-id\x00", "path": "/tmp"}]),
    ],
)
def test_require_selected_launch_target_rejects_malformed_authority(
    tmp_path,
    selected,
    targets,
):
    registry_path = tmp_path / "launch_targets.json"
    registry_path.write_text(json.dumps({"selected": selected, "targets": targets}))

    with pytest.raises(launch_targets.LaunchTargetUnavailable):
        launch_targets.require_selected_launch_target(registry_path)


def test_require_selected_launch_target_rejects_invalid_registry_encoding(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    registry_path.write_bytes(b"\xff")

    with pytest.raises(launch_targets.LaunchTargetUnavailable, match="registry is invalid"):
        launch_targets.require_selected_launch_target(registry_path)


def test_require_selected_launch_target_preserves_valid_absolute_route(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "label": "Reviewed build",
                        "path": str(checkout),
                        "env": {"ROUTE": "reviewed"},
                    }
                ],
            }
        )
    )

    target = launch_targets.require_selected_launch_target(registry_path)

    assert target == {
        "id": "reviewed",
        "label": "Reviewed build",
        "path": checkout,
        "enabled": True,
        "env": {"ROUTE": "reviewed"},
    }


def test_require_selected_launch_target_rejects_protected_authority_env(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(checkout),
                        "env": {"SPOKE_LAUNCH_TARGETS_PATH": "/tmp/other.json"},
                    }
                ],
            }
        )
    )

    with pytest.raises(
        launch_targets.LaunchTargetUnavailable,
        match="protected launch authority keys.*SPOKE_LAUNCH_TARGETS_PATH",
    ):
        launch_targets.require_selected_launch_target(registry_path)


def test_apply_selected_launch_target_env_repairs_missing_launcher_overrides(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "label": "Literal VAD-Off Recovery",
                        "path": str(checkout),
                        "env": {
                            "SPOKE_RETINA_LASSO_AUTO_WITNESS": "0",
                            "SPOKE_VAD_ENABLED": "0",
                        },
                    }
                ],
            }
        )
    )
    process_env = {
        "SPOKE_LAUNCH_TARGET_ID": "reviewed",
        "SPOKE_VAD_ENABLED": "1",
    }

    apply_runtime_env = getattr(
        launch_targets,
        "apply_selected_launch_target_env",
        None,
    )
    assert callable(apply_runtime_env), "runtime needs selected-target env conformance"

    receipt = apply_runtime_env(checkout, registry_path, process_env)

    assert process_env["SPOKE_VAD_ENABLED"] == "0"
    assert process_env["SPOKE_RETINA_LASSO_AUTO_WITNESS"] == "0"
    assert receipt == {
        "status": "repaired",
        "launch_target_id": "reviewed",
        "registry_path": str(registry_path.resolve()),
        "target_env_keys": [
            "SPOKE_RETINA_LASSO_AUTO_WITNESS",
            "SPOKE_VAD_ENABLED",
        ],
        "repaired_env_keys": [
            "SPOKE_LAUNCH_TARGETS_PATH",
            "SPOKE_RETINA_LASSO_AUTO_WITNESS",
            "SPOKE_VAD_ENABLED",
        ],
    }


def test_apply_selected_launch_target_env_reports_already_conformant(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(checkout),
                        "env": {"SPOKE_VAD_ENABLED": "0"},
                    }
                ],
            }
        )
    )
    process_env = {
        "SPOKE_LAUNCH_TARGET_ID": "reviewed",
        "SPOKE_LAUNCH_TARGETS_PATH": str(registry_path.resolve()),
        "SPOKE_VAD_ENABLED": "0",
    }

    receipt = launch_targets.apply_selected_launch_target_env(
        checkout,
        registry_path,
        process_env,
    )

    assert receipt["status"] == "conformant"
    assert receipt["registry_path"] == str(registry_path.resolve())
    assert receipt["target_env_keys"] == ["SPOKE_VAD_ENABLED"]
    assert receipt["repaired_env_keys"] == []


def test_apply_selected_launch_target_env_rejects_wrong_process_identity(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(checkout),
                        "env": {"SPOKE_VAD_ENABLED": "0"},
                    }
                ],
            }
        )
    )

    with pytest.raises(
        launch_targets.LaunchTargetUnavailable,
        match="process target.*other.*selected target.*reviewed",
    ):
        launch_targets.apply_selected_launch_target_env(
            checkout,
            registry_path,
            {"SPOKE_LAUNCH_TARGET_ID": "other"},
        )


def test_apply_selected_launch_target_env_rejects_wrong_checkout(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    other_checkout = tmp_path / "other"
    other_checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "reviewed",
                "targets": [
                    {
                        "id": "reviewed",
                        "path": str(checkout),
                        "env": {"SPOKE_VAD_ENABLED": "0"},
                    }
                ],
            }
        )
    )

    with pytest.raises(
        launch_targets.LaunchTargetUnavailable,
        match="process checkout.*does not match.*reviewed",
    ):
        launch_targets.apply_selected_launch_target_env(
            other_checkout,
            registry_path,
            {"SPOKE_LAUNCH_TARGET_ID": "reviewed"},
        )


def test_apply_selected_launch_target_env_leaves_manual_process_unmanaged(tmp_path):
    process_env = {"UNCHANGED": "yes"}

    receipt = launch_targets.apply_selected_launch_target_env(
        tmp_path,
        tmp_path / "missing.json",
        process_env,
    )

    assert process_env == {"UNCHANGED": "yes"}
    assert receipt == {
        "status": "unmanaged",
        "launch_target_id": None,
        "registry_path": None,
        "target_env_keys": [],
        "repaired_env_keys": [],
    }


def test_save_selected_launch_target_updates_registry_only(tmp_path, monkeypatch):
    registry_path = tmp_path / "launch_targets.json"
    main_target_file = tmp_path / "main-target"
    airstrike = tmp_path / "airstrike"
    butterfingers = tmp_path / "butterfingers"
    airstrike.mkdir()
    butterfingers.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "butterfingers",
                "targets": [
                    {"id": "butterfingers", "label": "Butterfingers", "path": str(butterfingers)},
                    {"id": "airstrike", "label": "Airstrike", "path": str(airstrike)},
                ],
            }
        )
    )
    monkeypatch.setenv("SPOKE_MAIN_TARGET_PATH", str(main_target_file))

    assert save_selected_launch_target("airstrike", registry_path) is True

    payload = json.loads(registry_path.read_text())
    assert payload["selected"] == "airstrike"
    assert not main_target_file.exists()


def test_current_launch_target_id_falls_back_to_selected_when_checkout_unregistered(
    tmp_path,
):
    registry_path = tmp_path / "launch_targets.json"
    airstrike = tmp_path / "airstrike"
    airstrike.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "airstrike",
                "targets": [
                    {"id": "airstrike", "label": "Airstrike", "path": str(airstrike)},
                ],
            }
        )
    )

    assert current_launch_target_id(tmp_path / "some-other-checkout", registry_path) == "airstrike"


def test_current_launch_target_returns_visible_label_for_registered_checkout(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "airstrike"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "airstrike",
                "targets": [
                    {
                        "id": "airstrike",
                        "label": "Assistant Backend on Main Next Airstrike",
                        "path": str(checkout),
                    }
                ],
            }
        )
    )

    target = current_launch_target(checkout, registry_path)

    assert target == {
        "id": "airstrike",
        "label": "Assistant Backend on Main Next Airstrike",
        "path": checkout,
        "enabled": True,
    }


def test_parse_env_overrides_expands_home_variables(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    env_file = tmp_path / ".spoke-smoke-env"
    env_file.write_text(
        'export SPOKE_OPERATOR_PING_EVENTS_PATH="$HOME/.local/state/epistaxis/events.jsonl"\n',
        encoding="utf-8",
    )

    overrides = parse_env_overrides(env_file)

    assert overrides["SPOKE_OPERATOR_PING_EVENTS_PATH"] == (
        str(home / ".local/state/epistaxis/events.jsonl")
    )


def test_resolve_launch_target_preserves_string_env_overrides(tmp_path):
    registry_path = tmp_path / "launch_targets.json"
    checkout = tmp_path / "switcher"
    checkout.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "selected": "switcher",
                "targets": [
                    {
                        "id": "switcher",
                        "label": "Live Diaulos Switcher",
                        "path": str(checkout),
                        "env": {
                            "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE": "0",
                            "SPOKE_RETINA_LASSO_AUTO_WITNESS": "0",
                            "": "ignored-empty-key",
                            "NOT_A_STRING": 7,
                        },
                    }
                ],
            }
        )
    )

    target = resolve_launch_target("switcher", registry_path)

    assert target["env"] == {
        "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE": "0",
        "SPOKE_RETINA_LASSO_AUTO_WITNESS": "0",
    }
