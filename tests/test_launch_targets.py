import json

from spoke.launch_targets import (
    current_launch_target,
    current_launch_target_id,
    parse_env_overrides,
    resolve_launch_target,
    save_selected_launch_target,
)


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
