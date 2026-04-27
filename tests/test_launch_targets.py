import json

from spoke.launch_targets import current_launch_target, current_launch_target_id, save_selected_launch_target


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
