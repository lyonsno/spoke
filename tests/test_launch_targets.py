import json
from pathlib import Path

from spoke.launch_targets import current_launch_target, current_launch_target_id


def test_current_launch_target_id_returns_matching_target(tmp_path):
    current_checkout = tmp_path / "target-a"
    current_checkout.mkdir()
    other_checkout = tmp_path / "target-b"
    other_checkout.mkdir()
    targets_file = tmp_path / "launch_targets.json"
    targets_file.write_text(
        (
            '{"selected":"target-b","targets":['
            '{"id":"target-a","label":"Target A","path":"%s"},'
            '{"id":"target-b","label":"Target B","path":"%s"}'
            "]}"
        )
        % (current_checkout, other_checkout)
    )

    assert current_launch_target_id(current_checkout, targets_file) == "target-a"


def test_current_launch_target_id_ignores_selected_when_checkout_not_registered(tmp_path):
    current_checkout = tmp_path / "unregistered"
    current_checkout.mkdir()
    targets_file = tmp_path / "launch_targets.json"
    targets_file.write_text(
        (
            '{"selected":"target-b","targets":['
            '{"id":"target-a","label":"Target A","path":"%s"},'
            '{"id":"target-b","label":"Target B","path":"%s"}'
            "]}"
        )
        % (tmp_path / "target-a", tmp_path / "target-b")
    )

    assert current_launch_target_id(current_checkout, targets_file) is None


def test_current_launch_target_returns_visible_label_for_registered_checkout(tmp_path):
    checkout = tmp_path / "airstrike"
    checkout.mkdir()
    targets_file = tmp_path / "launch_targets.json"
    targets_file.write_text(
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

    target = current_launch_target(checkout, targets_file)

    assert target == {
        "id": "airstrike",
        "label": "Assistant Backend on Main Next Airstrike",
        "path": checkout,
        "enabled": True,
    }
