"""Contract tests for the voice-native live Diaulos switcher."""

from __future__ import annotations

import json
import subprocess

import pytest

from spoke.diaulos_switcher import (
    DiaulosActivationError,
    DiaulosInventoryError,
    DiaulosSwitcherModel,
    EpistaxisDiaulosClient,
    parse_live_inventory,
)


def _payload(count: int = 3) -> dict:
    return {
        "status": "complete",
        "observed_at": "2026-07-17T20:00:00Z",
        "discovery_authority": "complete-live-pane-enumeration",
        "entries": [
            {
                "handle": f"thing-{index}",
                "diaulos_id": f"dia-{index}",
                "aliases": [f"thing number {index}"],
                "pane_id": index + 10,
                "tab_id": index + 20,
                "window_id": 1,
                "title": f"Thing {index}",
                "cwd": f"/tmp/thing-{index}",
                "thread_id": f"thread-{index}",
                "match_basis": ["endpoint_thread_id"],
            }
            for index in range(count)
        ],
        "excluded": [],
    }


def test_parse_live_inventory_is_uncapped_and_preserves_observation_identity():
    candidates = parse_live_inventory(_payload(160))

    assert len(candidates) == 160
    assert candidates[-1].handle == "thing-159"
    assert candidates[-1].pane_id == 169
    assert candidates[-1].observed_at == "2026-07-17T20:00:00Z"
    assert candidates[-1].discovery_authority == "complete-live-pane-enumeration"


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "failed", "entries": []},
        {
            "status": "complete",
            "observed_at": "",
            "discovery_authority": "complete-live-pane-enumeration",
            "entries": [],
        },
        {
            "status": "complete",
            "observed_at": "2026-07-17T20:00:00Z",
            "discovery_authority": "workspace-registry-fallback",
            "entries": [],
        },
        {
            "status": "complete",
            "observed_at": "2026-07-17T20:00:00Z",
            "discovery_authority": "complete-live-pane-enumeration",
            "entries": [{"handle": "missing-pane"}],
        },
    ],
)
def test_parse_live_inventory_rejects_false_authority(payload):
    with pytest.raises(DiaulosInventoryError):
        parse_live_inventory(payload)


def test_parse_live_inventory_rejects_duplicate_action_authority():
    payload = _payload(2)
    payload["entries"][1]["handle"] = payload["entries"][0]["handle"]

    with pytest.raises(DiaulosInventoryError, match="multiple panes"):
        parse_live_inventory(payload)


def test_parse_live_inventory_rejects_multiple_handles_for_one_pane():
    payload = _payload(2)
    payload["entries"][1]["pane_id"] = payload["entries"][0]["pane_id"]

    with pytest.raises(DiaulosInventoryError, match="multiple handles"):
        parse_live_inventory(payload)


def test_model_filters_handles_aliases_and_titles_without_mutating_inventory():
    candidates = parse_live_inventory(_payload())
    model = DiaulosSwitcherModel(candidates)

    model.set_query("number 2")
    assert [row.handle for row in model.filtered] == ["thing-2"]

    model.set_query("Thing 1")
    assert [row.handle for row in model.filtered] == ["thing-1"]
    assert len(model.all_candidates) == 3


def test_model_navigation_clamps_and_preserves_selected_identity():
    model = DiaulosSwitcherModel(parse_live_inventory(_payload()))
    assert model.selected.handle == "thing-0"

    model.move(1)
    model.move(1)
    model.move(1)
    assert model.selected.handle == "thing-2"

    model.move(-1)
    assert model.selected.handle == "thing-1"
    model.set_query("number 1")
    assert model.selected.handle == "thing-1"


def test_client_uses_compact_live_api_and_observation_bound_activation():
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["diaulos", "live"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(_payload()), "")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({
                "diaulos": "thing-0",
                "pane_id": 10,
                "expected_pane_id": 10,
            }),
            "",
        )

    client = EpistaxisDiaulosClient(runner=runner)
    candidate = client.load()[0]
    receipt = client.activate(candidate)

    assert calls[0] == ["epistaxis", "diaulos", "live", "--json"]
    assert calls[1] == [
        "epistaxis",
        "focus-pane",
        "--diaulos", "thing-0",
        "--expected-pane-id", "10",
        "--json",
    ]
    assert receipt["pane_id"] == 10


def test_client_rejects_malformed_output_and_activation_mismatch():
    responses = iter([
        subprocess.CompletedProcess([], 0, "not json", ""),
        subprocess.CompletedProcess([], 0, json.dumps(_payload()), ""),
        subprocess.CompletedProcess([], 0, json.dumps({
            "diaulos": "thing-0",
            "pane_id": 11,
            "expected_pane_id": 10,
        }), ""),
    ])
    client = EpistaxisDiaulosClient(runner=lambda *args, **kwargs: next(responses))

    with pytest.raises(DiaulosInventoryError):
        client.load()

    candidate = client.load()[0]
    with pytest.raises(DiaulosActivationError):
        client.activate(candidate)


def test_client_reports_malformed_activation_receipt_as_activation_error():
    responses = iter([
        subprocess.CompletedProcess([], 0, json.dumps(_payload()), ""),
        subprocess.CompletedProcess([], 0, json.dumps({
            "diaulos": "thing-0",
            "pane_id": "not-a-pane",
            "expected_pane_id": 10,
        }), ""),
    ])
    client = EpistaxisDiaulosClient(runner=lambda *args, **kwargs: next(responses))
    candidate = client.load()[0]

    with pytest.raises(DiaulosActivationError, match="pane_id is not an integer"):
        client.activate(candidate)
