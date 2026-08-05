"""Contract tests for the voice-native live Diaulos switcher."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

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


@pytest.fixture
def overlay_module(mock_pyobjc):
    sys.modules.pop("spoke.diaulos_switcher_overlay", None)
    module = importlib.import_module("spoke.diaulos_switcher_overlay")
    yield module
    sys.modules.pop("spoke.diaulos_switcher_overlay", None)


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


@pytest.mark.parametrize("field", ["tab_id", "window_id", "cwd"])
def test_parse_live_inventory_rejects_partial_activation_route(field):
    payload = _payload(1)
    payload["entries"][0].pop(field)

    with pytest.raises(DiaulosInventoryError, match=field):
        parse_live_inventory(payload)


@pytest.mark.parametrize("cwd", ["file://", "relative/path"])
def test_parse_live_inventory_rejects_malformed_activation_cwd(cwd):
    payload = _payload(1)
    payload["entries"][0]["cwd"] = cwd

    with pytest.raises(DiaulosInventoryError, match="cwd"):
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


def _live_panes(count: int = 3) -> list[dict]:
    return [
        {
            "pane_id": index + 10,
            "tab_id": index + 20,
            "window_id": 1,
            "title": f"Thing {index}",
            "cwd": f"file:///tmp/thing-{index}",
        }
        for index in range(count)
    ]


def test_client_loads_snapshot_without_epistaxis_and_activates_directly(
    tmp_path,
):
    calls: list[list[str]] = []
    snapshot = tmp_path / "live-diauloi.json"
    snapshot.write_text(json.dumps(_payload()))

    def runner(command, **kwargs):
        calls.append(command)
        if command[-3:] == ["list", "--format", "json"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(_live_panes()), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=snapshot,
        wezterm_executable="wezterm",
    )
    candidate = client.load()[0]
    receipt = client.activate(candidate)

    assert calls == [
        ["wezterm", "cli", "--no-auto-start", "list", "--format", "json"],
        ["wezterm", "cli", "--no-auto-start", "activate-pane", "--pane-id", "10"],
    ]
    assert receipt["pane_id"] == 10
    assert receipt["diaulos"] == "thing-0"
    assert receipt["verification"] == "direct-wezterm-pane-enumeration"


def test_refresh_atomically_persists_only_complete_inventory(tmp_path):
    snapshot = tmp_path / "live-diauloi.json"
    payload = _payload(4)

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    candidates = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=snapshot,
        epistaxis_executable="epistaxis",
    ).refresh()

    assert len(candidates) == 4
    assert json.loads(snapshot.read_text()) == payload
    assert not list(tmp_path.glob(".live-diauloi.json.*"))


def test_refresh_failure_preserves_exact_previous_snapshot(tmp_path):
    snapshot = tmp_path / "live-diauloi.json"
    previous = json.dumps(_payload(2), indent=2) + "\n"
    snapshot.write_text(previous)

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "Epistaxis live tools are not available at current",
        )

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=snapshot,
        epistaxis_executable="epistaxis",
    )

    with pytest.raises(DiaulosInventoryError, match="live tools are not available"):
        client.refresh()

    assert snapshot.read_text() == previous
    assert [row.handle for row in client.load()] == ["thing-0", "thing-1"]


def test_snapshot_load_does_not_resolve_or_execute_epistaxis(tmp_path, monkeypatch):
    snapshot = tmp_path / "live-diauloi.json"
    snapshot.write_text(json.dumps(_payload(1)))

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("snapshot load must not execute a subprocess")

    monkeypatch.setattr("spoke.diaulos_switcher.shutil.which", lambda *args, **kwargs: None)
    client = EpistaxisDiaulosClient(runner=forbidden_runner, snapshot_path=snapshot)

    assert client.load()[0].handle == "thing-0"


def test_missing_epistaxis_affects_refresh_only(tmp_path, monkeypatch):
    snapshot = tmp_path / "live-diauloi.json"
    snapshot.write_text(json.dumps(_payload(1)))
    monkeypatch.setattr("spoke.diaulos_switcher.shutil.which", lambda *args, **kwargs: None)
    client = EpistaxisDiaulosClient(snapshot_path=snapshot)

    assert client.load()[0].handle == "thing-0"
    with pytest.raises(
        DiaulosInventoryError,
        match="Epistaxis command is unavailable; searched the GUI-safe operator path",
    ):
        client.refresh()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pane_id", 11, "pane is not present exactly once"),
        ("tab_id", 999, "tab_id"),
        ("window_id", 999, "window_id"),
        ("cwd", "file:///tmp/recycled", "cwd"),
    ],
)
def test_direct_activation_refuses_recycled_route_identity(
    tmp_path,
    field,
    value,
    message,
):
    calls: list[list[str]] = []
    panes = _live_panes(1)
    panes[0][field] = value

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(panes), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        wezterm_executable="wezterm",
    )
    candidate = parse_live_inventory(_payload(1))[0]

    with pytest.raises(DiaulosActivationError, match=message):
        client.activate(candidate)

    assert len(calls) == 1
    assert calls[0][-3:] == ["list", "--format", "json"]


def test_client_rejects_malformed_snapshot_and_refresh_output(tmp_path):
    snapshot = tmp_path / "live-diauloi.json"
    snapshot.write_text("not json")
    client = EpistaxisDiaulosClient(
        runner=lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "not json", ""
        ),
        snapshot_path=snapshot,
        epistaxis_executable="epistaxis",
    )

    with pytest.raises(DiaulosInventoryError, match="snapshot returned invalid JSON"):
        client.load()
    with pytest.raises(DiaulosInventoryError, match="inventory returned invalid JSON"):
        client.refresh()


def test_activation_commit_cannot_be_dismissed_or_superseded(overlay_module):
    candidate = parse_live_inventory(_payload(2))[0]
    started = threading.Event()
    release = threading.Event()
    calls = []

    class BlockingClient:
        def activate(self, selected):
            calls.append(selected)
            started.set()
            assert release.wait(timeout=2.0)
            return {
                "diaulos": selected.handle,
                "pane_id": selected.pane_id,
                "expected_pane_id": selected.pane_id,
            }

    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._client = BlockingClient()
    overlay._model = DiaulosSwitcherModel([candidate])
    overlay._activation_generation = 0
    overlay._load_generation = 0
    overlay._panel = MagicMock()
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._previous_app = MagicMock()
    overlay._activation_in_flight = False
    overlay._activation_handle = None
    overlay.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()

    overlay.activate_selected()
    assert started.wait(timeout=1.0)
    try:
        assert overlay.hide() is False
        overlay.toggle()
        overlay.activate_selected()
        deadline = time.monotonic() + 0.5
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert overlay.visible is True
        assert calls == [candidate]
        overlay.cleanup()
        assert overlay.visible is False
        assert overlay._activation_in_flight is True
    finally:
        release.set()


def test_activation_failure_restores_visible_interaction(overlay_module):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._activation_generation = 4
    overlay._activation_in_flight = True
    overlay._activation_handle = "thing-0"
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._keyboard_monitor_available = True

    overlay.activationFinished_({"generation": 4, "error": "route moved"})

    assert overlay.visible is True
    assert overlay._activation_in_flight is False
    assert overlay._activation_handle is None
    overlay._search_field.setEnabled_.assert_called_once_with(True)
    overlay._panel.makeFirstResponder_.assert_called_once_with(
        overlay._search_field
    )
    overlay._status_label.setStringValue_.assert_called_once_with("route moved")


def test_activation_success_hides_panel_before_foregrounding_wezterm(overlay_module):
    events: list[str] = []
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._activation_generation = 4
    overlay._load_generation = 0
    overlay._activation_in_flight = True
    overlay._activation_handle = "thing-0"
    overlay._search_field = MagicMock()
    overlay._panel = MagicMock()
    overlay._panel.orderOut_.side_effect = lambda _: events.append("panel-hidden")
    overlay._previous_app = MagicMock()
    overlay._key_monitor_token = None
    overlay._key_monitor_handler = None
    overlay._keyboard_monitor_available = True
    overlay._activate_wezterm = MagicMock(
        side_effect=lambda: events.append("wezterm-foregrounded")
    )

    overlay.activationFinished_({"generation": 4, "receipt": {"pane_id": 10}})

    assert overlay.visible is False
    assert events == ["panel-hidden", "wezterm-foregrounded"]


def test_visible_overlay_owns_navigation_through_local_key_monitor(
    overlay_module,
    monkeypatch,
):
    candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay._model = DiaulosSwitcherModel([])
    overlay._search_field = MagicMock()
    overlay._count_label = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._document_view = MagicMock()
    overlay._previous_app = None
    overlay._load_generation = 0
    overlay._activation_generation = 0
    overlay._activation_in_flight = False
    overlay._activation_handle = None
    overlay._key_monitor_token = None
    overlay._key_monitor_handler = None
    overlay.visible = False
    thread = MagicMock()
    monkeypatch.setattr(
        overlay_module.threading,
        "Thread",
        MagicMock(return_value=thread),
    )

    overlay.show()

    appkit_event = sys.modules["AppKit"].NSEvent
    add_monitor = appkit_event.addLocalMonitorForEventsMatchingMask_handler_
    add_monitor.assert_called_once()
    handler = add_monitor.call_args.args[1]
    overlay._model = DiaulosSwitcherModel(candidates)
    overlay._render_rows = MagicMock()

    down = MagicMock()
    down.keyCode.return_value = overlay_module._DOWN_ARROW_KEYCODE
    assert handler(down) is None
    assert overlay._model.selected.handle == "thing-1"

    overlay.activate_selected = MagicMock()
    enter = MagicMock()
    enter.keyCode.return_value = next(iter(overlay_module._ENTER_KEYCODES))
    assert handler(enter) is None
    overlay.activate_selected.assert_called_once_with()

    ordinary = MagicMock()
    ordinary.keyCode.return_value = 0
    assert handler(ordinary) is ordinary

    monitor_token = overlay._key_monitor_token
    overlay.hide()
    appkit_event.removeMonitor_.assert_called_once_with(monitor_token)
    assert overlay._key_monitor_token is None


def test_monitor_installation_failure_survives_inventory_status(
    overlay_module,
    monkeypatch,
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay._model = DiaulosSwitcherModel([])
    overlay._search_field = MagicMock()
    overlay._count_label = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._document_view = MagicMock()
    overlay._previous_app = None
    overlay._load_generation = 0
    overlay._activation_generation = 0
    overlay._activation_in_flight = False
    overlay._activation_handle = None
    overlay._key_monitor_token = None
    overlay._key_monitor_handler = None
    overlay.visible = False
    thread = MagicMock()
    monkeypatch.setattr(
        overlay_module.threading,
        "Thread",
        MagicMock(return_value=thread),
    )
    appkit_event = sys.modules["AppKit"].NSEvent
    appkit_event.addLocalMonitorForEventsMatchingMask_handler_.return_value = None

    overlay.show()

    assert "Keyboard navigation unavailable" in (
        overlay._status_label.setStringValue_.call_args.args[0]
    )
    overlay.inventoryLoaded_(
        {
            "generation": overlay._load_generation,
            "candidates": parse_live_inventory(_payload(2)),
        }
    )
    assert "Keyboard navigation unavailable" in (
        overlay._status_label.setStringValue_.call_args.args[0]
    )


def test_show_retains_prior_inventory_while_refreshing(overlay_module, monkeypatch):
    old_candidate = parse_live_inventory(_payload(1))[0]
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._search_field = MagicMock()
    overlay._count_label = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._document_view = MagicMock()
    overlay._previous_app = None
    overlay._load_generation = 0
    overlay._load_in_flight = False
    overlay._activation_generation = 0
    overlay._activation_in_flight = False
    overlay._activation_handle = None
    overlay.visible = False
    thread = MagicMock()
    monkeypatch.setattr(
        overlay_module.threading,
        "Thread",
        MagicMock(return_value=thread),
    )

    overlay.show()

    assert overlay.visible is True
    assert overlay._model.selected == old_candidate
    assert overlay._model.all_candidates == [old_candidate]
    overlay._search_field.setStringValue_.assert_called_once_with("")
    overlay._search_field.setEnabled_.assert_called_once_with(True)
    thread.start.assert_called_once_with()


def test_hide_and_reopen_does_not_fan_out_inventory_refreshes(
    overlay_module,
    monkeypatch,
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay._model = DiaulosSwitcherModel(parse_live_inventory(_payload(1)))
    overlay._search_field = MagicMock()
    overlay._count_label = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._scroll_view = MagicMock()
    overlay._document_view = MagicMock()
    overlay._previous_app = None
    overlay._load_generation = 0
    overlay._load_in_flight = False
    overlay._activation_generation = 0
    overlay._activation_in_flight = False
    overlay._activation_handle = None
    overlay._key_monitor_token = None
    overlay._key_monitor_handler = None
    overlay._keyboard_monitor_available = True
    overlay.visible = False
    thread = MagicMock()
    thread_factory = MagicMock(return_value=thread)
    monkeypatch.setattr(overlay_module.threading, "Thread", thread_factory)

    overlay.show()
    overlay.hide()
    overlay.show()

    assert thread_factory.call_count == 1
    thread.start.assert_called_once_with()


def test_inventory_refresh_failure_retains_prior_inventory(overlay_module):
    old_candidate = parse_live_inventory(_payload(1))[0]
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._load_generation = 7
    overlay._load_in_flight = True
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._render_rows = MagicMock()

    overlay.inventoryLoaded_({"generation": 7, "error": "inventory unavailable"})

    assert overlay._load_in_flight is False
    assert overlay._model.all_candidates == [old_candidate]
    overlay._render_rows.assert_not_called()
    assert "inventory unavailable" in (
        overlay._status_label.setStringValue_.call_args.args[0]
    )


def test_load_worker_publishes_snapshot_before_failed_refresh(overlay_module):
    snapshot_candidates = parse_live_inventory(_payload(2))
    events: list[dict] = []

    class SnapshotThenFailureClient:
        def load(self):
            return snapshot_candidates

        def refresh(self):
            raise overlay_module.DiaulosInventoryError(
                "Epistaxis release is changing"
            )

    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay._client = SnapshotThenFailureClient()
    overlay.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock(
        side_effect=lambda selector, payload, wait: events.append(payload)
    )

    overlay._load_worker(12)

    assert events == [
        {
            "generation": 12,
            "candidates": snapshot_candidates,
            "refreshing": True,
        },
        {
            "generation": 12,
            "error": "Epistaxis release is changing",
            "refreshing": False,
        },
    ]


def test_cached_inventory_keeps_refresh_in_flight_and_is_immediately_filterable(
    overlay_module,
):
    candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._model = DiaulosSwitcherModel([])
    overlay._load_generation = 9
    overlay._load_in_flight = True
    overlay._search_field = MagicMock()
    overlay._search_field.stringValue.return_value = "number 1"
    overlay._status_label = MagicMock()
    overlay._render_rows = MagicMock()
    overlay._keyboard_monitor_available = True

    overlay.inventoryLoaded_(
        {
            "generation": 9,
            "candidates": candidates,
            "refreshing": True,
        }
    )

    assert overlay._load_in_flight is True
    assert [row.handle for row in overlay._model.filtered] == ["thing-1"]
    assert "Snapshot observation" in (
        overlay._status_label.setStringValue_.call_args.args[0]
    )


def test_inventory_refresh_completed_while_hidden_updates_cached_inventory(
    overlay_module,
):
    old_candidate = parse_live_inventory(_payload(1))[0]
    new_candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = False
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._load_generation = 8
    overlay._load_in_flight = True
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._render_rows = MagicMock()

    overlay.inventoryLoaded_(
        {"generation": 8, "candidates": new_candidates}
    )

    assert overlay._load_in_flight is False
    assert overlay._model.all_candidates == new_candidates
    overlay._render_rows.assert_not_called()
