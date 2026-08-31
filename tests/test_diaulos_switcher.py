"""Contract tests for the voice-native live Diaulos switcher."""

from __future__ import annotations

import importlib
import json
import os
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
        "runtime_lineage_required": True,
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
                "tty": f"/dev/ttys{index + 10:03d}",
                "resume_backend": "codex",
                "thread_id": f"thread-{index}",
                "match_basis": ["endpoint_thread_id"],
            }
            for index in range(count)
        ],
        "excluded": [],
    }


def _selected_pane_payload(candidate, **overrides) -> dict:
    entry = {
        "handle": candidate.handle,
        "diaulos_id": candidate.diaulos_id,
        "aliases": list(candidate.aliases),
        "pane_id": candidate.pane_id,
        "tab_id": candidate.tab_id,
        "window_id": candidate.window_id,
        "title": candidate.title,
        "cwd": candidate.cwd,
        "tty": candidate.tty,
        "resume_backend": candidate.resume_backend,
        "thread_id": candidate.thread_id,
        "match_basis": list(candidate.match_basis),
    }
    entry.update(overrides.pop("entry", {}))
    payload = {
        "status": "complete",
        "observed_at": "2026-08-31T12:10:51Z",
        "discovery_authority": "exact-selected-pane-enumeration",
        "observation_scope": "selected-pane",
        "requested_pane_id": candidate.pane_id,
        "runtime_lineage_required": True,
        "entries": [entry],
        "excluded": [],
    }
    payload.update(overrides)
    return payload


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
            "tty_name": f"/dev/ttys{index + 10:03d}",
        }
        for index in range(count)
    ]


def test_client_loads_snapshot_and_activates_after_selected_lineage_probe(
    tmp_path,
):
    calls: list[list[str]] = []
    snapshot = tmp_path / "live-diauloi.json"
    snapshot.write_text(json.dumps(_payload()))

    def runner(command, **kwargs):
        calls.append(command)
        if "diaulos" in command and "live" in command:
            candidate = parse_live_inventory(_payload(1))[0]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_selected_pane_payload(candidate)), ""
            )
        if command[-3:] == ["list", "--format", "json"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(_live_panes()), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=snapshot,
        epistaxis_executable="epistaxis",
        epistaxis_repo_root="/explicit/read-mirror",
        wezterm_executable="wezterm",
    )
    candidate = client.load()[0]
    receipt = client.activate(candidate)

    assert calls == [
        [
            "epistaxis", "diaulos", "live",
            "--repo-root", "/explicit/read-mirror",
            "--pane-id", "10",
            "--json",
        ],
        ["wezterm", "cli", "--no-auto-start", "list", "--format", "json"],
        ["wezterm", "cli", "--no-auto-start", "activate-pane", "--pane-id", "10"],
    ]
    assert receipt["pane_id"] == 10
    assert receipt["diaulos"] == "thing-0"
    assert receipt["verification"] == "selected-pane-lineage-and-direct-wezterm-enumeration"


def test_activation_refuses_recycled_pane_with_different_live_lineage(tmp_path):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    recycled = _selected_pane_payload(
        candidate,
        entry={
            "handle": "beaming-baby-cloud-milk",
            "diaulos_id": "dia-beaming",
            "thread_id": "different-thread",
            "tty": "/dev/ttys037",
        },
    )

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(recycled), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        epistaxis_repo_root="/explicit/read-mirror",
        wezterm_executable="wezterm",
    )

    with pytest.raises(DiaulosActivationError) as error:
        client.activate(candidate)

    message = str(error.value)
    assert "thing-0" in message
    assert "beaming-baby-cloud-milk" in message
    assert "thread-0" in message
    assert "different-thread" in message
    assert calls == [
        [
            "epistaxis", "diaulos", "live",
            "--repo-root", "/explicit/read-mirror",
            "--pane-id", "10",
            "--json",
        ]
    ]


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


def test_refresh_drops_inherited_wezterm_socket_without_castrating_environment(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "live-diauloi.json"
    payload = _payload(1)
    observed_environments: list[dict[str, str]] = []
    monkeypatch.setenv("WEZTERM_UNIX_SOCKET", "/tmp/dead-gui-sock")
    monkeypatch.setenv("SPOKE_SUBPROCESS_SENTINEL", "preserved")

    def runner(command, **kwargs):
        observed_environments.append(dict(kwargs.get("env", os.environ)))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=snapshot,
        epistaxis_executable="epistaxis",
    ).refresh()

    assert len(observed_environments) == 1
    assert "WEZTERM_UNIX_SOCKET" not in observed_environments[0]
    assert observed_environments[0]["SPOKE_SUBPROCESS_SENTINEL"] == "preserved"


def test_activation_drops_inherited_wezterm_socket_from_both_cli_calls(
    tmp_path,
    monkeypatch,
):
    observed_environments: list[dict[str, str]] = []
    monkeypatch.setenv("WEZTERM_UNIX_SOCKET", "/tmp/dead-gui-sock")

    def runner(command, **kwargs):
        observed_environments.append(dict(kwargs.get("env", os.environ)))
        if "diaulos" in command and "live" in command:
            candidate = parse_live_inventory(_payload(1))[0]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_selected_pane_payload(candidate)), ""
            )
        if command[-3:] == ["list", "--format", "json"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(_live_panes()),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )
    client.activate(parse_live_inventory(_payload(1))[0])

    assert len(observed_environments) == 3
    assert all(
        "WEZTERM_UNIX_SOCKET" not in environment
        for environment in observed_environments
    )


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
        if "diaulos" in command and "live" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_selected_pane_payload(candidate)), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(panes), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )
    candidate = parse_live_inventory(_payload(1))[0]

    with pytest.raises(DiaulosActivationError, match=message):
        client.activate(candidate)

    assert len(calls) == 2
    assert calls[0][-3:] == ["--pane-id", "10", "--json"]
    assert calls[1][-3:] == ["list", "--format", "json"]


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


def test_unexpected_activation_exception_returns_control_to_overlay(overlay_module):
    candidate = parse_live_inventory(_payload(1))[0]

    class BrokenClient:
        def activate(self, selected):
            raise ValueError("malformed activation environment")

    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay._client = BrokenClient()
    overlay.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()

    overlay._activation_worker(7, candidate)

    selector = overlay.performSelectorOnMainThread_withObject_waitUntilDone_
    selector.assert_called_once()
    name, payload, wait = selector.call_args.args
    assert name == "activationFinished:"
    assert payload["generation"] == 7
    assert "unexpected activation failure" in payload["error"]
    assert "malformed activation environment" in payload["error"]
    assert wait is False


@pytest.mark.parametrize(
    ("query", "expected_matches"),
    [("thing-1", 1), ("ordinary dictation with no Diaulos handle", 0)],
)
def test_dictation_filter_reports_applied_match_count(
    overlay_module, query, expected_matches
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._activation_in_flight = False
    overlay._model = DiaulosSwitcherModel(parse_live_inventory(_payload(3)))
    overlay._search_field = MagicMock()
    overlay._panel = MagicMock()
    overlay._render_rows = MagicMock()

    assert overlay.set_dictation_filter(query) == expected_matches
    overlay._search_field.setStringValue_.assert_called_once_with(query)


def test_dictation_filter_reports_not_applied_during_committed_activation(
    overlay_module,
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._activation_in_flight = True
    overlay._model = DiaulosSwitcherModel(parse_live_inventory(_payload(3)))
    overlay._search_field = MagicMock()

    assert overlay.set_dictation_filter("do not retarget") is None
    overlay._search_field.setStringValue_.assert_not_called()


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
    assert overlay.presentation_generation == 1
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


def test_show_orders_panel_front_before_cached_row_rebuild(
    overlay_module,
    monkeypatch,
):
    events: list[str] = []
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay._model = DiaulosSwitcherModel(parse_live_inventory(_payload(20)))
    overlay._search_field = MagicMock()
    overlay._count_label = MagicMock()
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._panel.makeKeyAndOrderFront_.side_effect = lambda _: events.append(
        "panel-front"
    )
    overlay._render_rows = MagicMock(side_effect=lambda: events.append("rows-rendered"))
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
    monkeypatch.setattr(
        overlay_module.threading,
        "Thread",
        MagicMock(return_value=thread),
    )

    overlay.show()

    assert events == ["panel-front", "rows-rendered"]


def test_prewarm_builds_panel_and_primes_snapshot_off_the_gesture_path(
    overlay_module,
    monkeypatch,
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.setup = MagicMock()
    overlay.visible = False
    overlay._model = DiaulosSwitcherModel([])
    overlay._prewarm_in_flight = False
    thread = MagicMock()
    thread_factory = MagicMock(return_value=thread)
    monkeypatch.setattr(overlay_module.threading, "Thread", thread_factory)

    overlay.prewarm()

    overlay.setup.assert_called_once_with()
    assert overlay._prewarm_in_flight is True
    thread_factory.assert_called_once()
    assert thread_factory.call_args.kwargs["target"] == overlay._prewarm_worker
    thread.start.assert_called_once_with()


def test_prewarm_completion_populates_and_renders_hidden_snapshot(overlay_module):
    candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = False
    overlay._model = DiaulosSwitcherModel([])
    overlay._load_in_flight = False
    overlay._prewarm_in_flight = True
    overlay._render_rows = MagicMock()

    overlay.prewarmFinished_({"candidates": candidates, "elapsed_ms": 12.5})

    assert overlay._prewarm_in_flight is False
    assert overlay._model.all_candidates == candidates
    overlay._render_rows.assert_called_once_with()


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


def test_inventory_completion_defers_row_rebuild_behind_committed_activation(
    overlay_module,
):
    old_candidate = parse_live_inventory(_payload(1))[0]
    new_candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._load_generation = 7
    overlay._load_in_flight = True
    overlay._activation_in_flight = True
    overlay._pending_inventory_payload = None
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._render_rows = MagicMock()

    payload = {"generation": 7, "candidates": new_candidates}
    overlay.inventoryLoaded_(payload)

    assert overlay._model.all_candidates == [old_candidate]
    assert overlay._pending_inventory_payload == payload
    overlay._render_rows.assert_not_called()


def test_activation_failure_retains_deferred_snapshot_before_refresh_error(
    overlay_module,
):
    old_candidate = parse_live_inventory(_payload(1))[0]
    new_candidates = parse_live_inventory(_payload(2))
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._load_generation = 7
    overlay._load_in_flight = True
    overlay._activation_generation = 4
    overlay._activation_in_flight = True
    overlay._activation_handle = "thing-0"
    overlay._pending_inventory_payload = None
    overlay._pending_inventory_error_payload = None
    overlay._search_field = MagicMock()
    overlay._search_field.stringValue.return_value = ""
    overlay._status_label = MagicMock()
    overlay._panel = MagicMock()
    overlay._render_rows = MagicMock()

    snapshot = {
        "generation": 7,
        "candidates": new_candidates,
        "refreshing": True,
    }
    refresh_error = {
        "generation": 7,
        "error": "Epistaxis release is changing",
        "refreshing": False,
    }
    overlay.inventoryLoaded_(snapshot)
    overlay.inventoryLoaded_(refresh_error)

    assert overlay._model.all_candidates == [old_candidate]
    assert overlay._load_in_flight is False
    overlay._render_rows.assert_not_called()

    overlay.activationFinished_({"generation": 4, "error": "route moved"})

    assert overlay._model.all_candidates == new_candidates
    assert overlay._activation_in_flight is False
    overlay._search_field.setEnabled_.assert_called_once_with(True)
    overlay._panel.makeFirstResponder_.assert_called_once_with(
        overlay._search_field
    )
    final_status = overlay._status_label.setStringValue_.call_args.args[0]
    assert "route moved" in final_status
    assert "Epistaxis release is changing" in final_status
    overlay._render_rows.assert_called_once_with()


def test_activation_success_caches_deferred_snapshot_before_refresh_error(
    overlay_module,
):
    old_candidate = parse_live_inventory(_payload(1))[0]
    new_candidates = parse_live_inventory(_payload(2))
    events: list[str] = []
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._model = DiaulosSwitcherModel([old_candidate])
    overlay._load_generation = 7
    overlay._load_in_flight = True
    overlay._activation_generation = 4
    overlay._activation_in_flight = True
    overlay._activation_handle = "thing-0"
    overlay._pending_inventory_payload = None
    overlay._pending_inventory_error_payload = None
    overlay._search_field = MagicMock()
    overlay._status_label = MagicMock()
    overlay._render_rows = MagicMock()

    def hide(*, restore_previous):
        events.append("panel-hidden")
        overlay.visible = False

    overlay.hide = MagicMock(side_effect=hide)
    overlay._activate_wezterm = MagicMock(
        side_effect=lambda: events.append("wezterm-foregrounded")
    )

    overlay.inventoryLoaded_(
        {
            "generation": 7,
            "candidates": new_candidates,
            "refreshing": True,
        }
    )
    overlay.inventoryLoaded_(
        {
            "generation": 7,
            "error": "Epistaxis release is changing",
            "refreshing": False,
        }
    )

    assert overlay._model.all_candidates == [old_candidate]
    overlay._render_rows.assert_not_called()

    overlay.activationFinished_({"generation": 4, "receipt": {"pane_id": 10}})

    assert events == ["panel-hidden", "wezterm-foregrounded"]
    assert overlay._model.all_candidates == new_candidates
    assert overlay._load_in_flight is False
    overlay._render_rows.assert_not_called()


def test_client_logs_subprocess_phase_duration(caplog):
    ticks = iter((10.0, 10.25))
    client = EpistaxisDiaulosClient(
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            "[]",
            "",
        ),
        clock=lambda: next(ticks),
    )

    with caplog.at_level("INFO", logger="spoke.diaulos_switcher"):
        client._run_process(
            ["wezterm", "cli", "--no-auto-start", "list", "--format", "json"],
            DiaulosActivationError,
        )

    assert "phase=wezterm_list" in caplog.text
    assert "elapsed_ms=250.0" in caplog.text
    assert "returncode=0" in caplog.text


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
