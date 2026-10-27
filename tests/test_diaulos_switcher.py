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
        "schema_version": 1,
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
        "schema_version": 1,
        "status": "complete",
        "observed_at": "2026-08-31T12:10:51Z",
        "discovery_authority": "exact-selected-pane-enumeration",
        "observation_scope": "selected-pane",
        "requested_pane_id": candidate.pane_id,
        "runtime_lineage_required": True,
        "entries": [entry],
        "excluded": [],
        "identity_conflicts": [],
        "identity_warnings": [],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def overlay_module(mock_pyobjc):
    mock_pyobjc["Quartz"].CATransaction = MagicMock()
    shared = sys.modules.pop("spoke.house_window", None)
    sys.modules.pop("spoke.diaulos_switcher_overlay", None)
    module = importlib.import_module("spoke.diaulos_switcher_overlay")
    yield module
    sys.modules.pop("spoke.diaulos_switcher_overlay", None)
    sys.modules.pop("spoke.house_window", None)
    if shared is not None:
        sys.modules["spoke.house_window"] = shared


def test_parse_live_inventory_is_uncapped_and_preserves_observation_identity():
    candidates = parse_live_inventory(_payload(160))

    assert len(candidates) == 160
    assert candidates[-1].handle == "thing-159"
    assert candidates[-1].pane_id == 169
    assert candidates[-1].observed_at == "2026-07-17T20:00:00Z"
    assert candidates[-1].discovery_authority == "complete-live-pane-enumeration"


def test_hiding_switcher_releases_only_its_optical_client(overlay_module):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(overlay_module.DiaulosSwitcherOverlay)
    overlay.visible = True
    overlay._activation_in_flight = False
    overlay._activation_generation = 0
    overlay._panel = MagicMock()
    overlay._previous_app = None
    overlay._shell_host = MagicMock()
    host = overlay._shell_host
    overlay.hide()
    host.release_client.assert_called_once_with("spoke.teleporter")


def test_teleporter_reports_native_fallback_when_async_capture_fails(overlay_module):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    host = MagicMock()
    overlay.visible = True
    overlay._shell_host = host
    overlay._panel = MagicMock()
    overlay._shell_unavailable = False
    overlay._status_label = MagicMock()
    overlay._keyboard_monitor_available = True

    overlay.opticalShellCaptureStateChanged_(
        {"host": host, "state": "failed", "error": "SCK denied"}
    )

    assert overlay._shell_unavailable is True
    status = overlay._status_label.setStringValue_.call_args.args[0]
    assert status == "Native presentation; optical shell unavailable (SCK denied)"


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


@pytest.mark.parametrize("field", ["pane_id", "tab_id", "window_id"])
@pytest.mark.parametrize("value", [True, False, 1.9, 1.0])
def test_parse_live_inventory_rejects_coercive_identity(field, value):
    payload = _payload(1)
    payload["entries"][0][field] = value

    with pytest.raises(DiaulosInventoryError, match=f"{field} is not an integer"):
        parse_live_inventory(payload)


def test_parse_live_inventory_preserves_integer_string_compatibility():
    payload = _payload(1)
    for field in ("pane_id", "tab_id", "window_id"):
        payload["entries"][0][field] = str(payload["entries"][0][field])

    candidate = parse_live_inventory(payload)[0]
    assert (candidate.pane_id, candidate.tab_id, candidate.window_id) == (10, 20, 1)


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
        ["wezterm", "cli", "--no-auto-start", "activate-pane", "--pane-id", "10"],
    ]
    assert receipt["pane_id"] == 10
    assert receipt["diaulos"] == "thing-0"
    assert receipt["verification"] == "exact-selected-pane-lineage"


@pytest.mark.parametrize(
    "field", ["requested_pane_id", "pane_id", "tab_id", "window_id"],
)
@pytest.mark.parametrize("value", [True, 1.9, 1.0, "1.9", None, {}, []])
def test_activation_rejects_malformed_identity_before_activation(
    tmp_path, field, value
):
    payload = _payload(1)
    payload["entries"][0].update(pane_id=1, tab_id=1, window_id=1)
    candidate = parse_live_inventory(payload)[0]
    selected = _selected_pane_payload(candidate)
    live = _live_panes(1)
    live[0].update(pane_id=1, tab_id=1, window_id=1)
    if field == "requested_pane_id":
        selected[field] = value
    else:
        selected["entries"][0][field] = value
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        result = selected if "diaulos" in command else live
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )
    with pytest.raises(DiaulosActivationError):
        client.activate(candidate)

    assert len(calls) == 1
    assert not any("activate-pane" in command for command in calls)


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


@pytest.mark.parametrize(
    "schema_version",
    [None, "1", 999],
    ids=["missing", "non-integer", "unsupported"],
)
def test_activation_rejects_unversioned_selected_pane_receipt(
    tmp_path, schema_version
):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    selected = _selected_pane_payload(candidate)
    if schema_version is None:
        selected.pop("schema_version")
    else:
        selected["schema_version"] = schema_version

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )

    with pytest.raises(DiaulosActivationError, match="schema version"):
        client.activate(candidate)

    assert len(calls) == 1
    assert calls[0][-3:] == ["--pane-id", "10", "--json"]


@pytest.mark.parametrize(
    ("evidence_field", "message"),
    [
        ("identity_conflicts", "identity conflict"),
        ("identity_warnings", "identity ambiguity"),
    ],
)
def test_activation_rejects_selected_pane_unresolved_identity(
    tmp_path, evidence_field, message
):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    selected = _selected_pane_payload(
        candidate,
        **{
            evidence_field: [{
                "pane_id": candidate.pane_id,
                "workspace_handle": candidate.handle,
                "endpoint_handle": "conflicting-owner",
                "reason": "endpoint_workspace_identity_conflict",
            }],
        },
    )

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )

    with pytest.raises(DiaulosActivationError, match=message):
        client.activate(candidate)

    assert len(calls) == 1
    assert calls[0][-3:] == ["--pane-id", "10", "--json"]


def test_activation_rejects_selected_pane_without_diaulos_id(tmp_path):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    selected = _selected_pane_payload(candidate, entry={"diaulos_id": ""})

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )

    with pytest.raises(DiaulosActivationError, match="Diaulos ID"):
        client.activate(candidate)

    assert len(calls) == 1
    assert calls[0][-3:] == ["--pane-id", "10", "--json"]


def test_activation_reports_selected_and_current_routes_from_exclusion(tmp_path):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    selected = _selected_pane_payload(
        candidate,
        entries=[],
        excluded=[{
            "handle": candidate.handle,
            "pane_id": candidate.pane_id,
            "reason": "route_identity_mismatch",
            "mismatched_fields": ["tty"],
            "selected_route": {
                "handle": candidate.handle,
                "diaulos_id": candidate.diaulos_id,
                "tty": candidate.tty,
                "resume_backend": candidate.resume_backend,
                "thread_id": candidate.thread_id,
            },
            "current_route": {
                "handle": "beaming-baby-cloud-milk",
                "diaulos_id": "dia-beaming",
                "tty": "/dev/ttys037",
                "resume_backend": "codex",
                "thread_id": "different-thread",
            },
        }],
    )

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), "")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )

    with pytest.raises(DiaulosActivationError) as error:
        client.activate(candidate)

    message = str(error.value)
    assert candidate.handle in message
    assert candidate.tty in message
    assert candidate.thread_id in message
    assert "beaming-baby-cloud-milk" in message
    assert "/dev/ttys037" in message
    assert "different-thread" in message
    assert len(calls) == 1


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

    assert len(observed_environments) == 2
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
        ("pane_id", 11, "pane_id"),
        ("tab_id", 999, "tab_id"),
        ("window_id", 999, "window_id"),
        ("cwd", "file:///tmp/recycled", "cwd"),
        ("tty", "/dev/ttys999", "tty"),
        ("handle", "replacement", "handle"),
        ("diaulos_id", "dia-replacement", "diaulos_id"),
        ("resume_backend", "claude", "resume_backend"),
        ("thread_id", "replacement-thread", "thread_id"),
    ],
)
def test_activation_refuses_every_changed_selected_route_field(
    tmp_path,
    field,
    value,
    message,
):
    calls: list[list[str]] = []
    candidate = parse_live_inventory(_payload(1))[0]
    selected = _selected_pane_payload(candidate, entry={field: value})

    def runner(command, **kwargs):
        calls.append(command)
        if "diaulos" in command and "live" in command:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(selected), ""
            )
        raise AssertionError("Changed route must not reach activation")

    client = EpistaxisDiaulosClient(
        runner=runner,
        snapshot_path=tmp_path / "unused.json",
        epistaxis_executable="epistaxis",
        wezterm_executable="wezterm",
    )
    candidate = parse_live_inventory(_payload(1))[0]

    with pytest.raises(DiaulosActivationError, match=message):
        client.activate(candidate)

    assert len(calls) == 1
    assert calls[0][-3:] == ["--pane-id", "10", "--json"]


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


def test_activation_commit_remains_single_flight_after_dismissal(overlay_module):
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
        assert overlay.hide() is True
        assert overlay.visible is False
        overlay.visible = True
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


def test_replacement_detaches_activation_without_stranding_latch(overlay_module):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(overlay_module.DiaulosSwitcherOverlay)
    overlay.visible = True
    overlay._activation_in_flight = True
    overlay._activation_handle = "target"
    overlay._activation_generation = 7
    overlay._panel = MagicMock()
    overlay._previous_app = None
    overlay._remove_key_monitor = MagicMock()
    overlay._finish_shell = MagicMock()
    overlay._search_field = MagicMock()
    overlay._activate_wezterm = MagicMock()
    overlay._apply_inventory_payload = MagicMock()
    assert overlay.hide(restore_previous=False, force=True)
    # Reopening the same panel does not give the old completion focus authority.
    overlay.visible = True
    overlay.activationFinished_({"generation": 7, "receipt": {}})
    assert not overlay._activation_in_flight
    assert overlay.visible
    overlay._activate_wezterm.assert_not_called()


def test_activation_completion_reports_main_dispatch_delay(overlay_module, monkeypatch, caplog):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(overlay_module.DiaulosSwitcherOverlay)
    overlay._activation_generation = 7
    overlay._activation_handle = "target"
    overlay._search_field = MagicMock()
    overlay._activation_detached = True
    overlay.visible = False
    monkeypatch.setattr(overlay_module.time, "monotonic", lambda: 15.0)
    with caplog.at_level("INFO", logger="spoke.diaulos_switcher_overlay"):
        overlay.activationFinished_({"generation": 7, "receipt": {},
                                     "selected_at": 10.0, "worker_finished_at": 12.0})
    assert "dispatch_ms=3000.0" in caplog.text
    assert "selection_to_callback_ms=5000.0" in caplog.text




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


@pytest.mark.parametrize(
    ("app_active", "panel_key", "activation_in_flight", "expected"),
    [
        (True, True, False, True),
        (False, True, False, False),
        (True, False, False, False),
        (False, False, True, True),
    ],
)
def test_switcher_accepts_dictation_only_during_active_interaction(
    overlay_module, monkeypatch, app_active, panel_key, activation_in_flight, expected
):
    overlay = overlay_module.DiaulosSwitcherOverlay.__new__(
        overlay_module.DiaulosSwitcherOverlay
    )
    overlay.visible = True
    overlay._activation_in_flight = activation_in_flight
    overlay._panel = MagicMock()
    overlay._panel.isKeyWindow.return_value = panel_key
    app = MagicMock()
    app.isActive.return_value = app_active
    monkeypatch.setattr(overlay_module, "NSApp", lambda: app)

    assert overlay.accepts_dictation() is expected


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


def _animated_switcher(module):
    from spoke.optical_lifecycle import HouseTransition
    from spoke.house_optical_primitive import compile_house_optical_shell_config

    overlay = module.DiaulosSwitcherOverlay.__new__(module.DiaulosSwitcherOverlay)
    overlay.visible = True
    overlay.presentation_generation = 1
    overlay._panel = MagicMock()
    overlay._previous_app = None
    overlay._activation_in_flight = False
    overlay._activation_generation = 0
    overlay._key_monitor_token = None
    overlay._shell_host = MagicMock()
    overlay._shell_registered = True
    overlay._shell_transition = HouseTransition()
    overlay._shell_transition.rest()
    overlay._shell_final = dict(compile_house_optical_shell_config(720, 560), client_id="spoke.teleporter")
    overlay._shell_epoch = 1
    overlay._shell_waiting = False
    overlay._shell_tick_timer = None
    overlay._shell_sidecars_registered = True
    overlay._set_shell_planes = MagicMock()
    return overlay


def test_animated_hide_releases_input_immediately_but_keeps_house_tail(overlay_module):
    overlay = _animated_switcher(overlay_module)
    host = overlay._shell_host
    overlay.hide()
    assert overlay.visible is False
    assert overlay.accepts_dictation() is False
    host.release_client.assert_not_called()
    overlay._panel.orderOut_.assert_not_called()
    assert overlay._shell_transition.phase == "dismiss"
    overlay._panel.setIgnoresMouseEvents_.assert_called_with(True)


def test_force_cleanup_releases_all_and_does_not_leave_tail_timer(overlay_module):
    overlay = _animated_switcher(overlay_module)
    host = overlay._shell_host
    timer = MagicMock()
    overlay._shell_tick_timer = timer
    overlay.hide(force=True)
    assert {call.args[0] for call in host.release_client.call_args_list} == {
        "spoke.teleporter", "spoke.teleporter.dismiss_seam", "spoke.teleporter.dismiss_radial_pucker",
    }
    timer.invalidate.assert_called_once()
    overlay._panel.orderOut_.assert_called_once()


def test_stale_first_present_cannot_rewind_or_resurrect_panel(overlay_module):
    overlay = _animated_switcher(overlay_module)
    assert callable(getattr(type(overlay), "opticalShellDidPresent_", None)), "Teleporter first-present admission missing"
    overlay._shell_waiting = True
    overlay.opticalShellDidPresent_({"epoch": 0, "host": overlay._shell_host})
    assert overlay._shell_waiting is True
    overlay.visible = False
    overlay.opticalShellDidPresent_({"epoch": 1, "host": overlay._shell_host})
    assert overlay._shell_waiting is True


def test_native_fallback_is_not_reanimated_by_late_first_present(overlay_module):
    overlay = _animated_switcher(overlay_module)
    assert callable(getattr(type(overlay), "opticalShellDidPresent_", None)), "Teleporter first-present admission missing"
    overlay._shell_waiting = False
    overlay.opticalShellDidPresent_({"epoch": 1, "host": overlay._shell_host})
    assert overlay._shell_transition.phase == "rest"
    overlay._set_shell_planes.assert_not_called()


@pytest.mark.parametrize("last_progress,hide_progress", [
    (0.0, .1), (.5, .9), (.8, .85),
    ("before_release", "after_release"), ("before_release", 1.001),
    (0.0, 2.0), (.99, 1.2),
])
def test_hide_between_ticks_preserves_last_applied_foreground(overlay_module, monkeypatch, last_progress, hide_progress):
    from spoke.house_optical_primitive import optical_text_release_progress, OPTICAL_MATERIALIZATION_S

    overlay = _animated_switcher(overlay_module)
    overlay._shell_transition.summon(10)
    release = optical_text_release_progress()
    last_progress = release - .005 if last_progress == "before_release" else last_progress
    hide_progress = release + .005 if hide_progress == "after_release" else hide_progress
    last_tick = 10 + last_progress * OPTICAL_MATERIALIZATION_S
    hide_time = 10 + hide_progress * OPTICAL_MATERIALIZATION_S
    overlay._publish_shell_frame(last_tick)
    before = overlay._set_shell_planes.call_args.args[1]
    monkeypatch.setattr(overlay_module.time, "perf_counter", lambda: hide_time)
    overlay.hide()
    timer = overlay._shell_tick_timer
    for interval in (1 / 60, 2 / 60, 3 / 60):
        monkeypatch.setattr(overlay_module.time, "perf_counter", lambda: hide_time + interval)
        overlay.opticalShellStep_(timer)
        after = overlay._set_shell_planes.call_args.args[1]
        assert after["alpha"] <= before["alpha"]
        assert after["height_frac"] <= before["height_frac"]
        assert after["width_frac"] <= before["width_frac"]


def test_rejected_publication_does_not_advance_applied_foreground(overlay_module):
    overlay = _animated_switcher(overlay_module)
    overlay._shell_transition.summon(10)
    overlay._publish_shell_frame(10)
    before = dict(overlay._shell_applied_text)
    overlay._shell_host.update_client_configs.return_value = False
    with pytest.raises(RuntimeError, match="publication rejected"):
        overlay._publish_shell_frame(11)
    assert overlay._shell_applied_text == before


def test_pending_capture_hide_releases_clients_without_an_invisible_tail(overlay_module):
    overlay = _animated_switcher(overlay_module)
    host = overlay._shell_host
    overlay._shell_waiting = True
    timer = MagicMock()
    overlay._shell_tick_timer = timer
    overlay.hide()
    assert overlay.visible is False
    assert overlay._shell_transition is None
    assert overlay._shell_tick_timer is None
    timer.invalidate.assert_called_once()
    assert host.release_client.call_count == 3
    overlay._panel.orderOut_.assert_called_once()


def test_reopen_during_close_rejects_the_old_timer(overlay_module, monkeypatch):
    overlay = _animated_switcher(overlay_module)
    overlay._delegate = MagicMock()
    # First hide creates a close timer; prepare replaces it on reopen.
    old_timer, new_timer = MagicMock(), MagicMock()
    monkeypatch.setattr(overlay_module.NSTimer,
        "scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_",
        MagicMock(side_effect=[old_timer, new_timer]))
    monkeypatch.setattr(overlay_module.time, "perf_counter", lambda: 10.0)
    overlay.hide()
    overlay.visible = True
    monkeypatch.setattr(overlay_module.time, "perf_counter", lambda: 10.05)
    overlay._prepare_shell_entrance()
    calls = overlay._set_shell_planes.call_count
    host = overlay._shell_host
    overlay.opticalShellStep_(old_timer)
    assert overlay._shell_tick_timer is new_timer
    assert overlay._set_shell_planes.call_count == calls
    host.release_client.assert_not_called()
    assert overlay._shell_transition.phase == "summon"
    assert overlay._shell_waiting is False
    frame = overlay._publish_shell_frame(10.05)
    assert frame.text["alpha"] == 0


def test_start_grace_applies_native_rest_without_late_reanimation(overlay_module, monkeypatch):
    overlay = _animated_switcher(overlay_module)
    overlay._shell_transition.summon(10)
    overlay._shell_waiting = True
    overlay._shell_wait_started = 10
    overlay._shell_tick_timer = timer = MagicMock()
    monkeypatch.setattr(overlay_module.time, "perf_counter", lambda: 10.26)
    overlay.opticalShellStep_(timer)
    assert overlay._shell_applied_text["alpha"] == 1
    assert overlay._shell_transition.phase == "rest"
    assert overlay._shell_tick_timer is None
    calls = overlay._set_shell_planes.call_count
    overlay.opticalShellDidPresent_({"epoch": 1, "host": overlay._shell_host})
    assert overlay._set_shell_planes.call_count == calls


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
