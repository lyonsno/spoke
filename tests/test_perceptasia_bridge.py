"""Tests for the Perceptasia shared selection bridge."""

from __future__ import annotations

import json


def test_load_selection_reads_current_perceptasia_payload_shape(tmp_path):
    from spoke.perceptasia_bridge import load_perceptasia_selection

    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "selected": "topos:codex-perceptasia-selection-spinaltap-bridge-0515",
                "kind": "topos",
                "name": "codex-perceptasia-selection-spinaltap-bridge-0515",
                "lifecycle": "active",
                "project": "spoke",
                "neighbors": [
                    "attractor:support-perceptasia-spoke-coordination-bridge",
                    "metadosis:perceptasia-spoke-coordination-bridge_2026-05-15",
                ],
                "filter_state": {"kinds_visible": ["topos", "attractor"]},
                "timestamp": "2026-05-15T12:34:56Z",
            }
        )
    )

    selection = load_perceptasia_selection(path)

    assert selection.available is True
    assert selection.active is True
    assert selection.selected == "topos:codex-perceptasia-selection-spinaltap-bridge-0515"
    assert selection.kind == "topos"
    assert selection.project == "spoke"
    assert selection.neighbors == (
        "attractor:support-perceptasia-spoke-coordination-bridge",
        "metadosis:perceptasia-spoke-coordination-bridge_2026-05-15",
    )
    assert selection.status_line == (
        "Perceptasia: topos codex-perceptasia-selection-spinaltap-bridge-0515"
    )
    assert "visual selection" in selection.prompt_context
    assert (
        "selected: topos:codex-perceptasia-selection-spinaltap-bridge-0515"
        in selection.prompt_context
    )
    assert (
        "neighbor: metadosis:perceptasia-spoke-coordination-bridge_2026-05-15"
        in selection.prompt_context
    )


def test_load_selection_accepts_metadosis_focused_project_alias(tmp_path):
    from spoke.perceptasia_bridge import load_perceptasia_selection

    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "selected": "attractor:support-visual-to-agent-routing",
                "focused_project": "epistaxis",
                "neighbors": [],
                "timestamp": "2026-05-15T12:35:00Z",
            }
        )
    )

    selection = load_perceptasia_selection(path)

    assert selection.active is True
    assert selection.project == "epistaxis"
    assert selection.kind == "attractor"
    assert selection.status_line == "Perceptasia: attractor support-visual-to-agent-routing"


def test_missing_selection_file_is_explicit_absent_state(tmp_path):
    from spoke.perceptasia_bridge import load_perceptasia_selection

    selection = load_perceptasia_selection(tmp_path / "missing-selection.json")

    assert selection.available is False
    assert selection.active is False
    assert selection.status_line == "Perceptasia: no selection file"
    assert selection.prompt_context == ""


def test_cleared_selection_file_is_explicit_no_visual_selection(tmp_path):
    from spoke.perceptasia_bridge import load_perceptasia_selection

    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected": None, "timestamp": "2026-05-15T12:36:00Z"}))

    selection = load_perceptasia_selection(path)

    assert selection.available is True
    assert selection.active is False
    assert selection.status_line == "Perceptasia: no visual selection"
    assert selection.prompt_context == ""
