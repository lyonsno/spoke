from __future__ import annotations

from spoke.seam_pucker_hud import _SLIDER_SPECS


def test_seam_pucker_hud_calls_the_diagnostic_control_transition_phase():
    label, key, min_value, max_value, fmt = _SLIDER_SPECS[0]

    assert label == "Transition Phase"
    assert key == "preview_progress"
    assert min_value == 0.0
    assert max_value == 0.42
    assert fmt == "{:.3f}"
