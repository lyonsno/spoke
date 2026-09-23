"""Selected diagnostic settings must reach the isolated NeMo child."""

import json

import pytest

from spoke import launch_targets, transcribe_nemotron


@pytest.mark.parametrize("inherited_timing", [None, "0"])
def test_selected_timing_reaches_nemotron_child(tmp_path, monkeypatch, inherited_timing):
    checkout = tmp_path / "diagnostic-release"
    checkout.mkdir()
    registry = tmp_path / "launch_targets.json"
    registry.write_text(json.dumps({
        "selected": "nemotron_cpu_full_buffer",
        "targets": [{
            "id": "nemotron_cpu_full_buffer",
            "path": str(checkout),
            "env": {"SPOKE_NEMOTRON_PHASE_TIMING": "1"},
        }],
    }))
    monkeypatch.setenv("SPOKE_LAUNCH_TARGET_ID", "nemotron_cpu_full_buffer")
    monkeypatch.setenv("SPOKE_LAUNCH_TARGETS_PATH", str(registry))
    monkeypatch.setenv("NEMO_SPEECH_TIMING", "0")
    monkeypatch.setenv("NEMO_SPEECH_UNRELATED", "inherited")
    # Track the key before production code mutates os.environ directly.
    monkeypatch.setenv("SPOKE_NEMOTRON_PHASE_TIMING", "0")
    if inherited_timing is None:
        monkeypatch.delenv("SPOKE_NEMOTRON_PHASE_TIMING", raising=False)
    else:
        monkeypatch.setenv("SPOKE_NEMOTRON_PHASE_TIMING", inherited_timing)

    assert transcribe_nemotron._phase_timing_enabled() is False
    reconcile = getattr(launch_targets, "apply_selected_launch_target_env", None)
    assert callable(reconcile), "managed startup must restore selected diagnostic env"
    receipt = reconcile(checkout)
    assert receipt["status"] == "repaired"
    assert "SPOKE_NEMOTRON_PHASE_TIMING" in receipt["repaired_env_keys"]
    assert transcribe_nemotron._phase_timing_enabled() is True

    child_env, removed = transcribe_nemotron._controlled_child_environment(
        phase_timing=transcribe_nemotron._phase_timing_enabled(),
    )
    assert child_env["NEMO_SPEECH_TIMING"] == "1"
    assert "NEMO_SPEECH_UNRELATED" not in child_env
    assert {"NEMO_SPEECH_TIMING", "NEMO_SPEECH_UNRELATED"} <= set(removed)
