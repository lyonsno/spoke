import subprocess
from pathlib import Path

import pytest


def test_observed_darwin_time_fixture_preserves_resource_units():
    from spoke.child_resources import parse_resources
    observed = Path(__file__).parent / "fixtures/nemotron/darwin-time-true.txt"
    result = parse_resources(observed.read_text())
    assert result["status"] == "complete"
    assert result["user_seconds"] == 0
    assert result["max_rss_bytes"] == 1327104
    assert result["page_faults"] == 0
    assert result["involuntary_context_switches"] == 31


@pytest.mark.parametrize("raw", ["", "0.00 real 0.00 user 0.00 sys\n", "nonsense", "1..2 real 0.00 user 0.00 sys"])
def test_partial_resources_are_not_complete(raw):
    from spoke.child_resources import parse_resources
    assert parse_resources(raw)["status"] != "complete"


def test_malformed_timing_with_other_fields_still_degrades():
    from spoke.child_resources import parse_resources
    raw = (Path(__file__).parent / "fixtures/nemotron/darwin-time-true.txt").read_text()
    assert parse_resources(raw.replace("0.00 real", "1..2 real"))["status"] == "partial"


def test_measured_run_keeps_output_and_exact_child_identity(tmp_path, monkeypatch):
    from spoke import child_resources as module
    fixture = (Path(__file__).parent / "fixtures/nemotron/darwin-time-true.txt").read_text()
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        if argv[:2] == ["/usr/bin/pmset", "-g"]:
            return subprocess.CompletedProcess(argv, 0, stdout="Battery Power 67%", stderr="")
        assert argv[:3] == ["/usr/bin/time", "-l", "-o"]
        Path(argv[3]).write_text(fixture)
        return subprocess.CompletedProcess(argv, 0, stdout=b"private transcript", stderr=b"native phases")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_time_available", lambda: True)
    monkeypatch.setattr(module, "power_snapshot", lambda: {"stdout": "Battery Power 67%"})
    result = module.run_measured(["native", "input.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert seen[0][4:] == ["native", "input.wav"]
    assert result.stdout == b"private transcript"
    assert result.stderr == b"native phases"
    assert result.resource_receipt["resources"]["status"] == "complete"
    assert result.resource_receipt["power_before"]["stdout"] == "Battery Power 67%"


def test_missing_resource_output_lowers_authority(tmp_path, monkeypatch):
    from spoke import child_resources as module
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout=b"ok", stderr=b""))
    monkeypatch.setattr(module, "_time_available", lambda: True)
    monkeypatch.setattr(module, "power_snapshot", lambda: {"status": "unavailable"})
    result = module.run_measured(["native"])
    assert result.resource_receipt["resources"]["status"] == "unavailable"


def test_explicit_timeout_keeps_direct_child_route(monkeypatch):
    from spoke import child_resources as module
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        module.run_measured(["native"], timeout=1)
    assert calls == [["native"]]


def test_native_resource_route_observed_without_inference():
    import sys
    from spoke.child_resources import run_measured
    if sys.platform != "darwin":
        pytest.skip("Darwin installed route")
    result = run_measured(["/usr/bin/true"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0
    from spoke.child_resources import _time_available
    if _time_available():
        assert result.resource_receipt["resources"]["status"] == "complete"
        assert result.resource_receipt["resources"]["max_rss_bytes"] > 0
        assert result.resource_receipt["native_executable"] == "/usr/bin/true"
    else:
        assert result.resource_receipt["resources"]["status"] == "unavailable"


def test_blocked_power_observation_does_not_block_snapshot(monkeypatch):
    import threading
    from spoke import child_resources as module
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    def run(argv, **kwargs):
        entered.set()
        release.wait()
        return subprocess.CompletedProcess(argv, 0, stdout="Battery Power", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_power_sampler", module.PowerSampler())
    caller = threading.Thread(target=lambda: (module.power_snapshot(), returned.set()))
    caller.start()
    try:
        assert entered.wait(2)
        assert returned.wait(0.1)
    finally:
        release.set()
        caller.join()


def test_cached_power_exposes_age_and_does_not_pretend_current(monkeypatch):
    from spoke import child_resources as module
    sampler = module.PowerSampler()
    sampler.latest = {"status": "observed", "observed_at_epoch": 100, "stdout": "Battery Power"}
    sampler.refreshing = True
    monkeypatch.setattr(module.time, "time", lambda: 109)
    result = sampler.snapshot()
    assert result["delivery"] == "cached_observation"
    assert result["age_seconds"] == 9


def test_denied_wrapper_does_not_prevent_native_execution(monkeypatch):
    from spoke import child_resources as module
    calls = []
    monkeypatch.setattr(module, "_time_available", lambda: False)
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"valid result", stderr=b"")
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.run_measured(["native"])
    assert calls == [["native"]]
    assert result.stdout == b"valid result"
    assert result.resource_receipt["resources"]["status"] == "unavailable"
