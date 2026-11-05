import plistlib
import subprocess
import threading
from pathlib import Path

import pytest


def observation(values=(100, 100), stamps=(9.5, 10)):
    return {"status": "observed", "samples": [
        {"status": "observed", "monotonic": stamp, "device_utilization_percent": value}
        for stamp, value in zip(stamps, values)]}


@pytest.mark.parametrize("values,expected", [((100, 100), True), ((95, 95), True),
    ((94, 100), False), ((100, 0), False), ((0, 0), False)])
def test_sustained_saturation_decision(values, expected):
    from spoke.gpu_routing import cpu_route_decision
    assert cpu_route_decision(observation(values), now=10)["cpu_selected"] is expected


@pytest.mark.parametrize("bad", [None, {}, {"status": "unavailable"},
    observation(stamps=(0, 1)), observation(stamps=(11, 12)),
    observation(stamps=(10, 10)), observation(values=(True, 100)),
    observation(values=(float("nan"), 100)), observation(values=(100, 101)),
    {"status": "observed", "samples": [{"monotonic": 10, "device_utilization_percent": 100}]},
    {"status": "observed", "samples": [{"status": "unavailable"}, {"status": "unavailable"}]}])
def test_unverified_or_stale_load_does_not_divert(bad):
    from spoke.gpu_routing import cpu_route_decision
    assert cpu_route_decision(bad, now=10)["cpu_selected"] is False


def test_parser_traverses_devices_and_preserves_unknown_fields():
    from spoke.gpu_routing import parse_gpu_load
    data = [{"IORegistryEntryChildren": [
        {"IORegistryEntryID": 1, "PerformanceStatistics": {"Device Utilization %": 10}},
        {"IORegistryEntryID": 2, "PerformanceStatistics": {"Device Utilization %": 100, "new field": 7}}]}]
    parsed = parse_gpu_load(plistlib.dumps(data))
    assert parsed["device_utilization_percent"] == 100
    assert len(parsed["devices"]) == 2
    assert parsed["devices"][1]["statistics"]["new field"] == 7


def test_observed_m4_max_ioreg_contract():
    from spoke.gpu_routing import parse_gpu_load
    # Projection of the unmodified statistics/name from native-gpu.plist,
    # ioreg -a -r -c IOAccelerator -d 1, macOS15.6/M4 Max, Nov5 00:35UTC.
    raw = (Path(__file__).parent / "fixtures/gpu-load-observed_2026-11-05.plist").read_bytes()
    parsed = parse_gpu_load(raw)
    assert parsed["device_utilization_percent"] == 0
    assert parsed["devices"][0]["name"] == "AGXAcceleratorG16X"
    assert parsed["devices"][0]["statistics"]["recoveryCount"] == 10


@pytest.mark.parametrize("data", [[], [{"PerformanceStatistics": {}}],
    [{"PerformanceStatistics": {"Device Utilization %": "100"}}],
    [{"PerformanceStatistics": {"Device Utilization %": True}}]])
def test_missing_or_retyped_counters_rejected(data):
    from spoke.gpu_routing import parse_gpu_load
    with pytest.raises(ValueError):
        parse_gpu_load(plistlib.dumps(data))


def test_slow_diagnostic_cannot_block_release_or_publish_after_finish(monkeypatch):
    from spoke import gpu_routing as module
    entered, release = threading.Event(), threading.Event()
    def read():
        entered.set()
        release.wait()
        return {"status": "observed", "device_utilization_percent": 100}
    monkeypatch.setattr(module, "read_gpu_load", read)
    monitor = module.CaptureGpuLoad()
    monitor.start()
    try:
        assert entered.wait(2)
        assert monitor.finish() == {"status": "unavailable", "samples": []}
    finally:
        release.set()
    assert monitor.finish()["samples"] == []


def test_diagnostic_failure_lowers_authority(monkeypatch):
    from spoke import gpu_routing as module
    monkeypatch.setattr(module.sys, "platform", "darwin")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(module.subprocess, "run", timeout)
    assert module.read_gpu_load()["status"] == "unavailable"

