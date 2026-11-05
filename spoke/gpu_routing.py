"""Off-path GPU observation and request-local ASR selection policy."""

import copy
import math
import plistlib
import subprocess
import sys
import threading
import time

SATURATION_PERCENT = 95
FRESH_SECONDS = 2.0
SAMPLE_INTERVAL = 0.5


def parse_gpu_load(raw):
    devices = []

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            stats = node.get("PerformanceStatistics")
            if isinstance(stats, dict) and "Device Utilization %" in stats:
                value = stats["Device Utilization %"]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 100):
                    raise ValueError("Invalid GPU device utilization")
                devices.append({"registry_id": node.get("IORegistryEntryID"),
                                "name": node.get("IORegistryEntryName"),
                                "device_utilization_percent": value,
                                "statistics": stats})
            visit(node.get("IORegistryEntryChildren", []))

    visit(plistlib.loads(raw))
    if not devices:
        raise ValueError("No GPU device-utilization counters in ioreg response")
    return {"status": "observed", "source": "ioreg-IOAccelerator",
            "device_utilization_percent": max(d["device_utilization_percent"] for d in devices),
            "devices": devices}


def read_gpu_load():
    if sys.platform != "darwin":
        return {"status": "unavailable", "reason": "unsupported_platform"}
    try:
        # This deadline bounds a disposable diagnostic child, never ASR or audio.
        result = subprocess.run(["/usr/sbin/ioreg", "-a", "-r", "-c", "IOAccelerator", "-d", "1"],
                                capture_output=True, timeout=1, check=False)
        if result.returncode:
            raise RuntimeError(f"ioreg exited {result.returncode}")
        return parse_gpu_load(result.stdout)
    except Exception as exc:
        return {"status": "unavailable", "reason": type(exc).__name__, "error": str(exc)}


class CaptureGpuLoad:
    """Only collect during one hold; release never waits for the diagnostic."""

    def __init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples = []

    def start(self):
        try:
            threading.Thread(target=self._run, name="spoke-capture-gpu-load", daemon=True).start()
        except RuntimeError:
            self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                sample = read_gpu_load()
            except Exception as exc:
                sample = {"status": "unavailable", "reason": type(exc).__name__}
            sample.update(monotonic=began, observed_at_epoch=time.time())
            with self._lock:
                if not self._stop.is_set():
                    self._samples.append(sample)
            if self._stop.wait(SAMPLE_INTERVAL):
                break

    def finish(self):
        self._stop.set()
        with self._lock:
            return {"status": "observed" if self._samples else "unavailable",
                    "samples": copy.deepcopy(self._samples)}


def cpu_route_decision(observation, *, now=None):
    now = time.monotonic() if now is None else now
    receipt = {"schema": "spoke.automatic-asr-routing.v1", "reason": "gpu_unverified",
               "cpu_selected": False, "threshold_percent": SATURATION_PERCENT,
               "fresh_seconds": FRESH_SECONDS, "gpu_observation": observation}
    if not isinstance(observation, dict) or observation.get("status") != "observed":
        return receipt
    samples = observation.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return receipt
    previous, latest = samples[-2:]
    for sample in (previous, latest):
        if not isinstance(sample, dict) or sample.get("status", "observed") != "observed":
            return receipt
        stamp, value = sample.get("monotonic"), sample.get("device_utilization_percent")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (stamp, value)) or not 0 <= value <= 100:
            return receipt
        if not 0 <= now - stamp <= FRESH_SECONDS:
            receipt["reason"] = "gpu_observation_stale"
            return receipt
    if latest["monotonic"] - previous["monotonic"] < SAMPLE_INTERVAL / 2:
        return receipt
    receipt["reason"] = "gpu_not_saturated"
    if all(sample["device_utilization_percent"] >= SATURATION_PERCENT for sample in (previous, latest)):
        receipt.update(reason="gpu_saturated", cpu_selected=True)
    return receipt
