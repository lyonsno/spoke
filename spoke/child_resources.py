"""Per-command diagnostics, separate from process-global child accounting."""

import math
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time


def parse_resources(raw):
    result = {"status": "unavailable", "raw": raw, "source": "darwin-time-l"}
    number = r"(?:\d+(?:\.\d+)?)"
    match = re.search(rf"^\s*({number}) real\s+({number}) user\s+({number}) sys\s*$", raw, re.MULTILINE)
    if match:
        for key, value in zip(("wall_seconds", "user_seconds", "system_seconds"), match.groups()):
            result[key] = float(value)
    fields = {
        "maximum resident set size": "max_rss_bytes",
        "page reclaims": "page_reclaims",
        "page faults": "page_faults",
        "swaps": "swaps",
        "voluntary context switches": "voluntary_context_switches",
        "involuntary context switches": "involuntary_context_switches",
    }
    for label, key in fields.items():
        match = re.search(r"^\s*(\d+)\s+" + re.escape(label) + r"\s*$", raw, re.MULTILINE)
        if match:
            result[key] = int(match[1])
    required = {"wall_seconds", "user_seconds", "system_seconds", *fields.values()}
    missing = sorted(required - result.keys())
    result["missing_fields"] = missing
    if not missing and all(math.isfinite(result[key]) and result[key] >= 0 for key in required):
        result["status"] = "complete"
    elif len(result) > 4:
        result["status"] = "partial"
    return result


def _read_power():
    started = time.monotonic()
    try:
        proc = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True)
        return {"source": "pmset-g-batt", "status": "observed" if proc.returncode == 0 else "unavailable",
                "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "collection_seconds": time.monotonic() - started, "observed_at_epoch": time.time()}
    except OSError as exc:
        return {"source": "pmset-g-batt", "status": "unavailable", "error": str(exc)}


class PowerSampler:
    """One in-flight observation; callers receive only timestamped cached data."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {"source": "pmset-g-batt", "status": "unavailable", "reason": "not_observed"}
        self.refreshing = False

    def snapshot(self):
        with self.lock:
            result = dict(self.latest)
            start = not self.refreshing
            if start:
                self.refreshing = True
        observed = result.get("observed_at_epoch")
        result["age_seconds"] = max(0, time.time() - observed) if observed is not None else None
        result["delivery"] = "cached_observation"
        if start:
            try:
                threading.Thread(target=self._refresh, name="spoke-power-observation", daemon=True).start()
            except Exception as exc:
                with self.lock:
                    self.refreshing = False
                result["refresh_error"] = type(exc).__name__
        return result

    def _refresh(self):
        try:
            value = _read_power()
        except Exception as exc:
            value = {"status": "unavailable", "error_type": type(exc).__name__}
        with self.lock:
            self.latest = value
            self.refreshing = False


_power_sampler = PowerSampler()


def power_snapshot():
    return _power_sampler.snapshot()


@functools.lru_cache(maxsize=1)
def _time_available():
    try:
        probe = subprocess.run(["/usr/bin/time", "-l", "/usr/bin/true"], capture_output=True)
        return probe.returncode == 0
    except OSError:
        return False


def run_measured(argv, **kwargs):
    # Preserve subprocess.run's timeout ownership: timing a wrapper would leave
    # its native child alive if the wrapper alone were killed on timeout.
    if sys.platform != "darwin" or kwargs.get("timeout") is not None or not _time_available():
        result = subprocess.run(argv, **kwargs)
        result.resource_receipt = {"resources": {"status": "unavailable", "reason": "unsupported-or-explicit-timeout"}}
        return result
    before = power_snapshot()
    try:
        temporary = tempfile.TemporaryDirectory(prefix="spoke-child-resources-", ignore_cleanup_errors=True)
    except OSError:
        result = subprocess.run(argv, **kwargs)
        result.resource_receipt = {"resources": {"status": "unavailable", "reason": "temporary_storage"}}
        return result
    with temporary as directory:
        output = Path(directory) / "resources.txt"
        result = subprocess.run(["/usr/bin/time", "-l", "-o", str(output), *argv], **kwargs)
        try:
            resources = parse_resources(output.read_text())
        except Exception as exc:
            resources = {"status": "unavailable", "error_type": type(exc).__name__}
        result.resource_receipt = {
            "schema": "spoke.child-resources.v1", "native_executable": argv[0],
            "argv_count": len(argv), "argv_sha256": hashlib.sha256(json.dumps(list(argv)).encode()).hexdigest(),
            "effective_wrapper": "/usr/bin/time -l -o", "resources": resources,
            "power_before": before, "power_after": power_snapshot(),
            "limits": "Child resource totals and timestamped cached battery observations; CPU seconds sum threads. No CPU frequency, thermal state, or GPU isolation claim",
        }
        result.args = list(argv)
        return result
