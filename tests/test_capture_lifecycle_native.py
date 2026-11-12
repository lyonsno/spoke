import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="installed Cocoa bridge conformance")
@pytest.mark.parametrize("matching_source", [False, True])
def test_native_capture_completion_and_source_identity(tmp_path, matching_source):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(b"".join((root / p).read_bytes()
        for p in ("spoke/__main__.py", "spoke/capture.py", "spoke/handsfree.py",
                  "spoke/glow.py"))).hexdigest()
    out = tmp_path / "capture-report.json"
    result = subprocess.run([sys.executable, str(root / "tests/native_capture_lifecycle_witness.py"),
        "--repo-root", str(root), "--expected-digest", digest if matching_source else "wrong",
        "--out", str(out)], capture_output=True, text=True)
    report = json.loads(out.read_text())
    assert report["passed"] == matching_source, result.stdout + result.stderr + str(report)
    assert result.returncode == (0 if matching_source else 1)
    assert report["source_digest"] == digest
    if matching_source:
        assert report["phase"] == "complete"
        assert report["pending_and_completion_passed"]
        assert report["capture_admitted_before_graphics"]
        assert report["stale_completion_ignored"]
        assert report["handsfree_completion_passed"]
        assert report["installed_pill_timer_clock_passed"]
        assert report["persistent_glow_release_and_retry_passed"]
        assert len(report["retained_fixture_audio"]) == 2
    else:
        assert report["phase"] == "source"
        assert "digest mismatch" in report["error"]
