import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="installed Cocoa bridge conformance")
@pytest.mark.parametrize("matching_source,suppress_history_cleanup", [
    (False, False), (True, False), (True, True),
], ids=["wrong-source", "production", "history-cleanup-omitted"])
def test_native_capture_completion_and_source_identity(tmp_path, matching_source,
                                                       suppress_history_cleanup):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(b"".join((root / p).read_bytes()
        for p in ("spoke/__main__.py", "spoke/capture.py", "spoke/handsfree.py",
                  "spoke/glow.py"))).hexdigest()
    out = tmp_path / "capture-report.json"
    argv = [str(root / "tests/native_capture_lifecycle_witness.py"),
        "--repo-root", str(root), "--expected-digest", digest if matching_source else "wrong",
        "--out", str(out)]
    command = [sys.executable, *argv]
    if suppress_history_cleanup:
        command = [sys.executable, "-c", """
import os, runpy, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[sys.argv.index("--repo-root") + 1])
os.environ["SPOKE_VAD_ENABLED"] = "0"
os.environ["SPOKE_MODEL_PREFERENCES_PATH"] = str(
    Path(sys.argv[sys.argv.index("--out") + 1]).parent / "preferences.json")
os.environ.pop("SPOKE_LAUNCH_TARGET_ID", None)
from spoke import __main__ as mod, glow
original = mod.SpokeAppDelegate._cancel_transcription_for_recovery
def omit_history_cleanup(self):
    with patch.object(glow.GlowOverlay, "hide", lambda self: None):
        return original(self)
with patch.object(mod.SpokeAppDelegate, "_cancel_transcription_for_recovery",
                  omit_history_cleanup):
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
""", *argv]
    result = subprocess.run(command, capture_output=True, text=True)
    report = json.loads(out.read_text())
    expected_pass = matching_source and not suppress_history_cleanup
    assert report["passed"] == expected_pass, result.stdout + result.stderr + str(report)
    assert result.returncode == (0 if expected_pass else 1)
    assert report["source_digest"] == digest
    if suppress_history_cleanup:
        assert report["phase"] == "history_glow_cleanup"
        assert "History did not request fresh glow window removal" in report["error"]
        assert not report.get("history_glow_cleanup_passed")
        counts = report["history_glow_window_removals"]
        assert counts["after"] == counts["before"]
    elif matching_source:
        assert report["phase"] == "complete"
        assert report["pending_and_completion_passed"]
        assert report["capture_admitted_before_graphics"]
        assert report["stale_completion_ignored"]
        assert report["handsfree_completion_passed"]
        assert report["installed_pill_timer_clock_passed"]
        assert report["persistent_glow_release_and_retry_passed"]
        assert len(report["retained_fixture_audio"]) == 2
        assert report["history_glow_cleanup_passed"]
        counts = report["history_glow_window_removals"]
        assert counts["after"] == counts["before"] + 1
    else:
        assert report["phase"] == "source"
        assert "digest mismatch" in report["error"]
