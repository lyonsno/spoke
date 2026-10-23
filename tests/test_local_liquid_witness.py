"""False-closure checks for the native localized presentation experiment."""

import pytest


def test_witness_rejects_missing_capture_and_fallback_dimensions():
    from tests import native_local_liquid_witness as witness

    with pytest.raises(ValueError, match="capture"):
        witness.validate_arm({"capture_sizes": [], "displayed_frames": 3}, (800, 600))
    with pytest.raises(ValueError, match="dimensions"):
        witness.validate_arm({"capture_sizes": [[2000, 1600]], "displayed_frames": 3}, (800, 600))


def test_witness_requires_native_presentation_not_submission():
    from tests import native_local_liquid_witness as witness

    with pytest.raises(ValueError, match="presentation"):
        witness.validate_arm({"capture_sizes": [[800, 600]], "displayed_frames": 0,
                              "presented_frames": 20}, (800, 600))


def test_witness_rejects_missing_or_failed_presentation_observer():
    from tests import native_local_liquid_witness as witness

    for rows in ([], [{"outcome": "observer_unavailable"}],
                 [{"outcome": "displayed", "presented_host_seconds": 0}]):
        with pytest.raises(ValueError, match="presentation"):
            witness.validate_arm({"capture_sizes": [[800, 600]], "displayed_frames": 3,
                                  "presentation_rows": rows}, (800, 600))


def test_witness_writes_identity_failure_before_native_work(tmp_path, monkeypatch, mock_pyobjc):
    import json
    import sys
    from tests import native_local_liquid_witness as witness

    monkeypatch.setattr(sys, "argv", ["probe", "--repo-root", str(tmp_path / "wrong"),
                                     "--out", str(tmp_path / "out")])
    assert witness.main() == 1
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["phase"] == "identity"
    assert not report["passed"]
    assert "Effective compositor differs" in report["error"]
