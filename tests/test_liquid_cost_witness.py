import pytest

from tests.native_liquid_cost_witness import summarize_capture


@pytest.mark.parametrize("rows", [[], [{"status": 1}],
    [{"status": 0, "size": [3456, 2234], "source_seconds": 1, "received_seconds": 2}],
    [{"status": 0, "size": [640, 360], "source_seconds": None, "received_seconds": 2}],
    [{"status": 0, "size": [640, 360], "source_seconds": 3, "received_seconds": 2}]])
def test_capture_probe_rejects_blank_wrong_route_and_missing_clock(rows):
    with pytest.raises(ValueError):
        summarize_capture(rows, (640, 360))


def test_capture_probe_reports_observed_source_age_only():
    result = summarize_capture([{"status": 0, "size": [640, 360],
        "source_seconds": 1, "received_seconds": 1.025}], (640, 360))
    assert result["complete_frames"] == 1
    assert result["source_to_callback_ms"]["median"] == pytest.approx(25)
    assert "excludes render" in result["claim"]


def test_native_probe_preserves_identity_failure_before_gpu_work(tmp_path, monkeypatch):
    import json
    import sys
    from tests import native_liquid_cost_witness as witness

    monkeypatch.setattr(sys, "argv", ["probe", "--repo-root", str(tmp_path / "wrong-root"),
        "--baseline-root", str(tmp_path / "baseline"), "--out", str(tmp_path / "evidence")])
    assert witness.main() == 1
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["passed"] is False
    assert report["phase"] == "identity"
    assert "Wrong source route" in report["error"]
