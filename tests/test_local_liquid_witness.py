"""False-closure checks for the native localized presentation experiment."""

import pytest


def _presented_arm():
    return {"capture_sizes": [[800, 600]], "displayed_frames": 3,
            "presentation_rows": [{"outcome": "displayed", "presented_host_seconds": 12}],
            "warp_rows": [{"capture_size": [800, 600], "drawable_size": [800, 600],
                           "dispatch_count": 1, "skip_reason": None,
                           "composited_to_drawable_texture": True}]}


def test_observed_house_dispatch_and_native_presentation_are_admitted():
    from tests import native_local_liquid_witness as witness

    witness.validate_arm(_presented_arm(), (800, 600))


@pytest.mark.parametrize("patch", [{"dispatch_count": 0, "skip_reason": "parameter_buffer_unavailable"},
                                   {"drawable_size": [1600, 1200]},
                                   {"composited_to_drawable_texture": False}])
def test_presented_passthrough_is_not_a_house_warp(patch):
    from tests import native_local_liquid_witness as witness

    arm = _presented_arm()
    arm["warp_rows"][0].update(patch)
    with pytest.raises(ValueError, match="warp"):
        witness.validate_arm(arm, (800, 600))


def test_second_arm_excludes_first_arm_pipeline_work():
    from types import SimpleNamespace
    from tests import native_local_liquid_witness as witness

    before = {"drawable_copy_frames": 10, "drawable_copy_pixels": 10000000,
              "mip_generation_frames": 0, "mip_generation_source_pixels": 0,
              "warp_dispatches": 10, "warp_dispatch_pixels": 1000000}
    after = {**before, "drawable_copy_frames": 20, "drawable_copy_pixels": 10100000,
             "warp_dispatches": 20, "warp_dispatch_pixels": 1010000,
             "avg_drawable_copy_mp": 0.505, "avg_warp_dispatch_mp": 0.0505}
    compositor = SimpleNamespace(diagnostics_snapshot=lambda: after,
                                 _pipeline=SimpleNamespace(diagnostics_snapshot=lambda: after))
    arm = {"pipeline_before": before}
    witness.finish_arm(compositor, arm)
    assert arm["drawable_copy_frames"] == 10
    assert arm["drawable_copy_pixels"] == 100000
    assert arm["avg_drawable_copy_mp"] == pytest.approx(0.01)
    assert arm["avg_warp_dispatch_mp"] == pytest.approx(0.001)


def test_failed_zero_work_arm_keeps_prior_work_only_in_lifetime_snapshots():
    from types import SimpleNamespace
    from tests import native_local_liquid_witness as witness

    previous = {"drawable_copy_frames": 187, "drawable_copy_pixels": 1443771648,
                "mip_generation_frames": 0, "mip_generation_source_pixels": 0,
                "warp_dispatches": 187, "warp_dispatch_pixels": 51026503}
    compositor = SimpleNamespace(diagnostics_snapshot=lambda: previous,
        _pipeline=SimpleNamespace(diagnostics_snapshot=lambda: previous))
    arm = {"pipeline_before": previous, "capture_sizes": [], "warp_rows": []}
    witness.finish_failed_arm(compositor, arm)
    assert arm["drawable_copy_frames"] == arm["drawable_copy_pixels"] == 0
    assert arm["warp_dispatches"] == arm["warp_dispatch_pixels"] == 0
    assert arm["pipeline_after"] == previous
    assert arm["pipeline_before"] == previous


def test_failure_diagnostic_error_preserves_original_failure_and_raw_rows():
    from types import SimpleNamespace
    from tests import native_local_liquid_witness as witness

    def broken():
        raise RuntimeError("diagnostic collection failed")

    rows = [{"capture_size": [800, 600]}]
    arm = {"error": "original startup failure", "warp_rows": rows}
    witness.finish_failed_arm(SimpleNamespace(diagnostics_snapshot=broken), arm)
    assert arm["error"] == "original startup failure"
    assert arm["warp_rows"] is rows
    assert "diagnostic collection failed" in arm["diagnostic_finalization_error"]
    assert arm["diagnostics_authority"] == "last-trustworthy-only"


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
