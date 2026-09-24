import json
import hashlib
from pathlib import Path
import struct
import zlib
from datetime import datetime, timezone

from spoke.optical_outcome_witness import (
    build_optical_outcome_report,
    main,
    write_optical_outcome_report,
)


def _png(path: Path) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x20\x30\x40\x50\x60\x70\x00\x11\x22\x33\x44\x55\x66"))
        + chunk(b"IEND", b"")
    )


def _evidence_files(tmp_path: Path, *, pid: int = 41) -> tuple[Path, Path, Path]:
    frame = tmp_path / "frame-000.png"
    _png(frame)
    manifest = tmp_path / "capture-manifest.json"
    manifest.write_text(json.dumps({"frames": [str(frame)]}) + "\n", encoding="utf-8")
    index = tmp_path / "witness-index.json"
    index.write_text(
        json.dumps(
            {
                "started_at": "2026-09-23T12:00:00Z",
                "ended_at": "2026-09-23T12:00:03Z",
                "retina_lasso_manifest": str(manifest),
                "trace_path": str((tmp_path / "trace.jsonl").resolve()),
                "frame_count": 1,
                "command": ["perceptasia-screen-capture"],
                "capture_profile": "low_perturbation",
                "source_app": "com.openai.codex",
                "source_window": "Spoke",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    event = {
        "timestamp": "2026-09-23T12:00:01Z",
        "event": "optical.witness.present",
        "pid": pid,
        "launch_id": "launch-a",
        "launch_target_id": "smoke",
        "source_root": str(tmp_path.resolve()),
        "source_revision": "abc123",
        "source_dirty": False,
        "consumer_id": "spoke.teleporter",
        "client_generation": 3,
        "requested_config_generation": 8,
        "rendered_config_generation": 8,
        "capture_attempt_generation": 2,
        "capture_state": "started",
        "capture_frame_generation": 19,
        "rendered_frame_generation": 19,
        "presented_count": 12,
        "visible": True,
        "transition_phase": "rest",
        "warp_applied": True,
        "warp_dispatch_count": 1,
        "warp_skip_reason": None,
        "optical_config": {"warp_mode": 1.0},
        "smoke_env_sha256": "a" * 64,
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
    return index, trace, frame


def _report(tmp_path: Path, index: Path, trace: Path) -> dict:
    return build_optical_outcome_report(
        capture_index_path=index,
        trace_path=trace,
        consumer="teleporter",
        expected_source_root=tmp_path,
        expected_source_revision="abc123",
        expected_launch_target_id="smoke",
        expected_source_app="com.openai.codex",
        expected_source_window="Spoke",
    )


def test_report_joins_same_process_consumer_generations_and_preserves_frame_hash(tmp_path):
    index, trace, frame = _evidence_files(tmp_path)

    report = _report(tmp_path, index, trace)

    assert report["status"] == "candidate_capture_window_inspection_required"
    assert report["source_identity"]["pid"] == 41
    assert report["source_identity"]["launch_id"] == "launch-a"
    assert report["presentation_receipts"][0]["requested_config_generation"] == 8
    assert report["frames"][0]["path"] == str(frame.resolve())
    assert report["frames"][0]["sha256"] == hashlib.sha256(frame.read_bytes()).hexdigest()
    assert report["capture_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert report["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert report["last_trustworthy_evidence"]["readable_png_count"] == 1
    assert report["failure_phase"] is None
    assert report["visual_assessment"].startswith("unassessed")
    assert "does not establish optical quality" in report["claim_limit"]


def test_report_rejects_mixed_processes_even_when_each_receipt_is_valid(tmp_path):
    index, trace, _frame_path = _evidence_files(tmp_path)
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    second = dict(events[0], pid=42, launch_id="launch-b")
    trace.write_text("\n".join(json.dumps(event) for event in (events[0], second)) + "\n")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert report["failure_phase"] == "process_consumer_generation_join"
    assert "multiple_process_or_source_identities_in_capture_window" in report["failures"]


def test_report_rejects_wrong_consumer_and_unrendered_generation(tmp_path):
    index, trace, _frame_path = _evidence_files(tmp_path)
    event = json.loads(trace.read_text(encoding="utf-8"))
    event["consumer_id"] = "perceptasia.throughglass"
    event["rendered_config_generation"] = 7
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert report["failure_phase"] == "process_consumer_generation_join"
    assert "no_matching_consumer_presentation_in_capture_window" in report["failures"]
    assert report["presentation_receipts"] == []


def test_report_rejects_unreadable_or_non_png_frame(tmp_path):
    index, trace, frame = _evidence_files(tmp_path)
    frame.write_bytes(b"not a png")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert report["failure_phase"] == "frame_verification"
    assert any(failure.startswith("frame_decode_failed:") for failure in report["failures"])
    assert report["frames"] == []


def test_report_rejects_capture_index_not_bound_to_trace_and_manifest(tmp_path):
    index, trace, _frame = _evidence_files(tmp_path)
    capture = json.loads(index.read_text(encoding="utf-8"))
    capture["trace_path"] = str(tmp_path / "another-trace.jsonl")
    capture["frame_count"] = 900
    capture["command"] = ["missing-capturer"]
    capture["capture_profile"] = "bogus"
    capture["source_app"] = "wrong.app"
    index.write_text(json.dumps(capture), encoding="utf-8")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert "capture_trace_path_mismatch" in report["failures"]
    assert "capture_frame_count_mismatch" in report["failures"]
    assert "capture_profile_unrecognized" in report["failures"]
    assert "capture_source_app_mismatch" in report["failures"]


def test_report_accepts_index_written_by_real_capture_index_producer(tmp_path):
    from spoke.retina_lasso_witness import write_witness_index

    frame = tmp_path / "frame.png"
    _png(frame)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"frames": [str(frame)]}), encoding="utf-8"
    )
    trace = tmp_path / "trace.jsonl"
    event = _evidence_files(tmp_path)[1]
    index = write_witness_index(
        output_dir=tmp_path,
        trace_path=trace,
        started_at=datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 9, 23, 12, 0, 3, tzinfo=timezone.utc),
        command=["perceptasia-screen-capture", "--source-app", "com.openai.codex"],
        trace_events=[json.loads(event.read_text(encoding="utf-8"))],
        source_app="com.openai.codex",
        source_window="Spoke",
    )

    report = _report(tmp_path, index, trace)

    assert report["status"] == "candidate_capture_window_inspection_required"
    assert report["capture_run"]["source_app"] == "com.openai.codex"
    assert report["capture_run"]["source_window"] == "Spoke"


def test_report_rejects_present_without_warp_dispatch(tmp_path):
    index, trace, _frame = _evidence_files(tmp_path)
    event = json.loads(trace.read_text(encoding="utf-8"))
    event["warp_applied"] = False
    event["warp_dispatch_count"] = 0
    event["warp_skip_reason"] = "empty_dispatch_box"
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert "no_warp_dispatch_in_capture_window" in report["failures"]


def test_cli_writes_input_loading_report_for_non_utf8_trace(tmp_path):
    index, trace, _frame = _evidence_files(tmp_path)
    trace.write_bytes(b"\xff")
    output = tmp_path / "outcome.json"

    result = main([
        "--capture-index", str(index), "--trace", str(trace),
        "--consumer", "teleporter", "--source-root", str(tmp_path),
        "--source-revision", "abc123", "--launch-target", "smoke",
        "--output", str(output),
    ])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 2
    assert report["failure_phase"] == "input_loading"
    assert any(item.startswith("trace_load_failed:UnicodeDecodeError") for item in report["failures"])


def test_report_can_join_throughglass_by_its_exact_client_id(tmp_path):
    index, trace, _frame_path = _evidence_files(tmp_path)
    event = json.loads(trace.read_text(encoding="utf-8"))
    event["consumer_id"] = "perceptasia.throughglass"
    trace.write_text(json.dumps(event) + "\n", encoding="utf-8")

    report = build_optical_outcome_report(
        capture_index_path=index,
        trace_path=trace,
        consumer="throughglass",
        expected_source_root=tmp_path,
        expected_source_revision="abc123",
        expected_launch_target_id="smoke",
    )

    assert report["status"] == "candidate_capture_window_inspection_required"
    assert report["consumer"]["client_id"] == "perceptasia.throughglass"


def test_cli_writes_a_failure_report_when_inputs_cannot_be_opened(tmp_path):
    output = tmp_path / "outcome.json"

    result = main(
        [
            "--capture-index",
            str(tmp_path / "missing-index.json"),
            "--trace",
            str(tmp_path / "missing-trace.jsonl"),
            "--consumer",
            "teleporter",
            "--source-root",
            str(tmp_path),
            "--source-revision",
            "abc123",
            "--launch-target",
            "smoke",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 2
    assert report["status"] == "incomplete"
    assert report["failure_phase"] == "input_loading"
    assert any(failure.startswith("capture_index_load_failed:") for failure in report["failures"])
    assert any(failure.startswith("trace_load_failed:") for failure in report["failures"])


def test_report_writer_refuses_to_replace_a_source_frame(tmp_path):
    index, trace, frame = _evidence_files(tmp_path)
    report = _report(tmp_path, index, trace)
    original = frame.read_bytes()

    try:
        write_optical_outcome_report(report, frame)
    except ValueError as exc:
        assert "must not overwrite captured evidence" in str(exc)
    else:
        raise AssertionError("report writer must protect source pixels")

    assert frame.read_bytes() == original


def test_report_writer_protects_corrupt_manifest_frame(tmp_path):
    index, trace, frame = _evidence_files(tmp_path)
    frame.write_bytes(b"corrupt evidence")
    report = _report(tmp_path, index, trace)
    original = frame.read_bytes()

    try:
        write_optical_outcome_report(report, frame)
    except ValueError as exc:
        assert "must not overwrite captured evidence" in str(exc)
    else:
        raise AssertionError("report writer must protect unreadable source pixels")

    assert frame.read_bytes() == original


def test_report_writer_protects_paths_parsed_before_malformed_manifest_entry(tmp_path):
    index, trace, frame = _evidence_files(tmp_path)
    capture = json.loads(index.read_text(encoding="utf-8"))
    manifest_path = Path(capture["retina_lasso_manifest"])
    manifest_path.write_text(
        json.dumps({"frames": [str(frame), {"unexpected": "no-path"}]}),
        encoding="utf-8",
    )
    index.write_text(json.dumps(capture), encoding="utf-8")
    report = _report(tmp_path, index, trace)
    original = frame.read_bytes()

    try:
        write_optical_outcome_report(report, frame)
    except ValueError as exc:
        assert "must not overwrite captured evidence" in str(exc)
    else:
        raise AssertionError("report writer must protect paths parsed from malformed manifests")

    assert frame.read_bytes() == original


def test_incomplete_report_does_not_claim_a_capture_window_join(tmp_path):
    index, trace, _frame = _evidence_files(tmp_path)
    trace.write_bytes(b"\xff")

    report = _report(tmp_path, index, trace)

    assert report["status"] == "incomplete"
    assert "overlaps" not in report["claim_limit"]
