"""Join optical presentation receipts to the exact passive pixel capture run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


CONSUMER_IDS = {
    "teleporter": "spoke.teleporter",
    "throughglass": "perceptasia.throughglass",
}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _frame_paths(capture_index: dict[str, Any], index_path: Path) -> tuple[Path | None, list[Path]]:
    raw_manifest = capture_index.get("retina_lasso_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest.strip():
        return None, []
    manifest_path = Path(raw_manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = index_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("capture manifest has no frames list")
    paths: list[Path] = []
    for frame in raw_frames:
        value = frame if isinstance(frame, str) else None
        if isinstance(frame, dict):
            value = frame.get("path") or frame.get("image") or frame.get("file")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("capture manifest contains a frame without a path")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path.resolve())
    return manifest_path, paths


def _read_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(event, dict):
                events.append(event)
            else:
                malformed += 1
    return events, malformed


def _frame_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    from .perceptasia_throughglass_witness import _read_png_rgb

    records = []
    failures = []
    seen: set[Path] = set()
    for index, path in enumerate(paths):
        if path in seen:
            failures.append(f"duplicate_frame_path:{path}")
            continue
        seen.add(path)
        try:
            contents = path.read_bytes()
            image = _read_png_rgb(path)
        except OSError as exc:
            failures.append(f"frame_unreadable:{path}:{type(exc).__name__}")
            continue
        except Exception as exc:
            failures.append(f"frame_decode_failed:{path}:{type(exc).__name__}")
            continue
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
            failures.append(f"frame_invalid_dimensions:{path}")
            continue
        records.append(
            {
                "index": index,
                "path": str(path),
                "bytes": len(contents),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    if not records:
        failures.append("no_readable_capture_frames")
    return records, failures


def build_optical_outcome_report(
    *,
    capture_index_path: str | Path,
    trace_path: str | Path,
    consumer: str,
    expected_source_root: str | Path,
    expected_source_revision: str,
    expected_launch_target_id: str,
) -> dict[str, Any]:
    """Join one Retina Lasso window to same-process, same-consumer present receipts.

    This reports evidence integrity and provenance only. It never grades warp quality.
    """
    capture_index_file = Path(capture_index_path).expanduser().resolve()
    trace_file = Path(trace_path).expanduser().resolve()
    expected_consumer_id = CONSUMER_IDS[consumer]
    expected_root = str(Path(expected_source_root).expanduser().resolve())
    failures: list[str] = []
    capture_window: dict[str, str | None] = {"started_at": None, "ended_at": None}
    frame_manifest_path: Path | None = None
    frames: list[Path] = []
    frame_records: list[dict[str, Any]] = []
    candidate_receipts: list[dict[str, Any]] = []
    malformed_trace_lines = 0
    capture_index: dict[str, Any] = {}
    capture_index_loaded = False
    trace_loaded = False

    try:
        capture_index = _load_json(capture_index_file)
        capture_index_loaded = True
        start = _parse_time(capture_index.get("started_at"))
        end = _parse_time(capture_index.get("ended_at"))
        capture_window = {
            "started_at": capture_index.get("started_at"),
            "ended_at": capture_index.get("ended_at"),
        }
        if start is None or end is None or end < start:
            failures.append("invalid_capture_window")
        frame_manifest_path, frames = _frame_paths(capture_index, capture_index_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        capture_index = {}
        failures.append(f"capture_index_load_failed:{type(exc).__name__}:{exc}")

    try:
        trace_events, malformed_trace_lines = _read_trace(trace_file)
        trace_loaded = True
    except OSError as exc:
        trace_events = []
        failures.append(f"trace_load_failed:{type(exc).__name__}:{exc}")
    if malformed_trace_lines:
        failures.append(f"malformed_trace_lines:{malformed_trace_lines}")

    start = _parse_time(capture_window.get("started_at"))
    end = _parse_time(capture_window.get("ended_at"))
    if start is not None and end is not None:
        for event in trace_events:
            if event.get("event") != "optical.witness.present":
                continue
            if event.get("consumer_id") != expected_consumer_id:
                continue
            event_time = _parse_time(event.get("timestamp"))
            if event_time is not None and start <= event_time <= end:
                candidate_receipts.append(event)

    if not candidate_receipts:
        failures.append("no_matching_consumer_presentation_in_capture_window")
    identity_keys = (
        "pid",
        "launch_id",
        "source_root",
        "source_revision",
        "launch_target_id",
        "smoke_env_sha256",
    )
    identity_values = [
        tuple(receipt.get(key) for key in identity_keys)
        for receipt in candidate_receipts
    ]
    identities = {json.dumps(value, sort_keys=True) for value in identity_values}
    if len(identities) > 1:
        failures.append("multiple_process_or_source_identities_in_capture_window")

    identity = None
    if len(identities) == 1:
        (
            pid,
            launch_id,
            source_root,
            source_revision,
            launch_target_id,
            smoke_env_sha256,
        ) = json.loads(
            next(iter(identities))
        )
        identity = {
            "pid": pid,
            "launch_id": launch_id,
            "source_root": source_root,
            "source_revision": source_revision,
            "launch_target_id": launch_target_id,
            "source_dirty": candidate_receipts[0].get("source_dirty"),
            "smoke_env_sha256": smoke_env_sha256,
        }
        if not isinstance(pid, int) or pid <= 0:
            failures.append("missing_process_id")
        if not isinstance(launch_id, str) or not launch_id:
            failures.append("missing_launch_id")
        if source_root != expected_root:
            failures.append("source_root_mismatch")
        if source_revision != expected_source_revision:
            failures.append("source_revision_mismatch")
        if launch_target_id != expected_launch_target_id:
            failures.append("launch_target_mismatch")
        if candidate_receipts[0].get("source_dirty") is not False:
            failures.append("source_worktree_not_proven_clean")

    for index, receipt in enumerate(candidate_receipts):
        required_positive_ints = (
            "client_generation",
            "requested_config_generation",
            "capture_attempt_generation",
            "capture_frame_generation",
            "rendered_frame_generation",
            "presented_count",
        )
        for key in required_positive_ints:
            value = receipt.get(key)
            if not isinstance(value, int) or value <= 0:
                failures.append(f"receipt_{index}_invalid_{key}")
        if receipt.get("capture_state") != "started":
            failures.append(f"receipt_{index}_capture_not_started")
        if receipt.get("visible") is not True:
            failures.append(f"receipt_{index}_consumer_not_visible")
        if receipt.get("transition_phase") != "rest":
            failures.append(f"receipt_{index}_consumer_not_settled")
        rendered = receipt.get("rendered_config_generation")
        requested = receipt.get("requested_config_generation")
        if (
            not isinstance(rendered, int)
            or not isinstance(requested, int)
            or rendered < requested
        ):
            failures.append(f"receipt_{index}_config_not_rendered")

    frame_records, frame_failures = _frame_records(frames)
    failures.extend(frame_failures)
    if any(failure.startswith(("capture_index_load_failed:", "trace_load_failed:")) for failure in failures):
        failure_phase = "input_loading"
    elif any(failure.startswith(("frame_", "no_readable_capture_frames", "duplicate_frame_path")) for failure in failures):
        failure_phase = "frame_verification"
    elif failures:
        failure_phase = "process_consumer_generation_join"
    else:
        failure_phase = None
    return {
        "schema": "spoke.optical_outcome_witness.v1",
        "status": "joined_evidence_inspection_required" if not failures else "incomplete",
        "failure_phase": failure_phase,
        "consumer": {"name": consumer, "client_id": expected_consumer_id},
        "source_identity": identity,
        "capture_window": capture_window,
        "capture_index": str(capture_index_file),
        "capture_index_sha256": _sha256(capture_index_file),
        "trace": str(trace_file),
        "trace_sha256": _sha256(trace_file),
        "capture_run": {
            "command": capture_index.get("command", []),
            "profile": capture_index.get("capture_profile"),
            "source_app": capture_index.get("source_app"),
            "source_window": capture_index.get("source_window"),
        },
        "frame_manifest": str(frame_manifest_path) if frame_manifest_path else None,
        "frame_manifest_sha256": _sha256(frame_manifest_path) if frame_manifest_path else None,
        "frame_count": len(frame_records),
        "frames": frame_records,
        "presentation_receipts": candidate_receipts,
        "malformed_trace_lines": malformed_trace_lines,
        "last_trustworthy_evidence": {
            "capture_index_loaded": capture_index_loaded,
            "capture_window": capture_window,
            "trace_loaded": trace_loaded,
            "trace_event_count": len(trace_events),
            "matching_presentation_receipt_count": len(candidate_receipts),
            "readable_png_count": len(frame_records),
        },
        "failures": failures,
        "visual_assessment": "unassessed; inspect the preserved frame files",
        "claim_limit": (
            "The capture window overlaps a same-process, same-consumer settled compositor presentation. "
            "Retina Lasso does not bind each PNG to an individual compositor frame generation, and "
            "this report does not establish optical quality or prove warp by pixel score."
        ),
    }


def write_optical_outcome_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    protected_paths = {
        report.get("capture_index"),
        report.get("trace"),
        report.get("frame_manifest"),
        *(frame.get("path") for frame in report.get("frames", [])),
    }
    if str(path) in protected_paths:
        raise ValueError("report output must not overwrite captured evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--consumer", required=True, choices=tuple(CONSUMER_IDS))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--launch-target", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_optical_outcome_report(
        capture_index_path=args.capture_index,
        trace_path=args.trace,
        consumer=args.consumer,
        expected_source_root=args.source_root,
        expected_source_revision=args.source_revision,
        expected_launch_target_id=args.launch_target,
    )
    output = write_optical_outcome_report(report, args.output)
    print(output)
    print(report["status"])
    return 0 if report["status"] == "joined_evidence_inspection_required" else 2


if __name__ == "__main__":
    raise SystemExit(main())
