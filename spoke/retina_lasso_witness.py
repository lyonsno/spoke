"""Trace-aligned Retina Lasso smoke witness for visible Spoke UI states."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PERCEPTASIA_ROOT = Path("/private/tmp/perceptasia-codex-screen-slice-smoke-loop-0521")
GLOBAL_CAPTURE_COMMAND_NAMES = {
    "global-witness-capture",
    "epistaxis-global-witness-capture",
    "global_witness_capture.py",
}
DEFAULT_LANE = "warpstorm-pit-boss"
DEFAULT_DIAULOS = "Warpstorm Pit Boss"
DEFAULT_SOURCE_APP = "Spoke"
DEFAULT_SOURCE_WINDOW = "Command Overlay"
INDEX_NAME = "witness-index.json"
SPACEBAR_KEYCODE = 49
RETURN_KEYCODE = 36
SUPPORTED_CAPTURE_PROFILES = {"low_perturbation", "stress"}
DEFAULT_PASSIVE_CAPTURE_PROFILE = "low_perturbation"
DEFAULT_STRESS_CAPTURE_PROFILE = "stress"
CAPTURE_PROFILE_FPS = {
    "low_perturbation": 6.0,
    "stress": 15.0,
}
DEFAULT_TRACE_TRIGGER_EVENTS = {
    "overlay.show.begin",
    "overlay.cancel_dismiss.begin",
    "overlay.fade_out.start",
    "overlay.fade_out.complete",
    "overlay.show.retarget_dismiss_to_summon",
}
DEFAULT_TRACE_TRIGGER_MAX_LAG_SECONDS = 1.0


def _default_uv_command() -> str:
    env_uv = os.environ.get("UV_BIN")
    if env_uv:
        return str(Path(env_uv).expanduser())
    which_uv = shutil.which("uv")
    if which_uv:
        return which_uv
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
        Path("/Users/noahlyons/.pyenv/shims/uv"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return "uv"


def _global_capture_command_is_healthy(command: str) -> bool:
    try:
        subprocess.run(
            [command, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2.0,
        )
    except Exception:
        return False
    return True


def _default_global_capture_command() -> str | None:
    env_command = (
        os.environ.get("SPOKE_RETINA_LASSO_CAPTURE_COMMAND")
        or os.environ.get("GLOBAL_WITNESS_CAPTURE_COMMAND")
    )
    if env_command:
        return str(Path(env_command).expanduser())
    for name in ("global-witness-capture", "epistaxis-global-witness-capture"):
        which_command = shutil.which(name)
        if which_command and _global_capture_command_is_healthy(which_command):
            return which_command
    for candidate in (
        Path.home() / ".local" / "bin" / "global-witness-capture",
        Path.home() / ".local" / "bin" / "epistaxis-global-witness-capture",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK) and _global_capture_command_is_healthy(
            str(candidate)
        ):
            return str(candidate)
    return None


def _is_global_capture_command(command: Sequence[str]) -> bool:
    return bool(command) and Path(command[0]).name in GLOBAL_CAPTURE_COMMAND_NAMES


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_trace_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_instant(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def capture_count_for_window(duration_seconds: float, fps: float) -> int:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, int(math.ceil(duration_seconds * fps)))


def capture_interval_for_fps(fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return 1.0 / fps


def _capture_profile_from_cli(value: str) -> str:
    return value.replace("-", "_")


def default_fps_for_capture_profile(capture_profile: str) -> float:
    if capture_profile not in SUPPORTED_CAPTURE_PROFILES:
        raise ValueError(f"unsupported capture profile: {capture_profile}")
    return CAPTURE_PROFILE_FPS[capture_profile]


def collect_trace_events(
    trace_path: str | Path,
    *,
    started_at: datetime,
    ended_at: datetime,
    interesting_events: set[str] | None = None,
) -> list[dict[str, Any]]:
    path = Path(trace_path).expanduser()
    if not path.exists():
        return []
    start = started_at.astimezone(timezone.utc)
    end = ended_at.astimezone(timezone.utc)
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
            event_time = _parse_trace_timestamp(str(event["timestamp"]))
        except Exception:
            continue
        if event_time < start or event_time > end:
            continue
        event_name = str(event.get("event", ""))
        if interesting_events is not None and event_name not in interesting_events:
            continue
        event = dict(event)
        event["trace_line"] = line_number
        events.append(event)
    return events


def read_trace_events_from_offset(
    trace_path: str | Path,
    *,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    path = Path(trace_path).expanduser()
    if not path.exists():
        return 0, []
    if offset > path.stat().st_size:
        offset = 0
    events: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for raw_line in handle:
            try:
                event = json.loads(raw_line.decode("utf-8"))
                _parse_trace_timestamp(str(event["timestamp"]))
            except Exception:
                continue
            events.append(dict(event))
        return handle.tell(), events


def trace_event_output_slug(event: dict[str, Any], *, index: int) -> str:
    event_name = str(event.get("event", "event"))
    safe_event = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in event_name)
    safe_event = safe_event.strip("-") or "event"
    timestamp = str(event.get("timestamp", "unknown"))
    safe_timestamp = "".join(ch if ch.isalnum() else "-" for ch in timestamp).strip("-") or "unknown"
    return f"{index:03d}-{safe_event}-{safe_timestamp}"


def build_retina_lasso_command(
    *,
    output_dir: str | Path,
    count: int,
    interval_seconds: float,
    lane: str,
    diaulos: str,
    source_app: str,
    source_window: str,
    trace_path: str | Path | None = None,
    capture_profile: str = DEFAULT_PASSIVE_CAPTURE_PROFILE,
    capture_command: str | Path | None = None,
    capture_mode: str | None = None,
    capture_rect: str | None = None,
    window_id: str | int | None = None,
    display_id: str | int | None = None,
    uv_command: str | Path | None = None,
) -> list[str]:
    resolved_capture_command = (
        str(Path(capture_command).expanduser())
        if capture_command is not None
        else _default_global_capture_command()
    )
    if resolved_capture_command is not None:
        command = [
            resolved_capture_command,
            "--output-dir",
            str(Path(output_dir).expanduser()),
            "--count",
            str(count),
            "--interval",
            f"{interval_seconds:.6f}",
            "--capture-profile",
            capture_profile.replace("_", "-"),
            "--lane",
            lane,
            "--diaulos",
            diaulos,
            "--source-app",
            source_app,
            "--source-window",
            source_window,
        ]
        if trace_path is not None:
            command.extend(["--trace-path", str(Path(trace_path).expanduser())])
        if capture_mode:
            command.extend(["--mode", capture_mode])
        if capture_rect:
            command.extend(["--rect", capture_rect])
        if window_id is not None:
            command.extend(["--window-id", str(window_id)])
        if display_id is not None:
            command.extend(["--display-id", str(display_id)])
        return command

    if capture_mode or capture_rect or window_id is not None or display_id is not None:
        raise ValueError("bounded capture requires global witness capture command")

    resolved_uv = str(Path(uv_command).expanduser()) if uv_command is not None else _default_uv_command()
    return [
        resolved_uv,
        "run",
        "perceptasia-screen-capture",
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--count",
        str(count),
        "--interval",
        f"{interval_seconds:.6f}",
        "--lane",
        lane,
        "--diaulos",
        diaulos,
        "--source-app",
        source_app,
        "--source-window",
        source_window,
    ]


def build_launch_target_command(repo_root: str | Path, target_id: str) -> list[str]:
    return [str(Path(repo_root).expanduser() / "scripts" / "launch-target.sh"), target_id]


def post_key_event(keycode: int, is_down: bool) -> None:
    from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap

    event = CGEventCreateKeyboardEvent(None, int(keycode), bool(is_down))
    CGEventPost(kCGHIDEventTap, event)


def post_space_enter_toggle_chord(*, key_pause_seconds: float = 0.035) -> None:
    """Synthesize the same space-first Enter chord the operator uses."""
    post_key_event(SPACEBAR_KEYCODE, True)
    time.sleep(key_pause_seconds)
    post_key_event(RETURN_KEYCODE, True)
    time.sleep(key_pause_seconds)
    post_key_event(RETURN_KEYCODE, False)
    time.sleep(key_pause_seconds)
    post_key_event(SPACEBAR_KEYCODE, False)


def drive_hammer_toggles(
    *,
    count: int,
    interval_seconds: float,
    key_pause_seconds: float = 0.035,
    toggle_chord: Callable[..., None] = post_space_enter_toggle_chord,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if count < 0:
        raise ValueError("count must be non-negative")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")
    for index in range(count):
        toggle_chord(key_pause_seconds=key_pause_seconds)
        if index < count - 1:
            sleep(interval_seconds)


def trace_event_is_open_ready(event: dict[str, Any]) -> bool:
    return (
        event.get("event") == "overlay.visual_ready.push"
        and event.get("presentation_text_state") == "visible"
        and event.get("presentation_body_state") == "body_ready"
    )


def _trace_event_open_generation(event: dict[str, Any]) -> int | None:
    if event.get("event") != "overlay.show.begin":
        return None
    if event.get("presentation_requested_state") != "opening":
        return None
    generation = event.get("presentation_generation")
    return generation if isinstance(generation, int) else None


def wait_for_open_ready_trace(
    trace_path: str | Path,
    *,
    offset: int = 0,
    timeout_seconds: float = 2.0,
    poll_interval_seconds: float = 0.025,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any] | None]:
    """Wait until a fresh show transition reaches human-publishable open state."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    deadline = monotonic() + timeout_seconds
    current_offset = offset
    open_generation: int | None = None
    while monotonic() < deadline:
        current_offset, events = read_trace_events_from_offset(trace_path, offset=current_offset)
        for event in events:
            generation = _trace_event_open_generation(event)
            if generation is not None:
                open_generation = generation
                continue
            if (
                open_generation is not None
                and event.get("presentation_generation") == open_generation
                and trace_event_is_open_ready(event)
            ):
                return current_offset, event
        sleep(poll_interval_seconds)
    current_offset, events = read_trace_events_from_offset(trace_path, offset=current_offset)
    for event in events:
        generation = _trace_event_open_generation(event)
        if generation is not None:
            open_generation = generation
            continue
        if (
            open_generation is not None
            and event.get("presentation_generation") == open_generation
            and trace_event_is_open_ready(event)
        ):
            return current_offset, event
    return current_offset, None


def drive_retarget_during_dismiss_pattern(
    *,
    repeats: int,
    open_dwell_seconds: float,
    dismiss_retarget_delay_seconds: float,
    reopen_dwell_seconds: float,
    cycle_pause_seconds: float = 0.2,
    trace_path: str | Path | None = None,
    open_ready_timeout_seconds: float = 2.0,
    open_ready_poll_interval_seconds: float = 0.025,
    key_pause_seconds: float = 0.035,
    toggle_chord: Callable[..., None] = post_space_enter_toggle_chord,
    sleep: Callable[[float], None] = time.sleep,
    ready_waiter: Callable[..., tuple[int, dict[str, Any] | None]] = wait_for_open_ready_trace,
) -> list[dict[str, Any]]:
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    trace_offset = Path(trace_path).expanduser().stat().st_size if trace_path and Path(trace_path).expanduser().exists() else 0
    results: list[dict[str, Any]] = []
    for index in range(repeats):
        toggle_chord(key_pause_seconds=key_pause_seconds)
        if trace_path is not None:
            trace_offset, event = ready_waiter(
                trace_path,
                offset=trace_offset,
                timeout_seconds=open_ready_timeout_seconds,
                poll_interval_seconds=open_ready_poll_interval_seconds,
                sleep=sleep,
            )
            results.append(
                {
                    "cycle": index + 1,
                    "phase": "open",
                    "status": "ready" if event is not None else "timeout",
                    "presentation_generation": event.get("presentation_generation") if event else None,
                }
            )
            if event is None:
                toggle_chord(key_pause_seconds=key_pause_seconds)
                if index < repeats - 1:
                    sleep(cycle_pause_seconds)
                continue
        sleep(open_dwell_seconds)
        toggle_chord(key_pause_seconds=key_pause_seconds)
        sleep(dismiss_retarget_delay_seconds)
        toggle_chord(key_pause_seconds=key_pause_seconds)
        if trace_path is not None:
            trace_offset, event = ready_waiter(
                trace_path,
                offset=trace_offset,
                timeout_seconds=open_ready_timeout_seconds,
                poll_interval_seconds=open_ready_poll_interval_seconds,
                sleep=sleep,
            )
            results.append(
                {
                    "cycle": index + 1,
                    "phase": "reopen",
                    "status": "ready" if event is not None else "timeout",
                    "presentation_generation": event.get("presentation_generation") if event else None,
                }
            )
        if trace_path is None or event is not None:
            sleep(reopen_dwell_seconds)
        toggle_chord(key_pause_seconds=key_pause_seconds)
        if index < repeats - 1:
            sleep(cycle_pause_seconds)
    return results


def build_evidence_split(
    *,
    manifest_loaded: bool,
    frame_count: int,
    trace_event_count: int,
    capture_profile: str,
) -> dict[str, Any]:
    """Describe what the witness can and cannot prove."""
    if capture_profile not in SUPPORTED_CAPTURE_PROFILES:
        raise ValueError(f"unsupported capture profile: {capture_profile}")
    return {
        "schema": "spoke.retina_lasso_evidence_split.v1",
        "visual_witness": {
            "role": "perturbing_visual_stress_witness",
            "manifest_loaded": manifest_loaded,
            "frame_count": frame_count,
            "capture_profile": capture_profile,
            "can_prove_absence": False,
            "can_raise_candidate_bad_frame": True,
            "known_perturbations": [
                "WindowServer/SCK capture pressure",
                "GPU pressure",
                "frame-cadence sampling gaps",
                "capture-path visual artifacts",
            ],
            "known_capture_artifact_signatures": [
                {
                    "signature": "horizontal_tear_or_phase_split",
                    "description": (
                        "A horizontal or banded discontinuity where one screen slice appears "
                        "one display phase older than another."
                    ),
                    "classification_without_trace": "witness_contamination_candidate",
                },
                {
                    "signature": "stale_slice_or_partial_frame_composite",
                    "description": (
                        "A still frame that appears composited from two different capture instants "
                        "while the live UI later lands in a lawful state."
                    ),
                    "classification_without_trace": "witness_contamination_candidate",
                },
            ],
        },
        "lifecycle_trace": {
            "role": "generation_lifecycle_receipts",
            "trace_event_count": trace_event_count,
            "required_for_extraction_clearance": True,
            "can_adjudicate_publication_law": True,
        },
        "classification_rule": {
            "witness_clean": "not_clearance",
            "witness_bad_frame": "candidate_violation_until_trace_correlated",
            "trace_unlawful_publication": "primitive_lifecycle_blocker",
            "trace_lawful_with_visual_artifact": "witness_reliability_or_capture_artifact",
            "known_capture_artifact_without_trace_violation": "not_a_primitive_blocker_by_itself",
        },
    }


def write_witness_index(
    *,
    output_dir: str | Path,
    trace_path: str | Path,
    started_at: datetime,
    ended_at: datetime,
    command: Sequence[str],
    trace_events: list[dict[str, Any]],
    manifest_name: str = "manifest.json",
    stimulus: dict[str, Any] | None = None,
    capture_profile: str = DEFAULT_PASSIVE_CAPTURE_PROFILE,
) -> Path:
    output = Path(output_dir).expanduser()
    manifest_path = output / manifest_name
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_loaded = manifest is not None
    frame_count = len((manifest or {}).get("frames", []))
    trace_event_count = len(trace_events)
    payload = {
        "schema": "spoke.retina_lasso_trace_witness.v1",
        "started_at": _format_instant(started_at),
        "ended_at": _format_instant(ended_at),
        "trace_path": str(Path(trace_path).expanduser()),
        "retina_lasso_manifest": str(manifest_path),
        "retina_lasso_manifest_loaded": manifest_loaded,
        "frame_count": frame_count,
        "trace_event_count": trace_event_count,
        "trace_events": trace_events,
        "evidence_split": build_evidence_split(
            manifest_loaded=manifest_loaded,
            frame_count=frame_count,
            trace_event_count=trace_event_count,
            capture_profile=capture_profile,
        ),
        "command": list(command),
        "stimulus": stimulus or {},
        "capture_profile": capture_profile,
        "uncertainty": [
            "Retina Lasso stills are visual evidence, not operator approval.",
            "Retina Lasso stills are perturbing stress evidence, not an absence oracle.",
            "Known Retina Lasso capture artifacts include horizontal tear/phase split and stale-slice composites.",
            "Still-frame cadence can miss a single display-refresh flash; use trace events to narrow the gap.",
            "Trace alignment is wall-clock based and does not prove exact compositor frame identity.",
        ],
    }
    index_path = output / INDEX_NAME
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return index_path


def run_autonomous_hammer_witness(
    *,
    trace_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    perceptasia_root: str | Path = DEFAULT_PERCEPTASIA_ROOT,
    duration_seconds: float = 8.0,
    fps: float | None = None,
    capture_profile: str = DEFAULT_STRESS_CAPTURE_PROFILE,
    hammer_toggles: int = 0,
    toggle_interval_seconds: float = 0.18,
    pre_hammer_delay_seconds: float = 0.35,
    retarget_during_dismiss_repeats: int = 0,
    open_dwell_seconds: float = 0.75,
    dismiss_retarget_delay_seconds: float = 0.08,
    reopen_dwell_seconds: float = 0.75,
    cycle_pause_seconds: float = 0.2,
    open_ready_timeout_seconds: float = 2.0,
    open_ready_poll_interval_seconds: float = 0.025,
    lane: str = DEFAULT_LANE,
    diaulos: str = DEFAULT_DIAULOS,
    source_app: str = DEFAULT_SOURCE_APP,
    source_window: str = DEFAULT_SOURCE_WINDOW,
    capture_command: str | Path | None = None,
    capture_mode: str | None = None,
    capture_rect: str | None = None,
    window_id: str | int | None = None,
    display_id: str | int | None = None,
    launch_target: str | None = None,
    launch_wait_seconds: float = 3.0,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    now: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
    hammer_driver: Callable[..., None] = drive_hammer_toggles,
    retarget_driver: Callable[..., None] = drive_retarget_during_dismiss_pattern,
) -> Path:
    if fps is None:
        fps = default_fps_for_capture_profile(capture_profile)
    if launch_target:
        runner(build_launch_target_command(repo_root, launch_target), check=True)
        sleep(launch_wait_seconds)

    count = capture_count_for_window(duration_seconds, fps)
    interval = capture_interval_for_fps(fps)
    command = build_retina_lasso_command(
        output_dir=output_dir,
        count=count,
        interval_seconds=interval,
        lane=lane,
        diaulos=diaulos,
        source_app=source_app,
        source_window=source_window,
        trace_path=trace_path,
        capture_profile=capture_profile,
        capture_command=capture_command,
        capture_mode=capture_mode,
        capture_rect=capture_rect,
        window_id=window_id,
        display_id=display_id,
    )
    started_at = now()
    capture_cwd = None if _is_global_capture_command(command) else Path(perceptasia_root).expanduser()
    capture = popen(command, cwd=capture_cwd)
    try:
        sleep(pre_hammer_delay_seconds)
        if retarget_during_dismiss_repeats:
            retarget_results = retarget_driver(
                repeats=retarget_during_dismiss_repeats,
                open_dwell_seconds=open_dwell_seconds,
                dismiss_retarget_delay_seconds=dismiss_retarget_delay_seconds,
                reopen_dwell_seconds=reopen_dwell_seconds,
                cycle_pause_seconds=cycle_pause_seconds,
                trace_path=trace_path,
                open_ready_timeout_seconds=open_ready_timeout_seconds,
                open_ready_poll_interval_seconds=open_ready_poll_interval_seconds,
            )
            stimulus = {
                "mode": "retarget-during-dismiss",
                "repeats": retarget_during_dismiss_repeats,
                "open_dwell_seconds": open_dwell_seconds,
                "dismiss_retarget_delay_seconds": dismiss_retarget_delay_seconds,
                "reopen_dwell_seconds": reopen_dwell_seconds,
                "cycle_pause_seconds": cycle_pause_seconds,
                "open_ready_timeout_seconds": open_ready_timeout_seconds,
                "open_ready_poll_interval_seconds": open_ready_poll_interval_seconds,
                "open_ready_gate": (
                    "fresh overlay.show.begin followed by overlay.visual_ready.push "
                    "with matching generation, visible text, and body_ready"
                ),
                "retarget_gate_results": retarget_results or [],
            }
        else:
            hammer_driver(
                count=hammer_toggles,
                interval_seconds=toggle_interval_seconds,
            )
            stimulus = {
                "mode": "hammer",
                "toggle_count": hammer_toggles,
                "toggle_interval_seconds": toggle_interval_seconds,
            }
        return_code = capture.wait()
    except BaseException:
        capture.terminate()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    ended_at = now()
    trace_events = collect_trace_events(trace_path, started_at=started_at, ended_at=ended_at)
    return write_witness_index(
        output_dir=output_dir,
        trace_path=trace_path,
        started_at=started_at,
        ended_at=ended_at,
        command=command,
        trace_events=trace_events,
        stimulus=stimulus,
        capture_profile=capture_profile,
    )


def run_witness_window(
    *,
    trace_path: str | Path,
    output_dir: str | Path,
    perceptasia_root: str | Path = DEFAULT_PERCEPTASIA_ROOT,
    duration_seconds: float = 8.0,
    fps: float | None = None,
    capture_profile: str = DEFAULT_PASSIVE_CAPTURE_PROFILE,
    lane: str = DEFAULT_LANE,
    diaulos: str = DEFAULT_DIAULOS,
    source_app: str = DEFAULT_SOURCE_APP,
    source_window: str = DEFAULT_SOURCE_WINDOW,
    capture_command: str | Path | None = None,
    capture_mode: str | None = None,
    capture_rect: str | None = None,
    window_id: str | int | None = None,
    display_id: str | int | None = None,
    stimulus: dict[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    if fps is None:
        fps = default_fps_for_capture_profile(capture_profile)
    count = capture_count_for_window(duration_seconds, fps)
    interval = capture_interval_for_fps(fps)
    command = build_retina_lasso_command(
        output_dir=output_dir,
        count=count,
        interval_seconds=interval,
        lane=lane,
        diaulos=diaulos,
        source_app=source_app,
        source_window=source_window,
        trace_path=trace_path,
        capture_profile=capture_profile,
        capture_command=capture_command,
        capture_mode=capture_mode,
        capture_rect=capture_rect,
        window_id=window_id,
        display_id=display_id,
    )
    started_at = now()
    capture_cwd = None if _is_global_capture_command(command) else Path(perceptasia_root).expanduser()
    runner(command, cwd=capture_cwd, check=True)
    ended_at = now()
    trace_events = collect_trace_events(trace_path, started_at=started_at, ended_at=ended_at)
    return write_witness_index(
        output_dir=output_dir,
        trace_path=trace_path,
        started_at=started_at,
        ended_at=ended_at,
        command=command,
        trace_events=trace_events,
        stimulus=stimulus,
        capture_profile=capture_profile,
    )


def run_trace_triggered_witness(
    *,
    trace_path: str | Path,
    output_dir: str | Path,
    perceptasia_root: str | Path = DEFAULT_PERCEPTASIA_ROOT,
    watch_timeout_seconds: float = 3600.0,
    event_capture_duration_seconds: float = 1.5,
    poll_interval_seconds: float = 0.05,
    max_captures: int = 48,
    max_trigger_lag_seconds: float = DEFAULT_TRACE_TRIGGER_MAX_LAG_SECONDS,
    fps: float | None = None,
    capture_profile: str = DEFAULT_STRESS_CAPTURE_PROFILE,
    trigger_events: set[str] | None = None,
    lane: str = DEFAULT_LANE,
    diaulos: str = DEFAULT_DIAULOS,
    source_app: str = DEFAULT_SOURCE_APP,
    source_window: str = DEFAULT_SOURCE_WINDOW,
    capture_command: str | Path | None = None,
    capture_mode: str | None = None,
    capture_rect: str | None = None,
    window_id: str | int | None = None,
    display_id: str | int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    now: Callable[[], datetime] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Path:
    if watch_timeout_seconds <= 0:
        raise ValueError("watch_timeout_seconds must be positive")
    if event_capture_duration_seconds <= 0:
        raise ValueError("event_capture_duration_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if max_captures <= 0:
        raise ValueError("max_captures must be positive")
    if max_trigger_lag_seconds <= 0:
        raise ValueError("max_trigger_lag_seconds must be positive")

    triggers = set(trigger_events or DEFAULT_TRACE_TRIGGER_EVENTS)
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    started_at = now()
    offset = Path(trace_path).expanduser().stat().st_size if Path(trace_path).expanduser().exists() else 0
    deadline = monotonic() + watch_timeout_seconds
    captures: list[dict[str, Any]] = []
    skipped_triggers: list[dict[str, Any]] = []

    while monotonic() < deadline and len(captures) < max_captures:
        offset, events = read_trace_events_from_offset(trace_path, offset=offset)
        for event in events:
            event_name = str(event.get("event", ""))
            if event_name not in triggers:
                continue
            trigger_seen_at = now()
            try:
                trigger_lag_seconds = (
                    trigger_seen_at - _parse_trace_timestamp(str(event["timestamp"]))
                ).total_seconds()
            except Exception:
                trigger_lag_seconds = 0.0
            if trigger_lag_seconds > max_trigger_lag_seconds:
                skipped_triggers.append(
                    {
                        "trigger_event": event,
                        "skipped_at": _format_instant(trigger_seen_at),
                        "trigger_lag_seconds": trigger_lag_seconds,
                        "max_trigger_lag_seconds": max_trigger_lag_seconds,
                        "reason": "stale_trigger_capture_suppressed",
                    }
                )
                continue
            capture_index = len(captures) + 1
            capture_output = root / trace_event_output_slug(event, index=capture_index)
            index_path = run_witness_window(
                trace_path=trace_path,
                output_dir=capture_output,
                perceptasia_root=perceptasia_root,
                duration_seconds=event_capture_duration_seconds,
                fps=fps,
                capture_profile=capture_profile,
                lane=lane,
                diaulos=diaulos,
                source_app=source_app,
                source_window=source_window,
                capture_command=capture_command,
                capture_mode=capture_mode,
                capture_rect=capture_rect,
                window_id=window_id,
                display_id=display_id,
                stimulus={
                    "mode": "trace-triggered",
                    "trigger_event": event,
                    "trigger_events": sorted(triggers),
                },
                runner=runner,
                now=now,
            )
            captures.append(
                {
                    "trigger_event": event,
                    "trigger_seen_at": _format_instant(trigger_seen_at),
                    "trigger_lag_seconds": trigger_lag_seconds,
                    "output_dir": str(capture_output),
                    "witness_index": str(index_path),
                }
            )
            if len(captures) >= max_captures:
                break
        if len(captures) >= max_captures:
            break
        sleep(poll_interval_seconds)

    ended_at = now()
    watch_index = {
        "schema": "spoke.retina_lasso_trace_trigger_watch.v1",
        "started_at": _format_instant(started_at),
        "ended_at": _format_instant(ended_at),
        "trace_path": str(Path(trace_path).expanduser()),
        "trigger_events": sorted(triggers),
        "capture_count": len(captures),
        "skipped_trigger_count": len(skipped_triggers),
        "max_captures": max_captures,
        "max_trigger_lag_seconds": max_trigger_lag_seconds,
        "watch_timeout_seconds": watch_timeout_seconds,
        "event_capture_duration_seconds": event_capture_duration_seconds,
        "capture_profile": capture_profile,
        "captures": captures,
        "skipped_triggers": skipped_triggers,
        "uncertainty": [
            "Trace-triggered capture starts after the triggering trace receipt, so it is not a pre-roll witness.",
            "Trace triggers older than max_trigger_lag_seconds are recorded as skipped instead of stale visual evidence.",
            "A single-frame flash before the first screencapture call can still be missed.",
        ],
    }
    watch_index_path = root / "watch-index.json"
    watch_index_path.write_text(json.dumps(watch_index, indent=2) + "\n", encoding="utf-8")
    return watch_index_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Retina Lasso capture window and index it against a Spoke trace."
    )
    parser.add_argument("--trace", required=True, help="Command overlay JSONL trace to align with captures.")
    parser.add_argument("--output-dir", required=True, help="Directory for Retina Lasso frames and witness index.")
    parser.add_argument("--perceptasia-root", default=str(DEFAULT_PERCEPTASIA_ROOT))
    parser.add_argument("--duration", type=float, default=8.0, help="Capture window length in seconds.")
    parser.add_argument(
        "--fps",
        type=float,
        help=(
            "Requested still capture cadence. Defaults to the selected capture profile "
            "(6fps low-perturbation, 15fps stress)."
        ),
    )
    parser.add_argument(
        "--capture-profile",
        choices=("low-perturbation", "stress"),
        default=None,
        help=(
            "Witness pressure profile. Passive/manual windows default to low-perturbation; "
            "hammer or launch-driven windows default to stress."
        ),
    )
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--diaulos", default=DEFAULT_DIAULOS)
    parser.add_argument("--source-app", default=DEFAULT_SOURCE_APP)
    parser.add_argument("--source-window", default=DEFAULT_SOURCE_WINDOW)
    parser.add_argument(
        "--capture-command",
        help=(
            "Optional global capture command. Defaults to global-witness-capture "
            "or epistaxis-global-witness-capture when installed; falls back to "
            "legacy uv-run perceptasia-screen-capture."
        ),
    )
    parser.add_argument(
        "--capture-mode",
        choices=("full-screen", "display", "rect", "window"),
        help="Global witness capture mode. Use rect/window/display to avoid full-screen contamination.",
    )
    parser.add_argument(
        "--capture-rect",
        help="Rectangle for --capture-mode rect, formatted as x,y,width,height in screen pixels.",
    )
    parser.add_argument("--window-id", help="Window id for --capture-mode window.")
    parser.add_argument("--display-id", help="Display id for --capture-mode display.")
    parser.add_argument(
        "--watch-trace",
        action="store_true",
        help="Watch the trace for lifecycle trigger events and capture short burst windows.",
    )
    parser.add_argument(
        "--watch-timeout",
        type=float,
        default=3600.0,
        help="Maximum seconds to keep a trace-triggered witness sidecar alive.",
    )
    parser.add_argument(
        "--event-capture-duration",
        type=float,
        default=1.5,
        help="Seconds of visual evidence to capture after each trace trigger.",
    )
    parser.add_argument(
        "--watch-poll-interval",
        type=float,
        default=0.05,
        help="Seconds between trace polls in watch mode.",
    )
    parser.add_argument(
        "--watch-max-captures",
        type=int,
        default=48,
        help="Maximum trigger capture bursts in one watch-mode sidecar.",
    )
    parser.add_argument(
        "--max-trigger-lag",
        type=float,
        default=DEFAULT_TRACE_TRIGGER_MAX_LAG_SECONDS,
        help=(
            "Maximum seconds between a trace trigger timestamp and starting its capture burst. "
            "Older triggers are recorded as skipped stale-trigger receipts."
        ),
    )
    parser.add_argument(
        "--trigger-event",
        action="append",
        dest="trigger_events",
        help=(
            "Trace event that should trigger a short capture burst. "
            "May be supplied more than once; defaults to show/dismiss/fade/retarget lifecycle edges."
        ),
    )
    parser.add_argument(
        "--hammer-toggles",
        type=int,
        default=0,
        help="If greater than zero, start capture and synthesize this many Spoke toggle chords.",
    )
    parser.add_argument(
        "--toggle-interval",
        type=float,
        default=0.18,
        help="Seconds between synthesized toggle chords in hammer mode.",
    )
    parser.add_argument(
        "--pre-hammer-delay",
        type=float,
        default=0.35,
        help="Seconds after capture starts before hammer-mode input begins.",
    )
    parser.add_argument(
        "--retarget-during-dismiss-repeats",
        type=int,
        default=0,
        help="Run a deterministic open/dismiss/reopen-during-dismiss pattern this many times.",
    )
    parser.add_argument(
        "--open-dwell",
        type=float,
        default=0.75,
        help="Seconds to hold the opened state before dismissing in retarget-during-dismiss mode.",
    )
    parser.add_argument(
        "--dismiss-retarget-delay",
        type=float,
        default=0.08,
        help="Seconds after dismiss begins before sending the reopen chord in retarget-during-dismiss mode.",
    )
    parser.add_argument(
        "--reopen-dwell",
        type=float,
        default=0.75,
        help="Seconds to hold the reopened state before final dismiss in retarget-during-dismiss mode.",
    )
    parser.add_argument(
        "--cycle-pause",
        type=float,
        default=0.2,
        help="Seconds between retarget-during-dismiss cycles.",
    )
    parser.add_argument(
        "--open-ready-timeout",
        type=float,
        default=2.0,
        help="Maximum seconds to wait for a trace-certified open state before dismissing.",
    )
    parser.add_argument(
        "--open-ready-poll-interval",
        type=float,
        default=0.025,
        help="Seconds between trace polls while waiting for a trace-certified open state.",
    )
    parser.add_argument(
        "--launch-target",
        help=(
            "Optional explicit Spoke launcher target to start before capture. "
            "This uses scripts/launch-target.sh and may replace the current live Spoke process."
        ),
    )
    parser.add_argument("--launch-wait", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    stimulus_mode = bool(
        args.watch_trace or args.hammer_toggles or args.retarget_during_dismiss_repeats or args.launch_target
    )
    capture_profile = _capture_profile_from_cli(
        args.capture_profile
        or (DEFAULT_STRESS_CAPTURE_PROFILE if stimulus_mode else DEFAULT_PASSIVE_CAPTURE_PROFILE)
    )
    if args.watch_trace:
        index_path = run_trace_triggered_witness(
            trace_path=args.trace,
            output_dir=args.output_dir,
            perceptasia_root=args.perceptasia_root,
            watch_timeout_seconds=args.watch_timeout,
            event_capture_duration_seconds=args.event_capture_duration,
            poll_interval_seconds=args.watch_poll_interval,
            max_captures=args.watch_max_captures,
            max_trigger_lag_seconds=args.max_trigger_lag,
            fps=args.fps,
            capture_profile=capture_profile,
            trigger_events=set(args.trigger_events) if args.trigger_events else None,
            lane=args.lane,
            diaulos=args.diaulos,
            source_app=args.source_app,
            source_window=args.source_window,
            capture_command=args.capture_command,
            capture_mode=args.capture_mode,
            capture_rect=args.capture_rect,
            window_id=args.window_id,
            display_id=args.display_id,
        )
    elif stimulus_mode:
        index_path = run_autonomous_hammer_witness(
            trace_path=args.trace,
            output_dir=args.output_dir,
            repo_root=Path(__file__).resolve().parents[1],
            perceptasia_root=args.perceptasia_root,
            duration_seconds=args.duration,
            fps=args.fps,
            capture_profile=capture_profile,
            hammer_toggles=args.hammer_toggles,
            toggle_interval_seconds=args.toggle_interval,
            pre_hammer_delay_seconds=args.pre_hammer_delay,
            retarget_during_dismiss_repeats=args.retarget_during_dismiss_repeats,
            open_dwell_seconds=args.open_dwell,
            dismiss_retarget_delay_seconds=args.dismiss_retarget_delay,
            reopen_dwell_seconds=args.reopen_dwell,
            cycle_pause_seconds=args.cycle_pause,
            open_ready_timeout_seconds=args.open_ready_timeout,
            open_ready_poll_interval_seconds=args.open_ready_poll_interval,
            lane=args.lane,
            diaulos=args.diaulos,
            source_app=args.source_app,
            source_window=args.source_window,
            capture_command=args.capture_command,
            capture_mode=args.capture_mode,
            capture_rect=args.capture_rect,
            window_id=args.window_id,
            display_id=args.display_id,
            launch_target=args.launch_target,
            launch_wait_seconds=args.launch_wait,
        )
    else:
        index_path = run_witness_window(
            trace_path=args.trace,
            output_dir=args.output_dir,
            perceptasia_root=args.perceptasia_root,
            duration_seconds=args.duration,
            fps=args.fps,
            capture_profile=capture_profile,
            lane=args.lane,
            diaulos=args.diaulos,
            source_app=args.source_app,
            source_window=args.source_window,
            capture_command=args.capture_command,
            capture_mode=args.capture_mode,
            capture_rect=args.capture_rect,
            window_id=args.window_id,
            display_id=args.display_id,
        )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
