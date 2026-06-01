"""Agent-owned visual witness recipes for live Spoke surfaces."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retina_lasso_witness import (
    DEFAULT_PASSIVE_CAPTURE_PROFILE,
    INDEX_NAME,
    build_evidence_split,
    build_retina_lasso_command,
    capture_count_for_window,
    capture_interval_for_fps,
    collect_trace_events,
    default_fps_for_capture_profile,
    write_witness_control_action,
)


DEFAULT_OUTPUT_ROOT = Path("/tmp/spoke-gold-durable-carbon-witnesses")
DEFAULT_TRACE_PATH = Path("/tmp/spoke-command-overlay-trace.jsonl")
DEFAULT_DIAULOS = "Warpstorm Pit Boss"
DEFAULT_LANE = "warpstorm-pit-boss"
DEFAULT_SOURCE_APP = "Spoke"
DEFAULT_LOG_PATHS = (
    Path.home() / "Library" / "Logs" / "spoke-main-launch.log",
    Path.home() / "Library" / "Logs" / "spoke-launch-target.log",
)


class VisualWitnessError(RuntimeError):
    """Raised after a witness writes a failure index for a false-closure condition."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_instant(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_slug(now: Callable[[], datetime] = _utc_now) -> str:
    return now().strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class VisualWitnessRecipe:
    name: str
    source_window: str
    show_action: str
    hide_action: str
    default_duration_seconds: float
    default_capture_profile: str = DEFAULT_PASSIVE_CAPTURE_PROFILE
    annotate_throughglass: bool = False


@dataclass(frozen=True)
class VisualWitnessPlan:
    recipe: VisualWitnessRecipe
    repo_root: Path
    output_dir: Path
    control_path: Path
    trace_path: Path
    command: list[str]
    show_action: str
    hide_action: str
    capture_profile: str
    fps: float
    duration_seconds: float
    lane: str
    diaulos: str
    source_app: str
    source_window: str


RECIPES: dict[str, VisualWitnessRecipe] = {
    "assistant-overlay-live": VisualWitnessRecipe(
        name="assistant-overlay-live",
        source_window="Command Overlay",
        show_action="show_command_overlay",
        hide_action="hide_command_overlay",
        default_duration_seconds=8.0,
    ),
    "throughglass-live": VisualWitnessRecipe(
        name="throughglass-live",
        source_window="Perceptasia Throughglass / Assistant Overlay",
        show_action="show_perceptasia_throughglass",
        hide_action="hide_perceptasia_throughglass",
        default_duration_seconds=10.0,
        annotate_throughglass=True,
    ),
}


def load_worktree_env(path: str | Path) -> dict[str, str]:
    env_path = Path(path).expanduser()
    if env_path.is_dir():
        env_path = env_path / ".spoke-smoke-env"
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _merged_env(repo_root: Path, env: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(load_worktree_env(repo_root))
    merged.update(os.environ)
    merged.update(dict(env or {}))
    return merged


def _recipe(name: str) -> VisualWitnessRecipe:
    try:
        return RECIPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown visual witness recipe: {name}") from exc


def build_visual_witness_plan(
    recipe_name: str,
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    duration_seconds: float | None = None,
    fps: float | None = None,
    capture_profile: str | None = None,
    control_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    lane: str = DEFAULT_LANE,
    diaulos: str = DEFAULT_DIAULOS,
    source_app: str = DEFAULT_SOURCE_APP,
    capture_command: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> VisualWitnessPlan:
    recipe = _recipe(recipe_name)
    root = Path(repo_root).expanduser().resolve()
    effective_env = _merged_env(root, env)
    resolved_control = Path(
        control_path
        or effective_env.get("SPOKE_WITNESS_CONTROL_PATH")
        or effective_env.get("SPOKE_RETINA_LASSO_TOGGLE_CONTROL_PATH")
        or root / ".spoke-witness-control.jsonl"
    ).expanduser()
    resolved_trace = Path(
        trace_path
        or effective_env.get("SPOKE_COMMAND_OVERLAY_TRACE_PATH")
        or DEFAULT_TRACE_PATH
    ).expanduser()
    resolved_duration = duration_seconds or recipe.default_duration_seconds
    resolved_profile = (capture_profile or recipe.default_capture_profile).replace("-", "_")
    resolved_fps = fps if fps is not None else default_fps_for_capture_profile(resolved_profile)
    resolved_output = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path(output_root).expanduser() / f"{recipe.name}-{_timestamp_slug(now)}"
    )
    command = build_retina_lasso_command(
        output_dir=resolved_output,
        count=capture_count_for_window(resolved_duration, resolved_fps),
        interval_seconds=capture_interval_for_fps(resolved_fps),
        lane=lane,
        diaulos=diaulos,
        source_app=source_app,
        source_window=recipe.source_window,
        trace_path=resolved_trace,
        capture_profile=resolved_profile,
        capture_command=capture_command,
    )
    return VisualWitnessPlan(
        recipe=recipe,
        repo_root=root,
        output_dir=resolved_output,
        control_path=resolved_control,
        trace_path=resolved_trace,
        command=command,
        show_action=recipe.show_action,
        hide_action=recipe.hide_action,
        capture_profile=resolved_profile,
        fps=resolved_fps,
        duration_seconds=resolved_duration,
        lane=lane,
        diaulos=diaulos,
        source_app=source_app,
        source_window=recipe.source_window,
    )


def _capture_cwd(command: Sequence[str]) -> Path | None:
    executable = Path(command[0]).name if command else ""
    if executable in {"global-witness-capture", "epistaxis-global-witness-capture"}:
        return None
    return Path("/private/tmp/perceptasia-codex-screen-slice-smoke-loop-0521")


def _read_manifest(output_dir: Path, manifest_name: str = "manifest.json") -> tuple[bool, int, str]:
    manifest_path = output_dir / manifest_name
    if not manifest_path.exists():
        return False, 0, str(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False, 0, str(manifest_path)
    return True, len(manifest.get("frames", [])), str(manifest_path)


def _write_visual_index(
    plan: VisualWitnessPlan,
    *,
    started_at: datetime,
    ended_at: datetime,
    trace_events: list[dict[str, Any]],
    show_receipt: dict[str, Any],
    hide_receipt: dict[str, Any] | None,
    command: Sequence[str],
) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_loaded, frame_count, manifest_path = _read_manifest(plan.output_dir)
    status = "passed" if manifest_loaded and frame_count > 0 else "failed"
    failure_phase = None if status == "passed" else "capture_frames"
    payload = {
        "schema": "spoke.visual_witness.v1",
        "status": status,
        "failure_phase": failure_phase,
        "recipe": plan.recipe.name,
        "started_at": _format_instant(started_at),
        "ended_at": _format_instant(ended_at),
        "retina_lasso_manifest": manifest_path,
        "retina_lasso_manifest_loaded": manifest_loaded,
        "frame_count": frame_count,
        "trace_path": str(plan.trace_path),
        "trace_event_count": len(trace_events),
        "trace_events": trace_events,
        "evidence_split": build_evidence_split(
            manifest_loaded=manifest_loaded,
            frame_count=frame_count,
            trace_event_count=len(trace_events),
            capture_profile=plan.capture_profile,
        ),
        "effective_route": {
            "repo_root": str(plan.repo_root),
            "control_path": str(plan.control_path),
            "trace_path": str(plan.trace_path),
            "capture_command": str(command[0]) if command else None,
            "lane": plan.lane,
            "diaulos": plan.diaulos,
            "source_app": plan.source_app,
            "source_window": plan.source_window,
        },
        "command": list(command),
        "stimulus": {
            "mode": "agent-owned-visual-witness",
            "show": show_receipt,
            "hide": hide_receipt,
        },
        "uncertainty": [
            "This witness proves the live route accepted an agent-owned stimulus only when matching trace receipts appear.",
            "Visual stills are evidence, not operator approval.",
            "Retina Lasso capture can perturb animation timing; route receipts and frames must be read together.",
        ],
    }
    index_path = plan.output_dir / INDEX_NAME
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_path


def run_visual_witness_recipe(
    recipe_name: str,
    *,
    repo_root: str | Path = ".",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    output_dir: str | Path | None = None,
    duration_seconds: float | None = None,
    fps: float | None = None,
    capture_profile: str | None = None,
    control_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    lane: str = DEFAULT_LANE,
    diaulos: str = DEFAULT_DIAULOS,
    source_app: str = DEFAULT_SOURCE_APP,
    capture_command: str | Path | None = None,
    pre_stimulus_delay_seconds: float = 0.25,
    settle_seconds: float = 0.25,
    cleanup: bool = True,
    env: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    plan = build_visual_witness_plan(
        recipe_name,
        repo_root=repo_root,
        output_root=output_root,
        output_dir=output_dir,
        duration_seconds=duration_seconds,
        fps=fps,
        capture_profile=capture_profile,
        control_path=control_path,
        trace_path=trace_path,
        lane=lane,
        diaulos=diaulos,
        source_app=source_app,
        capture_command=capture_command,
        env=env,
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()
    capture = popen(plan.command, cwd=_capture_cwd(plan.command))
    show_receipt: dict[str, Any] | None = None
    hide_receipt: dict[str, Any] | None = None
    try:
        sleep(pre_stimulus_delay_seconds)
        show_receipt = write_witness_control_action(
            plan.control_path,
            action=plan.show_action,
            nonce=f"{plan.recipe.name}-show",
            now=now,
        )
        sleep(max(0.0, plan.duration_seconds - pre_stimulus_delay_seconds - settle_seconds))
        return_code = capture.wait()
    except BaseException:
        capture.terminate()
        raise
    ended_at = now()
    if cleanup:
        hide_receipt = write_witness_control_action(
            plan.control_path,
            action=plan.hide_action,
            nonce=f"{plan.recipe.name}-hide",
            now=now,
        )
        sleep(settle_seconds)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, plan.command)
    trace_events = collect_trace_events(
        plan.trace_path,
        started_at=started_at,
        ended_at=ended_at,
    )
    index_path = _write_visual_index(
        plan,
        started_at=started_at,
        ended_at=ended_at,
        trace_events=trace_events,
        show_receipt=show_receipt or {},
        hide_receipt=hide_receipt,
        command=plan.command,
    )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    if index_payload.get("status") != "passed":
        raise VisualWitnessError(
            f"visual witness produced no captured frames; failure index: {index_path}"
        )
    if plan.recipe.annotate_throughglass:
        from .perceptasia_throughglass_witness import annotate_throughglass_contract

        annotate_throughglass_contract(index_path, log_paths=list(DEFAULT_LOG_PATHS))
    return index_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent-owned Spoke visual witness recipe.")
    parser.add_argument("recipe", choices=sorted(RECIPES))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--duration", type=float, dest="duration_seconds")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--capture-profile", choices=("low-perturbation", "stress"))
    parser.add_argument("--control-path")
    parser.add_argument("--trace-path")
    parser.add_argument("--capture-command")
    parser.add_argument("--pre-stimulus-delay", type=float, default=0.25)
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--no-cleanup", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    index_path = run_visual_witness_recipe(
        args.recipe,
        repo_root=args.repo_root,
        output_root=args.output_root,
        output_dir=args.output_dir,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        capture_profile=args.capture_profile,
        control_path=args.control_path,
        trace_path=args.trace_path,
        capture_command=args.capture_command,
        pre_stimulus_delay_seconds=args.pre_stimulus_delay,
        settle_seconds=args.settle,
        cleanup=not args.no_cleanup,
    )
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
