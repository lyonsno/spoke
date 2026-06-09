#!/bin/bash
# Launch spoke from a named registry target and replace any currently running
# local python-based spoke process.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HELPER_REPO_ROOT="${HELPER_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGETS_FILE="${SPOKE_LAUNCH_TARGETS_PATH:-$HOME/.config/spoke/launch_targets.json}"
TARGET_ID="${1:-${TARGET_ID:-}}"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-launch-target.log"

mkdir -p "$LOG_DIR"

if [ -z "$TARGET_ID" ]; then
  osascript -e 'display notification "No launch target selected" with title "Spoke Launch Target"' 2>/dev/null
  afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &
  exit 0
fi

export HELPER_REPO_ROOT TARGETS_FILE TARGET_ID LOG_FILE

/usr/bin/python3 - <<'PY'
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

helper_repo_root = Path(os.environ["HELPER_REPO_ROOT"])
if str(helper_repo_root) not in sys.path:
    sys.path.insert(0, str(helper_repo_root))

from spoke.launch_targets import parse_env_overrides, resolve_launch_target


def _resolve_uv_bin(repo_root: Path, child_env: dict[str, str]) -> Optional[Path]:
    candidates: list[Path] = []
    env_uv_bin = child_env.get("UV_BIN")
    if env_uv_bin:
        candidates.append(Path(env_uv_bin))
    candidates.append(repo_root / ".venv" / "bin" / "uv")
    which_uv = shutil.which("uv")
    if which_uv:
        candidates.append(Path(which_uv))
    candidates.extend(
        [
            Path.home() / ".local" / "bin" / "uv",
            Path.home() / ".cargo" / "bin" / "uv",
            Path("/opt/homebrew/bin/uv"),
            Path("/usr/local/bin/uv"),
        ]
    )
    candidates.append(Path("/Users/noahlyons/.pyenv/shims/uv"))

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        if "/.pyenv/shims/" in candidate_str:
            probe = subprocess.run(
                [candidate_str, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode != 0:
                continue
        return candidate
    return None


def _env_flag(child_env: dict[str, str], name: str) -> bool:
    return child_env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_path_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return slug.strip("-") or "selected"


def _positive_int_env(child_env: dict[str, str], name: str) -> bool:
    try:
        return int(child_env.get(name, "0").strip() or "0") > 0
    except ValueError:
        return False


def _append_retina_lasso_stimulus_args(
    args: list[str],
    *,
    child_env: dict[str, str],
    log,
) -> bool:
    """Append capture-first stimulus args and return whether one was armed."""
    capture_first = False
    hammer_toggles = child_env.get("SPOKE_RETINA_LASSO_HAMMER_TOGGLES", "").strip()
    if _positive_int_env(child_env, "SPOKE_RETINA_LASSO_HAMMER_TOGGLES"):
        args.extend(["--hammer-toggles", hammer_toggles])
        args.extend(["--toggle-interval", child_env.get("SPOKE_RETINA_LASSO_TOGGLE_INTERVAL_SECONDS", "0.18")])
        capture_first = True

    retarget_repeats = child_env.get("SPOKE_RETINA_LASSO_RETARGET_DURING_DISMISS_REPEATS", "").strip()
    if _positive_int_env(child_env, "SPOKE_RETINA_LASSO_RETARGET_DURING_DISMISS_REPEATS"):
        args.extend(["--retarget-during-dismiss-repeats", retarget_repeats])
        args.extend(["--open-dwell", child_env.get("SPOKE_RETINA_LASSO_OPEN_DWELL_SECONDS", "0.75")])
        args.extend(
            [
                "--dismiss-retarget-delay",
                child_env.get("SPOKE_RETINA_LASSO_DISMISS_RETARGET_DELAY_SECONDS", "0.08"),
            ]
        )
        args.extend(["--reopen-dwell", child_env.get("SPOKE_RETINA_LASSO_REOPEN_DWELL_SECONDS", "0.75")])
        args.extend(["--cycle-pause", child_env.get("SPOKE_RETINA_LASSO_CYCLE_PAUSE_SECONDS", "0.2")])
        args.extend(["--open-ready-timeout", child_env.get("SPOKE_RETINA_LASSO_OPEN_READY_TIMEOUT_SECONDS", "2.0")])
        args.extend(
            [
                "--open-ready-poll-interval",
                child_env.get("SPOKE_RETINA_LASSO_OPEN_READY_POLL_INTERVAL_SECONDS", "0.025"),
            ]
        )
        capture_first = True

    if capture_first:
        args.extend(["--pre-hammer-delay", child_env.get("SPOKE_RETINA_LASSO_PRE_HAMMER_DELAY_SECONDS", "0.35")])
        if _env_flag(child_env, "SPOKE_RETINA_LASSO_WATCH_TRACE"):
            log.write(
                "Retina Lasso auto witness: capture-first stimulus armed; "
                "trace-trigger watch mode suppressed for this sidecar.\n"
            )
    return capture_first


def _start_retina_lasso_witness(
    *,
    repo_root: Path,
    target_id: str,
    python_exe: Path,
    uv_bin: Optional[Path],
    child_env: dict[str, str],
    log,
) -> None:
    """Start the optional low-perturbation visual witness sidecar."""
    if not _env_flag(child_env, "SPOKE_RETINA_LASSO_AUTO_WITNESS"):
        return

    trace_path = child_env.get("SPOKE_COMMAND_OVERLAY_TRACE_PATH", "").strip()
    if not trace_path:
        log.write("Retina Lasso auto witness skipped: SPOKE_COMMAND_OVERLAY_TRACE_PATH is unset.\n")
        return

    throughglass_witness = _env_flag(child_env, "SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE")
    script = repo_root / "scripts" / "command-overlay-retina-lasso-witness.py"
    if not throughglass_witness and not script.is_file():
        log.write(f"Retina Lasso auto witness skipped: witness script missing at {script}.\n")
        return

    perceptasia_root = Path(
        child_env.get(
            "SPOKE_RETINA_LASSO_PERCEPTASIA_ROOT",
            "/private/tmp/perceptasia-codex-screen-slice-smoke-loop-0521",
        )
    ).expanduser()

    output_root_key = (
        "SPOKE_PERCEPTASIA_THROUGHGLASS_WITNESS_OUTPUT_ROOT"
        if throughglass_witness
        else "SPOKE_RETINA_LASSO_OUTPUT_ROOT"
    )
    output_root = Path(
        child_env.get(output_root_key, child_env.get("SPOKE_RETINA_LASSO_OUTPUT_ROOT", "/tmp/spoke-retina-lasso-witnesses"))
    ).expanduser()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = output_root / f"{_safe_path_slug(target_id)}-{stamp}"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if throughglass_witness:
        args = [
            "-m",
            "spoke.perceptasia_throughglass_witness",
            "--trace",
            trace_path,
            "--output-dir",
            str(output_dir),
            "--duration",
            child_env.get("SPOKE_RETINA_LASSO_DURATION_SECONDS", "45"),
            "--capture-profile",
            child_env.get("SPOKE_RETINA_LASSO_CAPTURE_PROFILE", "low-perturbation"),
            "--lane",
            child_env.get("SPOKE_RETINA_LASSO_LANE", "warpstorm-pit-boss"),
            "--diaulos",
            child_env.get("SPOKE_RETINA_LASSO_DIAULOS", "Warpstorm Pit Boss"),
            "--source-app",
            child_env.get("SPOKE_RETINA_LASSO_SOURCE_APP", "Spoke"),
            "--source-window",
            child_env.get("SPOKE_RETINA_LASSO_SOURCE_WINDOW", "Perceptasia Throughglass"),
            "--allow-unproven",
        ]
    else:
        args = [
            str(script),
            "--trace",
            trace_path,
            "--output-dir",
            str(output_dir),
            "--perceptasia-root",
            str(perceptasia_root),
            "--duration",
            child_env.get("SPOKE_RETINA_LASSO_DURATION_SECONDS", "45"),
            "--capture-profile",
            child_env.get("SPOKE_RETINA_LASSO_CAPTURE_PROFILE", "low-perturbation"),
            "--lane",
            child_env.get("SPOKE_RETINA_LASSO_LANE", "warpstorm-pit-boss"),
            "--diaulos",
            child_env.get("SPOKE_RETINA_LASSO_DIAULOS", "Warpstorm Pit Boss"),
            "--source-app",
            child_env.get("SPOKE_RETINA_LASSO_SOURCE_APP", "Spoke"),
            "--source-window",
            child_env.get("SPOKE_RETINA_LASSO_SOURCE_WINDOW", "Command Overlay"),
        ]
    fps = child_env.get("SPOKE_RETINA_LASSO_FPS", "").strip()
    if fps:
        args.extend(["--fps", fps])
    capture_command = (
        child_env.get("SPOKE_RETINA_LASSO_CAPTURE_COMMAND", "").strip()
        or child_env.get("SPOKE_RETINA_LASSO_CAPTURE_BIN", "").strip()
        or child_env.get("GLOBAL_WITNESS_CAPTURE_COMMAND", "").strip()
        or child_env.get("GLOBAL_WITNESS_CAPTURE_BIN", "").strip()
    )
    if capture_command:
        args.extend(["--capture-command", capture_command])
    if not throughglass_witness:
        capture_mode = child_env.get("SPOKE_RETINA_LASSO_CAPTURE_MODE", "").strip()
        if capture_mode:
            args.extend(["--capture-mode", capture_mode])
        capture_rect = child_env.get("SPOKE_RETINA_LASSO_CAPTURE_RECT", "").strip()
        if capture_rect:
            args.extend(["--capture-rect", capture_rect])
        window_id = child_env.get("SPOKE_RETINA_LASSO_WINDOW_ID", "").strip()
        if window_id:
            args.extend(["--window-id", window_id])
        display_id = child_env.get("SPOKE_RETINA_LASSO_DISPLAY_ID", "").strip()
        if display_id:
            args.extend(["--display-id", display_id])
    capture_first_stimulus = False
    if not throughglass_witness:
        capture_first_stimulus = _append_retina_lasso_stimulus_args(args, child_env=child_env, log=log)
    if not throughglass_witness and _env_flag(child_env, "SPOKE_RETINA_LASSO_WATCH_TRACE") and not capture_first_stimulus:
        args.append("--watch-trace")
        args.extend(["--watch-timeout", child_env.get("SPOKE_RETINA_LASSO_WATCH_TIMEOUT_SECONDS", "7200")])
        args.extend(
            [
                "--event-capture-duration",
                child_env.get("SPOKE_RETINA_LASSO_EVENT_CAPTURE_DURATION_SECONDS", "1.5"),
            ]
        )
        args.extend(["--watch-max-captures", child_env.get("SPOKE_RETINA_LASSO_WATCH_MAX_CAPTURES", "96")])
        args.extend(["--max-trigger-lag", child_env.get("SPOKE_RETINA_LASSO_MAX_TRIGGER_LAG_SECONDS", "1.0")])

    if python_exe.is_file():
        command = [str(python_exe), *args]
    elif uv_bin is not None:
        command = [str(uv_bin), "run", "--directory", str(repo_root), "python", *args]
    else:
        log.write("Retina Lasso auto witness skipped: no Python or UV runner found.\n")
        return
    witness_env = child_env.copy()
    if uv_bin is not None:
        witness_env["UV_BIN"] = str(uv_bin)

    log.write(f"Retina Lasso auto witness output: {output_dir}\n")
    log.write(f"Retina Lasso auto witness command: {command!r}\n")
    log.flush()
    subprocess.Popen(
        command,
        cwd=repo_root,
        env=witness_env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )


target_id = os.environ["TARGET_ID"]
targets_file = Path(
    os.environ.get("TARGETS_FILE") or os.environ["SPOKE_LAUNCH_TARGETS_PATH"]
).expanduser()
log_file = Path(os.environ["LOG_FILE"])
target = resolve_launch_target(target_id, targets_file)

with log_file.open("a", encoding="utf-8") as log:
    try:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"Requested launch target: {target_id}\n")
        log.write(f"Launch target registry: {targets_file}\n")
        if target is None:
            log.write(f"Launch target not found: {target_id}\n")
            raise SystemExit(1)

        repo_root = Path(target["path"])
        if not repo_root.is_dir():
            log.write(f"Launch target path missing: {repo_root}\n")
            raise SystemExit(1)

        child_env = os.environ.copy()
        # Clear inherited runtime overrides so the target's own env wins
        child_env.pop("SPOKE_VENV_PYTHON", None)
        child_env.pop("PYTHONPATH", None)
        secrets_env = Path.home() / ".config/spoke/secrets.env"
        child_env.update(parse_env_overrides(secrets_env))
        child_env.update(parse_env_overrides(repo_root / ".spoke-smoke-env"))
        child_env["REPO_ROOT"] = str(repo_root)
        child_env["SPOKE_LAUNCH_TARGET_ID"] = target_id
        child_env.pop("SPOKE_PREVIEW_MODEL", None)
        child_env.pop("SPOKE_TRANSCRIPTION_MODEL", None)
        child_env.pop("SPOKE_WHISPER_MODEL", None)

        python_exe = Path(
            child_env.get("SPOKE_VENV_PYTHON", str(repo_root / ".venv" / "bin" / "python"))
        )
        uv_bin = _resolve_uv_bin(repo_root, child_env)
        if python_exe.is_file():
            command = [str(python_exe), "-m", "spoke"]
        elif uv_bin is not None:
            command = [str(uv_bin), "run", "--directory", str(repo_root), "python", "-m", "spoke"]
        else:
            log.write("No repo .venv Python found and UV launcher is unavailable.\n")
            raise SystemExit(1)

        lock_file = Path.home() / "Library" / "Logs" / ".spoke.lock"
        old_pid = None
        try:
            old_pid = int(lock_file.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            pass

        if old_pid is not None and old_pid != os.getpid():
            import signal as _sig
            try:
                os.kill(old_pid, _sig.SIGTERM)
                log.write(f"Launch target handoff: sent SIGTERM to pid {old_pid}\n")
                for _ in range(25):
                    time.sleep(0.2)
                    try:
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(old_pid, _sig.SIGKILL)
                    log.write(f"Launch target handoff: escalated to SIGKILL for pid {old_pid}\n")
            except (ProcessLookupError, PermissionError):
                pass
        lock_file.unlink(missing_ok=True)

        log.write("Launch target handoff: prior instance cleared.\n")
        log.write(f"Launcher child command: {command!r}\n")
        log.flush()

        subprocess.Popen(
            command,
            cwd=repo_root,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        _start_retina_lasso_witness(
            repo_root=repo_root,
            target_id=target_id,
            python_exe=python_exe,
            uv_bin=uv_bin,
            child_env=child_env,
            log=log,
        )
    except Exception:
        traceback.print_exc(file=log)
        log.flush()
        raise SystemExit(1)
PY
