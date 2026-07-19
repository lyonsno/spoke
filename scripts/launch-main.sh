#!/bin/bash
# Launch spoke from the launcher registry's selected target.
# Bind to Ctrl+Opt+Cmd+Space via macOS Shortcuts or Automator.
#
# Architecture:
# 1. Read ~/.config/spoke/launch_targets.json → selected target → path
# 2. If path is valid and has a .venv: launch from there
# 3. If path is bad: fall back to the checkout containing this script
#    and flash red to indicate fallback
# 4. Kill any existing spoke instance before launching

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FALLBACK_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGETS_FILE="${HOME}/.config/spoke/launch_targets.json"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-main-launch.log"

mkdir -p "$LOG_DIR"

export FALLBACK_REPO_ROOT TARGETS_FILE LOG_FILE

/usr/bin/python3 - <<'PY'
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional


def _resolve_uv_bin(repo_root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    env_uv_bin = os.environ.get("UV_BIN")
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


def _read_selected_target(targets_file: Path) -> Optional[dict]:
    """Read the selected target from the launcher registry."""
    try:
        data = json.loads(targets_file.read_text(encoding="utf-8"))
        selected_id = data.get("selected")
        if not selected_id:
            return None
        for target in data.get("targets", []):
            if target.get("id") == selected_id:
                return target
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def _flash_notification(title: str, message: str, sound: str = "Basso") -> None:
    subprocess.run(
        ["osascript", "-e",
         f'display notification "{message}" with title "{title}"'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if sound:
        subprocess.Popen(
            ["afplay", f"/System/Library/Sounds/{sound}.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


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
        witness_route = "perceptasia-throughglass-pixel"
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
        witness_route = "command-overlay-retina-lasso"
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
    if throughglass_witness and _env_flag(child_env, "SPOKE_RETINA_LASSO_WATCH_TRACE"):
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
    log.write(
        "Retina Lasso auto witness route: "
        f"{witness_route} "
        f"(SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE={child_env.get('SPOKE_PERCEPTASIA_THROUGHGLASS_SMOKE', '')!r})\n"
    )
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


targets_file = Path(os.environ["TARGETS_FILE"])
fallback_repo_root = Path(os.environ["FALLBACK_REPO_ROOT"])
log_file = Path(os.environ["LOG_FILE"])

# Step 1: Try the registry
target = _read_selected_target(targets_file)
repo_root = None
target_source = "fallback"
is_fallback = False

if target is not None:
    candidate = Path(target["path"])
    if candidate.is_dir():
        repo_root = candidate
        target_source = f"registry:{target.get('id', '?')} ({target.get('label', '')})"
    else:
        _flash_notification(
            "Spoke Fallback",
            f"Target gone: {candidate.name}. Falling back to script checkout.",
        )
        is_fallback = True
else:
    _flash_notification(
        "Spoke Fallback",
        "No registry target selected. Falling back to script checkout.",
    )
    is_fallback = True

if repo_root is None:
    repo_root = fallback_repo_root
    target_source = f"fallback:{fallback_repo_root}"

# Build child env: clear inherited overrides, then apply machine-wide
# ~/.config/spoke/secrets.env (sourced first so per-worktree values can
# override it), then per-worktree .spoke-smoke-env so the worktree's
# own values win (matches launch-target.sh).
#
# The secrets file exists so that Automator-launched spoke processes
# receive API keys that live only in the user's shell profile
# (e.g. ~/.zshenv). Automator runs this launcher under non-interactive
# /bin/bash which does not source any zsh profile, so without this
# block secrets placed in shell profiles never reach spoke.
# See ~/dev/epistaxis/system/secrets.md for the cross-project pattern.
child_env = os.environ.copy()
child_env.pop("SPOKE_PREVIEW_MODEL", None)
child_env.pop("SPOKE_TRANSCRIPTION_MODEL", None)
child_env.pop("SPOKE_WHISPER_MODEL", None)
child_env.pop("SPOKE_VENV_PYTHON", None)
child_env.pop("PYTHONPATH", None)

def _apply_env_file(path: Path) -> None:
    """Apply KEY=value (or 'export KEY=value') overrides from path into
    child_env. Silent no-op if the file is missing or unreadable —
    launching must not crash on a fresh box or a permission glitch."""
    if not path.is_file():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
                val = val[1:-1]
            if key:
                child_env[key] = os.path.expanduser(os.path.expandvars(val))
    except Exception:
        pass

# Machine-wide secrets (populated once per box from the example
# template at scripts/secrets.env.example).
secrets_env = Path.home() / ".config" / "spoke" / "secrets.env"
_apply_env_file(secrets_env)

# Per-worktree overrides — win over machine-wide secrets.
smoke_env = repo_root / ".spoke-smoke-env"
_apply_env_file(smoke_env)
target_env = target.get("env") if target is not None else None
effective_target_env = {}
if isinstance(target_env, dict):
    effective_target_env = {
        key.strip(): os.path.expanduser(os.path.expandvars(value))
        for key, value in target_env.items()
        if isinstance(key, str)
        and key.strip()
        and isinstance(value, str)
    }
    child_env.update(effective_target_env)
if target is not None:
    child_env["SPOKE_LAUNCH_TARGET_ID"] = target.get("id", "")

uv_bin = _resolve_uv_bin(repo_root)

with log_file.open("a", encoding="utf-8") as log:
    try:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"Launcher PID {os.getpid()} (PPID {os.getppid()})\n")
        log.write(f"Launch target: {target_source}\n")
        log.write(f"Repo root: {repo_root}\n")
        if effective_target_env:
            log.write(f"Target env override keys: {sorted(effective_target_env)}\n")
        if is_fallback:
            log.write("WARNING: using fallback — registry target was missing or invalid\n")
        log.flush()

        python_exe = Path(
            child_env.get("SPOKE_VENV_PYTHON", str(repo_root / ".venv" / "bin" / "python"))
        )
        if python_exe.is_file():
            command = [str(python_exe), "-m", "spoke"]
        elif uv_bin is not None:
            command = [str(uv_bin), "run", "--directory", str(repo_root), "python", "-m", "spoke"]
        else:
            log.write("No repo .venv Python found and UV launcher is unavailable.\n")
            _flash_notification("Spoke Launch Failed", "No Python environment found.", "Sosumi")
            raise SystemExit(1)

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
            target_id=target.get("id", "selected") if target is not None else "fallback",
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
