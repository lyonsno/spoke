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
            val = val.strip().strip('"').strip("'")
            if key:
                child_env[key] = val
    except Exception:
        pass

# Machine-wide secrets (populated once per box from the example
# template at scripts/secrets.env.example).
secrets_env = Path.home() / ".config" / "spoke" / "secrets.env"
_apply_env_file(secrets_env)

# Per-worktree overrides — win over machine-wide secrets.
smoke_env = repo_root / ".spoke-smoke-env"
_apply_env_file(smoke_env)
if target is not None:
    child_env["SPOKE_LAUNCH_TARGET_ID"] = target.get("id", "")

uv_bin = _resolve_uv_bin(repo_root)

with log_file.open("a", encoding="utf-8") as log:
    try:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"Launcher PID {os.getpid()} (PPID {os.getppid()})\n")
        log.write(f"Launch target: {target_source}\n")
        log.write(f"Repo root: {repo_root}\n")
        if is_fallback:
            log.write("WARNING: using fallback — registry target was missing or invalid\n")
        log.flush()

        python_exe = repo_root / ".venv" / "bin" / "python"
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
    except Exception:
        traceback.print_exc(file=log)
        log.flush()
        raise SystemExit(1)
PY
