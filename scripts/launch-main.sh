#!/bin/bash
# Launch spoke main build. Bind to a hotkey via macOS Shortcuts or Automator.
# Kills any existing local python-based spoke instance first.
#
# If ~/.config/spoke/main-target exists, launch from the absolute repo/worktree
# path written there. Otherwise fall back to the checkout containing this script.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_TARGET_FILE="${HOME}/.config/spoke/main-target"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-main-launch.log"

mkdir -p "$LOG_DIR"

REPO_ROOT="$DEFAULT_REPO_ROOT"
TARGET_SOURCE="script checkout"

if [ -f "$MAIN_TARGET_FILE" ]; then
  IFS= read -r CONFIGURED_REPO_ROOT < "$MAIN_TARGET_FILE"
  if [ -z "$CONFIGURED_REPO_ROOT" ] || [ ! -d "$CONFIGURED_REPO_ROOT" ]; then
    osascript -e "display notification \"Target gone: $CONFIGURED_REPO_ROOT\" with title \"Spoke Main\" subtitle \"Set ~/.config/spoke/main-target\"" 2>/dev/null
    afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &
    exit 0
  fi
  REPO_ROOT="$CONFIGURED_REPO_ROOT"
  TARGET_SOURCE="~/.config/spoke/main-target"
fi

# Source per-target env overrides from the pinned main worktree if present.
# This keeps local main-smoke repairs in the target worktree instead of
# requiring edits to the shared launcher.
if [ -f "$REPO_ROOT/.spoke-smoke-env" ]; then
  # shellcheck source=/dev/null
  . "$REPO_ROOT/.spoke-smoke-env"
fi

{
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Launcher PID %d (PPID %d) invoked from %s\n' "$$" "$PPID" "$PWD"
  printf 'Launching Spoke main from %s (%s)\n' "$REPO_ROOT" "$TARGET_SOURCE"
} >>"$LOG_FILE"

export REPO_ROOT LOG_FILE
export VENV_PYTHON="${SPOKE_VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export UV_BIN="${UV_BIN:-}"
unset SPOKE_PREVIEW_MODEL
unset SPOKE_TRANSCRIPTION_MODEL
unset SPOKE_WHISPER_MODEL

/usr/bin/python3 - <<'PY'
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


repo_root = Path(os.environ["REPO_ROOT"])
log_file = Path(os.environ["LOG_FILE"])
python_exe = Path(os.environ.get("VENV_PYTHON", str(repo_root / ".venv" / "bin" / "python")))
uv_bin = _resolve_uv_bin(repo_root)
child_env = os.environ.copy()
child_env.pop("SPOKE_PREVIEW_MODEL", None)
child_env.pop("SPOKE_TRANSCRIPTION_MODEL", None)
child_env.pop("SPOKE_WHISPER_MODEL", None)

with log_file.open("a", encoding="utf-8") as log:
    try:
        if python_exe.is_file():
            command = [str(python_exe), "-m", "spoke"]
        elif uv_bin is not None:
            command = [str(uv_bin), "run", "--directory", str(repo_root), "python", "-m", "spoke"]
        else:
            log.write(
                "No repo .venv Python found and UV launcher is unavailable.\n"
            )
            log.flush()
            raise SystemExit(1)

        subprocess.run(
            ["pkill", "-TERM", "-f", "python.*spoke"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.5)
        log.write(f"Launcher PID context: pid={os.getpid()} ppid={os.getppid()}\n")
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
