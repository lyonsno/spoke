#!/bin/bash
# Launch spoke dev build. Bind to a hotkey via macOS Shortcuts or Automator.
# Lets the single-instance guard handle any existing instance.
#
# If ~/.config/spoke/dev-target exists, launch from the absolute repo/worktree
# path written there. Otherwise fall back to the checkout containing this script.
# This keeps the Automator binding stable while letting the actual launch target
# move to a fresh main/dev worktree.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_TARGET_FILE="${HOME}/.config/spoke/dev-target"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-dev-launch.log"
LOCK_FILE="${LOG_DIR}/.spoke.lock"

mkdir -p "$LOG_DIR"

REPO_ROOT="$DEFAULT_REPO_ROOT"
TARGET_SOURCE="script checkout"

if [ -f "$DEV_TARGET_FILE" ]; then
  IFS= read -r CONFIGURED_REPO_ROOT < "$DEV_TARGET_FILE"
  if [ -z "$CONFIGURED_REPO_ROOT" ] || [ ! -d "$CONFIGURED_REPO_ROOT" ]; then
    osascript -e "display notification \"Target gone: $CONFIGURED_REPO_ROOT\" with title \"Spoke Dev\" subtitle \"Set ~/.config/spoke/dev-target\"" 2>/dev/null
    afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &
    exit 0
  fi
  REPO_ROOT="$CONFIGURED_REPO_ROOT"
  TARGET_SOURCE="~/.config/spoke/dev-target"
fi

{
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Launcher PID %d (PPID %d) invoked from %s\n' "$$" "$PPID" "$PWD"
  printf 'Launching Spoke from %s (%s)\n' "$REPO_ROOT" "$TARGET_SOURCE"
} >>"$LOG_FILE"

OLD_PID=""
if [ -r "$LOCK_FILE" ]; then
  OLD_PID="$(tr -d '[:space:]' < "$LOCK_FILE")"
fi

if [[ "$OLD_PID" =~ ^[0-9]+$ ]]; then
  {
    printf 'Launcher preflight: observed lock-holder pid %s in %s\n' "$OLD_PID" "$LOCK_FILE"
    printf 'Launcher preflight: deferring termination to single-instance guard\n'
  } >>"$LOG_FILE"
else
  printf 'Launcher preflight: no numeric lock-holder pid in %s\n' "$LOCK_FILE" >>"$LOG_FILE"
fi

export REPO_ROOT LOG_FILE
export VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
export UV_BIN="${UV_BIN:-/Users/noahlyons/.pyenv/shims/uv}"
export SPOKE_COMMAND_URL="${SPOKE_COMMAND_URL:-http://localhost:8001}"
unset SPOKE_PREVIEW_MODEL
unset SPOKE_TRANSCRIPTION_MODEL
unset SPOKE_WHISPER_MODEL

# Source per-worktree env overrides (API keys, model dirs, voice, etc.)
if [ -f "$REPO_ROOT/.spoke-smoke-env" ]; then
  printf 'Sourcing %s/.spoke-smoke-env\n' "$REPO_ROOT" >>"$LOG_FILE"
  . "$REPO_ROOT/.spoke-smoke-env"
fi

/usr/bin/python3 - <<'PY'
import os
import subprocess
import traceback
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
log_file = Path(os.environ["LOG_FILE"])
python_exe = Path(os.environ.get("VENV_PYTHON", str(repo_root / ".venv" / "bin" / "python")))
uv_bin = Path(os.environ.get("UV_BIN", "/Users/noahlyons/.pyenv/shims/uv"))
child_env = os.environ.copy()
child_env.setdefault("SPOKE_COMMAND_URL", "http://localhost:8001")
child_env.pop("SPOKE_PREVIEW_MODEL", None)
child_env.pop("SPOKE_TRANSCRIPTION_MODEL", None)
child_env.pop("SPOKE_WHISPER_MODEL", None)

with log_file.open("a", encoding="utf-8") as log:
    try:
        if python_exe.is_file():
            command = [str(python_exe), "-m", "spoke"]
        elif uv_bin.is_file():
            command = [str(uv_bin), "run", "--directory", str(repo_root), "python", "-m", "spoke"]
        else:
            log.write(
                "No repo .venv Python found and UV launcher is unavailable.\n"
            )
            log.flush()
            raise SystemExit(1)

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
