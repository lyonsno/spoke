#!/bin/bash
# Launch spoke smoke-test build. Bind to a hotkey via macOS Shortcuts or Automator.
# Kills any existing instance first.
#
# The smoke target is read from ~/.config/spoke/smoke-target, which should
# contain the absolute path to the repo/worktree to launch from.
#
# If the target file doesn't exist or the target directory is gone,
# plays the system alert sound and exits — no modal dialog, no fallback.

SMOKE_TARGET_FILE="${HOME}/.config/spoke/smoke-target"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-smoke-launch.log"
mkdir -p "$LOG_DIR"

# Read smoke target
if [ ! -f "$SMOKE_TARGET_FILE" ]; then
  osascript -e 'display notification "No smoke target configured" with title "Spoke Smoke" subtitle "Set ~/.config/spoke/smoke-target"' 2>/dev/null
  afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &
  exit 0
fi

IFS= read -r REPO_ROOT < "$SMOKE_TARGET_FILE"

if [ -z "$REPO_ROOT" ] || [ ! -d "$REPO_ROOT" ]; then
  osascript -e "display notification \"Target gone: $REPO_ROOT\" with title \"Spoke Smoke\" subtitle \"Worktree may have been cleaned up\"" 2>/dev/null
  afplay /System/Library/Sounds/Basso.aiff 2>/dev/null &
  exit 0
fi

pkill -TERM -f "python.*spoke" 2>/dev/null
sleep 0.5
rm -f ~/Library/Logs/.spoke.lock

{
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Launching Spoke smoke from %s\n' "$REPO_ROOT"
} >>"$LOG_FILE"

export REPO_ROOT LOG_FILE
export VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
export UV_BIN="${UV_BIN:-/Users/noahlyons/.pyenv/shims/uv}"
unset SPOKE_PREVIEW_MODEL
unset SPOKE_TRANSCRIPTION_MODEL
unset SPOKE_WHISPER_MODEL

# Source per-branch env overrides from the smoke target if present.
# Each worktree can drop a .spoke-smoke-env to set feature flags
# (e.g. SPOKE_TTS_VOICE, SPOKE_COMMAND_URL) without editing this script.
if [ -f "$REPO_ROOT/.spoke-smoke-env" ]; then
  # shellcheck source=/dev/null
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
