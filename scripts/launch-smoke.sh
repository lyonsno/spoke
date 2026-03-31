#!/bin/bash
# Launch spoke smoke-test build. Bind to a hotkey via macOS Shortcuts or Automator.
# Lets the single-instance guard handle any existing instance.
#
# The smoke target is read from ~/.config/spoke/smoke-target, which should
# contain the absolute path to the repo/worktree to launch from.
#
# If the target file doesn't exist or the target directory is gone,
# plays the system alert sound and exits — no modal dialog, no fallback.

SMOKE_TARGET_FILE="${HOME}/.config/spoke/smoke-target"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-smoke-launch.log"
LOCK_FILE="${LOG_DIR}/.spoke.lock"
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

{
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Launcher PID %d (PPID %d) invoked from %s\n' "$$" "$PPID" "$PWD"
  printf 'Launching Spoke smoke from %s\n' "$REPO_ROOT"
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
unset SPOKE_PREVIEW_MODEL
unset SPOKE_TRANSCRIPTION_MODEL
unset SPOKE_WHISPER_MODEL

# Source shared box-local launch defaults first, then worktree-specific overrides.
SHARED_ENV_FILE="${SPOKE_SHARED_LAUNCH_ENV_PATH:-$HOME/.config/spoke/launch-env.sh}"
if [ -f "$SHARED_ENV_FILE" ]; then
  printf 'Sourcing shared launch env: %s\n' "$SHARED_ENV_FILE" >>"$LOG_FILE"
  # shellcheck source=/dev/null
  . "$SHARED_ENV_FILE"
fi

# Source per-branch env overrides from the smoke target if present.
# Each worktree can drop a .spoke-smoke-env to set feature flags
# (e.g. SPOKE_TTS_VOICE, SPOKE_COMMAND_URL) without editing this script.
if [ -f "$REPO_ROOT/.spoke-smoke-env" ]; then
  # shellcheck source=/dev/null
  . "$REPO_ROOT/.spoke-smoke-env"
fi

/usr/bin/python3 - <<'PY'
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional


def _resolve_uv_bin(repo_root: Path) -> Optional[Path]:
    env_uv_bin = os.environ.get("UV_BIN")
    candidates: list[Path] = []
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
tts_enabled = bool(child_env.get("SPOKE_TTS_VOICE"))

with log_file.open("a", encoding="utf-8") as log:
    try:
        if tts_enabled and uv_bin is not None:
            command = [
                str(uv_bin),
                "run",
                "--directory",
                str(repo_root),
                "--extra",
                "tts",
                "python",
                "-m",
                "spoke",
            ]
            log.write(
                "Smoke launcher: SPOKE_TTS_VOICE is set, using uv --extra tts for a coherent TTS runtime.\n"
            )
        elif python_exe.is_file():
            command = [str(python_exe), "-m", "spoke"]
        elif uv_bin is not None:
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
