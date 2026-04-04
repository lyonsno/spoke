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

        log.write(f"Launching Spoke target {target_id} from {repo_root}\n")
        log.write(f"Launcher child command: {command!r}\n")
        subprocess.run(
            ["pkill", "-TERM", "-f", "python.*spoke"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.5)
        log.write("Launch target handoff: terminated prior local python-based spoke processes.\n")
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
