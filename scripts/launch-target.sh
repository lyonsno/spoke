#!/bin/bash
# Compatibility entry point for callers that name a target. The requested
# target must already be the strict selected target; launch-main owns startup
# admission and the application child owns single-instance handoff.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HELPER_REPO_ROOT="${HELPER_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGETS_FILE="${SPOKE_LAUNCH_TARGETS_PATH:-${HOME}/.config/spoke/launch_targets.json}"
TARGET_ID="${1:-${TARGET_ID:-}}"
export HELPER_REPO_ROOT TARGETS_FILE TARGET_ID

exec /usr/bin/python3 - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

helper_repo_root = Path(os.environ["HELPER_REPO_ROOT"])
if str(helper_repo_root) not in sys.path:
    sys.path.insert(0, str(helper_repo_root))

from spoke.launch_targets import LaunchTargetUnavailable, require_selected_launch_target


def refuse(message: str) -> None:
    rendered = f"Spoke named-target launcher refused: {message}"
    print(rendered, file=sys.stderr)
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "Spoke Launch Failed"',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass
    raise SystemExit(1)


target_id = os.environ.get("TARGET_ID", "").strip()
if not target_id:
    refuse("no launch target was requested")

registry_path = Path(os.environ["TARGETS_FILE"]).expanduser().resolve()
try:
    target = require_selected_launch_target(registry_path)
except LaunchTargetUnavailable as exc:
    refuse(str(exc))
if target["id"] != target_id:
    refuse(
        f"requested target {target_id!r} is not the selected target {target['id']!r}"
    )

target_path = target["path"].resolve()
launcher = target_path / "scripts" / "launch-main.sh"
if not launcher.is_file() or not os.access(launcher, os.X_OK):
    refuse(f"selected target launcher is unavailable: {launcher}")

child_env = os.environ.copy()
child_env["SPOKE_LAUNCH_TARGETS_PATH"] = str(registry_path)
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_ID"] = target["id"]
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_PATH"] = str(target_path)
os.execve(launcher, [str(launcher)], child_env)
PY
