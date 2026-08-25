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
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

helper_repo_root = Path(os.environ["HELPER_REPO_ROOT"])
if str(helper_repo_root) not in sys.path:
    sys.path.insert(0, str(helper_repo_root))

from spoke.launch_targets import LaunchTargetUnavailable, require_selected_launch_target

target_id = os.environ.get("TARGET_ID", "").strip()
raw_registry_path = Path(os.environ["TARGETS_FILE"])
registry_path = None
selected_target_id = None
expected_target_path = None


def refuse(message: str) -> None:
    rendered = f"Spoke named-target launcher refused: {message}"
    print(rendered, file=sys.stderr)
    payload = {
        "status": "refused",
        "phase": "named_target_predelegation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
        "requested_target_id": target_id or None,
        "selected_target_id": selected_target_id,
        "registry_path": str(registry_path or raw_registry_path),
        "expected_target_path": expected_target_path,
        "reason": message,
    }
    try:
        log_path = Path.home() / "Library/Logs/spoke-launch-target-refusals.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload, sort_keys=True) + "\n")
            log.flush()
            os.fsync(log.fileno())
        os.chmod(log_path, 0o600)
    except Exception:
        pass
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


if not target_id:
    refuse("no launch target was requested")

try:
    registry_path = raw_registry_path.expanduser().resolve()
    target = require_selected_launch_target(registry_path)
except (LaunchTargetUnavailable, OSError, RuntimeError) as exc:
    selected_target_id = getattr(exc, "selected_target_id", None)
    refuse(str(exc))
selected_target_id = target["id"]
expected_target_path = str(target["path"])
if target["id"] != target_id:
    refuse(
        f"requested target {target_id!r} is not the selected target {target['id']!r}"
    )

target_path = Path(expected_target_path)
launcher = target_path / "scripts" / "launch-main.sh"
if not launcher.is_file() or not os.access(launcher, os.X_OK):
    refuse(f"selected target launcher is unavailable: {launcher}")

child_env = os.environ.copy()
child_env["SPOKE_LAUNCH_TARGETS_PATH"] = str(registry_path)
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_ID"] = target["id"]
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_PATH"] = str(target_path)
os.execve(launcher, [str(launcher)], child_env)
PY
