#!/bin/bash
# Stable Automator entry point. Dispatch launcher behavior to the selected
# target instead of pinning a workflow to one historical release.

set -euo pipefail

TARGETS_FILE="${SPOKE_LAUNCH_TARGETS_PATH:-${HOME}/.config/spoke/launch_targets.json}"
export TARGETS_FILE

exec /usr/bin/python3 - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path


def refuse(message: str) -> None:
    rendered = f"Spoke selected launcher refused: {message}"
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


registry_path = Path(os.environ["TARGETS_FILE"]).expanduser()
try:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    refuse(f"registry unavailable or invalid: {registry_path} ({exc})")

if not isinstance(payload, dict):
    refuse("registry must be an object")
selected = payload.get("selected")
if (
    not isinstance(selected, str)
    or not selected.strip()
    or selected != selected.strip()
    or not selected.isprintable()
):
    refuse("selected target id must be one nonblank printable string")
targets = payload.get("targets")
if not isinstance(targets, list):
    refuse("registry targets must be a list")
matches = [
    target
    for target in targets
    if isinstance(target, dict) and target.get("id") == selected
]
if len(matches) != 1:
    refuse(f"selected target {selected!r} must appear exactly once")

raw_path = matches[0].get("path")
if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
    refuse(f"selected target {selected!r} path must be a nonblank string")
target_path = Path(raw_path).expanduser()
if not target_path.is_absolute():
    refuse(f"selected target {selected!r} path must be absolute")
target_path = target_path.resolve()
registry_path = registry_path.resolve()
launcher = target_path / "scripts" / "launch-main.sh"
if not target_path.is_dir() or not launcher.is_file() or not os.access(launcher, os.X_OK):
    refuse(f"selected target {selected!r} launcher is unavailable: {launcher}")

child_env = os.environ.copy()
child_env["SPOKE_LAUNCH_TARGETS_PATH"] = str(registry_path)
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_ID"] = selected
child_env["SPOKE_EXPECTED_LAUNCH_TARGET_PATH"] = str(target_path)
os.execve(launcher, [str(launcher)], child_env)
PY
