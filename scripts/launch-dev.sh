#!/bin/bash
# Launch spoke dev build. Bind to a hotkey via macOS Shortcuts or Automator.
# Kills any existing instance first.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

pkill -TERM -f "python.*spoke" 2>/dev/null
sleep 0.5
rm -f ~/Library/Logs/.spoke.lock

nohup env -C "$REPO_ROOT" \
  SPOKE_WHISPER_MODEL="${SPOKE_WHISPER_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}" \
  "$REPO_ROOT/.venv/bin/python" -m spoke \
  </dev/null >/dev/null 2>&1 &
