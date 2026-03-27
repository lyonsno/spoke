#!/bin/bash
# Launch spoke dev build. Bind to a hotkey via macOS Shortcuts or Automator.
# Kills any existing instance first.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${HOME}/Library/Logs"
LOG_FILE="${LOG_DIR}/spoke-dev-launch.log"

pkill -TERM -f "python.*spoke" 2>/dev/null
sleep 0.5
rm -f ~/Library/Logs/.spoke.lock
mkdir -p "$LOG_DIR"

{
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Launching Spoke from %s\n' "$REPO_ROOT"
} >>"$LOG_FILE"

nohup env -C "$REPO_ROOT" \
  SPOKE_PREVIEW_MODEL="${SPOKE_PREVIEW_MODEL:-mlx-community/whisper-medium.en-mlx-8bit}" \
  SPOKE_TRANSCRIPTION_MODEL="${SPOKE_TRANSCRIPTION_MODEL:-mlx-community/whisper-large-v3-turbo}" \
  "$REPO_ROOT/.venv/bin/python" -m spoke \
  </dev/null >>"$LOG_FILE" 2>&1 &
