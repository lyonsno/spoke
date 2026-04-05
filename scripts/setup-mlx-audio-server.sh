#!/bin/bash
# setup-mlx-audio-server.sh — bootstrap spoke's canonical MLX-audio sidecar
#
# spoke owns the MLX-audio sidecar contract, but the implementation currently
# lives in a sibling Voxtral-capable fork checkout. We use that fork instead of
# "whatever mlx-audio is installed" because the published package is not the
# reliable surface for the TTS stack spoke tracks on this machine.
#
# Canonical sidecar contract:
# - repo: ../mlx-audio-pr-607-voxtral-tts
# - port: 9001
# - routes: /v1/models, /v1/audio/speech, /v1/audio/transcriptions
# - served TTS families: Voxtral, VibeVoice, Kokoro, Qwen3-TTS
#
# Usage:
#   ./scripts/setup-mlx-audio-server.sh
#   ./scripts/setup-mlx-audio-server.sh --start
#   ./scripts/setup-mlx-audio-server.sh --repo ~/dev/mlx-audio-pr-607-voxtral-tts --start --port 9001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_MLX_AUDIO_REPO="$REPO_ROOT/../mlx-audio-pr-607-voxtral-tts"

HOST="0.0.0.0"
PORT=9001
START=false
MLX_AUDIO_REPO="${MLX_AUDIO_REPO:-$DEFAULT_MLX_AUDIO_REPO}"

usage() {
    cat <<EOF
Usage: ./scripts/setup-mlx-audio-server.sh [--repo PATH] [--host HOST] [--port PORT] [--start]

Bootstraps the canonical MLX-audio sidecar that spoke tracks for remote TTS/STT.
By default it expects the Voxtral-capable fork at:
  $DEFAULT_MLX_AUDIO_REPO
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            MLX_AUDIO_REPO="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --start)
            START=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$MLX_AUDIO_REPO" ]]; then
    echo "MLX-audio repo not found: $MLX_AUDIO_REPO" >&2
    echo "Expected the Voxtral-capable fork checkout at $DEFAULT_MLX_AUDIO_REPO" >&2
    echo "Pass --repo /path/to/mlx-audio-pr-607-voxtral-tts to override." >&2
    exit 1
fi

if [[ ! -f "$MLX_AUDIO_REPO/mlx_audio/server.py" ]]; then
    echo "Not an mlx-audio checkout: $MLX_AUDIO_REPO" >&2
    exit 1
fi

if [[ ! -f "$MLX_AUDIO_REPO/mlx_audio/tts/models/voxtral_tts/__init__.py" ]]; then
    echo "This mlx-audio checkout does not include the Voxtral TTS backend: $MLX_AUDIO_REPO" >&2
    echo "spoke's tracked sidecar surface requires the Voxtral-capable fork." >&2
    exit 1
fi

echo "==> Syncing MLX-audio fork with required extras (server + tts + sts)..."
(
    cd "$MLX_AUDIO_REPO"
    uv sync --extra server --extra tts --extra sts
)

VENV_PYTHON="$MLX_AUDIO_REPO/.venv/bin/python"
SERVER_BIN="$MLX_AUDIO_REPO/.venv/bin/mlx_audio.server"

if [[ ! -x "$VENV_PYTHON" || ! -x "$SERVER_BIN" ]]; then
    echo "Expected fork venv binaries were not created under $MLX_AUDIO_REPO/.venv/bin" >&2
    exit 1
fi

echo "==> Verifying sidecar imports from fork venv..."
"$VENV_PYTHON" -c "
import importlib
import uvicorn
import webrtcvad
import mlx_audio.server
importlib.import_module('mlx_audio.tts.models.voxtral_tts')
print('All imports OK')
"

echo "==> MLX-audio sidecar ready."
echo "   Repo: $MLX_AUDIO_REPO"
echo "   Binary: $SERVER_BIN"
echo "   Port convention: $PORT"
echo "   Served TTS families tracked by spoke:"
echo "     - mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
echo "     - mlx-community/VibeVoice-Realtime-0.5B-fp16"
echo "     - mlx-community/Kokoro-82M-bf16"
echo "     - mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
echo
echo "==> Load models dynamically after start:"
echo "   curl -X POST \"http://localhost:${PORT}/v1/models?model_name=mlx-community/Voxtral-4B-TTS-2603-mlx-6bit\""
echo "   curl -X POST \"http://localhost:${PORT}/v1/models?model_name=mlx-community/VibeVoice-Realtime-0.5B-fp16\""
echo "   curl -X POST \"http://localhost:${PORT}/v1/models?model_name=mlx-community/Kokoro-82M-bf16\""
echo "   curl -X POST \"http://localhost:${PORT}/v1/models?model_name=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit\""

if $START; then
    echo "==> Starting on ${HOST}:${PORT}..."
    exec "$SERVER_BIN" --host "$HOST" --port "$PORT" --workers 1
fi
