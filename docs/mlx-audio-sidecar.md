# MLX-audio sidecar

`spoke` owns the contract for the MLX-audio sidecar it talks to. The current
implementation of that contract lives in the sibling Voxtral-capable fork at
`../mlx-audio-pr-607-voxtral-tts`, not in an arbitrary `mlx-audio` install.

This is the tracked serving surface on `MacBook-Pro-2.local` as of
`2026-04-04`:

- Port: `9001`
- Repo: `../mlx-audio-pr-607-voxtral-tts`
- Binary: `.venv/bin/mlx_audio.server`
- Routes: `GET /`, `GET/POST /v1/models`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`
- Served TTS families: Voxtral, VibeVoice, Kokoro, Qwen3-TTS

## Why this lives here

The sidecar is part of `spoke`'s runtime contract, so the launch path and
expected backend surface must be documented in this repo.

The MLX-audio source itself still lives in its own repo. `spoke` should own:

- which checkout is canonical,
- which port is canonical,
- which command boots it,
- and which models must be servable from that checkout.

## Canonical bootstrap

From the `spoke` repo root:

```sh
./scripts/setup-mlx-audio-server.sh --start --port 9001
```

That script expects the sibling fork checkout at
`../mlx-audio-pr-607-voxtral-tts` by default. Override it only if the fork
lives somewhere else:

```sh
./scripts/setup-mlx-audio-server.sh \
  --repo "$HOME/dev/mlx-audio-pr-607-voxtral-tts" \
  --start \
  --port 9001
```

Under the hood the script runs:

```sh
cd ../mlx-audio-pr-607-voxtral-tts
uv sync --extra server --extra tts --extra sts
.venv/bin/mlx_audio.server --host 0.0.0.0 --port 9001 --workers 1
```

The `sts` extra is not optional right now: `mlx_audio/server.py` hard-imports
`webrtcvad`, so `server + tts` alone is not a complete runtime.

## Required served models

This is the TTS surface `spoke` is currently tracking as known-good:

- `mlx-community/Voxtral-4B-TTS-2603-mlx-6bit`
- `mlx-community/VibeVoice-Realtime-0.5B-fp16`
- `mlx-community/Kokoro-82M-bf16`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`

Load them into the running server with:

```sh
curl -X POST "http://localhost:9001/v1/models?model_name=mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
curl -X POST "http://localhost:9001/v1/models?model_name=mlx-community/VibeVoice-Realtime-0.5B-fp16"
curl -X POST "http://localhost:9001/v1/models?model_name=mlx-community/Kokoro-82M-bf16"
curl -X POST "http://localhost:9001/v1/models?model_name=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
```

## Quick verification

These probes should work once the sidecar is up:

```sh
curl -sS http://127.0.0.1:9001/
curl -sS http://127.0.0.1:9001/openapi.json
curl -sS http://127.0.0.1:9001/v1/models
```

If the process is listening but `/` is unreachable from a sandboxed shell,
treat that as a sandbox limitation first, not as proof that the sidecar died.

## spoke integration

`spoke` can use this sidecar for remote TTS. The current default sidecar URL in
code is `http://MacBook-Pro-2.local:9001`, but the active backend, sidecar URL,
model, and voice all persist in `~/Library/Application Support/Spoke/model_preferences.json`.

Per-worktree `.spoke-smoke-env` files may still override local smoke behavior,
but they are not the canonical place to discover the MLX-audio fork path or the
required served model set. That contract lives here and in
`scripts/setup-mlx-audio-server.sh`.
