# spoke

Global hold-to-dictate for macOS.

Hold the spacebar anywhere on the system, speak, release, and `spoke` pastes the transcription at the current cursor. It runs as a menubar app, uses PyObjC instead of Swift, and supports three transcription modes:

- Local MLX Whisper
- Local Qwen3-ASR via MLX
- Remote OpenAI-compatible `/v1/audio/transcriptions` sidecar

<video src="https://github.com/user-attachments/assets/f05bafa9-f149-494b-b514-84070a6125e4" width="100%"></video>

## How It Works

```text
Hold spacebar → record mic audio → preview text while speaking → transcribe on release → paste at cursor
```

Quick taps still produce a normal space. Longer holds trigger recording, show the glow/overlay UI, then inject the final text with pasteboard save/restore plus synthetic `Cmd+V`.

## Features

- Global spacebar hold detection with normal tap passthrough
- Live preview overlay during recording
- Screen-edge glow driven by microphone amplitude
- Local transcription by default (Whisper or Qwen3-ASR) when `SPOKE_WHISPER_URL` is unset
- Optional remote sidecar mode for heavier models
- Single-instance app behavior
- Menubar-only UI with no Dock icon

## Requirements

- macOS 11+
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `portaudio`

Install the system audio dependency:

```sh
brew install portaudio
```

## Install

```sh
git clone https://github.com/lyonsno/spoke.git
cd spoke
uv sync
```

## Run

### Default: local MLX Whisper

If you do not set `SPOKE_WHISPER_URL`, `spoke` runs transcription locally with `mlx-whisper`.
By default, preview uses Whisper `medium.en` while final transcription uses Whisper `large-v3-turbo`
on machines that pass the existing RAM guard; otherwise both roles fall back to the lighter model.

```sh
uv run spoke
```

### Local Qwen3-ASR

Use a Qwen model name to switch the local backend:

```sh
SPOKE_WHISPER_MODEL=Qwen/Qwen3-ASR-0.6B uv run spoke
```

### Remote sidecar

Point `spoke` at any OpenAI-compatible transcription server:

```sh
SPOKE_WHISPER_URL=http://<host>:8000 uv run spoke
```

Example sidecar options on Apple Silicon:

```sh
uv tool install "mlx-audio[server]"
mlx-audio-server --host 0.0.0.0 --port 8000
```

## Permissions

On first run, macOS will ask for:

- Microphone access
- Accessibility access

Accessibility must be granted to the app that launches `spoke` if you run it from a terminal, or to `Spoke.app` if you run the bundled app.

## Configuration

### Core environment variables

The env var names use `WHISPER` for historical reasons — they control all backends, not just Whisper.
Preview and final transcription can now be configured independently, and the menu persists those
choices across relaunches.

| Variable | Default | Description |
|---|---|---|
| `SPOKE_WHISPER_URL` | unset | Remote transcription server. When unset, transcription runs locally. |
| `SPOKE_WHISPER_MODEL` | unset | Legacy single-model override. When set, both preview and final use the same model. |
| `SPOKE_PREVIEW_MODEL` | `mlx-community/whisper-medium.en-mlx-8bit` | Preview model identifier. Use `Qwen/Qwen3-ASR-0.6B` for local streaming preview, or any menu-listed Whisper variant. |
| `SPOKE_TRANSCRIPTION_MODEL` | `mlx-community/whisper-large-v3-turbo` | Final transcription model identifier. Use `Qwen/Qwen3-ASR-0.6B` or any menu-listed Whisper variant. |
| `SPOKE_HOLD_MS` | `250` | Spacebar hold threshold in milliseconds. Must be greater than `0`. |
| `SPOKE_RESTORE_DELAY_MS` | `1000` | Delay before the original pasteboard contents are restored. |

### UI tuning

The overlay and glow also expose advanced tuning env vars such as `SPOKE_GLOW_MULTIPLIER`, `SPOKE_TEXT_ALPHA_MIN`, and related `SPOKE_*` values in the overlay/glow modules.

## Development

Run the test suite:

```sh
uv run pytest -v
```

Each layer is independent and testable in isolation.

```text
spoke/
├── __main__.py          # app delegate and runtime wiring
├── input_tap.py         # global spacebar hold detection
├── capture.py           # sounddevice recording and WAV encoding
├── transcribe.py        # remote OpenAI-compatible client
├── transcribe_local.py  # local MLX Whisper backend
├── transcribe_qwen.py   # local Qwen3-ASR backend
├── inject.py            # pasteboard save/paste/restore
├── glow.py              # screen-edge amplitude glow
├── overlay.py           # live transcription overlay
└── menubar.py           # status item and menu
```

## Build

Build the macOS app bundle with PyInstaller:

```sh
./scripts/build.sh
```

Fast incremental rebuild:

```sh
./scripts/build.sh --fast
```

Create a DMG after building the app:

```sh
brew install create-dmg
./scripts/build-dmg.sh
```

The app bundle is written to `dist/Spoke.app`.

## Notes

- The bundled app logs to `~/Library/Logs/Spoke.log`.
- The local MLX backends may download model weights on first use.
- The app is designed for Apple Silicon-oriented local inference workflows, but remote sidecar mode works independently of local model availability.

## License

MIT
