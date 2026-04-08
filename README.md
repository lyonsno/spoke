# spoke

Speech-native control surface for macOS.

`spoke` is a menubar app built with PyObjC. Hold the spacebar anywhere on the
system to dictate, route the utterance into a tray for review, send it into a
tool-calling assistant, or keep recording hands-free. It is not just
"dictation with an AI mode": `spoke` treats direct text insertion, tray review,
assistant dispatch, and spoken playback as separate surfaces with explicit
transitions between them. Preview/final transcription, assistant inference, and
TTS each have their own backend selection and persist in
`~/Library/Application Support/Spoke/model_preferences.json`.

<video src="https://github.com/user-attachments/assets/f05bafa9-f149-494b-b514-84070a6125e4" width="100%"></video>

## Current surface

- System-wide hold-to-dictate with paste verification and tray fail-open
- Live preview overlay and screen-edge glow while recording
- Latched recording plus optional wake-word hands-free dictation
- Assistant pathway with streaming overlay output, tool calls, and thinking summaries
- Independent preview and final transcription backends: local, sidecar, or cloud
- TTS backends: local MLX runtime, MLX-audio sidecar, or Gemini cloud
- Menubar-driven backend, model, and launch-target control with persisted preferences
- Terror Form HUD for live topoi/status visibility
- Single-instance behavior with visible source and branch in the menubar

## Interaction model

```text
Hold spacebar -> speak -> release clean to paste at cursor
                        -> hold Shift at release to route into the tray
                        -> hold Enter at release to send to the assistant
Tap Shift while recording -> latch recording hands-free
Optional wake words -> start or stop hands-free dictation without touching the keyboard
```

Quick taps still produce a normal space. Longer holds trigger recording,
preview text, and the overlay/glow surface. If insertion cannot be verified,
`spoke` falls back to the tray instead of silently losing text.

Outside active recording, a single idle Shift tap toggles TTS audibility,
double-tap Shift toggles Terror Form, and pressing Enter during the pre-hold
`WAITING` window toggles the assistant overlay.

The full gesture surface lives in
[`docs/keyboard-grammar.md`](docs/keyboard-grammar.md).

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

Basic install:

```sh
git clone https://github.com/lyonsno/spoke.git
cd spoke
uv sync
```

If you want the full local speech stack, local TTS runtimes, and the usual dev
tooling, use:

```sh
uv sync --extra tts --group dev
```

## Run

```sh
uv run spoke
```

On first run macOS will ask for:

- Microphone access
- Accessibility access

Accessibility must be granted to the app that launches `spoke` if you run it
from a terminal, or to `Spoke.app` if you run the bundled app.

## Backend model

`spoke` starts with local transcription by default:

- Preview: `mlx-community/whisper-base.en-mlx-8bit`
- Final transcription: `mlx-community/whisper-medium.en-mlx-8bit`

After launch, the menubar is the primary control surface for backend selection.
Current choices persist across relaunches in
`~/Library/Application Support/Spoke/model_preferences.json`.

The current menu surface can independently control:

- `Preview Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Transcription Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Assistant Backend`: local OMLX, sidecar OMLX, or cloud
- `TTS Backend`: local runtime, MLX-audio sidecar, or Gemini cloud

Environment variables still matter, but mostly as seed values or smoke/test
overrides. In particular, assistant and narrator URLs are still live env-driven
inputs during bootstrap and smoke flows. Once preferences exist, the app uses
the saved backend/model state instead of pretending the env is the whole story.

## Remote sidecars

For the tracked MLX-audio serving surface, bootstrap the sibling fork with:

```sh
./scripts/setup-mlx-audio-server.sh --start --port 9001
```

That script syncs the expected fork checkout, installs the required extras, and
starts `.venv/bin/mlx_audio.server` on port `9001`. The canonical sidecar
contract, required models, and manual probes are documented in
[`docs/mlx-audio-sidecar.md`](docs/mlx-audio-sidecar.md).

If you want a quick health check for the local service fleet, run:

```sh
./scripts/spoke-doctor.sh
```

That script reports the current status of the assistant endpoint, narrator,
MLX-audio sidecar, remote Whisper sidecar, and the running `spoke` process.

## Configuration

The env vars use some legacy names (`WHISPER`, `COMMAND`) for historical
reasons. They now seed a broader backend model than the names suggest.

### Core runtime knobs

| Variable | Default | Description |
|---|---|---|
| `SPOKE_HOLD_MS` | `200` | Spacebar hold threshold in milliseconds. |
| `SPOKE_RESTORE_DELAY_MS` | `1000` | Delay before restoring the saved pasteboard contents. |
| `SPOKE_PREVIEW_MODEL` | `mlx-community/whisper-base.en-mlx-8bit` | Initial local preview model. |
| `SPOKE_TRANSCRIPTION_MODEL` | `mlx-community/whisper-medium.en-mlx-8bit` | Initial local final-transcription model. |
| `SPOKE_WHISPER_MODEL` | unset | Legacy single-model override for both preview and final roles. |
| `SPOKE_WHISPER_URL` | unset | Initial remote transcription sidecar URL for OpenAI-compatible `/v1/audio/transcriptions`. |
| `SPOKE_COMMAND_URL` | `http://localhost:8001` | Initial local or sidecar assistant endpoint. Menu persistence wins after first save. |
| `SPOKE_COMMAND_MODEL` | `qwen3p5-35B-A3B` | Initial assistant model id. |
| `SPOKE_COMMAND_API_KEY` | unset | Optional bearer token for the assistant endpoint. |
| `SPOKE_COMMAND_MODEL_DIR` | `~/.lmstudio/models` | Extra local model inventory to seed assistant menu entries. |
| `SPOKE_TTS_MODEL` | `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` | Initial local or sidecar TTS model selection. |
| `SPOKE_TTS_VOICE` | unset | Initial voice selection for local or sidecar TTS. |
| `GEMINI_API_KEY` | unset | Enables cloud assistant and Gemini cloud TTS. |

Cloud Whisper transcription is configured in-app from the menubar and persisted
to `model_preferences.json`; it is not currently driven by a dedicated env var.

### Optional integrations

| Variable | Default | Description |
|---|---|---|
| `SPOKE_PARAKEET_MODEL_DIR` | unset | Path to a `FluidInference/parakeet-ctc-110m-coreml` checkout or model snapshot. |
| `SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY` | unset | Enables wake-word hands-free mode. |
| `SPOKE_WAKEWORD_LISTEN` | `computer` | Wake word that starts hands-free dictation. |
| `SPOKE_WAKEWORD_SLEEP` | `terminator` | Wake word that returns hands-free mode to dormant. |
| `SPOKE_WAKEWORD_LISTEN_PPN` | unset | Optional custom Porcupine model file for the listen wake word. |
| `SPOKE_WAKEWORD_SLEEP_PPN` | unset | Optional custom Porcupine model file for the sleep wake word. |
| `SPOKE_NARRATOR_URL` | unset | Optional separate OpenAI-compatible narrator sidecar URL. Defaults to the assistant endpoint. |
| `SPOKE_NARRATOR_MODEL` | `Bonsai-8B-mlx-1bit` | Narrator model used for thinking summaries and loading-vamp lines. |
| `SPOKE_NARRATOR_API_KEY` | unset | Optional narrator bearer token. Falls back to the assistant API key. |
| `SPOKE_NARRATOR_ENABLED` | `1` | Set to `0` to disable narrator summaries entirely. |
| `SPOKE_MODEL_PREFERENCES_PATH` | unset | Override path for persisted backend/model preferences. Useful for isolated smoke/test surfaces. |
| `SPOKE_GMAIL_CREDENTIALS_PATH` | `~/Library/Application Support/Spoke/gmail_credentials.json` | Local Gmail OAuth material for the bounded `query_gmail` tool. |
| `SPOKE_GMAIL_CLIENT_ID` | unset | Optional Gmail OAuth client id override. |
| `SPOKE_GMAIL_CLIENT_SECRET` | unset | Optional Gmail OAuth client secret override. |
| `SPOKE_GMAIL_REFRESH_TOKEN` | unset | Optional Gmail OAuth refresh token override. |
| `SPOKE_GMAIL_TOKEN_URI` | `https://oauth2.googleapis.com/token` | Optional Gmail OAuth token endpoint override. |

The Gmail affordance is intentionally narrow and read-only: `query_gmail`
returns compact metadata plus snippets for matching messages rather than full
message bodies.

## Notes

- `spoke` keeps a bounded post-transcription repair pass for recurring
  project-specific vocabulary that is known to fail in real logs.
- The assistant tool surface includes local filesystem and screen-context
  affordances; the overlay is no longer just a text dump from a single local
  model.
- TTS is now a real routing surface rather than a single hardcoded backend.

## Development

Run the test suite:

```sh
uv run pytest -v
```

Core modules:

```text
spoke/
├── __main__.py           # app delegate, menu state, backend wiring, lifecycle
├── input_tap.py          # global key grammar and hold detection
├── capture.py            # sounddevice recording and WAV encoding
├── handsfree.py          # latched and wake-word-driven dictation controller
├── wakeword.py           # Picovoice Porcupine listener
├── transcribe.py         # remote OpenAI-compatible transcription client
├── transcribe_local.py   # local MLX Whisper backend
├── transcribe_qwen.py    # local Qwen3-ASR backend
├── transcribe_parakeet.py # local Parakeet CoreML backend
├── command.py            # assistant client and tool-call streaming
├── narrator.py           # thinking-summary sidecar
├── tts.py                # local, sidecar, and cloud TTS clients
├── command_overlay.py    # assistant overlay
├── overlay.py            # live transcription overlay
├── glow.py               # screen-edge glow
├── terraform_hud.py      # Terror Form HUD
├── menubar.py            # status item and menu
└── tool_dispatch.py      # local tool execution surface
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
- Local MLX backends may download model weights on first use.
- The local runtime is Apple Silicon-oriented, but sidecar and cloud backends
  work independently of local model availability.

## License

MIT
