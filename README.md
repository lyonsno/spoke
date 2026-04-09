# spoke

Speech-native control surface for macOS.

`spoke` is a menubar app built with PyObjC. Hold the spacebar anywhere on the
system to dictate, route the utterance into a tray for review, send it into a
tool-calling assistant, or keep recording hands-free. Direct text insertion,
tray review, assistant dispatch, and spoken playback are separate surfaces
with explicit transitions between them. Preview/final transcription, assistant
inference, and TTS each have their own backend selection and persist in
`~/Library/Application Support/Spoke/model_preferences.json`.

<video src="https://github.com/user-attachments/assets/f05bafa9-f149-494b-b514-84070a6125e4" width="100%"></video>

## What It Does

- Dictate anywhere on the system and paste directly into the focused field
- Fail open into a tray when insertion cannot be verified or when you want review first
- Send spoken utterances to an assistant with streamed responses and tool calls
- Keep recording hands-free with latched mode or wake words
- Read results back through local, sidecar, or cloud TTS backends
- Switch transcription, assistant, and TTS backends from the menubar and keep those choices across relaunches

## Product Shape

`spoke` is built around four connected surfaces:

- `Text`: hold space, speak, release cleanly, and the text lands at the cursor.
- `Tray`: hold shift at release to stage speech for review, recovery, recall, or later insertion.
- `Assistant`: hold enter at release to send the utterance into the assistant path.
- `Speech out`: assistant responses can be spoken back through the configured TTS backend.

The overlays and glow exist to make those transitions legible.

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
`spoke` falls back to the tray so the utterance is recoverable.

Hands-free mode can also be started by voice. Set
`SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY` (see the env-var table below) to enable
the wake-word listener; without that key the wake-word path is inert and only
the keyboard gestures above are active.

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

## Backend Selection

`spoke` starts with local transcription by default:

- Preview: `mlx-community/whisper-base.en-mlx-8bit`
- Final transcription: `mlx-community/whisper-medium.en-mlx-8bit`

After launch, the menubar is the canonical control surface for backend
selection. Current choices persist across relaunches in
`~/Library/Application Support/Spoke/model_preferences.json`.

The menus can independently control:

- `Preview Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Transcription Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Assistant Backend`: local OMLX, sidecar OMLX, or cloud
- `TTS Backend`: local runtime, MLX-audio sidecar, or Gemini cloud

For ordinary use, prefer the menus. The remaining environment variables are
smoke/debugging overrides and bootstrap plumbing.

### Reasoning continuity caveat

`spoke`'s current command pathway talks to hosted assistant backends through an
OpenAI-compatible `/v1/chat/completions` loop. That transport is good enough
for streamed text, tool calls, and visible thinking summaries, but it is not
the same thing as preserving first-class reasoning state across a multi-round
tool loop.

Concretely:

- `spoke` preserves assistant/tool message chains between tool rounds.
- `spoke` does not preserve first-class reasoning items for replay on the next
  round.
- Visible thinking summaries are downstream UI, not the semantic contract that
  lets a reasoning model continue the same hidden chain of thought after tool
  results come back.

So a cloud backend can look correct in the overlay and still lose reasoning
continuity during agentic tool use. If a provider requires replaying reasoning
blocks or thinking items to maintain performance across tool rounds, the
current generic chat-completions transport is semantically incomplete for that
provider.

This matters for backend selection. A provider's marketing around "reasoning"
is not enough; what matters is whether the protocol we are actually speaking
preserves the reasoning state the model needs on the next turn. If we want the
stronger contract, we likely need a reasoning-aware transport rather than
another generic OpenAI-compatible adapter.

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

That script reports the current status of the assistant endpoint,
MLX-audio sidecar, remote Whisper sidecar, and the running `spoke` process.

## Advanced Overrides

If you are running isolated smoke surfaces or debugging backend wiring, a small
set of env vars is still useful. For normal use, prefer the menus.

| Variable | Default | Description |
|---|---|---|
| `SPOKE_HOLD_MS` | `200` | Spacebar hold threshold in milliseconds. |
| `SPOKE_RESTORE_DELAY_MS` | `1000` | Delay before restoring the saved pasteboard contents. |
| `SPOKE_MODEL_PREFERENCES_PATH` | unset | Override path for persisted backend/model preferences. Useful for isolated smoke/test surfaces. |
| `SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY` | unset | Enables wake-word hands-free mode. |
| `SPOKE_WAKEWORD_LISTEN` | `computer` | Wake word that starts hands-free dictation. |
| `SPOKE_WAKEWORD_SLEEP` | `terminator` | Wake word that returns hands-free mode to dormant. |

If you need deeper backend or smoke-surface plumbing than that, you are in
developer territory and should inspect the codepaths in
[`spoke/__main__.py`](spoke/__main__.py) and related modules rather than treat
the README as a full configuration reference.

## Notes

- `spoke` keeps a bounded post-transcription repair pass for recurring
  project-specific vocabulary that is known to fail in real logs.
- The assistant tool surface includes local filesystem and screen-context
  affordances available to the model during a turn.
- TTS is a routing surface across local, sidecar, and cloud backends.
- Brief thinking summaries can be shown while the assistant is reasoning or
  loading, as a secondary affordance.
- The menubar also exposes launch-target switching, source/branch visibility,
  and the status HUD (`Terror Form`) for runtime legibility on local smoke
  surfaces.

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
├── narrator.py           # optional thinking-summary sidecar
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

## Runtime notes

- The bundled app logs to `~/Library/Logs/Spoke.log`.
- Local MLX backends may download model weights on first use.
- The local runtime is Apple Silicon-oriented, but sidecar and cloud backends
  work independently of local model availability.
- oMLX already exposes `POST /v1/messages` with Anthropic-compatible Messages
  semantics and adaptive thinking. That makes an Anthropic-style transport a
  plausible next step for `spoke` without changing providers. The important
  distinction is transport, not branding: preserving the stronger tool-use
  semantics requires `spoke` to speak the Messages contract end to end instead
  of flattening everything back through generic `/v1/chat/completions`.

## License

MIT
