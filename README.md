# spoke

Speech-native control surface for macOS.

`spoke` is a menubar app built with PyObjC for system-wide dictation,
review, and assistant control. Hold the spacebar anywhere on the system to
speak, then decide whether that utterance should land as text, stage in the
tray, go to the assistant, or stay live in hands-free mode. Preview/final
transcription, assistant inference, and TTS each keep their own backend and
model state in
`~/Library/Application Support/Spoke/model_preferences.json`.

<video src="https://github.com/user-attachments/assets/f05bafa9-f149-494b-b514-84070a6125e4" width="100%"></video>

## What It Does

- Dictate anywhere on the system and paste directly into the focused field
- Fail open into a stacked tray when insertion cannot be verified or when you want review first
- Use an assistant that can respond with streamed text, multimodal capture context, Brave Search, local files, Gmail, and background search subagents
- Show a narrator-style thinking and loading surface while the assistant is still working
- Compact long assistant histories so extended sessions can keep going without losing the thread
- Keep recording hands-free with latched mode or wake words
- Read results back through local, sidecar, or cloud TTS backends
- Switch transcription, model, assistant, and TTS choices from the menubar and keep them across relaunches

## Product Shape

`spoke` is built around four connected surfaces:

- `Text`: hold space, speak, release cleanly, and the text lands at the cursor.
- `Tray`: hold shift at release to stage speech in a stacked tray for review, recovery, recall, or later insertion.
- `Assistant`: hold enter at release to send the utterance into the assistant path, with streamed output and live thinking/loading summaries while it works.
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
`spoke` fails open into the tray so the utterance stays recoverable.

Hands-free mode can also be started by voice. Set
`SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY` (see the env-var table below) to enable
the wake-word listener; without that key the wake-word path is inert and only
the keyboard gestures above are active.

While dictating hands-free, simple spoken editing commands such as `new line`,
`new paragraph`, and `enter` are treated as controls rather than literal text.

If you want to prepare custom wakeword training material, `spoke`
now also ships a batch sample generator that renders WAVs through the same
local, sidecar, or Gemini cloud TTS surfaces:

```sh
uv run python -m spoke.wakeword_samples \
  --backend local \
  --text-file assets/wakewords/operation_mouthfeel_commands.txt \
  --voice-file assets/wakewords/kokoro_american_8.txt \
  --max-tokens 32 \
  --output-dir /tmp/mouthfeel-samples
```

Add `--text-file phrases.txt` or `--voice-file voices.txt` for one item per
line, switch to `--backend cloud` to use Gemini cloud TTS, or provide
`--sidecar-url` for a remote OpenAI-compatible speech sidecar. For short
wakewords, `--max-tokens` is useful when you want to keep the render from
wandering into trailing filler.

To carve a generated batch into per-keyword training groups:

```sh
uv run python -m spoke.wakeword_training \
  --batch-dir /tmp/mouthfeel-samples \
  --output-dir /tmp/mouthfeel-training
```

That writes one directory per keyword, with copied WAVs, a manifest, and a
`*-samples.zip` archive for each keyword. Those grouped artifacts are useful
for audition, curation, or handing off to whatever training pipeline the lane
is using.

If you want to train local `openWakeWord` models from that same batch, install
the training extra and run the trainer directly:

```sh
uv sync --extra wakeword-training --group dev
uv run --extra wakeword-training python -m spoke.openwakeword_training \
  --batch-dir /tmp/mouthfeel-samples \
  --output-dir /tmp/mouthfeel-openwakeword \
  --keyword tessera
```

That writes a local `.onnx` model plus JSON metrics and dataset manifests for
each requested keyword. The trainer accepts the Kokoro/sidecar sample batches
that `spoke.wakeword_samples` emits, including 24 kHz mono WAVs, and resamples
them to the 16 kHz runtime rate during feature extraction.

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

## Cloud And Wake-Word Setup

If you want cloud backends, sidecar bootstrap, or wake-word auth, create a
machine-local secrets file from the checked-in template:

```sh
mkdir -p ~/.config/spoke
cp scripts/secrets.env.example ~/.config/spoke/secrets.env
chmod 600 ~/.config/spoke/secrets.env
```

Then populate `~/.config/spoke/secrets.env` from your offline source of truth.
That keeps secrets out of the repo while giving `spoke` a stable place to find
cloud and wake-word credentials.

## Run

```sh
uv run spoke
```

On first run macOS will ask for:

- Microphone access
- Accessibility access

Accessibility must be granted to the app that launches `spoke` if you run it
from a terminal, or to `Spoke.app` if you run the bundled app.

## Backend And Model Selection

`spoke` starts with local transcription by default:

- Preview: `mlx-community/whisper-base.en-mlx-8bit`
- Final transcription: `mlx-community/whisper-medium.en-mlx-8bit`

After launch, the menubar is the canonical control surface for backend
selection. Current choices persist across relaunches in
`~/Library/Application Support/Spoke/model_preferences.json`.

The menus can independently control:

- `Preview Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Transcription Backend`: local Whisper, sidecar, or cloud OpenAI Whisper
- `Preview Model`: the fast speculative model used for live preview when the active backend supports model choice
- `Transcription Model`: the final commit model used when the active backend supports model choice
- `Assistant Backend`: Local OMLX, Sidecar OMLX, Google Cloud, or OpenRouter
- `Assistant Model`: the concrete model for the currently selected assistant surface
- `TTS Backend`: local runtime, MLX-audio sidecar, or Gemini cloud

Each surface remembers its own last selection, including the active assistant
model. If you want cloud backends or wake words, populate
`~/.config/spoke/secrets.env` from `scripts/secrets.env.example` first.

For ordinary use, prefer the menus. The remaining environment variables are
bootstrap, smoke-surface, or debugging overrides.

When you invoke the assistant, it can work from more than just the transcribed
utterance: it can inspect the frontmost screen through multimodal capture
context, search the web through Brave Search, search locally through
background subagents, search and read local files, query Gmail, compact older
history when a session gets long, place results into the tray, and speak text
back aloud when asked.

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

That script reports the current status of the assistant endpoint, speech
sidecars, and the running `spoke` process.

## Advanced Overrides

If you are running isolated smoke surfaces or debugging backend wiring, a small
set of env vars is still useful. For normal use, prefer the menus.

| Variable | Default | Description |
|---|---|---|
| `SPOKE_HOLD_MS` | `200` | Spacebar hold threshold in milliseconds. |
| `SPOKE_RESTORE_DELAY_MS` | `1000` | Delay before restoring the saved pasteboard contents. |
| `SPOKE_MODEL_PREFERENCES_PATH` | unset | Override path for persisted backend/model preferences. Useful for isolated smoke/test surfaces. |
| `SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY` | unset | Enables wake-word hands-free mode. |
| `SPOKE_WAKEWORD_BACKEND` | `porcupine` | Wake-word backend. Use `openwakeword` for local model files. |
| `SPOKE_WAKEWORD_LISTEN` | `computer` | Wake word that starts hands-free dictation. |
| `SPOKE_WAKEWORD_SLEEP` | `terminator` | Wake word that returns hands-free mode to dormant. |
| `SPOKE_WAKEWORD_LISTEN_MODEL` | unset | Path to the `openWakeWord` model for the listen role. |
| `SPOKE_WAKEWORD_SLEEP_MODEL` | unset | Path to the `openWakeWord` model for the sleep role. |
| `SPOKE_WAKEWORD_TESSERA_MODEL` | unset | Optional local `openWakeWord` model for the `tessera` Return command. When it fires, the command path wins and the matching Whisper segment is suppressed. |

If you need deeper backend or smoke-surface plumbing than that, you are in
developer territory. Use
[`docs/developer-operator-surfaces.md`](docs/developer-operator-surfaces.md)
and [`docs/local-smoke-runbook.md`](docs/local-smoke-runbook.md) as the
canonical deeper surfaces rather than treating the README as a full
configuration reference.

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
├── wakeword.py           # wakeword listener backends
├── wakeword_samples.py   # wake-word sample batch generator
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
├── terraform_hud.py      # smoke-status HUD
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

## Runtime Notes

- The bundled app logs to `~/Library/Logs/Spoke.log`.
- Local MLX backends may download model weights on first use.
- The local runtime is Apple Silicon-oriented, but sidecar and cloud backends
  work independently of local model availability.

## License

MIT
