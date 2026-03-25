# dictate

Global hold-to-dictate for macOS. Hold spacebar anywhere on the system — in your terminal, editor, browser, wherever — to record audio, transcribe it via a local Whisper server, and paste the result at your cursor.

Built with PyObjC. No Swift, no Xcode.

## How it works

```
Hold spacebar (400ms+) → Record audio → Transcribe on sidecar → Paste at cursor
```

A CGEventTap intercepts spacebar globally. Quick taps pass through as normal spaces. Holds trigger recording via `sounddevice`, send the audio to an OpenAI-compatible Whisper endpoint (e.g., `mlx-audio` on a sidecar machine), and inject the transcribed text via pasteboard + synthetic Cmd+V.

Runs as a menubar accessory app — mic icon in the menubar, no Dock icon.

## Architecture

```
dictate/
├── input_tap.py     # CGEventTap spacebar state machine (IDLE → WAITING → RECORDING)
├── capture.py       # sounddevice audio recording + RMS amplitude callback
├── transcribe.py    # httpx client for /v1/audio/transcriptions
├── inject.py        # pasteboard save → set text → Cmd+V → restore
├── menubar.py       # NSStatusItem with mic/mic.fill icons
└── __main__.py      # DictateAppDelegate — wires all layers together
```

Each layer is independent and testable in isolation.

## Requirements

- macOS 11+ (for SF Symbols and CGEventTap APIs)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- `portaudio` (for audio capture via sounddevice)
- A Whisper transcription server with an OpenAI-compatible `/v1/audio/transcriptions` endpoint

### System dependencies

```sh
brew install portaudio
```

### Transcription server

Any server that implements OpenAI's `/v1/audio/transcriptions` API works. Options for Apple Silicon:

```sh
# mlx-audio (Whisper on MLX):
uv tool install "mlx-audio[server]"
mlx-audio-server --host 0.0.0.0 --port 8000

# mlx-qwen3-asr (Qwen3-ASR on MLX — faster for short audio, no 30s padding):
# See https://github.com/moona3k/mlx-qwen3-asr
```

## Usage

```sh
# Clone and install
git clone https://github.com/lyonsno/dictate.git
cd dictate
uv sync

# Run (point at your Whisper server)
DICTATE_WHISPER_URL=http://<sidecar-ip>:8000 uv run dictate
```

On first run, macOS will prompt for Accessibility permission. Grant it to your terminal app (Terminal.app, iTerm2, Ghostty, etc.) in System Settings → Privacy & Security → Accessibility.

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DICTATE_WHISPER_URL` | Yes | — | Whisper server base URL |
| `DICTATE_WHISPER_MODEL` | No | `mlx-community/whisper-large-v3-turbo` | Model identifier |
| `DICTATE_HOLD_MS` | No | `400` | Hold threshold in milliseconds (must be > 0) |
| `DICTATE_RESTORE_DELAY_MS` | No | `1000` | Pasteboard restore delay in milliseconds |

## Tests

```sh
uv run pytest -v
```

67 tests covering the state machine, WAV encoding, HTTP client, injection, and menubar — all run headless with mocked PyObjC.

## Roadmap

### Phase 1 — Core dictation ✅

Global hold-to-dictate: spacebar hold detection, audio capture, Whisper transcription, paste at cursor. Menubar accessory with mic icon. 45 tests, all headless.

### Phase 2 — Visual feedback

**Menubar amplitude animation.** The menubar icon becomes a live visualizer while recording — a glowing bar that oscillates with voice amplitude, inspired by Claude Code's hold-to-dictate cursor. The RMS amplitude callback in `capture.py` already fires per-chunk; this phase connects it to a visual.

- [ ] Animated NSStatusItem that responds to amplitude in real time
- [ ] Smooth interpolation between amplitude samples (avoid jitter)
- [ ] Idle/recording state transitions with visual continuity

**Screen-border amplitude glow.** A subtle, ambient glow around the entire screen border that pulses with voice amplitude while recording. Invisible at idle; fades in on hold-start, breathes with speech, fades out on release. The goal is peripheral awareness — you feel it more than you see it.

- [ ] Borderless, transparent, click-through `NSWindow` at screen frame size, above all other windows
- [ ] Core Animation layer for the border glow (GPU-composited, no per-frame redraws)
- [ ] Amplitude-driven opacity/intensity: rise fast, decay slow (exponential moving average on RMS)
- [ ] Match screen corner radius (Big Sur+ rounded corners)
- [ ] Warm/neutral color — system accent or soft warm white, not attention-grabbing
- [ ] Fade in on recording start (~100ms), fade out on release (~200-300ms)
- [ ] Multi-display support (one window per screen, listen for display config changes)
- [ ] `setIgnoresMouseEvents_(True)` — must not intercept clicks
- [ ] `fullScreenAuxiliary` collection behavior for full-screen app compatibility

**Frosted transcription overlay.** A semi-transparent, system-font overlay appears on screen showing the transcription as it's produced. When recording ends and text is pasted at the cursor, the overlay fades out — the fade distracts from the paste so it feels seamless rather than jarring.

- [ ] Borderless `NSWindow` overlay with frosted/vibrancy background
- [ ] System font text rendering (SF Pro, matched to system appearance)
- [ ] Smooth fade-out animation timed to coincide with paste injection
- [ ] Overlay centered horizontally, fixed near bottom of screen (not cursor-tracking)
- [ ] Dark mode / light mode support via system appearance

### Phase 3 — Two-tier transcription

**Local preview + sidecar final.** A two-tier architecture for low-latency feedback without stitching complexity:

- **Preview tier (local):** A small model (MLX Whisper Tiny, ~75MB) runs on the local machine during recording, producing fast interim results shown in the overlay. Preview text is disposable — it exists only for visual feedback while you speak.
- **Final tier (sidecar):** On spacebar release, the full audio buffer is sent to a high-accuracy model on the sidecar (Qwen3-ASR 0.6B Q8, ~150-200ms for short clips) for a single clean transcription pass. This is what gets injected. No stitching, no chunk boundary reconciliation.

This avoids the fundamental problem with streaming STT stitching: chunk boundaries split words, context is lost between segments, and reconciling overlapping partial results is fragile. Full-buffer final transcription sees the complete utterance with full context.

- [ ] Local preview model integration (MLX Whisper Tiny) — modular, swappable
- [ ] Sidecar model switch from Whisper to Qwen3-ASR 0.6B Q8 (OpenAI-compatible API)
- [ ] Preview text shown in overlay during recording (lighter weight / lower opacity)
- [ ] Final transcription replaces preview on release
- [ ] Graceful fallback if local preview model is unavailable (amplitude-only feedback)
- [ ] Future: swap local preview from MLX to CoreML for ANE offload (zero unified memory pressure, lower power)

### Phase 4 — Polish

- [ ] **Self-contained DMG distribution** — PyInstaller bundle with embedded Python, MLX, and model downloader. No terminal, no dependencies. Drag to Applications and go.
- [ ] LaunchAgent for auto-start at login
- [ ] Config file (`~/.config/dictate/config.json`) — hold threshold, server URL, model, overlay preferences
- [ ] **Toggle mode** — menubar toggle between hold-to-record (current spacebar behavior) and press-to-start/press-to-stop with a configurable hotkey. Toggle mode for accessibility and workflows where holding is impractical; configurable hotkey for users who need bare spacebar (gaming, etc.)
- [ ] Settable hotkey via menubar dropdown (any key/modifier combo)

## License

MIT
