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

### Whisper server

Any server that implements OpenAI's audio transcription API works. For local inference on Apple Silicon:

```sh
# On your sidecar machine (or locally):
uv tool install "mlx-audio[server]"
mlx-audio-server --host 0.0.0.0 --port 8000
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
| `DICTATE_HOLD_MS` | No | `400` | Hold threshold in milliseconds |

## Tests

```sh
uv run pytest -v
```

45 tests covering the state machine, WAV encoding, HTTP client, injection, and menubar — all run headless with mocked PyObjC.

## Roadmap

- [ ] Menubar amplitude animation (Phase 2)
- [ ] Frosted overlay with interim transcription text (Phase 2)
- [ ] Incremental transcription during recording (Phase 3)
- [ ] LaunchAgent for auto-start at login (Phase 4)
- [ ] Config file (`~/.config/dictate/config.json`) (Phase 4)

## License

MIT
