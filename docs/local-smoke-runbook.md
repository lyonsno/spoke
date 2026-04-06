# Local Smoke Runbook

This runbook is for rebuilding or repairing the local `spoke` smoke
environment on a Mac without rediscovering the same launcher/runtime details
from scratch.

## What this covers

- pinned `main`, `smoke`, and optional extra smoke launch surfaces
- WezTerm hotkeys and how to reset them
- worktree target files under `~/.config/spoke/`
- Python/runtime setup for ASR plus TTS
- the current model-directory override used on local smoke surfaces
- the escape hatch when a fresh worktree venv crashes in MLX
- the log files that tell you whether the binding, launcher, or runtime failed

## Recommended local launch layout

The current box-local convention is:

- `Ctrl+Option+Cmd+Space` -> pinned `main`
- `Ctrl+Option+Cmd+K` -> current smoke target
- optional `Ctrl+Option+Cmd+S` -> separate experimental smoke branch

The launcher target files are:

- `~/.config/spoke/main-target`
- `~/.config/spoke/dev-target`
- `~/.config/spoke/smoke-target`
- optional `~/.config/spoke/smoke-branch-target`

Prefer changing those files when you only want a hotkey to point at a
different worktree. Do not edit launcher scripts just to retarget a branch.

## Baseline worktree setup

Create or refresh the worktree you want to smoke, then sync the runtime:

```sh
git fetch origin
git worktree add /tmp/spoke-main origin/main
cd /tmp/spoke-main
uv sync --extra tts --group dev
```

The same `uv sync --extra tts --group dev` shape is the baseline for `main`,
`dev`, and smoke worktrees when they need full local ASR/TTS support.

Checked-in launchers now treat a missing target-worktree `.venv` as a repairable
bootstrap problem, not as operator memory. If `main`, `dev`, or `smoke`
launches point at a fresh worktree with no local runtime yet, the launcher will
attempt:

```sh
uv sync --directory <target-worktree> --extra tts --group dev
```

before starting `spoke`. Manual `uv sync --extra tts --group dev` is still the
cleanest way to front-load dependency/download failures, but forgetting it
should no longer make first launch dead on arrival.

If you are doing this from a sandboxed shell that cannot write the default UV
cache, use:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra tts --group dev
```

## Dependency checks that matter

When TTS or local audio suddenly stops behaving coherently, confirm the
runtime actually has the packages the smoke surface expects:

- `mlx-whisper`
- `mlx-audio`
- `mistral-common`
- the repo itself installed into the venv

Useful checks:

```sh
uv run python -c "import importlib.metadata as m; print(m.version('mlx-whisper')); print(m.version('mlx-audio')); print(m.version('mistral-common'))"
uv run pytest -q tests/test_launch_script.py
```

If `uv sync --extra tts --group dev` succeeds but the runtime still crashes on
import, treat that as an MLX/runtime problem, not as proof that the hotkey is
wrong.

## Per-worktree overrides

Each worktree can carry a `.spoke-smoke-env` file at the repo root. Use that
for branch-local launch tweaks instead of editing the shared launcher.

Common overrides:

```sh
export SPOKE_COMMAND_URL="http://localhost:8001"
export SPOKE_COMMAND_API_KEY="1234"
export SPOKE_COMMAND_MODEL_DIR="$HOME/dev/scripts/quant/models"
export SPOKE_TTS_VOICE="casual_female"
export SPOKE_TTS_MODEL="mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
```

### OmniVoice local/quant note

`spoke`'s current OmniVoice path is not MLX-native. It loads
`k2-fsa/OmniVoice` through the upstream `omnivoice` PyTorch package, so treat
MLX conversion or MLX quantization as an experiment until there is an explicit
adapter for `model_type: omnivoice`.

As of `2026-04-05`, useful Hugging Face search terms are:

```text
OmniVoice
mlx-community OmniVoice
OmniVoice-bf16
OmniVoice-4bit
OmniVoice-6bit
OmniVoice-8bit
OmniVoice 4bit
OmniVoice 6bit
OmniVoice 8bit
omnivoice mlx
```

Current state on this box:

- upstream repo: `k2-fsa/OmniVoice`
- published mirrors found: `mlx-community/OmniVoice-bf16`, `drbaph/OmniVoice-bf16`
- published `4bit` / `6bit` / `8bit` OmniVoice repos found: none
- cached upstream snapshot footprint: about `3.0G` total (`model.safetensors`
  about `2.3G`, `audio_tokenizer/` about `768M`)

If you build or stage a local OmniVoice variant, point `SPOKE_TTS_MODEL` at the
local folder path instead of a Hub ID. The current loader forwards the value to
`OmniVoice.from_pretrained(...)`, so a local model directory is already a valid
shape:

```sh
export SPOKE_TTS_MODEL="/path/to/OmniVoice-local"
export SPOKE_TTS_VOICE="female, british accent"
```

### When a fresh worktree venv is sick

Some fresh worktrees can still abort on bare `import mlx.core` or
`import spoke.__main__` with `NSRangeException` in `libmlx.dylib`, even after
`uv sync`.

When that happens, keep running the target worktree's code but point the
launcher at a known-good interpreter:

```sh
export SPOKE_VENV_PYTHON="/private/tmp/spoke-dev/.venv/bin/python"
```

Put that in the target worktree's `.spoke-smoke-env`. Do not hardcode the
override into the shared launcher if the problem is only on one local surface.

## WezTerm hotkey reset

The live keymap is in:

```sh
~/.config/wezterm/wezterm.lua
```

When a hotkey stops working:

1. Confirm the binding still points at the expected launcher script.
2. Confirm the relevant `~/.config/spoke/*-target` file still points at a real worktree.
3. Reload or restart WezTerm after editing the config.
4. Press the hotkey once, then inspect the matching log file.

If `Space` is the only broken binding while `K` or `S` still work, that often
means the launcher or runtime failed, not that WezTerm ignored the key. Check
the log before changing the key binding again.

## Launcher logs

Use the logs to separate binding failures from runtime failures:

- `~/Library/Logs/spoke-main-launch.log`
- `~/Library/Logs/spoke-dev-launch.log`
- `~/Library/Logs/spoke-smoke-launch.log`
- optional `~/Library/Logs/spoke-smoke-branch-launch.log`

Typical interpretations:

- no new log entry after keypress: likely binding or reload problem
- log shows `Launcher bootstrap command:` followed by `uv sync ...`: target worktree was missing `.venv`; launcher is repairing it in place
- log entry with `No repo .venv Python found and UV launcher is unavailable.`: runtime not provisioned
- log entry followed by MLX/`libmlx.dylib` crash: launcher worked, target runtime is sick
- log shows the expected target path and child command but the app still feels wrong: the wrong surface may be healthy enough to start but not the one you meant to smoke; verify the target file and branch badge

## Branch identity while smoking

Local smoke/test branches should carry the menubar branch badge patch by
default so the running app can identify its own branch in the dropdown.

That badge is the fastest truth surface when the same box can launch `main`,
`dev`, and smoke branches in quick succession.

## Minimal repair checklist

If you need the short version:

1. Verify the target file points at the intended worktree.
2. Restore `.spoke-smoke-env` with the command-model dir and any TTS overrides.
3. Run `uv sync --extra tts --group dev` yourself if you want to front-load install errors; otherwise let the launcher bootstrap on first run.
4. If the worktree venv crashes in MLX, set `SPOKE_VENV_PYTHON` to a known-good interpreter.
5. Reload WezTerm.
6. Press the hotkey once and read the corresponding launcher log.
7. Ask the human whether the spacebar is actually working; logs cannot prove event-tap health.
