# Anaphora: Crash-Hardening Tests

## Scope / Assumptions

- Test-only review of the crash-hardening follow-up on
  `codex/spoke-butterfingers-0401`.
- This remained a provisional self-review lane via spawned subagents, not
  independent aposkepsis under the strict session-isolation rule.
- Reviewed surfaces: `tests/test_capture.py`, `tests/test_delegate.py`,
  `spoke/capture.py`, and `spoke/__main__.py`.

## Findings

### Cycle 1

1. Medium, addressed: runtime-phase writer race was untested.
2. Medium, addressed: callback-thread isolation was unpinned.
3. Medium, addressed: stop-path teardown/stale-callback behavior was untested.

### Cycle 2

1. High, addressed: pending queued callback work could still survive into
   teardown without direct coverage.
2. Medium, addressed: off-thread tests did not prove non-blocking callback
   behavior.
3. Medium, addressed: temp-path isolation was only indirectly covered.

### Cycle 3

1. Medium, addressed: cross-recording callback isolation on the real second
   `start()` path was not directly pinned.

## Addressed In This Pass

- Added concurrent runtime-phase snapshot tests that force competing writes and
  assert distinct temp filenames.
- Moved amplitude/VAD callback dispatch off the PortAudio thread and added
  tests for off-thread plus non-blocking behavior.
- Disabled callback delivery before teardown, added callback-generation
  rollover, and covered stale-event dropping on `stop()` and on a real second
  `start()`.
- Added stale-callback guard coverage for late `_audio_callback()` invocations
  after stream teardown.

## Verification Record

Targeted verification after closing the test-only review loop:

```bash
uv run pytest -q tests/test_capture.py tests/test_delegate.py
```

Result:

- `188 passed in 2.56s`

Broader combined verification after the test lane came back clean:

```bash
uv run pytest -q tests/test_capture.py tests/test_delegate.py tests/test_transcribe_local.py tests/test_transcribe_qwen.py tests/test_input_tap.py tests/test_tray.py tests/test_launch_script.py tests/test_mic_permission.py
```

Result:

- `355 passed in 10.53s`

## Residual Risk

- The startup MLX-import abort shape now leaves working breadcrumbs and the
  PortAudio callback path is materially hardened, but the live runtime still
  needs human soak to confirm whether the newly observed recording-time crash
  family is actually resolved on the smoke surface.
