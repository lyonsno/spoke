# Probole: Crash-Hardening Tests

## Scope

Review the test surface for the new crash-hardening follow-up only. This is a
test-focused lane: ensure the tests actually constrain the breadcrumb writer
race and the audio-callback handoff shape, rather than merely asserting that
the app can start.

## Commit Range

- Base: `46bfb34^` on `codex/spoke-butterfingers-0401`
- Review surface: current uncommitted crash-hardening follow-up on
  `codex/spoke-butterfingers-0401`

## How To Run

```bash
uv run pytest -q tests/test_capture.py tests/test_delegate.py
```

Likely focused subsets:

```bash
uv run pytest -q tests/test_capture.py -k 'callback or start or stop'
uv run pytest -q tests/test_delegate.py -k 'warmup or runtime'
```

## Prior Review State

- No prior review yet for this crash-hardening follow-up.
- Earlier Butterfingers test reviews are adjacent but not sufficient; they did
  not cover runtime-phase snapshot races or callback-thread isolation.

## Non-Diff Surfaces To Read

- `spoke/capture.py`
- `spoke/__main__.py`
- `tests/test_capture.py`
- `tests/test_delegate.py`
- `~/Library/Logs/DiagnosticReports/Python-2026-04-01-212549.ips`

## Questions For Review

1. Do the tests constrain that runtime-phase snapshot writes remain correct
   under repeated or concurrent phase updates, rather than only checking one
   happy-path write?
2. Do the tests constrain that PortAudio-thread callback work is handed off
   off-thread before UI-bound callbacks run?
3. Do the tests pin teardown behavior so worker/dispatch threads do not outlive
   `stop()` or leak stale callbacks across recordings?
4. Are there obvious acceptance gaps where a regression could reintroduce the
   callback-thread crash shape without failing the tests?
