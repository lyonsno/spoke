# Anaphora: Butterfingers Tests

## Scope / Assumptions

- Test-only review of the Butterfingers branch.
- This is a provisional self-review lane via spawned subagents, not an
  independent session under the strict aposkepsis rule.
- Reviewed surfaces: `tests/test_input_tap.py`, `tests/test_delegate.py`,
  `tests/test_launch_script.py`, with contract cross-checks against
  `docs/keyboard-grammar.md`, `spoke/input_tap.py`, and `spoke/__main__.py`.

## Findings

### Cycle 1

1. Medium, addressed: missing coverage for `force_end` and
   `safetyTimerFired_` while the send-disarm path was armed.
2. Medium, addressed: missing WAITING-branch release-order tests for
   `Space up first` toggle vs `Enter up then Space up` send.
3. Low, addressed: launcher bootstrap tests did not cover bootstrap failure
   followed by fallback to `uv run`.

### Cycle 2

1. Medium, addressed: `_suppress_enter_keyup` was not directly constrained,
   so Enter could have remained partially leaked even with keyDown tests.
2. Low, addressed: the visible-overlay dismiss side of
   `toggle_assistant_overlay` was not directly asserted.

## Addressed In This Pass

- Added Enter keyUp suppression tests for both WAITING and RECORDING
  chord states.
- Added WAITING-mode release-order tests, plus send-disarm retention on
  `force_end` and safety timeout.
- Added launcher bootstrap failure-fallback tests.
- Added direct delegate coverage for overlay recall from cached overlay text
  and for the visible-overlay dismiss branch.

## Verification Record

Implementation-local verification after addressing both cycles:

```bash
uv run pytest -q tests/test_input_tap.py tests/test_delegate.py tests/test_tray.py tests/test_launch_script.py
```

Result:

- `269 passed in 9.56s`

## Residual Risk

- The test surface is now materially tighter for the Butterfingers detector
  and overlay routing, but it does not resolve human-visible ambiguity around
  automatic recovery/tray entry after OCR verification failure. That remaining
  behavior needs fresh smoke rather than more local test expansion first.
