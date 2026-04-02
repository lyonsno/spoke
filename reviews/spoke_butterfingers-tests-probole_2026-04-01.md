# Probole: Butterfingers Tests

## Scope

Review the Butterfingers test surface beyond the immediate diff. This lane is
test-only: focus on whether the tests constrain the release-order contract,
overlay recall behavior, and launcher bootstrap fallbacks tightly enough to
catch the live regressions the human reported.

## Commit Range

- Base: `origin/main-next` at `29c123f`
- Review surface: `codex/spoke-butterfingers-0401`
- Include the uncommitted regression-fix delta that followed the initial
  Butterfingers landing on top of `7c50c8b`

## How To Run

```bash
uv run pytest -q tests/test_input_tap.py tests/test_delegate.py tests/test_tray.py tests/test_launch_script.py
```

Targeted subsets that matter most:

```bash
uv run pytest -q tests/test_input_tap.py -k 'enter or ForceEnd or safety_timer'
uv run pytest -q tests/test_delegate.py -k 'operator_toggle or recall'
uv run pytest -q tests/test_launch_script.py -k 'bootstrap or failed_bootstrap'
```

## Prior Review State

- Review is provisional only. The reviewer is a spawned subagent from the
  implementation thread, so this is useful self-review rather than independent
  aposkepsis.
- Cycle 1 already looked at the test surface and reported missing coverage for
  `force_end`, safety-timeout/send-disarm, WAITING-branch release order, and
  launcher bootstrap failure fallback.

## Non-Diff Surfaces To Read

- `docs/keyboard-grammar.md`
- `spoke/input_tap.py`
- `spoke/__main__.py`
- `tests/test_tray.py`

## Questions For Review

1. Do the tests directly constrain both halves of Enter consumption for the
   space-rooted chords, not just keyDown?
2. Do they distinguish overlay recall from dismiss when the command overlay is
   already visible?
3. Do they pin the WAITING and RECORDING branches separately for the
   `Enter up -> Space up` send window and the `Space up first` toggle path?
4. Do launcher bootstrap tests cover fallback when bootstrap fails, not just
   when the venv is missing?
