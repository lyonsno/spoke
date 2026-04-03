# Anaphora: Leaf 2 (`spoke/capture.py`) — 2026-04-03

Type: anaphora
Scope: `/private/tmp/spoke-careless-whisper-0402/spoke/capture.py`
Review target: Gemini commits `17985ef`, `30d661e`, `7dfe889`
Reviewer stance: independent leaf review, no implementation edits

## Scope assumptions

- I reviewed only the `spoke/capture.py` surface assigned to leaf 2.
- I treated existing local edits in `/private/tmp/spoke-careless-whisper-0402/spoke/__main__.py` as out of scope and did not modify them.
- I used `tests/test_capture.py` as the focused verification surface, then added direct runtime repros for grace-period behavior that the current tests do not cover.

## Findings ordered by severity

1. Severity: High
   Title: Grace period manufactures speech before any speech exists
   Evidence:
   - `spoke/capture.py:143` arms `_grace_chunks_remaining` at `start()`, before any voiced frame has been observed.
   - `spoke/capture.py:318-323` immediately flips `_is_speech` to `True` on the first callback while grace is active, even for silent input.
   - `spoke/capture.py:350-358` then routes those silent opening chunks through the in-speech path.
   Why this is material:
   - A grace period should extend a previously detected speech run; it should not assert “speech” from recording start.
   - This defeats the intended VAD gating during initial silence and makes the VAD state callback report speech for silence-only openings.
   Concrete repro:
   ```bash
   uv run python -c 'import numpy as np; from spoke.capture import AudioCapture; cap=AudioCapture(); cap.start(); cap._stream=type("S",(),{"active":True,"stop":lambda self:None,"close":lambda self:None})(); silence=np.zeros((1024,1),dtype=np.float32); [cap._audio_callback(silence,1024,None,0) for _ in range(10)]; print(cap._is_speech, cap._grace_chunks_remaining, len(cap._current_segment_chunks), len(cap._speech_chunks), len(cap._frames))'
   ```
   Observed result:
   - `True 68 10 10 10`
   - After ten silent chunks, the recorder still thinks it is in speech and has accumulated those silent chunks as speech payload.
   Fix direction:
   - Start grace only after real speech has been detected, or treat grace as a post-speech hold timer rather than a startup mode.

2. Severity: High
   Title: Silence-only recordings can emit bogus segment callbacks and untrimmed “speech” WAVs
   Evidence:
   - Because the startup grace marks silence as speech, `spoke/capture.py:371-373` can queue a segment composed entirely of silence once grace expires and the silence counter reaches `MIN_SILENCE_FRAMES`.
   - `spoke/capture.py:228-235` then uses `_speech_chunks` for final WAV output, so silence collected during grace is treated as trimmed speech content on `stop()`.
   Why this is material:
   - The commit intent was to trim dead air on release and opportunistically slice speech. The current behavior does the opposite for silence-only openings: it can emit silence to downstream opportunistic transcription and returns a non-empty WAV for a short silence-only hold.
   Concrete repros:
   ```bash
   uv run python -c 'import numpy as np; from unittest.mock import MagicMock; from spoke.capture import AudioCapture; cap=AudioCapture(); cb=MagicMock(); cap.start(segment_callback=cb); cap._stream=type("S",(),{"active":True,"stop":lambda self:None,"close":lambda self:None})(); silence=np.zeros((1024,1),dtype=np.float32); [cap._audio_callback(silence,1024,None,0) for _ in range(95)]; cap.stop(); print("calls", cb.call_count); print("first_len", len(cb.call_args_list[0][0][0]) if cb.call_count else 0)'
   ```
   Observed result:
   - `calls 1`
   - `first_len 182316`
   - A silence-only session produced one queued segment callback.
   ```bash
   uv run python -c 'import numpy as np; from spoke.capture import AudioCapture; cap=AudioCapture(); cap.start(); cap._stream=type("S",(),{"active":True,"stop":lambda self:None,"close":lambda self:None})(); silence=np.zeros((1024,1),dtype=np.float32); [cap._audio_callback(silence,1024,None,0) for _ in range(10)]; wav=cap.stop(); print(len(wav))'
   ```
   Observed result:
   - `20524`
   - Ten silent chunks produced a non-empty final WAV (`10 * 1024 * 2 + 44` bytes), so the silence was not trimmed away.
   Fix direction:
   - Do not append grace-only silence to `_current_segment_chunks` / `_speech_chunks` before the first real speech trigger.
   - Ensure segment emission and final WAV trimming key off actual speech detection, not startup grace bookkeeping.

## Verification evidence

- Focused suite:
  - `uv run pytest tests/test_capture.py -q`
  - Result: `24 passed in 0.08s`
- Gap exposed by review:
  - The current tests pass while still allowing the two grace-period regressions above, so the leaf should add fail-first coverage for:
    - initial silence before first speech
    - silence-only recording with `segment_callback`
    - silence-only recording without `segment_callback`

## Review conclusion

Leaf 2 is not ready to land as written. The 5-second grace-period change in `7dfe889` is wired as a startup speech assertion instead of a post-speech hold, and that breaks both VAD state correctness and silence trimming/segment emission on quiet openings.
