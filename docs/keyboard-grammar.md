# Keyboard Grammar

Internal reference for Spoke's input gesture design. Everything routes through
one physical key (spacebar) plus timing and modifier state.

## Core gestures

| Gesture | Duration | Shift at release | Result |
|---|---|---|---|
| Tap | < 400ms | No | Normal space character (forwarded to app) |
| Hold | ≥ 400ms | No | **Text pathway** — record, transcribe, paste at cursor |
| Hold | ≥ 400ms | Yes | **Command pathway** — record, transcribe, send to assistant |
| Hold | ≥ 400ms but < 800ms | Yes | **Recall** — skip transcription, show last Q&A pair |

The hold threshold defaults to 400ms (configurable via `SPOKE_HOLD_MS`, which
sets the threshold in the `SpacebarHoldDetector`).

## The fork-at-release principle

The intent fork happens at release, not at press. You hold spacebar, speak, and
at the moment you let go you decide what the utterance was:

- **Clean release** → text insertion (dictation)
- **Shift held at release** → command (send to assistant)

Shift can be pressed at any point during the hold — it is latched via
`kCGEventFlagsChanged`, so pressing shift after you start speaking still
routes to the command pathway. What matters is whether shift is down when
spacebar comes up.

## Modifier blocking

These modifier combinations pass through to the OS untouched and never trigger
recording:

- **Cmd+Space** — Spotlight, input source switching, etc.
- **Ctrl+Space** — system use
- **Alt/Option+Space** — system use

Shift is deliberately excluded from the blocking mask. It is a routing signal,
not a recording gate.

## Recording state machine

```
IDLE ──[spacebar down]──→ WAITING ──[400ms timer]──→ RECORDING ──[spacebar up]──→ IDLE
                              │                           │
                     [spacebar up early]          [300s safety timeout]
                              │                           │
                              ▼                           ▼
                      forward space char            force stop, IDLE
```

- **IDLE**: ready for input.
- **WAITING**: spacebar is down, hold timer running. Key repeats suppressed.
- **RECORDING**: audio capture active, overlays visible.

Synthetic space forwarding on early release uses a 100ms auto-clear timeout in
case the forwarded events are lost.

## Text pathway (clean release)

1. Audio capture stops.
2. Final transcription runs on a background thread.
3. Text is injected at the cursor via paste.
4. If no focused text field is detected → enters **recovery mode**.

## Command pathway (shift release)

1. Audio capture stops.
2. Audio is transcribed to text (the "utterance").
3. Utterance is streamed through the command client (assistant).
4. Command overlay shows the utterance and the streamed response.

### Short shift-hold (< 800ms with shift at release)

Treated as an instant recall/dismiss gesture — no transcription attempt:

- If the command client has history → recall and display the last Q&A pair.
- If no history or no command client → dismiss the command overlay.

## Recovery mode

Entered when text pathway transcription succeeds but paste verification fails
(no focused text field detected via OCR).

### Recovery gestures

| Gesture | Result |
|---|---|
| Spacebar release (no shift) | Retry paste — OCR verifies, bounces on failure |
| Shift+spacebar release | Send the transcribed text to assistant as a command |

### Recovery mouse targets

Three interactive columns in the recovery overlay:

| Column | Action |
|---|---|
| Dismiss | Restore original clipboard, hide overlay |
| Insert | Same as spacebar retry (paste + OCR verify) |
| Clipboard | Toggle between transcribed text and original clipboard on pasteboard |

### Recovery feedback

- **Bounce**: retry failed (scale 1.0 → 0.97 → 1.0, 80ms). Signals "nowhere to paste."
- **Insert flash**: Insert column flashes red (0.3s) on rejection.
- **Entrance pop**: scale 1.015 → 1.0 (200ms) on first appearance.

## Staging mode (planned)

Entered by releasing spacebar with shift held after a recording (the same
gesture that currently routes to the command pathway for long holds, and
recall for short holds). Instead of immediately pasting or sending to the
assistant, the transcribed text stays in the overlay for the user to decide
what to do with it.

### Staging mode entry

| Current state | Gesture | Result |
|---|---|---|
| Recording | Release spacebar while shift held (≥ 800ms) | Enter staging with transcribed text |

### Staging mode gestures

| Gesture | Result |
|---|---|
| Spacebar tap | Insert text into focused text field |
| Spacebar hold | Start a new recording (replaces staged text) |
| Enter | Send staged text to assistant |
| Shift tap | Cancel/dismiss staging |

### Staging mode state persistence

- **Staged text persists** until consumed (inserted into field or sent to
  assistant) or replaced (new recording).
- **Dismissing staging** (shift tap) puts the text aside — re-entering staging
  without an intervening dictation or recording restores it.
- **Normal dictation** (hold spacebar without shift, paste into field) clears
  the staged text. The staged text is "the last thing that didn't go where
  you wanted it." Once you've successfully dictated something else, that
  context is gone.

### Cancel gesture (during recording or staging)

| Gesture | Result |
|---|---|
| Tap shift while spacebar held | Cancel recording, discard audio |
| Tap shift (in staging, no spacebar) | Cancel/dismiss staging |

The shift-tap cancel is the universal "abort" — same gesture in both contexts.
It requires no hand repositioning since shift is adjacent to spacebar.

### Relationship to recovery mode

Recovery mode (triggered by OCR paste verification failure) is a subset of
staging mode. When staging mode ships, recovery mode becomes "staging mode
entered automatically because the paste failed" rather than a separate state.
The gestures and overlay layout are the same.

### Text editing in staging (future)

A separate hotkey (TBD — acceptable because text editing requires a keyboard)
could switch the overlay to editable mode where the keyboard types into the
staged text. In editable mode, spacebar types spaces rather than triggering
insert. Enter would send to the assistant (unchanged). The exit gesture from
editable mode is TBD.

## Recording cap

On machines with < 36GB RAM running local inference: 20-second maximum
recording duration. The last 3 seconds show a linear countdown glow. At 20s,
recording force-stops. No cap in sidecar mode.

## Visual feedback summary

| State | Overlay | Glow | Menu icon |
|---|---|---|---|
| Idle | hidden | off | unfilled mic |
| Recording | live preview (typewriter) | amplitude-reactive border | filled mic |
| Transcribing | preview holds | fading | filled mic |
| Recovery | three-column interactive | off | unfilled mic |
| Command streaming | command overlay (violet, pulsing) | off | filled mic |

## Key source files

- `spoke/input_tap.py` — `SpacebarHoldDetector`, CGEventTap, state machine
- `spoke/__main__.py` — hold callbacks, pathway routing, recovery mode
- `spoke/overlay.py` — text overlay, recovery UI, animations
- `spoke/command_overlay.py` — command response overlay
