# Keyboard Grammar

Internal reference for Spoke's input gesture design. Everything routes through
one physical key (spacebar) plus timing and modifier state.

## Design principle

Clean means you never need to move your hand, use the mouse, or look at the
keyboard. One hand resting on the desk near spacebar and shift — that's the
entire physical interface. If a gesture requires reaching for escape, tab,
or any key outside the spacebar/shift cluster, it has lost cleanliness status.
The only exception is features that inherently require a full keyboard (like
text editing), which can use additional keys because you're already there.

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

Entered by releasing spacebar while shift is held during any recording. Instead
of immediately pasting or sending to the assistant, the transcribed text stays
in the overlay for the user to decide what to do with it. There is no duration
gate — any recording that ends with shift held enters staging.

### Why staging exists

Staging solves three problems with one state:

1. **Preview before commit.** You can see what was transcribed before it goes
   anywhere — into a text field or to the assistant.
2. **Cancel path.** If you're recording and realize you don't want to land the
   text, shift+release takes you to staging, then dismiss from staging. No
   audio is wasted on a command or pasted somewhere wrong.
3. **Recovery unification.** Recovery mode (paste failed, no focused text field)
   becomes "staging entered automatically" rather than a separate state with
   its own gesture vocabulary.

### Staging entry

| Context | Gesture | Result |
|---|---|---|
| Recording | Release spacebar while shift held | Enter staging with transcription |
| Recording (short, < 800ms) | Release spacebar while shift held | Recall last Q&A into staging |
| Paste failure | *(automatic)* | Enter staging with transcribed text |

### Staging gestures (spacebar + shift only)

| Gesture | Result |
|---|---|
| Spacebar tap (no shift) | Insert text into focused text field |
| Spacebar hold (no shift) | Start a new recording (replaces staged text) |
| Shift + spacebar tap | Send staged text to assistant |
| Shift tap (no spacebar) | Dismiss staging |

The grammar stays on two keys. Spacebar = "do the thing." Shift = "not the
default thing." Shift alone = back out.

### Hold-through fast path (shift spring)

When shift is still held after spacebar release, the staging overlay appears
but the system starts a ~400ms commit timer. If shift stays held through the
timer, the text auto-sends to the assistant — same end result as the current
direct command path, with a brief preview window for free.

If shift is released before the timer fires, the spring relaxes and you stay
in staging to decide manually.

**Spring animation:** While the commit timer runs, the staging overlay drifts
downward with an ease-out curve (fast start, decelerating — like pulling a
spring). Maximum displacement is small (~8–10pt over 400ms). When the timer
fires, the overlay flicks upward and the text sends to the assistant. The
upward flick is the universal "sent to assistant" visual signature.

The same pull-and-flick animation plays on a manual send from staging
(shift+spacebar tap), just with a shorter, sharper pullback (~100ms). This
way "sent" always looks the same — the hold-through version just has a longer
windup.

If shift is released before the timer, the overlay eases back to its resting
position (spring relaxing, no flick).

**Timing with transcription:** The commit timer begins when spacebar comes up,
not when transcription completes. If the user holds shift through the timer
but transcription is still running, the hold-through is treated as
pre-authorization: the text sends as soon as transcription finishes. The
spring animation still plays during the wait — the user sees the overlay
loading (partials still streaming) while it physically winds up.

### State persistence

- **Staged text persists** until consumed (inserted or sent) or replaced
  (new recording).
- **Dismissing staging** (shift tap) puts the text aside — re-entering staging
  without an intervening dictation restores it.
- **Successful dictation** (hold spacebar, paste succeeds) clears the staged
  text. Staged text is "the last thing that didn't land." Once something
  else lands successfully, that context is gone.

### Migration from current command pathway

| Flow | Before staging | After staging |
|---|---|---|
| Fast command | shift+release → assistant | shift+release → hold shift 400ms → assistant |
| Deliberate command | *(same as fast)* | shift+release → release shift → staging → shift+space → assistant |
| Cancel recording | *(no clean path)* | shift+release → staging → shift tap → dismissed |
| Preview before paste | *(no path)* | shift+release → staging → spacebar tap → paste |

The fast command path adds ~400ms of shift-hold after release. During that
time the user sees their transcription and the spring winding up — it is
preview time, not dead time.

### Relationship to recovery mode

Recovery mode becomes staging mode entered automatically when paste
verification fails. The overlay, gestures, and state persistence are
identical. The only difference is the entry trigger: manual (shift+release)
vs automatic (OCR failure).

### Text editing in staging (future)

A separate hotkey (TBD — acceptable per the design principle, since text
editing inherently requires a keyboard) could switch the overlay to editable
mode where the keyboard types into the staged text. In editable mode,
spacebar types spaces rather than triggering insert. The exit gesture from
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
| Staging (planned) | same as recovery + pop entrance | off | unfilled mic |
| Command streaming | command overlay (slow full-spectrum hue rotation, pulsing) | off | filled mic |

## Key source files

- `spoke/input_tap.py` — `SpacebarHoldDetector`, CGEventTap, state machine
- `spoke/__main__.py` — hold callbacks, pathway routing, recovery mode
- `spoke/overlay.py` — text overlay, recovery UI, animations
- `spoke/command_overlay.py` — command response overlay
