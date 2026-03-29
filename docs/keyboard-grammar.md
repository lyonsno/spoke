# Keyboard Grammar

Internal reference for Spoke's input gesture design. Three physical keys —
spacebar, shift, enter — plus timing and modifier state.

## Design principle

Clean means you never need to move your hand, use the mouse, or look at the
keyboard. One hand resting on the desk near spacebar, shift, and enter —
that's the entire physical interface. Spacebar and shift are the primary
surface (adjacent keys, one hand, never leaves typing position). Enter is the
third key, used only in deliberate contexts: when the tray is up and you're
reviewing text, or held during recording for a confident send. You're already
in a decision-making posture when Enter becomes relevant, so reaching one key
over is not a flow break.

If a gesture requires reaching for escape, tab, or any key outside the
spacebar/shift/enter cluster, it has left the interaction surface. The only
exception is features that inherently require a full keyboard (like text
editing), which can use additional keys because you're already there.

## Key identity

Each key has a consistent identity across the entire grammar:

- **Spacebar** = go, do the thing, commit. Tap to insert, hold to record.
- **Shift** = not the default thing, navigate, review, back out. Enter the
  tray, scrub through history, dismiss, delete.
- **Enter** = send to assistant. The universal "submit" key.

## Core gestures

| Gesture | Result |
|---|---|
| Spacebar tap (< 400ms, no shift) | Normal space character (forwarded to app) |
| Spacebar hold (≥ 400ms), clean release | **Text pathway** — record, transcribe, paste at cursor |
| Spacebar hold, shift held at release | **Enter tray** — record, transcribe, stage for review |
| Spacebar hold, enter held at release | **Send to assistant** — record, transcribe, stream to assistant |
| Short spacebar hold (< 800ms), shift at release | **Recall** — enter tray with last tray entry (no recording) |

The hold threshold defaults to 400ms (configurable via `SPOKE_HOLD_MS`, which
sets the threshold in the `SpacebarHoldDetector`).

## Disposition at release

For the ordinary recording path, utterance disposition is decided when capture
ends, not when recording starts. You hold spacebar, speak, and at the moment
capture ends you decide what the utterance becomes:

- **Clean release** → text insertion (dictation)
- **Shift held at release** → tray (review before committing)
- **Enter held at release** → assistant (confident send)

Shift and enter can be pressed at any point during the hold — shift is latched
via `kCGEventFlagsChanged`, so pressing shift after you start speaking still
routes to the tray. For the ordinary path, what matters is what's held when
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
4. If no focused text field is detected → enters **tray** automatically.

## Command pathway (enter held at release)

1. Audio capture stops.
2. Audio is transcribed to text (the "utterance").
3. Utterance is streamed through the command client (assistant).
4. Command overlay shows the utterance and the streamed response.

This is the fast path for confident sends — you know before you finish
speaking that this is a command. Hold enter, release spacebar, done.

## Latched recording extension (planned)

Latched recording is a mid-capture state transition, not a change in
end-of-recording intent. The utterance's disposition is still chosen only
when capture ends.

### Entry

- Start with an ordinary recording hold.
- During active recording, tap shift while keeping spacebar held.
- That shift tap changes the recording state to **latched recording**.
- Once latched, releasing spacebar no longer ends capture.

This is deliberately different from the ordinary shift-at-release tray route.
The shift tap does not yet decide "tray" or "assistant." It only says "keep
recording even after spacebar comes up."

### Exits from latched recording

| Gesture | Result |
|---|---|
| Enter | Stop capture, transcribe, send to assistant |
| Hold shift + press spacebar, then release spacebar | Stop capture, transcribe, enter tray |

Latched recording is intentionally not a second direct-text pathway. Clean
release text insertion remains the ordinary pre-latch route. Once a recording
has been latched, it exits through tray or assistant.

### Relationship to the tray

Tray entry does not change. From the tray, spacebar hold (≥ 400ms) still
starts a new recording and pushes the current tray text down in the stack.
If that new recording should continue hands-free, tap shift during the
recording to latch it.

## The tray

The tray is a speech-native stacked clipboard. It holds recent transcriptions
for review, insertion, sending to the assistant, or later retrieval. It is the
central interaction surface between recording and committing.

### Why the tray exists

1. **Preview before commit.** See what was transcribed before it goes anywhere.
2. **Cancel path.** Shift+release enters the tray instead of pasting. Dismiss
   from the tray to cancel. No audio wasted on a bad paste or accidental send.
3. **History.** Every transcription that enters the tray is stacked. Scrub back
   through recent transcriptions with two keys.
4. **Speech-native clipboard.** The assistant can add arbitrary content to the
   tray via tool calls. Combined with voice commands, this is a full clipboard
   manager operated entirely by speech and three keys.
5. **Recovery unification.** Paste failure (no focused text field) enters the
   tray automatically rather than being a separate "recovery mode" state.

### Tray entry

| Context | Gesture | Result |
|---|---|---|
| Recording | Shift held + release spacebar | Enter tray with transcription |
| Recording (short, < 800ms) | Shift held + release spacebar | Recall last tray entry (no recording) |
| Paste failure | *(automatic)* | Enter tray with transcribed text |

### Tray gestures

| Gesture | Result |
|---|---|
| Spacebar tap (~150ms) | Insert tray text at cursor |
| Spacebar hold (≥ 400ms) | Start new recording (pushes current text down in stack) |
| Enter | Send tray text to assistant |
| Shift + release spacebar | Navigate up (more recent item; dismiss at top) |
| Spacebar + tap shift | Navigate down (older item) |
| Shift held + double-tap spacebar | Delete current tray entry |

The gesture vocabulary uses three keys with consistent identity: spacebar
commits (insert, record), shift navigates and manages (up, down, delete),
enter sends.

The delete gesture requires a double-tap of spacebar while shift is held
(two taps within ~300ms). The first tap navigates up as normal; the second
tap within the window deletes instead. This makes deletion deliberate —
you cannot accidentally delete with a single shift+spacebar tap.

### Navigation model

The tray is a vertical stack. The most recent transcription is at the top.

- **Shift + release spacebar** moves up toward more recent entries. Navigating
  up past the most recent entry dismisses the tray.
- **Spacebar + tap shift** moves down toward older entries. Navigating down
  past the oldest entry stops (no wrap).
- **Shift held + tap spacebar** deletes the currently displayed entry and
  shows the next one (or dismisses if the stack is now empty).

Navigation feels like rocking between two adjacent keys to scrub through
a list. The visual overlay updates to show the current entry's text as you
navigate.

### Tray stack lifecycle

- **New recording from tray** (spacebar hold ≥ 400ms) pushes the current tray
  text down in the stack and starts recording. The new transcription becomes
  the top of the stack when it lands.
- **Insert** (spacebar tap) consumes the current entry — it is removed from
  the stack after successful paste.
- **Send to assistant** (Enter) consumes the current entry.
- **Delete** (shift held + tap spacebar) removes the current entry.
- **Dismiss** (navigate up past top) hides the tray but preserves the stack.
  Re-entering the tray (shift+release from a short hold, or paste failure)
  shows the top of the stack.
- **Successful text pathway dictation** (hold spacebar, clean release, paste
  succeeds) does not affect the tray stack. The tray is a separate surface
  from the direct-paste pathway.

The stack has no hard depth limit. In practice it holds the last N
transcriptions that entered via shift+release or paste failure. Old entries
age out naturally as new ones push them down, or are explicitly deleted.

### Tray entry ownership

Every tray entry has an owner — either the user or the assistant — and the
overlay renders each entry in the owner's color language. The user's color is
the default dictation overlay style. The assistant's color is the command
overlay style (hue rotation, pulsing). There is no third "shared" or "modified
by both" color.

**Ownership rules:**

- **User-created entries** (dictation via shift+release, paste failure) appear
  in user color immediately and permanently.
- **Assistant-created entries** (placed via tool call) appear in assistant
  color when they arrive.
- **Ownership transfer is automatic and monotonic.** An assistant-created entry
  transitions to user color after one interaction turn — once the user has
  navigated to it, inserted it, or otherwise acknowledged it. Once an entry
  is in user color, it stays in user color. There is no reverse transition.
- **Assistant modifications to existing entries** re-paint the entry in
  assistant color for one turn, then it returns to user color on
  acknowledgment. The visual change is the signal — if the entry looks
  different, the assistant touched it.

**Design principle: failure is loud, stability is quiet.** Anything that has
been with the user in a persistent state is rendered as theirs. The user
should have the sense that their entries do not change silently. New or
modified assistant content pops out visually because it hasn't been
acknowledged yet. The moment it's acknowledged, it becomes the user's — quiet
and stable. This means the only time assistant color appears in the tray is
when something is new or has just changed. The resting state of the tray is
entirely user-colored.

### Spring animation

When the tray is entered from recording (shift held + release spacebar), the
tray overlay appears and, if shift is still held, begins a spring animation:
the overlay drifts downward with an ease-out curve (fast start, decelerating —
like pulling against a spring). Maximum displacement is small (~8–10pt over
400ms). This is purely a visual affordance — it signals "something is about
to happen if you keep holding."

If shift is released, the overlay eases back to resting position (spring
relaxing). The spring is a visual cue, not a commit mechanism — it no longer
triggers a send. The confident send path is Enter, not shift hold-through.

The upward flick animation plays on Enter (send to assistant) — the overlay
flicks up as the text departs for the command overlay. This is the universal
"sent to assistant" visual signature. A shorter, sharper pullback-and-flick
(~100ms) plays on Enter from the tray. The hold-through from recording
(Enter held at release) plays a longer version with the same visual shape.

**Why the flick works (accidental illusion).** The current command pathway
already produces a convincing illusion of continuous upward motion even though
no such animation exists. What actually happens: the preview overlay fades out
very quickly (fast ease-out) while the command overlay pops in just above it
at almost the same instant, with the user's utterance text appearing at the
top of the command overlay. Because the transitions are faster than the eye
can track, the brain perceives a single continuous motion — the preview text
flicking upward and fusing into the command overlay. The spring pullback
turns this optical coincidence into a deliberate gestalt: the downward drift
creates visible tension, the upward flick releases it, and the two
independent transitions (tray fade-out, command overlay fade-in) land in the
same perceptual moment as the flick's apex. The animation that the user
"sees" never actually exists as a single coordinated motion — it is two
unrelated transitions that rhyme.

### Migration from current command pathway

| Flow | Before tray | After tray |
|---|---|---|
| Fast command | shift+release → assistant | enter+release → assistant |
| Deliberate command | *(same as fast)* | shift+release → tray → Enter → assistant |
| Cancel recording | *(no clean path)* | shift+release → tray → navigate up past top → dismissed |
| Preview before paste | *(no path)* | shift+release → tray → spacebar tap → paste |
| Recall history | shift+short-hold → command overlay | shift+short-hold → tray with last entry |
| Clipboard management | *(no path)* | tray stack + navigation + delete |

The fast command path moves from shift to enter. The accidental-send problem
(shift held by mistake) goes away entirely — shift now means "review," never
"send."

For assistant recall, Enter also wins if both modifiers are involved during a
short empty hold. That lets a slightly sloppy Shift+Enter overlap still
toggle the assistant overlay instead of falling back to tray recall.

### Relationship to recovery mode

Recovery mode becomes tray entered automatically when paste verification
fails. The overlay, gestures, and stack lifecycle are identical. The only
difference is the entry trigger: manual (shift+release) vs automatic (OCR
failure).

### Text editing in tray (future)

A separate hotkey (TBD — acceptable per the design principle, since text
editing inherently requires a keyboard) could switch the overlay to editable
mode where the keyboard types into the tray text. In editable mode, spacebar
types spaces rather than triggering insert, and shift/enter revert to their
standard typing roles. The exit gesture from editable mode is TBD.

## Recording cap

On machines with < 16GB RAM running local inference: 20-second maximum
recording duration. The last 3 seconds show a linear countdown glow. At 20s,
recording force-stops. No cap in sidecar mode.

## Visual feedback summary

| State | Overlay | Glow | Menu icon |
|---|---|---|---|
| Idle | hidden | off | unfilled mic |
| Recording | live preview (typewriter) | amplitude-reactive border | filled mic |
| Transcribing | preview holds | fading | filled mic |
| Tray | tray overlay (entry in owner color — user or assistant) | off | unfilled mic |
| Command streaming | command overlay (slow full-spectrum hue rotation, pulsing) | off | filled mic |

## Key source files

- `spoke/input_tap.py` — `SpacebarHoldDetector`, CGEventTap, state machine
- `spoke/__main__.py` — hold callbacks, pathway routing, tray state
- `spoke/overlay.py` — text overlay, tray UI, animations
- `spoke/command_overlay.py` — command response overlay
