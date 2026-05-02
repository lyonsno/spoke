# Keyboard Grammar

Internal reference for Spoke's input gesture design.

## Design principle

Clean means you never need to move your hand, use the mouse, or look at the
keyboard. One hand resting on the desk near spacebar, shift, enter, `]` —
that's the entire physical interface. Spacebar and shift are the primary
surface (adjacent keys, one hand, never leaves typing position). Enter and `]`
are the secondary surface, one key over — reachable without shifting hand
position.

If a gesture requires reaching for escape, tab, or any key outside the
spacebar/shift/enter/`]` cluster, it has left the interaction surface. The
exceptions are route keys (number row, used during recording for destination
selection) and the tray's editable mode (full keyboard for typing).

## Key identity

Each key has a consistent identity across the entire grammar:

- **Spacebar** = go, do the thing, commit. Tap to type, hold to record.
  Release sends output to the cursor (the passive default).
- **Shift** = not the default thing, navigate, review, back out. Enter the
  tray, scrub through history, dismiss, delete. Switch navigation surface
  (tray ↔ agent cards, future).
- **Enter** = send to the active destination. Always. From the tray, during
  recording, from anywhere — Enter means "send this to whoever I'm talking
  to." The active destination is the assistant by default, or whatever sticky
  route is set.
- **`]`** = route to destination. Tap during recording to select where the
  utterance goes. Part of the send chord. The assistant is the default
  destination; other destinations are bound via route keys.

## Core gestures

| Gesture | Result |
|---|---|
| Spacebar tap (< 200ms, no shift) | Normal space character (forwarded to app) |
| Spacebar hold (≥ 200ms), clean release | **Text pathway** — record, transcribe, paste at cursor |
| Spacebar hold, shift held at release | **Enter tray** — record, transcribe, stage for editing |
| Spacebar hold, tap `]` during recording | **Route to assistant** — on release, utterance goes to assistant |
| Spacebar hold, tap route key during recording | **Route to destination** — on release, utterance goes to that route's destination |
| Spacebar hold, tap shift, then release spacebar | **Latched recording** — keep recording hands-free until explicit exit |
| Short spacebar hold (< 800ms), shift at release | **Recall** — enter tray with last tray entry (no recording) |

The hold threshold defaults to 200ms (configurable via `SPOKE_HOLD_MS`, which
sets the threshold in the `SpacebarHoldDetector`).

## Idle controls

These gestures are available when you are not already in the tray and not in
the middle of an active hold:

| Gesture | Result |
|---|---|
| Shift tap while idle | Toggle TTS audibility |
| Double-tap Shift while idle | Toggle Terror Form HUD |
| Space + Delete (while assistant overlay is visible) | **Cancel generation / end assistant turn.** Stops the current streamed response and dismisses or freezes the command overlay. |

## Disposition at release

For the ordinary recording path, utterance disposition is decided by route key
state when capture ends, not when recording starts. You hold spacebar, speak,
optionally tap a route key to select a destination, and release:

- **Clean release, no route key selected** → text insertion (dictation)
- **Shift held at release** → tray (edit before committing)
- **Route key selected (e.g. `]`)** → route to that destination on release

Shift can be pressed at any point during the hold — it is latched via
`kCGEventFlagsChanged`, so pressing shift after you start speaking still
routes to the tray.

Route keys can also be pressed at any point during the hold. Tapping a route
key selects it; tapping the same key again deselects it. Only one route key
is active at a time.

If both shift and a route key are active at release, shift wins — the
utterance enters the tray with the route key's destination pre-selected, so
the send chord (Enter + `]`) will honor the pre-selected route.

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
IDLE ──[spacebar down]──→ WAITING ──[200ms timer]──→ RECORDING ──[spacebar up]──→ IDLE
                              │                           │
                     [spacebar up early]          [300s safety timeout]
                              │                           │
                              ▼                           ▼
                      forward space char            force stop, IDLE
```

- **IDLE**: ready for input.
- **WAITING**: spacebar is down, hold timer running. Key repeats suppressed.
- **RECORDING**: audio capture active, overlays visible.

Latched recording adds one more state on top of the base hold detector:

- **LATCHED**: recording remains active after spacebar is released.
- **Shift tap during RECORDING** enters **LATCHED**.
- **`]`** or route key selection + release exits **LATCHED** to the selected
  destination.
- **Shift + spacebar release** exits **LATCHED** to the tray.

Synthetic space forwarding on early release uses a 100ms auto-clear timeout in
case the forwarded events are lost.

## Text pathway (clean release)

1. Audio capture stops.
2. Final transcription runs on a background thread.
3. Text is injected at the cursor via paste.
4. If no focused text field is detected → enters **tray** automatically.

## Command pathway (`]` route key selected)

1. Audio capture stops.
2. Audio is transcribed to text (the "utterance").
3. Utterance is streamed through the command client (assistant or hot route).
4. Command overlay shows the utterance and the streamed response.

## Latched recording

Latched recording is a mid-capture state transition, not a change in
end-of-recording intent. The utterance's disposition is still chosen by
route key state when capture ends.

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
| Clean release (no shift, no route key) | Stop capture, transcribe, paste at cursor |
| Shift held + spacebar release | Stop capture, transcribe, enter tray |
| Route key selected + release | Stop capture, transcribe, route to destination |

Latched recording exits follow the same disposition rules as ordinary
recording. The latch only changes *when* capture ends, not *where* the
utterance goes.

### Relationship to the tray

Tray entry does not change. From the tray, spacebar hold (≥ 200ms) still
starts a new recording. The transcription appends at the cursor position in
the tray's editable text. If that new recording should continue hands-free,
tap shift during the recording to latch it.

## The tray

The tray is an editable speech-native stacked buffer. It holds recent
transcriptions for editing, insertion, sending to a destination, or later
retrieval. It is the central interaction surface between recording and
committing.

### Why the tray exists

1. **Edit before commit.** See and modify what was transcribed before it goes
   anywhere. Fix whisper errors, restructure, add context.
2. **General input buffer.** Type into it, paste into it, dictate into it. The
   tray is the way to get any text — spoken or typed or pasted — to the
   assistant or any other route destination.
3. **Cancel path.** Shift+release enters the tray instead of pasting. Dismiss
   from the tray to cancel. No audio wasted on a bad paste or accidental send.
4. **History.** Every transcription that enters the tray is stacked. Scrub back
   through recent transcriptions with two keys.
5. **Speech-native clipboard.** The assistant can add arbitrary content to the
   tray via tool calls. Combined with voice commands, this is a full clipboard
   manager operated entirely by speech and a few keys.
6. **Recovery unification.** Paste failure (no focused text field) enters the
   tray automatically rather than being a separate "recovery mode" state.

### Tray entry

| Context | Gesture | Result |
|---|---|---|
| Recording | Shift held + release spacebar | Enter tray with transcription at cursor |
| Recording (short, < 800ms) | Shift held + release spacebar | Recall last tray entry (no recording) |
| Paste failure | *(automatic)* | Enter tray with transcribed text |

### Editable tray

When the tray is up, it is editable by default. The keyboard types into the
tray text — spacebar types spaces, shift and enter have their standard typing
roles, all keys pass through to the tray's text field. The cursor is in the
tray.

This is the critical difference from the old read-only tray: you can fix
transcription errors, type new text, and paste from the system clipboard
(Cmd+V). The tray is no longer just a display case for transcriptions — it is
a workspace.

Recording from within the tray still works: spacebar hold (≥ 200ms) starts a
new recording, and the transcription appends at the cursor position in the
tray when it lands. This requires the 200ms hold threshold to distinguish
"typing a space" from "starting a recording." The threshold is the same as
the global one.

### Tray gestures

| Gesture | Result |
|---|---|
| Spacebar tap | Type a space character in the tray |
| Spacebar hold (≥ 200ms) | Start new recording (transcription appends at cursor in tray) |
| Enter | **Send** — send tray text to active destination (assistant by default, sticky route if set) |
| Shift+Enter | Insert a newline character in the tray text |
| Enter + `]` | **Send chord** — send tray text to a specific route destination |
| Double-tap space, then hold | **Insert at cursor** — insert tray text at cursor in frontmost app and dismiss |
| Shift + release spacebar | Navigate up (more recent item; dismiss at top) |
| Spacebar + tap shift | Navigate down (older item) |
| Shift held + double-tap spacebar | Delete current tray entry |

**Enter = send.** Bare Enter from the tray always sends to the active
destination. The active destination is the assistant by default, or whatever
sticky route is set via the sticky toggle. This matches universal compose
surface convention — Enter sends, Shift+Enter inserts a newline.

**Enter + `]`** is the explicit send chord for targeting a specific route
destination. It overrides the default active destination. `]` specifies
*where*, Enter specifies *go*.

**Double-tap space, then hold** is the insert-at-cursor gesture. Tap space
twice (each tap types a space character), then on the third press, hold.
The hold enters a visual wind-up state. On release, the tray entry is
pasted at the cursor in the frontmost app and the tray is dismissed. The
double-tap requirement makes this virtually impossible to trigger
accidentally during normal typing.

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

- **New recording from tray** (spacebar hold ≥ 200ms) appends the
  transcription at the cursor position in the current tray entry.
- **Send** (Enter, or Enter + route key) consumes the current entry — it is
  removed from the stack after delivery. Bare Enter sends to the active
  destination; Enter + route key overrides.
- **Insert at cursor** (double-tap space, then hold) pastes the current entry
  at the cursor in the frontmost app and dismisses the tray. The double-tap
  requirement prevents accidental triggers during typing.
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

- **User-created entries** (dictation via shift+release, paste failure, typed)
  appear in user color immediately and permanently.
- **Assistant-created entries** (placed via tool call) appear in assistant
  color when they arrive.
- **Ownership transfer is automatic and monotonic.** An assistant-created entry
  transitions to user color after one interaction turn — once the user has
  navigated to it, edited it, or otherwise acknowledged it. Once an entry
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
200ms). This is purely a visual affordance — it signals "something is about
to happen if you keep holding."

If shift is released, the overlay eases back to resting position (spring
relaxing). The spring is a visual cue, not a commit mechanism.

The upward flick animation plays on send (Enter + `]`) — the overlay flicks
up as the text departs for the command overlay. This is the universal "sent"
visual signature.

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

### Relationship to recovery mode

Recovery mode becomes tray entered automatically when paste verification
fails. The overlay, gestures, and stack lifecycle are identical. The only
difference is the entry trigger: manual (shift+release) vs automatic (OCR
failure).

## Route keys

Route keys replace timing-based destination selection with explicit,
visible, toggleable selection made during recording. You choose where
your utterance goes while you're still speaking, with visual confirmation,
and you can change your mind before release.

### Physical surface

Two surfaces:

**`]` — the assistant route key.** Sits under the right hand's middle/ring
finger at rest, one key over from enter. The most reachable non-spacebar key
after enter and delete. This is the primary and most common route key. It
routes to the assistant by default, or to whatever destination is bound to
it.

**The number row** from the right side of the keyboard, reachable without
moving the right hand:

```
6  7  8  9  0  -  =
```

Seven keys. Each can be bound to a different destination: operator shell
modes, Epistaxis interaction, read-aloud, web research, or user-defined
shortcuts. Modifiers (shift, etc.) could extend the surface later.

### Visual presentation

While recording (spacebar held or latched), a row of small ghost indicators
appears above the user preview overlay — one per route key, all lined up.
They render very faint by default: present but quiet, like watermarks. Each
ghost shows its key label and possibly a tiny icon or abbreviation for what
it routes to.

The assistant route key `]` renders as a faint vertical pill to the right of
the user preview overlay, separate from the number row ghosts. It is the most
common non-default destination and deserves a distinct visual presence.

When you tap a route key, its ghost sharpens — becomes noticeably more legible,
stands out from the others. You are now routed to that key's destination.
Tap the same key again and it goes faint — you're back to the default route
(cursor insertion / text pathway).

Only one route key is active at a time. Tapping a different key deactivates
the previous one and activates the new one.

The ghosts are not buttons. They are not clickable. They exist only as a
visual readout of current routing state. The input surface is the physical
keyboard, not the overlay.

### Lifecycle

Route keys are only active during recording — while spacebar is held or
while in latched recording. On release, the utterance goes to whatever
destination is currently selected (default text pathway if nothing is
selected). After release, the route key state resets — unless sticky routing
is active.

### Route flavors

Not all route keys behave the same way after the utterance is delivered.
Three flavors:

1. **Persistent.** Activates a mode that stays on after release. The mode
   persists across subsequent recordings until explicitly deactivated (tap
   the route key again during a future recording, or the mode times out, or
   an exit command is spoken). Example: entering Epistaxis interaction mode,
   read-aloud mode, web research mode.

2. **Contingent.** Activates silently for the current utterance and continues
   if the context suggests continuation, but backs off if the model senses
   you're not interested in continuing. Ghost continuation — the mode is warm
   but will cool on its own. Example: a follow-up question that might or
   might not become a multi-turn thread.

3. **One-shot.** Routes the current utterance once, then deactivates. No
   persistent mode, no continuation. Example: a single command dispatch, a
   quick lookup.

The flavor is a property of the route key's binding, not of the gesture.
The user always does the same thing (tap to select, release to send). The
route key's configuration determines what happens after delivery.

### Mappability

Route key bindings are configurable. The initial set will be opinionated
(specific modes and destinations wired to specific keys), but the mapping
should be a data structure, not hardcoded routing logic. Future operator
shell modes, Epistaxis verbs, or user-defined shortcuts can be bound to
route keys without changing the grammar machinery.

### The send chord: Enter + `]`

**Enter + `]`** sends text to the hot route destination. This chord works
everywhere text can be sent:

- From the tray: sends the current tray entry to the route destination.
- The `]` key specifies *where*. Enter specifies *go*. Together they are
  unambiguous and cannot fire by accident.

If a different route key is active (sticky routed to an Epistaxis mode, for
example), Enter + that route key sends to that destination instead. The
pattern is always **Enter + route key = send to that route**.

The flick animation plays on send — the tray text flicks upward toward the
command overlay.

### Sticky routing toggle: Space + Enter + `]`

Normally, route key selection resets after each recording. The sticky toggle
locks the current routing so it persists across recordings — every subsequent
utterance goes to the selected destination on release without re-tapping the
route key each time.

The chord is **Space + Enter + `]`** — all under the right hand at rest.
Press it once to lock; press it again to unlock and return to per-recording
selection.

Sticky routing captures the keyboard. While locked, typing goes to the hot
route — keystrokes are intercepted by Spoke and delivered to whatever
destination is currently locked, not to the frontmost app. This is the
full keyboard capture mode: if you're sticky-routed to the assistant,
typing goes to the assistant. If you're sticky-routed to an Epistaxis
mode, typing goes to the Epistaxis interaction surface.

This means sticky routing is a deliberate mode shift. You are choosing
to give Spoke the keyboard. The toggle chord (Space + Enter + `]` again)
gives it back. The visual distinction between sticky and non-sticky must
be strong enough that you always know which state you're in.

Sticky routing works with any route key, not just `]`. Lock the assistant
route with Space + Enter + `]`. Lock a number-row mode route with
Space + Enter + that number key. The pattern is always
**Space + Enter + route key = lock that route**.

Visual feedback: when sticky routing is active, the selected route key's
ghost (or the assistant pill) renders with a persistent glow or underline
rather than the transient sharpening that fades after release. The visual
distinction between "selected for this recording" and "locked across
recordings" should be immediately legible.

Sticky routing has no special send semantics. Sending from the tray is
always Enter + `]` (or Enter + the relevant route key), whether sticky
routing is active or not. Sticky routing only affects where dictations go
on release — it does not change the tray's send chord.

### Relationship to the tray

Route keys and the tray are complementary, not competing. Route keys select
a destination before release. The tray is a post-release editing surface.
A natural flow: tap `]` to select the assistant, but then shift at release
to enter the tray anyway — edit the text, then send from the tray with
Enter + `]`. The route key pre-selects the destination; the tray provides the
editing and review step.

### Relationship to operator modes

Route keys can activate operator modes, but they are not the only way to
enter a mode. A voice command ("enter Epistaxis mode") can also activate a
mode. Route keys are a physical shortcut into mode activation — faster and
more reliable than voice for known, frequent destinations.

Once a persistent mode is active, subsequent recordings may route through
that mode's logic regardless of route key state. The mode is the authority;
the route key was the entry gesture.

## Double-tap-then-hold gesture family

Double-tap-then-hold is a universal gesture modifier: tap a key twice, then
hold it on the third press. The taps commit the key's normal action; the
hold that follows triggers the deeper action. The double-tap requirement
makes accidental activation nearly impossible during normal use.

This is the grammar's bridge gesture — it connects surfaces that are
otherwise closed. The tray owns the keyboard, so there's no spare key for
"insert at cursor in the app behind the tray." Double-tap-then-hold
provides the bridge without stealing any key from its normal role.

| Key | Double-tap-then-hold action | Status |
|---|---|---|
| Spacebar | Insert tray entry at cursor in frontmost app | Wired |
| Enter | Send to assistant regardless of context | Future |
| Shift | Switch navigation surface (tray ↔ agent cards) | Future |

The detection window is ~300ms between taps. The hold must follow the
second tap within the same window. Each key tracks its own tap history
independently — tapping space then enter does not open a window on enter.

### Surface switching (future)

Double-tap-then-hold Shift switches what the space/shift rocking gesture
navigates. By default, rocking cycles through tray history. After a
surface switch, the same rocking gesture cycles through agent cards instead
— selecting which agent's full output is shown in the assistant overlay
and which agents are displayed as compact bearing cards.

This is the operator's primary instrument for managing a fleet of concurrent
agent sessions through the keyboard grammar. The agent card system
(SpinalTap thread cards) provides the data; the surface switch provides
the control surface. Double-tap-then-hold Shift again flips back to tray
navigation.

## Recording cap

On machines with < 16GB RAM running local inference: 20-second maximum
recording duration. The last 3 seconds show a linear countdown glow. At 20s,
recording force-stops. No cap in sidecar mode.

## Visual feedback summary

| State | Overlay | Glow | Menu icon |
|---|---|---|---|
| Idle | hidden | off | unfilled mic |
| Recording | live preview (typewriter) + route key ghosts | amplitude-reactive border | filled mic |
| Transcribing | preview holds | fading | filled mic |
| Tray | editable tray overlay (entry in owner color) | off | unfilled mic |
| Command streaming | command overlay (slow full-spectrum hue rotation, pulsing) | off | filled mic |
| Sticky routing | persistent route key glow + keyboard captured | off | filled mic (or distinct icon TBD) |

## Key source files

- `spoke/input_tap.py` — `SpacebarHoldDetector`, CGEventTap, state machine
- `spoke/__main__.py` — hold callbacks, pathway routing, tray state
- `spoke/overlay.py` — text overlay, tray UI, animations
- `spoke/command_overlay.py` — command response overlay
