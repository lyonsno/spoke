# Screen Context V1

Narrow design note for the first `spoke` screen-context affordance.

This is intentionally small. The goal is not a full perception stack. The goal
is to give the assistant one fast, local way to ground "that on screen" well
enough to support `read_aloud` without forcing text through files or asking the
assistant to regenerate text that already exists locally.

## Goal

Add a first assistant-callable screen-context affordance that:

1. captures a downsampled local view quickly enough for conversational use;
2. augments the image with OCR and lightweight app/window metadata;
3. stores the result as a local `SceneCapture` artifact with stable refs; and
4. lets a follow-on `read_aloud(source_ref)` resolve exact local text from that
   artifact and hand it to Voxtral.

The key property is indirection:

- the assistant should prefer a local ref to visible text;
- the app should resolve the ref to exact text locally;
- TTS should speak the resolved text, not a regenerated paraphrase.

## Why This First

The current output/TTS work already proves the assistant can speak local text.
The missing affordance is referential grounding for visible things that are not
already sitting in a file, the clipboard, or command history.

Examples:

- "read that tab title"
- "read the dialog on screen"
- "read the text in that pane"
- "read what this says"

Those are not file problems. They are screen-reference problems.

## V1 Scope

V1 is deliberately narrow:

- one capture tool: `capture_context`
- one output tool: `read_aloud`
- one primary capture target: the frontmost app's active window
- one fallback capture target: the main screen
- one default image strategy: downsample before model use
- one reference format: stable local `source_ref` handles backed by a
  `SceneCapture`

V1 does not attempt:

- always-on screen watching
- action execution
- rich scene graphs
- persistent strand routing
- cursor grounding
- region selection UI
- multi-step visual planning
- full desktop fidelity

## Design Principle

The planner should see structured scene state, not just pixels.

The image is still useful, especially for layout, icons, and cases where OCR is
imperfect, but the preferred path is:

1. app captures local scene
2. app extracts OCR + AX hints + window metadata
3. assistant resolves a referent against that scene
4. app resolves the chosen ref to exact text
5. Voxtral speaks the text

This keeps the expensive visual pass small and makes `read_aloud` exact.

## Capture Strategy

### Default target

Capture the frontmost app's active window first.

Why:

- much smaller token budget than full desktop
- better grounding for "this tab", "that pane", "the dialog"
- lower chance of irrelevant chrome from other apps
- faster enough to feel conversational

If active-window capture is unavailable or fails, fall back to the main screen.

### Default resolution

Downsample before the image reaches the model.

Initial target:

- capture at native resolution locally
- produce a model-facing raster at roughly `0.5x` linear scale by default
- equivalent to roughly `25%` of the original pixel count

This is the default because it preserves global layout while cutting prefill
cost dramatically. It should be configurable later, but V1 should hard-code a
single sane default so smoke and latency are easy to reason about.

If the active window is already small, skip unnecessary downsampling.

### Companion context

Each capture should also bundle lightweight local context:

- frontmost app name
- bundle identifier if available
- active window title if available
- capture scope (`active_window` or `screen`)
- capture size and downsample size
- timestamp

V1 should also include OCR text and OCR blocks.

Accessibility data is useful enough to include in a small form if it is cheap:

- focused element role
- focused element label/value if available
- a shallow list of obvious named UI elements near focus

Do not block V1 on deep AX traversal. Cheap hints only.

## Scene Model

V1 should create a local `SceneCapture` artifact.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class OCRBlock:
    ref: str
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass
class AXHint:
    ref: str
    role: str
    label: str | None = None
    value: str | None = None


@dataclass
class SceneCapture:
    scene_ref: str
    created_at: float
    scope: Literal["active_window", "screen"]
    app_name: str | None
    bundle_id: str | None
    window_title: str | None
    image_path: str
    image_size: tuple[int, int]
    model_image_size: tuple[int, int]
    ocr_text: str
    ocr_blocks: list[OCRBlock]
    ax_hints: list[AXHint]
```

The `scene_ref` is the durable handle the assistant sees. The artifact lives
locally under app support or a short-lived temp cache. The assistant should not
need to know the file path.

## Source Refs

`read_aloud` should consume source refs, not just literal text.

V1 needs only a small set:

```python
@dataclass
class SourceRef:
    kind: Literal[
        "last_response",
        "selection",
        "clipboard",
        "scene_block",
        "ax_hint",
        "literal",
    ]
    value: str
```

Examples:

- `last_response:current`
- `selection:frontmost`
- `clipboard:current`
- `scene_block:scene-abc:block-12`
- `ax_hint:scene-abc:focus`

The important design choice is that `scene_block` and `ax_hint` refs are local
artifact pointers, not copied text blobs.

## Tool Surface

### 1. `capture_context`

Assistant-facing shape:

```json
{
  "scope": "active_window"
}
```

V1 should support:

- `active_window`
- `screen`

Return shape:

```json
{
  "scene_ref": "scene-abc",
  "scope": "active_window",
  "app_name": "Safari",
  "window_title": "Pull request - GitHub",
  "summary": "Active browser window with visible tab strip and main content area.",
  "ocr_blocks": [
    {
      "ref": "scene-abc:block-1",
      "text": "Pull request",
      "bbox": [102, 58, 164, 24]
    }
  ],
  "ax_hints": [
    {
      "ref": "scene-abc:focus",
      "role": "AXWebArea",
      "label": "Pull request conversation"
    }
  ]
}
```

The image itself stays local. If the vision model needs it, the tool execution
layer can pass it internally; the assistant should primarily operate on the
summary and returned refs.

### 2. `read_aloud`

Assistant-facing shape:

```json
{
  "source_ref": "scene_block:scene-abc:block-1",
  "voice": "casual_female"
}
```

Accepted V1 inputs:

- `last_response`
- `selection`
- `clipboard`
- `scene_block`
- `ax_hint`
- `literal`

Resolution happens locally:

- `scene_block` resolves to exact OCR text for that block
- `ax_hint` resolves to label/value text
- `selection` and `clipboard` resolve from the current desktop state
- `last_response` resolves from command history

If a ref resolves to empty text, fail clearly instead of silently speaking
nothing.

## Minimal Call Flow

### Case 1: read visible UI text

User:

`read that tab title`

Flow:

1. assistant calls `capture_context(scope="active_window")`
2. assistant resolves the likely tab/title block from returned refs
3. assistant calls `read_aloud(source_ref=...)`
4. app resolves exact text locally
5. Voxtral speaks it

### Case 2: read currently selected text

User:

`read this`

Flow:

1. assistant prefers `read_aloud(source_ref="selection:frontmost")`
2. no capture required unless selection resolution fails

### Case 3: ambiguous visible referent

User:

`read what this says`

Flow:

1. assistant calls `capture_context`
2. if returned refs are insufficient, the execution layer may attach the local
   downsampled image to the model-side resolution step
3. assistant chooses a ref
4. assistant calls `read_aloud`

## V1 Resolution Policy

Prefer the cheapest grounded source first:

1. explicit local text source already available
2. OCR/AX-backed scene refs from `capture_context`
3. image-aided disambiguation against the same scene
4. literal text only when the content exists nowhere else locally

The assistant should not restate visible text just to make TTS work.

## Latency Budget

V1 should optimize for "fast enough to use in live conversation," not perfect
scene understanding.

Target shape:

- local capture: cheap
- OCR: cheap enough to run every time for V1
- AX hints: cheap, shallow only
- vision grounding: use the downsampled image, not full retina

That means:

- capture once per request
- store one local artifact
- avoid full-resolution model inspection by default
- avoid repeated screen reads during a single `read_aloud` request

## Implementation Plan

### Step 1

Create the local capture artifact and a `capture_context(scope)` entry point.

Implementation notes:

- use existing macOS capture primitives already present in `spoke`
- add OCR extraction next to capture, not as a separate user-facing tool
- write artifact files under a local cache directory
- return refs plus small structured metadata

### Step 2

Add `read_aloud(source_ref)` with local ref resolution.

Implementation notes:

- reuse the existing Voxtral TTS client surface
- add a resolver module that maps refs to exact text
- keep truncation/chunking rules minimal for V1

### Step 3

Teach the assistant prompt/tool policy to prefer refs over regenerated text.

Implementation notes:

- if the user refers to visible UI, call `capture_context`
- if the user refers to selected text or the last response, skip capture
- do not copy large visible text into tool arguments when a ref exists

## Out of Scope for V1

These are good follow-ons, but they should not block the first cut:

- region crop selection
- cursor-relative grounding
- richer AX tree export
- action execution from scene refs
- recapture-after-action verification loops
- persistent scene history
- multi-screen routing
- automatic background capture
- generalized screenshot tool for everything

## Immediate Next Cut After V1

Once `capture_context -> read_aloud(source_ref)` works reliably, broaden
screen context rather than broadening TTS.

The next likely expansions are:

1. better referent resolution inside a captured scene
2. region or cursor-local capture
3. richer app/window metadata
4. action-side reuse of the same `SceneCapture` model

That keeps the first cut small while still pushing directly toward the bigger
operator-shell direction.
