# Operator Shell Roadmap

Spoke began as global hold-to-dictate for macOS. This document describes the
architectural direction that grows it into a speech-native local operator shell —
a system where local AI stops being a destination you visit and becomes a
faculty you invoke.

## The Architecture

Six layers, each building on the last. Spoke already implements Layer 0. The
remaining layers are ordered by dependency and value: each one is useful on its
own, and each one makes the next one possible.

### Layer 0 — Ingress (exists)

Spacebar-gated voice capture with local MLX Whisper inference. Hold the
spacebar, speak, release, and the transcription lands at the cursor. Live
preview overlay during recording. Screen-edge glow driven by microphone
amplitude. This layer solved vocal ingress: making speech feel native rather
than bolted on.

The key property that carries forward: the body decides *after* speaking what
kind of act it was. The recording and transcription flow is identical regardless
of destination. The fork happens at release, not before speech.

### Layer 1 — Telos Routing

A modifier at release determines the destination class of the utterance:

- **Normal release** — text insertion (existing path). The transcription is
  pasted at the cursor via the current `inject.py` pipeline.
- **Shift-release** — command utterance (new path). The transcription is sent
  to a local model for interpretation and action.

Same intake, same bodily habit, same screen waking into listening mode. The
user does not decide before speaking whether they are "using the AI." They
decide after, at the release boundary, what class of act it was. This removes a
cognitive fork at the exact point where cognition should remain continuous.

**Scope:**

1. Shift-release detection in `input_tap.py`. On spacebar release, check
   whether shift is held. If yes, route the transcribed text to the command
   pathway instead of text injection.
2. Command dispatch module (`command.py`). Takes the transcribed command,
   sends it to an OMLX server endpoint (`/v1/chat/completions`), streams the
   response. Configuration via `SPOKE_COMMAND_URL` and `SPOKE_COMMAND_MODEL`.
3. Output overlay. A second overlay surface, visually kin to the input overlay
   — same ethereal transparency, same floating treatment — but differentiated
   by color and rhythm. The input overlay breathes with amplitude (non-
   repetitive, tuned to the voice). The output overlay pulses with a simple
   ease-in/ease-out rhythm (mechanical but gentle, distinct from the organic
   input). Response text streams in token by token with the pulse. Streaming
   from day one — a dead pause followed by a wall of text would kill the
   feeling of liveness that makes the whole thing work.

**Inference target:** OMLX server running a 35B-A3B MoE tool-caller. OMLX
already supports MCP tool configs, persistent KV caches, and streaming. Spoke
talks to it over the same OpenAI-compatible HTTP shape it already uses for
remote transcription.

### Layer 2 — First Tool: Semantic Scratch Space

The command pathway's first MCP tool. Not a clipboard manager — a
semantically-addressed scratch space where storage and retrieval are mediated
by natural language.

**How it works:**

- The user says "grab that error message" or "save that URL." The model
  resolves "that" against screen context (initially: whatever was just spoken
  about or recently pasted; later: actual screen perception) and stores the
  content with a semantic key and optional context tag.
- The user says "paste the API key from earlier." The model resolves the
  natural language query against the scratch manifest and retrieves the
  matching entry.
- Delivery uses the existing `inject.py` paste pipeline — content goes to
  whatever text field is currently focused. The model resolves *what*, the
  cursor resolves *where*.

**Manifest:** A lightweight structured document (JSON or markdown) maintained
by the model at runtime. Each entry carries: content (or truncation for large
blobs), semantic key, optional context for why it was saved, and timestamp.
Bounded at ~50 entries by default, prunable by the model when it gets stale.
This is a tiny inward-facing Epistaxis — same pattern, narrower scope.

**MCP tool schema:** `scratch_store`, `scratch_retrieve`, `scratch_list`,
`scratch_remove`. This is the proving ground for the entire tool-calling
pipeline. Building this means building the first complete loop: voice command
-> model -> tool call -> world change.

### Layer 3 — Perception Router

Not "a VLM looking at screenshots." An adaptive stack that makes the screen
semantically legible to the planner without paying full-frame vision cost on
every query.

**Priority order:**

1. **Free structural hints.** Accessibility API, window bounds, app identity,
   focused element, URL bar contents. These are free, fast, and often
   sufficient.
2. **Cheap scout pass.** A small VLM (0.8B–2B class) on a downsampled crop
   when structural hints are insufficient. Determines where the relevant
   content lives, or answers simple queries outright.
3. **Sharper pass.** A larger VLM or more targeted crop when the scout is
   ambiguous. Still local, still fast enough for interactive use.

The output of this layer is always a **textual scene description**, not raw
pixels forwarded to the planner. The heavy model never sees images. It sees
the world the perception router already made small.

This is the layer where the tab experiments showed the key discovery: screen
state is usually legible enough through cheaper channels that full-frame VLM
inference is rarely needed. The problem is engineering, not research.

### Layer 4 — Action Surface + Verification Loop

The planner (Layer 3's downstream consumer — probably the same 35B-A3B
tool-caller, just receiving text instead of images) emits structured tool
calls against a fixed action schema: browser operations, OS operations,
selection, window management, menu invocation, tab switching, click, drag,
typed insertion.

**The critical property:** act, then immediately verify. Small recapture after
every action. Did the state change as intended? If yes, continue. If no,
retry or escalate ambiguity. Without this, it's a hallucination cannon. With
it, it's a primitive hand.

Tool calls execute against macOS APIs (Accessibility, AppleScript, CGEvent)
and browser automation where applicable. The action schema is fixed and
validated — the model cannot invent new action types, only compose from the
available set.

### Layer 5 — Output Surface

No god-box. No chat window as destination. Output is summoned and transient:

- **Silent actions.** Tool actions that change the world produce no textual
  output, or a faint provenance ghost — an almost-invisible inscription of
  what happened, appearing and vanishing before it gets vulgar. Enough to
  orient, not enough to narrate.
- **Brief answers.** Floating overlay, semi-transparent, pulsing with
  generation. Visually kin to the input overlay but distinct tissue — same
  organism, different rhythm. Fades after completion or yields to a longer
  strand.
- **Conversational strands.** When the interaction actually wants continuity,
  a persistent surface emerges. Not a chat window — a summoned, contextual
  thread that knows what it's about and can be reopened later.

The visual language: effervescent, ghostly, present-but-not-heavy. The
machine thinking visibly for a moment, not a panel opening. Chat is not a
destination. Chat is an occasion.

### Layer 6 — Continuity Substrate

Inward-facing Epistaxis. Lighter than the full cross-project version, but
governed by the same principles:

- **Strands** carry scoped state: app, window, repo, recent action lineage,
  current user intention layer, maybe active visual referent.
- **Lifecycle states:** active, soak (provisionally resolved, still watched),
  blocked, idle.
- **Routing:** new utterances are matched to the right strand or fork a new
  one. The system resolves "what thread is this?" before bothering the human.
- **Intent layering:** not just "what are we doing" but what class of doing
  is active.
- **Incoherence semantics:** when the machine's interpretation of the screen,
  the continuity state, and the new utterance disagree materially, that
  surfaces rather than getting smeared over.
- **Bounded revision loops:** the system does not become a self-licking cult
  of local seriousness.
- **Provenance for meta-rules:** if the interaction law revises itself, the
  revision carries a record of what contradiction forced the change.

This is the layer that turns a collection of features into a coherent
organism. It is also the layer most informed by Epistaxis — the law is already
written and battle-tested across hundreds of sessions, it just needs to be
transposed into a runtime state substrate.

## Implementation Order

1. **Layer 1: shift-release routing + command dispatch + output overlay.**
   This is the cut that makes Spoke feel like a different product. Voice
   commands go to a local 35B-A3B via OMLX, responses stream back into a
   pulsing output overlay. Immediate daily-driver value: ask questions, get
   answers, search the web via MCP tools.
2. **Layer 2: semantic scratch space.** First MCP tool. First complete
   voice -> model -> tool call -> world change loop. Forcing function for
   the entire tool-calling pipeline.
3. **Layer 3: perception router.** AX hints first, cheap VLM scout when
   needed, sharper pass when ambiguous. Makes "that" and "this" resolvable
   in voice commands.
4. **Layer 4: action surface.** Fixed tool schema for OS/browser operations.
   Act-then-verify loop. The primitive hand.
5. **Layer 5: output surface refinement.** Provenance ghosts, summoned
   strands, the full visual language. This builds on top of the V1 output
   overlay from Layer 1 — it's iterative refinement, not a rewrite.
6. **Layer 6: continuity substrate.** Runtime Epistaxis. Strand state,
   lifecycle management, intent layering, incoherence surfacing. Last because
   it requires all other layers to exist before it has something to govern.

## Design Principles

- **The body decides after speaking.** The recording flow is always the same.
  The telos fork happens at release, not before speech.
- **The model is the index.** Storage and retrieval are natural language
  operations, not position lookups.
- **Perception is adaptive, not brute-force.** Free structural hints before
  cheap vision before expensive vision. The planner sees text, not pixels.
- **Act then verify.** Every world-changing action is followed by state
  recapture. No trust without verification.
- **Output is summoned, not visited.** No god-box. No permanent chat panel.
  Surfaces appear when the interaction wants them and fade when they don't.
- **Coherence is governed, not hoped for.** Intent layering, incoherence
  surfacing, bounded loops, and provenance are structural properties of the
  system, not aspirations.
- **Local by default.** The acted-on surface is the private desktop. The
  inference runs on local hardware. The state lives in local files. Cloud
  services are optional, not load-bearing.

## Machine Context

Primary development and daily-driver machine: Mac Studio M4 Max, 128GB
unified memory. This machine can keep the 35B-A3B tool-caller, multiple
smaller VLMs (0.8B–4B), and the Whisper transcription models all warm
simultaneously without memory pressure. The 80B-A3B is available as a
swap-in for complex reasoning tasks. OMLX server handles model lifecycle,
persistent KV caches, MCP tool routing, and batch parallel inference.

## Relationship to Existing Work

- **Spoke** (this repo) is the product. All layers live here. The repo does
  not split.
- **Epistaxis** is the governance substrate that taught the concurrency
  semantics, state management, and coherence law that Layer 6 will transpose
  into runtime. It remains the cross-project memory surface.
- **OMLX** is the inference backend. Spoke talks to it over HTTP. Model
  lifecycle, KV caches, and MCP tool routing are OMLX's responsibility.
- **mlx-openai-server / mlx-lm / mlx-quant-toolkit** are upstream
  infrastructure that OMLX builds on. Spoke doesn't interact with them
  directly.
