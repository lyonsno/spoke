# Converge And The Operator-Shell Substrate

This document is the current best-effort architecture map for what `spoke`
already is, what Converge adds to it, and what the operator shell is growing
toward.

It is intentionally not a public README surface. Some of the load-bearing
architecture here depends on private Epistaxis custody, and some adjacent
features are still branch-only or partially staged. The point of this document
is to describe the real system honestly, not to market a cleaner version of it.

## Thesis

`spoke` is no longer just global hold-to-dictate for macOS. It is turning into
a local operator shell with three coupled layers:

1. a voice-native command surface
2. a bounded local tool surface
3. a continuity substrate that starts in the private Epistaxis world and is now
   being ported inward through Converge

The important architectural move is not "voice plus tools." It is that the same
local command model can now:

- answer interactively through Grapheus and OMLX
- call bounded local tools
- write into a lighter-weight substrate of personal attractors and embedding
  caches in the background
- read both private Epistaxis state and its own local Converge state as context
  for future turns

That is the beginning of an operator shell with memory, coordination, and
background cognition, not just a speech UI.

## The Shape Of The System

At a high level, the runtime stack looks like this:

```text
user speech
  -> spacebar-gated capture and transcription
  -> shift-release command routing
  -> spoke CommandClient
  -> Grapheus proxy (:8090)
  -> OMLX OpenAI-compatible endpoint (:8001)
  -> local command model
  -> bounded tool calls and overlays
  -> background Converge carve/embed work
  -> local + private substrate state available to later turns
```

In parallel, a separate narrator path can run through a second Grapheus proxy
and a smaller model, and TTS/STT sidecars remain isolated services rather than
being folded into the core app process.

## What Is Landed Now

### 1. Voice ingress and command routing

The old `spoke` core is still intact: spacebar-gated recording, local
transcription, overlay feedback, and insertion on normal release. What is now
landed on top of that is the command path:

- shift-release routes an utterance into the command stack instead of text
  insertion
- `spoke.command.CommandClient` streams against an OpenAI-compatible endpoint
- the canonical local endpoint is Grapheus, not raw OMLX
- the command stack keeps a bounded persistent ring buffer at
  `~/.config/spoke/history.json`
- command output has its own overlay surface rather than reusing the input UI

This is the first real cut that makes `spoke` feel like an operator shell
instead of just dictation with a side chat.

### 2. Grapheus as the command membrane

The command plane is intentionally mediated:

- `spoke` talks to Grapheus
- Grapheus talks to OMLX
- Grapheus is the durable OpenAI-compatible command surface
- OMLX remains the upstream inference server

This matters because the operator shell wants one stable local API even while
models, providers, and sidecars change underneath it. It also gives the system
an observation point for inference traffic and a place to keep the command path
legible.

The current fleet shape in `services.yaml` is:

- Grapheus for commands on `localhost:8090`
- OMLX upstream on `localhost:8001`
- optional narrator path through a second Grapheus proxy
- `mlx-audio` sidecar for TTS / optional STT
- optional remote Whisper sidecar

### 3. The landed bounded tool surface

On current `origin/main`, the command model gets a bounded 16-tool surface.
Grouped by role, that surface is:

- Perception and expression:
  - `capture_context`
  - `read_aloud`
  - `add_to_tray`
- Local file and repo exploration:
  - `list_directory`
  - `read_file`
  - `write_file`
  - `edit_file`
  - `search_file`
  - `find_file`
- Private substrate bridge:
  - `run_epistaxis_ops`
- External bounded read surface:
  - `query_gmail`
- Background work coordination:
  - `launch_subagent`
  - `list_subagents`
  - `get_subagent_result`
  - `cancel_subagent`
- Context management:
  - `compact_history`

This is already enough to do real operator-shell work: inspect local state,
change files in bounded ways, query private review/state custody, kick off
background search, and compress context instead of just letting the prompt bloat.

### 4. Targeted file editing is now real

`Project Sniper` is the most important recent operator-shell hardening step.
The old file surface had a bad cliff between `read_file` and full overwrite via
`write_file`. That gap is now closed by `edit_file`.

The landed contract is:

- exact `file` / `old_string` / `new_string`
- unique-match requirement
- explicit `not_found`, `not_unique`, and `malformed_request` failure modes
- normalization for line endings, trailing whitespace, final newline, and
  indentation-aware matching
- compactable return values including `applied`, `edited_range`,
  `normalization_applied`, `failure_reason`, and `match_count`

That matters architecturally because the operator shell now has a trustworthy
middle mutation surface. It no longer has to choose only between pure read and
whole-file replacement.

### 5. Subagents are landed in narrow form

`spoke.subagents` is not a general agent harness yet, but it is not imaginary
either. The current form is:

- operator-owned background jobs
- launched from the same command surface
- currently only `kind="search"`
- read-only local file search tools only
- isolated prompt and history-free execution lane
- asynchronous lifecycle with list/get/cancel

This is important because it proves the operator-shell idea is not restricted to
single-threaded conversation. The shell already has a small internal worker
pattern.

### 6. Converge is integrated, not just proposed

The most important thing to understand is that Converge is no longer only a
roadmap idea. The live app imports and initializes `TurnCarver` in
`spoke.__main__`, and it runs after command turns complete.

What the landed Converge layer does today:

- observes completed command turns
- embeds every user utterance
- carves only more substantive turns into personal-attractor operations
- fires this work asynchronously in the background
- relies on OMLX batch parallel scheduling so background carve/embed requests do
  not require a second architecture
- writes personal attractors to `~/.config/spoke/attractors/`
- writes a rolling turn-embedding cache to
  `~/.config/spoke/turn-embeddings.npz`
- writes trace events to `~/.config/spoke/converge-trace.jsonl`

This is already the inward-facing substrate port in embryo. It is smaller than
full Epistaxis, but it is the same pattern:

- durable state outside the immediate thread
- extraction instead of raw transcript hoarding
- explicit attractor objects
- later-turn reuse through a context mechanism

### 7. Compaction has crossed into semantic territory

`compact_history` is not just token shedding. It has three modes:

- `drop_tool_results`
- `summarize`
- `guided`

The important one is `guided`. In guided mode, `spoke.converge` loads an
attractor embedding index, compares the current history slice against the turn
embedding cache, and returns retention flags ranked by cosine similarity.

That means context compaction is already starting to use a semantic substrate,
not just recency and truncation. It is still modest and bounded, but it is a
real substrate-aware memory operation.

### 8. The private Epistaxis bridge is narrow on purpose

`run_epistaxis_ops` is not a shell escape hatch. It is a deliberately narrow
bridge into private Epistaxis custody. It is restricted to:

- reading the repo note
- listing review tickets
- writing review tickets
- appending review pointers
- staging review artifacts
- git status / commit / push on the dedicated Epistaxis worktree branch

That boundedness is load-bearing. The operator shell needs access to private
coordination state, but it cannot turn the whole private repo into a raw mutable
shell surface just because the model wants convenience.

## The Architectural Layering Now

The old six-layer roadmap is still useful, but the live system has started to
collapse parts of it together.

### Layer A: ingress, routing, overlays

This is the user-facing body interface:

- hold to speak
- release to insert or command
- command overlay for streamed responses
- optional narrator layer

This part is mature enough to feel product-like.

### Layer B: command membrane and tool dispatch

This is the local command core:

- CommandClient prompt assembly
- Grapheus mediation
- streamed tool calling
- bounded tool dispatch
- persistent ring buffer

This is the current operational center of the shell.

### Layer C: continuity substrate port

This is where Converge lives:

- personal attractor carving
- turn embeddings
- semantic compaction support
- trace logs for observation and tuning

This is not full runtime Epistaxis yet, but it is already the start of a
continuity substrate inside `spoke`.

### Layer D: private substrate adjacency

This is the relationship to the existing private Epistaxis world:

- the command prompt knows the Epistaxis layout
- the file tools can read Epistaxis directly
- `run_epistaxis_ops` can perform narrow write-side custody tasks
- the external private substrate remains the source of truth for project-level
  coordination, reviews, attractors, topoi, and metadosis

The important distinction is that the command shell is not replacing Epistaxis.
It is learning how to participate in it.

## What Is Partly Staged Or Not Yet Live

This part matters because the repo currently contains several future-adjacent
surfaces in different states.

### Brave search: module present, not live on current main

There is a `spoke.brave_search_operator` module and an attractor for Brave
Search MCP integration. The command prompt also talks as if a `search_web` tool
exists.

But on current `origin/main`, `get_tool_schemas()` does not expose
`search_web`, and the live runtime tool bundle does not include it.

So the honest state is:

- the concept is staged
- the code module exists
- the live tool surface does not expose it yet

### Terminal tool: real branch, not landed

There is also a real bounded terminal tool lane on
`origin/cc/operator-terminal-tool-0421`. That branch adds:

- `run_terminal_command`
- parsed argv instead of raw shell strings
- allow / deny / approval-required policy classification
- bounded command families and cwd constraints

But it is not landed on `main`, and the active review note says the approval
path is still incomplete: approval-gated commands become dead ends because the
conversation loop has no full approval token / retry round-trip yet.

So the terminal tool is not vapor, but it is not part of the current operator
shell trunk either.

### Git tooling: carved as direction, not built

There is already an attractor for bounded read-only git access from the
operator shell. That is a natural next tool surface, but it is still design
direction rather than landed runtime.

## What The System Is Building Toward

The future direction now looks more specific than the original roadmap did.

### 1. More internal background workers

The active Converge lane in Epistaxis has already named the key realization:
OMLX batch-parallel scaling makes local background jobs cheap enough that the
shell does not have to stay single-threaded.

The likely next step is not "many equal agents." It is:

- one user-facing operator
- several bounded internal workers
- all sharing one local inference substrate

Search subagents are the first proof of shape.

### 2. A richer bounded tool surface

The shell is already crossing out the bluntest tool gaps:

- exact text perception and read-aloud
- bounded private state mutation
- Gmail query
- targeted file editing

The obvious next expansions are:

- web search
- read-only git inspection
- terminal execution with approval flow
- more trustworthy file/tool coordination around background tasks

The system is moving toward "bounded operator affordances with clear contracts,"
not toward arbitrary shell freedom.

### 3. A deeper inward Epistaxis

The bigger architectural target is still what the original Converge thread
described:

- hot working set rather than flat chat
- attractor and topos style continuity objects
- graded persistence
- semantic retrieval and compaction
- conflict / incoherence surfacing instead of silent smear
- background substrate maintenance

Today the local inward-facing substrate is still much thinner than full
Epistaxis. It has personal attractors, embeddings, traces, and semantic
compaction support. It does not yet have the full runtime equivalents of topoi,
metadosis, or lifecycle governance.

But the direction is now legible: private Epistaxis remains the full
cross-session coordination substrate, while Converge builds a runtime-local
substrate inside `spoke` that can eventually support the same style of
continuity for voice interaction.

## The Most Important Architectural Difference From A Chat App

The shell is not trying to become a bigger chat window.

The command pathway, tool surfaces, Converge background work, and Epistaxis
adjacency all point in the same direction:

- speech is the ingress
- overlays are the transient expression layer
- the important durable objects are not chats but work signals and continuity
  artifacts
- the model is not just responding; it is operating over local bounded tools
  and a persistent substrate

That is the real convergence here. `spoke` is becoming a local operator shell
whose memory and coordination model are substrate-first rather than
conversation-first.

## Current Status Summary

If you need the short version:

- The voice-native command shell is real.
- Grapheus-mediated local inference is real.
- The bounded local tool surface is real.
- `edit_file` is real and now good enough to count as a serious operator tool.
- Gmail query is real.
- bounded Epistaxis ops are real.
- search subagents are real in a narrow local-search form.
- Converge is real in first form: background carving, embeddings, and guided
  compaction support are integrated into the app.
- Brave search is staged but not yet live on current main.
- the bounded terminal tool is branch-real but not yet landed.
- the full runtime continuity substrate is still ahead of the current code, but
  the port has started.

That is enough to say the operator shell exists now in early form. What remains
is expansion, hardening, and making the substrate layer as coherent in runtime
as it already is in private design custody.
