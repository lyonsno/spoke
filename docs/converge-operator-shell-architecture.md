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

On current trunk, the command model gets a bounded 17-tool surface.
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
- External bounded read surfaces:
  - `search_web`
  - `query_gmail`
- Local bounded terminal surface:
  - `run_terminal_command`
- Background work coordination:
  - `launch_subagent`
  - `list_subagents`
  - `get_subagent_result`
  - `cancel_subagent`
- Context management:
  - `compact_history`

This is already enough to do real operator-shell work: inspect local state,
change files in bounded ways, search the public web, query mail, run bounded
local terminal commands under an approval contract, kick off background search,
and compress context instead of just letting the prompt bloat.

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

- observes completed command turns through a rolling context window (last 4
  turns; user utterances never truncated, assistant turns middle-out truncated
  to preserve intent and conclusion while dropping agent-loop middles)
- carves every 2nd substantive turn, with a staleness override that forces a
  carve when the context buffer has fully rotated past anything the last carve
  saw (no turns pass through unseen)
- embeds every user utterance via OMLX /v1/embeddings
- runs four independent carve passes per event, staggered by 0.5s to avoid
  concurrent prefill contention on Apple Silicon:
  - **attractors** — forces pulling work into existence, with extinguishable
    satisfaction conditions (the extinguishment test: if the thing were true
    right now, would you stop caring about it?)
  - **anamnesis** — factual/relational observations worth remembering
  - **tópoi** — current state of ongoing work, naturally decaying
  - **observed policy** — standing rules and principles with compliance, not
    satisfaction (you can comply or violate but never finish)
- after all four passes produce candidates, a **beast species-classification
  pass** reviews every candidate and decides: pass (correctly routed), kill
  (wrong species / ephemeral command misidentified as signal), or reroute to
  a different surface. The beast fails open — if it can't run, all candidates
  pass through unchanged
- reinforce/expand ops on all surfaces trigger **recompile-on-write** via a
  generic recompile prompt, producing current-state summaries instead of
  appending chronological evidence lines. The file always reads as "what this
  entry says now," not a changelog
- inference parameters tuned for the local model (Qwen3.6-35B-A3B-oQ8):
  temp 0.6, top_p 0.95, top_k 20 — found via gradient probe across 5 prompt
  variants and 3 temperature settings
- all carve events traced with prompt_hash (SHA-256 prefix of the system
  prompt that produced each result) for eval reproducibility
- writes to:
  - `~/.config/spoke/attractors/`
  - `~/.config/spoke/anamnesis/`
  - `~/.config/spoke/topoi/`
  - `~/.config/spoke/policy/`
  - `~/.config/spoke/turn-embeddings.npz`
  - `~/.config/spoke/converge-trace.jsonl`

The key architectural distinctions between the four surfaces:

- **Attractors** are prescriptive — they describe how the world should be.
  Satisfying an attractor extinguishes it; the pressure goes away.
- **Policy** is prescriptive but non-extinguishable — you comply or violate
  but never finish. "Development always happens in worktrees" is policy.
- **Anamnesis** is descriptive — facts, observations, relational knowledge.
  No satisfaction condition. "The server runs on port 8001."
- **Tópoi** are descriptive and temporal — current work state that decays
  naturally. "Working on the context window branch."

The beast species filter exists because these boundaries are subtle and a 35B
local model making the routing decision alone gets it wrong on edge cases,
especially when existing entries act as reinforcement magnets. The beast sees
all four surfaces' candidates together and catches cross-surface duplication
and misrouting that no individual pass can see.

This is the inward-facing substrate port in substantial form. It is still
lighter than full Epistaxis, but it is now multi-surface, species-filtered,
recompile-on-write, and semantically aware:

### 7. Compaction has crossed into semantic territory

`compact_history` is not just token shedding. It now has four modes:

- `drop_tool_results`
- `summarize`
- `guided`
- `reset_to_summary`

The important one is `guided`. In guided mode, `spoke.converge` loads an
attractor embedding index, compares the current history slice against the turn
embedding cache, and returns retention flags ranked by cosine similarity.

That means context compaction is already starting to use a semantic substrate,
not just recency and truncation. It is still modest and bounded, but it is a
real substrate-aware memory operation.

### 8. Epistaxis access is real, but the old helper is demoted

The command shell can read Epistaxis directly through file tools and bounded
terminal reads, and current trunk also carries a runbook-gated terminal path
for Epistaxis git operations. The important design shift is that the assistant
is no longer supposed to rely on `run_epistaxis_ops` as a default first-class
surface.

`run_epistaxis_ops` still exists in the implementation, but it has been demoted
from the default advertised tool surface because the old shape was too easy to
mix incoherently with normal file tools and terminal git. The current system is
biased toward:

- direct file reads and bounded file writes for Epistaxis contents
- bounded terminal git with an explicit runbook gate for merge/push flows
- a future stricter Epistaxis write membrane rather than the old partial helper

That demotion is load-bearing. The operator shell needs access to private
coordination state, but it also needs one coherent mutation story rather than
two overlapping brittle ones.

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

- four-pass carving across attractors, anamnesis, tópoi, and policy
- beast species-classification filter with pass/kill/reroute verdicts
- recompile-on-write for all surfaces (current-state files, not changelogs)
- rolling context window (4 turns) with staleness-aware carving cadence
- staggered prefill launches for Apple Silicon Metal contention
- turn embeddings via OMLX /v1/embeddings
- semantic compaction support (guided mode with attractor-aware retention)
- prompt-versioned trace logs for observation and eval reproducibility
- tuned inference parameters (temp 0.6 / top_p 0.95 / top_k 20)

Planned next surfaces:

- **coherence gardening pass** — offline full-state review with license to
  merge, split, migrate, archive, or rewrite entries across all surfaces.
  Runs manually or nightly, not on the hot path.
- **Paint Dry reader adaptation** — live terminal observability for the carve
  pipeline, consuming the extracted Paint Dry narrator/sink/reader architecture
- **rough** — a catch-all surface for light-mode operation (2 calls instead
  of 5) where the carver routes to attractor or rough, and periodic beast
  reconciliation promotes rough entries to their correct surfaces

This is not full runtime Epistaxis yet, but it is already a multi-surface
continuity substrate with species-filtered routing inside `spoke`.

### Layer D: private substrate adjacency

This is the relationship to the existing private Epistaxis world:

- the command prompt knows the Epistaxis layout
- the file tools can read Epistaxis directly
- bounded terminal git can operate against Epistaxis under a runbook gate
- the legacy helper exists but is not part of the default advertised surface
- the external private substrate remains the source of truth for project-level
  coordination, reviews, attractors, topoi, and metadosis

The important distinction is that the command shell is not replacing Epistaxis.
It is learning how to participate in it.

## What Has Recently Crossed From Staged To Live

Some surfaces that used to be branch-only or merely described are now part of
the live shell and are worth calling out explicitly.

### Bounded terminal execution is landed

`run_terminal_command` is no longer a branch-only idea. Current trunk carries:

- parsed argv instead of raw shell strings
- allow / deny / approval-required policy classification
- bounded command families and cwd constraints
- explicit Enter/Delete approval grammar
- durable pending-approval recovery across recall and restart
- terminal preview truncation and explicit tool-output truncation signaling

That matters because it closes one of the biggest gaps between "chat with some
tools" and "actual operator shell": the model can now perform bounded terminal
work without escaping into arbitrary shell semantics.

### Brave search is landed as a bounded read surface

`search_web` is now part of the live default tool surface. It is a bounded
read-only public web lookup, not a general browser automation path.

## What Is Still Future-Adjacent

This part matters because the repo still contains several next-surface ideas in
design or attractor form rather than as landed runtime.

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

### 3. A real Epistaxis workspace membrane

The current demotion of `run_epistaxis_ops` is not the end state. It is a
holding pattern until the shell can offer an Epistaxis mutation surface that is
actually simpler than mixing file tools, bounded terminal git, and a partial
helper.

The direction we want is:

- explicit Epistaxis workspace handles rather than raw filesystem paths
- no hidden mutable "current Epistaxis workspace" state machine
- one workspace per ops call
- one canonical write membrane for Epistaxis mutation flows

The likely shape is:

- `ensure_epistaxis_workspace(...)` or equivalent
  - resolves or creates a named Epistaxis worktree
  - returns a stable session-local handle plus metadata like path, branch, base,
    dirty/clean state, and detached/not-detached state
- `list_epistaxis_workspaces()`
  - exposes the currently known handles and enough summary information to pick
    the right one without raw path recall
- `run_epistaxis_ops(workspace_handle=..., operations=[...])`
  - executes bounded operations against exactly one explicit workspace handle
  - no cross-worktree mutation plan in one call

The important architectural decision is that this should become the canonical
Epistaxis write path if it exists at all. If the shell is going to pay the
prompt and cognitive cost of an Epistaxis-specific helper, it has to collapse
friction and ambiguity, not add a second half-overlapping mutation surface.

That means the target contract is not "another way to do some git and file
steps." It is:

- prepare or resolve the right Epistaxis workspace
- perform bounded writes inside that one workspace
- stage / commit / push through one coherent membrane

Until that exists, the honest default remains direct file access plus
runbook-gated terminal git, because that is less magical and less confusing
than the older partial helper shape.

### 4. A deeper inward Epistaxis

The bigger architectural target is still what the original Converge thread
described:

- hot working set rather than flat chat
- attractor and topos style continuity objects
- graded persistence
- semantic retrieval and compaction
- conflict / incoherence surfacing instead of silent smear
- background substrate maintenance

The local inward-facing substrate is now multi-surface: personal attractors,
anamnesis, tópoi, observed policy, embeddings, traces, semantic compaction, and
species-filtered routing. It does not yet have the full runtime equivalents of
metadosis or lifecycle governance, but tópoi and policy are now landed surfaces
rather than future-adjacent ideas.

The direction is now legible and partially realized: private Epistaxis remains
the full cross-session coordination substrate, while Converge builds a
runtime-local substrate inside `spoke` that already supports four-surface
continuity for voice interaction, with the coherence gardener and Paint Dry
observability layer as the next expansions.

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
- Epistaxis access is real; the legacy helper still exists, but the shell is
  now biased toward direct files plus runbook-gated terminal git.
- search subagents are real in a narrow local-search form.
- Converge is real in substantial form: four-pass background carving with beast
  species filter, recompile-on-write, staggered prefills, tuned inference
  params, prompt-versioned tracing, and guided/reset compaction support are
  integrated into the app.
- Brave search is live as a bounded read surface.
- the bounded terminal tool is live, including approval and recovery plumbing.
- the full runtime continuity substrate is still ahead of the current code, but
  the port is now multi-surface and species-filtered.

That is enough to say the operator shell exists now in working form. What
remains is the coherence gardener, Paint Dry observability adaptation, eval
harness formalization (Lane G), and making the substrate layer as deep in
runtime as it already is in private design custody.
