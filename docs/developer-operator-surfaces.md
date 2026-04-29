# Developer And Operator Surfaces

This document holds real `spoke` capabilities that do not belong on the
public README but still need a durable canonical home.

## Bounded Post-Transcription Repair Pass

`spoke` keeps a bounded post-transcription repair pass for recurring
project-specific vocabulary observed in real logs.

This is a developer-facing correction surface, not a public product promise.
The implementation currently lives in [`spoke/dedup.py`](../spoke/dedup.py),
and README omission is intentional unless the repair pass becomes a visible
user-facing control or configuration surface.

## Modal Agent Shell Sessions

`spoke` carries subscription-auth local coding-agent transport, but those
backends are not generic tools for the default assistant to call. The
operator-facing contract is **Agent Shell**: a modal route destination where
input goes to the selected agent backend session, while explicit Spoke-owned
mode-control input remains under the operator shell.

The menubar exposes an `Agent Shell` provider selector (`Off`, `Codex`, `Claude
Code`). This is intentionally separate from `Assistant Backend`: the local
assistant remains the fuzzy-intent resolver and router, while local coding
agents are modal worker shells selected by route/mode state.

Epistaxis-shaped utterances are not intercepted by Agent Shell today. The
previous keyword-based interception was removed because it could not distinguish
between "run this Epistaxis command" and "talk to the current agent about
Epistaxis", and there is no dedicated Epistaxis command executor seated behind
that route yet.

The lower-level provider contract currently wires `codex` through the local
Codex CLI JSON event stream and requires `codex login` to report ChatGPT
subscription auth. Billing-backed credentials are stripped from the child
environment and are not a fallback path. `claude-code` is a reserved backend id
for the Claude Code CLI transport, but it is disabled in the menu until that
CLI contract is measured.

Provider sessions are asynchronous, keep Spoke-owned ids distinct from
provider session/thread ids, carry the requested working directory, preserve
structured backend events for future command/tool-loop rendering, and surface
backend-unavailable failures as operator-visible state rather than as raw
terminal-command failures.

Backend event presentation is a shared policy layer, not Codex-specific UI
glue. `spoke.agent_backend_presenter` maps provider events into compact actions
such as response deltas, tool start/end, status lines, narrator summaries, and
errors. Raw command/tool output remains hidden by default; operator surfaces
should show liveness and status first, with scrollback or expansion built on
the stored structured events later.
