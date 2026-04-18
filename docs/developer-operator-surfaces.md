# Developer And Operator Surfaces

This document is the canonical home for `spoke` capabilities that are real,
maintained, and intentionally omitted from the public README because they serve
developer or operator audiences rather than ordinary product onboarding.

The routing contract for these decisions lives in
`docs/documentation_surfaces.toml`.

## Bounded Post-Transcription Repair Pass

`spoke` keeps a bounded post-transcription repair pass for recurring
project-specific vocabulary observed in real logs.

This pass lives in [`spoke/dedup.py`](../spoke/dedup.py) and is applied by the
transcription clients after transcription text comes back but before the text is
committed downstream. It exists to repair known ontology vocabulary misses
without turning the public README into a chronicle of transcription internals.

This belongs on a developer-facing surface unless the repair path grows visible
user controls, configuration, or user-facing behavior that would make the
public README an appropriate contract.

## Smoke-Surface Runtime Affordances

On local smoke surfaces, the menubar also exposes launch-target switching,
source/branch visibility, and the status HUD (`Terror Form`) so you can confirm
which runtime surface is actually live.

These affordances are for operator legibility on machines that can launch
multiple nearby surfaces in quick succession. The public README is not their
home because they describe runtime verification and local surface identity, not
core product onboarding. The deeper bring-up and verification flow still lives
in [`docs/local-smoke-runbook.md`](./local-smoke-runbook.md).
