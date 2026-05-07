# Optical Field Consumer Contract

This document defines the public House-facing consumer contract in
`spoke.optical_field`. It is intentionally a request/adapter surface, not a
shader or compositor implementation guide.

House owns optical execution: lifecycle clocks, transition phase, coalescing,
interruption, SDF/fill/ridge/material composition, and private geometry queues.
Consumers own semantic intent: stable identity, lifecycle intent, desired
geometry, content frame, coordinate custody metadata, presentation role, layout
recipe, motion intent, continuity, finite signals, freshness, provisional/final
state, and confidence.

## Request Fields

`OpticalFieldRequest` is the production request object for UI consumers.

- `caller_id`: stable caller identity for the optical presence.
- `continuity_key`: stable identity across materialize/rest/retarget/dismiss
  when it differs from `caller_id`; defaults to `caller_id`.
- `bounds`: optical envelope requested by the consumer.
- `content_frame`: legible/rendered content area when it differs from the
  optical envelope; defaults to `bounds`.
- `coordinate_space`: explicit custody marker for geometry units. The contract
  records the value; coordinate normalization remains a helper/consumer surface.
- `display_epoch`, `source_epoch`, `freshness_epoch`: freshness metadata for
  display/capture/request validity.
- `role`: semantic surface role such as `assistant_shell`, `preview`, or
  `agent_card`.
- `state`: lifecycle intent. Current legal values are `hidden`,
  `materialize`, `rest`, `resize`, `recenter`, `retarget`, and `dismiss`.
- `presentation_layer`: presentation ordering family as data.
- `layout_recipe`: finite layout recipe such as `direct_positioned`,
  `semantic-placement-candidate`, `rail`, `deck`, or `selected-handoff`.
- `motion`: `OpticalFieldMotionIntent`; consumers request strategy and urgency,
  while House chooses execution curves and interruption behavior.
- `continuity`: continuity mode such as `preserve_identity`, `handoff`,
  `new_presence`, or `replace`.
- `signals`: finite `OpticalFieldSignal` values routed into House-owned recipes
  and profiles.
- `provisional`: `true` for candidate/working desired state; `false` for final
  desired state.
- `confidence`: optional diagnostic confidence from a semantic or adapter
  source, constrained to `0.0..1.0`.
- `profile`, `disturbances`, `visible`, and `z_index`: existing adapter data for
  profile selection, composable gestures, visibility, and ordering.

## Motion And Signals

`OpticalFieldMotionIntent` is data only. A semantic positioning consumer may
request `strategy="auto"` or a named strategy such as `squirt` or
`dematerialize_rematerialize`, but House owns the overlap metric, thresholds,
coalescing, and visual phase.

`OpticalFieldSignal` carries finite semantic/material inputs such as
`audio_rms`, `background_luminance`, `pending_resolution`, or
`attention_pulse`. Signals may carry their own freshness epoch and bounded
parameters. They are not animation frame controls.

## Progress And Phase Boundary

Production consumers must not author `progress`, `phase`,
`transition.phase`, or equivalent frame-step animation custody. The public
request schema rejects those names for finite signals and does not expose them
as request or disturbance fields.

Lawful phase-like surfaces are separate from this production request contract:

- House-owned internal transition metadata.
- explicit diagnostic/tuner scrubbers.
- Optical Witness manifests and frame-strip metadata.

Those surfaces are debug or implementation evidence. They are not required for
normal consumers to make the primitive behave.

## Migration Notes

Assistant shell consumers should send stable `caller_id`/`continuity_key`,
`role="assistant_shell"`, lifecycle `state`, explicit bounds/content frame,
freshness epochs when semantic positioning or capture is involved,
`presentation_layer`, `layout_recipe`, `motion`, and finite material signals.
They should not rebuild CPU fill state or pass shader phase through request
metadata.

User-preview consumers should send `role="preview"`, the preview optical bounds
and content frame, a fast preview profile, lifecycle state, RMS/audio signals,
and timing/profile intent as data. Preview materialize/dismiss must not depend
on public progress fields or private bottom-left hidden-origin animation state.

Agent-card and HUD consumers should send stable card/session identity, sibling
presentation role, explicit coordinate-space metadata, content frames, layout
recipes, selected/handoff continuity, and finite readiness/activity signals.
Card/provider truth remains owned by card contracts, not by House.

Semantic positioning should emit desired geometry, source/display freshness,
provisional/final status, confidence, `motion.strategy`, and continuity. The VLM
continues to choose targets; House chooses whether the visual motion retargets,
coalesces, snaps, or rematerializes.
