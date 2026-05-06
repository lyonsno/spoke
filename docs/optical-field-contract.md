# Optical Field Primitive Contract

This is the consumer-facing contract for shared optical-shell UI surfaces.
Consumers describe desired state; the primitive/backend owns how the field
materializes, resizes, recenters, rests, dismisses, and cleans itself up.

## Consumer Shape

Use `spoke.optical_field.OpticalFieldRequest` as the adapter boundary:

```python
from spoke.optical_field import (
    OpticalFieldBounds,
    OpticalFieldProfileRef,
    OpticalFieldRequest,
    OpticalFieldTransitionTiming,
)

request = OpticalFieldRequest(
    caller_id="agent.card.codex-1",
    bounds=OpticalFieldBounds(x=40, y=80, width=320, height=96),
    role="agent_card",
    profile=OpticalFieldProfileRef(base="agent_card"),
).as_materializing(
    timing=OpticalFieldTransitionTiming(
        duration_s=0.18,
        attack_curve="ease_out_cubic",
        release_curve="critically_damped",
    )
)
```

Required consumer-owned data:

- `caller_id`: stable namespaced visual id, such as `agent.card.<session-id>`.
- `bounds`: desired compositor-space rectangle.
- `role`: coarse caller role for ordering and diagnostics.
- `profile`: named material language plus bounded profile/slot overrides.
- `transition_timing`: optional data-only timing and attack/release curve hints.
- `disturbances`: optional data-only transient or persistent field requests.

Primitive-owned behavior:

- lifecycle progress and frame stepping
- shader constants and field composition
- old-to-new resize interpolation
- center-to-center recenter interpolation
- SDF/mask/fill continuity
- interruption and cleanup of stranded summon/dismiss/resize/recenter state

## Lifecycle Helpers

Prefer helpers over hand-written state strings:

```python
request = request.as_materializing()
request = request.as_resting()
request = request.resize_to(OpticalFieldBounds(x=60, y=80, width=420, height=120))
request = request.recenter_to(center_x=480, center_y=160)
request = request.as_dismissing()
request = request.as_hidden()
```

`resize_to(...)` records `previous_bounds`, sets `state == "resize"`, and
uses the supplied target bounds as the latest desired rectangle.

`recenter_to(...)` records `previous_bounds`, sets `state == "recenter"`, and
preserves the current width and height while changing the target center.

Both helpers accept `timing=OpticalFieldTransitionTiming(...)`. Timing is data,
not progress. Consumers may name duration and attack/release curves, but they
must not tick animation steps, mutate phase progress, or call shader/fill phases
directly.

## Profiles And Slots

Profiles are named visual languages. Current bases are available from
`available_optical_field_profiles()` and include:

- `assistant_shell`
- `preview_pill`
- `agent_card`
- `quiet_chip`

Per-slot overrides are allowed as data:

```python
from spoke.optical_field import OpticalFieldSlotOverride

profile = OpticalFieldProfileRef(
    base="agent_card",
    slots={
        "resize": OpticalFieldSlotOverride(
            params={"ring_amplitude_frac": 0.08}
        ),
        "recenter": OpticalFieldSlotOverride(
            params={"band_width_frac": 0.07}
        ),
    },
)
```

Slots map to primitive lifecycle phases: `rest`, `materialize`, `dismiss`,
`resize`, and `recenter`. `hidden` compiles no visible shell config.

## Mailbox Boundary

The placeholder backend stores the latest request per `caller_id`, so repeated
resize/recenter/materialize/dismiss/rest requests for one caller expose only the
newest desired state to the compiler. The full retarget mailbox owns stale epoch
rejection, sampled-presented-bounds custody, and interruption details. Consumers
should send fresh desired state rather than FIFO animation queues.

## Do Not

- Do not choose semantic destination rectangles in this primitive layer.
- Do not fork shader parameters into consumer code.
- Do not hand-drive materialization, dismiss, resize, or recenter progress.
- Do not compensate locally for stranded displacement, missing fill, wrong
  scale, resize catching, or old-rect residue.
- Do not treat `AgentShellPrimitive.material` as already equal to this API; use
  an adapter that maps rendered surfaces to `OpticalFieldRequest`.

If a consumer sees stranded warp, missing fill, wrong scale, resize slow-motion,
recenter drift, or summon/dismiss residue, file it against the House primitive
or packet rather than patching around it downstream.

## Placeholder Backend

`OpticalFieldPlaceholderBackend` is intentionally a compatibility bridge. It
keeps the request API stable while compiling into today's shell-config
dictionaries. Consumers may use it for tests and early adapters, but should keep
their code shaped around `OpticalFieldRequest`, not around the compiled config.
