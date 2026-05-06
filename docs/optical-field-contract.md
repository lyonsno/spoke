# Optical Field Contract

The optical-field request contract is the finite data boundary for shared
House-owned optical surfaces. Consumers describe identity, desired geometry,
surface role, presentation order, visibility, profile intent, freshness,
provisional/final status, disturbances, and handoff metadata. House and the
compositor own clocks, lifecycle execution, material/fill behavior, geometry
continuity, coalescing, and teardown.

## Sibling Surface Arbitration

Assistant shell, user preview, agent card, and HUD surfaces are sibling optical
presences. A consumer must not make one of those surfaces a hidden child of the
assistant overlay or repair ordering through private AppKit window-level
changes.

Use `OpticalFieldPresentation` on `OpticalFieldRequest` when ordering matters:

```python
OpticalFieldRequest(
    caller_id="preview.transcription",
    role="user_preview",
    bounds=OpticalFieldBounds(x=20.0, y=30.0, width=320.0, height=72.0),
    presentation=OpticalFieldPresentation(layer="user_preview", order=40),
    z_index=0,
)
```

The compositor sorts visible siblings by:

1. `presentation.order`
2. `z_index`
3. original request order for the placeholder backend, or client id in the
   shared compositor host

Default presentation layers are derived from existing roles for compatibility:

| Role | Presentation layer | Order |
| --- | --- | --- |
| `assistant` / `assistant_shell` | `assistant_shell` | `20` |
| `agent_card` | `agent_card` | `20` |
| `preview` / `user_preview` | `user_preview` | `30` |
| `hud` / `tray` / `recovery` | `hud` | `50` |

Sibling surfaces use `visibility_scope="independent"`. Production requests for
`agent_card`, `preview`, `user_preview`, and `hud` are rejected if they attempt
to follow assistant-shell visibility. Hiding the assistant shell must not hide a
visible card, preview, or HUD request.

Selected card handoff is metadata, not card truth. Use
`OpticalFieldSelectedHandoff` to describe the optical boundary between an agent
card and the assistant shell:

```python
OpticalFieldSelectedHandoff(
    from_caller_id="agent.card.codex-1",
    to_caller_id="assistant.command",
    continuity_key="codex-session-1",
    mode="handoff",
)
```

The handoff fields round-trip through `optical_field` metadata. They do not
redefine provider/session identity, selected-card semantics, transcript
visibility, or lifecycle phase.

## Retargetable Geometry Mailbox

Each `caller_id` has one latest accepted target. The mailbox is not FIFO:
newer valid same-caller geometry replaces older unplayed geometry unless a
future request explicitly declares choreographed sequencing. Current requests
support these lifecycle states:

- `materialize`
- `rest`
- `resize`
- `recenter`
- `dismiss`
- `hidden`

The backend keeps primitive-owned transition state for each caller:

- `previous_bounds`: the bounds a new transition starts from
- `presented_bounds`: the latest sampled visual bounds currently on screen
- `target_bounds`: the latest accepted desired bounds
- `target_request`: the latest accepted request data
- `pending_request`: always `None` in the default coalescing mailbox

When an in-flight transition is interrupted, the primitive retargets from the
latest sampled `presented_bounds`, not from stale requested `from_bounds`.
Consumers may report sampled visual bounds through
`sample_presented_bounds(caller_id, bounds)` or pass `presented_bounds` while
upserting the next request.

## Freshness

Requests may carry:

- `display_epoch`: coordinate-space or display-capture freshness
- `source_epoch`: semantic/capture/source freshness when applicable
- `provisional`: whether this is an optimizer/intermediate target

For a single caller, requests with older `display_epoch` or `source_epoch` than
the latest accepted request are rejected and do not alter the current target.
Final targets can interrupt provisional targets at the same epoch. A provisional
target cannot overwrite a final target unless it carries a newer freshness
epoch.

## Dismiss And Hidden

`dismiss` and `hidden` are latest desired states for the same caller, not side
queues. Accepting either one clears any pending materialize, resize, or recenter
intent by replacing the caller's target. `hidden` requests do not compile to a
visible shell config.

## Compositor Payload

Compiled placeholder shell configs preserve the existing `optical_field`
metadata. When a request carries freshness or active geometry custody, the
metadata also includes a `transition` block:

```python
{
    "from_bounds": (x, y, width, height),
    "presented_bounds": (x, y, width, height),
    "target_bounds": (x, y, width, height),
    "display_epoch": 3,
    "source_epoch": 9,
    "provisional": False,
}
```

This payload is compositor-owned state. Consumers must not reconstruct previous
visual bounds, install private geometry queues, or drive warp/fill phases to
compensate for frequent geometry updates.
