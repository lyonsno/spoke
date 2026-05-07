# Optical Field Contract

The optical-field request contract is the finite data boundary for shared
House-owned optical surfaces. Consumers describe identity, desired geometry,
surface role, presentation order, visibility, profile intent, disturbances, and
handoff metadata. House and the compositor own clocks, lifecycle execution,
material/fill behavior, coalescing, and teardown.

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
