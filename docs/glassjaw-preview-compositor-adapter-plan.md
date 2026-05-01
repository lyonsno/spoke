# Glassjaw Preview Compositor Adapter Plan

Status: accepted for the first `preview.transcription` shared-compositor
migration on `cc/glassjaw-integration-carry-0428`. Tray and recovery remain
overlay-local and are not separate shared-host clients in this slice.

## Gate

The Glassjaw packet allowed the `preview-adapter` lane to plan and write
fail-first witnesses before production migration. The steward accepted the
adapter plan after the assistant half of Glassjaw smoked clean, so this carry
now consumes the accepted host/client API rather than redefining it.

## Target Contract

`TranscriptionOverlay` remains the owner of preview text, typewriter state,
tray state, recovery state, colors, local window choreography, and paste
recovery callbacks. The shared compositor host receives only render snapshots:
identity, generation, visibility, geometry, material, excluded window ids, and
z ordering.

The first shared-host preview client is:

```text
client_id: preview.transcription
role: preview
```

The display id comes from the injected `OverlayCompositorHost`. The adapter
must not derive display ownership from `windowNumber()` alone.

## Adapter Shape

The first migration adds these methods to `TranscriptionOverlay`:

```python
def set_compositor_registry(self, registry) -> None: ...
def _publish_preview_compositor_snapshot(self, *, visible: bool) -> bool: ...
def _release_preview_compositor_client(self) -> None: ...
def _preview_compositor_geometry_snapshot(self) -> OpticalShellGeometrySnapshot: ...
def _preview_compositor_material_snapshot(self) -> OpticalShellMaterialSnapshot: ...
```

The adapter should use the injected registry:

```python
host = self._compositor_registry.host_for_screen(self._screen)
identity = OverlayClientIdentity(
    client_id="preview.transcription",
    display_id=host.display_id,
    role="preview",
)
self._preview_compositor_client = host.register_client(
    identity,
    window=self._window,
    content_view=self._content_view,
)
self._preview_compositor_client.publish(snapshot)
```

Do not call `start_overlay_compositor(...)` from the preview adapter. The
compatibility shim is for the assistant command overlay path; preview must use
the shared registry so it cannot create a second display owner or a second SCK
stream for the same display.

## Geometry

Preview geometry should match the host's current display-local pixel contract:

- center is derived from screen frame, overlay window frame, content view frame,
  and backing scale
- content width, content height, corner radius, band width, and tail width are
  scaled to the same units currently consumed by `FullScreenCompositor`
- excluded window ids include the preview overlay window id

The dataclass field names still say `*_points`; the landed command-overlay
path currently publishes pixel-scaled values. Preview should follow the landed
consumer contract unless host-contract-hardening changes the field meaning.

## Lifecycle

`show()`:

- keeps the existing overlay-local show behavior
- resets preview-local typewriter state
- registers `preview.transcription` through the injected registry
- publishes a visible snapshot with a fresh generation
- does not touch `assistant.command`

`set_text()` and typewriter/layout updates:

- keep all text and typewriter semantics overlay-local
- publish only geometry/material changes needed by the shared compositor
- never put transcript text into the host snapshot

`hide()` and `order_out()`:

- publish a final `visible=False` snapshot for `preview.transcription`
- release only the preview client
- leave any sibling `assistant.command` client and host alive

`show_tray(...)`:

- remains overlay-local in this first migration
- releases or hides the preview client if one is active so stale preview state
  cannot remain in the shared host
- does not register `preview.tray` until the steward accepts a separate tray
  adapter decision

`show_recovery(...)`:

- remains overlay-local in this first migration
- keeps mouse/callback/clipboard behavior in `TranscriptionOverlay`
- releases or hides the preview client if one is active
- does not register `preview.recovery` until the steward accepts a separate
  recovery adapter decision

## Witness Surface

The fail-first witnesses in `tests/test_preview_compositor_adapter.py` pin the
first gate:

- preview uses an injected registry rather than `start_overlay_compositor(...)`
- preview publishes `OverlayRenderSnapshot` as `preview.transcription`
- hide publishes `visible=False` and releases only the preview client
- existing assistant client state is not mutated by preview publish/hide
- tray mode stays overlay-local and does not register a shared-host tray client

These witnesses intentionally failed before the migration because
`TranscriptionOverlay` had no preview compositor adapter yet. On the carry they
now pass with the adapter in place.

## Acceptance

The adapter plan has been accepted for the first preview transcription slice.
The implemented scope remains deliberately narrow: one `preview.transcription`
client, one shared registry/host path, no tray/recovery client split, and no
assistant command-overlay behavior rewrite.
