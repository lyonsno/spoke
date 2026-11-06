# Standalone WezTerm sidebar

`python -m spoke.wezterm_lanes_sidebar` runs the standalone native accessory
used by `wezterm-lanes-hud`. It uses the focused WezTerm producer's rows and
per-window visibility. `--ensure-daemon` starts it only when its existing
singleton lock is available. This does not launch the full Spoke app.

The pinned Greenroom footer reads a derived snapshot; Greenroom owns all
requests, decisions, escalation, priority and freshness policy. Clicking its
queue summary opens the monitor. Clicking the leading request opens that
item in the monitor, where the operator can act. The attention count includes
every unresolved item supplied by the producer; the monitor is the full list.
Long titles elide visually and retain their full text in the tooltip.

Choose the snapshot with `--greenroom-attention-path /absolute/path.json`,
or `SPOKE_GREENROOM_ATTENTION_PATH`. The default is
`$GPU_GREENROOM_DIR/operator-attention.json`, falling back to
`~/.local/state/gpu-greenroom/operator-attention.json`. An explicit CLI path
is forwarded when `--ensure-daemon` spawns the helper. Changing a running
helper's configuration requires restarting that helper.

The agreed producer schema is `gpu-greenroom.operator-attention.v1`:

```json
{
  "schema": "gpu-greenroom.operator-attention.v1",
  "observed_at": "2026-11-06T12:00:00Z",
  "valid_until": "2026-11-06T12:01:00Z",
  "monitor_url": "http://localhost:8768/",
  "queue": {"running": 1, "waiting": 2},
  "items": [{
    "id": "attempt-bound-request-id",
    "title": "Confirm performance window",
    "status": "pending",
    "target_url": "http://localhost:8768/#attention-attempt-bound-request-id"
  }]
}
```

These example URLs and times are illustrative. The effective producer supplies
working links and its shared monitor deadline. Timestamps require explicit
timezones; queue counts are nonnegative integers; item IDs are unique. Items
contain the complete unresolved set in producer priority order. The consumer
does not interpret status names as state transitions. Unknown additive fields
remain compatible. Navigation accepts HTTP(S) URLs only.

The reader stats the file on the sidebar's existing visible update cadence
(approximately 480 ms), reparses only after inode/mtime/size changes, and checks
the producer deadline without rereading unchanged JSON. Hidden sidebar polling
does no Greenroom reads. No queue CLI, HTTP poll, worker or new timer is added.
Missing or malformed snapshots show unavailable and clear navigation. Expired
snapshots show stale and disable request navigation; the validated monitor link
remains available. Future observations show clock mismatch. Clicks recheck
freshness and current targets before opening a link. The footer remains below
the lane scroll view across window changes.

Focused checks: `uv run pytest -q tests/test_wezterm_greenroom_footer.py
tests/test_wezterm_lanes_sidebar.py`. Native tests run actual AppKit controls
without displaying another window; URL opening is intercepted to avoid browser
activation. A real generated snapshot and monitor response remain the consumer
conformance witness for producer integration.
