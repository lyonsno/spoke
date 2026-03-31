# Claude Code Instructions

## Repo Identity

The canonical repo and product name is `spoke`.

When writing or updating docs, reviews, Epistaxis notes, PR text, release notes, or other outward-facing references for this repo, use `spoke` rather than `donttype` or `dictate`.

Treat the repo as renamed for documentation purposes and keep naming consistent with `spoke`.

## Testing

Always run `uv run pytest -q` after code changes and before committing. All tests must pass.

## Smoke testing

There are two Automator-bound launcher scripts:

- `scripts/launch-dev.sh` — launches from `~/.config/spoke/dev-target` when that file exists; otherwise falls back to the checkout containing the script.
- `scripts/launch-smoke.sh` — launches from whatever worktree path is written
  in `~/.config/spoke/smoke-target`.

When you want the stable dev hotkey to follow a fresh main/dev worktree, point
the dev launcher at it:

```sh
echo '/path/to/worktree' > ~/.config/spoke/dev-target
```

When a change is ready for human smoke testing, point the smoke launcher at
the active worktree and tell the user it's ready:

```sh
echo '/path/to/worktree' > ~/.config/spoke/smoke-target
```

The user triggers the smoke Automator hotkey themselves. Do not kill the
running process or relaunch — `launch-smoke.sh` handles that.

Per-worktree env overrides can go in `.spoke-smoke-env` at the worktree root
(e.g. `SPOKE_COMMAND_URL`, `SPOKE_TTS_VOICE`).

After pointing the smoke target, **ask the user if the spacebar is working**
before doing anything else. There is no way to verify event tap functionality
from logs or process state.

When changing or resetting box-local hotkeys:

- treat `~/.config/wezterm/wezterm.lua` and `~/.config/spoke/*-target` files as one contract
- prefer retargeting the target files over editing scripts when only the launched worktree changes
- check the corresponding `~/Library/Logs/spoke-*-launch.log` before assuming the binding itself is dead
- record durable hotkey/reset rules in repo docs and `spoke` Epistaxis

When the launch-target menu feature is in play:

- treat `~/.config/spoke/launch_targets.json` as the curated source for menu-visible launch targets
- agents may add, remove, or retarget entries there when preparing or retiring local smoke surfaces
- there is no dedicated `smoke_branch` slot; additional prepared surfaces should appear as their own explicit registry entries
- prefer stable ids and short human labels
- when `⌃⌥⌘K` and the menu should refer to the same smoke surface, keep `~/.config/spoke/smoke-target` and the registry entry with id `smoke` aligned
- do not silently assume the selected target also carries the launch-target submenu; call that out when preparing a target that does not
- record durable registry conventions or machine-local target changes in `spoke` Epistaxis

## Building the .app bundle

For .app distribution testing (not normal dev smoke testing):

```sh
pkill -TERM -f "Spoke" 2>/dev/null
sleep 1
rm -f ~/Library/Logs/.spoke.lock
./scripts/build.sh --fast
rm -rf ~/Applications/Spoke.app
cp -r dist/Spoke.app ~/Applications/
open ~/Applications/Spoke.app
```

For full clean builds (after dependency changes, spec file changes, or when
`--fast` builds behave unexpectedly):

```sh
./scripts/build.sh
```

## Permissions

- **Ad-hoc signed .app bundles cannot reliably get Accessibility on Sequoia.** System Settings shows the toggle as on but silently drops the grant — it never writes to the TCC database. This is a Sequoia limitation, not a bug in Spoke.
- **Workaround for development**: Run via `uv run spoke` from Terminal. Terminal.app has a real Apple signature with a working Accessibility grant, and the Python process inherits it.
- **Permanent fix**: Sign with a Developer ID certificate ($99/yr Apple Developer Program). This is also required for distribution (notarization, no Gatekeeper warnings).
- Every rebuild changes the ad-hoc signature, which may invalidate macOS TCC permissions (Accessibility, Microphone).
- The app has a graceful permissions flow that retries automatically — the user just needs to grant when prompted.
- If permissions stop working after a rebuild (app says "no permissions" despite being granted in System Settings), the TCC daemon has cached a stale CDHash. Fix: `sudo tccutil reset Accessibility com.noahlyons.spoke`, re-grant in System Settings, and **reboot** to flush the daemon's in-memory cache. The reset alone is not sufficient without a reboot.
- Do NOT run `tccutil reset` in build scripts — it causes more problems than it solves. Only use it as a manual recovery step.
- Do NOT change the bundle identifier to work around TCC — Sequoia won't prompt for Accessibility for unknown ad-hoc apps.
- Use `pkill -TERM` (not `-9`) to kill the app so the SIGTERM handler can cleanly uninstall the CGEventTap.
- After rebuilding and relaunching, **ask the user if the spacebar is working** before doing anything else. There is no way to verify event tap functionality from logs or process state.

## Local assistant (command pathway)

The assistant requires a local OpenAI-compatible model server. The app defaults
to `http://localhost:8001` (OMLX) when `SPOKE_COMMAND_URL` is not set.

Authentication: the app reads `SPOKE_COMMAND_API_KEY` first, then falls back to
`OMLX_SERVER_API_KEY` from the environment. Both are typically set in the
user's shell profile. If the assistant menu appears but is empty ("couldn't
reach the model"), the most likely cause is the model server not running or the
API key not reaching the app process.

When preparing a new worktree or smoke surface, do not assume the assistant
will work without checking. The command pathway is always enabled but the model
server and API key must be reachable from the launched process.

## Epistaxis

When reading epistaxis, if any recorded state doesn't match what you observe in the code or thread, flag it before proceeding — even if the mismatch might just be stale rather than wrong. Multiple sessions may write to the same epistaxis file concurrently; merge your changes without overwriting entries you didn't write.

## Epistaxis Intent Model

For `spoke`, do not treat `Repo/task` in `**Current intent**` as a single
repo-global active intent that must summarize the whole repository.

`spoke` can carry one durable strategic direction while multiple active
surfaces proceed in parallel. In this repo, use the layers below:

- `Session:` the active intent for the current thread.
- `Repo/task:` the specific surface, branch, worktree, or task this session is
  advancing. It does not need to summarize unrelated concurrent work.
- Strategic direction: durable product-level direction belongs in repo
  Epistaxis status/decisions or roadmap surfaces, not in the per-session
  `Repo/task` line.

When updating `spoke` Epistaxis state:

- Keep concurrent surfaces as separate scoped local state entries.
- Name a default continuation surface only when one is actually intended as the
  default for future pickup.
- Do not churn `**Current intent**` just because another unrelated surface is
  also active.
- Treat incoherence as contested surface ownership, landing target, shared
  invariant, or contradictory strategic direction, not merely the existence of
  several active branches.

## Demo video

- `scripts/demo-convert.sh` converts screen recordings (.mov) to optimized MP4s. Run with `--help` for options.
- GitHub README only renders inline `<video>` with `user-attachments` URLs. Release download URLs and raw.githubusercontent URLs are silently stripped.
- The only way to get a `user-attachments` URL is drag-and-drop into a GitHub issue/PR comment in the browser. Wait for the "Uploading..." placeholder to resolve to an actual URL before submitting — if you submit while it still says uploading, the upload is lost.
- `gh` CLI cannot upload to `user-attachments`. Don't waste time trying.

## Commits

Use descriptive commit messages. Include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in all commits.
Unless the user explicitly says otherwise, push commits after creating them.
