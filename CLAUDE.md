# Claude Code Instructions

## Repo Identity

The canonical repo and product name is `spoke`.

When writing or updating docs, reviews, Epistaxis notes, PR text, release notes, or other outward-facing references for this repo, use `spoke` rather than `donttype` or `dictate`.

Treat the repo as renamed for documentation purposes and keep naming consistent with `spoke`.

## Branching

**`main` is the integration branch.** `main-next` is retired — it was
force-pushed onto `main` on 2026-04-04 and the two are now identical.
All new feature branches, fix branches, and worktrees must be sliced
from `origin/main`. Do not use `main-next`, `dev`, or any other branch
as a base.

Before creating a worktree: `git fetch origin main` and branch from
`origin/main`.

## Testing

Always run `uv run pytest -q` after code changes and before committing. All tests must pass.

## Smoke testing

One Automator-bound launcher script: `scripts/launch-main.sh`, bound to
`Ctrl+Opt+Cmd+Space`. It reads the launcher registry
(`~/.config/spoke/launch_targets.json`), launches the selected target, and
kills any existing spoke instance first. If the selected target path is
missing or invalid, it falls back to the checkout containing the script and
shows a macOS notification to indicate fallback.

There are no file-based launcher targets (`main-target`, `dev-target`,
`smoke-target`). Those are retired. The registry is the single source of
truth for which worktree launches.

**Every smoke-ready surface must be added to the launcher registry** with:
- `id`: short snake_case identifier
- `label`: human-readable label (use the operation codename if one exists)
- `path`: absolute path to the worktree
- `note` (optional): branch name and one-line description

If the surface has a sēmeion (operation codename), use it as the label.
Include the branch name in the `note` field so the user can identify which
code is running. Select the target in the registry's `selected` field.

Per-worktree env overrides can go in `.spoke-smoke-env` at the worktree root
(e.g. `SPOKE_COMMAND_URL`, `SPOKE_TTS_VOICE`).

Do not tell the user a surface is "smoke-ready" unless the launcher path is
ready too. For `spoke`, that means:

- the registry entry has been added and selected
- required `.spoke-smoke-env` values are present and correct
- `~/Library/Logs/spoke-main-launch.log` shows the launcher choosing the
  intended worktree/path rather than a fallback
- if the app has been relaunched, the menubar `Source:` and `Branch:` lines
  match the claimed surface

If only the code is ready but the launcher contract has not been checked,
describe it as branch-ready or code-ready, not smoke-ready.

After updating the registry, **ask the user if the spacebar is working**
before doing anything else. There is no way to verify event tap functionality
from logs or process state.

## Launch target registry policy

When the launch-target menu feature is in play:

- for `spoke`, a surface is not smoke-ready unless it is present in `~/.config/spoke/launch_targets.json` and launchable from the visible launcher UI on that machine
- treat `~/.config/spoke/launch_targets.json` as the curated source for menu-visible launch targets
- agents may add, remove, or retarget entries there when preparing or retiring local smoke surfaces
- there is no dedicated `smoke_branch` slot; additional prepared surfaces should appear as their own explicit registry entries
- prefer stable ids and short human labels; the entry should identify a purposeful surface, not a temporary hunk of local reasoning
- when `⌃⌥⌘K` and the menu should refer to the same smoke surface, keep `~/.config/spoke/smoke-target` and the registry entry with id `smoke` aligned
- do not silently assume the selected target also carries the launch-target affordance; if the target branch lacks the feature, say so when preparing the surface
- smoke-worthy surfaces must carry the launcher/menu commits that make the target selectable and legible in the menubar; registry prep alone is not enough
- record durable registry conventions or machine-local target changes in `spoke` Epistaxis when another session would need them to resume coherently

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
If commit/push is the required next step, needing sandbox/escalation approval is not a reason to defer it or leave work local-only. Request the permission and continue.
