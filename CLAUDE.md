# Claude Code Instructions

# Spoke Policy Core

<!-- Shared source of truth. Compiled into CLAUDE.md and AGENTS.md by tools/build-policy.sh -->

## Repo Identity

The canonical repo and product name is `spoke`.

When writing or updating docs, reviews, Operator Memory notes, PR text, release notes, or other outward-facing references for this repo, use `spoke` rather than `donttype` or `dictate`.

Treat the repo as renamed for documentation purposes and keep naming consistent with `spoke`.

## Branching

**`main` is the integration branch.** `main-next` is retired — it was
force-pushed onto `main` on 2026-04-04 and the two are now identical.
All new feature branches, fix branches, and worktrees must be sliced
from `origin/main`. Do not use `main-next`, `dev`, or any other branch
as a base unless the human explicitly asks for some historical or
non-trunk surface.

Before creating a worktree: `git fetch origin main` and branch from
`origin/main`.

Treat remote `origin/main` as the source of truth rather than any older
local trunk witness or smoke worktree. Do not present an older local
worktree as "the current tip" just because it was the last place a smoke
run happened.

## Integration Landings

When the user says to merge or land something on the integration branch,
that means remote `origin/main`.

An intermediate branch may still be used as a temporary landing carrier
for verification, but it is only a short-lived transport surface:

- cut it fresh from the then-current `origin/main`
- port the intended change there
- run the relevant verification there
- remote-merge it to `main`
- delete the branch and worktree immediately after merge

Do not present an intermediate landing branch as a second integration
branch, as a durable trunk variant, or as a user-facing launch target
unless the human explicitly asked for that separate surface.

If a feature branch was not directly smoke-ready on its own base and had
to be rebased, cherry-picked, or otherwise carried onto trunk-compatible
support work, the smoke-ready surface must itself be re-sliced from the
then-current `origin/main` before being called ready. Do not smoke or
hand off an older pre-carry surface as though it were current trunk.

## Testing

Always run `uv run pytest -q` after code changes and before committing. All tests must pass.

## Topothesia

`spoke` uses Topothesia for README-vs-operator/developer routing. Before
restoring or removing a README note about a real capability, check
`docs/documentation_surfaces.toml` and the routed canonical surface first.

When the human says `Make this durable for review`, treat that as a request to
choose the narrowest durable review-control surface:

- use Topothesia review surfaces when the issue is authority, canonical-vs-
  fallback interpretation, allowed divergence, or review-routing semantics;
  consult `docs/review_surfaces.toml` and `docs/review-authority-surfaces.md`
- use Prilosec when the issue is a recurring acknowledged false positive or
  accepted finding family that future reviews should suppress or demote
- use a review-artifact disposition only when the issue is specific to one
  reviewed commit and does not need a standing repo-level rule

For the current `spoke` contract, the public README is intentionally not the
home for smoke-only runtime affordances or developer-only repair-pass notes.

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

When a new smoke-ready `spoke` surface is ready to hand to the human, repoint
the relevant launcher state to that worktree before calling it ready. Update
both the stable launcher pin and any menu/registry state that governs the same
surface. This repointing step is autonomous and is distinct from relaunching.

Do not present a surface as smoke-ready if invoking the intended launcher would
still reopen an older worktree. Do not present a surface as smoke-worthy unless
that branch already carries the menubar launch-target affordance needed to
select and identify it from the visible launcher UI.

"Smoke-ready" for `spoke` means all of the following are true:

- the branch carries any launcher or launch-target code it depends on
- the relevant launcher target or registry entry points at the intended worktree
- any required `.spoke-smoke-env` is present and correct
- the launcher log shows it is launching from the intended path, not a fallback
- if exercised live, the app's `Source:` and `Branch:` lines match

If those conditions are not met, say the branch is code-ready or branch-ready,
not smoke-ready.

## Local hotkey policy

When changing or resetting local smoke hotkeys:

- treat the live WezTerm mapping and the launcher target files as one contract
- keep the meaning legible: `Space` for pinned `main`, `K` for the current smoke target, and any optional extra smoke binding clearly named and logged
- prefer retargeting `~/.config/spoke/*-target` files over editing launcher scripts when only the destination worktree changes
- if a hotkey fails, check the corresponding `~/Library/Logs/spoke-*-launch.log` first to distinguish dead binding from launcher/runtime failure
- treat missing target-worktree `.venv` bootstrap as launcher responsibility by default
- if a launcher needs a known-good interpreter, set `SPOKE_VENV_PYTHON` in that worktree's `.spoke-smoke-env` instead of hardcoding a machine path
- record any durable remap or reset rule in repo docs and `spoke` Operator Memory

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
- record durable registry conventions or machine-local target changes in `spoke` Operator Memory when another session would need them to resume coherently

## Secrets

Machine-local secrets (API keys, access tokens) for every spoke
surface on a box live in a single file at `~/.config/spoke/secrets.env`.
The launcher (`scripts/launch-main.sh`) sources this file into the
child environment before applying per-worktree `.spoke-smoke-env`
overrides.

**Never commit real secret values anywhere.** `~/.config/spoke/secrets.env`
lives outside the repo on purpose. A tracked template at
`scripts/secrets.env.example` documents the expected shape with empty
values — copy it to the real location on each new box:

```sh
mkdir -p ~/.config/spoke
cp scripts/secrets.env.example ~/.config/spoke/secrets.env
chmod 600 ~/.config/spoke/secrets.env
# then edit and populate from your offline source of truth
```

The single cross-project registry that lists secret-file locations,
provenance, and rotation history (never values) lives at
`~/dev/operator-memory/system/secrets.md`. Consult that rather than
re-deriving where to put things. When adding a new secret to spoke,
update both `scripts/secrets.env.example` and that registry.

Gemini API key specifics: spoke reads `GEMINI_API_KEY_INACTIVE` before
`GEMINI_API_KEY` via `spoke/__main__.py::_gemini_api_key_env()`. The
`_INACTIVE` alias exists so a spoke-only key can sit in the same shell
or secrets file without colliding with the Gemini CLI.

After updating the registry, **ask the user if the spacebar is working**
before doing anything else. There is no way to verify event tap functionality
from logs or process state.

## Service fleet

Spoke runs as a coordinated fleet of isolated services. The core app handles
UI, orchestration, and input; sidecar services handle inference workloads in
their own venvs.

**Fleet manifest:** `services.yaml` at the repo root describes every service —
ports, env vars, health endpoints, rebuild instructions, and which box runs
what. Read it before debugging connectivity or preparing a new smoke surface.

**Health check:** `scripts/spoke-doctor.sh` pings every expected service and
reports what's up, what's down, and what URL it tried.

```sh
./scripts/spoke-doctor.sh            # quick status
./scripts/spoke-doctor.sh --verbose  # include response snippets
```

The fleet topology as of 2026-04-07:

| Service | Default URL | Required? |
|---------|------------|-----------|
| Grapheus (commands) | `localhost:8090` | Yes |
| OMLX upstream | `localhost:8001` | No |
| Narrator | Falls back to Grapheus | No |
| MLX-audio (TTS/STT) | `MacBook-Pro-2.local:9001` | Yes |
| Whisper (remote) | `nlm2pr.local:7001` | No (local fallback) |

When preparing a new worktree or smoke surface, do not assume sidecars are
reachable. Run `spoke-doctor.sh` first.

## Local assistant (command pathway)

The assistant requires a local OpenAI-compatible command endpoint. The app
defaults to `http://localhost:8090` (Grapheus) when `SPOKE_COMMAND_URL` is not
set. Grapheus is the canonical local command path for `spoke`; it proxies the
local OMLX server at `http://localhost:8001` and captures structured logs.
On `MacBook-Pro-2.local`, `phylax` owns the `grapheus_local` service lifecycle.

Authentication: the app reads `SPOKE_COMMAND_API_KEY` first, then falls back to
`OMLX_SERVER_API_KEY` from the environment.

### Grapheus headers (`X-Spoke-*`)

Every LLM call through Grapheus must send these headers:

- `X-Spoke-Pathway`: which pathway is making the call (e.g. `command`, `narrator`, `positioning`)
- `X-Spoke-Utterance-ID`: a stable ID tying all calls from one user action together

Without `Utterance-ID`, Grapheus cannot group calls into logical requests.
`X-Spoke-Turn` (round number) and `X-Spoke-Step` (pipeline stage) are
optional but recommended for multi-round or multi-step pathways.

## Operator Memory

When reading operator-memory, if any recorded state doesn't match what you observe
in the code or thread, flag it before proceeding — even if the mismatch might
just be stale rather than wrong. Multiple sessions may write to the same
operator-memory file concurrently; merge your changes without overwriting entries
you didn't write.

## Operator Memory Intent Model

For `spoke`, do not treat `Repo/task` in `**Current intent**` as a single
repo-global active intent that must summarize the whole repository.

`spoke` can carry one durable strategic direction while multiple active
surfaces proceed in parallel. In this repo, use the layers below:

- `Session:` the active intent for the current thread.
- `Repo/task:` the specific surface, branch, worktree, or task this session is
  advancing. It does not need to summarize unrelated concurrent work.
- Strategic direction: durable product-level direction belongs in repo
  Operator Memory status/decisions or roadmap surfaces, not in the per-session
  `Repo/task` line.

When updating `spoke` Operator Memory state:

- Keep concurrent surfaces as separate scoped local state entries.
- Name a default continuation surface only when one is actually intended as the
  default for future pickup.
- Do not churn `**Current intent**` just because another unrelated surface is
  also active.
- Treat incoherence as contested surface ownership, landing target, shared
  invariant, or contradictory strategic direction, not merely the existence of
  several active branches.

## Commits

Use descriptive commit messages. Include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in all commits.
Unless the user explicitly says otherwise, push commits after creating them.
If commit/push is the required next step, needing sandbox/escalation approval is not a reason to defer it or leave work local-only. Request the permission and continue.

## Demo video

- `scripts/demo-convert.sh` converts screen recordings (.mov) to optimized MP4s. Run with `--help` for options.
- GitHub README only renders inline `<video>` with `user-attachments` URLs. Release download URLs and raw.githubusercontent URLs are silently stripped.
- The only way to get a `user-attachments` URL is drag-and-drop into a GitHub issue/PR comment in the browser.
- `gh` CLI cannot upload to `user-attachments`. Don't waste time trying.

---

<!-- Sources: policy/shared/core.md — read it first, this file adds Claude Code deltas only -->

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
- If permissions stop working after a rebuild, the TCC daemon has cached a stale CDHash. Fix: `sudo tccutil reset Accessibility com.noahlyons.spoke`, re-grant in System Settings, and **reboot**.
- Do NOT run `tccutil reset` in build scripts.
- Do NOT change the bundle identifier to work around TCC.
- Use `pkill -TERM` (not `-9`) to kill the app so the SIGTERM handler can cleanly uninstall the CGEventTap.
- After rebuilding and relaunching, **ask the user if the spacebar is working** before doing anything else.
