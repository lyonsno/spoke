# Repo AGENTS.md

This file adds repo-specific policy on top of the global AGENTS guidance.

## Repo Identity

The canonical repo and product name is `spoke`.

When writing or updating docs, reviews, Epistaxis notes, PR text, release notes, or other outward-facing references for this repo, use `spoke` rather than `donttype` or `dictate`.

Treat the repo as renamed for documentation purposes and keep naming consistent with `spoke`.

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

## Commits

Unless the user explicitly says otherwise, push commits after creating them.

## Integration Tip

For `spoke`, the canonical integration tip is remote `origin/main-next`, not
whichever older local `main-next` carry or smoke worktree happens to exist.

When a dedicated local trunk surface is in use, refer to it by its visible
launcher label, `Main Next Trunk`, but only treat it as the current integration
tip after it has been refreshed or recreated from current `origin/main-next`.

Do not present an older local `main-next` worktree as "the current tip" just
because it is named like a trunk surface or was the last place a smoke run
happened.

## Integration Landings

When the user says to merge or land something on the integration branch, that
means remote `origin/main-next`.

An intermediate branch may still be used as a temporary landing carrier for
verification, but it is only a short-lived transport surface:

- cut it fresh from the then-current `origin/main-next`
- port the intended change there
- run the relevant verification there
- remote-merge it to `main-next`
- delete the branch and worktree immediately after merge

Do not present an intermediate landing branch as a second integration branch,
as a durable trunk variant, or as a user-facing launch target unless the human
explicitly asked for that separate surface.

If a feature branch was not directly smoke-ready on its own base and had to be
rebased, cherry-picked, or otherwise carried onto trunk-compatible support
work, the smoke-ready surface must itself be re-sliced from the then-current
`origin/main-next` before being called ready. Do not smoke or hand off an older
pre-carry surface as though it were current trunk.

`Main Next Trunk` is the only ordinary local witness label for trunk. Extra
launcher-visible trunk-like labels or surfaces need an explicit human request.

## Smoke-test branch launches

When the user asks to spin up a separate fun or smoke-test branch, treat that as a request to launch the dedicated worktree for that branch rather than the stable default launcher path.

When a new smoke-ready `spoke` surface is ready to hand to the human, repoint the
relevant launcher state to that worktree before calling it ready. Update both
the stable launcher pin and any menu/registry state that governs the same
surface, such as `~/.config/spoke/main-target`, `dev-target`, `smoke-target`,
or `launch_targets.json`, whichever the requested launcher path actually uses.

This repointing step is autonomous and is distinct from relaunching. It does
not authorize killing or relaunching the currently running app. If the old
surface is still live, report that the launcher now points at the new surface
and that the next manual launch will land there.

Do not present a surface as smoke-ready if invoking the intended launcher would
still reopen an older worktree.

Do not present a surface as smoke-worthy unless that branch already carries the
menubar launch-target affordance needed to select and identify it from the
visible launcher UI. If the branch still lacks the relevant launcher/menu
commits, it is not smoke-ready yet even if the underlying feature work is.

Before launching that branch:
- pull or otherwise update the target branch/worktree
- kill the currently running Spoke process
- relaunch from the target worktree's launcher script

Do not silently fall back to the stable Automator or `main` launcher when the user explicitly asked for the branch variant.

## Local hotkey policy

When changing or resetting local smoke hotkeys:

- treat the live WezTerm mapping and the launcher target files as one contract
- keep the meaning legible: `Space` for pinned `main`, `K` for the current smoke target, and any optional extra smoke binding clearly named and logged
- prefer retargeting `~/.config/spoke/*-target` files over editing launcher scripts when only the destination worktree changes
- if a hotkey fails, check the corresponding `~/Library/Logs/spoke-*-launch.log` first to distinguish dead binding from launcher/runtime failure
- if a launcher needs a known-good interpreter because a fresh worktree venv crashes, set `SPOKE_VENV_PYTHON` in that worktree's `.spoke-smoke-env` instead of hardcoding a machine path into the shared launcher
- record any durable remap or reset rule in repo docs and `spoke` Epistaxis rather than leaving it only in thread history

## Launch target registry policy

When the launch-target menu feature is in play:

- treat `~/.config/spoke/launch_targets.json` as the curated source for menu-visible launch targets
- agents may add, remove, or retarget entries there when preparing or retiring local smoke surfaces
- there is no dedicated `smoke_branch` slot; additional prepared surfaces should appear as their own explicit registry entries
- prefer stable ids and short human labels; the entry should identify a purposeful surface, not a temporary hunk of local reasoning
- when `⌃⌥⌘K` and the menu should refer to the same smoke surface, keep `~/.config/spoke/smoke-target` and the registry entry with id `smoke` aligned
- do not silently assume the selected target also carries the launch-target affordance; if the target branch lacks the feature, say so when preparing the surface
- smoke-worthy surfaces must carry the launcher/menu commits that make the target selectable and legible in the menubar; registry prep alone is not enough
- record durable registry conventions or machine-local target changes in `spoke` Epistaxis when another session would need them to resume coherently
