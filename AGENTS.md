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

## Smoke-test branch launches

When the user asks to spin up a separate fun or smoke-test branch, treat that as a request to launch the dedicated worktree for that branch rather than the stable default launcher path.

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
- keep box-local launch defaults in `~/.config/spoke/launch-env.sh`; treat per-worktree `.spoke-smoke-env` files as branch-specific overrides layered on top of that shared launch env
- treat `.spoke-smoke-env` as a durable launcher contract, not an informal shell snippet: keep any required command/TTS runtime vars explicit there, and do not assume the default repo venv will still provide Voxtral or other local smoke-only backends on future branches
- record any durable remap or reset rule in repo docs and `spoke` Epistaxis rather than leaving it only in thread history

## Launch target registry policy

When the launch-target menu feature is in play:

- treat `~/.config/spoke/launch_targets.json` as the curated source for menu-visible launch targets
- agents may add, remove, or retarget entries there when preparing or retiring local smoke surfaces
- when a surface is ready for human smoke, add it to the registry and rely on the shared launch env plus any branch-specific `.spoke-smoke-env` override instead of copy-pasting the full box-local runtime block into every worktree
- there is no dedicated `smoke_branch` slot; additional prepared surfaces should appear as their own explicit registry entries
- prefer stable ids and short human labels; the entry should identify a purposeful surface, not a temporary hunk of local reasoning
- when `⌃⌥⌘K` and the menu should refer to the same smoke surface, keep `~/.config/spoke/smoke-target` and the registry entry with id `smoke` aligned
- do not silently assume the selected target also carries the launch-target affordance; if the target branch lacks the feature, say so when preparing the surface
- record durable registry conventions or machine-local target changes in `spoke` Epistaxis when another session would need them to resume coherently
