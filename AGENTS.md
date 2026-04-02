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

Do not call a surface "smoke-ready" unless the intended user-facing launcher path is actually ready, not just the code in the worktree. "Smoke-ready" for `spoke` means all of the following are true:

- the branch carries any launcher or launch-target code it depends on; do not rely on unrelated neighboring branches for launcher behavior
- the relevant launcher target file or launch-target registry entry points at the intended existing worktree
- any required per-worktree `.spoke-smoke-env` is present and does not silently force a different surface or endpoint than the one being claimed
- the relevant launcher log shows it is launching from the intended worktree/path, not a fallback surface
- if the launcher was exercised live, the running app's `Source:` and `Branch:` lines agree with the claimed surface

If those conditions are not met, say the branch is code-ready or branch-ready, not smoke-ready.

## Local hotkey policy

When changing or resetting local smoke hotkeys:

- treat the live WezTerm mapping and the launcher target files as one contract
- keep the meaning legible: `Space` for pinned `main`, `K` for the current smoke target, and any optional extra smoke binding clearly named and logged
- prefer retargeting `~/.config/spoke/*-target` files over editing launcher scripts when only the destination worktree changes
- if a hotkey fails, check the corresponding `~/Library/Logs/spoke-*-launch.log` first to distinguish dead binding from launcher/runtime failure
- treat missing target-worktree `.venv` bootstrap as launcher responsibility by default: checked-in launchers should repair it with `uv sync --extra tts --group dev` before first run rather than assuming a separate remembered manual step
- if a launcher needs a known-good interpreter because a fresh worktree venv crashes, set `SPOKE_VENV_PYTHON` in that worktree's `.spoke-smoke-env` instead of hardcoding a machine path into the shared launcher
- record any durable remap or reset rule in repo docs and `spoke` Epistaxis rather than leaving it only in thread history
