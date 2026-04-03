# Gemini CLI Instructions

## Repo Identity

The canonical repo and product name is `spoke`.

When writing or updating docs, reviews, Epistaxis notes, PR text, release
notes, or other outward-facing references for this repo, use `spoke` rather
than `donttype` or `dictate`.

Treat the repo as renamed for documentation purposes and keep naming
consistent with `spoke`.

## Branching

**`main-next` is the active integration branch.** For `spoke`, all new
feature branches, fix branches, worktrees, and temporary integration-carrier
branches should be sliced from current remote `origin/main-next`, not from
`main` or `dev`, unless the human explicitly asks for some other historical
or non-trunk surface.

Treat remote `origin/main-next` as the source of truth rather than any older
local trunk witness or smoke worktree. A local surface such as `Main Next
Trunk` is only a refreshed witness; do not call it the current tip unless it
has been refreshed or recreated from current `origin/main-next`.

When the user asks to land work on the integration branch, that means remote
`origin/main-next`. Any temporary carrier branch used for verification should
be cut fresh from current `origin/main-next`, verified there, remote-merged,
and then cleaned up.

## Commits

Unless the user explicitly says otherwise, push commits after creating them.

## Epistaxis

When reading Epistaxis, if recorded state conflicts with what you observe in
the code, repo, or thread, flag the mismatch before proceeding. Multiple
sessions may write concurrently; merge your changes without overwriting
entries you did not author.
