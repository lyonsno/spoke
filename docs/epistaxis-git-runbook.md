# Epistaxis Git Runbook

Reference for the spoke assistant when performing git operations in the
Epistaxis repo (`~/dev/epistaxis`).

## Repo layout

| Path | Role |
|------|------|
| `~/dev/epistaxis` | Main checkout. Always on `main`. Inert sync surface — do not do feature work here. |
| `~/dev/epistaxis-wt` | Active worktree. Feature branches live here. |

The main checkout and the worktree share the same `.git` directory, so
they see each other's branches and refs. Git will refuse to checkout a
branch in the worktree if it is already checked out in the main
checkout (and vice versa). This is normal — do not fight it.

## Divergence on main is normal

The main checkout frequently shows "have diverged, N and M different
commits" when comparing local main to origin/main. This is expected.
Multiple agents and automated systems (Panopticon, other lanes) push
to origin/main independently. The local main checkout does not
auto-pull.

**Do not treat divergence as a problem.** A `git pull --rebase` on the
main checkout will almost always resolve cleanly because the local
commits are typically already on origin (pushed by whichever lane
authored them). The rebase drops the duplicates and fast-forwards.

## Merging a feature branch to main

This is the standard flow for landing Epistaxis work from a worktree
branch onto main:

### Step 1: Push the branch (from the worktree)

```
cd ~/dev/epistaxis-wt
git push origin <branch-name>
```

If on a detached HEAD, create a branch first:
```
git checkout -b <branch-name>
git push -u origin <branch-name>
```

### Step 2: Sync main (from the main checkout)

```
cd ~/dev/epistaxis
```

If the working tree is dirty (staged or unstaged changes), commit or
stash them first:
```
git add -A && git commit -m "wip: in-flight state before sync"
# or: git stash
```

Then sync:
```
git pull --rebase origin main
```

This will fast-forward or cleanly rebase. If it conflicts (rare),
resolve and continue the rebase.

### Step 3: Merge the branch (from the main checkout)

```
git merge <branch-name>
git push origin main
```

### Step 4: Clean up

```
git branch -d <branch-name>
git push origin --delete <branch-name>
```

## Common mistakes

- **Trying to checkout main in the worktree.** Git won't allow this
  because main is checked out in `~/dev/epistaxis`. Operate from the
  main checkout instead.
- **Panicking about divergence.** A diverged main just means the local
  checkout hasn't synced recently. `git pull --rebase` fixes it.
- **Skipping the dirty-tree step.** The main checkout often has staged
  review artifacts or modified epistaxis.md from other lanes. Commit
  or stash before pulling.
- **Trying to merge from the worktree.** The merge onto main must
  happen from the main checkout where main is checked out.
