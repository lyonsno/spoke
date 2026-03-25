# Claude Code Instructions

## After making changes

When a change is ready for smoke testing, run the build and install pipeline:

```sh
pkill -TERM -f "DontType" 2>/dev/null
sleep 1
rm -f ~/Library/Logs/.donttype.lock
./scripts/build.sh --fast
rm -rf ~/Applications/DontType.app
cp -r dist/DontType.app ~/Applications/
open ~/Applications/DontType.app
```

This kills any running instance, rebuilds incrementally, copies to Applications, and relaunches. The user will grant permissions if prompted.

For full clean builds (after dependency changes, spec file changes, or when --fast builds behave unexpectedly):

```sh
./scripts/build.sh
```

## Testing

Always run `uv run pytest -q` after code changes and before committing. All tests must pass.

## Permissions

- Every rebuild changes the ad-hoc signature, which may invalidate macOS TCC permissions (Accessibility, Microphone).
- The app has a graceful permissions flow that retries automatically — the user just needs to grant when prompted.
- Do NOT run `tccutil reset` in scripts — it causes more problems than it solves.
- Use `pkill -TERM` (not `-9`) to kill the app so the SIGTERM handler can cleanly uninstall the CGEventTap.

## Epistaxis

When reading epistaxis, if any recorded state doesn't match what you observe in the code or thread, flag it before proceeding — even if the mismatch might just be stale rather than wrong. Multiple sessions may write to the same epistaxis file concurrently; merge your changes without overwriting entries you didn't write.

## Commits

Use descriptive commit messages. Include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in all commits.
