# Claude Code Instructions

## After making changes

When a change is ready for smoke testing, run the build and install pipeline:

```sh
pkill -TERM -f "Spoke" 2>/dev/null
sleep 1
rm -f ~/Library/Logs/.spoke.lock
./scripts/build.sh --fast
rm -rf ~/Applications/Spoke.app
cp -r dist/Spoke.app ~/Applications/
open ~/Applications/Spoke.app
```

This kills any running instance, rebuilds incrementally, copies to Applications, and relaunches. The user will grant permissions if prompted.

For full clean builds (after dependency changes, spec file changes, or when --fast builds behave unexpectedly):

```sh
./scripts/build.sh
```

## Testing

Always run `uv run pytest -q` after code changes and before committing. All tests must pass.

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

## Epistaxis

When reading epistaxis, if any recorded state doesn't match what you observe in the code or thread, flag it before proceeding — even if the mismatch might just be stale rather than wrong. Multiple sessions may write to the same epistaxis file concurrently; merge your changes without overwriting entries you didn't write.

## Demo video

- `scripts/demo-convert.sh` converts screen recordings (.mov) to optimized MP4s. Run with `--help` for options.
- GitHub README only renders inline `<video>` with `user-attachments` URLs. Release download URLs and raw.githubusercontent URLs are silently stripped.
- The only way to get a `user-attachments` URL is drag-and-drop into a GitHub issue/PR comment in the browser. Wait for the "Uploading..." placeholder to resolve to an actual URL before submitting — if you submit while it still says uploading, the upload is lost.
- `gh` CLI cannot upload to `user-attachments`. Don't waste time trying.

## Commits

Use descriptive commit messages. Include `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` in all commits.
Unless the user explicitly says otherwise, push commits after creating them.
