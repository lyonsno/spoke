"""One-shot notifier for Greenroom's operator-needed transition.

Every invocation sends a notification. Greenroom owns event latching and must
call this only when a smoke first needs the operator, not from a status-poll
loop.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


_TITLE = "GPU Greenroom smoke needs you"
_NOTIFICATION_SCRIPT = "display notification (item 2 of argv) with title (item 1 of argv)"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spoke-greenroom-ping",
        description="Send a macOS notification when a Greenroom smoke needs the operator.",
        epilog=(
            "Every invocation sends a notification. Greenroom must call once "
            "on the operator-needed transition, not on status polls."
        ),
    )
    parser.add_argument("--job-id", required=True, help="Greenroom job identifier")
    parser.add_argument("--agent-id", required=True, help="Greenroom request agent_id")
    parser.add_argument("--reason", default="", help="Short operator-needed reason")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    body = f"Job {args.job_id} | {args.agent_id}"
    if args.reason:
        body += f" - {args.reason}"

    command = [
        "/usr/bin/osascript",
        "-e",
        "on run argv",
        "-e",
        _NOTIFICATION_SCRIPT,
        "-e",
        "end run",
        _TITLE,
        body,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as exc:
        print(f"spoke-greenroom-ping: could not run osascript: {exc}", file=sys.stderr)
        return 1

    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        print(f"spoke-greenroom-ping: osascript exited {result.returncode}{suffix}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
