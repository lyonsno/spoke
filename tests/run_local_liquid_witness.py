"""Keep the permitted native witness host inside a Greenroom external lease."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "repo-root", "greenroom-root", "queue-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--startup-seconds", type=float, default=15)
    parser.add_argument("--native-capture-library", type=Path)
    parser.add_argument("--render-on-capture", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    route = ("typed Swift/SCK capture feeding existing Python House renderer" if args.native_capture_library else
             "native Python/AppKit/SCK/Metal full vs localized House")
    receipt = {"passed": False, "phase": "permission", "pid": os.getpid(),
               "repo_root": str(args.repo_root), "queue": str(args.queue_dir),
               "route": route,
               "lease_id": args.lease_id}
    path = args.out / "lease-run.json"

    def save():
        path.write_text(json.dumps(receipt, indent=2) + "\n")

    lease, queue = None, None
    save()
    try:
        import Quartz
        receipt["screen_recording_access"] = bool(Quartz.CGPreflightScreenCaptureAccess())
        if not receipt["screen_recording_access"]:
            raise RuntimeError("Effective host lacks Screen Recording permission")
        sys.path.insert(0, str(args.greenroom_root))
        from gpu_queue.queue import GPUQueue
        queue = GPUQueue(args.queue_dir)
        receipt["claim"] = "Correctness observation under natural load, not a controlled performance comparison"
        if any((queue.queue_dir / "pending").iterdir()) or any((queue.queue_dir / "running").iterdir()):
            raise RuntimeError("Existing queued work has priority; no lease claimed")
        lease = queue.claim_lease(owner="warpstorm-pit-boss", agent_id="warpstorm-pit-boss",
            repo_root=str(args.repo_root), pid=os.getpid(), effective_route=receipt["route"],
            backend="SCK/Metal", device="M4 Max/WindowServer", profile="localized House experiment",
            supports_checkpoints=False, interruptible=True, lease_id=args.lease_id)
        command = [sys.executable, str(args.repo_root / "tests/native_local_liquid_witness.py"),
                   "--repo-root", str(args.repo_root), "--out", str(args.out),
                   "--startup-seconds", str(args.startup_seconds)]
        if args.native_capture_library:
            command.extend(["--native-capture-library", str(args.native_capture_library)])
        if args.render_on_capture:
            command.append("--render-on-capture")
        receipt.update(phase="execution", command=command,
            witness_sha256=hashlib.sha256((args.repo_root / "tests/native_local_liquid_witness.py").read_bytes()).hexdigest())
        save()
        with (args.out / "stdout.log").open("w") as stdout, (args.out / "stderr.log").open("w") as stderr:
            receipt["exit_code"] = subprocess.run(command, cwd=args.repo_root, stdout=stdout, stderr=stderr).returncode
        result = json.loads((args.out / "report.json").read_text())
        if receipt["exit_code"] or not result.get("passed"):
            raise RuntimeError("Native experiment failed; retained report identifies the last phase")
        receipt.update(phase="complete", passed=True)
    except BaseException:
        receipt["error"] = traceback.format_exc()
    finally:
        if lease is not None:
            queue.release_lease(lease.lease_id, released_by="warpstorm-pit-boss",
                                reason="Owned native compositor process terminal")
            receipt["lease_released"] = True
        save()
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
