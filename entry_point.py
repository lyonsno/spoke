"""PyInstaller entry point — runs donttype as a package."""
import sys
import os
import fcntl
import signal
import time

# Log to file so we can debug the bundled app
log_path = os.path.expanduser("~/Library/Logs/DontType.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = sys.stderr = open(log_path, "a")

# Help MLX find its metallib inside the PyInstaller bundle
if getattr(sys, '_MEIPASS', None):
    bundle_dir = sys._MEIPASS
    # Check multiple possible locations
    for candidate in [
        os.path.join(bundle_dir, 'mlx.metallib'),
        os.path.join(bundle_dir, 'mlx', 'lib', 'mlx.metallib'),
    ]:
        if os.path.exists(candidate):
            os.environ['MLX_METAL_LIB_PATH'] = candidate
            print(f"Set MLX_METAL_LIB_PATH={candidate}", file=sys.stderr)
            break
    else:
        print(f"WARNING: mlx.metallib not found in {bundle_dir}", file=sys.stderr)

# Single-instance guard — prevent multiple copies from running.
# If an old instance is stuck (e.g., mid-inference crash), kill it
# and take the lock.
_lock_path = os.path.expanduser("~/Library/Logs/.donttype.lock")
_lock_file = open(_lock_path, "a+")
_lock_file.seek(0)
try:
    fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    # Lock held — try to read the old PID and kill it
    try:
        _lock_file.seek(0)
        old_pid = int(_lock_file.read().strip())
        print(f"Killing old instance (pid={old_pid})", file=sys.stderr)
        os.kill(old_pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pass

    # Retry with backoff — old process needs time to die and release lock
    for _attempt in range(10):
        time.sleep(0.2)
        try:
            fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if _attempt == 4:
                # SIGTERM didn't work (e.g. stuck in Metal SIGABRT) — escalate
                try:
                    os.kill(old_pid, signal.SIGKILL)
                    print(f"Escalated to SIGKILL (pid={old_pid})", file=sys.stderr)
                except (ProcessLookupError, PermissionError, NameError):
                    pass
            continue
    else:
        print("DontType is already running. Exiting.", file=sys.stderr)
        sys.exit(0)

# Write our PID so the next instance can kill us if needed
_lock_file.seek(0)
_lock_file.truncate()
_lock_file.write(str(os.getpid()))
_lock_file.flush()

from donttype.__main__ import main

if __name__ == "__main__":
    main()
