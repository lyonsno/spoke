"""PyInstaller entry point — runs dictate as a package."""
import sys
import os
import fcntl

# Log to file so we can debug the bundled app
log_path = os.path.expanduser("~/Library/Logs/Dictate.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = sys.stderr = open(log_path, "a")

# Single-instance guard — prevent multiple copies from running
_lock_path = os.path.expanduser("~/Library/Logs/.dictate.lock")
_lock_file = open(_lock_path, "w")
try:
    fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    # Another instance is already running
    print("Dictate is already running. Exiting.", file=sys.stderr)
    sys.exit(0)

from dictate.__main__ import main

if __name__ == "__main__":
    main()
