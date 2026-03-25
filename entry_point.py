"""PyInstaller entry point — runs dictate as a package."""
import sys
import os

# Log to file so we can debug the bundled app
log_path = os.path.expanduser("~/Library/Logs/Dictate.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = sys.stderr = open(log_path, "a")

from dictate.__main__ import main

if __name__ == "__main__":
    main()
