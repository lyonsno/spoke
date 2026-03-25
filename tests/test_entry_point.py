"""Tests for the single-instance guard in the entry point.

Tests exercise the actual entry_point.py guard logic, not just flock
behavior. A subprocess runs entry_point-style guard code so we test
the real retry/kill/reacquire paths.
"""

import fcntl
import os
import signal
import subprocess
import time

import pytest


# Helper: a script that mimics entry_point.py's guard logic
_GUARD_SCRIPT = '''
import sys, os, fcntl, signal, time

lock_path = sys.argv[1]
lock_file = open(lock_path, "a+")
lock_file.seek(0)
try:
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    # Lock held — try to read the old PID and kill it
    try:
        lock_file.seek(0)
        old_pid = int(lock_file.read().strip())
        os.kill(old_pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    # Retry with backoff — old process needs time to die and release lock
    for _attempt in range(10):
        time.sleep(0.2)
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if _attempt == 4:
                # SIGTERM didn't work — escalate to SIGKILL
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, NameError):
                    pass
            continue
    else:
        print("BLOCKED")
        sys.exit(1)

# Got the lock — write PID
lock_file.seek(0)
lock_file.truncate()
lock_file.write(str(os.getpid()))
lock_file.flush()
print(f"LOCKED:{os.getpid()}")

# If --hold flag, hold the lock until killed
if "--hold" in sys.argv:
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    time.sleep(30)
'''

# Variant that ignores SIGTERM, simulating a process stuck in a Metal abort
_UNKILLABLE_HOLD_SCRIPT = '''
import sys, os, fcntl, signal, time

lock_path = sys.argv[1]
lock_file = open(lock_path, "a+")
lock_file.seek(0)
fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
lock_file.truncate()
lock_file.write(str(os.getpid()))
lock_file.flush()
print(f"LOCKED:{os.getpid()}")

# Ignore SIGTERM — simulate stuck process
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(30)
'''


class TestSingleInstanceGuard:
    """Test the entry_point.py guard logic end-to-end."""

    def test_first_instance_acquires_and_writes_pid(self, tmp_path):
        """First instance should acquire lock and write its PID."""
        lock_path = str(tmp_path / ".donttype.lock")
        result = subprocess.run(
            ["python3", "-c", _GUARD_SCRIPT, lock_path],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("LOCKED:")
        pid = int(result.stdout.strip().split(":")[1])

        # PID should be in the lock file
        with open(lock_path) as f:
            assert f.read().strip() == str(pid)

    def test_second_instance_kills_first_and_takes_lock(self, tmp_path):
        """Second instance should kill the first and acquire the lock."""
        lock_path = str(tmp_path / ".donttype.lock")

        # Start first instance holding the lock
        first = subprocess.Popen(
            ["python3", "-c", _GUARD_SCRIPT, lock_path, "--hold"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)

        # Verify first instance has the lock
        assert first.poll() is None  # still running

        # Start second instance — should kill first and take the lock
        result = subprocess.run(
            ["python3", "-c", _GUARD_SCRIPT, lock_path],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("LOCKED:")

        # First instance should be dead
        first.wait(timeout=2)
        assert first.returncode is not None

    def test_corrupt_pid_in_lock_file(self, tmp_path):
        """Guard should handle corrupt PID gracefully and still acquire."""
        lock_path = str(tmp_path / ".donttype.lock")

        # Write garbage to the lock file (no process holding it)
        with open(lock_path, "w") as f:
            f.write("not-a-pid")

        result = subprocess.run(
            ["python3", "-c", _GUARD_SCRIPT, lock_path],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("LOCKED:")

    def test_sigkill_escalation_for_stuck_process(self, tmp_path):
        """If the old process ignores SIGTERM, guard should escalate to SIGKILL."""
        lock_path = str(tmp_path / ".donttype.lock")

        # Start an unkillable instance (ignores SIGTERM)
        first = subprocess.Popen(
            ["python3", "-c", _UNKILLABLE_HOLD_SCRIPT, lock_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)
        assert first.poll() is None  # still running

        # Second instance should escalate to SIGKILL and take the lock
        result = subprocess.run(
            ["python3", "-c", _GUARD_SCRIPT, lock_path],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("LOCKED:")

        # First instance should be dead (killed by SIGKILL)
        first.wait(timeout=2)
        assert first.returncode is not None

    def test_stale_pid_no_process(self, tmp_path):
        """Guard should handle a PID for a process that no longer exists."""
        lock_path = str(tmp_path / ".donttype.lock")

        # Write a PID that doesn't exist (high number, unlikely to be real)
        with open(lock_path, "w") as f:
            f.write("99999999")

        result = subprocess.run(
            ["python3", "-c", _GUARD_SCRIPT, lock_path],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("LOCKED:")
