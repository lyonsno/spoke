"""Tests for the production single-instance guard in the entry point."""

from __future__ import annotations

import signal
import errno
import types
from unittest.mock import MagicMock, patch


class TestSingleInstanceGuard:
    """Exercise spoke.__main__._acquire_instance_lock directly."""

    def test_first_instance_acquires_and_writes_pid(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        with patch.object(main_module, "_LOCK_PATH", str(lock_path)):
            with patch.object(main_module.os, "getpid", return_value=12345):
                main_module._acquire_instance_lock()

        assert lock_path.read_text(encoding="utf-8") == "12345"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_replacement_waits_for_actual_release_after_old_retry_window(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("111", encoding="utf-8")
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
            flock=MagicMock(side_effect=[OSError("busy")] * 15 + [None]))
        with patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                patch.object(main_module, "_LOCK_PATH", str(lock_path)), \
                patch.object(main_module.os, "getpid", return_value=444), \
                patch.object(main_module.os, "kill") as killed, \
                patch.object(main_module, "_is_process_alive", return_value=False), \
                patch("time.sleep"):
            main_module._acquire_instance_lock()
        assert killed.call_args_list.count(((111, signal.SIGKILL),)) == 1
        assert lock_path.read_text() == "444"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_waiter_leaves_a_new_replacement_owner_alone(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("111", encoding="utf-8")
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
            flock=MagicMock(side_effect=OSError("busy")))
        with patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                patch.object(main_module, "_LOCK_PATH", str(lock_path)), \
                patch.object(main_module.os, "getpid", return_value=444), \
                patch.object(main_module.os, "kill") as killed, \
                patch("time.sleep", side_effect=lambda _: lock_path.write_text("222")):
            import pytest
            with pytest.raises(SystemExit) as exited:
                main_module._acquire_instance_lock()
        assert exited.value.code == 0
        assert killed.call_args_list == [((111, signal.SIGTERM),)]
        assert lock_path.read_text() == "222"

    def test_invalid_owner_pid_never_signals_a_process_group(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("0")
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
            flock=MagicMock(side_effect=[OSError("busy"), None]))
        with patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                patch.object(main_module, "_LOCK_PATH", str(lock_path)), \
                patch.object(main_module.os, "getpid", return_value=444), \
                patch.object(main_module.os, "kill") as killed, patch("time.sleep"):
            main_module._acquire_instance_lock()
        killed.assert_not_called()
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_flock_error_is_not_treated_as_a_live_predecessor(self, main_module, tmp_path):
        import pytest
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("111")
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
            flock=MagicMock(side_effect=[OSError(errno.EBADF, "invalid descriptor"),
                                         RuntimeError("unexpected retry")]))
        with patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                patch.object(main_module, "_LOCK_PATH", str(lock_path)), \
                patch.object(main_module.os, "kill") as killed, patch("time.sleep"):
            with pytest.raises(OSError, match="invalid descriptor"):
                main_module._acquire_instance_lock()
        killed.assert_not_called()

    def test_second_instance_kills_first_and_takes_lock(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        first_lock = open(lock_path, "a+", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(first_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            first_lock.seek(0)
            first_lock.truncate()
            first_lock.write("111")
            first_lock.flush()
            kill_calls = []

            def fake_kill(pid, sig_num):
                kill_calls.append((pid, sig_num))
                if sig_num == signal.SIGTERM:
                    fcntl.flock(first_lock, fcntl.LOCK_UN)
                    return
                if sig_num == 0:
                    raise ProcessLookupError()

            with patch.object(main_module, "_LOCK_PATH", str(lock_path)):
                with patch.object(main_module.os, "getpid", return_value=222):
                    with patch.object(main_module.os, "kill", side_effect=fake_kill):
                        with patch("time.sleep"):
                            main_module._acquire_instance_lock()
        finally:
            first_lock.close()

        assert (111, signal.SIGTERM) in kill_calls
        assert lock_path.read_text(encoding="utf-8") == "222"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_corrupt_pid_in_lock_file(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("not-a-pid", encoding="utf-8")

        with patch.object(main_module, "_LOCK_PATH", str(lock_path)):
            with patch.object(main_module.os, "getpid", return_value=333):
                main_module._acquire_instance_lock()

        assert lock_path.read_text(encoding="utf-8") == "333"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_sigkill_escalation_for_stuck_process(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("111", encoding="utf-8")
        fake_fcntl = types.SimpleNamespace(
            LOCK_EX=1,
            LOCK_NB=2,
            flock=MagicMock(side_effect=[OSError("busy")] * 6 + [None]),
        )
        kill_calls = []

        def fake_kill(pid, sig_num):
            kill_calls.append((pid, sig_num))
            if sig_num == 0:
                raise ProcessLookupError()

        with patch.dict("sys.modules", {"fcntl": fake_fcntl}):
            with patch.object(main_module, "_LOCK_PATH", str(lock_path)):
                with patch.object(main_module.os, "getpid", return_value=444):
                    with patch.object(main_module.os, "kill", side_effect=fake_kill):
                        with patch.object(main_module, "_is_process_alive", return_value=False):
                            with patch("time.sleep"):
                                main_module._acquire_instance_lock()

        assert (111, signal.SIGTERM) in kill_calls
        assert (111, signal.SIGKILL) in kill_calls
        assert lock_path.read_text(encoding="utf-8") == "444"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")

    def test_stale_pid_no_process(self, main_module, tmp_path):
        lock_path = tmp_path / ".spoke.lock"
        lock_path.write_text("99999999", encoding="utf-8")

        with patch.object(main_module, "_LOCK_PATH", str(lock_path)):
            with patch.object(main_module.os, "getpid", return_value=555):
                main_module._acquire_instance_lock()

        assert lock_path.read_text(encoding="utf-8") == "555"
        main_module._acquire_instance_lock._lock_file.close()
        delattr(main_module._acquire_instance_lock, "_lock_file")
