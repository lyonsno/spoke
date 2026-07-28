from __future__ import annotations

import os


def test_reopens_replaced_log_inode_for_both_streams(tmp_path) -> None:
    from spoke.log_reattach import reattach_replaced_log

    visible_path = tmp_path / "spoke.log"
    visible_path.write_text("new\n")
    detached_path = tmp_path / "detached.log"
    detached_path.write_text("old\n")

    stdout_fd = os.open(detached_path, os.O_WRONLY | os.O_APPEND)
    stderr_fd = os.dup(stdout_fd)
    try:
        assert reattach_replaced_log(
            visible_path, stdout_fd=stdout_fd, stderr_fd=stderr_fd
        )
        os.write(stdout_fd, b"stdout-visible\n")
        os.write(stderr_fd, b"stderr-visible\n")
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)

    visible = visible_path.read_text()
    assert "stdout-visible" in visible
    assert "stderr-visible" in visible
    assert detached_path.read_text() == "old\n"


def test_same_inode_is_a_noop(tmp_path) -> None:
    from spoke.log_reattach import reattach_replaced_log

    path = tmp_path / "spoke.log"
    path.write_text("")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        assert not reattach_replaced_log(path, stdout_fd=fd, stderr_fd=fd)
    finally:
        os.close(fd)
