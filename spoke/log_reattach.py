"""Repair stdout/stderr when a launcher log path is replaced underneath Spoke."""

from __future__ import annotations

import os
from pathlib import Path


def reattach_replaced_log(
    path: Path,
    *,
    stdout_fd: int = 1,
    stderr_fd: int = 2,
) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    changed = False
    try:
        target_stat = os.fstat(target_fd)
        for stream_fd in dict.fromkeys((stdout_fd, stderr_fd)):
            try:
                stream_stat = os.fstat(stream_fd)
            except OSError:
                stream_stat = None
            if stream_stat is not None and (
                stream_stat.st_dev,
                stream_stat.st_ino,
            ) == (target_stat.st_dev, target_stat.st_ino):
                continue
            os.dup2(target_fd, stream_fd)
            changed = True
        if changed:
            os.write(
                stderr_fd,
                (
                    "Spoke log stream reattached after path inode replacement: "
                    f"{path}\n"
                ).encode("utf-8", errors="replace"),
            )
    finally:
        os.close(target_fd)
    return changed
