"""Local delivery and presentation for Greenroom operator-needed pings."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import socket
import socketserver
import stat
import threading
from collections.abc import Callable, Mapping
from typing import Any


GREENROOM_MONITOR_URL = "http://127.0.0.1:8766/"
GREENROOM_PING_SCHEMA = "spoke.greenroom-ping.v1"
GREENROOM_PING_SOCKET = (
    Path.home() / "Library" / "Caches" / "Spoke" / "greenroom-ping" / "notify.sock"
)


def valid_greenroom_ping(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if set(payload) != {"schema", "job_id", "agent_id", "reason"}:
        return False
    return (
        payload.get("schema") == GREENROOM_PING_SCHEMA
        and isinstance(payload.get("job_id"), str)
        and bool(payload["job_id"].strip())
        and isinstance(payload.get("agent_id"), str)
        and bool(payload["agent_id"].strip())
        and isinstance(payload.get("reason"), str)
    )


def greenroom_url_for_notification(user_info: Any) -> str | None:
    if isinstance(user_info, Mapping):
        kind = user_info.get("kind")
    else:
        object_for_key = getattr(user_info, "objectForKey_", None)
        kind = object_for_key("kind") if object_for_key is not None else None
    if kind == "greenroom-ping":
        return GREENROOM_MONITOR_URL
    return None


class _PingHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            payload = json.loads(self.rfile.readline())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._respond("rejected")
            return
        if not valid_greenroom_ping(payload):
            self._respond("rejected")
            return
        try:
            self.server.callback(payload)  # type: ignore[attr-defined]
        except Exception:
            self._respond("failed")
            return
        self._respond("accepted")

    def _respond(self, status: str) -> None:
        self.wfile.write(json.dumps({"status": status}).encode("utf-8") + b"\n")


class _ThreadedUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, socket_path: str, callback: Callable[[dict[str, str]], None]):
        self.callback = callback
        super().__init__(socket_path, _PingHandler)


class GreenroomPingServer:
    """Serve a same-user local socket and dispatch accepted pings to Spoke."""

    def __init__(
        self,
        callback: Callable[[dict[str, str]], None],
        *,
        socket_path: Path = GREENROOM_PING_SOCKET,
    ) -> None:
        self.socket_path = Path(socket_path)
        self._callback = callback
        self._server: _ThreadedUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._socket_identity: tuple[int, int] | None = None

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Greenroom ping server is already started")
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_metadata = self.socket_path.parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode) or parent_metadata.st_uid != os.getuid():
            raise RuntimeError(f"Greenroom ping directory is not owned by this user: {self.socket_path.parent}")
        os.chmod(self.socket_path.parent, 0o700)
        self._remove_stale_socket()
        previous_umask = os.umask(0o177)
        try:
            server = _ThreadedUnixServer(str(self.socket_path), self._dispatch)
        finally:
            os.umask(previous_umask)
        os.chmod(self.socket_path, 0o600)
        metadata = self.socket_path.lstat()
        self._socket_identity = (metadata.st_dev, metadata.st_ino)
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="spoke-greenroom-ping",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if (metadata.st_dev, metadata.st_ino) == self._socket_identity and stat.S_ISSOCK(metadata.st_mode):
            self.socket_path.unlink()
        self._socket_identity = None

    def _dispatch(self, payload: dict[str, str]) -> None:
        self._callback(payload)

    def _remove_stale_socket(self) -> None:
        try:
            metadata = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"Greenroom ping path exists and is not a socket: {self.socket_path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                raise
            self.socket_path.unlink()
        else:
            probe.sendall(b"\n")
            probe.shutdown(socket.SHUT_WR)
            probe.recv(1024)
            raise RuntimeError(f"another Spoke Greenroom ping listener is active at {self.socket_path}")
        finally:
            probe.close()
