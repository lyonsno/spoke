import json
import errno
import os
import socket
from pathlib import Path
import uuid
import threading

import pytest

from spoke.greenroom_notifications import (
    GREENROOM_MONITOR_URL,
    GreenroomPingServer,
    greenroom_url_for_notification,
)
from spoke.greenroom_ping import main


def _socket_path() -> Path:
    return (
        Path("/tmp")
        / f"spoke-gr-test-{os.getuid()}-{uuid.uuid4().hex}"
        / "notify.sock"
    )


def test_local_ping_socket_dispatches_valid_event_and_acknowledges():
    socket_path = _socket_path()
    received = []
    server = GreenroomPingServer(
        received.append,
        socket_path=socket_path,
    )
    server.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(server.socket_path))
            client.sendall(
                json.dumps(
                    {
                        "schema": "spoke.greenroom-ping.v1",
                        "job_id": "job-42",
                        "agent_id": "handy-fire-man",
                        "reason": "Capture ready",
                    }
                ).encode()
                + b"\n"
            )
            response = json.loads(client.makefile("rb").readline())

        assert response == {"status": "accepted"}
        assert received == [
            {
                "schema": "spoke.greenroom-ping.v1",
                "job_id": "job-42",
                "agent_id": "handy-fire-man",
                "reason": "Capture ready",
            }
        ]
    finally:
        server.close()
        socket_path.parent.rmdir()


def test_local_ping_socket_rejects_arbitrary_open_url():
    socket_path = _socket_path()
    received = []
    server = GreenroomPingServer(received.append, socket_path=socket_path)
    server.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(server.socket_path))
            client.sendall(
                b'{"schema":"spoke.greenroom-ping.v1","job_id":"j","agent_id":"a",'
                b'"reason":"x","url":"https://attacker.invalid"}\n'
            )
            response = json.loads(client.makefile("rb").readline())

        assert response["status"] == "rejected"
        assert received == []
    finally:
        server.close()
        socket_path.parent.rmdir()


def test_only_greenroom_notifications_open_the_fixed_monitor_url():
    assert greenroom_url_for_notification({"kind": "greenroom-ping"}) == GREENROOM_MONITOR_URL
    assert greenroom_url_for_notification({"kind": "other"}) is None
    assert greenroom_url_for_notification({"kind": "greenroom-ping", "url": "https://evil.invalid"}) is None


def test_pyobjc_dictionary_notification_info_opens_monitor():
    class NotificationInfo:
        def objectForKey_(self, key):
            return "greenroom-ping" if key == "kind" else None

    assert greenroom_url_for_notification(NotificationInfo()) == GREENROOM_MONITOR_URL


def test_existing_ping_command_reaches_spoke_listener_without_duplicate_fallback(monkeypatch):
    socket_path = _socket_path()
    received = []
    server = GreenroomPingServer(received.append, socket_path=socket_path)
    server.start()
    monkeypatch.setattr("spoke.greenroom_ping.GREENROOM_PING_SOCKET", socket_path)
    monkeypatch.setattr(
        "spoke.greenroom_ping.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("successful local delivery must not send a second banner")
        ),
    )
    try:
        assert main(["--job-id", "job-7", "--agent-id", "sammy", "--reason", "Inspect this"] ) == 0
        assert received == [
            {
                "schema": "spoke.greenroom-ping.v1",
                "job_id": "job-7",
                "agent_id": "sammy",
                "reason": "Inspect this",
            }
        ]
    finally:
        server.close()
        socket_path.parent.rmdir()


def test_second_listener_cannot_replace_live_spoke_listener():
    socket_path = _socket_path()
    first = GreenroomPingServer(lambda payload: None, socket_path=socket_path)
    second = GreenroomPingServer(lambda payload: None, socket_path=socket_path)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another Spoke Greenroom ping listener"):
            second.start()
        assert socket_path.exists()
    finally:
        first.close()
        socket_path.parent.rmdir()


def test_live_listener_probe_handles_server_closing_after_reply(monkeypatch):
    class RepliedProbe:
        def setblocking(self, value):
            assert value is False

        def connect(self, path):
            pass

        def sendall(self, data):
            assert data == b"\n"

        def shutdown(self, how):
            raise OSError(errno.ENOTCONN, "Socket is not connected")

        def recv(self, size):
            return b'{"status":"rejected"}\n'

        def close(self):
            pass

    socket_path = _socket_path()
    first = GreenroomPingServer(lambda payload: None, socket_path=socket_path)
    first.start()
    try:
        with monkeypatch.context() as patch:
            patch.setattr("spoke.greenroom_notifications.socket.socket", lambda *args: RepliedProbe())
            with pytest.raises(RuntimeError, match="another Spoke Greenroom ping listener"):
                first._remove_stale_socket()
        assert socket_path.exists()
    finally:
        first.close()
        socket_path.parent.rmdir()


def test_stale_socket_path_is_not_replaced_by_regular_file(tmp_path):
    path = tmp_path / "greenroom.sock"
    path.write_text("not a socket")
    server = GreenroomPingServer(lambda payload: None, socket_path=path)

    with pytest.raises(RuntimeError, match="not a socket"):
        server.start()


def test_occupied_listener_does_not_wait_for_a_protocol_reply():
    path = _socket_path()
    path.parent.mkdir()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen()
    server = GreenroomPingServer(lambda _: None, socket_path=path)
    finished = threading.Event()
    errors = []
    def start():
        try:
            server.start()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()
    worker = threading.Thread(target=start)
    worker.start()
    peer, _ = listener.accept()
    try:
        assert finished.wait(0.2), 'occupied optional listener must not block app startup on a reply'
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert path.is_socket()
    finally:
        peer.close()
        listener.close()
        worker.join()
        server.close()
        path.unlink()
        path.parent.rmdir()


def test_malformed_url_gets_an_explicit_wire_rejection():
    path = _socket_path()
    received = []
    server = GreenroomPingServer(received.append, socket_path=path)
    server.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(path))
            client.sendall(json.dumps({'schema': 'spoke.greenroom-ping.v1',
                'job_id': 'job', 'agent_id': 'owner', 'reason': '', 'url': 'http://[broken/'}).encode() + b'\n')
            assert client.makefile('rb').readline() == b'{"status": "rejected"}\n'
        assert received == []
    finally:
        server.close()
        path.parent.rmdir()


def test_closed_occupancy_probe_is_quiet_and_does_not_dispatch(monkeypatch):
    from spoke import greenroom_notifications as notifications
    completed = threading.Event()
    errors = []
    received = []
    class ObservedServer(notifications._ThreadedUnixServer):
        def finish_request(self, request, client_address):
            try:
                super().finish_request(request, client_address)
            finally:
                completed.set()

        def handle_error(self, request, client_address):
            errors.append(client_address)
    monkeypatch.setattr(notifications, '_ThreadedUnixServer', ObservedServer)
    path = _socket_path()
    server = notifications.GreenroomPingServer(received.append, socket_path=path)
    server.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(path))
        assert completed.wait(0.2)
        server.close()
        assert not errors and not received
    finally:
        server.close()
        path.parent.rmdir()
