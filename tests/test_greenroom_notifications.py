import json
import os
import socket
from pathlib import Path
import uuid

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
    assert greenroom_url_for_notification({"kind": "greenroom-ping", "url": "https://evil.invalid"}) == GREENROOM_MONITOR_URL


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


def test_stale_socket_path_is_not_replaced_by_regular_file(tmp_path):
    path = tmp_path / "greenroom.sock"
    path.write_text("not a socket")
    server = GreenroomPingServer(lambda payload: None, socket_path=path)

    with pytest.raises(RuntimeError, match="not a socket"):
        server.start()
