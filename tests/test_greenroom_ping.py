from subprocess import CompletedProcess
import errno
import io

from spoke.greenroom_ping import main


def test_listener_can_close_after_accepting_newline_framed_ping(monkeypatch):
    class AcceptedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, path):
            pass

        def sendall(self, data):
            assert data.endswith(b"\n")

        def shutdown(self, how):
            raise OSError(errno.ENOTCONN, "Socket is not connected")

        def makefile(self, mode):
            return io.BytesIO(b'{"status": "accepted"}\n')

    monkeypatch.setattr("spoke.greenroom_ping.socket.socket", lambda *args: AcceptedSocket())
    assert main(["--job-id", "job-fast", "--agent-id", "handy-handy-man"]) == 0


def test_greenroom_ping_passes_notification_text_as_arguments(monkeypatch, capsys):
    observed = {}

    def no_spoke_listener(payload):
        raise FileNotFoundError("Spoke notification socket is absent")

    def fake_run(command, *, capture_output, check, text):
        observed["command"] = command
        observed["capture_output"] = capture_output
        observed["check"] = check
        observed["text"] = text
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("spoke.greenroom_ping.subprocess.run", fake_run)
    monkeypatch.setattr("spoke.greenroom_ping._send_to_spoke", no_spoke_listener)

    result = main(
        [
            "--job-id",
            "job-42",
            "--agent-id",
            "handy-fire-man",
            "--reason",
            'Smoke is ready: say "go"; then inspect $HOME',
        ]
    )

    command = observed["command"]
    assert result == 0
    assert command[0] == "/usr/bin/osascript"
    assert "display notification (item 2 of argv) with title (item 1 of argv)" in command
    assert command[-2:] == [
        "GPU Greenroom smoke needs you",
        'Job job-42 | handy-fire-man - Smoke is ready: say "go"; then inspect $HOME',
    ]
    assert observed["capture_output"] is True
    assert observed["check"] is False
    assert observed["text"] is True
    assert "non-clickable notification fallback" in capsys.readouterr().err


def test_greenroom_ping_reports_notification_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "spoke.greenroom_ping._send_to_spoke",
        lambda payload: (_ for _ in ()).throw(FileNotFoundError("Spoke is not running")),
    )

    def fake_run(command, **kwargs):
        return CompletedProcess(command, 1, stdout="", stderr="User denied notification")

    monkeypatch.setattr("spoke.greenroom_ping.subprocess.run", fake_run)

    result = main(["--job-id", "job-42", "--agent-id", "worker"])

    assert result == 1
    stderr = capsys.readouterr().err
    assert "osascript exited 1" in stderr
    assert "User denied notification" in stderr


def test_greenroom_ping_prefers_spoke_owned_clickable_notification(monkeypatch):
    observed = {}

    def send_to_spoke(payload):
        observed["payload"] = payload
        return {"status": "accepted"}

    def no_fallback(*args, **kwargs):
        raise AssertionError("osascript fallback must not duplicate a Spoke notification")

    monkeypatch.setattr("spoke.greenroom_ping._send_to_spoke", send_to_spoke)
    monkeypatch.setattr("spoke.greenroom_ping.subprocess.run", no_fallback)

    result = main(["--job-id", "job-42", "--agent-id", "handy-fire-man", "--reason", "Capture ready"])

    assert result == 0
    assert observed["payload"] == {
        "schema": "spoke.greenroom-ping.v1",
        "job_id": "job-42",
        "agent_id": "handy-fire-man",
        "reason": "Capture ready",
    }
