from subprocess import CompletedProcess

from spoke.greenroom_ping import main


def test_greenroom_ping_passes_notification_text_as_arguments(monkeypatch):
    observed = {}

    def fake_run(command, *, capture_output, check, text):
        observed["command"] = command
        observed["capture_output"] = capture_output
        observed["check"] = check
        observed["text"] = text
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("spoke.greenroom_ping.subprocess.run", fake_run)

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


def test_greenroom_ping_reports_notification_failure(monkeypatch, capsys):
    def fake_run(command, **kwargs):
        return CompletedProcess(command, 1, stdout="", stderr="User denied notification")

    monkeypatch.setattr("spoke.greenroom_ping.subprocess.run", fake_run)

    result = main(["--job-id", "job-42", "--agent-id", "diaulos"])

    assert result == 1
    stderr = capsys.readouterr().err
    assert "osascript exited 1" in stderr
    assert "User denied notification" in stderr
