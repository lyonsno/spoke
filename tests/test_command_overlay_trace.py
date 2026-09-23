import json

from spoke.command_overlay_trace import record_command_overlay_trace


def test_command_overlay_trace_writes_jsonl_when_path_is_set(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(path))
    monkeypatch.setenv("SPOKE_LAUNCH_ID", "launch-a")
    monkeypatch.setenv("SPOKE_LAUNCH_TARGET_ID", "smoke")

    record_command_overlay_trace("gesture.test", visible=True, ignored=None)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "gesture.test"
    assert payload["visible"] is True
    assert "ignored" not in payload
    assert isinstance(payload["pid"], int)
    assert payload["launch_id"] == "launch-a"
    assert payload["launch_target_id"] == "smoke"
    assert payload["source_root"].startswith("/")
    assert payload["source_root"].endswith("spoke-warpstorm-optical-outcome-witness-0923")
    assert isinstance(payload["source_revision"], str)
    assert isinstance(payload["source_dirty"], bool)


def test_command_overlay_trace_is_noop_without_path(monkeypatch, tmp_path):
    monkeypatch.delenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", raising=False)

    record_command_overlay_trace("gesture.test", path=str(tmp_path / "unused.jsonl"))

    assert not (tmp_path / "unused.jsonl").exists()
