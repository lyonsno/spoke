import json
import threading
from pathlib import Path

from spoke.command_overlay_trace import flush_command_overlay_trace, record_command_overlay_trace


def test_command_overlay_trace_writes_jsonl_when_path_is_set(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(path))
    monkeypatch.setenv("SPOKE_LAUNCH_ID", "launch-a")
    monkeypatch.setenv("SPOKE_LAUNCH_TARGET_ID", "smoke")

    record_command_overlay_trace("gesture.test", visible=True, ignored=None)
    flush_command_overlay_trace()

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


def test_enqueue_does_not_defer_a_trace_when_disabled(monkeypatch):
    import spoke.command_overlay_trace as trace

    class CaptureQueue:
        item = None

        def put(self, item):
            self.item = item

    capture_queue = CaptureQueue()
    monkeypatch.setattr(trace, "_TRACE_QUEUE", capture_queue)
    monkeypatch.delenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", raising=False)

    trace.enqueue_command_overlay_trace("disabled.event")

    assert capture_queue.item is None


def test_failed_trace_write_leaves_a_sidecar_loss_receipt(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(path))
    monkeypatch.setattr(
        trace,
        "_write_command_overlay_trace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    trace.enqueue_command_overlay_trace("optical.witness.present", generation=9)
    trace.flush_command_overlay_trace()

    failure = json.loads(Path(f"{path}.failures.jsonl").read_text(encoding="utf-8"))
    assert failure["event"] == "trace.write.failed"
    assert failure["source_event"] == "optical.witness.present"
    assert failure["error_type"] == "OSError"


def test_enqueued_trace_captures_event_time_thread_and_destination(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    class CaptureQueue:
        item = None

        def put(self, item):
            self.item = item

    capture_queue = CaptureQueue()
    monkeypatch.setattr(trace, "_TRACE_QUEUE", capture_queue)
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(path))
    caller = threading.current_thread().name

    trace.enqueue_command_overlay_trace("optical.witness.present", generation=7)

    event, details = capture_queue.item
    assert event == "optical.witness.present"
    assert details["timestamp"]
    assert details["event_thread"] == caller
    assert details["trace_path"] == str(path)
