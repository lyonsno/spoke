import json
import threading
import queue
import pytest
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
    assert Path(payload["source_root"]).resolve() == Path(__file__).resolve().parents[1]
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

    trace.enqueue_command_overlay_trace("optical.witness.present", generation=7)

    event, details = capture_queue.item
    assert event == "optical.witness.present"
    assert details["event_time_unix_seconds"] > 0
    assert details["event_thread_id"] == threading.get_ident()
    assert details["trace_path"] == str(path)


def test_producer_does_not_format_time_or_create_foreign_thread_objects(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    class CaptureQueue:
        def put(self, item):
            self.item = item

    captured = CaptureQueue()
    monkeypatch.setattr(trace, "_TRACE_QUEUE", captured)
    monkeypatch.setenv("SPOKE_COMMAND_OVERLAY_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr(trace, "datetime", None)
    monkeypatch.setattr(trace.threading, "current_thread", lambda: pytest.fail("producer created a thread object"))
    trace.enqueue_command_overlay_trace("native.callback")
    assert captured.item[1]["event_time_unix_seconds"] > 0


def test_writer_drains_available_records_with_one_file_open(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    path = tmp_path / "trace.jsonl"
    items = [("frame", {"trace_path": str(path), "frame": i}) for i in range(3)]

    class PendingQueue:
        completed = 0

        def get(self):
            if not items:
                raise StopIteration
            return items.pop(0)

        def get_nowait(self):
            if not items:
                raise queue.Empty
            return items.pop(0)

        def qsize(self):
            return len(items)

        def task_done(self):
            self.completed += 1

    pending = PendingQueue()
    original = Path.open
    opened = []

    def tracked_open(self, *args, **kwargs):
        if self == path:
            opened.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(trace, "_TRACE_QUEUE", pending)
    monkeypatch.setattr(Path, "open", tracked_open)
    with pytest.raises(StopIteration):
        trace._trace_writer()
    assert len(opened) == 1
    assert pending.completed == 3
    assert [json.loads(line)["frame"] for line in path.read_text().splitlines()] == [0, 1, 2]


def test_writer_preserves_event_time_and_explicit_timestamp(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(trace, "_source_identity", lambda: {})
    trace._write_command_overlay_trace("first", {"trace_path": str(path), "event_time_unix_seconds": 1.25, "event_thread_id": 42})
    trace._write_command_overlay_trace("second", {"trace_path": str(path), "timestamp": "original", "event_time_unix_seconds": 1.25})
    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    from datetime import datetime
    assert datetime.fromisoformat(first["timestamp"]).timestamp() == 1.25
    assert first["thread"] == "thread-42"
    assert first["event_thread_id"] == 42
    assert second["timestamp"] == "original"


def test_buffered_close_failure_is_receipted_before_queue_completion(monkeypatch, tmp_path):
    import spoke.command_overlay_trace as trace

    path = tmp_path / "trace.jsonl"
    items = [("frame", {"trace_path": str(path), "trace_sequence": 9})]

    class PendingQueue:
        completed = False

        def get(self):
            if not items:
                raise StopIteration
            return items.pop(0)

        def qsize(self):
            return 0

        def task_done(self):
            assert Path(f"{path}.failures.jsonl").exists()
            self.completed = True

    class BrokenFile:
        def __enter__(self):
            return self

        def write(self, line):
            pass

        def __exit__(self, *args):
            raise OSError("buffered flush failed")

    pending = PendingQueue()
    original = Path.open
    monkeypatch.setattr(trace, "_TRACE_QUEUE", pending)
    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: BrokenFile() if self == path else original(self, *a, **kw))
    with pytest.raises(StopIteration):
        trace._trace_writer()
    assert pending.completed
    failure = json.loads(Path(f"{path}.failures.jsonl").read_text())
    assert failure["trace_sequence"] == 9
    assert failure["error"] == "buffered flush failed"
