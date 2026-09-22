import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from uuid import uuid4

import pytest

from spoke.smoke_requests import DirectoryWatch, SmokeRequests, return_response


@pytest.fixture
def smoke_request(tmp_path):
    return {
        "id": str(uuid4()), "kind": "interactive-smoke",
        "source": {"diaulos": "handy-handy-man", "thread_id": str(uuid4()),
                   "repo_root": str(tmp_path)},
        "title": "Kiln comparison", "prompt": "Move the camera and compare A with B.",
        "url": "http://127.0.0.1:8156/kiln", "availability": "preparation-needed",
        "availability_note": "GPU access still needed.",
    }


def test_submit_is_idempotent_but_cannot_change_the_request(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    first = queue.submit(smoke_request)
    assert queue.submit(deepcopy(smoke_request)) == first
    changed = {**smoke_request, "prompt": "Different judgment"}
    with pytest.raises(ValueError, match="conflict"):
        queue.submit(changed)
    assert queue.get(smoke_request["id"])["request"] == smoke_request


@pytest.mark.parametrize("change", [
    {"kind": "job-completed"}, {"kind": "ordinary-question"},
    {"url": "file:///etc/passwd"}, {"url": "javascript:alert(1)"},
    {"url": "https://user:password@example.org"}, {"url": "http://"},
    {"id": "../../other"}, {"availability": "free-gpu"}, {"prompt": ""},
])
def test_only_explicit_interactive_smoke_contract_is_admitted(tmp_path, smoke_request, change):
    with pytest.raises(ValueError):
        SmokeRequests(tmp_path).submit({**smoke_request, **change})
    assert not list(tmp_path.glob("*.json"))


def test_later_and_open_do_not_answer_or_discard_request(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.act(smoke_request["id"], "later")
    queue.act(smoke_request["id"], "opened")
    row = SmokeRequests(tmp_path).get(smoke_request["id"])
    assert row["status"] == "pending"
    assert row["response"] is None
    assert row["later_at"] and row["opened_at"]
    assert queue.claim_notifications() == []


def test_notification_claim_survives_restart_and_concurrent_observers(tmp_path, smoke_request):
    SmokeRequests(tmp_path).submit(smoke_request)
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda _: SmokeRequests(tmp_path).claim_notifications(), range(2)))
    assert sum(len(rows) for rows in claims) == 1
    assert SmokeRequests(tmp_path).claim_notifications() == []
    row = SmokeRequests(tmp_path).get(smoke_request["id"])
    assert row["notification"]["state"] == "attempted"
    assert row["response"] is None


def test_withdrawal_retains_evidence_and_rejects_late_reply(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.act(smoke_request["id"], "withdraw", "Server retired")
    with pytest.raises(ValueError, match="withdrawn"):
        queue.reply(smoke_request["id"], "Looks good")
    assert queue.get(smoke_request["id"])["withdrawal_reason"] == "Server retired"
    assert queue.claim_notifications() == []


def test_withdrawal_after_saved_reply_blocks_a_new_return(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    queue.act(smoke_request["id"], "withdraw", "Server retired")
    with pytest.raises(ValueError, match="withdrawn"):
        queue.deliver(
            smoke_request["id"],
            lambda row: pytest.fail("withdrawn response must not be sent"),
            retry=True,
        )


def test_preparation_state_can_advance_once_with_request_digest_binding(
    tmp_path, smoke_request
):
    queue = SmokeRequests(tmp_path)
    original = queue.submit(smoke_request)
    with pytest.raises(ValueError, match="digest"):
        queue.set_availability(
            smoke_request["id"], "prepared", "Server is live.", "wrong-digest"
        )
    prepared = queue.set_availability(
        smoke_request["id"],
        "prepared",
        "Server is live.",
        original["request_digest"],
    )
    assert prepared["readiness"]["state"] == "prepared"
    assert prepared["readiness"]["note"] == "Server is live."
    persisted = SmokeRequests(tmp_path).get(smoke_request["id"])
    assert persisted["readiness"] == prepared["readiness"]
    with pytest.raises(ValueError, match="terminal"):
        queue.set_availability(
            smoke_request["id"],
            "unavailable",
            "Late stale update.",
            original["request_digest"],
        )


def test_exact_response_survives_failure_without_automatic_resend(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    answer = "  A has less detail.\nKeep B.\n"
    queue.reply(smoke_request["id"], answer)
    calls = []
    def failed(row):
        calls.append(row)
        raise OSError("broker unavailable")
    row = queue.deliver(smoke_request["id"], failed)
    assert row["response"]["text"] == answer
    assert row["delivery"]["state"] == "unconfirmed"
    assert "broker unavailable" in row["delivery"]["receipt"]["error"]
    queue.deliver(smoke_request["id"], failed)
    assert len(calls) == 1
    assert SmokeRequests(tmp_path).get(smoke_request["id"])["response"]["text"] == answer


def test_failed_reply_write_keeps_pending_request(tmp_path, smoke_request, monkeypatch):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    monkeypatch.setattr(queue, "_write", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        queue.reply(smoke_request["id"], "Keep B")
    assert SmokeRequests(tmp_path).get(smoke_request["id"])["response"] is None


def test_corrupt_item_does_not_hide_other_requests(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    corrupt = str(uuid4())
    (tmp_path / f"{corrupt}.json").write_text("{broken")
    rows, errors = queue.scan()
    assert [row["request"]["id"] for row in rows] == [smoke_request["id"]]
    assert len(errors) == 1 and corrupt in errors[0]


@pytest.mark.parametrize("change", [
    {"created_at": None}, {"created_at": "not a timestamp"},
    {"notification": []}, {"delivery": []}, {"response": []},
    {"status": "withdrawn"}, {"status": "pending", "response": {
        "source": "operator-spoke", "request_digest": "not-this-request", "text": "yes"}},
])
def test_malformed_state_is_quarantined_per_item(tmp_path, smoke_request, change):
    queue = SmokeRequests(tmp_path)
    good = queue.submit(smoke_request)
    bad_id = str(uuid4())
    bad = queue.submit({**smoke_request, "id": bad_id})
    bad.update(change)
    queue.path(bad_id).write_text(json.dumps(bad))
    rows, errors = queue.scan()
    assert rows == [good]
    assert len(errors) == 1 and bad_id in errors[0]


def test_notification_scan_survives_a_request_disappearing(tmp_path, smoke_request, monkeypatch):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    other_id = str(uuid4())
    queue.submit({**smoke_request, "id": other_id})
    original_scan = queue.scan
    def scan_then_remove():
        result = original_scan()
        queue.path(smoke_request["id"]).unlink()
        return result
    monkeypatch.setattr(queue, "scan", scan_then_remove)
    assert [row["request"]["id"] for row in queue.claim_notifications()] == [other_id]


def test_interrupted_delivery_can_explicitly_recover_saved_answer(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep the exact reply.\n")
    def interrupted(row):
        raise SystemExit("simulated process interruption after durable start")
    with pytest.raises(SystemExit):
        queue.deliver(smoke_request["id"], interrupted)
    restarted = SmokeRequests(tmp_path)
    calls = []
    def recover(row):
        calls.append(row)
        return {"transport_verified": True, "semantic_receipt": False}
    assert restarted.deliver(smoke_request["id"], recover)["delivery"]["state"] == "sending"
    assert not calls
    result = restarted.deliver(smoke_request["id"], recover, retry=True)
    assert result["delivery"]["state"] == "delivered"
    assert len(calls) == 1
    assert calls[0]["response"]["text"] == "Keep the exact reply.\n"
    assert result["delivery"]["previous"]["state"] == "sending"


def test_concurrent_explicit_returns_have_one_sender(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    entered, release = threading.Event(), threading.Event()
    calls = []
    def send(row):
        calls.append(row)
        entered.set()
        assert release.wait(5), "test did not release sender"
        return {"transport_verified": True}
    with ThreadPoolExecutor(1) as pool:
        worker = pool.submit(queue.deliver, smoke_request["id"], send)
        try:
            assert entered.wait(5), "sender did not enter"
            result = SmokeRequests(tmp_path).deliver(smoke_request["id"], send, retry=True)
            assert result["delivery"]["state"] == "sending"
            assert len(calls) == 1
        finally:
            release.set()
        assert worker.result()["delivery"]["state"] == "delivered"


def test_withdrawal_during_inflight_return_preserves_verified_transport(
    tmp_path, smoke_request
):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    entered, release = threading.Event(), threading.Event()

    def send(row):
        entered.set()
        assert release.wait(5), "test did not release sender"
        return {"transport_verified": True}

    with ThreadPoolExecutor(1) as pool:
        worker = pool.submit(queue.deliver, smoke_request["id"], send)
        assert entered.wait(5), "sender did not enter"
        queue.act(smoke_request["id"], "withdraw", "Server retired")
        release.set()
        result = worker.result()

    assert result["status"] == "withdrawn"
    assert result["delivery"]["state"] == "delivered"
    assert result["withdrawal_reason"] == "Server retired"


def test_watch_close_tolerates_thread_exit_between_probe_and_wakeup():
    from unittest.mock import Mock
    watch = DirectoryWatch.__new__(DirectoryWatch)
    reader, watch.write_fd = os.pipe()
    os.close(reader)
    watch.thread = Mock()
    watch.thread.is_alive.return_value = True
    watch._close_lock = threading.Lock()
    watch.close()
    watch.close()
    assert watch.write_fd is None


def test_tampered_response_binding_is_not_delivered(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    file = tmp_path / f"{smoke_request['id']}.json"
    row = json.loads(file.read_text())
    row["request"]["prompt"] = "A different question"
    file.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="digest"):
        queue.deliver(smoke_request["id"], lambda row: pytest.fail("must not send"))


def test_peer_return_uses_fixed_argv_exact_source_and_strict_receipt(tmp_path, smoke_request):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "A; $(touch should-not-run)\nB")
    commands = []
    def runner(argv, **kwargs):
        commands.append(argv)
        assert kwargs.get("shell", False) is False
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "schema": "epistaxis.pty_broker.peer_send_result.v1",
            "source_diaulos": "handy-handy-man", "target_diaulos": "handy-handy-man",
            "status": "submitted", "transport_verified": True,
            "submit_signal_written": True, "semantic_receipt": False,
            "submit_result": {
                "schema": "epistaxis.pty_broker.submit_result.v1",
                "status": "submitted",
                "request_id": f"smoke-response-{smoke_request['id']}",
                "target_diaulos": "handy-handy-man",
                "transport_verified": True, "submit_signal_written": True,
                "semantic_receipt": False,
                "response": {
                    "schema": "epistaxis.pty_broker.control_response.v1",
                    "status": "submitted",
                    "request_id": f"smoke-response-{smoke_request['id']}",
                    "target_diaulos": "handy-handy-man",
                    "submit_signal_written": True, "semantic_receipt": False,
                },
                "durable_receipt": {
                    "required": True, "write_verified": True, "submit_verified": True}},
        }), "")
    row = queue.get(smoke_request["id"])
    result = return_response(row, executable="/bin/epistaxis-test", runner=runner)
    assert result["transport_verified"] is True
    assert commands[0][0:3] == ["/bin/epistaxis-test", "pty-broker", "peer-send"]
    text = commands[0][commands[0].index("--text") + 1]
    assert smoke_request["source"]["thread_id"] in text
    assert json.dumps(row["response"]["text"]) in text


@pytest.mark.parametrize(("path", "value"), [
    (("status",), "failed"),
    (("submit_signal_written",), False),
    (("submit_result", "schema"), "wrong.schema"),
    (("submit_result", "status"), "failed"),
    (("submit_result", "transport_verified"), False),
    (("submit_result", "submit_signal_written"), False),
    (("submit_result", "response", "request_id"), "wrong-request"),
    (("submit_result", "response", "status"), "failed"),
    (("submit_result", "response", "submit_signal_written"), False),
])
def test_contradictory_peer_receipt_is_never_verified(
    tmp_path, smoke_request, path, value
):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    identity = f"smoke-response-{smoke_request['id']}"
    payload = {
        "schema": "epistaxis.pty_broker.peer_send_result.v1",
        "source_diaulos": "handy-handy-man",
        "target_diaulos": "handy-handy-man",
        "status": "submitted",
        "transport_verified": True,
        "submit_signal_written": True,
        "semantic_receipt": False,
        "submit_result": {
            "schema": "epistaxis.pty_broker.submit_result.v1",
            "status": "submitted",
            "request_id": identity,
            "target_diaulos": "handy-handy-man",
            "transport_verified": True,
            "submit_signal_written": True,
            "semantic_receipt": False,
            "response": {
                "schema": "epistaxis.pty_broker.control_response.v1",
                "status": "submitted",
                "request_id": identity,
                "target_diaulos": "handy-handy-man",
                "submit_signal_written": True,
                "semantic_receipt": False,
            },
            "durable_receipt": {
                "required": True,
                "write_verified": True,
                "submit_verified": True,
            },
        },
    }
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = return_response(
        queue.get(smoke_request["id"]),
        executable="/test/epistaxis",
        runner=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, json.dumps(payload), ""
        ),
    )
    assert result["transport_verified"] is False


@pytest.mark.parametrize("payload", ["", "not json", "{}", '{"transport_verified":true}',
                                      '{"schema":"epistaxis.pty_broker.peer_send_result.v1","target_diaulos":"wrong"}'])
def test_blank_partial_or_wrong_route_receipt_is_not_delivery(tmp_path, smoke_request, payload):
    queue = SmokeRequests(tmp_path)
    queue.submit(smoke_request)
    queue.reply(smoke_request["id"], "Keep B")
    result = return_response(queue.get(smoke_request["id"]), executable="/test/epistaxis",
        runner=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, payload, ""))
    assert result["transport_verified"] is False
