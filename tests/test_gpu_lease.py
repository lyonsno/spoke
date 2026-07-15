"""Spoke-side lifecycle and evidence contracts for GPU Greenroom leases."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


class _DeferredThread:
    def __init__(self, *, target, daemon, name):
        self.target = target
        self.daemon = daemon
        self.name = name
        self.started = False

    def start(self):
        self.started = True


def _fake_process() -> MagicMock:
    process = MagicMock()
    process.pid = 4242
    process.stdin = MagicMock()
    process.stdout = iter(())
    process.poll.return_value = None
    return process


def test_disabled_manager_does_not_launch(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    popen = MagicMock()
    manager = GPUInteractiveLeaseManager(
        enabled=False,
        binary=tmp_path / "gpu-greenroom",
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=popen,
    )

    assert manager.request("utterance-1") is None
    popen.assert_not_called()


def test_request_launches_asynchronously_with_caller_owned_receipt(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    process = _fake_process()
    popen = MagicMock(return_value=process)
    threads: list[_DeferredThread] = []

    def thread_factory(**kwargs):
        thread = _DeferredThread(**kwargs)
        threads.append(thread)
        return thread

    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        route_id="gpu-greenroom@733554a",
        popen_factory=popen,
        thread_factory=thread_factory,
    )

    lease = manager.request("utterance-2")

    assert lease is not None
    assert threads and threads[0].started is True
    process.wait.assert_not_called()
    command = popen.call_args.args[0]
    assert command[:4] == [
        str(binary),
        "--queue-dir",
        str(tmp_path / "queue"),
        "lease",
    ]
    assert command[4:6] == ["acquire", "--lease-id"]
    greenroom_receipt = tmp_path / "receipts" / "utterance-2.greenroom.json"
    assert command[command.index("--receipt-path") + 1] == str(greenroom_receipt)
    report = json.loads((tmp_path / "receipts" / "utterance-2.spoke.json").read_text())
    assert report["requested"] is True
    assert report["effective"] is False
    assert report["state"] == "launching"
    assert report["effective_route"] is None
    assert report["requested_route"]["binary"] == str(binary)
    assert report["requested_route"]["queue_dir"] == str(tmp_path / "queue")
    assert report["requested_route"]["route_id"] == "gpu-greenroom@733554a"
    assert report["scheduling_posture"] == "scheduler-unverified"


def test_missing_binary_writes_failure_report_without_launch(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    popen = MagicMock()
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=tmp_path / "missing-greenroom",
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=popen,
    )

    lease = manager.request("utterance-3")

    assert lease is not None
    popen.assert_not_called()
    report = lease.snapshot()
    assert report["state"] == "failed"
    assert report["failure_phase"] == "launch"
    assert report["effective"] is False
    assert report["last_trustworthy_event"] == "launch-rejected-missing-binary"
    assert Path(report["report_path"]).exists()


def test_effective_requires_event_and_live_holder_process(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    process = _fake_process()
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(return_value=process),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    lease = manager.request("utterance-4")

    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-4",
        "state": "requested",
        "effective_at": None,
    })
    assert lease.snapshot()["effective"] is False

    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-4",
        "state": "effective",
        "effective_at": 123.0,
    })
    assert lease.snapshot()["effective"] is True
    assert lease.snapshot()["effective_route"]["state"] == "effective"
    assert lease.snapshot()["scheduling_posture"] == "lease-effective"

    process.poll.return_value = 9
    assert lease.snapshot()["effective"] is False
    assert lease.snapshot()["state"] == "stale-effective"
    assert lease.snapshot()["scheduling_posture"] == "scheduler-unverified"


def test_effective_event_without_timestamp_fails_protocol_validation(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(return_value=_fake_process()),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    lease = manager.request("utterance-malformed-effective")

    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-malformed-effective",
        "state": "effective",
        "effective_at": None,
    })

    report = lease.snapshot()
    assert report["state"] == "failed"
    assert report["failure_phase"] == "protocol"
    assert report["effective"] is False
    assert report["scheduling_posture"] == "scheduler-unverified"
    assert report["last_trustworthy_event"] == "invalid-effective-event"


def test_protocol_failure_cannot_be_resurrected_to_effective(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(return_value=_fake_process()),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    lease = manager.request("utterance-bad-then-effective")

    lease._fail("protocol", "invalid-jsonl-event", "bad holder output")
    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-bad-then-effective",
        "state": "effective",
        "effective_at": 123.0,
    })

    report = lease.snapshot()
    assert report["state"] == "failed"
    assert report["effective"] is False
    assert report["scheduling_posture"] == "scheduler-unverified"
    assert report["last_trustworthy_event"] == "invalid-jsonl-event"


def test_release_requested_cannot_be_resurrected_to_effective(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(return_value=_fake_process()),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    lease = manager.request("utterance-release-race")

    lease.release()
    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-release-race",
        "state": "effective",
        "effective_at": 123.0,
    })

    report = lease.snapshot()
    assert report["state"] == "release-requested"
    assert report["effective"] is False
    assert report["scheduling_posture"] == "scheduler-unverified"
    assert report["last_trustworthy_event"] == "spoke-release-requested"


def test_release_requested_accepts_terminal_holder_event(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(return_value=_fake_process()),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    lease = manager.request("utterance-release-terminal")

    lease.release()
    lease._handle_event({
        "schema": "gpu-greenroom.interactive-lease.v1",
        "lease_id": "utterance-release-terminal",
        "state": "released",
        "effective_at": None,
    })

    report = lease.snapshot()
    assert report["state"] == "released"
    assert report["effective"] is False
    assert report["scheduling_posture"] == "scheduler-unverified"
    assert report["last_trustworthy_event"] == "greenroom-released"


def test_release_is_nonblocking_and_manager_close_releases_all(tmp_path):
    from spoke.gpu_lease import GPUInteractiveLeaseManager

    binary = tmp_path / "gpu-greenroom"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    processes = [_fake_process(), _fake_process()]
    manager = GPUInteractiveLeaseManager(
        enabled=True,
        binary=binary,
        queue_dir=tmp_path / "queue",
        receipt_dir=tmp_path / "receipts",
        popen_factory=MagicMock(side_effect=processes),
        thread_factory=lambda **kwargs: _DeferredThread(**kwargs),
    )
    first = manager.request("utterance-5")
    second = manager.request("utterance-6")

    first.release()
    processes[0].stdin.close.assert_called_once_with()
    processes[0].wait.assert_not_called()

    manager.close_all()
    processes[1].stdin.close.assert_called_once_with()
    processes[1].wait.assert_not_called()


def test_selected_smoke_surface_names_reviewed_greenroom_lease_route():
    smoke_env = Path(".spoke-smoke-env").read_text()

    assert 'export SPOKE_GPU_INTERACTIVE_LEASE="1"' in smoke_env
    assert (
        'export SPOKE_GPU_GREENROOM_ROUTE_ID="gpu-greenroom@733554a"'
        in smoke_env
    )
    assert (
        'export SPOKE_GPU_GREENROOM_BINARY="/private/tmp/'
        'gpu-greenroom-interactive-asr-lease-0714/'
        '.venv/bin/gpu-greenroom"'
        in smoke_env
    )
