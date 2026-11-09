"""Native lifecycle ownership without touching a real microphone."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import spoke.capture as capture


@pytest.fixture
def cap(monkeypatch):
    monkeypatch.setenv("SPOKE_VAD_ENABLED", "0")
    return capture.AudioCapture()


def test_teardown_remains_busy_until_native_stop_finishes(cap):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    stream = MagicMock()
    stream.stop.side_effect = lambda: (entered.set(), release.wait())
    stream.close.side_effect = closed.set
    with patch.object(capture.sd, "InputStream", return_value=stream):
        cap.start()
        cap._frames = [np.ones(1024, dtype=np.float32) * 0.2]
        try:
            wav = cap.stop()
            assert entered.wait(1)
            assert len(wav) == 44 + 2048
            assert cap._stream_closing, "Detached native teardown was reported complete"
        finally:
            release.set()
            assert closed.wait(1)


def test_async_open_is_off_caller_and_cancelled_release_never_starts_stream(cap):
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    threads = []
    stream = MagicMock()

    def open_stream(**kwargs):
        threads.append(threading.get_ident())
        entered.set()
        release.wait()
        return stream

    assert callable(getattr(cap, "start_async", None)), "Capture has no nonblocking start route"
    with patch.object(capture.sd, "InputStream", side_effect=open_stream):
        future = cap.start_async()
        try:
            assert entered.wait(1)
            assert not future.done()
            assert threads == [threads[0]] and threads[0] != caller
            assert cap.stop() == b""
        finally:
            release.set()
        with pytest.raises(capture.CaptureStartCancelled):
            future.result(timeout=1)
    stream.start.assert_not_called()
    stream.close.assert_called_once()
    assert not cap.is_recording


def test_async_start_refuses_to_overlap_blocked_teardown(cap):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    stream = MagicMock()
    stream.stop.side_effect = lambda: (entered.set(), release.wait())
    stream.close.side_effect = closed.set
    with patch.object(capture.sd, "InputStream", return_value=stream) as opened:
        cap.start()
        try:
            cap.stop()
            assert entered.wait(1)
            assert callable(getattr(cap, "start_async", None)), "Capture has no nonblocking start route"
            with pytest.raises(capture.CaptureBusyError):
                cap.start_async()
            assert opened.call_count == 1
        finally:
            release.set()
            assert closed.wait(1)


def test_start_thread_failure_does_not_leave_capture_permanently_busy(cap):
    with patch.object(capture.threading, "Thread") as worker:
        worker.return_value.start.side_effect = RuntimeError("thread unavailable")
        with pytest.raises(RuntimeError, match="thread unavailable"):
            cap.start_async()
    stream = MagicMock()
    with patch.object(capture.sd, "InputStream", return_value=stream):
        cap.start_async().result(timeout=1)
        cap.stop()
        assert cap._teardown_complete.wait(1)


def test_release_during_failed_open_does_not_reset_or_retry_native_audio(cap):
    def fail_after_release(**kwargs):
        cap.stop()
        raise RuntimeError("native open failed")
    with patch.object(capture.sd, "InputStream", side_effect=fail_after_release) as opened, \
            patch.object(cap, "_reset_portaudio") as reset:
        with pytest.raises(capture.CaptureStartCancelled):
            cap.start_async().result(timeout=1)
    assert opened.call_count == 1
    reset.assert_not_called()


def test_old_audio_callback_cannot_append_into_new_generation(cap):
    cap._stream = MagicMock()
    cap._stream_generation = 7
    sliced = MagicMock()
    def replace_during_copy():
        cap._stream_generation = 8
        cap._frames = []
        return np.ones(1024, dtype=np.float32)
    sliced.copy.side_effect = replace_during_copy
    indata = MagicMock()
    indata.__getitem__.return_value = sliced
    cap._audio_callback(indata, 1024, None, None, generation=7)
    assert cap._frames == []


def test_failed_native_close_does_not_authorize_reopening(cap):
    stream = MagicMock()
    stream.close.side_effect = RuntimeError("device did not close")
    with patch.object(capture.sd, "InputStream", return_value=stream):
        cap.start()
        cap.stop()
        assert cap._teardown_complete.wait(1)
        assert cap._stream_closing, "Failed close was presented as a released microphone"
        with pytest.raises(capture.CaptureBusyError):
            cap.start_async()


def test_teardown_thread_failure_still_returns_the_complete_recording(cap):
    stream = MagicMock()
    cap._stream = stream
    cap._frames = [np.ones(1024, dtype=np.float32) * 0.2]
    with patch.object(capture.threading, "Thread") as worker:
        worker.return_value.start.side_effect = RuntimeError("no thread available")
        wav = cap.stop()
    assert len(wav) == 44 + 2048
    assert cap._stream_closing and cap._teardown_complete.is_set()
    assert cap._failed_teardown_stream is stream
    stream.stop.assert_not_called()
    stream.close.assert_not_called()


def test_frames_delivered_inside_native_start_are_retained(cap):
    stream = MagicMock()
    def opened(**kwargs):
        stream.start.side_effect = lambda: kwargs["callback"](
            np.ones((1024, 1), dtype=np.float32) * 0.2, 1024, None, None)
        return stream
    with patch.object(capture.sd, "InputStream", side_effect=opened):
        cap.start_async().result(timeout=1)
        try:
            assert len(cap.get_buffer()) == 44 + 2048
        finally:
            cap.stop()
            assert cap._teardown_complete.wait(1)


def test_release_while_native_start_is_returning_keeps_received_audio(cap):
    entered, release = threading.Event(), threading.Event()
    stream = MagicMock()
    def opened(**kwargs):
        def started():
            kwargs["callback"](np.ones((1024, 1), dtype=np.float32) * 0.2, 1024, None, None)
            entered.set()
            release.wait()
        stream.start.side_effect = started
        return stream
    with patch.object(capture.sd, "InputStream", side_effect=opened):
        future = cap.start_async()
        try:
            assert entered.wait(1)
            wav = cap.stop()
            assert len(wav) == 44 + 2048
        finally:
            release.set()
            with pytest.raises(capture.CaptureStartCancelled):
                future.result(timeout=1)
