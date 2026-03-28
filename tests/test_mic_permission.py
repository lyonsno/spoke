"""Tests for async mic permission checking.

The mic permission probe (sd.rec) must run off the main/AppKit thread.
Blocking the main thread with sd.rec(blocking=True) deadlocks the NSRunLoop,
causing a beachball that survives SIGTERM.
"""

import threading
import time
from unittest.mock import MagicMock, patch, call


def _make_mic_delegate(main_module, monkeypatch):
    """Create a minimal SpokeAppDelegate for mic permission tests."""
    monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")
    delegate = main_module.SpokeAppDelegate.__new__(main_module.SpokeAppDelegate)
    delegate._menubar = MagicMock()
    delegate._detector = MagicMock()
    delegate._detector.install.return_value = True
    delegate._models_ready = False
    delegate._warm_error = None
    delegate._mic_probe_in_flight = False
    delegate.performSelectorOnMainThread_withObject_waitUntilDone_ = MagicMock()
    return delegate


class TestMicPermissionAsync:
    """Mic permission probe must never block the main thread."""

    def test_request_mic_permission_does_not_call_sd_rec_on_calling_thread(
        self, main_module, monkeypatch
    ):
        """sd.rec(blocking=True) must not execute on the thread that calls
        _request_mic_permission — it should be dispatched to a background thread."""
        delegate = _make_mic_delegate(main_module, monkeypatch)

        rec_called = threading.Event()
        rec_thread_names = []
        original_thread = threading.current_thread().name

        def fake_rec(*args, **kwargs):
            rec_thread_names.append(threading.current_thread().name)
            rec_called.set()

        with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
            import sys
            sys.modules["sounddevice"].rec = fake_rec

            delegate._request_mic_permission()
            rec_called.wait(timeout=2.0)

        assert len(rec_thread_names) >= 1, "sd.rec was never called"
        for name in rec_thread_names:
            assert name != original_thread, (
                f"sd.rec ran on the main thread ({original_thread}) — "
                "this will deadlock the NSRunLoop"
            )

    def test_mic_permission_success_dispatches_granted_to_main_thread(
        self, main_module, monkeypatch
    ):
        """When mic probe succeeds on the background thread, micPermissionGranted:
        must be dispatched to the main thread."""
        delegate = _make_mic_delegate(main_module, monkeypatch)

        rec_called = threading.Event()

        def fake_rec(*args, **kwargs):
            rec_called.set()

        with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
            import sys
            sys.modules["sounddevice"].rec = fake_rec

            delegate._request_mic_permission()
            rec_called.wait(timeout=2.0)
            # Give dispatch a moment to fire
            time.sleep(0.05)

        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_with(
            "micPermissionGranted:", None, False
        )

    def test_mic_permission_failure_dispatches_denied_and_returns_promptly(
        self, main_module, monkeypatch
    ):
        """When sd.rec raises, micPermissionDenied: should be dispatched to
        the main thread, and _request_mic_permission must return promptly."""
        delegate = _make_mic_delegate(main_module, monkeypatch)

        rec_called = threading.Event()

        def blocking_rec(*args, **kwargs):
            rec_called.set()
            raise RuntimeError("PortAudio error")

        with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
            import sys
            sys.modules["sounddevice"].rec = blocking_rec

            start = time.monotonic()
            delegate._request_mic_permission()
            elapsed = time.monotonic() - start

            rec_called.wait(timeout=2.0)
            time.sleep(0.05)

        assert elapsed < 0.5, (
            f"_request_mic_permission blocked for {elapsed:.2f}s — "
            "sd.rec is probably running on the main thread"
        )
        delegate.performSelectorOnMainThread_withObject_waitUntilDone_.assert_called_with(
            "micPermissionDenied:", None, False
        )

    def test_retry_mic_permission_does_not_block_main_thread(
        self, main_module, monkeypatch
    ):
        """retryMicPermission_ must not call sd.rec(blocking=True) synchronously
        on the main thread either."""
        delegate = _make_mic_delegate(main_module, monkeypatch)

        rec_called = threading.Event()
        rec_thread_names = []
        calling_thread = threading.current_thread().name

        def fake_rec(*args, **kwargs):
            rec_thread_names.append(threading.current_thread().name)
            rec_called.set()
            raise RuntimeError("Still no mic")

        with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
            import sys
            sys.modules["sounddevice"].rec = fake_rec

            delegate.retryMicPermission_(MagicMock())
            rec_called.wait(timeout=2.0)

        assert len(rec_thread_names) >= 1, "sd.rec was never called in retry"
        for name in rec_thread_names:
            assert name != calling_thread, (
                "retryMicPermission_ called sd.rec on the main thread"
            )

    def test_concurrent_probe_is_deduplicated(self, main_module, monkeypatch):
        """If a probe is already in flight, a second call should be a no-op."""
        delegate = _make_mic_delegate(main_module, monkeypatch)

        call_count = 0
        first_entered = threading.Event()
        release = threading.Event()

        def slow_rec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            first_entered.set()
            release.wait(timeout=2.0)

        with patch.dict("sys.modules", {"sounddevice": MagicMock()}):
            import sys
            sys.modules["sounddevice"].rec = slow_rec

            delegate._request_mic_permission()
            first_entered.wait(timeout=2.0)

            # Second call while first is still in-flight
            delegate._request_mic_permission()
            release.set()
            time.sleep(0.1)

        assert call_count == 1, (
            f"sd.rec called {call_count} times — concurrent probe should be deduplicated"
        )


class TestSigtermMenuBarCleanup:
    """SIGTERM must remove the NSStatusItem to prevent ghost menu bar icons."""

    def test_sigterm_handler_calls_menubar_cleanup(self, main_module, monkeypatch):
        """The SIGTERM handler should call menubar.cleanup() before
        calling NSApp.terminate_."""
        monkeypatch.setenv("SPOKE_WHISPER_URL", "http://test:8000")

        import signal as signal_mod
        import sys

        captured_handlers = {}

        def capture_signal(signum, handler):
            captured_handlers[signum] = handler

        with patch.object(main_module, "NSApp", MagicMock()):
            with patch.object(main_module, "NSApplication", MagicMock()) as mock_nsapp_cls:
                mock_nsapp_cls.sharedApplication.return_value = MagicMock()
                with patch.object(signal_mod, "signal", side_effect=capture_signal):
                    delegate = MagicMock()
                    delegate._menubar = MagicMock()
                    delegate._detector = MagicMock()

                    with patch.object(
                        main_module.SpokeAppDelegate, "alloc", return_value=MagicMock()
                    ) as mock_alloc:
                        mock_alloc.return_value.init.return_value = delegate

                        with patch("PyObjCTools.AppHelper.runEventLoop"):
                            main_module.main()

        assert signal_mod.SIGTERM in captured_handlers, (
            "main() did not install a SIGTERM handler"
        )

        captured_handlers[signal_mod.SIGTERM](signal_mod.SIGTERM, None)

        delegate._menubar.cleanup.assert_called_once()
