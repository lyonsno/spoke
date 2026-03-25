"""Entry point for donttype — macOS global hold-to-dictate.

Run with:  uv run donttype
    or:    uv run python -m donttype

Configure via environment variables:
    DICTATE_WHISPER_URL    Sidecar server URL (optional — if unset, uses local MLX Whisper)
    DICTATE_WHISPER_MODEL  Model name (default: mlx-community/whisper-large-v3-turbo)
    DICTATE_HOLD_MS        Hold threshold in ms (default: 250, must be > 0)
    DICTATE_RESTORE_DELAY_MS  Pasteboard restore delay in ms (default: 1000)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

import objc
from AppKit import (
    NSAlert,
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
)
from Foundation import NSObject

from .capture import AudioCapture
from .glow import GlowOverlay
from .inject import inject_text
from .input_tap import SpacebarHoldDetector
from .menubar import MenuBarIcon
from .overlay import TranscriptionOverlay
from .transcribe import TranscriptionClient
from .transcribe_local import LocalTranscriptionClient
from .transcribe_qwen import LocalQwenClient

logger = logging.getLogger(__name__)


def _get_ram_gb() -> float:
    """Return system RAM in GB via sysctl."""
    import subprocess
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


# Recording cap: 20s on machines with < 36GB RAM to avoid Metal GPU crashes
# on long MLX inference buffers. No cap in sidecar mode (inference is remote).
_RAM_GB = _get_ram_gb()
_MAX_RECORD_SECS: float | None = 15.0 if _RAM_GB < 36 else None


class DontTypeAppDelegate(NSObject):
    """Main application delegate — wires input → capture → transcribe → inject."""

    def init(self):
        self = objc.super(DontTypeAppDelegate, self).init()
        if self is None:
            return None

        whisper_url = os.environ.get("DICTATE_WHISPER_URL", "")
        model = os.environ.get(
            "DICTATE_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo"
        )
        hold_ms_raw = os.environ.get("DICTATE_HOLD_MS", "250")
        try:
            hold_ms = int(hold_ms_raw)
        except ValueError:
            logger.error("DICTATE_HOLD_MS must be an integer, got %r", hold_ms_raw)
            print(
                f"ERROR: DICTATE_HOLD_MS must be an integer, got {hold_ms_raw!r}.\n"
                "  Example: DICTATE_HOLD_MS=400 uv run donttype",
                file=sys.stderr,
            )
            sys.exit(1)

        if hold_ms <= 0:
            logger.error("DICTATE_HOLD_MS must be > 0, got %d", hold_ms)
            print(
                f"ERROR: DICTATE_HOLD_MS must be > 0, got {hold_ms}.\n"
                "  Example: DICTATE_HOLD_MS=400 uv run donttype",
                file=sys.stderr,
            )
            sys.exit(1)

        self._capture = AudioCapture()
        self._capture.warmup()
        if whisper_url:
            logger.info("Using sidecar transcription: %s", whisper_url)
            self._client = TranscriptionClient(base_url=whisper_url, model=model)
            self._preview_client = TranscriptionClient(base_url=whisper_url, model=model)
            self._local_mode = False
        else:
            if model.startswith("Qwen/"):
                logger.info("Using local Qwen3 ASR: %s", model)
                self._client = LocalQwenClient(model=model)
            else:
                logger.info("Using local transcription: %s", model)
                self._client = LocalTranscriptionClient(model=model)
            self._preview_client = self._client  # share model instance
            self._local_mode = True
        self._detector = SpacebarHoldDetector.alloc().initWithHoldStart_holdEnd_holdMs_(
            self._on_hold_start,
            self._on_hold_end,
            hold_ms,
        )
        self._menubar: MenuBarIcon | None = None
        self._glow: GlowOverlay | None = None
        self._overlay: TranscriptionOverlay | None = None
        self._transcribing = False
        self._transcription_token = 0
        self._preview_active = False
        self._preview_thread: threading.Thread | None = None
        self._record_start_time: float = 0.0
        self._cap_fired = False

        if self._local_mode and _MAX_RECORD_SECS is not None:
            logger.info(
                "RAM %.0fGB < 36GB — recording capped at %.0fs to avoid Metal crashes",
                _RAM_GB, _MAX_RECORD_SECS,
            )

        return self

    # ── NSApplication delegate ──────────────────────────────

    def applicationDidFinishLaunching_(self, notification) -> None:
        self._menubar = MenuBarIcon.alloc().initWithQuitCallback_(self._quit)
        self._menubar.setup()

        self._glow = GlowOverlay.alloc().initWithScreen_(None)
        self._glow.setup()

        self._overlay = TranscriptionOverlay.alloc().initWithScreen_(None)
        self._overlay.setup()

        # Step 1: Request mic permission with a test recording.
        # This triggers the system prompt before we start listening for spacebar.
        self._menubar.set_status_text("Requesting mic access…")
        self._request_mic_permission()

    def _request_mic_permission(self) -> None:
        """Trigger mic permission prompt by attempting a short recording."""
        import sounddevice as sd
        from Foundation import NSTimer
        try:
            # Record 0.1s — just enough to trigger the permission prompt
            sd.rec(1600, samplerate=16000, channels=1, dtype='float32', blocking=True)
            logger.info("Microphone access granted")
            self._setup_event_tap()
        except Exception as e:
            logger.warning("Mic permission not yet granted: %s", e)
            self._menubar.set_status_text("Grant mic access, then wait…")
            # Retry every 2 seconds until mic works
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryMicPermission:", None, False
            )

    def retryMicPermission_(self, timer) -> None:
        """Retry mic access check."""
        import sounddevice as sd
        from Foundation import NSTimer
        try:
            sd.rec(1600, samplerate=16000, channels=1, dtype='float32', blocking=True)
            logger.info("Microphone access granted")
            self._setup_event_tap()
        except Exception:
            logger.debug("Still waiting for mic permission")
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryMicPermission:", None, False
            )

    def _setup_event_tap(self) -> None:
        """Install the event tap after permissions are confirmed."""
        if not self._detector.install():
            self._show_accessibility_alert()
            self._menubar.set_status_text("Grant accessibility, then wait…")
            # Retry every 2 seconds
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryEventTap:", None, False
            )
            return

        logger.info("donttype ready — hold spacebar to record")
        self._menubar.set_status_text("Ready — hold spacebar")

    def retryEventTap_(self, timer) -> None:
        """Retry event tap installation."""
        if self._detector.install():
            logger.info("donttype ready — hold spacebar to record")
            self._menubar.set_status_text("Ready — hold spacebar")
        else:
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryEventTap:", None, False
            )

    # ── hold callbacks (called on main thread) ──────────────

    def _on_hold_start(self) -> None:
        if self._transcribing:
            logger.warning("Hold started while transcription in flight — ignoring")
            return

        logger.info("Hold started — recording")
        if self._menubar is not None:
            self._menubar.set_recording(True)
            self._menubar.set_status_text("Recording…")
        if self._glow is not None:
            self._glow.show()
        if self._overlay is not None:
            self._overlay.show()
        self._capture.start(amplitude_callback=self._on_amplitude)
        self._record_start_time = time.monotonic()
        self._cap_fired = False

        # Start the adaptive preview loop
        self._preview_active = True
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def _on_amplitude(self, rms: float) -> None:
        """Called from PortAudio thread — marshal to main thread."""
        from Foundation import NSNumber
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "amplitudeUpdate:", NSNumber.numberWithFloat_(rms), False
        )

    def amplitudeUpdate_(self, rms_number) -> None:
        """Main thread: forward amplitude to glow and overlay text."""
        rms = float(rms_number)

        # Recording cap: check elapsed time on every amplitude tick
        if self._local_mode and _MAX_RECORD_SECS is not None and not self._cap_fired:
            elapsed = time.monotonic() - self._record_start_time
            _CAP_WARN_SECS = 5.0

            # Update countdown glow in the last 5 seconds
            warn_start = _MAX_RECORD_SECS - _CAP_WARN_SECS
            if elapsed >= warn_start and self._glow is not None:
                progress = min((elapsed - warn_start) / _CAP_WARN_SECS, 1.0)
                eased = progress * progress  # ease-in: slow start, accelerating
                self._glow._cap_factor = 1.0 - eased

            # Fire the cap
            if elapsed >= _MAX_RECORD_SECS:
                self._cap_fired = True
                logger.warning(
                    "Recording capped at %.0fs (%.0fGB RAM) — transcribing what we have",
                    _MAX_RECORD_SECS, _RAM_GB,
                )
                self._detector.force_end()
                return

        if self._glow is not None:
            self._glow.update_amplitude(rms)
        if self._overlay is not None and self._glow is not None:
            # Text breathing: glow's smoothed amplitude, scaled independently
            self._overlay.update_text_amplitude(
                min(self._glow._smoothed_amplitude * 18.0, 1.0)
            )
            # Inner glow: track screen glow, saturate aggressively
            glow_opacity = self._glow._glow_layer.opacity()
            self._overlay.update_glow_amplitude(
                min(glow_opacity * 2.5, 1.0),
                cap_factor=self._glow._cap_factor,
            )

    def _preview_loop(self) -> None:
        """Background thread: preview transcription during recording.

        Uses streaming (KV-cache reuse) when the client supports it,
        falling back to full-buffer re-transcription otherwise.
        """
        use_streaming = getattr(self._preview_client, 'supports_streaming', False)

        if use_streaming:
            self._preview_loop_streaming()
        else:
            self._preview_loop_batch()

    def _preview_loop_streaming(self) -> None:
        """Streaming preview: feed incremental audio chunks, read state.text."""
        _FEED_INTERVAL = 0.15  # feed new audio every 150ms

        # Wait for some audio to accumulate before first feed
        time.sleep(0.4)

        try:
            self._preview_client.start_stream()
        except Exception:
            logger.exception("Failed to start streaming — falling back to batch")
            self._preview_loop_batch()
            return

        while self._preview_active:
            loop_start = time.monotonic()

            frames = self._capture.get_new_frames()

            try:
                text = self._preview_client.feed(frames)
            except Exception:
                logger.debug("Streaming feed failed", exc_info=True)
                time.sleep(0.5)
                continue

            if text and self._preview_active:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "previewTextUpdate:", text, False
                )

            elapsed = time.monotonic() - loop_start
            remaining = _FEED_INTERVAL - elapsed
            if remaining > 0 and self._preview_active:
                time.sleep(remaining)

    def _preview_loop_batch(self) -> None:
        """Batch preview: re-transcribe the full buffer each tick."""
        _MIN_INTERVAL = 0.5 if self._local_mode else 0.75
        _INITIAL_DELAY = 0.4 if self._local_mode else 0.3

        time.sleep(_INITIAL_DELAY)

        while self._preview_active:
            loop_start = time.monotonic()

            wav_bytes = self._capture.get_buffer()
            if not wav_bytes:
                time.sleep(0.2)
                continue

            try:
                text = self._preview_client.transcribe(wav_bytes)
            except Exception:
                logger.debug("Preview transcription failed", exc_info=True)
                time.sleep(0.5)
                continue

            if text and self._preview_active:
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "previewTextUpdate:", text, False
                )

            elapsed = time.monotonic() - loop_start
            remaining = _MIN_INTERVAL - elapsed
            if remaining > 0 and self._preview_active:
                time.sleep(remaining)

    def previewTextUpdate_(self, text: str) -> None:
        """Main thread: update the overlay with preview transcription text."""
        if self._overlay is not None:
            self._overlay.set_text(text)

    def _on_hold_end(self) -> None:
        logger.info("Hold ended — transcribing")
        self._preview_active = False
        wav_bytes = self._capture.stop()

        if self._glow is not None:
            self._glow.hide()
        if self._menubar is not None:
            self._menubar.set_recording(False)
            self._menubar.set_status_text("Transcribing…")

        if not wav_bytes:
            logger.warning("No audio captured")
            if self._overlay is not None:
                self._overlay.hide()
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")
            return

        # Invalidate any in-flight transcription so its result is discarded
        self._transcription_token += 1
        token = self._transcription_token

        self._transcribing = True
        self._transcribe_start = time.monotonic()
        thread = threading.Thread(
            target=self._transcribe_worker, args=(wav_bytes, token), daemon=True
        )
        thread.start()

    def _transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: finalize transcription and marshal result to main thread."""
        # Wait for preview loop to finish so we don't hit the model concurrently
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            # If the preview client was streaming, finalize it for the final text
            # (this runs the tail refinement pass with the existing KV cache)
            if getattr(self._client, 'supports_streaming', False) and self._client._stream_state is not None:
                text = self._client.finish_stream()
            else:
                text = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "transcriptionFailed:", {"token": token}, False
            )
            return

        elapsed_ms = (time.monotonic() - self._transcribe_start) * 1000
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "transcriptionComplete:",
            {"token": token, "text": text, "elapsed_ms": elapsed_ms},
            False,
        )

    def transcriptionComplete_(self, payload: dict) -> None:
        """Main thread: inject transcribed text at cursor."""
        if payload["token"] != self._transcription_token:
            logger.info("Discarding stale transcription (token %d)", payload["token"])
            return
        self._transcribing = False
        text = payload["text"]
        if text:
            def _on_clipboard_restored():
                if self._menubar is not None:
                    self._menubar.set_status_text("Ready — hold spacebar")

            # Show final text in overlay briefly before fading
            if self._overlay is not None:
                self._overlay.set_text(text)

            inject_text(text, on_restored=_on_clipboard_restored)
            elapsed_ms = payload.get("elapsed_ms", 0)
            logger.info("Injected: %r (%.0fms)", text, elapsed_ms)
            if self._menubar is not None:
                self._menubar.set_status_text("Pasted!")

            # Fade overlay out after a brief display of final text
            if self._overlay is not None:
                from Foundation import NSTimer
                NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    0.5, self, "hideOverlayAfterInject:", None, False
                )
            return
        if self._overlay is not None:
            self._overlay.hide()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def transcriptionFailed_(self, payload: dict) -> None:
        """Main thread: handle transcription error."""
        if payload["token"] != self._transcription_token:
            return  # stale failure, ignore
        self._transcribing = False
        logger.error("Transcription failed — no text injected")
        if self._overlay is not None:
            self._overlay.hide()
        if self._menubar is not None:
            self._menubar.set_status_text("Error — try again")

    def hideOverlayAfterInject_(self, timer) -> None:
        """Hide the overlay after briefly showing the final transcription."""
        if self._overlay is not None:
            self._overlay.hide()

    # ── helpers ─────────────────────────────────────────────

    def _quit(self) -> None:
        self._detector.uninstall()
        self._preview_active = False
        self._client.close()
        if self._preview_client is not None and self._preview_client is not self._client:
            self._preview_client.close()
        NSApp.terminate_(None)

    def _show_accessibility_alert(self) -> None:
        """Show a dialog explaining the Accessibility permission requirement."""
        alert = NSAlert.new()
        alert.setMessageText_("Accessibility Permission Required")
        alert.setInformativeText_(
            "DontType needs Accessibility access to detect spacebar holds.\n\n"
            "Go to System Settings → Privacy & Security → Accessibility "
            "and enable access for your terminal app (Terminal, iTerm2, etc.).\n\n"
            "Then relaunch DontType."
        )
        alert.addButtonWithTitle_("OK")
        # Temporarily become a regular app so the alert is visible
        NSApp.setActivationPolicy_(1)  # NSApplicationActivationPolicyRegular
        alert.runModal()


def _acquire_instance_lock() -> None:
    """Single-instance guard — kill any stuck old instance and take the lock."""
    import fcntl
    import signal as sig
    import time

    lock_path = os.path.expanduser("~/Library/Logs/.donttype.lock")
    lock_file = open(lock_path, "a+")
    lock_file.seek(0)
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            lock_file.seek(0)
            old_pid = int(lock_file.read().strip())
            logger.info("Killing old instance (pid=%d)", old_pid)
            os.kill(old_pid, sig.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        for _attempt in range(10):
            time.sleep(0.2)
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if _attempt == 4:
                    try:
                        os.kill(old_pid, sig.SIGKILL)
                        logger.info("Escalated to SIGKILL (pid=%d)", old_pid)
                    except (ProcessLookupError, PermissionError, NameError):
                        pass
                continue
        else:
            logger.warning("Another instance is running — exiting")
            sys.exit(0)

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    # Keep lock_file alive for process lifetime
    _acquire_instance_lock._lock_file = lock_file


def main() -> None:
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _acquire_instance_lock()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    delegate = DontTypeAppDelegate.alloc().init()
    app.setDelegate_(delegate)

    # Clean shutdown on SIGTERM — uninstall event tap before dying
    # so we don't leave a zombie tap that eats keyboard events
    def _handle_sigterm(signum, frame):
        logger.info("Received SIGTERM — cleaning up")
        delegate._detector.uninstall()
        NSApp.terminate_(None)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    from PyObjCTools import AppHelper

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
