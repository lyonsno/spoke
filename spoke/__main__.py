"""Entry point for spoke — macOS global hold-to-dictate.

Run with:  uv run spoke
    or:    uv run python -m spoke

Configure via environment variables:
    SPOKE_WHISPER_URL    Sidecar server URL (optional — if unset, uses local MLX Whisper)
    SPOKE_WHISPER_MODEL  Model name (default: mlx-community/whisper-large-v3-turbo)
    SPOKE_HOLD_MS        Hold threshold in ms (default: 200, must be > 0)
    SPOKE_RESTORE_DELAY_MS  Pasteboard restore delay in ms (default: 1000)
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
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
from .command import CommandClient
from .focus_check import has_focused_text_input
from .glow import GlowOverlay
from .inject import inject_text, save_pasteboard, restore_pasteboard, set_pasteboard_only
from .input_tap import SpacebarHoldDetector
from .menubar import MenuBarIcon
from .overlay import TranscriptionOverlay
from .transcribe import TranscriptionClient
from .transcribe_local import LocalTranscriptionClient, supports_eager_eval
from .transcribe_qwen import LocalQwenClient
from .tts import TTSClient

logger = logging.getLogger(__name__)

_DEFAULT_PREVIEW_MODEL = "mlx-community/whisper-base.en-mlx-8bit"
_DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-medium.en-mlx-8bit"
_DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT = 30.0
_DEFAULT_LOCAL_WHISPER_EAGER_EVAL = False

_NOT_CAPTURED = object()  # sentinel for _pre_paste_clipboard


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
_MAX_RECORD_SECS: float | None = 20.0 if _RAM_GB < 36 else None


class SpokeAppDelegate(NSObject):
    """Main application delegate — wires input → capture → transcribe → inject."""

    def init(self):
        self = objc.super(SpokeAppDelegate, self).init()
        if self is None:
            return None

        whisper_url = os.environ.get("SPOKE_WHISPER_URL", "")
        hold_ms_raw = os.environ.get("SPOKE_HOLD_MS", "200")
        try:
            hold_ms = int(hold_ms_raw)
        except ValueError:
            logger.error("SPOKE_HOLD_MS must be an integer, got %r", hold_ms_raw)
            print(
                f"ERROR: SPOKE_HOLD_MS must be an integer, got {hold_ms_raw!r}.\n"
                "  Example: SPOKE_HOLD_MS=400 uv run spoke",
                file=sys.stderr,
            )
            sys.exit(1)

        if hold_ms <= 0:
            logger.error("SPOKE_HOLD_MS must be > 0, got %d", hold_ms)
            print(
                f"ERROR: SPOKE_HOLD_MS must be > 0, got {hold_ms}.\n"
                "  Example: SPOKE_HOLD_MS=400 uv run spoke",
                file=sys.stderr,
            )
            sys.exit(1)

        self._whisper_url = whisper_url
        self._preview_model_id, self._transcription_model_id = self._resolve_model_ids()
        self._client_cache: dict[tuple[str, str], object] = {}
        self._capture = AudioCapture()
        self._capture.warmup()
        self._local_mode = not bool(whisper_url)
        (
            self._local_whisper_decode_timeout,
            self._local_whisper_eager_eval,
        ) = self._resolve_local_whisper_settings()
        self._client = self._get_client(whisper_url, self._transcription_model_id)
        self._preview_client = self._get_client(whisper_url, self._preview_model_id)
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
        self._preview_done = threading.Event()
        self._preview_done.set()
        self._preview_session_token = 0
        self._local_inference_lock = threading.Lock()
        self._record_start_time: float = 0.0
        self._cap_fired = False
        self._last_preview_text = ""
        self._models_ready = False
        self._warm_error: Exception | None = None
        self._mic_probe_in_flight = False

        # Command pathway — initialized if SPOKE_COMMAND_URL is configured
        command_url = os.environ.get("SPOKE_COMMAND_URL", "")
        if command_url:
            self._command_client = CommandClient()
            self._command_overlay: TranscriptionOverlay | None = None
            logger.info("Command pathway enabled: %s", command_url)
        else:
            self._command_client = None
            self._command_overlay = None

        # TTS autoplay — initialized if SPOKE_TTS_VOICE is set
        self._tts_client = TTSClient.from_env()
        if self._tts_client is not None:
            self._tts_client._gpu_lock = self._local_inference_lock
            logger.info("TTS enabled: voice=%s", self._tts_client._voice)

        # Recovery mode state
        # _NOT_CAPTURED sentinel distinguishes "not captured yet" from
        # "captured but clipboard was empty (None)".
        self._pre_paste_clipboard: list[tuple[str, bytes]] | None | object = _NOT_CAPTURED
        self._verify_paste_text: str | None = None
        self._verify_paste_attempt: int = 0
        self._recovery_saved_clipboard: list[tuple[str, bytes]] | None = None
        self._recovery_text: str | None = None
        self._recovery_clipboard_state: str = "idle"
        self._recovery_pending_insert = None
        self._recovery_hold_active: bool = False
        self._recovery_retry_pending: bool = False

        if self._local_mode and _MAX_RECORD_SECS is not None:
            logger.info(
                "RAM %.0fGB < 36GB — recording capped at %.0fs to avoid Metal crashes",
                _RAM_GB, _MAX_RECORD_SECS,
            )

        return self

    # ── NSApplication delegate ──────────────────────────────

    def applicationDidFinishLaunching_(self, notification) -> None:
        self._menubar = MenuBarIcon.alloc().initWithQuitCallback_selectModelCallback_(
            self._quit, self._handle_model_menu_action
        )
        self._menubar.setup()

        self._glow = GlowOverlay.alloc().initWithScreen_(None)
        self._glow.setup()

        self._overlay = TranscriptionOverlay.alloc().initWithScreen_(None)
        self._overlay.setup()

        # Command output overlay — separate surface for command responses
        if self._command_client is not None:
            from .command_overlay import CommandOverlay
            self._command_overlay = CommandOverlay.alloc().initWithScreen_(None)
            self._command_overlay.setup()

        # Step 1: Request mic permission with a test recording.
        # This triggers the system prompt before we start listening for spacebar.
        self._menubar.set_status_text("Requesting mic access…")
        self._request_mic_permission()

    def _request_mic_permission(self) -> None:
        """Trigger mic permission prompt by attempting a short recording.

        The sd.rec(blocking=True) call can deadlock the main thread if PortAudio
        hits a hardware error or the mic permission prompt is pending.  Run the
        probe on a background thread and dispatch the result back to the main
        thread so the NSRunLoop stays responsive.
        """
        if self._mic_probe_in_flight:
            return
        self._mic_probe_in_flight = True
        threading.Thread(
            target=self._probe_mic_permission, daemon=True, name="mic-probe"
        ).start()

    def _probe_mic_permission(self) -> None:
        """Background-thread mic probe — dispatches result to main thread."""
        import sounddevice as sd
        try:
            sd.rec(1600, samplerate=16000, channels=1, dtype='float32', blocking=True)
            logger.info("Microphone access granted")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionGranted:", None, False
            )
        except Exception as e:
            logger.warning("Mic permission not yet granted: %s", e)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionDenied:", None, False
            )

    def micPermissionGranted_(self, _sender) -> None:
        """Main-thread callback after mic probe succeeds."""
        self._mic_probe_in_flight = False
        self._setup_event_tap()

    def micPermissionDenied_(self, _sender) -> None:
        """Main-thread callback after mic probe fails — schedule retry."""
        from Foundation import NSTimer
        self._mic_probe_in_flight = False
        if self._menubar is not None:
            self._menubar.set_status_text("Grant mic access, then wait…")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            2.0, self, "retryMicPermission:", None, False
        )

    def retryMicPermission_(self, timer) -> None:
        """Retry mic access check — dispatches to background thread."""
        if self._mic_probe_in_flight:
            return
        self._mic_probe_in_flight = True
        threading.Thread(
            target=self._probe_mic_permission, daemon=True, name="mic-probe-retry"
        ).start()

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

        self._complete_event_tap_startup()

    def _complete_event_tap_startup(self) -> None:
        if self._menubar is not None:
            self._menubar.set_status_text("Loading models…")
        try:
            self._prepare_clients()
        except Exception as exc:
            self._models_ready = False
            self._warm_error = exc
            logger.exception("Model preparation failed")
            self._show_model_load_alert(exc)
            if self._menubar is not None:
                self._menubar.set_status_text("Model load failed — choose another model")
            return

        logger.info("spoke ready — hold spacebar to record")
        self._models_ready = True
        self._warm_error = None
        self._menubar.set_status_text("Ready — hold spacebar")

        # Warm TTS after Whisper is loaded to avoid Metal GPU contention
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            tts.warm()

    def retryEventTap_(self, timer) -> None:
        """Retry event tap installation."""
        if self._detector.install():
            self._complete_event_tap_startup()
        else:
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                2.0, self, "retryEventTap:", None, False
            )

    # ── hold callbacks (called on main thread) ──────────────

    def _on_hold_start(self) -> None:
        if not getattr(self, "_models_ready", True):
            logger.warning("Hold started before models were ready — ignoring")
            if self._menubar is not None:
                if getattr(self, "_warm_error", None) is not None:
                    self._menubar.set_status_text("Model load failed — choose another model")
                else:
                    self._menubar.set_status_text("Loading models…")
            return
        # If a command is actively streaming, cancel it
        if self._transcribing:
            logger.info("Hold during active stream — cancelling")
            self._transcription_token += 1
            self._transcribing = False
            # Fall through to start recording
        # Note: if command overlay is visible but finished, leave it up.
        # It will be dismissed if the user says nothing (empty recording)
        # or replaced if they send a new command.

        # Cancel any in-flight TTS playback
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            tts.cancel()

        # If recovery overlay is active, don't start recording.
        # The hold-end handler will check shift state and either:
        #   - Spacebar alone: retry Insert
        #   - Shift+Space: dismiss recovery
        self._verify_paste_text = None
        if getattr(self, "_recovery_text", None) is not None:
            self._recovery_hold_active = True
            logger.info("Hold started during recovery — waiting for release")
            return
        self._recovery_hold_active = False

        shift_at_press = getattr(self._detector, '_shift_at_press', False)
        logger.info("Hold started — recording (shift_at_press=%s)", shift_at_press)
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
        self._last_preview_text = ""
        self._preview_cancelled_on_release = False
        self._preview_session_token = getattr(self, "_preview_session_token", 0) + 1
        if getattr(self, "_preview_done", None) is not None:
            self._preview_done.clear()

        # Start the adaptive preview loop
        self._preview_active = True
        token = self._preview_session_token
        self._preview_thread = threading.Thread(
            target=self._preview_loop, args=(token,), daemon=True
        )
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
            _CAP_WARN_SECS = 3.0

            # Update countdown glow in the last 3 seconds
            warn_start = _MAX_RECORD_SECS - _CAP_WARN_SECS
            if elapsed >= warn_start and self._glow is not None:
                progress = min((elapsed - warn_start) / _CAP_WARN_SECS, 1.0)
                self._glow._cap_factor = 1.0 - progress  # linear

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

    def _preview_loop(self, token: int | None = None) -> None:
        """Background thread: preview transcription during recording.

        Uses streaming (KV-cache reuse) when the client supports it,
        falling back to full-buffer re-transcription otherwise.
        """
        use_streaming = getattr(self._preview_client, 'supports_streaming', False)

        if use_streaming:
            self._preview_loop_streaming(token)
        else:
            self._preview_loop_batch(token)

    def _preview_loop_streaming(self, token: int | None = None) -> None:
        """Streaming preview: feed incremental audio chunks, read state.text."""
        _FEED_INTERVAL = 0.15  # feed new audio every 150ms
        token = getattr(self, "_preview_session_token", 0) if token is None else token
        delegated_to_batch = False

        try:
            # Wait for some audio to accumulate before first feed
            time.sleep(0.4)

            try:
                with self._local_inference_context(self._preview_client):
                    self._preview_client.start_stream()
            except Exception:
                logger.exception("Failed to start streaming — falling back to batch")
                delegated_to_batch = True
                self._preview_loop_batch(token)
                return

            while self._preview_active:
                loop_start = time.monotonic()

                frames = self._capture.get_new_frames()

                try:
                    with self._local_inference_context(self._preview_client):
                        text = self._preview_client.feed(frames)
                except Exception:
                    logger.debug("Streaming feed failed", exc_info=True)
                    time.sleep(0.5)
                    continue

                if text and self._preview_active:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "previewTextUpdate:", {"token": token, "text": text}, False
                    )

                elapsed = time.monotonic() - loop_start
                remaining = _FEED_INTERVAL - elapsed
                if remaining > 0 and self._preview_active:
                    time.sleep(remaining)
        finally:
            if not delegated_to_batch:
                # Clean up streaming state if preview client differs from final client,
                # so the upstream library doesn't hold stale KV cache / session state.
                if (
                    self._preview_client is not self._client
                    and getattr(self._preview_client, "has_active_stream", False)
                ):
                    try:
                        if getattr(self, "_preview_cancelled_on_release", False):
                            cancel_stream = getattr(self._preview_client, "cancel_stream", None)
                            if callable(cancel_stream):
                                cancel_stream()
                        else:
                            self._preview_client.finish_stream()
                    except Exception:
                        logger.debug("Preview stream cleanup failed", exc_info=True)
                if getattr(self, "_preview_done", None) is not None:
                    self._preview_done.set()

    def _preview_loop_batch(self, token: int | None = None) -> None:
        """Batch preview: re-transcribe the full buffer each tick."""
        _MIN_INTERVAL = 0.2 if self._local_mode else 0.75
        _INITIAL_DELAY = 0.15 if self._local_mode else 0.3
        token = getattr(self, "_preview_session_token", 0) if token is None else token

        try:
            time.sleep(_INITIAL_DELAY)

            while self._preview_active:
                loop_start = time.monotonic()

                wav_bytes = self._capture.get_buffer()
                if not wav_bytes:
                    time.sleep(0.1 if self._local_mode else 0.2)
                    continue

                try:
                    with self._local_inference_context(self._preview_client):
                        text = self._preview_client.transcribe(wav_bytes)
                except Exception:
                    logger.debug("Preview transcription failed", exc_info=True)
                    time.sleep(0.5)
                    continue

                if text and self._preview_active:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "previewTextUpdate:",
                        {"token": token, "text": text},
                        False,
                    )

                elapsed = time.monotonic() - loop_start
                remaining = _MIN_INTERVAL - elapsed
                if remaining > 0 and self._preview_active:
                    time.sleep(remaining)
        finally:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.set()

    def previewTextUpdate_(self, payload) -> None:
        """Main thread: update the overlay with preview transcription text."""
        if isinstance(payload, dict):
            token = payload.get("token")
            text = payload.get("text", "")
        else:
            token = None
            text = payload
        if token is not None and token != getattr(self, "_preview_session_token", token):
            return
        if not self._preview_active:
            return
        self._last_preview_text = text
        if self._overlay is not None:
            self._overlay.set_text(text)

    def _on_hold_end(self, shift_held: bool = False) -> None:
        # Recovery overlay intercept: spacebar retries Insert, shift+space sends to assistant
        recovery_active = getattr(self, "_recovery_text", None) is not None
        if recovery_active or getattr(self, "_recovery_hold_active", False):
            self._recovery_hold_active = False
            if shift_held:
                if self._command_client is not None and self._recovery_text:
                    # Send the recovery text to the command pathway as an utterance
                    logger.info("Shift+space during recovery — sending to assistant: %r",
                                self._recovery_text[:50])
                    text = self._recovery_text
                    self._cancel_recovery()
                    self._send_text_as_command(text)
                else:
                    logger.info("Shift+space during recovery — dismissing (no command client)")
                    self._cancel_recovery()
                    if self._menubar is not None:
                        self._menubar.set_status_text("Ready — hold spacebar")
            else:
                logger.info("Spacebar during recovery — retrying Insert")
                self._recovery_retry_insert()
            return

        logger.info("Hold ended — %s", "command" if shift_held else "transcribing")
        self._preview_active = False
        self._preview_cancelled_on_release = True
        wav_bytes = self._capture.stop()

        # Short shift-hold (under 800ms of recording) = instant recall/dismiss
        # The user didn't have time to say anything meaningful
        elapsed = time.monotonic() - self._record_start_time if self._record_start_time else 0
        if shift_held and elapsed < 0.8:
            logger.info("Short shift-hold (%.0fms) — treating as instant", elapsed * 1000)
            wav_bytes = b""  # force the empty-audio path

        # Glow/dimmer: hide immediately for text insertion, persist for commands
        if not shift_held and self._glow is not None:
            self._glow.hide()
        if self._menubar is not None:
            self._menubar.set_recording(False)

        if not wav_bytes:
            logger.info("No audio — instant path (shift=%s)", shift_held)
            if self._overlay is not None:
                self._overlay.hide()
            if self._glow is not None:
                self._glow.hide()

            command_visible = (
                self._command_overlay is not None
                and getattr(self._command_overlay, '_visible', False)
            )

            if shift_held and not command_visible and self._command_client is not None:
                # Shift + empty recording + no overlay = recall last response
                history = self._command_client.history
                if history:
                    last_utterance, last_response = history[-1]
                    logger.info("Shift+empty — recalling last response")
                    if self._command_overlay is not None:
                        self._command_overlay.show()
                        self._command_overlay.set_utterance(last_utterance)
                        # Append the full response at once
                        for token in last_response:
                            self._command_overlay.append_token(token)
                        self._command_overlay.finish()
                else:
                    logger.info("Shift+empty — no history to recall")
            elif command_visible:
                # Empty recording with overlay visible = dismiss
                logger.info("Empty recording — dismissing command overlay")
                self._command_overlay.cancel_dismiss()

            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")
            return

        # Invalidate any in-flight transcription so its result is discarded
        self._transcription_token += 1
        token = self._transcription_token

        self._transcribing = True
        self._transcribe_start = time.monotonic()

        if shift_held and self._command_client is not None:
            # Command pathway: transcribe then send to OMLX
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing command…")
            thread = threading.Thread(
                target=self._command_transcribe_worker,
                args=(wav_bytes, token),
                daemon=True,
            )
        else:
            # Text pathway: transcribe and paste
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing…")
            thread = threading.Thread(
                target=self._transcribe_worker, args=(wav_bytes, token), daemon=True
            )
        thread.start()

    def _transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: finalize transcription and marshal result to main thread."""
        release_cutover = getattr(self, "_preview_cancelled_on_release", False)

        # Normally we wait for preview shutdown before the final pass. On release
        # cutover, stop blocking on preview wind-down and let the local inference
        # lock serialize any in-flight work that still needs to exit.
        if self._preview_thread is not None and not release_cutover:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            with self._local_inference_context(self._client):
                # If the preview client was streaming, finalize it for the final text
                # (this runs the tail refinement pass with the existing KV cache).
                if (
                    release_cutover
                    and getattr(self._client, 'supports_streaming', False)
                    and self._client is self._preview_client
                    and getattr(self._client, "has_active_stream", False)
                ):
                    cancel_stream = getattr(self._client, "cancel_stream", None)
                    if callable(cancel_stream):
                        cancel_stream()
                    text = self._client.transcribe(wav_bytes)
                elif (
                    getattr(self._client, 'supports_streaming', False)
                    and self._client is self._preview_client
                    and getattr(self._client, "has_active_stream", False)
                ):
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
            elapsed_ms = payload.get("elapsed_ms", 0)
            logger.info("Injected: %r (%.0fms)", text, elapsed_ms)
            self._inject_result_text(text, "Pasted!")
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
        if self._last_preview_text:
            logger.warning(
                "Final transcription failed — falling back to latest preview text"
            )
            self._inject_result_text(self._last_preview_text, "Pasted preview")
            return
        logger.error("Transcription failed — no text injected")
        if self._overlay is not None:
            self._overlay.hide()
        if self._menubar is not None:
            self._menubar.set_status_text("Error — try again")

    def hideOverlayAfterInject_(self, timer) -> None:
        """Hide the overlay after briefly showing the final transcription."""
        if self._overlay is not None:
            self._overlay.hide()

    def _recallLastResponse_(self, payload) -> None:
        """Main thread: recall the last command/response from history."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        if self._overlay is not None:
            self._overlay.hide()
        if self._glow is not None:
            self._glow.hide()

        if self._command_client is not None:
            history = self._command_client.history
            if history:
                last_utterance, last_response = history[-1]
                logger.info("Recalling last response: %r", last_utterance[:50])
                if self._command_overlay is not None:
                    self._command_overlay.show()
                    self._command_overlay.set_utterance(last_utterance)
                    for ch in last_response:
                        self._command_overlay.append_token(ch)
                    self._command_overlay.finish()
                if self._menubar is not None:
                    self._menubar.set_status_text("Ready — hold spacebar")
                return

        logger.info("No history to recall")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def _resetStatusAfterCancel_(self, timer) -> None:
        """Reset menubar status after a cancel."""
        if self._menubar is not None and not self._transcribing:
            self._menubar.set_status_text("Ready — hold spacebar")

    # ── command pathway ────────────────────────────────────

    def _send_text_as_command(self, text: str) -> None:
        """Send pre-transcribed text to the command pathway.

        Used when shift+space is pressed during recovery — the text is
        already transcribed, so we skip the audio transcription step and
        go straight to OMLX streaming.
        """
        self._transcription_token += 1
        token = self._transcription_token
        self._transcribing = True
        self._transcribe_start = time.monotonic()

        if self._menubar is not None:
            self._menubar.set_status_text("Sending to assistant…")

        # Dispatch utterance to the main thread for overlay setup
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandUtteranceReady:",
            {"token": token, "utterance": text},
            False,
        )

        # Stream the command in a background thread
        def _stream():
            try:
                for content_token in self._command_client.stream_command(text):
                    if token != self._transcription_token:
                        break  # stale
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "commandToken:",
                        {"token": token, "text": content_token},
                        False,
                    )
            except Exception:
                logger.exception("Command stream failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "commandFailed:", {"token": token, "error": "Command failed"}, False
                )
                return

            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandComplete:", {"token": token}, False
            )

        threading.Thread(target=_stream, daemon=True).start()

    def _command_transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe then send command to OMLX."""
        # Wait for preview loop to finish
        if self._preview_thread is not None:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        # Step 1: Transcribe the audio
        try:
            with self._local_inference_context(self._client):
                if (
                    getattr(self._client, 'supports_streaming', False)
                    and self._client is self._preview_client
                    and getattr(self._client, "has_active_stream", False)
                ):
                    utterance = self._client.finish_stream()
                else:
                    utterance = self._client.transcribe(wav_bytes)
        except Exception:
            logger.exception("Command transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandFailed:", {"token": token, "error": "Transcription failed"}, False
            )
            return

        if not utterance:
            # No speech with shift held = recall last response
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "_recallLastResponse:", {"token": token}, False
            )
            return

        transcribe_ms = (time.monotonic() - self._transcribe_start) * 1000
        logger.info("Command utterance: %r (%.0fms)", utterance, transcribe_ms)

        # Show the utterance in the input overlay before hiding it
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandUtteranceReady:",
            {"token": token, "utterance": utterance},
            False,
        )

        # Step 2: Stream the command response
        full_response = ""
        try:
            for content_token in self._command_client.stream_command(utterance):
                if token != self._transcription_token:
                    break  # stale
                full_response += content_token
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "commandToken:",
                    {"token": token, "text": content_token},
                    False,
                )
        except Exception:
            logger.exception("Command stream failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandFailed:", {"token": token, "error": "Command failed"}, False
            )
            return

        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "commandComplete:", {"token": token, "response": full_response}, False
        )

    def commandUtteranceReady_(self, payload: dict) -> None:
        """Main thread: show transcribed command, prepare output overlay."""
        if payload["token"] != self._transcription_token:
            return
        utterance = payload["utterance"]
        # Hide the input overlay
        if self._overlay is not None:
            self._overlay.hide()
        # Show the command overlay with the utterance as context
        if self._command_overlay is not None:
            self._command_overlay.show()
            self._command_overlay.set_utterance(utterance)
        self._command_first_token = True
        if self._menubar is not None:
            self._menubar.set_status_text("Thinking…")

    def commandToken_(self, payload: dict) -> None:
        """Main thread: append a streamed token to the command overlay."""
        if payload["token"] != self._transcription_token:
            return
        # First content token: invert the thinking timer and update status
        if getattr(self, "_command_first_token", False):
            self._command_first_token = False
            if self._command_overlay is not None:
                self._command_overlay.invert_thinking_timer()
            if self._menubar is not None:
                self._menubar.set_status_text("Responding…")
        if self._command_overlay is not None:
            self._command_overlay.append_token(payload["text"])

    def commandComplete_(self, payload: dict) -> None:
        """Main thread: command response finished streaming."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        if self._command_overlay is not None:
            self._command_overlay.finish()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")
        # Autoplay response via TTS if enabled — keep glow alive for amplitude
        response = payload.get("response", "")
        tts = getattr(self, "_tts_client", None)
        if response and tts is not None:
            # Reset glow state for TTS — the noise floor has adapted up during
            # command streaming and would eat the TTS signal otherwise
            if self._glow is not None:
                self._glow._noise_floor = 0.0
                self._glow._smoothed_amplitude = 0.0
            tts.speak_async(
                response,
                amplitude_callback=self._on_amplitude,
                done_callback=lambda: self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "ttsFinished:", None, False
                ),
            )
        elif self._glow is not None:
            self._glow.hide()

    def ttsFinished_(self, _) -> None:
        """Main thread: TTS playback ended — hide glow."""
        if self._glow is not None:
            self._glow.hide()

    def commandFailed_(self, payload: dict) -> None:
        """Main thread: show error in the command overlay, then fade."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        error = payload.get("error", "Unknown error")
        logger.error("Command pathway error: %s", error)
        if self._glow is not None:
            self._glow.hide()
        if self._overlay is not None:
            self._overlay.hide()
        # Show the error in the command overlay like a response
        if self._command_overlay is not None:
            if not self._command_overlay._visible:
                self._command_overlay.show()
            self._command_overlay.append_token("couldn't reach the model — try again in a moment")
            self._command_overlay.finish()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def verifyPaste_(self, timer) -> None:
        """Background OCR verification — confirm the pasted text appeared on screen."""
        text = getattr(self, "_verify_paste_text", None)
        if text is None:
            return

        attempt = getattr(self, "_verify_paste_attempt", 0)

        # Run OCR in background thread to avoid blocking the main thread
        import threading
        def _verify():
            from .paste_verify import capture_screen_text, text_appears_on_screen
            screen_text = capture_screen_text()
            found = text_appears_on_screen(text, screen_text)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "verifyPasteResult:",
                {"found": found, "text": text, "attempt": attempt},
                False,
            )
        threading.Thread(target=_verify, daemon=True).start()

    def verifyPasteResult_(self, payload) -> None:
        """Main thread: handle OCR verification result."""
        found = payload["found"]
        text = payload["text"]
        attempt = payload["attempt"]

        # If we've moved on (new recording started, or recovery already active),
        # discard this result.
        if getattr(self, "_verify_paste_text", None) != text:
            return

        is_retry = getattr(self, "_recovery_retry_pending", False)

        if found:
            logger.info("Paste verified by OCR (attempt %d)", attempt + 1)
            self._verify_paste_text = None
            if is_retry:
                # Retry succeeded — clear recovery state
                self._recovery_retry_pending = False
                self._clear_recovery_state()
                if self._menubar is not None:
                    self._menubar.set_status_text("Pasted!")
            return

        if attempt == 0:
            # First check failed — retry once at 300ms to give slow apps time
            logger.debug("Paste not verified on first check, retrying in 200ms")
            self._verify_paste_attempt = 1
            from Foundation import NSTimer
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.2, self, "verifyPaste:", None, False
            )
            return

        # Second check also failed
        self._verify_paste_text = None
        if is_retry:
            # Retry from recovery failed — bounce the overlay back
            logger.warning("Recovery retry not verified by OCR — bouncing back")
            self._recovery_retry_pending = False
            self._enter_recovery_mode(text)
        else:
            # Normal paste failed — enter recovery for the first time
            logger.warning("Paste not verified by OCR after %d attempts — entering recovery", attempt + 1)
            self._enter_recovery_mode(text)

    # ── helpers ─────────────────────────────────────────────

    _MODEL_OPTIONS = [
        ("mlx-community/whisper-tiny.en-mlx", "Tiny.en (float16)"),
        ("mlx-community/whisper-base.en-mlx", "Base.en (float16)"),
        ("mlx-community/whisper-base.en-mlx-8bit", "Base.en (8bit)"),
        ("mlx-community/whisper-small.en-mlx", "Small.en (float16)"),
        ("mlx-community/whisper-small.en-mlx-8bit", "Small.en (8bit)"),
        ("mlx-community/whisper-medium.en-mlx-4bit", "Medium.en (4bit)"),
        ("mlx-community/whisper-medium.en-mlx-8bit", "Medium.en (8bit)"),
        ("mlx-community/whisper-medium.en-mlx", "Medium.en (float16)"),
        ("mlx-community/whisper-large-v3-turbo-4bit", "v3 Large Turbo (4bit)"),
        ("mlx-community/whisper-large-v3-turbo-8bit", "v3 Large Turbo (8bit)"),
        ("mlx-community/whisper-large-v3-turbo", "v3 Large Turbo (float16)"),
        ("Qwen/Qwen3-ASR-0.6B", "Qwen3 ASR 0.6B (streaming)"),
    ]

    def _select_model(self, model_id):
        """Model picker. Pass None to get the menu list, or a model ID to switch."""
        if model_id is None:
            # Return (model_id, label, enabled) tuples — hide disallowed models
            return [
                (mid, label, True)
                for mid, label in self._MODEL_OPTIONS
                if self._model_allowed(mid)
            ]
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        self._apply_model_selection(
            preview_model=current_preview if model_id is None else model_id,
            transcription_model=current_transcription if model_id is None else model_id,
        )

    def _handle_model_menu_action(self, selection):
        """Menu callback for preview/transcription role-specific model choices."""
        if selection is None:
            state = {
                "transcription": {
                    "selected": getattr(
                        self, "_transcription_model_id", self._default_transcription_model()
                    ),
                    "models": self._select_model(None),
                },
                "preview": {
                    "selected": getattr(
                        self, "_preview_model_id", _DEFAULT_PREVIEW_MODEL
                    ),
                    "models": self._select_model(None),
                },
            }
            if self._local_whisper_controls_available():
                eager_eval_available = self._local_whisper_eager_eval_available()
                state["local_whisper"] = {
                    "title": "Local Whisper",
                    "items": [
                        (
                            "decode_timeout",
                            "Decode timeout guard (30s)",
                            getattr(
                                self,
                                "_local_whisper_decode_timeout",
                                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
                            )
                            is not None,
                            True,
                        ),
                        (
                            "eager_eval",
                            (
                                "Stability mode (eager eval)"
                                if eager_eval_available
                                else "Stability mode (eager eval) [mlx-whisper update needed]"
                            ),
                            (
                                getattr(
                                    self,
                                    "_local_whisper_eager_eval",
                                    _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
                                )
                                if eager_eval_available
                                else False
                            ),
                            eager_eval_available,
                        ),
                    ],
                }
            return state
        if not isinstance(selection, tuple) or len(selection) != 2:
            self._select_model(selection)
            return
        role, model_id = selection
        if role == "local_whisper":
            self._toggle_local_whisper_setting(model_id)
            return
        if role not in {"preview", "transcription"}:
            return
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        preview_model = current_preview
        transcription_model = current_transcription
        if role == "preview":
            preview_model = model_id
        else:
            transcription_model = model_id
        self._apply_model_selection(preview_model, transcription_model)

    def _apply_model_selection(self, preview_model: str, transcription_model: str) -> None:
        if not self._model_allowed(preview_model):
            logger.warning(
                "Preview model %s not available on this machine (%.0fGB RAM)",
                preview_model,
                _RAM_GB,
            )
            return
        if not self._model_allowed(transcription_model):
            logger.warning(
                "Transcription model %s not available on this machine (%.0fGB RAM)",
                transcription_model,
                _RAM_GB,
            )
            return
        current_preview = getattr(
            self,
            "_preview_model_id",
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or _DEFAULT_PREVIEW_MODEL,
        )
        current_transcription = getattr(
            self,
            "_transcription_model_id",
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or os.environ.get("SPOKE_WHISPER_MODEL")
            or self._default_transcription_model(),
        )
        if (
            preview_model == current_preview
            and transcription_model == current_transcription
        ):
            prefs = self._load_model_preferences()
            if (
                prefs.get("preview_model") != preview_model
                or prefs.get("transcription_model") != transcription_model
            ):
                logger.info(
                    "Repairing stale model preferences without relaunch: preview=%s, transcription=%s",
                    preview_model,
                    transcription_model,
                )
                self._save_model_preferences(preview_model, transcription_model)
            return
        logger.info(
            "Switching models (relaunching): preview %s → %s, transcription %s → %s",
            current_preview,
            preview_model,
            current_transcription,
            transcription_model,
        )
        if not self._save_model_preferences(preview_model, transcription_model):
            logger.warning(
                "Skipping relaunch because the new model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save model selection")
            return
        os.environ["SPOKE_PREVIEW_MODEL"] = preview_model
        os.environ["SPOKE_TRANSCRIPTION_MODEL"] = transcription_model
        if preview_model == transcription_model:
            os.environ["SPOKE_WHISPER_MODEL"] = preview_model
        else:
            os.environ.pop("SPOKE_WHISPER_MODEL", None)
        self._relaunch()

    def _default_transcription_model(self) -> str:
        if self._model_allowed(_DEFAULT_TRANSCRIPTION_MODEL):
            return _DEFAULT_TRANSCRIPTION_MODEL
        return _DEFAULT_PREVIEW_MODEL

    def _resolve_model_ids(self) -> tuple[str, str]:
        prefs = self._load_model_preferences()
        legacy_model = os.environ.get("SPOKE_WHISPER_MODEL")
        preview_model = (
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or prefs.get("preview_model")
            or legacy_model
            or _DEFAULT_PREVIEW_MODEL
        )
        transcription_model = (
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or prefs.get("transcription_model")
            or legacy_model
            or self._default_transcription_model()
        )
        return preview_model, transcription_model

    def _resolve_local_whisper_settings(self) -> tuple[float | None, bool]:
        prefs = self._load_local_whisper_preferences()
        decode_timeout_raw = os.environ.get("SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT")
        if decode_timeout_raw is None and "decode_timeout" in prefs:
            decode_timeout = prefs["decode_timeout"]
        else:
            if decode_timeout_raw is None:
                decode_timeout_raw = _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
            decode_timeout = self._coerce_decode_timeout_setting(decode_timeout_raw)

        eager_eval_raw = os.environ.get("SPOKE_LOCAL_WHISPER_EAGER_EVAL")
        if eager_eval_raw is None:
            if "eager_eval" in prefs:
                eager_eval_raw = prefs["eager_eval"]
            else:
                eager_eval_raw = _DEFAULT_LOCAL_WHISPER_EAGER_EVAL
        eager_eval = self._coerce_eager_eval_setting(eager_eval_raw)
        return decode_timeout, eager_eval

    def _load_local_whisper_preferences(self) -> dict:
        raw_prefs = self._load_preferences()
        prefs = {}
        if "local_whisper_decode_timeout" in raw_prefs:
            prefs["decode_timeout"] = raw_prefs["local_whisper_decode_timeout"]
        if "local_whisper_eager_eval" in raw_prefs:
            prefs["eager_eval"] = raw_prefs["local_whisper_eager_eval"]
        return prefs

    def _preferences_path(self) -> Path:
        override = os.environ.get("SPOKE_MODEL_PREFERENCES_PATH")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Library/Application Support/Spoke/model_preferences.json"

    def _load_preferences(self) -> dict:
        path = self._preferences_path()
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except Exception:
            logger.warning("Failed to read model preferences from %s", path, exc_info=True)
            return {}

    def _load_model_preferences(self) -> dict:
        prefs = self._load_preferences()
        return {
            "preview_model": prefs.get("preview_model"),
            "transcription_model": prefs.get("transcription_model"),
        }

    def _save_model_preferences(
        self, preview_model: str, transcription_model: str
    ) -> bool:
        payload = self._load_preferences()
        payload["preview_model"] = preview_model
        payload["transcription_model"] = transcription_model
        return self._save_preferences(payload)

    def _save_local_whisper_preferences(
        self, decode_timeout: float | None, eager_eval: bool
    ) -> bool:
        payload = self._load_preferences()
        payload["local_whisper_decode_timeout"] = decode_timeout
        payload["local_whisper_eager_eval"] = eager_eval
        return self._save_preferences(payload)

    def _save_preferences(self, payload: dict) -> bool:
        path = self._preferences_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(path)
            return True
        except Exception:
            logger.warning("Failed to save model preferences to %s", path, exc_info=True)
            return False

    def _get_client(self, whisper_url: str, model_id: str):
        cache_key = (whisper_url, model_id)
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]
        client = self._build_client(whisper_url, model_id)
        self._client_cache[cache_key] = client
        return client

    def _build_client(self, whisper_url: str, model_id: str):
        if whisper_url:
            logger.info("Using sidecar transcription: %s (%s)", whisper_url, model_id)
            return TranscriptionClient(base_url=whisper_url, model=model_id)
        if model_id.startswith("Qwen/"):
            logger.info("Using local Qwen3 ASR: %s", model_id)
            return LocalQwenClient(model=model_id)
        logger.info("Using local transcription: %s", model_id)
        return LocalTranscriptionClient(
            model=model_id,
            decode_timeout=getattr(
                self,
                "_local_whisper_decode_timeout",
                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
            ),
            eager_eval=getattr(
                self,
                "_local_whisper_eager_eval",
                _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
            ),
        )

    def _prepare_clients(self) -> None:
        seen_clients = []
        ordered_clients = [
            ("transcription", getattr(self, "_transcription_model_id", None), self._client),
            ("preview", getattr(self, "_preview_model_id", None), self._preview_client),
        ]
        for role, model_id, client in ordered_clients:
            if client is None or any(existing is client for existing in seen_clients):
                continue
            seen_clients.append(client)
            if model_id is not None and not self._model_allowed(model_id):
                raise RuntimeError(
                    f"{role.title()} model {model_id} is not supported on this machine."
                )
            prepare = getattr(client, "prepare", None)
            if callable(prepare):
                with self._local_inference_context(client):
                    prepare()

    def _close_clients(self) -> None:
        seen_clients = []
        for client in list(getattr(self, "_client_cache", {}).values()) + [
            getattr(self, "_client", None),
            getattr(self, "_preview_client", None),
        ]:
            if client is None or any(existing is client for existing in seen_clients):
                continue
            seen_clients.append(client)
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _local_whisper_controls_available(self) -> bool:
        if not getattr(self, "_local_mode", False):
            return False
        model_ids = [
            getattr(
                self,
                "_preview_model_id",
                os.environ.get("SPOKE_PREVIEW_MODEL")
                or os.environ.get("SPOKE_WHISPER_MODEL")
                or _DEFAULT_PREVIEW_MODEL,
            ),
            getattr(
                self,
                "_transcription_model_id",
                os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
                or os.environ.get("SPOKE_WHISPER_MODEL")
                or self._default_transcription_model(),
            ),
        ]
        return any(model_id and not model_id.startswith("Qwen/") for model_id in model_ids)

    def _local_whisper_eager_eval_available(self) -> bool:
        return self._local_whisper_controls_available() and supports_eager_eval()

    def _toggle_local_whisper_setting(self, setting: str) -> None:
        if not self._local_whisper_controls_available():
            return
        if setting == "eager_eval" and not self._local_whisper_eager_eval_available():
            logger.info(
                "Ignoring eager_eval toggle because the installed mlx-whisper build does not support it yet"
            )
            return
        decode_timeout = getattr(
            self,
            "_local_whisper_decode_timeout",
            _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
        )
        eager_eval = getattr(
            self,
            "_local_whisper_eager_eval",
            _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
        )
        if setting == "decode_timeout":
            decode_timeout = (
                None
                if decode_timeout is not None
                else _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
            )
        elif setting == "eager_eval":
            eager_eval = not eager_eval
        else:
            return
        self._apply_local_whisper_settings(decode_timeout, eager_eval)

    def _apply_local_whisper_settings(
        self, decode_timeout: float | None, eager_eval: bool
    ) -> None:
        current_timeout = getattr(
            self,
            "_local_whisper_decode_timeout",
            _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
        )
        current_eager = getattr(
            self,
            "_local_whisper_eager_eval",
            _DEFAULT_LOCAL_WHISPER_EAGER_EVAL,
        )
        if decode_timeout == current_timeout and eager_eval == current_eager:
            return
        logger.info(
            "Switching local Whisper settings (relaunching): decode_timeout %s -> %s, eager_eval %s -> %s",
            current_timeout,
            decode_timeout,
            current_eager,
            eager_eval,
        )
        self._local_whisper_decode_timeout = decode_timeout
        self._local_whisper_eager_eval = eager_eval
        self._save_local_whisper_preferences(decode_timeout, eager_eval)
        os.environ["SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT"] = self._format_decode_timeout_env(
            decode_timeout
        )
        os.environ["SPOKE_LOCAL_WHISPER_EAGER_EVAL"] = "1" if eager_eval else "0"
        self._relaunch()

    @staticmethod
    def _coerce_decode_timeout_setting(value) -> float | None:
        if value is None:
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        if isinstance(value, (int, float)):
            return None if value <= 0 else float(value)
        normalized = str(value).strip().lower()
        if normalized in {"", "default"}:
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        if normalized in {"off", "none", "false"}:
            return None
        try:
            timeout = float(normalized)
        except ValueError:
            logger.warning(
                "Invalid SPOKE_LOCAL_WHISPER_DECODE_TIMEOUT=%r; using default %.1fs",
                value,
                _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT,
            )
            return _DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT
        return None if timeout <= 0 else timeout

    @staticmethod
    def _coerce_eager_eval_setting(value) -> bool:
        if value is None:
            return _DEFAULT_LOCAL_WHISPER_EAGER_EVAL
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _format_decode_timeout_env(value: float | None) -> str:
        if value is None:
            return "off"
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)

    def _relaunch(self) -> None:
        self._detector.uninstall()
        self._preview_active = False
        self._close_clients()
        os.execv(sys.executable, [sys.executable, "-m", "spoke"])

    def _local_inference_context(self, client):
        lock = getattr(self, "_local_inference_lock", None)
        if lock is None or isinstance(client, TranscriptionClient):
            return nullcontext()
        return lock

    def _inject_result_text(self, text: str, status_text: str) -> None:
        # Remove the overlay from screen so it doesn't appear in the
        # verification screenshot or mask the focused element.
        if self._overlay is not None:
            self._overlay.order_out()

        # Always attempt the paste immediately — the user perceives no delay.
        # Save clipboard state before inject_text overwrites it, in case we
        # need to enter recovery mode after OCR verification.
        self._pre_paste_clipboard = save_pasteboard()

        def _on_clipboard_restored():
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

        inject_text(text, on_restored=_on_clipboard_restored)
        if self._menubar is not None:
            self._menubar.set_status_text(status_text)

        # Schedule background OCR verification to confirm the paste landed.
        # If it didn't, we'll enter recovery mode.
        self._verify_paste_text = text
        self._verify_paste_attempt = 0
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.15, self, "verifyPaste:", None, False
        )

    def _enter_recovery_mode(self, text: str) -> None:
        """Show the recovery overlay with Dismiss / Insert / Clipboard buttons."""
        # Use pre-saved clipboard if available (from _inject_result_text).
        # _pre_paste_clipboard is _NOT_CAPTURED when not set, and None when
        # the clipboard was empty — both are valid states. Only fall back to
        # save_pasteboard() if we never captured it at all.
        pre = getattr(self, "_pre_paste_clipboard", _NOT_CAPTURED)
        if pre is not _NOT_CAPTURED:
            self._recovery_saved_clipboard = pre
        else:
            self._recovery_saved_clipboard = save_pasteboard()
        self._pre_paste_clipboard = _NOT_CAPTURED
        self._recovery_text = text
        self._recovery_clipboard_state = "idle"

        # Put transcription on pasteboard for manual paste
        set_pasteboard_only(text)

        if self._overlay is not None:
            self._overlay.show_recovery(
                text,
                on_dismiss=self._on_recovery_dismiss,
                on_insert=self._on_recovery_insert,
                on_clipboard_toggle=self._on_recovery_clipboard_toggle,
            )
        if self._menubar is not None:
            self._menubar.set_status_text("No text field — ⌘V to paste")

    def _recovery_retry_insert(self) -> None:
        """Spacebar retry: attempt Insert from recovery, OCR verify after.

        Always attempts the paste (just like the normal flow), then uses
        OCR to verify. If it didn't land, bounces the overlay back.
        If it landed, clears recovery state.
        """
        if self._recovery_text is None:
            return

        # No focused element at all → bounce immediately, don't waste time
        if not has_focused_text_input():
            logger.info("Recovery retry — no focused element, bouncing")
            if self._overlay is not None:
                self._overlay.bounce()
            return

        text = self._recovery_text or ""

        # Dismiss overlay so it doesn't appear in the OCR screenshot
        if self._overlay is not None:
            self._overlay.order_out()

        # Paste
        inject_text(text)

        # OCR verify — reuse the same verification pipeline
        self._verify_paste_text = text
        self._verify_paste_attempt = 0
        # Override the verify result handler to bounce-back on failure
        # instead of entering recovery (we're already in recovery)
        self._recovery_retry_pending = True
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.15, self, "verifyPaste:", None, False
        )

    def _on_recovery_dismiss(self) -> None:
        """Dismiss button: restore clipboard and hide overlay."""
        if self._recovery_text is None:
            return  # already dismissed
        restore_pasteboard(self._recovery_saved_clipboard)
        if self._overlay is not None:
            self._overlay.dismiss_recovery()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")
        self._clear_recovery_state()

    def _on_recovery_insert(self) -> None:
        """Insert button: same as spacebar retry — check first, bounce on failure."""
        self._recovery_retry_insert()

    def doRecoveryInsert_(self, timer) -> None:
        """Delayed paste after recovery Insert — gives target app time to refocus."""
        pending = getattr(self, "_recovery_pending_insert", None)
        if pending is None:
            return
        text, saved = pending
        self._recovery_pending_insert = None

        if not has_focused_text_input():
            # Still no text field — the target app didn't refocus in time.
            # Re-enter recovery mode rather than losing the text.
            logger.warning("Insert failed — target app did not refocus, re-entering recovery")
            self._enter_recovery_mode(text)
            return

        def _on_restored():
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

        inject_text(text, on_restored=_on_restored)
        if self._menubar is not None:
            self._menubar.set_status_text("Pasted!")
        self._clear_recovery_state()

    def _on_recovery_clipboard_toggle(self) -> None:
        """Clipboard button: toggle between transcription and original clipboard."""
        if self._recovery_clipboard_state == "idle":
            # First click: transcription is already on clipboard from _enter_recovery_mode
            self._recovery_clipboard_state = "transcription_on_clipboard"
            # Show preview of what was clobbered
            old_text = self._get_clipboard_preview_text(self._recovery_saved_clipboard)
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(old_text)
        elif self._recovery_clipboard_state == "transcription_on_clipboard":
            # Restore old clipboard, show transcription preview
            restore_pasteboard(self._recovery_saved_clipboard)
            self._recovery_clipboard_state = "original_on_clipboard"
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(self._recovery_text or "")
        elif self._recovery_clipboard_state == "original_on_clipboard":
            # Put transcription back on clipboard, show old preview
            set_pasteboard_only(self._recovery_text or "")
            self._recovery_clipboard_state = "transcription_on_clipboard"
            old_text = self._get_clipboard_preview_text(self._recovery_saved_clipboard)
            if self._overlay is not None:
                self._overlay.set_clipboard_preview(old_text)

    def _cancel_recovery(self) -> None:
        """Cancel recovery mode if active. Restores original clipboard."""
        if getattr(self, "_recovery_text", None) is not None:
            restore_pasteboard(getattr(self, "_recovery_saved_clipboard", None))
            if self._overlay is not None:
                self._overlay.dismiss_recovery()
            self._clear_recovery_state()

    def _clear_recovery_state(self) -> None:
        """Reset recovery state fields."""
        self._recovery_saved_clipboard = None
        self._recovery_text = None
        self._recovery_clipboard_state = "idle"
        self._recovery_pending_insert = None

    @staticmethod
    def _get_clipboard_preview_text(saved: list[tuple[str, bytes]] | None) -> str:
        """Extract a text preview from saved clipboard contents."""
        if not saved:
            return "(empty)"
        for ptype, data in saved:
            if "utf8" in ptype.lower() or "string" in ptype.lower() or "text" in ptype.lower():
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    pass
        # Non-text content
        return "(non-text)"

    @staticmethod
    def _model_allowed(model_id: str) -> bool:
        """Guard: whisper-large-v3-turbo requires >= 36GB RAM."""
        if "large-v3-turbo" in model_id and _RAM_GB < 36:
            return False
        return True

    def _quit(self) -> None:
        self._detector.uninstall()
        self._preview_active = False
        self._close_clients()
        NSApp.terminate_(None)

    def _show_accessibility_alert(self) -> None:
        """Show a dialog explaining the Accessibility permission requirement."""
        alert = NSAlert.new()
        alert.setMessageText_("Accessibility Permission Required")
        alert.setInformativeText_(
            "Spoke needs Accessibility access to detect spacebar holds.\n\n"
            "Go to System Settings → Privacy & Security → Accessibility "
            "and enable access for your terminal app (Terminal, iTerm2, etc.).\n\n"
            "Then relaunch Spoke."
        )
        alert.addButtonWithTitle_("OK")
        # Temporarily become a regular app so the alert is visible
        NSApp.setActivationPolicy_(1)  # NSApplicationActivationPolicyRegular
        alert.runModal()

    def _show_model_load_alert(self, error: Exception) -> None:
        alert = NSAlert.new()
        alert.setMessageText_("Model Load Failed")
        alert.setInformativeText_(
            "Spoke could not prepare the selected models.\n\n"
            f"{error}\n\n"
            "Choose a different model from the menu and Spoke will relaunch."
        )
        alert.addButtonWithTitle_("OK")
        NSApp.setActivationPolicy_(1)
        alert.runModal()


def _acquire_instance_lock() -> None:
    """Single-instance guard — kill any stuck old instance and take the lock."""
    import fcntl
    import signal as sig
    import time

    lock_path = os.path.expanduser("~/Library/Logs/.spoke.lock")
    current_pid = os.getpid()
    parent_pid = os.getppid()
    logger.info(
        "Single-instance guard starting (pid=%d ppid=%d lock=%s)",
        current_pid,
        parent_pid,
        lock_path,
    )
    lock_file = open(lock_path, "a+")
    lock_file.seek(0)
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info(
            "Single-instance guard found an existing lock holder (pid=%d lock=%s)",
            current_pid,
            lock_path,
        )
        try:
            lock_file.seek(0)
            old_pid = int(lock_file.read().strip())
            logger.warning(
                "Single-instance guard sending SIGTERM to prior pid=%d from pid=%d",
                old_pid,
                current_pid,
            )
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
    lock_file.write(str(current_pid))
    lock_file.flush()
    logger.info(
        "Single-instance guard acquired lock (pid=%d lock=%s)",
        current_pid,
        lock_path,
    )
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

    delegate = SpokeAppDelegate.alloc().init()
    app.setDelegate_(delegate)

    # Clean shutdown on SIGTERM — uninstall event tap and remove status item
    # before dying so we don't leave a zombie tap or ghost menu bar icon
    def _handle_sigterm(signum, frame):
        lock_pid = "unavailable"
        try:
            with open(os.path.expanduser("~/Library/Logs/.spoke.lock"), encoding="utf-8") as lock_file:
                lock_pid = lock_file.read().strip() or "empty"
        except OSError:
            pass
        logger.info(
            "Received SIGTERM — cleaning up (pid=%d ppid=%d cwd=%s lock_pid=%s)",
            os.getpid(),
            os.getppid(),
            os.getcwd(),
            lock_pid,
        )
        delegate._detector.uninstall()
        if delegate._menubar is not None:
            delegate._menubar.cleanup()
        NSApp.terminate_(None)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    from PyObjCTools import AppHelper

    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
