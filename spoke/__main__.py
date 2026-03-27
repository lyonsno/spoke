"""Entry point for spoke — macOS global hold-to-dictate.

Run with:  uv run spoke
    or:    uv run python -m spoke

Configure via environment variables:
    SPOKE_WHISPER_URL    Sidecar server URL (optional — if unset, uses local MLX Whisper)
    SPOKE_WHISPER_MODEL  Model name (default: mlx-community/whisper-large-v3-turbo)
    SPOKE_HOLD_MS        Hold threshold in ms (default: 250, must be > 0)
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
from .glow import GlowOverlay
from .inject import inject_text
from .input_tap import SpacebarHoldDetector
from .menubar import MenuBarIcon
from .overlay import TranscriptionOverlay
from .transcribe import TranscriptionClient
from .transcribe_local import LocalTranscriptionClient
from .transcribe_qwen import LocalQwenClient

logger = logging.getLogger(__name__)

_DEFAULT_PREVIEW_MODEL = "mlx-community/whisper-medium.en-mlx-8bit"
_DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-large-v3-turbo"


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
        hold_ms_raw = os.environ.get("SPOKE_HOLD_MS", "250")
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
        self._last_preview_text = ""
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
            if not delegated_to_batch and getattr(self, "_preview_done", None) is not None:
                self._preview_done.set()

    def _preview_loop_batch(self, token: int | None = None) -> None:
        """Batch preview: re-transcribe the full buffer each tick."""
        _MIN_INTERVAL = 0.5 if self._local_mode else 0.75
        _INITIAL_DELAY = 0.4 if self._local_mode else 0.3
        token = getattr(self, "_preview_session_token", 0) if token is None else token

        try:
            time.sleep(_INITIAL_DELAY)

            while self._preview_active:
                loop_start = time.monotonic()

                wav_bytes = self._capture.get_buffer()
                if not wav_bytes:
                    time.sleep(0.2)
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
        # Wait for preview loop to finish so we don't hit the model concurrently.
        if self._preview_thread is not None:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            with self._local_inference_context(self._client):
                # If the preview client was streaming, finalize it for the final text
                # (this runs the tail refinement pass with the existing KV cache).
                if (
                    getattr(self._client, 'supports_streaming', False)
                    and self._client is self._preview_client
                    and getattr(self._client, "_stream_state", None) is not None
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

    # ── helpers ─────────────────────────────────────────────

    _MODEL_OPTIONS = [
        ("mlx-community/whisper-medium.en-mlx-4bit", "Medium.en (4bit)"),
        ("mlx-community/whisper-medium.en-mlx-8bit", "Medium.en (8bit)"),
        ("mlx-community/whisper-medium.en-mlx", "Medium.en (bf16)"),
        ("mlx-community/whisper-large-v3-turbo-4bit", "v3 Large Turbo (4bit)"),
        ("mlx-community/whisper-large-v3-turbo-8bit", "v3 Large Turbo (8bit)"),
        ("mlx-community/whisper-large-v3-turbo", "v3 Large Turbo (bf16)"),
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
            return {
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
        if not isinstance(selection, tuple) or len(selection) != 2:
            self._select_model(selection)
            return
        role, model_id = selection
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
            return
        logger.info(
            "Switching models (relaunching): preview %s → %s, transcription %s → %s",
            current_preview,
            preview_model,
            current_transcription,
            transcription_model,
        )
        self._save_model_preferences(preview_model, transcription_model)
        os.environ["SPOKE_PREVIEW_MODEL"] = preview_model
        os.environ["SPOKE_TRANSCRIPTION_MODEL"] = transcription_model
        if preview_model == transcription_model:
            os.environ["SPOKE_WHISPER_MODEL"] = preview_model
        else:
            os.environ.pop("SPOKE_WHISPER_MODEL", None)
        self._detector.uninstall()
        self._preview_active = False
        self._close_clients()
        os.execv(sys.executable, [sys.executable, "-m", "spoke"])

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

    def _preferences_path(self) -> Path:
        override = os.environ.get("SPOKE_MODEL_PREFERENCES_PATH")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Library/Application Support/Spoke/model_preferences.json"

    def _load_model_preferences(self) -> dict:
        path = self._preferences_path()
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except Exception:
            logger.warning("Failed to read model preferences from %s", path, exc_info=True)
            return {}

    def _save_model_preferences(
        self, preview_model: str, transcription_model: str
    ) -> None:
        path = self._preferences_path()
        payload = {
            "preview_model": preview_model,
            "transcription_model": transcription_model,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
        except Exception:
            logger.warning("Failed to save model preferences to %s", path, exc_info=True)

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
        return LocalTranscriptionClient(model=model_id)

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

    def _local_inference_context(self, client):
        lock = getattr(self, "_local_inference_lock", None)
        if lock is None or isinstance(client, TranscriptionClient):
            return nullcontext()
        return lock

    def _inject_result_text(self, text: str, status_text: str) -> None:
        def _on_clipboard_restored():
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

        if self._overlay is not None:
            self._overlay.set_text(text)

        inject_text(text, on_restored=_on_clipboard_restored)
        if self._menubar is not None:
            self._menubar.set_status_text(status_text)

        if self._overlay is not None:
            from Foundation import NSTimer

            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.5, self, "hideOverlayAfterInject:", None, False
            )

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

    delegate = SpokeAppDelegate.alloc().init()
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
