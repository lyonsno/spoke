"""Entry point for spoke — macOS global hold-to-dictate.

Run with:  uv run spoke
    or:    uv run python -m spoke

Configure via environment variables:
    SPOKE_WHISPER_URL          Sidecar server URL (optional — if unset, uses local MLX Whisper)
    SPOKE_WHISPER_MODEL        Model name (default: mlx-community/whisper-large-v3-turbo)
    SPOKE_HOLD_MS              Hold threshold in ms (default: 200, must be > 0)
    SPOKE_RESTORE_DELAY_MS     Pasteboard restore delay in ms (default: 1000)
    SPOKE_PARAKEET_MODEL_DIR   Path to FluidInference/parakeet-ctc-110m-coreml clone dir
                               (enables Parakeet CoreML/ANE preview model in the menu)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
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
from .command import CommandClient, _DEFAULT_COMMAND_MODEL, _DEFAULT_COMMAND_URL
from .focus_check import has_focused_text_input
from .scene_capture import SceneCaptureCache
from .tool_dispatch import execute_tool, get_tool_schemas
from .glow import GlowOverlay
from .inject import inject_text, save_pasteboard, restore_pasteboard, set_pasteboard_only
from .input_tap import SpacebarHoldDetector
from .launch_targets import (
    current_launch_target_id,
    iter_launch_targets,
    save_selected_launch_target,
)
from .menubar import MenuBarIcon
from .overlay import TranscriptionOverlay
from .transcribe import TranscriptionClient
from .transcribe_local import LocalTranscriptionClient, supports_eager_eval
from .transcribe_parakeet import ParakeetCoreMLClient, _PARAKEET_MODEL_ID
from .transcribe_qwen import LocalQwenClient
from .tts import TTSClient

logger = logging.getLogger(__name__)

_DEFAULT_PREVIEW_MODEL = "mlx-community/whisper-base.en-mlx-8bit"
_DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-medium.en-mlx-8bit"
_DEFAULT_LOCAL_WHISPER_DECODE_TIMEOUT = 30.0
_DEFAULT_LOCAL_WHISPER_EAGER_EVAL = False
_DEFAULT_COMMAND_MODEL_DIR = Path.home() / ".lmstudio" / "models"
_CURATED_LOCAL_COMMAND_MODEL_IDS = [
    "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-6bit",
    "mlx-community/Qwen3-4B-Thinking-2507-8bit",
    "lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit",
    "lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-8bit",
    "alexgusevski/LFM2.5-1.2B-Nova-Function-Calling-mlx",
]

_TTS_MODELS = [
    ("mlx-community/Voxtral-4B-TTS-2603-mlx-4bit", "Voxtral 4B (4-bit)"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "Qwen3-TTS 1.7B CustomVoice (8-bit)"),
    ("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "Qwen3-TTS 0.6B CustomVoice (8-bit)"),
    ("mlx-community/VibeVoice-Realtime-0.5B-fp16", "VibeVoice 0.5B Realtime (fp16)"),
    ("mlx-community/Kokoro-82M-bf16", "Kokoro 82M (bf16)"),
]

_NOT_CAPTURED = object()  # sentinel for _pre_paste_clipboard


def _get_ram_gb() -> float:
    """Return system RAM in GB via sysctl."""
    import subprocess
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


_MIN_RAM_GB_FOR_V3_TURBO = 16.0
_MIN_RAM_GB_FOR_UNCAPPED_RECORDING = 16.0
_LOW_RAM_RECORDING_CAP_SECS = 20.0


def _max_record_secs_for_ram(ram_gb: float) -> float | None:
    """Return the recording cap for local inference on this RAM tier."""
    if ram_gb < _MIN_RAM_GB_FOR_UNCAPPED_RECORDING:
        return _LOW_RAM_RECORDING_CAP_SECS
    return None


def _iter_local_command_model_ids(model_dir: Path) -> list[str]:
    """Return a curated list of installed MLX model ids from a local model directory."""
    if not model_dir.is_dir():
        return []

    discovered: set[str] = set()
    for child in sorted(model_dir.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        child_entries = sorted(child.iterdir(), key=lambda path: path.name.lower())
        if _is_local_command_model_leaf(child):
            discovered.add(child.name)
            continue
        for grandchild in child_entries:
            if _is_local_command_model_leaf(grandchild):
                discovered.add(f"{child.name}/{grandchild.name}")
    return [
        model_id
        for model_id in _CURATED_LOCAL_COMMAND_MODEL_IDS
        if model_id in discovered
    ]


def _is_local_command_model_leaf(model_path: Path) -> bool:
    """Return True when a leaf directory looks like an installed MLX model."""
    if not model_path.is_dir():
        return False

    file_names = {
        entry.name
        for entry in model_path.iterdir()
        if entry.is_file()
    }
    if "config.json" not in file_names or "tokenizer.json" not in file_names:
        return False
    if "model.safetensors" in file_names or "model.safetensors.index.json" in file_names:
        return True
    return any(name.endswith(".safetensors") for name in file_names)
# Recording cap: let 16GB+ local boxes that can run v3 turbo record freely.
# Keep the 20s cap only for smaller local machines. No cap in sidecar mode.
_RAM_GB = _get_ram_gb()
_MAX_RECORD_SECS: float | None = _max_record_secs_for_ram(_RAM_GB)


@dataclass
class TrayEntry:
    """A tray entry with minimal ownership/provenance metadata."""

    text: str
    owner: str = "user"
    acknowledged: bool = True

    def __eq__(self, other):
        if isinstance(other, TrayEntry):
            return (
                self.text == other.text
                and self.owner == other.owner
                and self.acknowledged == other.acknowledged
            )
        if isinstance(other, str):
            return self.text == other
        return NotImplemented

    @property
    def display_owner(self) -> str:
        if self.owner == "assistant" and not self.acknowledged:
            return "assistant"
        return "user"


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
        # Wire tray callbacks on the detector
        self._detector._on_shift_tap = self._on_tray_shift_tap
        self._detector._on_shift_tap_during_hold = self._on_tray_navigate_up
        self._detector._on_shift_tap_idle = self._on_audio_shift_tap
        self._detector._on_enter_pressed = self._on_tray_enter_pressed
        self._detector._on_tray_delete = self._on_tray_delete_gesture
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
        self._warmup_in_flight = False
        self._hold_rejected_during_warmup = False
        self._mic_probe_in_flight = False

        # Command pathway — always enabled, defaults to localhost:8001
        command_url = os.environ.get("SPOKE_COMMAND_URL", _DEFAULT_COMMAND_URL)
        if command_url:
            self._command_model_id = (
                os.environ.get("SPOKE_COMMAND_MODEL")
                or self._load_command_model_preference()
                or _DEFAULT_COMMAND_MODEL
            )
            self._command_client = CommandClient(model=self._command_model_id)
            self._command_model_options = self._seed_command_model_options(
                self._command_model_id
            )
            self._command_models_refresh_in_flight = False
            self._command_overlay: TranscriptionOverlay | None = None
            self._scene_cache = SceneCaptureCache(max_captures=10)
            self._tool_schemas = get_tool_schemas()
            logger.info(
                "Command pathway enabled: %s (%s)",
                command_url,
                self._command_model_id,
            )
        else:
            self._command_client = None
            self._command_model_id = None
            self._command_model_options = []
            self._command_models_refresh_in_flight = False
            self._command_overlay = None
            self._scene_cache = None
            self._tool_schemas = None

        # TTS autoplay — initialized if SPOKE_TTS_VOICE is set.
        # Preference-based model override takes priority over env var.
        tts_model_pref = self._load_preferences().get("tts_model")
        if tts_model_pref:
            os.environ["SPOKE_TTS_MODEL"] = tts_model_pref
        self._tts_client = TTSClient.from_env(gpu_lock=self._local_inference_lock)
        self._command_tool_used_tts = False
        if self._tts_client is not None:
            logger.info("TTS enabled: model=%s voice=%s", self._tts_client._model_id, self._tts_client._voice)

        # Tray state — speech-native stacked clipboard
        self._tray_stack: list[TrayEntry | str] = []
        self._tray_index: int = 0
        self._tray_active: bool = False
        self._tray_tool_result: dict | None = None

        # Recovery mode state (implementation detail of tray)
        # _NOT_CAPTURED sentinel distinguishes "not captured yet" from
        # "captured but clipboard was empty (None)".
        self._pre_paste_clipboard: list[tuple[str, bytes]] | None | object = _NOT_CAPTURED
        self._verify_paste_text: str | None = None
        self._verify_paste_attempt: int = 0
        self._result_pending_inject = None
        self._recovery_saved_clipboard: list[tuple[str, bytes]] | None = None
        self._recovery_text: str | None = None
        self._recovery_clipboard_state: str = "idle"
        self._recovery_pending_insert = None
        self._recovery_hold_active: bool = False
        self._recovery_retry_pending: bool = False

        if self._local_mode and _MAX_RECORD_SECS is not None:
            logger.info(
                "RAM %.0fGB < %.0fGB — recording capped at %.0fs to avoid Metal crashes",
                _RAM_GB,
                _MIN_RAM_GB_FOR_UNCAPPED_RECORDING,
                _MAX_RECORD_SECS,
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
            self._refresh_command_model_options_async()

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
            self._run_mic_permission_probe(sd)
            logger.info("Microphone access granted")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "micPermissionGranted:", None, False
            )
        except Exception as exc:
            if self._is_permission_probe_denial(exc):
                logger.warning("Mic permission not yet granted: %s", exc)
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micPermissionDenied:", None, False
                )
                return

            logger.warning(
                "Mic probe hit audio runtime failure — resetting PortAudio",
                exc_info=True,
            )
            self._reset_portaudio_probe(sd)

            try:
                self._run_mic_permission_probe(sd)
                logger.info("Microphone access granted after PortAudio reset")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micPermissionGranted:", None, False
                )
            except Exception as retry_exc:
                if self._is_permission_probe_denial(retry_exc):
                    logger.warning("Mic permission not yet granted: %s", retry_exc)
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "micPermissionDenied:", None, False
                    )
                    return
                logger.warning(
                    "Mic probe failed after PortAudio reset",
                    exc_info=True,
                )
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "micProbeFailed:", None, False
                )

    def _run_mic_permission_probe(self, sd_module) -> None:
        sd_module.rec(
            1600,
            samplerate=16000,
            channels=1,
            dtype='float32',
            blocking=True,
        )

    def _reset_portaudio_probe(self, sd_module) -> None:
        try:
            sd_module._terminate()
        except Exception:
            logger.debug("Mic probe PortAudio terminate failed", exc_info=True)
        try:
            sd_module._initialize()
        except Exception:
            logger.debug("Mic probe PortAudio initialize failed", exc_info=True)

    def _is_permission_probe_denial(self, exc: Exception) -> bool:
        message = str(exc).lower()
        permission_markers = (
            "permission",
            "not permitted",
            "permission denied",
            "access denied",
            "access was denied",
            "operation not permitted",
        )
        return any(marker in message for marker in permission_markers)

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

    def micProbeFailed_(self, _sender) -> None:
        """Main-thread callback after a non-permission mic probe failure."""
        from Foundation import NSTimer
        self._mic_probe_in_flight = False
        if self._menubar is not None:
            self._menubar.set_status_text("Mic unavailable — retrying…")
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
        self._hold_rejected_during_warmup = False
        self._models_ready = False
        self._warm_error = None
        self._refresh_startup_status()
        if self._warmup_in_flight:
            return
        self._warmup_in_flight = True
        threading.Thread(
            target=self._prepare_clients_in_background,
            daemon=True,
            name="client-warmup",
        ).start()

    def _prepare_clients_in_background(self) -> None:
        try:
            self._prepare_clients()
        except Exception as exc:
            self._warm_error = exc
            logger.exception("Model preparation failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "clientWarmupFailed:", None, False
            )
            return

        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "clientWarmupSucceeded:", None, False
        )

    def clientWarmupSucceeded_(self, _sender) -> None:
        self._warmup_in_flight = False
        self._models_ready = True
        self._warm_error = None
        logger.info("spoke ready — hold spacebar to record")
        self._menubar.set_status_text("Ready — hold spacebar")
        self._hide_startup_status()

        # Warm TTS after Whisper is loaded, but keep it off the main thread.
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            threading.Thread(
                target=self._warm_tts_in_background,
                daemon=True,
                name="tts-warmup",
            ).start()

    def clientWarmupFailed_(self, _sender) -> None:
        self._warmup_in_flight = False
        self._models_ready = False
        exc = self._warm_error or RuntimeError("Model warmup failed")
        self._refresh_startup_status()
        self._show_model_load_alert(exc)

    def _warm_tts_in_background(self) -> None:
        tts = getattr(self, "_tts_client", None)
        if tts is None:
            return
        try:
            tts.warm()
        except Exception:
            logger.exception("TTS warmup failed")

    def _show_startup_status(self, text: str) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return
        overlay.show()
        overlay.set_text(text)

    def _hide_startup_status(self) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is None:
            return
        overlay.hide()

    def _refresh_startup_status(self) -> None:
        if getattr(self, "_warm_error", None) is not None:
            self._show_startup_status(
                "Model load failed.\nChoose another model from the menu."
            )
            if self._menubar is not None:
                self._menubar.set_status_text(
                    "Model load failed — choose another model"
                )
            return

        self._show_startup_status(
            "Loading models...\nFirst launch may download selected models."
        )
        if self._menubar is not None:
            self._menubar.set_status_text("Loading models…")

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
            self._hold_rejected_during_warmup = True
            self._refresh_startup_status()
            return
        # If a command is actively streaming, cancel it
        if self._transcribing:
            logger.info("Hold during active stream — cancelling")
            self._transcription_token += 1
            self._transcribing = False
            # Fall through to start recording
        tts = getattr(self, "_tts_client", None)
        if tts is not None:
            # User speech should interrupt assistant speech before we reopen input.
            tts.cancel()
        # Clear Enter suppression — new hold replaces/dismisses the overlay.
        self._detector.command_overlay_active = False
        # Note: if command overlay is visible but finished, leave it up.
        # It will be dismissed if the user says nothing (empty recording)
        # or replaced if they send a new command.

        # Tray intercept: shift+space from tray = navigation, plain space = record.
        self._verify_paste_text = None
        if getattr(self, "_tray_active", False):
            shift_at_press = getattr(self._detector, '_shift_at_press', False)
            logger.info("Tray hold: shift_at_press=%s, shift_latched=%s",
                         shift_at_press, getattr(self._detector, '_shift_latched', 'N/A'))
            if shift_at_press or getattr(self._detector, '_shift_latched', False):
                # Shift+space hold during tray = navigation gesture, not recording.
                # Wait for release to route through _on_hold_end as navigate up.
                self._recovery_hold_active = True
                logger.info("Hold started during tray with shift — waiting for release (navigate)")
                return
            # Plain spacebar hold during tray = dismiss tray, start recording
            logger.info("Hold started during tray — dismissing tray, starting new recording")
            self._dismiss_tray()
            # Fall through to start recording
        elif getattr(self, "_recovery_text", None) is not None:
            self._recovery_hold_active = True
            logger.info("Hold started during recovery — waiting for release")
            return
        self._recovery_hold_active = False

        shift_at_press = getattr(self._detector, '_shift_at_press', False)
        logger.info(
            "Hold started — recording (shift_at_press=%s tray_active=%s recovery_active=%s recovery_hold_active=%s verify_pending=%s overlay_visible=%s)",
            shift_at_press,
            getattr(self, "_tray_active", False),
            getattr(self, "_recovery_text", None) is not None,
            getattr(self, "_recovery_hold_active", False),
            getattr(self, "_verify_paste_text", None) is not None,
            getattr(self._overlay, "_visible", False) if self._overlay is not None else False,
        )
        if self._menubar is not None:
            self._menubar.set_recording(True)
            self._menubar.set_status_text("Recording…")
        if self._glow is not None:
            self._glow.show()
        if self._overlay is not None:
            if self._glow is not None:
                self._overlay.set_brightness(
                    getattr(self._glow, "_brightness", 0.0),
                    immediate=True,
                )
            self._overlay.show()
        try:
            self._capture.start(amplitude_callback=self._on_amplitude)
        except Exception:
            logger.exception("Audio capture failed to start")
            if self._glow is not None:
                self._glow.hide()
            if self._overlay is not None:
                self._overlay.hide()
            if self._menubar is not None:
                self._menubar.set_recording(False)
                self._menubar.set_status_text("Audio input error — try again")
            return
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
            glow_brightness = getattr(self._glow, "_brightness", 0.0)
            if glow_brightness != getattr(self._overlay, "_brightness", 0.0):
                self._overlay.set_brightness(glow_brightness)
            self._sync_command_overlay_brightness()
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
            logger.info(
                "Dropping stale preview update (token=%s current=%s len=%d)",
                token,
                getattr(self, "_preview_session_token", None),
                len(text),
            )
            return
        if not self._preview_active:
            logger.info(
                "Dropping preview update while preview inactive (token=%s current=%s len=%d)",
                token,
                getattr(self, "_preview_session_token", None),
                len(text),
            )
            return
        self._last_preview_text = text
        if self._overlay is not None:
            self._overlay.set_text(text)

    def _on_hold_end(self, shift_held: bool = False, enter_held: bool = False) -> None:
        if getattr(self, "_hold_rejected_during_warmup", False):
            self._hold_rejected_during_warmup = False
            logger.info(
                "Hold ended before models were ready — keeping startup status"
            )
            if not getattr(self, "_models_ready", True):
                self._refresh_startup_status()
            return

        # ── Tray intercept ──
        # When the tray is active, gestures route through the tray handler.
        tray_active = getattr(self, "_tray_active", False)
        recovery_active = getattr(self, "_recovery_text", None) is not None
        if tray_active or recovery_active or getattr(self, "_recovery_hold_active", False):
            self._recovery_hold_active = False
            if shift_held:
                # Shift held + release spacebar from tray = navigate down (older)
                logger.info("Shift+space during tray — navigate down")
                self._tray_navigate_down()
            elif tray_active:
                # Spacebar from tray (tap or hold release) = insert
                logger.info("Spacebar during tray — inserting text")
                self._tray_insert_current()
            else:
                # Spacebar from recovery (non-tray) = retry insert
                logger.info("Spacebar during recovery — retrying Insert")
                self._recovery_retry_insert()
            return

        # ── Normal recording end ──
        logger.info("Hold ended — shift=%s enter=%s", shift_held, enter_held)
        self._preview_active = False
        self._preview_cancelled_on_release = True
        wav_bytes = self._capture.stop()

        # Short shift-hold (under 800ms of recording) = recall into tray
        elapsed = time.monotonic() - self._record_start_time if self._record_start_time else 0
        if shift_held and elapsed < 0.8:
            logger.info("Short shift-hold (%.0fms) — recalling into tray", elapsed * 1000)
            wav_bytes = b""  # force the empty-audio path

        # Glow/dimmer: hide immediately for text insertion
        if not shift_held and not enter_held and self._glow is not None:
            self._glow.hide()
        if self._menubar is not None:
            self._menubar.set_recording(False)

        if not wav_bytes:
            logger.info("No audio — instant path (shift=%s, enter=%s)", shift_held, enter_held)
            if self._overlay is not None:
                self._overlay.hide()
            if self._glow is not None:
                self._glow.hide()

            if enter_held and self._command_client is not None:
                # Enter + empty recording = recall last assistant response
                command_visible = (
                    self._command_overlay is not None
                    and getattr(self._command_overlay, '_visible', False)
                )
                if command_visible:
                    # Already showing — dismiss it
                    logger.info("Enter+empty — dismissing command overlay")
                    self._command_overlay.cancel_dismiss()
                    self._detector.command_overlay_active = False
                else:
                    # Not showing — recall last response
                    history = self._command_client.history
                    if history:
                        last_utterance, last_response = history[-1]
                        logger.info("Enter+empty — recalling last response")
                        if self._command_overlay is not None:
                            try:
                                self._command_overlay.show()
                                self._command_overlay.set_utterance(last_utterance)
                                self._command_overlay.append_token(last_response)
                                self._command_overlay.finish()
                                self._detector.command_overlay_active = True
                            except Exception:
                                logger.exception("Recall overlay failed")
                    else:
                        logger.info("Enter+empty — no history to recall")
            elif shift_held:
                # Shift + empty recording = recall tray
                if self._tray_stack:
                    logger.info("Shift+empty — recalling tray (stack has %d entries)", len(self._tray_stack))
                    self._tray_active = True
                    self._detector.tray_active = True
                    self._tray_index = len(self._tray_stack) - 1
                    self._show_tray_current(acknowledge=True)
                    return
                else:
                    logger.info("Shift+empty — no tray entries to recall")
            else:
                command_visible = (
                    self._command_overlay is not None
                    and getattr(self._command_overlay, '_visible', False)
                )
                if command_visible:
                    logger.info("Empty recording — dismissing command overlay")
                    self._command_overlay.cancel_dismiss()
                    self._detector.command_overlay_active = False

            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")
            return

        # Invalidate any in-flight transcription so its result is discarded
        self._transcription_token += 1
        token = self._transcription_token

        self._transcribing = True
        self._transcribe_start = time.monotonic()

        if enter_held and self._command_client is not None:
            # Command pathway (enter held): transcribe then send to OMLX
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing command…")
            thread = threading.Thread(
                target=self._command_transcribe_worker,
                args=(wav_bytes, token),
                daemon=True,
            )
        elif shift_held:
            # Tray pathway: transcribe, then enter tray with the result.
            # Don't set _tray_active yet — defer until _enter_tray is called
            # from trayTranscriptionComplete_. Setting it prematurely would
            # let gestures fire on stale/empty stack state during transcription.
            if self._menubar is not None:
                self._menubar.set_status_text("Transcribing…")
            thread = threading.Thread(
                target=self._tray_transcribe_worker,
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
                    self._sync_command_overlay_brightness(immediate=True)
                    self._command_overlay.show()
                    self._command_overlay.set_utterance(last_utterance)
                    for ch in last_response:
                        self._command_overlay.append_token(ch)
                    self._command_overlay.finish()
                    self._detector.command_overlay_active = True
                if self._menubar is not None:
                    self._menubar.set_status_text("Ready — hold spacebar")
                return

        logger.info("No history to recall")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def _sync_command_overlay_brightness(self, immediate: bool = False) -> None:
        if self._command_overlay is None or self._glow is None:
            return
        self._command_overlay.set_brightness(
            getattr(self._glow, "_brightness", 0.0),
            immediate=immediate,
        )

    def _resetStatusAfterCancel_(self, timer) -> None:
        """Reset menubar status after a cancel."""
        if self._menubar is not None and not self._transcribing:
            self._menubar.set_status_text("Ready — hold spacebar")

    # ── tray ───────────────────────────────────────────────

    def _tray_transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe audio, then enter tray on main thread."""
        release_cutover = getattr(self, "_preview_cancelled_on_release", False)

        if self._preview_thread is not None and not release_cutover:
            if getattr(self, "_preview_done", None) is not None:
                self._preview_done.wait(timeout=2.0)
            self._preview_thread.join(timeout=2.0)
            self._preview_thread = None

        try:
            with self._local_inference_context(self._client):
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
            logger.exception("Tray transcription failed")
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "trayTranscriptionFailed:", {"token": token}, False
            )
            return

        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "trayTranscriptionComplete:",
            {"token": token, "text": text},
            False,
        )

    def trayTranscriptionComplete_(self, payload: dict) -> None:
        """Main thread: transcription done — enter tray with the text."""
        if payload["token"] != self._transcription_token:
            logger.info("Discarding stale tray transcription (token %d)", payload["token"])
            return
        self._transcribing = False
        text = payload["text"]
        if not text:
            # Empty transcription — use last preview text if available
            if self._last_preview_text:
                logger.info("Tray transcription empty — using last preview text")
                text = self._last_preview_text
            else:
                logger.info("Tray transcription returned empty — dismissing")
                if self._overlay is not None:
                    self._overlay.hide()
                if self._glow is not None:
                    self._glow.hide()
                self._tray_active = False
                self._detector.tray_active = False
                if self._menubar is not None:
                    self._menubar.set_status_text("Ready — hold spacebar")
                return
        self._enter_tray(text)

    def trayTranscriptionFailed_(self, payload: dict) -> None:
        """Main thread: tray transcription failed — fall back to preview text."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        if self._last_preview_text:
            logger.warning("Tray transcription failed — using preview text")
            self._enter_tray(self._last_preview_text)
            return
        logger.error("Tray transcription failed — no text")
        if self._overlay is not None:
            self._overlay.hide()
        if self._glow is not None:
            self._glow.hide()
        self._tray_active = False
        self._detector.tray_active = False
        if self._menubar is not None:
            self._menubar.set_status_text("Error — try again")

    def _enter_tray(self, text: str) -> None:
        """Enter the tray with new text, pushing it onto the stack."""
        self._add_tray_entry(text, owner="user", activate=True)

    def _coerce_tray_entry(self, entry: TrayEntry | str) -> TrayEntry:
        if isinstance(entry, TrayEntry):
            return entry
        return TrayEntry(text=str(entry))

    def _get_tray_entry(self, index: int) -> TrayEntry:
        entry = self._coerce_tray_entry(self._tray_stack[index])
        self._tray_stack[index] = entry
        return entry

    def _acknowledge_tray_entry(self, index: int) -> None:
        if index < 0 or index >= len(self._tray_stack):
            return
        entry = self._get_tray_entry(index)
        if entry.owner == "assistant" and not entry.acknowledged:
            entry.acknowledged = True

    def _add_tray_entry(
        self,
        text: str,
        *,
        owner: str = "user",
        activate: bool = True,
    ) -> TrayEntry:
        entry = TrayEntry(
            text=text,
            owner=owner,
            acknowledged=(owner != "assistant"),
        )
        self._tray_stack.append(entry)
        self._tray_index = len(self._tray_stack) - 1

        if activate:
            self._tray_active = True
            self._detector.tray_active = True
            logger.info(
                "Entering tray (entries=%d index=%d text_len=%d)",
                len(self._tray_stack),
                self._tray_index,
                len(text),
            )

            if self._glow is not None:
                if hasattr(self._glow, "show_tray_dim"):
                    self._glow.show_tray_dim()
                else:
                    self._glow.hide()

            self._show_tray_current()

        return entry

    def _show_tray_current(self, *, acknowledge: bool = False) -> None:
        """Update the tray overlay to display the current stack entry."""
        if not self._tray_stack:
            self._dismiss_tray()
            return
        # Defensive bounds clamp
        if self._tray_index >= len(self._tray_stack):
            self._tray_index = len(self._tray_stack) - 1
        if acknowledge:
            self._acknowledge_tray_entry(self._tray_index)
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text
        # Set recovery_text for compatibility with existing dismiss/cleanup
        self._recovery_text = text
        self._recovery_clipboard_state = "idle"
        if self._overlay is not None:
            self._overlay.show_tray(text, owner=entry.display_owner)
        if self._menubar is not None:
            pos = f"{self._tray_index + 1}/{len(self._tray_stack)}"
            self._menubar.set_status_text(f"Tray [{pos}]")

    def _dismiss_tray(self) -> None:
        """Dismiss the tray overlay. Stack is preserved for re-entry."""
        logger.info(
            "Dismissing tray (entries=%d index=%d recovery_active=%s)",
            len(self._tray_stack),
            self._tray_index,
            getattr(self, "_recovery_text", None) is not None,
        )
        self._tray_active = False
        self._detector.tray_active = False
        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")

    def _on_tray_shift_tap(self) -> None:
        """Shift tap (no spacebar) during tray = dismiss."""
        if self._tray_active:
            logger.info("Shift tap during tray — dismiss")
            self._acknowledge_tray_entry(self._tray_index)
            self._dismiss_tray()

    def _on_audio_shift_tap(self) -> None:
        """Shift tap while idle toggles current TTS audibility."""
        if self._tray_active:
            return
        tts = getattr(self, "_tts_client", None)
        if tts is None:
            return
        audible = tts.toggle_audio()
        logger.info("Idle shift tap — audio target now %s", "on" if audible else "off")
        if self._menubar is not None:
            self._menubar.set_status_text("Audio on" if audible else "Audio muted")

    def _on_tray_navigate_up(self) -> None:
        """Spacebar held + shift tapped during tray = navigate up (more recent)."""
        if self._tray_active:
            logger.info("Shift tap during hold — navigate up")
            self._tray_navigate_up()

    def _on_tray_enter_pressed(self) -> None:
        """Enter pressed during tray = send current entry to assistant."""
        if self._tray_active:
            logger.info("Enter during tray — sending to assistant")
            self._tray_send_current()

    def _on_tray_delete_gesture(self) -> None:
        """Shift held + double-tap spacebar = delete current tray entry."""
        if self._tray_active:
            logger.info("Double-tap delete during tray")
            self._tray_delete_current()

    def _tray_cycle(self) -> None:
        """Cycle to the next tray entry, wrapping at the edges."""
        if not self._tray_active or len(self._tray_stack) < 2:
            return
        self._tray_index = (self._tray_index - 1) % len(self._tray_stack)
        self._show_tray_current()

    def _tray_navigate_up(self) -> None:
        """Navigate up toward more recent entries. Dismiss at top."""
        if not self._tray_active:
            return
        if self._tray_index >= len(self._tray_stack) - 1:
            # Already at the top — dismiss
            self._dismiss_tray()
        else:
            self._tray_index += 1
            self._show_tray_current(acknowledge=True)

    def _tray_navigate_down(self) -> None:
        """Navigate down toward older entries. Stop at bottom."""
        if not self._tray_active:
            return
        if self._tray_index > 0:
            self._tray_index -= 1
            self._show_tray_current(acknowledge=True)

    def _tray_delete_current(self) -> None:
        """Delete the currently displayed tray entry."""
        if not self._tray_active or not self._tray_stack:
            return
        del self._tray_stack[self._tray_index]
        if not self._tray_stack:
            self._dismiss_tray()
            return
        # Adjust index: stay at same position or move up if we were at the end
        if self._tray_index >= len(self._tray_stack):
            self._tray_index = len(self._tray_stack) - 1
        self._show_tray_current(acknowledge=True)

    def _tray_insert_current(self) -> None:
        """Insert the current tray entry at cursor and consume it."""
        if not self._tray_active or not self._tray_stack:
            return
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text

        # Dismiss tray first, then inject. Dismissing before inject ensures
        # the tray overlay doesn't interfere with focus on the target app.
        self._tray_active = False
        self._detector.tray_active = False

        # Remove consumed entry from stack
        del self._tray_stack[self._tray_index]
        if self._tray_index >= len(self._tray_stack) and self._tray_stack:
            self._tray_index = len(self._tray_stack) - 1

        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()
        if self._overlay is not None:
            self._overlay.hide()

        # Small delay to let focus settle after overlay dismissal,
        # then inject the text
        self._tray_pending_inject = text
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "trayInjectDelayed:", None, False
        )

    def trayInjectDelayed_(self, timer) -> None:
        """Inject tray text after a short delay for focus to settle."""
        text = getattr(self, "_tray_pending_inject", None)
        self._tray_pending_inject = None
        if not text:
            return

        inject_text(text)
        logger.info("Tray injected: %r", text[:50])
        if self._menubar is not None:
            self._menubar.set_status_text("Pasted!")

    def _tray_send_current(self) -> None:
        """Send the current tray entry to the assistant and consume it."""
        if not self._tray_stack:
            # Nothing to send — make sure tray state is clean
            if self._tray_active:
                self._dismiss_tray()
            return
        if not self._tray_active:
            return
        entry = self._get_tray_entry(self._tray_index)
        text = entry.text

        # Remove consumed entry from stack
        del self._tray_stack[self._tray_index]
        if self._tray_index >= len(self._tray_stack) and self._tray_stack:
            self._tray_index = len(self._tray_stack) - 1

        self._tray_active = False
        self._detector.tray_active = False
        if self._glow is not None:
            self._glow.hide()
        self._cancel_recovery()

        if self._command_client is not None:
            self._send_text_as_command(text)
        else:
            logger.warning("Tray send — no command client configured")
            if self._menubar is not None:
                self._menubar.set_status_text("Ready — hold spacebar")

    def _add_assistant_content_to_tray(self, text: str) -> dict:
        """Place assistant-created content into the tray on the main thread.

        Must be called from a background thread — uses waitUntilDone=True to
        synchronously dispatch tray mutation onto the main thread. Relies on
        tool calls being executed sequentially (the command.py tool loop) so
        that _tray_tool_result is never written concurrently.
        """
        self._tray_tool_result = None
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "toolAddToTray:",
            {"text": text},
            True,
        )
        return self._tray_tool_result or {"error": "Tray update failed"}

    def toolAddToTray_(self, payload: dict) -> None:
        """Main-thread tray mutation entrypoint for command tool calls."""
        text = payload.get("text", "")
        reveal_now = bool(self._tray_active)
        self._add_tray_entry(
            text,
            owner="assistant",
            activate=reveal_now,
        )
        self._tray_tool_result = {
            "status": "added",
            "tray_visible": reveal_now,
            "stack_size": len(self._tray_stack),
            "owner": "assistant",
        }

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
        self._command_tool_used_tts = False
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
            full_response = ""
            try:
                for event in self._command_client.stream_command_events(
                    text,
                    tools=self._tool_schemas,
                    tool_executor=self._make_tool_executor(),
                ):
                    if token != self._transcription_token:
                        break  # stale
                    if event.kind == "assistant_delta":
                        self.performSelectorOnMainThread_withObject_waitUntilDone_(
                            "commandToken:",
                            {"token": token, "text": event.text},
                            False,
                        )
                    elif event.kind == "assistant_final":
                        full_response = event.text
            except Exception:
                logger.exception("Command stream failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "commandFailed:", {"token": token, "error": "Command failed"}, False
                )
                return

            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandComplete:", {"token": token, "response": full_response}, False
            )

        threading.Thread(target=_stream, daemon=True).start()

    def _make_tool_executor(self):
        """Build a tool executor closure with current app state."""
        scene_cache = self._scene_cache
        raw_tts_client = getattr(self, "_tts_client", None)
        tts_client = raw_tts_client
        # Get last assistant response for last_response refs
        last_response = None
        try:
            history = self._command_client.history if self._command_client else []
            if history:
                _, last_response = history[-1]
        except (TypeError, ValueError):
            pass

        if raw_tts_client is not None:
            delegate = self

            class _ToolTTSProxy:
                def __init__(self, client):
                    self._client = client

                def speak_async(self, text, *args, **kwargs):
                    result = self._client.speak_async(text, *args, **kwargs)
                    delegate._command_tool_used_tts = True
                    return result

                def speak(self, text, *args, **kwargs):
                    result = self._client.speak(text, *args, **kwargs)
                    delegate._command_tool_used_tts = True
                    return result

                def __getattr__(self, name):
                    return getattr(self._client, name)

            tts_client = _ToolTTSProxy(raw_tts_client)

        def _executor(name, arguments, **kwargs):
            return execute_tool(
                name=name,
                arguments=arguments,
                scene_cache=scene_cache,
                last_response=last_response,
                tts_client=tts_client,
                tray_writer=self._add_assistant_content_to_tray,
            )
        return _executor

    def _command_transcribe_worker(self, wav_bytes: bytes, token: int) -> None:
        """Background thread: transcribe then send command to OMLX."""
        self._command_tool_used_tts = False
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
            for event in self._command_client.stream_command_events(
                utterance,
                tools=self._tool_schemas,
                tool_executor=self._make_tool_executor(),
            ):
                if token != self._transcription_token:
                    break  # stale
                if event.kind == "assistant_delta":
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "commandToken:",
                        {"token": token, "text": event.text},
                        False,
                    )
                elif event.kind == "assistant_final":
                    full_response = event.text
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
            self._sync_command_overlay_brightness(immediate=True)
            self._command_overlay.show()
            self._command_overlay.set_utterance(utterance)
            self._detector.command_overlay_active = True
        self._command_first_token = True
        if self._menubar is not None:
            self._menubar.set_status_text("Thinking…")

    def commandToken_(self, payload: dict) -> None:
        """Main thread: append a streamed token to the command overlay."""
        if payload["token"] != self._transcription_token:
            return
        overlay = self._command_overlay
        # First content token: invert the thinking timer and update status
        if getattr(self, "_command_first_token", False):
            self._command_first_token = False
            if overlay is not None:
                try:
                    overlay.invert_thinking_timer()
                except Exception:
                    logger.exception("Command overlay failed to invert thinking timer")
            if self._menubar is not None:
                self._menubar.set_status_text("Responding…")
        if overlay is not None:
            try:
                overlay.append_token(payload["text"])
            except Exception:
                logger.exception("Command overlay failed to append streamed token")

    def commandComplete_(self, payload: dict) -> None:
        """Main thread: command response finished streaming."""
        if payload["token"] != self._transcription_token:
            return
        self._transcribing = False
        overlay = self._command_overlay
        response = payload.get("response", "")
        if overlay is not None and response:
            try:
                overlay.set_response_text(response)
            except Exception:
                logger.exception("Command overlay failed to apply final response text")
        if overlay is not None:
            try:
                overlay.finish()
            except Exception:
                logger.exception("Command overlay finish failed")
        if self._menubar is not None:
            self._menubar.set_status_text("Ready — hold spacebar")
        # Autoplay response via TTS if enabled — glow hides, overlay breathes with voice
        tts = getattr(self, "_tts_client", None)
        tool_used_tts = getattr(self, "_command_tool_used_tts", False)
        self._command_tool_used_tts = False
        logger.info(
            "TTS autoplay decision: response=%d chars, tts_client=%s, tool_used_tts=%s, model=%s",
            len(response) if response else 0,
            tts is not None,
            tool_used_tts,
            getattr(tts, "_model_id", "?") if tts else "none",
        )
        if not response:
            logger.info("TTS autoplay: skipped — no response text")
        elif tts is None:
            logger.info("TTS autoplay: skipped — no TTS client")
            if self._menubar is not None:
                self._menubar.set_status_text("TTS: not configured")
        elif tool_used_tts:
            logger.info("TTS autoplay: skipped — tool already used TTS")
        if response and tts is not None and not tool_used_tts:
            if self._glow is not None:
                self._glow.hide()
            if overlay is not None:
                try:
                    overlay.tts_start()
                except Exception:
                    logger.exception("Command overlay TTS start failed")
            try:
                logger.info("TTS autoplay: calling speak_async with %d chars", len(response))
                tts.speak_async(
                    response,
                    amplitude_callback=self._on_tts_amplitude,
                    done_callback=lambda: self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "ttsFinished:", None, False
                    ),
                )
                logger.info("TTS autoplay: speak_async returned (queued)")
            except Exception:
                logger.exception("Command autoplay failed to start")
                if overlay is not None:
                    try:
                        overlay.tts_stop()
                    except Exception:
                        logger.exception("Command overlay TTS stop failed")
        elif self._glow is not None:
            self._glow.hide()

    def _on_tts_amplitude(self, rms: float) -> None:
        """Called from TTS thread — marshal to main thread."""
        from Foundation import NSNumber
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "ttsAmplitudeUpdate:", NSNumber.numberWithFloat_(rms), False
        )

    def ttsAmplitudeUpdate_(self, rms_number) -> None:
        """Main thread: forward TTS amplitude to command overlay."""
        rms = float(rms_number)
        if self._command_overlay is not None:
            try:
                self._command_overlay.update_tts_amplitude(rms)
            except Exception:
                logger.exception("Command overlay amplitude update failed")

    def ttsFinished_(self, _) -> None:
        """Main thread: TTS playback ended — restore overlay opacity."""
        if self._command_overlay is not None:
            try:
                self._command_overlay.tts_stop()
            except Exception:
                logger.exception("Command overlay TTS stop failed")

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
                self._sync_command_overlay_brightness(immediate=True)
                try:
                    self._command_overlay.show()
                except Exception:
                    logger.exception("Command overlay show failed during error presentation")
            try:
                self._command_overlay.append_token("couldn't reach the model — try again in a moment")
            except Exception:
                logger.exception("Command overlay append failed during error presentation")
            try:
                self._command_overlay.finish()
                self._detector.command_overlay_active = True
            except Exception:
                logger.exception("Command overlay finish failed during error presentation")
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
            logger.warning(
                "Paste not verified by OCR after %d attempts — entering recovery",
                attempt + 1,
            )
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
        (_PARAKEET_MODEL_ID, "Parakeet CTC-110M (CoreML/ANE, preview only)"),
    ]

    # Parakeet is preview-only: too rough for final transcription
    _PREVIEW_ONLY_MODELS = frozenset({_PARAKEET_MODEL_ID})

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
        current_preview, current_transcription = self._sanitize_model_ids(
            current_preview,
            current_transcription,
        )
        self._apply_model_selection(
            preview_model=current_preview if model_id is None else model_id,
            transcription_model=current_transcription if model_id is None else model_id,
        )

    def _handle_model_menu_action(self, selection):
        """Menu callback for preview/transcription role-specific model choices."""
        if selection is None:
            current_preview = getattr(
                self, "_preview_model_id", _DEFAULT_PREVIEW_MODEL
            )
            current_transcription = getattr(
                self, "_transcription_model_id", self._default_transcription_model()
            )
            current_preview, current_transcription = self._sanitize_model_ids(
                current_preview,
                current_transcription,
            )
            state = {
                "transcription": {
                    "selected": current_transcription,
                    "models": self._select_model(None),
                },
                "preview": {
                    "selected": current_preview,
                    "models": self._select_model(None),
                },
            }
            launch_target = self._launch_target_menu_state()
            if launch_target is not None:
                state["launch_target"] = launch_target
            if self._command_client is not None:
                state["assistant"] = {
                    "selected": self._command_model_id,
                    "models": self._command_model_options,
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
            tts = getattr(self, "_tts_client", None)
            if tts is not None:
                current_tts_model = tts._model_id
                tts_models = [
                    (model_id, label, True)
                    for model_id, label in _TTS_MODELS
                ]
                state["tts"] = {
                    "selected": current_tts_model,
                    "models": tts_models,
                }
            return state
        if not isinstance(selection, tuple) or len(selection) != 2:
            self._select_model(selection)
            return
        role, model_id = selection
        if role == "assistant":
            self._apply_command_model_selection(model_id)
            return
        if role == "launch_target":
            self._apply_launch_target_selection(model_id)
            return
        if role == "tts":
            self._apply_tts_model_selection(model_id)
            return
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
        current_preview, current_transcription = self._sanitize_model_ids(
            current_preview,
            current_transcription,
        )
        preview_model = current_preview
        transcription_model = current_transcription
        if role == "preview":
            preview_model = model_id
        else:
            transcription_model = model_id
        self._apply_model_selection(preview_model, transcription_model)

    def _current_checkout_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _launch_target_menu_state(self) -> dict | None:
        targets = iter_launch_targets()
        if not targets:
            return None
        current_target = current_launch_target_id(self._current_checkout_root())
        return {
            "title": "Launch Target",
            "selected": current_target,
            "items": [
                (target["id"], target["label"], target["enabled"]) for target in targets
            ],
        }

    def _persist_launch_target_selection(self, target_id: str) -> bool:
        return save_selected_launch_target(target_id)

    def _invoke_launch_target_helper(self, target_id: str) -> bool:
        helper_path = Path(__file__).resolve().parents[1] / "scripts" / "launch-target.sh"
        if not helper_path.is_file():
            logger.warning("Launch target helper is missing: %s", helper_path)
            return False
        import subprocess

        subprocess.Popen(
            ["/bin/bash", str(helper_path), target_id],
            cwd=helper_path.parent.parent,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True

    def _apply_launch_target_selection(self, target_id: str) -> None:
        current_target = current_launch_target_id(self._current_checkout_root())
        if not self._persist_launch_target_selection(target_id):
            logger.warning(
                "Skipping launch-target switch because %s could not be persisted",
                target_id,
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save launch target")
            return
        if target_id == current_target:
            return
        logger.info("Switching launch target (handoff): %s -> %s", current_target, target_id)
        if not self._invoke_launch_target_helper(target_id):
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't switch launch target")
            return
        if self._menubar is not None:
            self._menubar.set_status_text(f"Switching to {target_id}…")

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

    def _fallback_model_for_role(self, role: str) -> str:
        if role == "preview":
            return _DEFAULT_PREVIEW_MODEL
        return self._default_transcription_model()

    def _sanitize_model_id(self, model_id: str, *, role: str) -> str:
        # Preview-only models (e.g. Parakeet) must not be used for transcription
        if role == "transcription" and model_id in self._PREVIEW_ONLY_MODELS:
            fallback = self._default_transcription_model()
            logger.warning(
                "Model %s is preview-only and cannot be used for transcription — falling back to %s",
                model_id,
                fallback,
            )
            return fallback
        if self._model_allowed(model_id):
            return model_id
        fallback = self._fallback_model_for_role(role)
        logger.warning(
            "%s model %s not available on this machine (%.0fGB RAM) — falling back to %s",
            role.title(),
            model_id,
            _RAM_GB,
            fallback,
        )
        return fallback

    def _sanitize_model_ids(
        self, preview_model: str, transcription_model: str
    ) -> tuple[str, str]:
        return (
            self._sanitize_model_id(preview_model, role="preview"),
            self._sanitize_model_id(transcription_model, role="transcription"),
        )

    def _resolve_model_ids(self) -> tuple[str, str]:
        prefs = self._load_model_preferences()
        legacy_model = os.environ.get("SPOKE_WHISPER_MODEL")
        raw_preview_model = (
            os.environ.get("SPOKE_PREVIEW_MODEL")
            or prefs.get("preview_model")
            or legacy_model
            or _DEFAULT_PREVIEW_MODEL
        )
        raw_transcription_model = (
            os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            or prefs.get("transcription_model")
            or legacy_model
            or self._default_transcription_model()
        )
        preview_model, transcription_model = self._sanitize_model_ids(
            raw_preview_model,
            raw_transcription_model,
        )
        if (
            not os.environ.get("SPOKE_PREVIEW_MODEL")
            and not os.environ.get("SPOKE_TRANSCRIPTION_MODEL")
            and not legacy_model
            and (
                prefs.get("preview_model") is not None
                or prefs.get("transcription_model") is not None
            )
            and (
                preview_model != raw_preview_model
                or transcription_model != raw_transcription_model
            )
        ):
            logger.info(
                "Repairing unsupported saved model preferences: preview=%s, transcription=%s",
                preview_model,
                transcription_model,
            )
            self._save_model_preferences(preview_model, transcription_model)
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

    def _load_command_model_preference(self) -> str | None:
        return self._load_preferences().get("command_model")

    def _save_model_preferences(
        self, preview_model: str, transcription_model: str
    ) -> bool:
        payload = self._load_preferences()
        payload["preview_model"] = preview_model
        payload["transcription_model"] = transcription_model
        return self._save_preferences(payload)

    def _save_command_model_preference(self, model_id: str) -> bool:
        payload = self._load_preferences()
        payload["command_model"] = model_id
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
        if model_id == _PARAKEET_MODEL_ID:
            model_dir = self._resolve_parakeet_model_dir()
            logger.info("Using Parakeet CoreML: %s", model_dir)
            return ParakeetCoreMLClient(model_dir=model_dir)
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

    def _resolve_parakeet_model_dir(self) -> Path:
        """Return the Parakeet CoreML model directory.

        Checks SPOKE_PARAKEET_MODEL_DIR env var first, then the default HF
        cache location (~/.cache/huggingface/hub/models--FluidInference--parakeet-ctc-110m-coreml/snapshots/).
        Falls back to an empty path so the client can log a clear warning.
        """
        env_dir = os.environ.get("SPOKE_PARAKEET_MODEL_DIR", "")
        if env_dir:
            return Path(env_dir).expanduser()

        # Try HF hub cache — look for the most recent snapshot
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        snapshots = hf_cache / "models--FluidInference--parakeet-ctc-110m-coreml" / "snapshots"
        if snapshots.is_dir():
            candidates = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for candidate in candidates:
                if (candidate / "AudioEncoder.mlmodelc").exists():
                    return candidate

        # No model found — return a path that will produce a clear warning
        return Path("/nonexistent/parakeet-ctc-110m-coreml")

    def _discover_command_models(
        self, selected_model: str
    ) -> list[tuple[str, str, bool]]:
        server_model_ids: list[str] = []
        if self._command_client is not None:
            try:
                server_model_ids = self._command_client.list_models()
            except Exception:
                logger.warning("Failed to fetch assistant models from OMLX", exc_info=True)
        local_model_dir = Path(
            os.environ.get("SPOKE_COMMAND_MODEL_DIR", str(_DEFAULT_COMMAND_MODEL_DIR))
        ).expanduser()
        local_model_ids = _iter_local_command_model_ids(local_model_dir)
        if local_model_ids:
            local_model_set = set(local_model_ids)
            model_ids = [
                model_id
                for model_id in server_model_ids
                if model_id in local_model_set
            ]
            model_ids.extend(
                model_id for model_id in local_model_ids if model_id not in model_ids
            )
        else:
            model_ids = server_model_ids
        seen: set[str] = set()
        options = []
        for model_id in model_ids:
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            options.append((model_id, model_id, model_id == selected_model))
        return options

    def _seed_command_model_options(
        self, selected_model: str
    ) -> list[tuple[str, str, bool]]:
        """Seed the Assistant menu from local disk without hitting /v1/models."""
        local_model_dir = Path(
            os.environ.get("SPOKE_COMMAND_MODEL_DIR", str(_DEFAULT_COMMAND_MODEL_DIR))
        ).expanduser()
        local_model_ids = _iter_local_command_model_ids(local_model_dir)
        if local_model_ids:
            return [
                (model_id, model_id, model_id == selected_model)
                for model_id in local_model_ids
            ]
        return [(selected_model, selected_model, True)] if selected_model else []

    def _refresh_command_model_options_async(self) -> None:
        if self._command_client is None or self._command_models_refresh_in_flight:
            return
        self._command_models_refresh_in_flight = True

        def _load():
            options = self._discover_command_models(self._command_model_id)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "commandModelsDiscovered:",
                {"options": options},
                False,
            )

        threading.Thread(
            target=_load,
            daemon=True,
            name="command-model-refresh",
        ).start()

    def commandModelsDiscovered_(self, payload: dict) -> None:
        self._command_models_refresh_in_flight = False
        options = payload.get("options") or []
        self._command_model_options = options
        if self._menubar is not None:
            self._menubar.refresh_menu()

    def _apply_command_model_selection(self, model_id: str) -> None:
        current_model = self._command_model_id
        if model_id == current_model:
            if self._load_command_model_preference() != model_id:
                logger.info(
                    "Repairing stale assistant model preference without relaunch: %s",
                    model_id,
                )
                self._save_command_model_preference(model_id)
            return
        logger.info(
            "Switching assistant model (relaunching): %s -> %s",
            current_model,
            model_id,
        )
        if not self._save_command_model_preference(model_id):
            logger.warning(
                "Skipping relaunch because the assistant model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save model selection")
            return
        self._command_model_id = model_id
        os.environ["SPOKE_COMMAND_MODEL"] = model_id
        self._relaunch()

    def _apply_tts_model_selection(self, model_id: str) -> None:
        tts = getattr(self, "_tts_client", None)
        current_model = tts._model_id if tts else None
        if model_id == current_model:
            return
        logger.info(
            "Switching TTS model (relaunching): %s -> %s",
            current_model,
            model_id,
        )
        payload = self._load_preferences()
        payload["tts_model"] = model_id
        if not self._save_preferences(payload):
            logger.warning(
                "Skipping relaunch because the TTS model selection could not be persisted"
            )
            if self._menubar is not None:
                self._menubar.set_status_text("Couldn't save TTS model selection")
            return
        self._relaunch()

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

        # Save clipboard state before inject_text overwrites it, in case we
        # need to enter recovery mode after OCR verification.
        self._pre_paste_clipboard = save_pasteboard()
        # Give the target app a brief moment to refocus after the overlay
        # disappears before we synthesize Cmd+V.
        self._result_pending_inject = (text, status_text)
        from Foundation import NSTimer
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "resultInjectDelayed:", None, False
        )

    def resultInjectDelayed_(self, timer) -> None:
        """Paste normal-path text after a short post-overlay refocus delay."""
        pending = getattr(self, "_result_pending_inject", None)
        self._result_pending_inject = None
        if pending is None:
            return
        text, status_text = pending

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
        """Paste verification failed — enter the tray automatically.

        Recovery mode is tray entered automatically when paste verification
        fails. The tray subsumes recovery as a special case.
        """
        # Save clipboard state for potential restore
        pre = getattr(self, "_pre_paste_clipboard", _NOT_CAPTURED)
        if pre is not _NOT_CAPTURED:
            self._recovery_saved_clipboard = pre
        else:
            self._recovery_saved_clipboard = save_pasteboard()
        self._pre_paste_clipboard = _NOT_CAPTURED

        # Put transcription on pasteboard for manual paste
        set_pasteboard_only(text)

        # Enter tray with the failed paste text
        self._enter_tray(text)

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
        """Cancel recovery/tray overlay if active. Restores original clipboard."""
        if getattr(self, "_recovery_text", None) is not None:
            restore_pasteboard(getattr(self, "_recovery_saved_clipboard", None))
            if self._overlay is not None:
                # dismiss_recovery handles button-based recovery mode cleanup.
                # For tray mode (show_tray), we just need to hide the overlay.
                if self._overlay._recovery_mode:
                    self._overlay.dismiss_recovery()
                else:
                    self._overlay.hide()
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
        """Guard: some models require specific hardware or installed files."""
        if "large-v3-turbo" in model_id and _RAM_GB < _MIN_RAM_GB_FOR_V3_TURBO:
            return False
        if model_id == _PARAKEET_MODEL_ID:
            # Only show if the model files are present or the user has pointed us at them
            env_dir = os.environ.get("SPOKE_PARAKEET_MODEL_DIR", "")
            if env_dir and (Path(env_dir).expanduser() / "AudioEncoder.mlmodelc").exists():
                return True
            hf_snapshots = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--FluidInference--parakeet-ctc-110m-coreml"
                / "snapshots"
            )
            if hf_snapshots.is_dir():
                return any(
                    (s / "AudioEncoder.mlmodelc").exists() for s in hf_snapshots.iterdir()
                )
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
