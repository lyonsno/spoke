"""Hands-free continuous dictation mode.

Coordinates wake word detection (Porcupine or openWakeWord), audio capture (VAD-based
segment slicing), transcription, and text injection into a continuous
loop:

    Wake word "listen" → start dictating
    Speech → silence → transcribe segment → inject at cursor + space → loop
    Wake word "sleep" → stop dictating

Voice commands detected post-transcription:
    "new line" / "enter"  → inject \\n
    "new paragraph"       → inject \\n\\n
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def handsfree_env_ready() -> bool:
    """True when the environment provides a runnable wakeword backend."""
    backend = os.environ.get("SPOKE_WAKEWORD_BACKEND", "porcupine").strip().lower()
    if backend == "openwakeword":
        return bool(
            os.environ.get("SPOKE_WAKEWORD_LISTEN_MODEL")
            and os.environ.get("SPOKE_WAKEWORD_SLEEP_MODEL")
        )
    return bool(os.environ.get("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY"))


# ── Voice commands (matched against full transcription, case-insensitive) ──
#
# Each entry maps to a tuple of (action, value):
#   ("inject", "\n")            — paste this text at cursor
#   ("keystroke", "return")     — synthesize a bare keystroke
#   ("chord", "cmd+shift+[")    — synthesize a modifier chord
#   ("tray_enter", "")          — enter tray dictation mode
#   ("tray_insert", "")         — insert current tray entry at cursor

VOICE_COMMANDS: dict[str, tuple[str, str]] = {
    # Text insertion
    "new line": ("inject", "\n"),
    "newline": ("inject", "\n"),
    "new paragraph": ("inject", "\n\n"),
    # θές — Enter keystroke
    "tessera": ("keystroke", "return"),
    "tesserae": ("keystroke", "return"),
    "tessara": ("keystroke", "return"),
    "tessura": ("keystroke", "return"),
    "kessura": ("keystroke", "return"),
    "pessera": ("keystroke", "return"),
    "kessera": ("keystroke", "return"),
    # ἐφήμερα — enter tray mode
    "ephemera": ("tray_enter", ""),
    # εὕρηκα — insert from tray
    "eureka": ("tray_insert", ""),
    # tab switching — Cmd+Shift+[ / Cmd+Shift+]
    "alpha": ("chord", "cmd+shift+["),
    "omega": ("chord", "cmd+shift+]"),
}


def match_voice_command(text: str) -> tuple[str, str] | None:
    """If *text* is a voice command, return (action, value). Else None."""
    normalized = text.strip().lower().rstrip(".,!?")
    return VOICE_COMMANDS.get(normalized)


def _is_repeated_keyword_phrase(text: str, keyword: str) -> bool:
    """True when *text* is one or more standalone repetitions of *keyword*."""
    normalized_keyword = keyword.strip().lower().rstrip(".,!?")
    tokens = [token.strip(".,!?") for token in text.strip().lower().split()]
    tokens = [token for token in tokens if token]
    return bool(tokens) and all(token == normalized_keyword for token in tokens)


class HandsFreeState(enum.Enum):
    DORMANT = "dormant"        # feature disabled
    LISTENING = "listening"    # wake word active, waiting for "listen"
    DICTATING = "dictating"    # mic open, VAD listening for speech
    TRANSCRIBING = "transcribing"  # utterance captured, transcription in flight


class HandsFreeController:
    """Manages the hands-free dictation lifecycle.

    The controller is instantiated once and lives for the app's lifetime.
    It coordinates WakeWordListener, AudioCapture, transcription clients,
    and the injection flow.

    Parameters
    ----------
    delegate : SpokeAppDelegate
        The app delegate — provides access to capture, transcription
        clients, injection, menubar, overlay.
    """

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._state = HandsFreeState.DORMANT
        self._wakeword = None
        self._saved_clipboard = None  # saved at dictation start, restored at stop

        # One-shot routing: if set, the next segment routes here then resets.
        # Values: None (normal cursor inject), "tray"
        self._next_segment_dest: str | None = None

        # Resolve wake word configuration
        self._access_key = os.environ.get("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY", "")
        self._wakeword_backend = os.environ.get("SPOKE_WAKEWORD_BACKEND", "porcupine").strip().lower()
        self._listen_keyword = os.environ.get("SPOKE_WAKEWORD_LISTEN", "computer")
        self._sleep_keyword = os.environ.get("SPOKE_WAKEWORD_SLEEP", "terminator")
        self._listen_ppn = os.environ.get("SPOKE_WAKEWORD_LISTEN_PPN")
        self._sleep_ppn = os.environ.get("SPOKE_WAKEWORD_SLEEP_PPN")
        self._listen_sensitivity = self._read_keyword_sensitivity(
            "SPOKE_WAKEWORD_LISTEN_SENSITIVITY"
        )
        self._sleep_sensitivity = self._read_keyword_sensitivity(
            "SPOKE_WAKEWORD_SLEEP_SENSITIVITY"
        )
        self._listen_model = os.environ.get("SPOKE_WAKEWORD_LISTEN_MODEL")
        self._sleep_model = os.environ.get("SPOKE_WAKEWORD_SLEEP_MODEL")
        self._tessera_model = os.environ.get("SPOKE_WAKEWORD_TESSERA_MODEL")
        self._pending_transcription_override_phrase: str | None = None
        self._pending_transcription_override_deadline = 0.0

        # Callbacks set by the delegate
        self.on_state_change: Callable[[HandsFreeState], None] | None = None

    @property
    def state(self) -> HandsFreeState:
        return self._state

    @property
    def is_active(self) -> bool:
        """True if hands-free is in any non-dormant state."""
        return self._state != HandsFreeState.DORMANT

    @property
    def is_dictating(self) -> bool:
        return self._state in (HandsFreeState.DICTATING, HandsFreeState.TRANSCRIBING)

    def _read_keyword_sensitivity(self, env_name: str) -> float | None:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r (expected 0.0-1.0)", env_name, raw)
            return None
        if not 0.0 <= value <= 1.0:
            logger.warning("Ignoring out-of-range %s=%r (expected 0.0-1.0)", env_name, raw)
            return None
        return value

    def _optional_existing_model_path(self, path: str | None, *, label: str) -> str | None:
        if not path:
            return None
        if os.path.exists(path):
            return path
        logger.warning("%s missing at %s — continuing without it", label, path)
        return None

    def _set_state(self, new_state: HandsFreeState) -> None:
        old = self._state
        self._state = new_state
        logger.info("Hands-free: %s → %s", old.value, new_state.value)
        if self.on_state_change is not None:
            self.on_state_change(new_state)

    # ── Enable / Disable ─────────────────────────────────────

    def enable(self) -> None:
        """Activate wake word listening. DORMANT → LISTENING."""
        if self._state != HandsFreeState.DORMANT:
            logger.warning("enable() called in state %s", self._state.value)
            return

        if self._wakeword_backend != "openwakeword" and not self._access_key:
            logger.error("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY not set — cannot enable hands-free")
            return

        from .wakeword import WakeWordListener
        sensitivities = None
        if self._listen_sensitivity is not None or self._sleep_sensitivity is not None:
            sensitivities = [
                self._listen_sensitivity if self._listen_sensitivity is not None else 0.5,
                self._sleep_sensitivity if self._sleep_sensitivity is not None else 0.5,
            ]
        tessera_model = self._optional_existing_model_path(
            self._tessera_model,
            label="Optional tessera wakeword model",
        )

        # Build keyword lists
        if self._wakeword_backend == "openwakeword":
            if not self._listen_model or not self._sleep_model:
                logger.error(
                    "openWakeWord backend selected but SPOKE_WAKEWORD_LISTEN_MODEL "
                    "or SPOKE_WAKEWORD_SLEEP_MODEL is not set"
                )
                return
            model_paths = [self._listen_model, self._sleep_model]
            if tessera_model:
                model_paths.append(tessera_model)
            self._wakeword = WakeWordListener(
                access_key="",
                backend="openwakeword",
                model_paths=model_paths,
                on_wake=self._on_wake_word,
            )
            self._keyword_map = {
                self._listen_model.rsplit("/", 1)[-1].rsplit(".", 1)[0].removesuffix("_model"): "listen",
                self._sleep_model.rsplit("/", 1)[-1].rsplit(".", 1)[0].removesuffix("_model"): "sleep",
            }
            if tessera_model:
                self._keyword_map[
                    tessera_model.rsplit("/", 1)[-1].rsplit(".", 1)[0].removesuffix("_model")
                ] = "tessera"
        elif self._listen_ppn and self._sleep_ppn:
            self._wakeword = WakeWordListener(
                access_key=self._access_key,
                backend="porcupine",
                keyword_paths=[self._listen_ppn, self._sleep_ppn],
                sensitivities=sensitivities,
                model_paths=[tessera_model] if tessera_model else None,
                on_wake=self._on_wake_word,
            )
            self._keyword_map = {
                self._listen_ppn: "listen",
                self._sleep_ppn: "sleep",
            }
            if tessera_model:
                self._keyword_map[
                    tessera_model.rsplit("/", 1)[-1].rsplit(".", 1)[0].removesuffix("_model")
                ] = "tessera"
        else:
            self._wakeword = WakeWordListener(
                access_key=self._access_key,
                backend="porcupine",
                keywords=[self._listen_keyword, self._sleep_keyword],
                sensitivities=sensitivities,
                model_paths=[tessera_model] if tessera_model else None,
                on_wake=self._on_wake_word,
            )
            self._keyword_map = {
                self._listen_keyword: "listen",
                self._sleep_keyword: "sleep",
            }
            if tessera_model:
                self._keyword_map[
                    tessera_model.rsplit("/", 1)[-1].rsplit(".", 1)[0].removesuffix("_model")
                ] = "tessera"

        try:
            self._wakeword.start()
        except Exception:
            logger.exception("Failed to start wake word listener")
            self._wakeword = None
            return

        self._set_state(HandsFreeState.LISTENING)

    def disable(self, *, reason: str = "unspecified") -> None:
        """Fully shut down hands-free mode. Any state → DORMANT."""
        if self._state == HandsFreeState.DORMANT:
            return
        logger.info(
            "Disabling hands-free: reason=%s state=%s",
            reason,
            self._state.value,
        )

        # Stop any active dictation capture
        if self._state in (HandsFreeState.DICTATING, HandsFreeState.TRANSCRIBING):
            self._stop_dictation_capture()

        # Stop wake word listener
        if self._wakeword is not None:
            self._wakeword.stop()
            self._wakeword = None

        self._set_state(HandsFreeState.DORMANT)

    # ── Wake word callback ───────────────────────────────────

    def _on_wake_word(self, keyword: str) -> None:
        """Called from the wakeword audio thread."""
        role = self._keyword_map.get(keyword, keyword)
        logger.info("Wake word role: %s (keyword=%s, state=%s)", role, keyword, self._state.value)

        # Marshal to main thread
        from Foundation import NSObject
        d = self._delegate
        d.performSelectorOnMainThread_withObject_waitUntilDone_(
            "handleWakeWord:", {"role": role}, False,
        )

    def handle_wake_word(self, role: str) -> None:
        """Process wake word on the main thread."""
        if role == "listen" and self._state == HandsFreeState.LISTENING:
            self._start_dictating()
        elif role == "sleep":
            if self._state in (HandsFreeState.DICTATING, HandsFreeState.TRANSCRIBING):
                self._stop_dictating()
            elif self._state == HandsFreeState.LISTENING:
                logger.info("Sleep wake word received while already passively listening")
        elif role == "tessera":
            self._arm_transcription_override("tessera")
            self._delegate.handsFreeInject_({"text": "tessera", "dest": "cursor"})

    # ── Dictation loop ───────────────────────────────────────

    def _start_dictating(self) -> None:
        """Begin continuous dictation. LISTENING → DICTATING."""
        logger.info("Starting hands-free dictation")

        # Save clipboard once at dictation start — restored when dictation stops.
        # Individual segment injections use inject_text_raw() to avoid
        # per-segment save/restore overhead (M1 clipboard thrashing fix).
        from .inject import save_pasteboard
        self._saved_clipboard = save_pasteboard()

        self._set_state(HandsFreeState.DICTATING)

        delegate = self._delegate
        capture = delegate._capture

        # Set up segment callback — each silence-bounded segment triggers
        # transcription and injection.
        def on_segment(wav_bytes: bytes):
            self._on_segment(wav_bytes)

        def on_vad_state(is_speech: bool):
            if delegate._menubar is not None:
                delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "updateVadState:", is_speech, False,
                )

        try:
            capture.start(
                amplitude_callback=delegate._on_amplitude,
                vad_state_callback=on_vad_state,
                segment_callback=on_segment,
            )
        except Exception:
            logger.exception("Failed to start audio capture for hands-free")
            self._set_state(HandsFreeState.LISTENING)
            return

        if delegate._glow is not None:
            delegate._glow.show()
        if delegate._menubar is not None:
            delegate._menubar.set_recording(True)

    def _stop_dictating(self) -> None:
        """Exit dictation mode. DICTATING/TRANSCRIBING → LISTENING."""
        logger.info("Stopping hands-free dictation")
        self._stop_dictation_capture()
        self._set_state(HandsFreeState.LISTENING)

    def _stop_dictation_capture(self) -> None:
        """Stop audio capture and clean up UI."""
        delegate = self._delegate
        capture = delegate._capture

        if capture.is_recording:
            capture.stop()

        # Restore clipboard saved at dictation start (M1 fix).
        if self._saved_clipboard is not None:
            from .inject import restore_pasteboard
            restore_pasteboard(self._saved_clipboard)
            self._saved_clipboard = None

        if delegate._glow is not None:
            delegate._glow.hide()
        if delegate._menubar is not None:
            delegate._menubar.set_recording(False)
            delegate._menubar.set_vad_state(False, False)

    # ── Segment handling ─────────────────────────────────────

    def _on_segment(self, wav_bytes: bytes) -> None:
        """Called from the capture encode worker thread when a silence boundary
        is detected. Transcribes the segment and marshals injection to the
        main thread."""
        if self._state != HandsFreeState.DICTATING:
            return

        # Capture and clear the one-shot routing destination
        dest = self._next_segment_dest
        self._next_segment_dest = None

        logger.info("Hands-free segment received (%d bytes, dest=%s)", len(wav_bytes), dest or "cursor")

        # Transcribe on this thread (encode worker), marshal result to main
        def _work():
            try:
                client = self._delegate._client
                text = client.transcribe(wav_bytes)
            except Exception:
                logger.exception("Hands-free segment transcription failed")
                return

            if text and text.strip():
                normalized = text.strip()
                if self._consume_pending_transcription_override(normalized):
                    return
                if _is_repeated_keyword_phrase(normalized, self._sleep_keyword):
                    self._delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "handleWakeWord:", {"role": "sleep"}, False,
                    )
                    return

                payload = {"text": normalized, "dest": dest or "cursor"}
                self._delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "handsFreeInject:", payload, False,
                )

        thread = threading.Thread(target=_work, daemon=True, name="hf-transcribe")
        thread.start()

    # ── One-shot routing ───────────────────────────────────────

    def route_next_segment(self, dest: str) -> None:
        """Tag the next segment to route to *dest* instead of cursor.
        Resets to cursor after that segment is dispatched."""
        logger.info("Hands-free: next segment → %s", dest)
        self._next_segment_dest = dest

    def _arm_transcription_override(self, phrase: str, window_s: float = 2.0) -> None:
        self._pending_transcription_override_phrase = phrase
        self._pending_transcription_override_deadline = time.monotonic() + window_s

    def _consume_pending_transcription_override(self, text: str) -> bool:
        phrase = self._pending_transcription_override_phrase
        if not phrase:
            return False
        if time.monotonic() > self._pending_transcription_override_deadline:
            self._pending_transcription_override_phrase = None
            self._pending_transcription_override_deadline = 0.0
            return False
        if _is_repeated_keyword_phrase(text, phrase):
            logger.info(
                "Hands-free: suppressing transcription %r because wakeword %r already fired",
                text,
                phrase,
            )
            self._pending_transcription_override_phrase = None
            self._pending_transcription_override_deadline = 0.0
            return True
        return False

    # ── Manual entry (e.g. long-press spacebar) ──────────────

    def toggle(self) -> None:
        """Toggle hands-free mode from any state."""
        if self._state == HandsFreeState.DORMANT:
            self.enable()
        elif self._state == HandsFreeState.LISTENING:
            self._start_dictating()
        elif self._state in (HandsFreeState.DICTATING, HandsFreeState.TRANSCRIBING):
            self._stop_dictating()
