"""Hands-free continuous dictation mode.

Coordinates wake word detection (Porcupine), audio capture (VAD-based
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
        self._listen_keyword = os.environ.get("SPOKE_WAKEWORD_LISTEN", "computer")
        self._sleep_keyword = os.environ.get("SPOKE_WAKEWORD_SLEEP", "terminator")
        self._listen_ppn = os.environ.get("SPOKE_WAKEWORD_LISTEN_PPN")
        self._sleep_ppn = os.environ.get("SPOKE_WAKEWORD_SLEEP_PPN")

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

        if not self._access_key:
            logger.error("SPOKE_PICOVOICE_PORCUPINE_ACCESS_KEY not set — cannot enable hands-free")
            return

        from .wakeword import WakeWordListener

        # Build keyword lists
        if self._listen_ppn and self._sleep_ppn:
            self._wakeword = WakeWordListener(
                access_key=self._access_key,
                keyword_paths=[self._listen_ppn, self._sleep_ppn],
                on_wake=self._on_wake_word,
            )
            self._keyword_map = {
                self._listen_ppn: "listen",
                self._sleep_ppn: "sleep",
            }
        else:
            self._wakeword = WakeWordListener(
                access_key=self._access_key,
                keywords=[self._listen_keyword, self._sleep_keyword],
                on_wake=self._on_wake_word,
            )
            self._keyword_map = {
                self._listen_keyword: "listen",
                self._sleep_keyword: "sleep",
            }

        try:
            self._wakeword.start()
        except Exception:
            logger.exception("Failed to start wake word listener")
            self._wakeword = None
            return

        self._set_state(HandsFreeState.LISTENING)

    def disable(self) -> None:
        """Fully shut down hands-free mode. Any state → DORMANT."""
        if self._state == HandsFreeState.DORMANT:
            return

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
        """Called from the Porcupine audio thread."""
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
        elif role == "sleep" and self._state != HandsFreeState.DORMANT:
            self.disable()

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
                normalized = text.strip().lower().rstrip(".,!?")
                if normalized == self._sleep_keyword.strip().lower():
                    self._delegate.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "handleWakeWord:", {"role": "sleep"}, False,
                    )
                    return

                payload = {"text": text.strip(), "dest": dest or "cursor"}
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

    # ── Manual entry (e.g. long-press spacebar) ──────────────

    def toggle(self) -> None:
        """Toggle hands-free mode from any state."""
        if self._state == HandsFreeState.DORMANT:
            self.enable()
        elif self._state == HandsFreeState.LISTENING:
            self._start_dictating()
        elif self._state in (HandsFreeState.DICTATING, HandsFreeState.TRANSCRIBING):
            self._stop_dictating()
