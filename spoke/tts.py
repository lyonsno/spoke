"""Local text-to-speech via Voxtral on MLX.

Loads a Voxtral TTS model lazily and plays generated audio through
sounddevice.  Designed to be driven from the command-completion pathway
in __main__.py — speak_async() returns immediately, running generation
and playback on a background thread.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
_DEFAULT_VOICE = "casual_female"


def tts_load(model_id: str):
    """Load a Voxtral TTS model.  Separated for easy patching in tests."""
    from mlx_audio.tts import load
    return load(model_id)


class TTSClient:
    """Lazy-loading TTS client with cancellation support."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        voice: str = _DEFAULT_VOICE,
    ):
        self._model_id = model_id
        self._voice = voice
        self._model = None
        self._cancelled = False
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Optional["TTSClient"]:
        """Create a TTSClient from environment variables, or None if disabled.

        Set SPOKE_TTS_VOICE to enable TTS (e.g. "casual_female").
        Optionally set SPOKE_TTS_MODEL to override the default model.
        """
        voice = os.environ.get("SPOKE_TTS_VOICE")
        if not voice:
            return None
        model_id = os.environ.get("SPOKE_TTS_MODEL", _DEFAULT_MODEL_ID)
        return cls(model_id=model_id, voice=voice)

    def _ensure_model(self):
        """Load the model on first use."""
        if self._model is None:
            logger.info("Loading TTS model %s …", self._model_id)
            self._model = tts_load(self._model_id)
            logger.info("TTS model loaded.")

    def speak(self, text: str) -> None:
        """Generate speech and play it synchronously.  Blocks until done."""
        if not text:
            return
        if self._cancelled:
            return

        self._ensure_model()

        for result in self._model.generate(text=text, voice=self._voice):
            if self._cancelled:
                return
            sd.play(result.audio, result.sample_rate)
            sd.wait()

    def speak_async(self, text: str) -> threading.Thread:
        """Generate and play speech on a background daemon thread."""
        self._cancelled = False
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def cancel(self) -> None:
        """Cancel any in-flight or future speak() call and stop playback."""
        self._cancelled = True
        sd.stop()
