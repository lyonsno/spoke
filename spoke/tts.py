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
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TOP_K = 50
_DEFAULT_TOP_P = 0.95


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
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
        top_p: float = _DEFAULT_TOP_P,
    ):
        self._model_id = model_id
        self._voice = voice
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._model = None
        self._cancelled = False
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None

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
        temperature = float(os.environ.get("SPOKE_TTS_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
        top_k = int(os.environ.get("SPOKE_TTS_TOP_K", str(_DEFAULT_TOP_K)))
        top_p = float(os.environ.get("SPOKE_TTS_TOP_P", str(_DEFAULT_TOP_P)))
        return cls(model_id=model_id, voice=voice, temperature=temperature, top_k=top_k, top_p=top_p)

    def _ensure_model(self):
        """Load the model on first use."""
        if self._model is None:
            logger.info("Loading TTS model %s …", self._model_id)
            self._model = tts_load(self._model_id)
            logger.info("TTS model loaded.")

    def warm(self) -> None:
        """Pre-load the model in a background thread so first speak() is fast."""
        threading.Thread(target=self._ensure_model, daemon=True).start()

    def speak(self, text: str) -> None:
        """Generate speech and play it synchronously.  Blocks until done.

        Uses a dedicated OutputStream so cancel() only kills TTS playback
        without interfering with microphone recording via sounddevice.
        """
        if not text:
            return
        if self._cancelled:
            return

        self._ensure_model()

        import numpy as np

        for result in self._model.generate(
            text=text,
            voice=self._voice,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        ):
            if self._cancelled:
                return
            audio = np.array(result.audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            done = threading.Event()
            stream = sd.OutputStream(
                samplerate=result.sample_rate,
                channels=audio.shape[1],
                dtype="float32",
                finished_callback=lambda: done.set(),
            )
            self._stream = stream
            stream.start()
            stream.write(audio)
            # Wait for playback to finish or cancellation
            while not done.is_set():
                if self._cancelled:
                    break
                done.wait(timeout=0.05)
            stream.stop()
            stream.close()
            self._stream = None

    def speak_async(self, text: str) -> threading.Thread:
        """Generate and play speech on a background daemon thread."""
        self._cancelled = False
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def cancel(self) -> None:
        """Cancel any in-flight or future speak() call and stop playback.

        Only stops the TTS output stream — does not touch global sounddevice
        state, so microphone recording is unaffected.
        """
        self._cancelled = True
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except sd.PortAudioError:
                pass
