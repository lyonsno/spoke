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
from typing import Callable, Optional

import numpy as np
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
        gpu_lock: threading.Lock | None = None,
    ):
        self._model_id = model_id
        self._voice = voice
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._model = None
        self._cancelled = False
        self._lock = threading.Lock()
        self._gpu_lock = gpu_lock
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
        def _warm():
            from contextlib import nullcontext
            lock_ctx = self._gpu_lock if self._gpu_lock is not None else nullcontext()
            with lock_ctx:
                self._ensure_model()
        threading.Thread(target=_warm, daemon=True).start()

    def speak(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Generate speech and play it synchronously.  Blocks until done.

        Holds gpu_lock during model.generate() to prevent concurrent MLX
        inference (which crashes Metal).  Releases the lock before audio
        playback so Whisper can proceed while audio plays.

        If amplitude_callback is provided, it is called with the RMS value
        of each ~64ms audio chunk during playback (same interface as the
        microphone amplitude callback used by the glow overlay).
        """
        if not text:
            return
        if self._cancelled:
            return

        from contextlib import nullcontext

        lock_ctx = self._gpu_lock if self._gpu_lock is not None else nullcontext()

        with lock_ctx:
            self._ensure_model()
            if self._cancelled:
                return
            # Generate audio while holding the GPU lock
            results = []
            for result in self._model.generate(
                text=text,
                voice=self._voice,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            ):
                if self._cancelled:
                    return
                results.append(result)
        # GPU lock released — play audio without blocking Whisper
        for result in results:
            if self._cancelled:
                return
            audio = np.array(result.audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            sr = result.sample_rate
            # ~64ms chunks for amplitude updates (matches mic capture cadence)
            chunk_size = int(sr * 0.064)

            done = threading.Event()
            stream = sd.OutputStream(
                samplerate=sr,
                channels=audio.shape[1],
                dtype="float32",
                finished_callback=lambda: done.set(),
            )
            self._stream = stream
            stream.start()

            # Write in chunks, emitting RMS for each
            offset = 0
            while offset < len(audio):
                if self._cancelled:
                    break
                end = min(offset + chunk_size, len(audio))
                chunk = audio[offset:end]
                stream.write(chunk)
                if amplitude_callback is not None:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    amplitude_callback(rms)
                offset = end

            # Wait for remaining audio to finish playing
            while not done.is_set():
                if self._cancelled:
                    break
                done.wait(timeout=0.05)
            stream.stop()
            stream.close()
            self._stream = None

            # Signal zero amplitude when playback ends
            if amplitude_callback is not None:
                amplitude_callback(0.0)

    def speak_async(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
        done_callback: Callable[[], None] | None = None,
    ) -> threading.Thread:
        """Generate and play speech on a background daemon thread."""
        self._cancelled = False
        def _run():
            self.speak(text, amplitude_callback=amplitude_callback)
            if done_callback is not None:
                done_callback()
        t = threading.Thread(target=_run, daemon=True)
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
