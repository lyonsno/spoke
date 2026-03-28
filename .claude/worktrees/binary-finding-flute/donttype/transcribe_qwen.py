"""Local Qwen3 ASR transcription via mlx-qwen3-asr.

Runs Qwen3-ASR in-process on Apple Silicon via MLX. Downloads the model
from HuggingFace on first use. Decodes WAV in-process via numpy to
avoid requiring ffmpeg.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
import mlx_qwen3_asr

from .dedup import truncate_repetition, is_hallucination

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


class LocalQwenClient:
    """Transcribe audio locally via mlx-qwen3-asr.

    Parameters
    ----------
    model : str
        HuggingFace model identifier. Downloaded on first use.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._session = None

    def _ensure_session(self):
        """Lazy-init the Session so model loads on first transcribe, not import."""
        if self._session is not None:
            return
        logger.info("Loading Qwen3 ASR model %s (first use may download ~1.2GB)", self._model)
        self._session = mlx_qwen3_asr.Session(model=self._model)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text.

        Decodes WAV to numpy float32 array in-process (no ffmpeg needed),
        then passes directly to the Qwen3 ASR session.
        """
        if not wav_bytes:
            return ""

        self._ensure_session()

        audio = self._decode_wav(wav_bytes)
        result = self._session.transcribe(audio, language="English")

        text = result.text.strip()
        text = truncate_repetition(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info("Qwen3 transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes to float32 numpy array at 16kHz mono."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels")
            if wf.getsampwidth() != 2:
                raise ValueError(f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit")
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0

    def close(self) -> None:
        """Release the model session."""
        self._session = None
