"""Cohere Transcribe via mlx-audio STT.

Runs CohereLabs/cohere-transcribe-03-2026 in-process on Apple Silicon via MLX.
~1.5 GB memory, 1.97s for 3 minutes of speech (94x RTF) on M4 Max.

Requires mlx-audio with STT support (installed via the [tts] extra which
pulls in mlx-audio, and the stt subpackage is included).

The model is gated on HuggingFace (Apache 2.0). First use downloads ~1.5 GB.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from .dedup import truncate_repetition, is_hallucination

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
_SAMPLE_RATE = 16000


def _load_stt_model(model_id: str):
    """Deferred import + load so mlx-audio is only required at runtime."""
    from mlx_audio.stt.generate import load_model

    return load_model(model_id)


class LocalCohereClient:
    """Transcribe audio locally via Cohere Transcribe on MLX.

    Parameters
    ----------
    model : str
        HuggingFace model identifier. Downloaded on first use.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model_id = model
        self._model = None
        self._loaded = False

    def prepare(self) -> None:
        """Load the model. First call may download ~1.5 GB."""
        if self._loaded:
            return
        logger.info(
            "Loading Cohere Transcribe model %s (first use may download ~1.5 GB)",
            self._model_id,
        )
        self._model = _load_stt_model(self._model_id)
        self._loaded = True
        logger.info("Cohere Transcribe model ready")

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text.

        Decodes WAV to numpy float32 array in-process (no ffmpeg needed),
        then passes directly to the Cohere model.
        """
        if not wav_bytes:
            return ""

        if not self._loaded:
            self.prepare()

        if self._model is None:
            logger.warning("Cohere model not available — returning empty string")
            return ""

        audio = self._decode_wav(wav_bytes)

        try:
            results = self._model.transcribe(
                language="english",
                audio_arrays=[audio],
                sample_rates=[_SAMPLE_RATE],
            )
        except Exception:
            logger.warning("Cohere transcription failed", exc_info=True)
            return ""

        if not results:
            return ""

        text = results[0].strip()
        text = truncate_repetition(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info(
            "Cohere transcription: %r (%d bytes audio)", text, len(wav_bytes)
        )
        return text

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes to float32 numpy array at 16kHz mono."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(
                    f"Expected mono audio, got {wf.getnchannels()} channels"
                )
            if wf.getsampwidth() != 2:
                raise ValueError(
                    f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit"
                )
            frames = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(frames, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0

    def close(self) -> None:
        """Release the model."""
        self._model = None
        self._loaded = False
