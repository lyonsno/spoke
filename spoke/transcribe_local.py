"""Local MLX Whisper transcription — no server required.

Runs mlx-whisper in-process on the local machine. Downloads the model
from HuggingFace on first use. Decodes WAV in-process via numpy to
avoid requiring ffmpeg.
"""

from __future__ import annotations

import importlib
import io
import logging
import wave

import mlx.core as mx
import numpy as np
import mlx_whisper
from mlx_whisper.load_models import load_model

from .dedup import truncate_repetition, is_hallucination

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class LocalTranscriptionClient:
    """Transcribe audio locally via mlx-whisper.

    Parameters
    ----------
    model : str
        HuggingFace model identifier. Downloaded on first use.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._loaded = False
        self._model_instance = None

    def _ensure_model(self) -> None:
        """Trigger model download if not cached."""
        self.prepare()

    def _install_model_holder(self) -> None:
        """Point mlx-whisper's singleton cache at this client's loaded model."""
        if self._model_instance is None:
            raise RuntimeError("Whisper model was not loaded before installation")
        transcribe_module = importlib.import_module("mlx_whisper.transcribe")
        transcribe_module.ModelHolder.model = self._model_instance
        transcribe_module.ModelHolder.model_path = self._model

    def prepare(self) -> None:
        """Warm the Whisper model cache without running a transcription."""
        if self._loaded:
            self._install_model_holder()
            return
        logger.info("Preloading Whisper model %s", self._model)
        self._model_instance = load_model(self._model, dtype=mx.float16)
        self._loaded = True
        self._install_model_holder()

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text.

        Decodes WAV to numpy float32 array in-process (no ffmpeg needed),
        then passes directly to mlx_whisper.transcribe().
        """
        if not wav_bytes:
            return ""

        self._ensure_model()
        self._install_model_holder()

        # Decode WAV bytes to float32 numpy array — bypass ffmpeg entirely
        audio = self._decode_wav(wav_bytes)

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language="en",
            decode_timeout=30.0,
        )

        text = result.get("text", "").strip()
        text = truncate_repetition(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info("Local transcription: %r (%d bytes audio)", text, len(wav_bytes))
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
        """No-op — no persistent resources to clean up."""
        pass
