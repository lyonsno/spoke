"""Local MLX Whisper transcription — no server required.

Runs mlx-whisper in-process on the local machine. Downloads the model
from HuggingFace on first use.
"""

from __future__ import annotations

import io
import logging
import tempfile
import os

import mlx_whisper

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

    def _ensure_model(self) -> None:
        """Trigger model download if not cached."""
        if self._loaded:
            return
        logger.info("Loading model %s (first use may download ~400MB)", self._model)
        self._loaded = True

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text.

        Writes to a temp file because mlx_whisper expects a file path.
        """
        if not wav_bytes:
            return ""

        self._ensure_model()

        # mlx_whisper.transcribe() expects a file path, not bytes
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            tmp.write(wav_bytes)
            tmp.close()

            result = mlx_whisper.transcribe(
                tmp.name,
                path_or_hf_repo=self._model,
                language="en",
            )

            text = result.get("text", "").strip()
            logger.info("Local transcription: %r (%d bytes audio)", text, len(wav_bytes))
            return text
        finally:
            os.unlink(tmp.name)

    def close(self) -> None:
        """No-op — no persistent resources to clean up."""
        pass
