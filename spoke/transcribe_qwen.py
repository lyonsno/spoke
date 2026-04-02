"""Local Qwen3 ASR transcription via mlx-qwen3-asr.

Runs Qwen3-ASR in-process on Apple Silicon via MLX. Downloads the model
from HuggingFace on first use. Supports both batch transcription (full
buffer) and streaming transcription (incremental KV-cache reuse).
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

from .dedup import truncate_repetition, is_hallucination

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"

mlx_qwen3_asr = None


def _get_mlx_qwen3_asr():
    global mlx_qwen3_asr
    if mlx_qwen3_asr is None:
        import spoke.patch_qwen3_streaming  # noqa: F401

        mlx_qwen3_asr = __import__("mlx_qwen3_asr")
    return mlx_qwen3_asr


class LocalQwenClient:
    """Transcribe audio locally via mlx-qwen3-asr.

    Supports two modes:
    - Batch: transcribe(wav_bytes) for full-buffer transcription
    - Streaming: start_stream() / feed(frames) / finish_stream() for
      incremental transcription with KV-cache reuse

    Parameters
    ----------
    model : str
        HuggingFace model identifier. Downloaded on first use.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._session = None
        self._stream_state = None
        self._last_preview_text = ""

    def _ensure_session(self):
        """Lazy-init the Session so model loads on first transcribe, not import."""
        if self._session is not None:
            return
        logger.info("Loading Qwen3 ASR model %s (first use may download ~1.2GB)", self._model)
        self._session = _get_mlx_qwen3_asr().Session(model=self._model)

    def prepare(self) -> None:
        """Warm the Qwen session without starting a transcription."""
        self._ensure_session()

    # ── Batch transcription ──────────────────────────────────

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

    # ── Streaming transcription (KV-cache reuse) ─────────────

    def start_stream(self) -> None:
        """Initialize a new streaming transcription session.

        Call feed() to incrementally add audio, then finish_stream()
        to finalize and get the complete transcription.
        """
        self._ensure_session()
        self._last_preview_text = ""
        self._stream_state = self._session.init_streaming(
            language="English",
            chunk_size_sec=1.5,
        )
        logger.info("Qwen3 streaming session started")

    def feed(self, frames: np.ndarray) -> str:
        """Feed new audio frames and return current transcription.

        Parameters
        ----------
        frames : np.ndarray
            Float32 mono audio at 16kHz. Only new frames since last feed().

        Returns
        -------
        str
            Current best transcription (may revise trailing tokens).
        """
        if self._stream_state is None:
            return ""
        if frames.size == 0:
            return self._stream_state.text.strip() if self._stream_state.text else ""

        self._stream_state = self._session.feed_audio(frames, self._stream_state)

        text = self._stream_state.text.strip() if self._stream_state.text else ""
        text = truncate_repetition(text)
        if is_hallucination(text):
            return ""
        if text and text != self._last_preview_text:
            self._last_preview_text = text
            logger.info("Qwen3 stream preview: %r", text)
        return text

    def finish_stream(self) -> str:
        """Finalize the streaming session and return the complete transcription.

        Runs a tail refinement pass on any remaining buffered audio.
        """
        if self._stream_state is None:
            return ""

        self._stream_state = self._session.finish_streaming(self._stream_state)
        text = self._stream_state.text.strip() if self._stream_state.text else ""
        self._stream_state = None

        text = truncate_repetition(text)
        if is_hallucination(text):
            logger.info("Discarding hallucination from stream: %r", text)
            return ""
        logger.info("Qwen3 stream finalized: %r", text)
        return text

    def cancel_stream(self) -> None:
        """Abandon any active streaming session without a tail refinement pass."""
        if self._stream_state is None:
            return
        logger.info("Qwen3 stream cancelled")
        self._stream_state = None
        self._last_preview_text = ""

    @property
    def has_active_stream(self) -> bool:
        """Whether a streaming session is currently in progress."""
        return self._stream_state is not None

    @property
    def supports_streaming(self) -> bool:
        """Whether this client supports incremental streaming."""
        return True

    # ── Helpers ───────────────────────────────────────────────

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
        """Release the model session and any streaming state."""
        self._stream_state = None
        self._session = None
