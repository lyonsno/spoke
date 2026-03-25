"""Audio capture via sounddevice.

Records 16 kHz mono float32 audio and provides:
- Per-chunk RMS amplitude callback for visual feedback
- WAV byte output for transcription
- Growing-buffer snapshots for incremental transcription (Phase 3)
"""

from __future__ import annotations

import io
import logging
import struct
import threading
import wave
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 1024  # ~64 ms at 16 kHz — 16Hz amplitude updates


class AudioCapture:
    """Record audio from the default input device.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz. Default 16000 (optimal for Whisper).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._amplitude_cb: Callable[[float], None] | None = None
        self._read_cursor: int = 0  # index into _frames for incremental reads

    def warmup(self) -> None:
        """Pre-initialize PortAudio so first start() is fast."""
        try:
            sd.query_devices()
            logger.info("PortAudio warmed up (%d devices)", len(sd.query_devices()))
        except Exception:
            logger.debug("PortAudio warmup failed", exc_info=True)

    def start(self, amplitude_callback: Callable[[float], None] | None = None) -> None:
        """Begin recording.

        Parameters
        ----------
        amplitude_callback : callable, optional
            Called with RMS amplitude (float, 0.0–1.0) per audio chunk.
            Called from the PortAudio thread — keep it fast.
        """
        # Stop any existing stream to avoid leaking PortAudio resources
        if self._stream is not None:
            logger.warning("start() called while already recording — stopping previous stream")
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._frames = []
        self._read_cursor = 0
        self._amplitude_cb = amplitude_callback
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio capture started")

    def stop(self) -> bytes:
        """Stop recording and return the complete audio as WAV bytes."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped (%d chunks)", len(self._frames))
            return self._encode_wav(self._get_all_frames())

        # No active stream — clear any stale frames and return empty
        self._frames = []
        return b""

    def get_buffer(self) -> bytes:
        """Return the current recording buffer as WAV bytes (non-destructive).

        Used for incremental transcription — returns everything recorded so far.
        """
        return self._encode_wav(self._get_all_frames())

    def get_new_frames(self) -> np.ndarray:
        """Return audio frames accumulated since the last call (incremental).

        Returns a float32 numpy array at 16kHz mono. Returns an empty array
        if no new frames are available. Advances an internal cursor so each
        chunk is returned exactly once.
        """
        with self._lock:
            if self._read_cursor >= len(self._frames):
                return np.array([], dtype=np.float32)
            new = self._frames[self._read_cursor:]
            self._read_cursor = len(self._frames)
        return np.concatenate(new)

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    # ── private ─────────────────────────────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by PortAudio on its own thread. Must be fast."""
        if status:
            logger.warning("sounddevice status: %s", status)

        chunk = indata[:, 0].copy()  # mono, float32

        with self._lock:
            self._frames.append(chunk)

        if self._amplitude_cb is not None:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            self._amplitude_cb(rms)

    def _get_all_frames(self) -> np.ndarray:
        """Concatenate all captured frames into a single array."""
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._frames)

    def _encode_wav(self, samples: np.ndarray) -> bytes:
        """Encode float32 samples as 16-bit PCM WAV bytes."""
        if samples.size == 0:
            return b""

        # Convert float32 [-1, 1] to int16
        pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm.tobytes())

        return buf.getvalue()
