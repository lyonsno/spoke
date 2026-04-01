"""Audio capture via sounddevice.

Records 16 kHz mono float32 audio and provides:
- Per-chunk RMS amplitude callback for visual feedback
- WAV byte output for transcription
- Growing-buffer snapshots for incremental transcription (Phase 3)
- Silence-sliced segment callbacks for opportunistic transcription (Phase 5)
"""

from __future__ import annotations

import io
import logging
import queue
import struct
import threading
import wave
from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 1024  # ~64 ms at 16 kHz — 16Hz amplitude updates

# VAD parameters for opportunistic slicing
MIN_SPEECH_FRAMES = 3      # ~192ms of consecutive speech to trigger
MIN_SILENCE_FRAMES = 12    # ~768ms of consecutive silence to slice
PRE_SPEECH_MARGIN = 6      # ~384ms padding before speech
NOISE_FLOOR_WINDOW = 50    # ~3.2s of history for noise floor
THRESHOLD_MULTIPLIER = 2.5
MIN_THRESHOLD = 0.001
MAX_SEGMENT_CHUNKS = 468   # ~30s maximum segment duration


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
        self._segment_cb: Callable[[bytes], None] | None = None
        self._read_cursor: int = 0  # index into _frames for incremental reads
        
        # VAD state
        self._is_speech: bool = False
        self._speech_trigger_count: int = 0
        self._silence_trigger_count: int = 0
        self._noise_floor_history: deque[float] = deque(maxlen=NOISE_FLOOR_WINDOW)
        self._current_segment_chunks: list[np.ndarray] = []
        self._ring_buffer: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_MARGIN)
        
        # Async encoding state
        self._encode_queue: queue.Queue | None = None
        self._encode_thread: threading.Thread | None = None

    def warmup(self) -> None:
        """Pre-initialize PortAudio so first start() is fast."""
        try:
            sd.query_devices()
            logger.info("PortAudio warmed up (%d devices)", len(sd.query_devices()))
        except Exception:
            logger.debug("PortAudio warmup failed", exc_info=True)

    def _close_stream(self) -> None:
        """Best-effort stream teardown for normal stop and failed starts."""
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            logger.debug("Audio stream stop failed during teardown", exc_info=True)
        try:
            stream.close()
        except Exception:
            logger.debug("Audio stream close failed during teardown", exc_info=True)

    def _reset_portaudio(self) -> None:
        """Best-effort PortAudio reset after a dead input stream."""
        try:
            sd._terminate()
        except Exception:
            logger.debug("PortAudio terminate failed", exc_info=True)
        try:
            sd._initialize()
        except Exception:
            logger.debug("PortAudio initialize failed", exc_info=True)

    def start(
        self, 
        amplitude_callback: Callable[[float], None] | None = None,
        segment_callback: Callable[[bytes], None] | None = None,
        vad_state_callback: Callable[[bool], None] | None = None
    ) -> None:
        """Begin recording.

        Parameters
        ----------
        amplitude_callback : callable, optional
            Called with RMS amplitude (float, 0.0–1.0) per audio chunk.
            Called from the PortAudio thread — keep it fast.
        segment_callback : callable, optional
            Called with bounded WAV bytes when a silence boundary is reached.
            Invoked from a background worker thread.
        """
        # Stop any existing stream to avoid leaking PortAudio resources
        if self._stream is not None:
            logger.warning("start() called while already recording — stopping previous stream")
            self._close_stream()

        self._frames = []
        self._read_cursor = 0
        self._amplitude_cb = amplitude_callback
        self._segment_cb = segment_callback
        self._vad_cb = vad_state_callback
        
        # Reset VAD state
        self._is_speech = False
        self._speech_trigger_count = 0
        self._silence_trigger_count = 0
        self._noise_floor_history.clear()
        self._current_segment_chunks = []
        self._speech_chunks = []
        self._ring_buffer.clear()
        
        if self._segment_cb is not None:
            self._encode_queue = queue.Queue()
            self._encode_thread = threading.Thread(target=self._encode_worker, daemon=True)
            self._encode_thread.start()

        last_exc: Exception | None = None
        for attempt in range(2):
            stream = None
            try:
                if attempt:
                    logger.warning("Retrying audio capture start after PortAudio reset")
                    self._reset_portaudio()
                stream = sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=BLOCKSIZE,
                    callback=self._audio_callback,
                )
                stream.start()
                self._stream = stream
                logger.info("Audio capture started")
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Audio capture start failed (attempt %d/2)",
                    attempt + 1,
                    exc_info=True,
                )
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug(
                            "Audio stream close failed after start error",
                            exc_info=True,
                        )
                self._stream = None
        
        self._amplitude_cb = None
        self._segment_cb = None
        if self._encode_queue is not None:
            self._encode_queue.put(None)
            self._encode_queue = None
            self._encode_thread = None
            
        assert last_exc is not None
        raise last_exc

    def stop(self) -> bytes:
        """Stop recording and return the complete audio as WAV bytes."""
        if self._stream is not None:
            self._close_stream()
            chunk_count = len(self._frames)
            logger.info("Audio capture stopped (%d chunks)", chunk_count)
            if chunk_count == 0:
                logger.warning(
                    "Audio capture produced zero chunks — resetting PortAudio"
                )
                self._reset_portaudio()
            
            # Emit final segment if we were in the middle of speech
            if self._segment_cb is not None and self._is_speech and self._current_segment_chunks:
                segment_samples = np.concatenate(self._current_segment_chunks)
                if self._encode_queue is not None:
                    self._encode_queue.put(segment_samples)
                    
            if self._encode_queue is not None:
                self._encode_queue.put(None)  # Sentinel to stop thread
            if self._encode_thread is not None:
                self._encode_thread.join(timeout=1.0)

            self._encode_queue = None
            self._encode_thread = None
            self._current_segment_chunks = []
            self._ring_buffer.clear()
            self._noise_floor_history.clear()

            self._amplitude_cb = None
            self._segment_cb = None
            
            # Use trimmed chunks if available
            speech_chunks = getattr(self, "_speech_chunks", None)
            if speech_chunks is not None and len(speech_chunks) > 0:
                final_chunks = list(speech_chunks)
                if self._is_speech:
                    final_chunks.extend(self._current_segment_chunks)
                wav_bytes = self._encode_wav(np.concatenate(final_chunks))
            else:
                wav_bytes = self._encode_wav(self._get_all_frames())

            if hasattr(self, '_speech_chunks'):
                self._speech_chunks = []
            return wav_bytes

        # No active stream — clear any stale frames and return empty
        self._frames = []
        self._amplitude_cb = None
        self._segment_cb = None
        self._current_segment_chunks = []
        self._ring_buffer.clear()
        self._noise_floor_history.clear()
        
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
    
    def _encode_worker(self) -> None:
        """Background thread to encode and dispatch segments to avoid blocking PortAudio."""
        while True:
            if self._encode_queue is None:
                break
            try:
                samples = self._encode_queue.get()
                if samples is None:
                    break
                
                if self._segment_cb is not None:
                    wav_bytes = self._encode_wav(samples)
                    self._segment_cb(wav_bytes)
            except Exception:
                logger.error("Error in segment callback worker", exc_info=True)

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

        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if self._amplitude_cb is not None:
            self._amplitude_cb(rms)
            
        if self._segment_cb is not None or hasattr(self, '_vad_cb'):
            # Ensure it runs if either callback is present!
            self._noise_floor_history.append(rms)
            noise_floor = min(self._noise_floor_history) if self._noise_floor_history else 0.0
            threshold = max(noise_floor * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
            
            # Debug log every 50 chunks (~3 seconds)
            if len(self._frames) % 50 == 0:
                logger.info(f"VAD tick: rms={rms:.4f}, thr={threshold:.4f}, is_speech={self._is_speech}, trigger={self._speech_trigger_count}/{MIN_SPEECH_FRAMES}")

            if not self._is_speech:
                self._ring_buffer.append(chunk)

                if rms > threshold:
                    self._speech_trigger_count += 1
                    if self._speech_trigger_count >= MIN_SPEECH_FRAMES:
                        self._is_speech = True
                        if hasattr(self, '_vad_cb') and self._vad_cb is not None:
                            self._vad_cb(True)
                        self._current_segment_chunks.extend(self._ring_buffer)
                        if getattr(self, '_speech_chunks', None) is None:
                            self._speech_chunks = []
                        self._speech_chunks.extend(self._ring_buffer)
                        self._ring_buffer.clear()
                else:
                    self._speech_trigger_count = 0
            else:
                self._current_segment_chunks.append(chunk)
                if getattr(self, '_speech_chunks', None) is None:
                    self._speech_chunks = []
                self._speech_chunks.append(chunk)
                
                force_slice = len(self._current_segment_chunks) >= MAX_SEGMENT_CHUNKS
                
                if force_slice or rms <= threshold:
                    if rms <= threshold:
                        self._silence_trigger_count += 1
                    else:
                        self._silence_trigger_count = MIN_SILENCE_FRAMES  # Force immediate slice
                        
                    if self._silence_trigger_count >= MIN_SILENCE_FRAMES:
                        self._is_speech = False
                        if hasattr(self, '_vad_cb') and self._vad_cb is not None:
                            self._vad_cb(False)
                        self._speech_trigger_count = 0
                        self._silence_trigger_count = 0
                        
                        if self._encode_queue is not None:
                            segment_samples = np.concatenate(self._current_segment_chunks)
                            self._encode_queue.put(segment_samples)
                            
                        self._current_segment_chunks = []
                else:
                    self._silence_trigger_count = 0

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
