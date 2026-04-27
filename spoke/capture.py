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
import time
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
SILERO_CHUNK = 512  # Silero VAD requires exactly 512 samples at 16kHz

# VAD parameters for opportunistic slicing (Silero-based)
SPEECH_PROB_THRESHOLD = 0.5   # Silero probability threshold for speech
MIN_SPEECH_FRAMES = 3         # ~192ms (3 × 64ms callbacks) to confirm speech
MIN_SILENCE_FRAMES = 12       # ~768ms (12 × 64ms callbacks) to confirm silence
PRE_SPEECH_MARGIN = 6         # ~384ms padding before speech (in BLOCKSIZE chunks)
MAX_SEGMENT_CHUNKS = 468      # ~30s maximum segment duration (in BLOCKSIZE chunks)
VAD_GRACE_PERIOD_SECS = 5.0
NON_SILENT_SAMPLE_THRESHOLD = 1e-4

# Path to cached Silero VAD JIT model
_SILERO_VAD_JIT_PATHS = [
    "/Users/noahlyons/.cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.jit",
]


def _load_silero_vad():
    """Load Silero VAD JIT model. Returns (model, sample_rate_tensor) or (None, None)."""
    try:
        import torch
    except ImportError:
        logger.warning("torch not available — Silero VAD disabled")
        return None, None

    for path in _SILERO_VAD_JIT_PATHS:
        try:
            model = torch.jit.load(path)
            model.eval()
            sr = torch.tensor(SAMPLE_RATE)
            model(torch.zeros(1, SILERO_CHUNK), sr)
            model.reset_states()
            logger.info("Silero VAD loaded from %s", path)
            return model, sr
        except Exception:
            continue

    try:
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        sr = torch.tensor(SAMPLE_RATE)
        model(torch.zeros(1, SILERO_CHUNK), sr)
        model.reset_states()
        logger.info("Silero VAD loaded from torch hub")
        return model, sr
    except Exception:
        logger.warning("Failed to load Silero VAD — VAD disabled", exc_info=True)
        return None, None


def _has_non_silent_samples(chunk: np.ndarray) -> bool:
    return bool(np.any(np.abs(chunk) > NON_SILENT_SAMPLE_THRESHOLD))


class AudioCapture:
    """Record audio from the default input device.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz. Default 16000 (optimal for Whisper).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, metrics=None) -> None:
        self._sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._metrics = metrics
        self._amplitude_cb: Callable[[float], None] | None = None
        self._segment_cb: Callable[[bytes], None] | None = None
        self._vad_cb: Callable[[bool], None] | None = None
        self._read_cursor: int = 0  # index into _frames for incremental reads
        
        # VAD state (Silero-based)
        self._is_speech: bool = False
        self._speech_trigger_count: int = 0
        self._silence_trigger_count: int = 0
        self._current_segment_chunks: list[np.ndarray] = []
        self._ring_buffer: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_MARGIN)
        self._speech_ring_buffer: deque[tuple[np.ndarray, bool]] = deque(maxlen=PRE_SPEECH_MARGIN)
        self._grace_chunks_remaining: int = 0

        # Silero VAD model (loaded once, reused across recordings)
        self._silero_model, self._silero_sr = _load_silero_vad()
        self._torch = None
        self._silero_warned = False
        if self._silero_model is not None:
            import torch
            self._torch = torch
        
        # Async encoding state
        self._encode_queue: queue.Queue | None = None
        self._encode_thread: threading.Thread | None = None
        self._callback_queue: queue.Queue | None = None
        self._callback_thread: threading.Thread | None = None
        self._callbacks_enabled = False
        self._callback_generation = 0
        self._stream_closing = False

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
        self._stream_closing = True
        self._stream = None
        if stream is None:
            self._stream_closing = False
            return
        try:
            stream.stop()
        except Exception:
            logger.debug("Audio stream stop failed during teardown", exc_info=True)
        try:
            stream.close()
        except Exception:
            logger.debug("Audio stream close failed during teardown", exc_info=True)
        finally:
            self._stream_closing = False

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

    def _stop_callback_dispatch(self) -> None:
        self._callbacks_enabled = False
        queue_ref = self._callback_queue
        thread_ref = self._callback_thread
        if queue_ref is not None:
            while True:
                try:
                    queue_ref.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        queue_ref.get_nowait()
                    except queue.Empty:
                        break
        if thread_ref is not None:
            thread_ref.join(timeout=1.0)
        self._callback_queue = None
        self._callback_thread = None

    def _queue_callback_event(self, kind: str, payload: object) -> None:
        queue_ref = self._callback_queue
        if queue_ref is None:
            return
        try:
            queue_ref.put_nowait((self._callback_generation, kind, payload))
        except queue.Full:
            if kind == "amplitude":
                return
            try:
                queue_ref.get_nowait()
            except queue.Empty:
                return
            try:
                queue_ref.put_nowait((self._callback_generation, kind, payload))
            except queue.Full:
                pass

    def _callback_worker(self) -> None:
        """Dispatch UI-bound callbacks off the PortAudio thread."""
        while True:
            queue_ref = self._callback_queue
            if queue_ref is None:
                break
            try:
                item = queue_ref.get()
            except Exception:
                logger.error("Error in callback dispatch worker", exc_info=True)
                continue

            if item is None:
                break

            generation, kind, payload = item
            try:
                if not self._callbacks_enabled or generation != self._callback_generation:
                    continue
                if kind == "amplitude" and self._amplitude_cb is not None:
                    self._amplitude_cb(float(payload))
                elif kind == "vad" and self._vad_cb is not None:
                    self._vad_cb(bool(payload))
            except Exception:
                logger.error("Error in callback dispatch worker", exc_info=True)

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

        self._stop_callback_dispatch()
        self._callback_generation += 1
        self._frames = []
        self._read_cursor = 0
        self._stream_closing = False
        self._amplitude_cb = amplitude_callback
        self._segment_cb = segment_callback
        self._vad_cb = vad_state_callback
        
        # Reset VAD state
        self._is_speech = False
        self._speech_trigger_count = 0
        self._silence_trigger_count = 0
        self._current_segment_chunks = []
        self._speech_chunks = []
        self._ring_buffer.clear()
        self._speech_ring_buffer.clear()
        self._flushed_segment_count = 0
        self._grace_chunks_remaining = int(VAD_GRACE_PERIOD_SECS * SAMPLE_RATE / BLOCKSIZE)
        if self._silero_model is not None:
            self._silero_model.reset_states()
        
        if self._segment_cb is not None:
            self._encode_queue = queue.Queue()
            self._encode_thread = threading.Thread(target=self._encode_worker, daemon=True)
            self._encode_thread.start()
        if self._amplitude_cb is not None or self._vad_cb is not None:
            self._callbacks_enabled = True
            self._callback_queue = queue.Queue(maxsize=8)
            self._callback_thread = threading.Thread(
                target=self._callback_worker,
                daemon=True,
                name="audio-callback-dispatch",
            )
            self._callback_thread.start()

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
        self._vad_cb = None
        if self._encode_queue is not None:
            self._encode_queue.put(None)
            self._encode_queue = None
            self._encode_thread = None
        self._stop_callback_dispatch()
            
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
                    self._flushed_segment_count += 1
                    
            if self._encode_queue is not None:
                self._encode_queue.put(None)  # Sentinel to stop thread
            if self._encode_thread is not None:
                self._encode_thread.join(timeout=1.0)

            self._encode_queue = None
            self._encode_thread = None
            had_vad = self._vad_cb is not None or self._segment_cb is not None
            self._amplitude_cb = None
            self._vad_cb = None
            self._stop_callback_dispatch()
            self._current_segment_chunks = []
            self._ring_buffer.clear()
            self._speech_ring_buffer.clear()

            self._segment_cb = None

            # Use trimmed speech chunks if available; if VAD never detected
            # speech, return empty for VAD-aware callers.
            speech_chunks = getattr(self, "_speech_chunks", None)
            if speech_chunks is not None and len(speech_chunks) > 0:
                final_chunks = list(speech_chunks)
                wav_bytes = self._encode_wav(np.concatenate(final_chunks))
            elif had_vad:
                wav_bytes = b""
            else:
                wav_bytes = self._encode_wav(self._get_all_frames())

            if hasattr(self, '_speech_chunks'):
                self._speech_chunks = []
            return wav_bytes

        # No active stream — clear any stale frames and return empty
        self._frames = []
        self._amplitude_cb = None
        self._segment_cb = None
        self._vad_cb = None
        self._stop_callback_dispatch()
        self._current_segment_chunks = []
        self._ring_buffer.clear()
        self._speech_ring_buffer.clear()
        
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
        poll_start = time.perf_counter()
        with self._lock:
            available = len(self._frames) - self._read_cursor
            if available <= 0:
                poll_end = time.perf_counter()
                if self._metrics is not None:
                    self._metrics.record_capture_poll(
                        0,
                        elapsed_ms=(poll_end - poll_start) * 1000.0,
                        now=poll_end,
                    )
                return np.array([], dtype=np.float32)
            new = self._frames[self._read_cursor:]
            self._read_cursor = len(self._frames)
        result = np.concatenate(new)
        if self._metrics is not None:
            poll_end = time.perf_counter()
            self._metrics.record_capture_poll(
                len(new),
                elapsed_ms=(poll_end - poll_start) * 1000.0,
                now=poll_end,
            )
        return result

    def get_tail_buffer(self) -> bytes:
        """Return WAV bytes for audio since the last segment boundary.

        When segments are being accumulated via segment_callback, this returns
        only the unflushed tail — the current in-progress segment plus any
        silence/ring-buffer audio since the last flush.  Returns the full
        buffer if no segment callback is active (i.e. falls back to
        get_buffer behaviour).
        """
        if self._segment_cb is None:
            return self.get_buffer()
        chunks = list(self._current_segment_chunks)
        if not chunks:
            # Between segments (silence) — return ring buffer contents
            rb = list(self._ring_buffer)
            if not rb:
                return b""
            return self._encode_wav(np.concatenate(rb))
        return self._encode_wav(np.concatenate(chunks))

    @property
    def flushed_segment_count(self) -> int:
        """Number of segments flushed to segment_callback so far."""
        return getattr(self, "_flushed_segment_count", 0)

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
        if self._stream is None or self._stream_closing:
            return
        tick_start = time.perf_counter()
        if status:
            logger.warning("sounddevice status: %s", status)
        try:
            chunk = indata[:, 0].copy()  # mono, float32

            with self._lock:
                self._frames.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if self._amplitude_cb is not None:
                self._queue_callback_event("amplitude", rms)

            if self._stream_closing:
                return

            if self._segment_cb is not None or self._vad_cb is not None:
                # Grace period: suppress silence transitions but do NOT force speech.
                # Silero still decides — grace only prevents premature silence-idle
                # transitions during the first few seconds of recording.
                if self._grace_chunks_remaining > 0:
                    self._grace_chunks_remaining -= 1

                # Run Silero VAD on each 512-sample sub-chunk, take max probability
                if self._silero_model is not None:
                    torch = self._torch
                    max_prob = 0.0
                    for offset in range(0, len(chunk), SILERO_CHUNK):
                        if self._stream_closing:
                            return
                        sub = chunk[offset:offset + SILERO_CHUNK]
                        if len(sub) < SILERO_CHUNK:
                            break
                        tensor = torch.from_numpy(sub).unsqueeze(0)
                        prob = self._silero_model(tensor, self._silero_sr).item()
                        max_prob = max(max_prob, prob)
                    self._process_vad_decision(max_prob >= SPEECH_PROB_THRESHOLD, chunk, max_prob)
                else:
                    # Fallback: RMS-based detection when Silero is unavailable
                    self._process_vad_decision(rms > 0.01, chunk, rms)
                    if not self._silero_warned:
                        logger.warning("Silero VAD unavailable — using RMS fallback (degraded)")
                        self._silero_warned = True
        finally:
            if self._metrics is not None:
                tick_end = time.perf_counter()
                self._metrics.record_capture_tick(
                    elapsed_ms=(tick_end - tick_start) * 1000.0,
                    now=tick_end,
                )

    def _process_vad_decision(self, is_speech_now: bool, chunk: np.ndarray, prob: float) -> None:
        """Process a single Silero VAD decision and manage state transitions."""
        if logger.isEnabledFor(logging.DEBUG) and len(self._frames) % 50 == 0:
            logger.debug(
                "VAD tick: prob=%.4f, is_speech=%s, speech_trig=%d/%d, silence_trig=%d/%d, grace=%d",
                prob, self._is_speech,
                self._speech_trigger_count, MIN_SPEECH_FRAMES,
                self._silence_trigger_count, MIN_SILENCE_FRAMES,
                self._grace_chunks_remaining,
            )

        if not self._is_speech:
            self._ring_buffer.append(chunk)
            self._speech_ring_buffer.append((chunk, is_speech_now))
            if is_speech_now:
                self._speech_trigger_count += 1
                if self._speech_trigger_count >= MIN_SPEECH_FRAMES:
                    self._is_speech = True
                    if self._vad_cb is not None:
                        self._queue_callback_event("vad", True)
                    self._current_segment_chunks.extend(self._ring_buffer)
                    self._speech_chunks.extend(
                        ring_chunk
                        for ring_chunk, was_speech in self._speech_ring_buffer
                        if was_speech or _has_non_silent_samples(ring_chunk)
                    )
                    self._ring_buffer.clear()
                    self._speech_ring_buffer.clear()
            else:
                self._speech_trigger_count = 0
        else:
            self._current_segment_chunks.append(chunk)
            self._speech_chunks.append(chunk)

            force_slice = len(self._current_segment_chunks) >= MAX_SEGMENT_CHUNKS

            if self._grace_chunks_remaining <= 0 and (force_slice or not is_speech_now):
                if not is_speech_now:
                    self._silence_trigger_count += 1
                else:
                    self._silence_trigger_count = MIN_SILENCE_FRAMES

                if self._silence_trigger_count >= MIN_SILENCE_FRAMES:
                    self._is_speech = False
                    if self._vad_cb is not None:
                        self._queue_callback_event("vad", False)
                    self._speech_trigger_count = 0
                    self._silence_trigger_count = 0

                    if not force_slice and self._speech_chunks:
                        del self._speech_chunks[-MIN_SILENCE_FRAMES:]

                    if self._encode_queue is not None:
                        segment_samples = np.concatenate(self._current_segment_chunks)
                        self._encode_queue.put(segment_samples)
                        self._flushed_segment_count += 1

                    self._current_segment_chunks = []
            elif is_speech_now:
                self._silence_trigger_count = max(0, self._silence_trigger_count - 2)

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
