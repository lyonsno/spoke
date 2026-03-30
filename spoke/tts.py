"""Local text-to-speech via Voxtral on MLX.

Loads a Voxtral TTS model lazily and plays generated audio through
sounddevice.  Designed to be driven from the command-completion pathway
in __main__.py — speak_async() returns immediately, running generation
and playback on a background thread.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
_DEFAULT_VOICE = "casual_female"
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TOP_K = 50
_DEFAULT_TOP_P = 0.95
_AUDIO_TOGGLE_FADE_S = 0.5
_ABBREVIATION_SUFFIXES = (
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
)


@dataclass(slots=True)
class _PlaybackResult:
    """Materialized audio payload safe to play after generation yields it."""

    audio: np.ndarray
    sample_rate: int


def _playback_device_summary() -> str:
    """Return a compact description of the active playback device context."""
    try:
        default_device = getattr(sd.default, "device", None)
        if isinstance(default_device, (list, tuple)):
            output_index = default_device[1] if len(default_device) > 1 else default_device[0]
        else:
            output_index = default_device
        device_info = sd.query_devices(output_index)
        if isinstance(device_info, dict):
            device_name = device_info.get("name", "<unknown>")
        else:
            device_name = getattr(device_info, "name", "<unknown>")
        return f"default={default_device!r} output={output_index!r} name={device_name}"
    except Exception as exc:
        return f"default=<unavailable: {exc}>"


def _split_sentences(text: str) -> list[str]:
    """Split text conservatively on sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char not in ".!?":
            i += 1
            continue

        end = i + 1
        while end < len(text) and text[end] in "\"'”’)]}":
            end += 1
        candidate = text[start:end].strip()
        lowered = candidate.lower()

        if char == ".":
            if any(lowered.endswith(suffix) for suffix in _ABBREVIATION_SUFFIXES):
                i += 1
                continue
            if i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue

        next_nonspace = end
        while next_nonspace < len(text) and text[next_nonspace].isspace():
            next_nonspace += 1

        if next_nonspace < len(text) and text[next_nonspace].islower() and char == ".":
            i += 1
            continue

        if candidate:
            sentences.append(candidate)
        start = next_nonspace
        i = next_nonspace

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def tts_load(model_id: str):
    """Load a Voxtral TTS model.  Separated for easy patching in tests."""
    if "voxtral" in model_id.lower():
        try:
            mlx_audio_pkg = importlib.import_module("mlx_audio")
            importlib.import_module("mlx_audio.tts.models.voxtral_tts")
        except Exception as exc:
            mlx_audio_path = getattr(mlx_audio_pkg, "__file__", "<unresolved>") if "mlx_audio_pkg" in locals() else "<unresolved>"
            py_path = os.environ.get("PYTHONPATH", "")
            raise RuntimeError(
                "Voxtral TTS backend is unavailable in the active mlx_audio runtime. "
                f"Resolved mlx_audio from {mlx_audio_path}. "
                f"PYTHONPATH={py_path or '<unset>'}. "
                "Expected mlx_audio.tts.models.voxtral_tts to be importable. "
                "If you are using a local smoke/runtime checkout, ensure the branch-local "
                ".spoke-smoke-env restores the intended PYTHONPATH override."
            ) from exc
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
        self._gpu_lock = gpu_lock
        self._stream: sd.OutputStream | None = None
        self._last_chunk: np.ndarray | None = None
        self._speak_lock = threading.Lock()
        self._audio_fade_lock = threading.Lock()
        self._playback_active = False
        self._audio_fade_start_gain = 1.0
        self._audio_fade_target_gain = 1.0
        self._audio_fade_started_at = time.monotonic()

    @classmethod
    def from_env(cls, gpu_lock: threading.Lock | None = None) -> Optional["TTSClient"]:
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
        return cls(model_id=model_id, voice=voice, temperature=temperature, top_k=top_k, top_p=top_p, gpu_lock=gpu_lock)

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

    def _current_audio_gain_locked(self, now: float) -> float:
        progress = min(max((now - self._audio_fade_started_at) / _AUDIO_TOGGLE_FADE_S, 0.0), 1.0)
        eased = progress * progress * (3.0 - 2.0 * progress)
        return self._audio_fade_start_gain + (
            self._audio_fade_target_gain - self._audio_fade_start_gain
        ) * eased

    def _current_audio_gain(self) -> float:
        with self._audio_fade_lock:
            return self._current_audio_gain_locked(time.monotonic())

    def toggle_audio(self) -> bool:
        """Toggle audible playback with a 500ms eased fade."""
        with self._audio_fade_lock:
            now = time.monotonic()
            if self._playback_active or self._stream is not None:
                current_gain = self._current_audio_gain_locked(now)
            else:
                current_gain = self._audio_fade_target_gain
            target_gain = 0.0 if self._audio_fade_target_gain > 0.0 else 1.0
            self._audio_fade_start_gain = current_gain
            self._audio_fade_target_gain = target_gain
            self._audio_fade_started_at = now
            return target_gain > 0.0

    def _play_result(
        self,
        result,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        audio = np.array(result.audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        sr = result.sample_rate
        chunk_size = int(sr * 0.064)
        playback_device = _playback_device_summary()
        logger.info(
            "TTS playback starting: samples=%d sample_rate=%d channels=%d chunk_size=%d device=%s",
            len(audio),
            sr,
            audio.shape[1],
            chunk_size,
            playback_device,
        )

        done = threading.Event()
        stream = sd.OutputStream(
            samplerate=sr,
            channels=audio.shape[1],
            dtype="float32",
            finished_callback=lambda: done.set(),
        )
        self._stream = stream
        self._last_chunk = None
        stream.start()

        try:
            offset = 0
            while offset < len(audio):
                if self._cancelled:
                    break
                end = min(offset + chunk_size, len(audio))
                chunk = audio[offset:end]
                gain = self._current_audio_gain()
                shaped_chunk = chunk * gain
                self._last_chunk = shaped_chunk
                stream.write(shaped_chunk)
                if amplitude_callback is not None:
                    rms = float(np.sqrt(np.mean(shaped_chunk ** 2)))
                    amplitude_callback(rms)
                offset = end

            while not done.is_set():
                if self._cancelled:
                    break
                done.wait(timeout=0.05)

            if self._cancelled and not done.is_set() and self._last_chunk is not None:
                fade_samples = int(sr * 0.05)
                last_amp = float(np.mean(np.abs(self._last_chunk[-1:])))
                fade_ramp = np.linspace(last_amp, 0.0, fade_samples, dtype=np.float32).reshape(-1, 1)
                try:
                    stream.write(fade_ramp)
                except Exception:
                    pass
            logger.info(
                "TTS playback finished: samples=%d sample_rate=%d channels=%d cancelled=%s device=%s",
                len(audio),
                sr,
                audio.shape[1],
                self._cancelled,
                playback_device,
            )
        except Exception:
            logger.warning(
                "TTS playback failed: samples=%d sample_rate=%d channels=%d cancelled=%s device=%s",
                len(audio),
                sr,
                audio.shape[1],
                self._cancelled,
                playback_device,
                exc_info=True,
            )
            raise
        finally:
            stream.stop()
            stream.close()
            self._stream = None
            self._last_chunk = None

        if amplitude_callback is not None:
            amplitude_callback(0.0)

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

        from contextlib import nullcontext

        sentences = _split_sentences(text)
        if not sentences:
            return

        lock_ctx = self._gpu_lock if self._gpu_lock is not None else nullcontext()
        with self._speak_lock:
            if self._cancelled:
                return
            with self._audio_fade_lock:
                self._playback_active = True
            try:
                for sentence in sentences:
                    if self._cancelled:
                        return

                    with lock_ctx:
                        self._ensure_model()
                        if self._cancelled:
                            return
                        results = self._model.generate(
                            text=sentence,
                            voice=self._voice,
                            temperature=self._temperature,
                            top_k=self._top_k,
                            top_p=self._top_p,
                        )

                    while True:
                        if self._cancelled:
                            return
                        with lock_ctx:
                            try:
                                result = next(results)
                            except StopIteration:
                                break
                            materialized = _PlaybackResult(
                                audio=np.asarray(result.audio, dtype=np.float32),
                                sample_rate=int(result.sample_rate),
                            )
                        self._play_result(materialized, amplitude_callback=amplitude_callback)
            finally:
                with self._audio_fade_lock:
                    self._playback_active = False

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
        """Cancel any in-flight or future speak() call.

        Sets the cancelled flag so the playback loop writes a short fade-out
        and exits cleanly. Does not call stream.abort() — avoids racing with
        the fade-out write on the playback thread.
        """
        self._cancelled = True
