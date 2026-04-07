"""Text-to-speech via local MLX or a remote speech sidecar.

Loads a local TTS model lazily (any mlx_audio-supported architecture) or
fetches synthesized audio from an OpenAI-compatible remote sidecar, then
plays audio through sounddevice. Designed to be driven from the
command-completion pathway in __main__.py — speak_async() returns
immediately, running synthesis and playback on a background thread.
"""

from __future__ import annotations

import importlib
import io
import json as _json
import inspect
import logging
import os
import platform
import queue
import sys
import threading
import time
import wave
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
_SENTENCE_FADE_S = 0.015  # 15ms fade-in/fade-out at sentence boundaries
_OMNIVOICE_SAMPLE_RATE = 24000
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


def _resolve_output_device() -> int | None:
    """Return the current system default output device index.

    Queries PortAudio each time so the stream follows device changes
    (e.g. Bluetooth connect/disconnect).  Returns None on failure,
    which lets sounddevice fall back to its own default.
    """
    try:
        default_device = getattr(sd.default, "device", None)
        if isinstance(default_device, (list, tuple)) and len(default_device) > 1:
            return int(default_device[1])
        if isinstance(default_device, int):
            return default_device
        # Ask PortAudio directly
        info = sd.query_devices(kind="output")
        if isinstance(info, dict):
            return info.get("index")
    except Exception:
        pass
    return None


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
    """Load an mlx_audio TTS model.  Separated for easy patching in tests."""
    if _is_omnivoice_model(model_id):
        return omnivoice_load(model_id)

    # Pre-import model backends so mlx_audio.tts.load() can find them.
    # The Voxtral backend lives in a PYTHONPATH fork that isn't auto-discovered.
    if "voxtral" in model_id.lower():
        try:
            importlib.import_module("mlx_audio.tts.models.voxtral_tts")
        except ImportError:
            mlx_audio_path = "<unresolved>"
            try:
                import mlx_audio
                mlx_audio_path = getattr(mlx_audio, "__file__", "<unresolved>")
            except ImportError:
                pass
            py_path = os.environ.get("PYTHONPATH", "")
            raise RuntimeError(
                "Voxtral TTS backend is unavailable in the active mlx_audio runtime. "
                f"Resolved mlx_audio from {mlx_audio_path}. "
                f"PYTHONPATH={py_path or '<unset>'}. "
                "Ensure .spoke-smoke-env sets PYTHONPATH to the mlx-audio fork with Voxtral support."
            )
    from mlx_audio.tts import load
    return load(model_id)


def _is_omnivoice_model(model_id: str) -> bool:
    return model_id.strip().lower() == "k2-fsa/omnivoice"


def _default_voice_for_model(model_id: str) -> str | None:
    """Return the implicit local voice to use for a model, if any."""
    return None if _is_omnivoice_model(model_id) else _DEFAULT_VOICE


def _omnivoice_dtype(torch_module, device_map: str):
    dtype_name = os.environ.get("SPOKE_OMNIVOICE_DTYPE", "").strip()
    if dtype_name:
        dtype = getattr(torch_module, dtype_name, None)
        if dtype is None:
            raise RuntimeError(
                f"SPOKE_OMNIVOICE_DTYPE={dtype_name!r} does not name a torch dtype"
            )
        return dtype
    return torch_module.float16 if device_map != "cpu" else torch_module.float32


def _omnivoice_device_map(torch_module) -> str:
    explicit = os.environ.get("SPOKE_OMNIVOICE_DEVICE_MAP", "").strip()
    if explicit:
        return explicit
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda:0"
    return "cpu"


def omnivoice_load(model_id: str):
    """Load OmniVoice through its native package instead of mlx_audio."""
    try:
        import torch
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError(
            "OmniVoice TTS backend is unavailable in the active runtime. "
            "Install the `tts` extra with OmniVoice support."
        ) from exc

    device_map = _omnivoice_device_map(torch)
    dtype = _omnivoice_dtype(torch, device_map)
    logger.info(
        "Loading OmniVoice model %s with device_map=%s dtype=%s",
        model_id,
        device_map,
        getattr(dtype, "__name__", repr(dtype)),
    )
    model = OmniVoice.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
    )
    setattr(model, "sample_rate", _OMNIVOICE_SAMPLE_RATE)
    return model


def _generate_kwargs(model, *, text: str, voice: str | None,
                     temperature: float, top_k: int, top_p: float,
                     model_id: str | None = None) -> dict:
    """Build kwargs for model.generate(), passing only params it accepts.

    If the model's generate() signature can't be introspected (e.g. it
    accepts **kwargs with no named params beyond self), all params are
    forwarded — the model can ignore what it doesn't need.
    """
    try:
        sig = inspect.signature(model.generate)
        params = sig.parameters
    except (ValueError, TypeError):
        params = {}
    # If there are no named params (just *args/**kwargs), pass everything
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    named = {n for n, p in params.items()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)}
    is_omnivoice = model_id is not None and _is_omnivoice_model(model_id)
    voice_key = "instruct" if is_omnivoice else "voice"
    all_extras: dict[str, object] = {}
    if voice:
        all_extras[voice_key] = voice
    if not is_omnivoice:
        all_extras.update(
            {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
    kwargs: dict = {"text": text}
    if not named or has_var_keyword:
        # Can't tell what's accepted — forward everything
        kwargs.update(all_extras)
    else:
        for k, v in all_extras.items():
            if k in named:
                kwargs[k] = v
            elif k == "voice" and "instruct" in named and v:
                # OmniVoice voice-design prompts use `instruct` instead of `voice`.
                kwargs["instruct"] = v
    return kwargs


def _normalize_audio_array(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1 and array.shape[1] > 1:
        array = array.T
    return array


def _materialize_generation_result(result, sample_rate_hint: int | None) -> _PlaybackResult:
    if hasattr(result, "audio"):
        audio = _normalize_audio_array(result.audio)
        sample_rate = int(getattr(result, "sample_rate"))
    else:
        if sample_rate_hint is None:
            raise RuntimeError("TTS backend returned raw audio without a sample-rate hint")
        audio = _normalize_audio_array(result)
        sample_rate = int(sample_rate_hint)
    return _PlaybackResult(audio=audio, sample_rate=sample_rate)


def _iter_playback_results(results, sample_rate_hint: int | None):
    if results is None:
        return
    if isinstance(results, (list, tuple)):
        iterable = results
    else:
        iterable = results
    for result in iterable:
        yield _materialize_generation_result(result, sample_rate_hint)


def _apply_sentence_fades(audio: np.ndarray, sample_rate: int,
                          fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
    """Apply short fade-in/fade-out ramps to smooth sentence boundaries."""
    fade_samples = int(sample_rate * _SENTENCE_FADE_S)
    if fade_samples < 2 or len(audio) < fade_samples * 2:
        return audio
    audio = audio.copy()
    if fade_in:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        if audio.ndim == 2:
            ramp = ramp.reshape(-1, 1)
        audio[:fade_samples] *= ramp
    if fade_out:
        ramp = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        if audio.ndim == 2:
            ramp = ramp.reshape(-1, 1)
        audio[-fade_samples:] *= ramp
    return audio


_SENTENCE_BOUNDARY = object()  # sentinel pushed into queue between sentences


class TTSClient:
    """Lazy-loading TTS client with cancellation support."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        voice: str | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_k: int = _DEFAULT_TOP_K,
        top_p: float = _DEFAULT_TOP_P,
        gpu_lock: threading.Lock | None = None,
    ):
        self._model_id = model_id
        self._voice = _default_voice_for_model(model_id) if voice is None else voice
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
        self._warm_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        self._warm_in_progress = False
        self._warm_completed = threading.Event()
        self._warm_error: Exception | None = None

    @classmethod
    def from_env(cls, gpu_lock: threading.Lock | None = None) -> Optional["TTSClient | RemoteTTSClient"]:
        """Create a TTS client from environment variables, or None if disabled.

        Set SPOKE_TTS_VOICE to enable TTS (e.g. "casual_female").
        Optionally set SPOKE_TTS_MODEL to override the default model.
        Set SPOKE_TTS_URL to route synthesis to a remote OpenAI-compatible
        /v1/audio/speech sidecar while keeping playback local.
        """
        voice = os.environ.get("SPOKE_TTS_VOICE")
        if not voice:
            return None
        base_url = os.environ.get("SPOKE_TTS_URL", "").rstrip("/")
        model_id = os.environ.get("SPOKE_TTS_MODEL", _DEFAULT_MODEL_ID)
        if base_url:
            return RemoteTTSClient(base_url=base_url, model_id=model_id, voice=voice)
        temperature = float(os.environ.get("SPOKE_TTS_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
        top_k = int(os.environ.get("SPOKE_TTS_TOP_K", str(_DEFAULT_TOP_K)))
        top_p = float(os.environ.get("SPOKE_TTS_TOP_P", str(_DEFAULT_TOP_P)))
        return cls(model_id=model_id, voice=voice, temperature=temperature, top_k=top_k, top_p=top_p, gpu_lock=gpu_lock)

    def _ensure_model(self):
        """Load the model on first use."""
        if self._model is None:
            logger.info("Loading TTS model %s …", self._model_id)
            self._model = tts_load(self._model_id)
            self._warm_error = None
            self._warm_completed.set()
            logger.info("TTS model loaded.")

    def warm(self) -> None:
        """Pre-load the model in a background thread so first speak() is fast."""
        def _warm():
            from contextlib import nullcontext
            lock_ctx = self._gpu_lock if self._gpu_lock is not None else nullcontext()
            with lock_ctx:
                try:
                    self._ensure_model()
                except Exception as exc:
                    self._warm_error = exc
                    logger.warning("TTS model warm-up failed", exc_info=True)
                finally:
                    with self._warm_lock:
                        self._warm_in_progress = False
                        self._warm_completed.set()

        with self._warm_lock:
            if self._model is not None:
                self._warm_error = None
                self._warm_completed.set()
                return
            if self._warm_in_progress and self._warm_thread is not None and self._warm_thread.is_alive():
                return
            self._warm_error = None
            self._warm_completed.clear()
            self._warm_in_progress = True
            self._warm_thread = threading.Thread(target=_warm, daemon=True)
            self._warm_thread.start()

    def unload(self) -> None:
        """Release the TTS model to free memory."""
        if self._model is None:
            return
        logger.info("Unloading TTS model %s", self._model_id)
        self._model = None
        self._warm_completed.clear()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident in memory."""
        return self._model is not None

    @property
    def is_warming(self) -> bool:
        """Whether a background warmup is currently in flight."""
        with self._warm_lock:
            if self._warm_in_progress and self._warm_thread is not None and not self._warm_thread.is_alive():
                self._warm_in_progress = False
            return self._warm_in_progress

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for an in-flight warmup to finish."""
        if self._model is not None:
            return True
        if not self.is_warming:
            return self._model is not None
        finished = self._warm_completed.wait(timeout=timeout)
        if not finished:
            return False
        return self._model is not None

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

        stream = sd.OutputStream(
            samplerate=sr,
            channels=audio.shape[1],
            dtype="float32",
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

            if self._cancelled and self._last_chunk is not None:
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

    def _start_generation(self, sentence: str):
        """Start generation for a sentence, returning the chunk iterator.

        Holds gpu_lock during model.generate() call but NOT during chunk
        iteration (the iterator does Metal work internally but holding the
        lock deadlocks with Whisper transcription on another thread).

        Returns None if the model fails to load or generation is cancelled.
        """
        from contextlib import nullcontext
        lock_ctx = self._gpu_lock if self._gpu_lock is not None else nullcontext()

        with lock_ctx:
            try:
                self._ensure_model()
            except Exception:
                logger.warning("TTS model load failed — cannot speak", exc_info=True)
                return None
            if self._cancelled:
                logger.info("TTS speak: cancelled after model load")
                return None
            gen_kwargs = _generate_kwargs(
                self._model,
                text=sentence,
                voice=self._voice,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                model_id=self._model_id,
            )
            results = self._model.generate(**gen_kwargs)
            sample_rate_hint = getattr(self._model, "sample_rate", None)
            logger.info("TTS speak: generate() returned for: %s", sentence[:40])
        return _iter_playback_results(results, sample_rate_hint)

    def speak(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Generate speech and play it synchronously.  Blocks until done.

        Uses a producer/consumer pipeline: a generation thread produces
        _PlaybackResults into a queue while a playback thread (this thread)
        drains and plays them.  Generation runs ahead of playback so the
        next sentence is ready when the current one finishes.

        Short fade-in/fade-out ramps are applied at sentence boundaries to
        smooth the transition between sentences.

        Holds gpu_lock during model.generate() to prevent concurrent MLX
        inference (which crashes Metal).  Releases the lock before audio
        playback so Whisper can proceed while audio plays.

        If amplitude_callback is provided, it is called with the RMS value
        of each ~64ms audio chunk during playback (same interface as the
        microphone amplitude callback used by the glow overlay).
        """
        if not text:
            logger.info("TTS speak: empty text, skipping")
            return

        sentences = _split_sentences(text)
        if not sentences:
            logger.info("TTS speak: no sentences after split, skipping")
            return

        logger.info("TTS speak: %d sentences, %d chars, model=%s, cancelled=%s",
                     len(sentences), len(text), self._model_id, self._cancelled)
        with self._speak_lock:
            self._cancelled = False
            with self._audio_fade_lock:
                self._playback_active = True

            # Queue connects generation → playback.  None sentinel = done.
            playback_queue: queue.Queue[_PlaybackResult | None] = queue.Queue(maxsize=4)

            def _generate_all() -> None:
                """Generate all sentences, pushing results into the queue.

                Streams chunks as they arrive — each chunk is enqueued
                immediately so playback can start before the full sentence
                is generated.  Sentence-boundary fades are applied to the
                first chunk of each sentence (fade-in) and a deferred
                fade-out is applied to the last chunk once we know it's
                the last.
                """
                try:
                    for idx, sentence in enumerate(sentences):
                        if self._cancelled:
                            return
                        results_iter = self._start_generation(sentence)
                        if results_iter is None:
                            continue
                        chunk_count = 0
                        for materialized in results_iter:
                            if self._cancelled:
                                return
                            # Apply fade-in to first chunk of each sentence
                            if chunk_count == 0:
                                materialized.audio = _apply_sentence_fades(
                                    materialized.audio, materialized.sample_rate,
                                    fade_in=True, fade_out=False,
                                )
                            playback_queue.put(materialized)
                            chunk_count += 1
                        # Signal sentence boundary (playback reopens stream
                        # here to follow device changes)
                        if chunk_count > 0:
                            playback_queue.put(_SENTENCE_BOUNDARY)
                        if chunk_count > 0:
                            logger.info("TTS speak: generated sentence %d/%d (%d chunks)",
                                       idx + 1, len(sentences), chunk_count)
                finally:
                    playback_queue.put(None)  # sentinel

            gen_thread = threading.Thread(target=_generate_all, daemon=True)
            gen_thread.start()

            stream: sd.OutputStream | None = None
            try:
                chunk_count = 0
                write_chunk_size: int = 0
                while True:
                    if self._cancelled:
                        break
                    try:
                        item = playback_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    if item is _SENTENCE_BOUNDARY:
                        # Reopen stream at sentence boundary to follow
                        # the current default output device (e.g. Bluetooth
                        # connected mid-playback).  Wait for the buffer to
                        # drain before closing to avoid truncation.
                        if stream is not None:
                            try:
                                stream.wait()
                            except Exception:
                                pass
                            stream.stop()
                            stream.close()
                            stream = None
                            self._stream = None
                            self._last_chunk = None
                        continue

                    audio = np.array(item.audio, dtype=np.float32)
                    if audio.ndim == 1:
                        audio = audio.reshape(-1, 1)
                    sr = item.sample_rate

                    # Open stream lazily on first chunk (sample rate may vary).
                    # Explicitly resolve the current default output device so
                    # the stream follows Bluetooth/headphone changes.
                    if stream is None:
                        write_chunk_size = int(sr * 0.064)
                        output_device = _resolve_output_device()
                        stream = sd.OutputStream(
                            samplerate=sr,
                            channels=audio.shape[1],
                            dtype="float32",
                            device=output_device,
                        )
                        self._stream = stream
                        self._last_chunk = None
                        stream.start()
                        logger.info(
                            "TTS playback stream opened: sample_rate=%d channels=%d device=%r (%s)",
                            sr, audio.shape[1], output_device, _playback_device_summary(),
                        )

                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info("TTS speak: first audio chunk: %d samples @ %dHz",
                                   len(audio), sr)

                    # Write audio in ~64ms sub-chunks with gain modulation
                    offset = 0
                    while offset < len(audio):
                        if self._cancelled:
                            break
                        end = min(offset + write_chunk_size, len(audio))
                        chunk = audio[offset:end]
                        gain = self._current_audio_gain()
                        shaped_chunk = chunk * gain
                        self._last_chunk = shaped_chunk
                        stream.write(shaped_chunk)
                        if amplitude_callback is not None:
                            rms = float(np.sqrt(np.mean(shaped_chunk ** 2)))
                            amplitude_callback(rms)
                        offset = end

                logger.info("TTS speak: finished (%d chunks played)", chunk_count)
            finally:
                # Fade out on cancel
                if self._cancelled and stream is not None and self._last_chunk is not None:
                    fade_samples = int(stream.samplerate * 0.05)
                    last_amp = float(np.mean(np.abs(self._last_chunk[-1:])))
                    fade_ramp = np.linspace(last_amp, 0.0, fade_samples, dtype=np.float32).reshape(-1, 1)
                    try:
                        stream.write(fade_ramp)
                    except Exception:
                        pass
                # Close the single stream
                if stream is not None:
                    stream.stop()
                    stream.close()
                    self._stream = None
                    self._last_chunk = None
                if amplitude_callback is not None:
                    amplitude_callback(0.0)
                # Drain queue so gen thread doesn't block on put()
                self._cancelled = True
                while True:
                    try:
                        playback_queue.get_nowait()
                    except queue.Empty:
                        break
                gen_thread.join(timeout=5)
                with self._audio_fade_lock:
                    self._playback_active = False

    def speak_async(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
        done_callback: Callable[[], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        """Generate and play speech on a background daemon thread."""
        self.cancel()
        def _run():
            try:
                self.speak(text, amplitude_callback=amplitude_callback)
            except Exception as exc:
                logger.exception("TTS local speak failed")
                if error_callback is not None:
                    error_callback(str(exc))
            finally:
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
        import traceback
        logger.info("TTS cancel() called from:\n%s", "".join(traceback.format_stack()[-8:-1]))
        self._cancelled = True


class RemoteTTSClient:
    """HTTP-backed TTS client for a remote OpenAI-compatible speech sidecar.

    Fetches synthesized audio from a remote /v1/audio/speech endpoint and
    plays it locally via sounddevice. Shares the cancel/speak_async interface
    with TTSClient so callers can swap backends transparently.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str = _DEFAULT_MODEL_ID,
        voice: str = _DEFAULT_VOICE,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/v1/audio/speech"
        self._model_id = model_id
        self._voice = voice
        self._timeout = timeout
        self._cancelled = False
        self._stream: sd.OutputStream | None = None
        self._last_chunk: np.ndarray | None = None

    def warm(self) -> None:
        """Remote speech is ready once the HTTP client exists."""
        return None

    def _decode_wav(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        return audio.reshape(-1, channels), sample_rate

    def _play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Play float32 audio locally, emitting optional RMS updates."""
        if self._cancelled:
            return
        if audio.size == 0:
            if amplitude_callback is not None:
                amplitude_callback(0.0)
            return
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        chunk_size = int(sample_rate * 0.064)
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=audio.shape[1],
            dtype="float32",
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
                self._last_chunk = chunk
                stream.write(chunk)
                if amplitude_callback is not None:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    amplitude_callback(rms)
                offset = end

            if self._cancelled and self._last_chunk is not None:
                fade_samples = int(sample_rate * 0.05)
                last_amp = float(np.mean(np.abs(self._last_chunk[-1:])))
                fade_ramp = np.linspace(last_amp, 0.0, fade_samples, dtype=np.float32).reshape(-1, 1)
                try:
                    stream.write(fade_ramp)
                except Exception:
                    pass
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
        if not text:
            return
        self._cancelled = False

        import urllib.request
        import urllib.error
        payload = {
            "model": self._model_id,
            "voice": self._voice,
            "input": text,
            "response_format": "wav",
        }
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(
            "TTS sidecar request: url=%s model=%s voice=%s text=%d chars",
            self._url, self._model_id, self._voice, len(text),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                wav_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"TTS sidecar HTTP {exc.code} from {self._url}: {body or exc.reason} "
                f"(model={self._model_id}, voice={self._voice})"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"TTS sidecar unreachable at {self._base_url}: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"TTS sidecar request failed ({self._url}, model={self._model_id}, "
                f"voice={self._voice}): {type(exc).__name__}: {exc}"
            ) from exc

        if self._cancelled:
            return

        logger.info(
            "TTS sidecar response: %d bytes, content-type=%s",
            len(wav_bytes), content_type,
        )

        if not wav_bytes:
            raise RuntimeError(
                f"TTS sidecar returned empty response (model={self._model_id}, "
                f"voice={self._voice}, url={self._url})"
            )

        try:
            audio, sample_rate = self._decode_wav(wav_bytes)
        except Exception as exc:
            # Show first bytes to help diagnose format issues
            preview = wav_bytes[:64]
            raise RuntimeError(
                f"TTS sidecar returned invalid audio (model={self._model_id}, "
                f"voice={self._voice}, content-type={content_type}, "
                f"size={len(wav_bytes)}, starts_with={preview!r}): {exc}"
            ) from exc

        self._play_audio(audio, sample_rate, amplitude_callback=amplitude_callback)

    def speak_async(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
        done_callback: Callable[[], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        """Generate and play speech on a background daemon thread."""
        self._cancelled = False

        def _run():
            try:
                self.speak(text, amplitude_callback=amplitude_callback)
            except Exception as exc:
                logger.exception("TTS sidecar speak failed")
                if error_callback is not None:
                    error_callback(str(exc))
            finally:
                if done_callback is not None:
                    done_callback()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel(self) -> None:
        """Cancel any in-flight or future speak() call."""
        self._cancelled = True


# ---------------------------------------------------------------------------
# Gemini cloud voices
# ---------------------------------------------------------------------------

GEMINI_VOICES: list[tuple[str, str]] = [
    ("Aoede", "Aoede"),
    ("Charon", "Charon"),
    ("Fenrir", "Fenrir"),
    ("Kore", "Kore"),
    ("Leda", "Leda"),
    ("Orus", "Orus"),
    ("Puck", "Puck"),
    ("Zephyr", "Zephyr"),
]

_DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_GEMINI_TTS_SAMPLE_RATE = 24000


class CloudTTSClient:
    """TTS client backed by the Gemini generateContent API with audio output.

    Uses the ``responseModalities: ["AUDIO"]`` capability of Gemini 2.0 Flash
    to synthesize speech.  The API returns base64-encoded WAV inline in JSON,
    which this client decodes and plays locally via sounddevice.

    Auth is via ``x-goog-api-key`` header (same ``GEMINI_API_KEY`` used by the
    cloud assistant backend).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_GEMINI_TTS_MODEL,
        voice: str = "Aoede",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._model_id = model  # compat with menu introspection
        self._voice = voice
        self._timeout = timeout
        self._cancelled = False
        self._stream: sd.OutputStream | None = None
        self._last_chunk: np.ndarray | None = None

    # -- public interface (same shape as TTSClient / RemoteTTSClient) -------

    def warm(self) -> None:
        """No local model to warm — cloud is always ready."""
        return None

    @property
    def is_loaded(self) -> bool:
        return True

    def speak(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        if not text:
            return
        self._cancelled = False

        sentences = _split_sentences(text)
        if not sentences:
            return

        for idx, sentence in enumerate(sentences):
            if self._cancelled:
                break
            wav_bytes = self._synthesize(sentence)
            if self._cancelled or wav_bytes is None:
                break
            audio, sample_rate = self._decode_wav(wav_bytes)
            # Apply sentence boundary fades
            audio = _apply_sentence_fades(
                audio, sample_rate,
                fade_in=(idx == 0),
                fade_out=(idx == len(sentences) - 1),
            )
            self._play_audio(audio, sample_rate, amplitude_callback=amplitude_callback)

    def speak_async(
        self,
        text: str,
        amplitude_callback: Callable[[float], None] | None = None,
        done_callback: Callable[[], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ) -> threading.Thread:
        self._cancelled = False

        def _run():
            try:
                self.speak(text, amplitude_callback=amplitude_callback)
            except Exception as exc:
                logger.exception("TTS cloud speak failed")
                if error_callback is not None:
                    error_callback(str(exc))
            finally:
                if done_callback is not None:
                    done_callback()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def cancel(self) -> None:
        self._cancelled = True

    # -- internals ----------------------------------------------------------

    def _synthesize(self, text: str) -> bytes | None:
        """Call Gemini generateContent with audio output and return raw WAV bytes."""
        import base64
        import urllib.request
        import urllib.error

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self._voice,
                        }
                    }
                },
            },
        }
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        logger.info(
            "TTS cloud request: model=%s voice=%s text=%d chars",
            self._model, self._voice, len(text),
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Gemini TTS HTTP {exc.code}: {detail or exc.reason} "
                f"(model={self._model}, voice={self._voice})"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Gemini TTS unreachable: {exc.reason}"
            ) from exc

        response = _json.loads(body)

        # Navigate: candidates[0].content.parts[0].inlineData
        try:
            candidates = response["candidates"]
            parts = candidates[0]["content"]["parts"]
            inline_data = parts[0]["inlineData"]
            b64_audio = inline_data["data"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Gemini TTS response missing audio data: {response!r:.500}"
            ) from exc

        wav_bytes = base64.b64decode(b64_audio)
        logger.info("TTS cloud response: %d bytes audio", len(wav_bytes))
        return wav_bytes

    def _decode_wav(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes to float32 numpy array + sample rate."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
        except Exception:
            # Gemini may return raw PCM (L16) without WAV headers
            # Assume 24kHz mono 16-bit signed
            logger.info("TTS cloud: WAV decode failed, trying raw PCM")
            audio = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio.reshape(-1, 1), _GEMINI_TTS_SAMPLE_RATE

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        return audio.reshape(-1, channels), sample_rate

    def _play_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        amplitude_callback: Callable[[float], None] | None = None,
    ) -> None:
        if self._cancelled:
            return
        if audio.size == 0:
            if amplitude_callback is not None:
                amplitude_callback(0.0)
            return
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        chunk_size = int(sample_rate * 0.064)
        output_device = _resolve_output_device()
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=audio.shape[1],
            dtype="float32",
            device=output_device,
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
                self._last_chunk = chunk
                stream.write(chunk)
                if amplitude_callback is not None:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    amplitude_callback(rms)
                offset = end

            if self._cancelled and self._last_chunk is not None:
                fade_samples = int(sample_rate * 0.05)
                last_amp = float(np.mean(np.abs(self._last_chunk[-1:])))
                fade_ramp = np.linspace(last_amp, 0.0, fade_samples, dtype=np.float32).reshape(-1, 1)
                try:
                    stream.write(fade_ramp)
                except Exception:
                    pass
        finally:
            stream.stop()
            stream.close()
            self._stream = None
            self._last_chunk = None

        if amplitude_callback is not None:
            amplitude_callback(0.0)
