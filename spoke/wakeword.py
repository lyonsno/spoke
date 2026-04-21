"""Always-on wake word detection via pluggable wakeword backends.

Runs a lightweight audio stream independent of the main AudioCapture,
listening for custom wake words to trigger hands-free mode transitions.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordListener:
    """Wake word detector with its own audio stream.

    Parameters
    ----------
    access_key : str
        Picovoice access key for Porcupine backends.
    backend : str
        Wake word backend. Supported values: ``porcupine`` and ``openwakeword``.
    keywords : list[str]
        Built-in keyword names (e.g. ["computer", "terminator"]).
    keyword_paths : list[str] | None
        Paths to custom .ppn files. Takes precedence over *keywords*.
    model_paths : list[str] | None
        Paths to custom openWakeWord model files. Used when ``backend`` is
        ``openwakeword`` and may also be provided alongside Porcupine keywords
        to add auxiliary command-model detection on the same audio stream.
    on_wake : callable
        Called with the detected keyword name. Invoked from the audio
        processing thread — callers should marshal to the main thread.
    """

    def __init__(
        self,
        access_key: str,
        backend: str = "porcupine",
        keywords: list[str] | None = None,
        keyword_paths: list[str] | None = None,
        sensitivities: list[float] | None = None,
        model_paths: list[str] | None = None,
        on_wake: Callable[[str], None] | None = None,
    ) -> None:
        self._access_key = access_key
        self._backend = backend.strip().lower()
        self._keywords = keywords or []
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities
        self._model_paths = model_paths or []
        self._on_wake = on_wake
        self._porcupine = None
        self._engine = None
        self._stream = None
        self._running = False
        self._porcupine_lock = threading.Lock()
        self._keyword_labels = list(self._keywords)
        self._oww_keyword_labels: list[str] = []
        self._trigger_threshold = 0.5
        self._armed_labels: dict[str, bool] = {}
        self._oww_armed_labels: dict[str, bool] = {}

    def start(self) -> None:
        """Start the wake word listener."""
        if self._running:
            logger.warning("WakeWordListener already running")
            return

        import sounddevice as sd

        if self._backend == "openwakeword":
            if not self._model_paths:
                raise RuntimeError(
                    "openWakeWord backend requested but no model_paths were provided"
                )

            self._start_openwakeword_engine(self._model_paths)
            self._keyword_labels = list(self._oww_keyword_labels)
            self._armed_labels = dict(self._oww_armed_labels)
            frame_length = 1280
            sample_rate = 16000
            logger.info(
                "openWakeWord initialized: models=%s frame_length=%d sample_rate=%d",
                self._model_paths,
                frame_length,
                sample_rate,
            )
        else:
            import pvporcupine

            kw_kwargs = {}
            if self._keyword_paths:
                kw_kwargs["keyword_paths"] = self._keyword_paths
                self._keyword_labels = self._keyword_paths
            else:
                kw_kwargs["keywords"] = self._keywords
                self._keyword_labels = self._keywords
            if self._sensitivities is not None:
                kw_kwargs["sensitivities"] = self._sensitivities

            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                **kw_kwargs,
            )

            frame_length = self._porcupine.frame_length
            sample_rate = self._porcupine.sample_rate

            logger.info(
                "Porcupine initialized: keywords=%s sensitivities=%s frame_length=%d sample_rate=%d",
                self._keyword_labels,
                self._sensitivities,
                frame_length,
                sample_rate,
            )
            if self._model_paths:
                self._start_openwakeword_engine(self._model_paths)
                logger.info("Auxiliary openWakeWord models enabled: %s", self._model_paths)

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
            callback=self._audio_callback,
        )
        self._running = True
        self._stream.start()
        logger.info("Wake word listener audio stream started")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback: feed audio frames into Porcupine."""
        if status:
            logger.debug("Wake word stream status: %s", status)
        if not self._running:
            return
        try:
            pcm = np.asarray(indata[:, 0], dtype=np.int16)
            keyword = self._process_frame(pcm)
            if keyword is not None and self._on_wake is not None:
                logger.info("Wake word detected: %s", keyword)
                self._on_wake(keyword)
        except Exception:
            logger.exception("Wake word listener callback failed")

    def _process_frame(self, pcm: np.ndarray) -> str | None:
        if self._backend == "openwakeword":
            return self._process_openwakeword_frame(pcm)

        detected = None
        with self._porcupine_lock:
            if not self._running:
                return None
            porcupine = self._porcupine
            if porcupine is not None:
                result = porcupine.process(pcm)
                if result >= 0:
                    detected = self._keyword_labels[result]
        if detected is not None:
            return detected
        return self._process_openwakeword_frame(pcm)

    def _start_openwakeword_engine(self, model_paths: list[str]) -> None:
        from openwakeword.model import Model

        self._engine = Model(wakeword_model_paths=model_paths)
        self._oww_keyword_labels = [self._label_for_model_path(path) for path in model_paths]
        self._oww_armed_labels = {label: True for label in self._oww_keyword_labels}

    def _process_openwakeword_frame(self, pcm: np.ndarray) -> str | None:
        model = self._engine
        if model is None:
            return None
        predictions = model.predict(pcm)
        detected: str | None = None
        labels = self._oww_keyword_labels or self._keyword_labels
        armed_labels = self._oww_armed_labels if self._oww_keyword_labels else self._armed_labels
        for label in labels:
            score = float(predictions.get(label, 0.0))
            armed = armed_labels.get(label, True)
            if score >= self._trigger_threshold and armed and detected is None:
                detected = label
                armed_labels[label] = False
            elif score < self._trigger_threshold:
                armed_labels[label] = True
        return detected

    def _label_for_model_path(self, path: str) -> str:
        stem = path.rsplit("/", 1)[-1]
        stem = stem.rsplit(".", 1)[0]
        lowered = stem.lower()
        if lowered.endswith("_model"):
            lowered = lowered[:-6]
        return lowered

    def stop(self) -> None:
        """Stop the wake word listener and release resources."""
        self._running = False
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                logger.debug("WakeWordListener stream abort failed during stop", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("WakeWordListener stream close failed during stop", exc_info=True)
        with self._porcupine_lock:
            porcupine = self._porcupine
            self._porcupine = None
            if porcupine is not None:
                porcupine.delete()
        engine = self._engine
        self._engine = None
        if engine is not None and hasattr(engine, "reset"):
            engine.reset()
        self._stream = None
        logger.info("Wake word listener stopped")
        logger.info("WakeWordListener stopped and released")

    @property
    def is_running(self) -> bool:
        return self._running
