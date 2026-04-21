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
        ``openwakeword``.
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
        model_paths: list[str] | None = None,
        on_wake: Callable[[str], None] | None = None,
    ) -> None:
        self._access_key = access_key
        self._backend = backend.strip().lower()
        self._keywords = keywords or []
        self._keyword_paths = keyword_paths
        self._model_paths = model_paths or []
        self._on_wake = on_wake
        self._porcupine = None
        self._engine = None
        self._stream = None
        self._running = False
        self._porcupine_lock = threading.Lock()
        self._keyword_labels = list(self._keywords)
        self._trigger_threshold = 0.5
        self._armed_labels: dict[str, bool] = {}

    def start(self) -> None:
        """Start the wake word listener."""
        if self._running:
            logger.warning("WakeWordListener already running")
            return

        import sounddevice as sd

        if self._backend == "openwakeword":
            from openwakeword.model import Model

            if not self._model_paths:
                raise RuntimeError(
                    "openWakeWord backend requested but no model_paths were provided"
                )

            self._engine = Model(wakeword_models=self._model_paths)
            self._keyword_labels = [self._label_for_model_path(path) for path in self._model_paths]
            self._armed_labels = {label: True for label in self._keyword_labels}
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

            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                **kw_kwargs,
            )

            frame_length = self._porcupine.frame_length
            sample_rate = self._porcupine.sample_rate

            logger.info(
                "Porcupine initialized: keywords=%s frame_length=%d sample_rate=%d",
                self._keyword_labels,
                frame_length,
                sample_rate,
            )

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
            model = self._engine
            if model is None:
                return None
            predictions = model.predict(pcm)
            detected: str | None = None
            for label in self._keyword_labels:
                score = float(predictions.get(label, 0.0))
                armed = self._armed_labels.get(label, True)
                if score >= self._trigger_threshold and armed and detected is None:
                    detected = label
                    self._armed_labels[label] = False
                elif score < self._trigger_threshold:
                    self._armed_labels[label] = True
            return detected

        with self._porcupine_lock:
            if not self._running:
                return None
            porcupine = self._porcupine
            if porcupine is None:
                return None
            result = porcupine.process(pcm)
        if result >= 0:
            return self._keyword_labels[result]
        return None

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
