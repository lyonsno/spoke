"""Always-on wake word detection via Picovoice Porcupine.

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
    """Porcupine-based wake word detector with its own audio stream.

    Parameters
    ----------
    access_key : str
        Picovoice access key.
    keywords : list[str]
        Built-in keyword names (e.g. ["computer", "terminator"]).
    keyword_paths : list[str] | None
        Paths to custom .ppn files. Takes precedence over *keywords*.
    on_wake : callable
        Called with the detected keyword name. Invoked from the audio
        processing thread — callers should marshal to the main thread.
    """

    def __init__(
        self,
        access_key: str,
        keywords: list[str] | None = None,
        keyword_paths: list[str] | None = None,
        on_wake: Callable[[str], None] | None = None,
    ) -> None:
        self._access_key = access_key
        self._keywords = keywords or []
        self._keyword_paths = keyword_paths
        self._on_wake = on_wake
        self._porcupine = None
        self._stream = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the wake word listener."""
        if self._running:
            logger.warning("WakeWordListener already running")
            return

        import pvporcupine
        import sounddevice as sd

        kw_kwargs = {}
        if self._keyword_paths:
            kw_kwargs["keyword_paths"] = self._keyword_paths
            kw_labels = self._keyword_paths
        else:
            kw_kwargs["keywords"] = self._keywords
            kw_labels = self._keywords

        self._porcupine = pvporcupine.create(
            access_key=self._access_key,
            **kw_kwargs,
        )

        frame_length = self._porcupine.frame_length
        sample_rate = self._porcupine.sample_rate

        logger.info(
            "Porcupine initialized: keywords=%s frame_length=%d sample_rate=%d",
            kw_labels, frame_length, sample_rate,
        )

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(frame_length, sample_rate),
            daemon=True,
            name="wakeword-listener",
        )
        self._thread.start()

    def _listen_loop(self, frame_length: int, sample_rate: int) -> None:
        """Blocking loop that reads audio and feeds Porcupine."""
        import sounddevice as sd

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_length,
            ) as stream:
                self._stream = stream
                logger.info("Wake word listener audio stream started")
                while self._running:
                    data, status = stream.read(frame_length)
                    if status:
                        logger.debug("Wake word stream status: %s", status)
                    if not self._running:
                        break

                    pcm = data[:, 0]  # mono int16
                    result = self._porcupine.process(pcm)
                    if result >= 0:
                        if self._keyword_paths:
                            keyword = self._keyword_paths[result]
                        else:
                            keyword = self._keywords[result]
                        logger.info("Wake word detected: %s", keyword)
                        if self._on_wake is not None:
                            self._on_wake(keyword)
        except Exception:
            logger.exception("Wake word listener failed")
        finally:
            self._stream = None
            logger.info("Wake word listener stopped")

    def stop(self) -> None:
        """Stop the wake word listener and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        self._stream = None
        logger.info("WakeWordListener stopped and released")

    @property
    def is_running(self) -> bool:
        return self._running
