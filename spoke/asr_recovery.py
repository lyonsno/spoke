"""Independent final-transcription recovery through WhisperKit."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable

from .dedup import is_hallucination, repair_ontology_terms, truncate_repetition

logger = logging.getLogger(__name__)

_WHISPERKIT_PROCESS_LOCK = threading.Lock()
_DEFAULT_MODEL = "medium.en"
_DEFAULT_ENCODER_COMPUTE = "cpuAndNeuralEngine"
_DEFAULT_DECODER_COMPUTE = "cpuOnly"
_KNOWN_BINARY_PATHS = (
    "/opt/homebrew/bin/whisperkit-cli",
    "/usr/local/bin/whisperkit-cli",
    "~/.local/bin/whisperkit-cli",
)


def _resolve_whisperkit_binary(explicit: str | None) -> str | None:
    if explicit:
        return str(Path(explicit).expanduser())

    env_binary = os.environ.get("SPOKE_WHISPERKIT_RECOVERY_BINARY", "").strip()
    if env_binary:
        return str(Path(env_binary).expanduser())

    discovered = shutil.which("whisperkit-cli")
    if discovered:
        return discovered

    for raw_path in _KNOWN_BINARY_PATHS:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


class WhisperKitRecoveryClient:
    """Run one process-serialized WhisperKit decode outside the MLX route."""

    def __init__(
        self,
        *,
        binary: str | None = None,
        model: str | None = None,
        encoder_compute: str | None = None,
        decoder_compute: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._binary = _resolve_whisperkit_binary(binary)
        self._model = model or os.environ.get(
            "SPOKE_WHISPERKIT_RECOVERY_MODEL", _DEFAULT_MODEL
        )
        self._encoder_compute = encoder_compute or os.environ.get(
            "SPOKE_WHISPERKIT_RECOVERY_ENCODER_COMPUTE", _DEFAULT_ENCODER_COMPUTE
        )
        self._decoder_compute = decoder_compute or os.environ.get(
            "SPOKE_WHISPERKIT_RECOVERY_DECODER_COMPUTE", _DEFAULT_DECODER_COMPUTE
        )
        self._runner = runner
        self.last_route: str | None = None

    @property
    def route(self) -> str:
        return (
            f"whisperkit-cli[{self._binary}]:{self._model}:"
            f"encoder={self._encoder_compute}:"
            f"decoder={self._decoder_compute}"
        )

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        if not self._binary:
            raise RuntimeError("whisperkit-cli is not installed or not on PATH")

        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="spoke-whisperkit-recovery-",
                suffix=".wav",
                delete=False,
            ) as wav_file:
                wav_file.write(wav_bytes)
                wav_path = Path(wav_file.name)

            command = [
                self._binary,
                "transcribe",
                "--model",
                self._model,
                "--audio-path",
                str(wav_path),
                "--audio-encoder-compute-units",
                self._encoder_compute,
                "--text-decoder-compute-units",
                self._decoder_compute,
                "--chunking-strategy",
                "none",
                "--concurrent-worker-count",
                "1",
                "--language",
                "en",
                "--without-timestamps",
                "--skip-special-tokens",
            ]
            self.last_route = self.route
            started = time.monotonic()
            logger.warning("ASR recovery starting via %s", self.last_route)
            with _WHISPERKIT_PROCESS_LOCK:
                result = self._runner(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            elapsed = time.monotonic() - started
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "no output"
                raise RuntimeError(
                    f"WhisperKit recovery failed with exit {result.returncode}: {detail}"
                )

            text = result.stdout.strip()
            if not text:
                raise RuntimeError("WhisperKit recovery returned a blank transcript")
            text = repair_ontology_terms(truncate_repetition(text))
            if is_hallucination(text):
                raise RuntimeError(
                    f"WhisperKit recovery returned a filtered hallucination: {text!r}"
                )
            logger.info(
                "ASR recovery completed via %s in %.3fs", self.last_route, elapsed
            )
            return text
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)
