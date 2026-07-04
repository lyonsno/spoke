"""WhisperKit ANE transcription — runs encoder+decoder on Apple Neural Engine.

Calls whisperkit-cli (brew install whisperkit-cli) as a subprocess.
Both encoder and decoder run on the ANE by default, leaving the GPU
free for other workloads. First run for a given model is slow (~4 min)
due to ANE compilation; subsequent runs use the cached compilation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms

logger = logging.getLogger(__name__)

# Model ID prefix that routes to this client
WHISPERKIT_PREFIX = "whisperkit/"

# Default WhisperKit model variant
DEFAULT_WHISPERKIT_MODEL = "medium.en"
DEFAULT_WHISPERKIT_CHUNKING_STRATEGY = "none"
_WHISPERKIT_CHUNKING_STRATEGIES = {"none", "vad"}


_HOMEBREW_PATHS = [
    "/opt/homebrew/bin/whisperkit-cli",
    "/usr/local/bin/whisperkit-cli",
]


def _find_whisperkit_cli() -> str | None:
    """Locate the whisperkit-cli binary."""
    env_path = os.environ.get("SPOKE_WHISPERKIT_CLI")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    found = shutil.which("whisperkit-cli")
    if found:
        return found
    for path in _HOMEBREW_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _whisperkit_chunking_strategy() -> str:
    """Return the effective WhisperKit CLI chunking strategy for Spoke utterances."""
    requested = os.environ.get(
        "SPOKE_WHISPERKIT_CHUNKING_STRATEGY",
        DEFAULT_WHISPERKIT_CHUNKING_STRATEGY,
    ).strip()
    if requested in _WHISPERKIT_CHUNKING_STRATEGIES:
        return requested
    logger.warning(
        "Invalid SPOKE_WHISPERKIT_CHUNKING_STRATEGY=%r; using %s",
        requested,
        DEFAULT_WHISPERKIT_CHUNKING_STRATEGY,
    )
    return DEFAULT_WHISPERKIT_CHUNKING_STRATEGY


class WhisperKitClient:
    """Transcribe audio via whisperkit-cli on the Apple Neural Engine.

    Parameters
    ----------
    model : str
        WhisperKit model variant (e.g. "medium.en", "base.en", "large-v3").
        Passed to ``whisperkit-cli transcribe --model``.
    """

    def __init__(self, model: str = DEFAULT_WHISPERKIT_MODEL) -> None:
        self._model = model
        self._cli_path = _find_whisperkit_cli()

    @staticmethod
    def available() -> bool:
        """Whether whisperkit-cli is installed and reachable."""
        return _find_whisperkit_cli() is not None

    def prepare(self) -> None:
        """Verify the CLI is available. Model download happens on first transcribe."""
        if self._cli_path is None:
            self._cli_path = _find_whisperkit_cli()
        if self._cli_path is None:
            logger.warning(
                "whisperkit-cli not found. Install with: brew install whisperkit-cli"
            )

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text."""
        if not wav_bytes:
            return ""

        cli = self._cli_path or _find_whisperkit_cli()
        if cli is None:
            logger.error("whisperkit-cli not found — cannot transcribe")
            return ""

        total_start = time.perf_counter()
        write_start = total_start
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        write_ms = (time.perf_counter() - write_start) * 1000

        try:
            chunking_strategy = _whisperkit_chunking_strategy()
            cmd = [
                cli, "transcribe",
                "--audio-path", tmp_path,
                "--model", self._model,
                "--language", "en",
                "--audio-encoder-compute-units", "cpuAndNeuralEngine",
                "--text-decoder-compute-units", "cpuAndNeuralEngine",
                "--chunking-strategy", chunking_strategy,
                "--skip-special-tokens",
                "--without-timestamps",
            ]
            logger.debug(
                "WhisperKit command: %s (model=%s, chunking=%s, bytes=%d)",
                " ".join(cmd),
                self._model,
                chunking_strategy,
                len(wav_bytes),
            )
            subprocess_start = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess_ms = (time.perf_counter() - subprocess_start) * 1000
            if result.returncode != 0:
                logger.error(
                    "whisperkit-cli failed (exit %d, model=%s, chunking=%s, "
                    "write=%.0fms, subprocess=%.0fms): %s",
                    result.returncode,
                    self._model,
                    chunking_strategy,
                    write_ms,
                    subprocess_ms,
                    result.stderr.strip(),
                )
                return ""

            text = result.stdout.strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        postprocess_start = time.perf_counter()
        text = truncate_repetition(text)
        text = repair_ontology_terms(text)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        if is_hallucination(text):
            logger.info("Discarding hallucination: %r", text)
            return ""
        logger.info(
            "WhisperKit ANE transcription (%s): %r "
            "(%d bytes audio; chunking=%s; write=%.0fms, subprocess=%.0fms, "
            "postprocess=%.0fms, total=%.0fms)",
            self._model,
            text,
            len(wav_bytes),
            chunking_strategy,
            write_ms,
            subprocess_ms,
            postprocess_ms,
            total_ms,
        )
        return text

    def unload(self) -> None:
        """No-op — subprocess-based, no resident model to unload."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Always False — subprocess-based, model is not resident."""
        return False

    def close(self) -> None:
        """No-op."""
        pass
