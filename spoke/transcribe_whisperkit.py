"""WhisperKit ANE transcription — runs encoder+decoder on Apple Neural Engine.

Calls whisperkit-cli (brew install whisperkit-cli) as a subprocess.
Both encoder and decoder run on the ANE by default, leaving the GPU
free for other workloads. First run for a given model is slow (~4 min)
due to ANE compilation; subsequent runs use the cached compilation.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms

logger = logging.getLogger(__name__)

# Model ID prefix that routes to this client
WHISPERKIT_PREFIX = "whisperkit/"

# Default WhisperKit model variant
DEFAULT_WHISPERKIT_MODEL = "medium.en"
DEFAULT_WHISPERKIT_CHUNKING_STRATEGY = "none"
_WHISPERKIT_CHUNKING_STRATEGIES = {"none", "vad"}
_WHISPERKIT_ENCODER_COMPUTE_UNITS = "cpuAndNeuralEngine"
_WHISPERKIT_DECODER_COMPUTE_UNITS = "cpuAndNeuralEngine"
_WHISPERKIT_TIMEOUT_SECONDS = 120
_WHISPERKIT_SUSPECT_MIN_SECONDS = 8.0
_WHISPERKIT_SUSPECT_MIN_CHARS = 80
_WHISPERKIT_SUSPECT_MIN_CHARS_PER_SECOND = 3.0
_WHISPERKIT_SUSPECT_RETRY_COUNT = 1


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


def _wav_duration_seconds(wav_bytes: bytes) -> float | None:
    """Return WAV duration when the input is parseable."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frame_rate = wf.getframerate()
            if frame_rate <= 0:
                return None
            return wf.getnframes() / frame_rate
    except (wave.Error, EOFError, OSError, ValueError):
        return None


def _whisperkit_suspect_spool_dir() -> Path:
    configured = os.environ.get("SPOKE_WHISPERKIT_SUSPECT_SPOOL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "Spoke" / "whisperkit-suspect-spool"


def _whisperkit_suspicion_reason(text: str, duration_s: float | None) -> str | None:
    if duration_s is None or duration_s < _WHISPERKIT_SUSPECT_MIN_SECONDS:
        return None
    required_chars = max(
        _WHISPERKIT_SUSPECT_MIN_CHARS,
        int(duration_s * _WHISPERKIT_SUSPECT_MIN_CHARS_PER_SECOND),
    )
    if len(text.strip()) < required_chars:
        return "too_short_for_audio_duration"
    return None


def _record_whisperkit_suspect_bundle(
    *,
    wav_bytes: bytes,
    status: str,
    model: str,
    cli_path: str,
    chunking_strategy: str,
    duration_s: float | None,
    write_ms: float,
    postprocess_ms: float,
    total_ms: float,
    attempts: list[dict[str, object]],
    chosen_output: str,
) -> None:
    spool_dir = _whisperkit_suspect_spool_dir()
    try:
        spool_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(wav_bytes).hexdigest()[:12]
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        stem = f"whisperkit-suspect-{stamp}-{digest}"
        audio_path = spool_dir / f"{stem}.wav"
        report_path = spool_dir / f"{stem}.json"
        audio_path.write_bytes(wav_bytes)
        report = {
            "phase": "whisperkit_suspicious_success",
            "status": status,
            "effective_cli_path": cli_path,
            "effective_model": model,
            "effective_chunking_strategy": chunking_strategy,
            "audio_encoder_compute_units": _WHISPERKIT_ENCODER_COMPUTE_UNITS,
            "text_decoder_compute_units": _WHISPERKIT_DECODER_COMPUTE_UNITS,
            "timeout_seconds": _WHISPERKIT_TIMEOUT_SECONDS,
            "audio_path": str(audio_path),
            "audio_bytes": len(wav_bytes),
            "duration_seconds": duration_s,
            "write_ms": write_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "attempts": attempts,
            "chosen_output": chosen_output,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        logger.warning(
            "WhisperKit suspect bundle preserved at %s (status=%s, model=%s, "
            "chunking=%s, duration=%s, bytes=%d)",
            report_path,
            status,
            model,
            chunking_strategy,
            f"{duration_s:.1f}s" if duration_s is not None else "unknown",
            len(wav_bytes),
        )
    except OSError as exc:
        logger.error("Failed to preserve WhisperKit suspect bundle in %s: %s", spool_dir, exc)


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
                "--audio-encoder-compute-units", _WHISPERKIT_ENCODER_COMPUTE_UNITS,
                "--text-decoder-compute-units", _WHISPERKIT_DECODER_COMPUTE_UNITS,
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
            duration_s = _wav_duration_seconds(wav_bytes)
            attempts: list[dict[str, object]] = []
            text = ""
            postprocess_ms = 0.0
            final_suspicion_reason: str | None = None
            for attempt_index in range(_WHISPERKIT_SUSPECT_RETRY_COUNT + 1):
                subprocess_start = time.perf_counter()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_WHISPERKIT_TIMEOUT_SECONDS,
                )
                subprocess_ms = (time.perf_counter() - subprocess_start) * 1000
                raw_stdout = result.stdout or ""
                raw_stderr = result.stderr or ""
                raw_text = raw_stdout.strip()
                if result.returncode != 0:
                    attempts.append(
                        {
                            "attempt": attempt_index + 1,
                            "returncode": result.returncode,
                            "stdout_len": len(raw_stdout),
                            "stderr_len": len(raw_stderr),
                            "subprocess_ms": subprocess_ms,
                            "suspicious": False,
                            "suspicion_reason": None,
                        }
                    )
                    logger.error(
                        "whisperkit-cli failed (exit %d, model=%s, chunking=%s, "
                        "write=%.0fms, subprocess=%.0fms): %s",
                        result.returncode,
                        self._model,
                        chunking_strategy,
                        write_ms,
                        subprocess_ms,
                        raw_stderr.strip(),
                    )
                    return ""

                postprocess_start = time.perf_counter()
                attempt_text = truncate_repetition(raw_text)
                attempt_text = repair_ontology_terms(attempt_text)
                attempt_postprocess_ms = (time.perf_counter() - postprocess_start) * 1000
                postprocess_ms += attempt_postprocess_ms
                if is_hallucination(attempt_text):
                    logger.info("Discarding hallucination candidate: %r", attempt_text)
                    attempt_text = ""
                suspicion_reason = _whisperkit_suspicion_reason(attempt_text, duration_s)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "returncode": result.returncode,
                        "stdout_len": len(raw_stdout),
                        "stderr_len": len(raw_stderr),
                        "subprocess_ms": subprocess_ms,
                        "postprocess_ms": attempt_postprocess_ms,
                        "suspicious": suspicion_reason is not None,
                        "suspicion_reason": suspicion_reason,
                    }
                )
                text = attempt_text
                final_suspicion_reason = suspicion_reason
                if suspicion_reason is None:
                    break
                if attempt_index < _WHISPERKIT_SUSPECT_RETRY_COUNT:
                    logger.warning(
                        "WhisperKit returned suspiciously short success; retrying "
                        "(model=%s, chunking=%s, duration=%s, stdout_len=%d, text_len=%d)",
                        self._model,
                        chunking_strategy,
                        f"{duration_s:.1f}s" if duration_s is not None else "unknown",
                        len(raw_stdout),
                        len(attempt_text),
                    )
            total_ms = (time.perf_counter() - total_start) * 1000
            if any(attempt["suspicious"] for attempt in attempts):
                status = (
                    "suspicious_unrecovered"
                    if final_suspicion_reason is not None
                    else "recovered_by_retry"
                )
                _record_whisperkit_suspect_bundle(
                    wav_bytes=wav_bytes,
                    status=status,
                    model=self._model,
                    cli_path=cli,
                    chunking_strategy=chunking_strategy,
                    duration_s=duration_s,
                    write_ms=write_ms,
                    postprocess_ms=postprocess_ms,
                    total_ms=total_ms,
                    attempts=attempts,
                    chosen_output=text,
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        logger.info(
            "WhisperKit ANE transcription (%s): %r "
            "(%d bytes audio; chunking=%s; write=%.0fms, subprocess=%.0fms, "
            "postprocess=%.0fms, total=%.0fms)",
            self._model,
            text,
            len(wav_bytes),
            chunking_strategy,
            write_ms,
            sum(float(attempt["subprocess_ms"]) for attempt in attempts),
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
