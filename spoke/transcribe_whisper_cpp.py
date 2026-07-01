"""whisper.cpp CoreML transcription client.

This route is an explicit escape hatch for heavy MLX/Metal contention. It is
only available when the caller has seated a whisper.cpp binary, a ggml model,
and the matching CoreML encoder bundle next to that model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms

logger = logging.getLogger(__name__)

_WHISPER_CPP_COREML_MODEL_ID = "whisper.cpp/coreml"
_DEFAULT_TIMEOUT_SECONDS = 60.0


class WhisperCppCoreMLError(RuntimeError):
    """Raised when the configured whisper.cpp CoreML route cannot transcribe."""


def _coreml_companion_path(model_path: Path) -> Path:
    """Return whisper.cpp's expected CoreML encoder bundle beside a ggml model."""
    return model_path.with_name(f"{model_path.stem}-encoder.mlmodelc")


def _configured_path(env_name: str) -> Path | None:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_binary(binary: str | os.PathLike[str] | None = None) -> Path | None:
    if binary is not None:
        return Path(binary).expanduser()
    configured = _configured_path("SPOKE_WHISPER_CPP_BINARY")
    if configured is not None:
        return configured
    discovered = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    return Path(discovered) if discovered else None


def _resolve_model_path(model_path: str | os.PathLike[str] | None = None) -> Path | None:
    if model_path is not None:
        return Path(model_path).expanduser()
    return _configured_path("SPOKE_WHISPER_CPP_MODEL")


def _resolve_timeout(timeout: float | None = None) -> float:
    if timeout is not None:
        return float(timeout)
    value = os.environ.get("SPOKE_WHISPER_CPP_TIMEOUT", "").strip()
    if not value:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        return float(value)
    except ValueError as exc:
        raise WhisperCppCoreMLError(
            f"SPOKE_WHISPER_CPP_TIMEOUT must be numeric, got {value!r}"
        ) from exc


def _binary_is_executable(binary: Path | None) -> bool:
    return binary is not None and binary.is_file() and os.access(binary, os.X_OK)


class WhisperCppCoreMLClient:
    """Transcribe WAV bytes through whisper.cpp's CoreML-backed CLI path."""

    def __init__(
        self,
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self._binary = _resolve_binary(binary)
        self._model_path = _resolve_model_path(model_path)
        self._timeout = _resolve_timeout(timeout)
        self._model_id = _WHISPER_CPP_COREML_MODEL_ID

    @staticmethod
    def available(
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        resolved_binary = _resolve_binary(binary)
        resolved_model = _resolve_model_path(model_path)
        if not _binary_is_executable(resolved_binary):
            return False
        if resolved_model is None or not resolved_model.is_file():
            return False
        return _coreml_companion_path(resolved_model).is_dir()

    def _missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not _binary_is_executable(self._binary):
            missing.append(
                "SPOKE_WHISPER_CPP_BINARY or PATH whisper-cli executable"
            )
        if self._model_path is None or not self._model_path.is_file():
            missing.append("SPOKE_WHISPER_CPP_MODEL ggml model file")
        elif not _coreml_companion_path(self._model_path).is_dir():
            missing.append(
                f"CoreML companion bundle {_coreml_companion_path(self._model_path)}"
            )
        return missing

    def prepare(self) -> None:
        """Validate the configured route without doing any hidden fallback."""
        missing = self._missing_requirements()
        if missing:
            raise WhisperCppCoreMLError(
                "whisper.cpp CoreML route is not seated: " + "; ".join(missing)
            )

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        self.prepare()
        assert self._binary is not None
        assert self._model_path is not None

        with tempfile.TemporaryDirectory(prefix="spoke-whisper-cpp-coreml-") as td:
            workdir = Path(td)
            wav_path = workdir / "input.wav"
            output_prefix = workdir / "output"
            wav_path.write_bytes(wav_bytes)

            cmd = [
                str(self._binary),
                "-m",
                str(self._model_path),
                "-f",
                str(wav_path),
                "-l",
                "en",
                "-otxt",
                "-of",
                str(output_prefix),
                "-np",
            ]
            logger.info(
                "Running whisper.cpp CoreML route: binary=%s model=%s companion=%s",
                self._binary,
                self._model_path,
                _coreml_companion_path(self._model_path),
            )
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise WhisperCppCoreMLError(
                    f"whisper.cpp timed out after {self._timeout:.1f}s"
                ) from exc

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise WhisperCppCoreMLError(
                    f"whisper.cpp failed with exit {result.returncode}: {detail}"
                )

            text_path = output_prefix.with_suffix(".txt")
            if not text_path.exists():
                raise WhisperCppCoreMLError(
                    f"whisper.cpp did not produce expected text output at {text_path}"
                )
            text = text_path.read_text().strip()

        text = truncate_repetition(text)
        text = repair_ontology_terms(text)
        if is_hallucination(text):
            logger.info("Discarding whisper.cpp hallucination: %r", text)
            return ""
        logger.info("whisper.cpp CoreML transcription: %r (%d bytes audio)", text, len(wav_bytes))
        return text

    def close(self) -> None:
        """No resident model is owned by the Python process."""
        return None
