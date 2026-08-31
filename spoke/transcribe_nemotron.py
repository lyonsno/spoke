"""Full-buffer Nemotron ASR through NVIDIA's CPU-only NeMo-Speech CLI."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import wave

from .dedup import is_hallucination, repair_ontology_terms, truncate_repetition
from .transcription_prompt import TranscriptionPromptProvider

logger = logging.getLogger(__name__)

_NEMOTRON_CPU_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b@cpu"
_NEMOTRON_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
_NEMOTRON_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
_NEMOTRON_FILENAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
_NEMOTRON_EXPECTED_SHA256 = (
    "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae"
)
_DEFAULT_BINARY = Path.home() / ".local" / "share" / "spoke" / "nemo-speech-cpu" / "bin" / "nemo-speech"
_DEFAULT_MODEL = (
    Path.home()
    / "Library"
    / "Caches"
    / "NeMoSpeech"
    / "models"
    / "nvidia"
    / "nemotron-3.5-asr-streaming-0.6b"
    / _NEMOTRON_REVISION
    / _NEMOTRON_FILENAME
)
_DEFAULT_FAILURE_DIR = (
    Path.home() / "Library" / "Application Support" / "Spoke" / "nemotron-failures"
)
_DEFAULT_SPEECH_CONTEXT_BOOST = 2.5


class NemotronCPUError(RuntimeError):
    """Raised when the explicit CPU route cannot return a valid transcript."""


def _configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


def _resolve_timeout(timeout: float | None = None) -> float | None:
    if timeout is not None:
        return float(timeout)
    value = os.environ.get("SPOKE_NEMOTRON_TIMEOUT", "").strip()
    if not value or value.lower() in {"none", "off", "unlimited"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise NemotronCPUError(
            f"SPOKE_NEMOTRON_TIMEOUT must be numeric or 'off', got {value!r}"
        ) from exc
    if parsed <= 0:
        return None
    return parsed


def _resolve_boost(boost: float | None = None) -> float:
    if boost is not None:
        return float(boost)
    value = os.environ.get("SPOKE_NEMOTRON_SPEECH_CONTEXT_BOOST", "").strip()
    if not value:
        return _DEFAULT_SPEECH_CONTEXT_BOOST
    try:
        return float(value)
    except ValueError as exc:
        raise NemotronCPUError(
            "SPOKE_NEMOTRON_SPEECH_CONTEXT_BOOST must be numeric, "
            f"got {value!r}"
        ) from exc


def _prompt_phrases(text: str) -> list[str]:
    """Map the caller-owned comma/newline lexicon to uncapped decoder phrases."""
    phrases: list[str] = []
    for value in re.split(r"[,\n]", text):
        phrase = value.strip().rstrip(".")
        if phrase.lower().startswith("vocabulary:"):
            phrase = phrase.split(":", 1)[1].strip()
        if phrase:
            phrases.append(phrase)
    return phrases


def _wav_duration_seconds(wav_bytes: bytes) -> float | None:
    try:
        import io

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            rate = wav_file.getframerate()
            return wav_file.getnframes() / float(rate) if rate > 0 else None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NemotronCPUClient:
    """Transcribe one complete WAV through a pinned, CPU-only runtime."""

    def __init__(
        self,
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
        speech_context_boost: float | None = None,
        expected_model_sha256: str = _NEMOTRON_EXPECTED_SHA256,
        prompt_provider: TranscriptionPromptProvider | None = None,
        failure_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._binary = (
            Path(binary).expanduser()
            if binary is not None
            else _configured_path("SPOKE_NEMO_SPEECH_BINARY", _DEFAULT_BINARY)
        )
        self._model_path = (
            Path(model_path).expanduser()
            if model_path is not None
            else _configured_path("SPOKE_NEMOTRON_MODEL", _DEFAULT_MODEL)
        )
        self._timeout = _resolve_timeout(timeout)
        self._speech_context_boost = _resolve_boost(speech_context_boost)
        self._expected_model_sha256 = expected_model_sha256.lower()
        self._model_sha256_actual: str | None = None
        self._verified_model_stat: tuple[int, int, int, int] | None = None
        self._prompt_provider = (
            prompt_provider or TranscriptionPromptProvider.from_environment()
        )
        self._failure_dir = (
            Path(failure_dir).expanduser()
            if failure_dir is not None
            else _configured_path("SPOKE_NEMOTRON_FAILURE_DIR", _DEFAULT_FAILURE_DIR)
        )
        self._last_receipt: dict | None = None

    @staticmethod
    def available(
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        resolved_binary = (
            Path(binary).expanduser()
            if binary is not None
            else _configured_path("SPOKE_NEMO_SPEECH_BINARY", _DEFAULT_BINARY)
        )
        resolved_model = (
            Path(model_path).expanduser()
            if model_path is not None
            else _configured_path("SPOKE_NEMOTRON_MODEL", _DEFAULT_MODEL)
        )
        return (
            resolved_binary.is_file()
            and os.access(resolved_binary, os.X_OK)
            and resolved_model.is_file()
        )

    def prepare(self) -> None:
        missing: list[str] = []
        if not self._binary.is_file() or not os.access(self._binary, os.X_OK):
            missing.append(f"CPU nemo-speech executable {self._binary}")
        if not self._model_path.is_file():
            missing.append(f"pinned Nemotron GGUF {self._model_path}")
        if missing:
            raise NemotronCPUError("Nemotron CPU route is not seated: " + "; ".join(missing))
        model_stat = self._model_path.stat()
        stat_identity = (
            model_stat.st_dev,
            model_stat.st_ino,
            model_stat.st_size,
            model_stat.st_mtime_ns,
        )
        if stat_identity != self._verified_model_stat:
            actual_sha256 = _sha256_file(self._model_path)
            if actual_sha256 != self._expected_model_sha256:
                raise NemotronCPUError(
                    "Nemotron GGUF SHA-256 mismatch: "
                    f"expected {self._expected_model_sha256}, got {actual_sha256} "
                    f"for {self._model_path}"
                )
            self._model_sha256_actual = actual_sha256
            self._verified_model_stat = stat_identity

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        self.prepare()

        prompt = self._prompt_provider.resolve()
        phrases = _prompt_phrases(prompt.text)
        prompt_receipt = prompt.receipt(
            supported=True,
            effective=bool(phrases),
        )
        audio_sha256 = hashlib.sha256(wav_bytes).hexdigest()
        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="spoke-nemotron-cpu-") as td:
            wav_path = Path(td) / "input.wav"
            wav_path.write_bytes(wav_bytes)
            cmd = [
                str(self._binary),
                "--json",
                "transcribe",
                str(wav_path),
                "--model",
                str(self._model_path),
                "--device",
                "cpu",
                "--language",
                "en",
                "--format",
                "json",
                "--no-batching",
            ]
            for phrase in phrases:
                cmd.extend(("--speech-context", phrase))
            if phrases:
                cmd.extend(
                    ("--speech-context-boost", str(self._speech_context_boost))
                )

            route = {
                "schema": "spoke.nemotron-cpu-transcription.v1",
                "requested_model": _NEMOTRON_CPU_MODEL_ID,
                "effective_model_path": str(self._model_path),
                "model_repo": _NEMOTRON_REPO,
                "model_revision": _NEMOTRON_REVISION,
                "model_sha256_expected": self._expected_model_sha256,
                "model_sha256_actual": self._model_sha256_actual,
                "effective_binary": str(self._binary),
                "effective_device": "cpu",
                "streaming": False,
                "endpointing": False,
                "vad": False,
                "audio_sha256": audio_sha256,
                "audio_bytes": len(wav_bytes),
                "audio_duration_seconds": _wav_duration_seconds(wav_bytes),
                "prompt": prompt_receipt,
                "speech_context_count": len(phrases),
                "speech_context_boost": self._speech_context_boost if phrases else None,
            }
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
                report = self._write_failure(
                    route,
                    phase="transcribe_timeout",
                    detail=str(exc),
                    wall_seconds=time.monotonic() - started,
                )
                raise NemotronCPUError(
                    f"Nemotron CPU transcription timed out; report={report}"
                ) from exc

            wall_seconds = time.monotonic() - started
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                report = self._write_failure(
                    route,
                    phase="transcribe_process",
                    detail=detail,
                    wall_seconds=wall_seconds,
                    exit_code=result.returncode,
                )
                raise NemotronCPUError(
                    f"Nemotron CPU transcription exited {result.returncode}; report={report}"
                )

            try:
                payload = json.loads(result.stdout)
                text = payload["text"]
                if not isinstance(text, str):
                    raise TypeError("text is not a string")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                report = self._write_failure(
                    route,
                    phase="parse_output",
                    detail=f"{type(exc).__name__}: {exc}",
                    wall_seconds=wall_seconds,
                )
                raise NemotronCPUError(
                    f"Nemotron CPU returned invalid JSON output; report={report}"
                ) from exc

        text = truncate_repetition(text.strip())
        text = repair_ontology_terms(text)
        if is_hallucination(text):
            text = ""
        self._last_receipt = {
            **route,
            "status": "success",
            "wall_seconds": wall_seconds,
            "reported_duration_seconds": payload.get("duration"),
        }
        logger.info(
            "Nemotron CPU transcription: wall=%.3fs audio=%s bytes=%d prompt=%s",
            wall_seconds,
            route["audio_duration_seconds"],
            len(wav_bytes),
            prompt.sha256,
        )
        return text

    def _write_failure(
        self,
        route: dict,
        *,
        phase: str,
        detail: str,
        wall_seconds: float,
        exit_code: int | None = None,
    ) -> Path:
        self._failure_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = self._failure_dir / f"{stamp}-{route['audio_sha256'][:12]}.json"
        payload = {
            **route,
            "status": "failure",
            "failure_phase": phase,
            "detail": detail,
            "exit_code": exit_code,
            "wall_seconds": wall_seconds,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
        self._last_receipt = payload
        return path

    def close(self) -> None:
        return None


def _write_replay_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_replay(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    client: NemotronCPUClient | None = None,
) -> dict:
    """Replay one retained WAV and preserve success or failure evidence."""
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    wav_bytes = source.read_bytes()
    expected_audio_sha256 = hashlib.sha256(wav_bytes).hexdigest()
    active_client = client or NemotronCPUClient()
    started = time.monotonic()
    try:
        transcript = active_client.transcribe(wav_bytes)
        receipt = active_client._last_receipt
        if not isinstance(receipt, dict):
            raise NemotronCPUError("Replay route identity is missing")
        identity_ok = (
            receipt.get("requested_model") == _NEMOTRON_CPU_MODEL_ID
            and receipt.get("effective_device") == "cpu"
            and receipt.get("audio_sha256") == expected_audio_sha256
            and receipt.get("streaming") is False
            and receipt.get("endpointing") is False
            and receipt.get("vad") is False
        )
        if not identity_ok:
            report = {
                "schema": "spoke.nemotron-cpu-replay.v1",
                "status": "failure",
                "failure_phase": "route_identity",
                "input_path": str(source),
                "expected_audio_sha256": expected_audio_sha256,
                "wall_seconds": time.monotonic() - started,
                **receipt,
            }
            report.pop("transcript", None)
            _write_replay_report(destination, report)
            raise NemotronCPUError(
                f"Nemotron replay route identity did not prove CPU/full-buffer use; report={destination}"
            )
        report = {
            "schema": "spoke.nemotron-cpu-replay.v1",
            "status": "success",
            "input_path": str(source),
            "transcript": transcript,
            "receipt": receipt,
        }
        _write_replay_report(destination, report)
        return report
    except Exception as exc:
        if not destination.exists():
            receipt = getattr(active_client, "_last_receipt", None)
            report = {
                "schema": "spoke.nemotron-cpu-replay.v1",
                "status": "failure",
                "failure_phase": (
                    receipt.get("failure_phase", "transcription")
                    if isinstance(receipt, dict)
                    else "transcription"
                ),
                "input_path": str(source),
                "expected_audio_sha256": expected_audio_sha256,
                "wall_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "receipt": receipt,
            }
            _write_replay_report(destination, report)
        raise


def replay_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay retained Spoke WAV audio through Nemotron's CPU-only route."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_replay(args.input, args.output)
    return 0
