"""Serial non-Metal recovery and durable route evidence for final ASR."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Iterable

from .transcribe import TranscriptionClient
from .transcription_prompt import TranscriptionPromptProvider


_REPORT_SCHEMA = "spoke.asr-route-report.v1"
_DEFAULT_REPORT_ROOT = (
    Path.home() / "Library" / "Application Support" / "Spoke" / "asr-reports"
)
_DEFAULT_REMOTE_MODEL = "mlx-community/whisper-large-v3-turbo"
_DEFAULT_WHISPERKIT_MODEL = "medium.en"


class ASRRecoveryError(RuntimeError):
    """Every configured serial escape route failed."""


def default_report_path(
    wav_bytes: bytes,
    *,
    spool_record=None,
) -> Path:
    if spool_record is not None:
        metadata_path = getattr(spool_record, "metadata_path", None)
        if metadata_path is not None:
            return Path(metadata_path).with_suffix(".asr.json")
    root = Path(
        os.environ.get("SPOKE_ASR_REPORT_DIR", str(_DEFAULT_REPORT_ROOT))
    ).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(wav_bytes).hexdigest()[:12]
    return root / f"asr-{stamp}-{digest}.json"


class ASRRouteReporter:
    """Persist requested/effective route identity throughout one utterance."""

    def __init__(
        self,
        path: Path,
        *,
        wav_bytes: bytes,
        requested_route: dict,
    ) -> None:
        self.path = Path(path)
        self._payload = {
            "schema": _REPORT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "failure_phase": None,
            "last_trustworthy_evidence": "raw_audio",
            "requested_route": dict(requested_route),
            "effective_route": None,
            "audio": {
                "sha256": hashlib.sha256(wav_bytes).hexdigest(),
                "byte_count": len(wav_bytes),
            },
            "backend_events": [],
            "primary_failure": None,
            "recovery_attempts": [],
        }
        self._write()

    def backend_event(self, event: dict) -> None:
        self._payload["backend_events"].append(dict(event))
        self._payload["last_trustworthy_evidence"] = str(
            event.get("event") or "backend_event"
        )
        self._write()

    def record_primary_failure(self, error: Exception) -> None:
        self._payload["primary_failure"] = _error_identity(error)
        self._payload["status"] = "recovering"
        self._payload["failure_phase"] = "primary_route"
        self._payload["last_trustworthy_evidence"] = "primary_failure"
        self._write()

    def record_recovery_attempt(self, attempt: dict) -> None:
        self._payload["recovery_attempts"].append(dict(attempt))
        self._payload["last_trustworthy_evidence"] = "recovery_attempt"
        self._write()

    def succeed(self, *, route: dict, transcript: str) -> None:
        self._payload["status"] = "succeeded"
        self._payload["failure_phase"] = None
        self._payload["effective_route"] = dict(route)
        self._payload["transcript"] = {
            "char_count": len(transcript),
            "sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        }
        self._payload["last_trustworthy_evidence"] = "transcript"
        self._write()

    def fail(self, *, phase: str, error: Exception) -> None:
        self._payload["status"] = "failed"
        self._payload["failure_phase"] = phase
        self._payload["terminal_error"] = _error_identity(error)
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(self._payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        )
        tmp.write_bytes(encoded)
        tmp.replace(self.path)


class SerialASRRecovery:
    """Try distinct non-Metal routes one at a time."""

    def __init__(self, routes: Iterable[tuple[str, object]]) -> None:
        self._routes = list(routes)
        self._lock = threading.Lock()

    def recover(
        self,
        wav_bytes: bytes,
        *,
        primary_failure: Exception,
        reporter: ASRRouteReporter,
    ) -> str:
        with self._lock:
            return self._recover_serial(
                wav_bytes,
                primary_failure=primary_failure,
                reporter=reporter,
            )

    def _recover_serial(
        self,
        wav_bytes: bytes,
        *,
        primary_failure: Exception,
        reporter: ASRRouteReporter,
    ) -> str:
        reporter.record_primary_failure(primary_failure)
        if not self._routes:
            error = ASRRecoveryError("no serial ASR recovery routes are available")
            reporter.fail(phase="route_selection", error=error)
            raise error from primary_failure

        errors = []
        for route_name, client in self._routes:
            route = dict(client.route_identity())
            started = time.monotonic()
            try:
                text = client.transcribe(wav_bytes)
                if not text.strip():
                    raise ASRRecoveryError(
                        f"{route_name} returned a blank recovery transcript"
                    )
            except Exception as exc:
                route = dict(client.route_identity())
                attempt = {
                    "requested_route": route_name,
                    "effective_route": route,
                    "status": "failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": _error_identity(exc),
                }
                reporter.record_recovery_attempt(attempt)
                errors.append(f"{route_name}: {exc}")
                continue

            route = dict(client.route_identity())
            attempt = {
                "requested_route": route_name,
                "effective_route": route,
                "status": "succeeded",
                "elapsed_seconds": time.monotonic() - started,
                "transcript_char_count": len(text),
            }
            reporter.record_recovery_attempt(attempt)
            reporter.succeed(route=route, transcript=text)
            return text

        error = ASRRecoveryError(
            "serial ASR recovery failed: " + "; ".join(errors)
        )
        reporter.fail(phase="serial_recovery", error=error)
        raise error from primary_failure


class RemoteASREscapeClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = _DEFAULT_REMOTE_MODEL,
        api_key: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = TranscriptionClient(
            self._base_url,
            model=model,
            api_key=api_key,
        )

    def route_identity(self) -> dict:
        return {
            "route": "remote-openai-whisper",
            "url": self._base_url,
            "model": self._model,
            "prompt": getattr(self._client, "_last_prompt_receipt", None),
        }

    def transcribe(self, wav_bytes: bytes) -> str:
        return self._client.transcribe(wav_bytes)

    def close(self) -> None:
        self._client.close()


class WhisperKitEscapeClient:
    def __init__(
        self,
        *,
        cli_path: str,
        model: str = _DEFAULT_WHISPERKIT_MODEL,
        prompt_provider: TranscriptionPromptProvider | None = None,
    ) -> None:
        self._cli_path = cli_path
        self._model = model
        self._prompt_provider = (
            prompt_provider or TranscriptionPromptProvider.from_environment()
        )
        self._last_prompt_receipt = None

    def route_identity(self) -> dict:
        return {
            "route": "whisperkit-cli",
            "cli_path": self._cli_path,
            "model": self._model,
            "audio_encoder_compute_units": "cpuAndNeuralEngine",
            "text_decoder_compute_units": "cpuOnly",
            "chunking_strategy": "none",
            "concurrent_worker_count": 1,
            "prompt": self._last_prompt_receipt,
        }

    def transcribe(self, wav_bytes: bytes) -> str:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio:
                audio.write(wav_bytes)
                temp_path = audio.name
            command = [
                self._cli_path,
                "transcribe",
                "--model",
                self._model,
                "--audio-path",
                temp_path,
                "--audio-encoder-compute-units",
                "cpuAndNeuralEngine",
                "--text-decoder-compute-units",
                "cpuOnly",
                "--chunking-strategy",
                "none",
                "--concurrent-worker-count",
                "1",
                "--language",
                "en",
                "--without-timestamps",
                "--skip-special-tokens",
            ]
            prompt = self._prompt_provider.resolve()
            if prompt.text:
                command.extend(["--prompt", prompt.text])
            self._last_prompt_receipt = prompt.receipt(
                supported=True,
                effective=bool(prompt.text),
            )
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"whisperkit-cli exited {result.returncode}: {detail}"
                )
            return result.stdout.strip()
        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink()
                except FileNotFoundError:
                    pass


def build_serial_asr_recovery(
    *,
    remote_url: str = "",
    remote_model: str = _DEFAULT_REMOTE_MODEL,
    remote_api_key: str = "",
) -> SerialASRRecovery:
    routes: list[tuple[str, object]] = []
    effective_remote_url = (
        os.environ.get("SPOKE_ASR_RECOVERY_URL", "").strip()
        or remote_url.strip()
    )
    if effective_remote_url:
        routes.append(
            (
                "remote",
                RemoteASREscapeClient(
                    effective_remote_url,
                    model=(
                        os.environ.get("SPOKE_ASR_RECOVERY_MODEL", "").strip()
                        or remote_model
                    ),
                    api_key=remote_api_key,
                ),
            )
        )

    cli_path = _find_whisperkit_cli()
    if cli_path is not None:
        routes.append(
            (
                "whisperkit",
                WhisperKitEscapeClient(
                    cli_path=cli_path,
                    model=(
                        os.environ.get("SPOKE_WHISPERKIT_RECOVERY_MODEL", "").strip()
                        or _DEFAULT_WHISPERKIT_MODEL
                    ),
                ),
            )
        )
    return SerialASRRecovery(routes)


def _find_whisperkit_cli() -> str | None:
    configured = os.environ.get("SPOKE_WHISPERKIT_CLI", "").strip()
    candidates = [
        configured,
        shutil.which("whisperkit-cli"),
        "/opt/homebrew/bin/whisperkit-cli",
        "/usr/local/bin/whisperkit-cli",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _error_identity(error: Exception) -> dict:
    identity = {
        "type": type(error).__name__,
        "message": str(error),
    }
    for name in (
        "phase",
        "timeout",
        "window_index",
        "temperature",
        "token_count",
        "partial_result",
    ):
        if hasattr(error, name):
            identity[name] = getattr(error, name)
    return identity
