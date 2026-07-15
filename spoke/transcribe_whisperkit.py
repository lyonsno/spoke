"""WhisperKit transcription with explicit CoreML compute routing.

Calls whisperkit-cli (brew install whisperkit-cli). The default path keeps a
resident OpenAI-compatible WhisperKit server warm; one-shot CLI transcription is
an explicit fallback/escape hatch because spawning a fresh model process per
utterance is too slow and brittle under heavy box contention.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from .dedup import truncate_repetition, is_hallucination, repair_ontology_terms
from .transcription_prompt import TranscriptionPrompt, TranscriptionPromptProvider

logger = logging.getLogger(__name__)

# Model ID prefix that routes to this client
WHISPERKIT_PREFIX = "whisperkit/"

# Default WhisperKit model variant
DEFAULT_WHISPERKIT_MODEL = "medium.en"
DEFAULT_WHISPERKIT_CHUNKING_STRATEGY = "none"
_WHISPERKIT_CHUNKING_STRATEGIES = {"none", "vad"}
_WHISPERKIT_ENCODER_COMPUTE_UNITS = "cpuAndNeuralEngine"
_WHISPERKIT_DECODER_COMPUTE_UNITS = "cpuOnly"
_WHISPERKIT_COMPUTE_UNITS = {
    "all",
    "cpuOnly",
    "cpuAndGPU",
    "cpuAndNeuralEngine",
    "random",
}
_WHISPERKIT_TIMEOUT_SECONDS = 30.0
_WHISPERKIT_SUSPECT_MIN_SECONDS = 8.0
_WHISPERKIT_SUSPECT_MIN_CHARS = 80
_WHISPERKIT_SUSPECT_MIN_CHARS_PER_SECOND = 3.0
# Repeating the same deterministic decode added latency without changing route or recovery odds.
_WHISPERKIT_SUSPECT_ATTEMPTS = 1
_WHISPERKIT_SERVER_HOST = "localhost"
_WHISPERKIT_SERVER_START_TIMEOUT_SECONDS = 20.0


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


def _whisperkit_compute_units(env_name: str, default: str) -> str:
    requested = os.environ.get(env_name, default).strip()
    if requested in _WHISPERKIT_COMPUTE_UNITS:
        return requested
    logger.warning("Invalid %s=%r; using %s", env_name, requested, default)
    return default


def _whisperkit_encoder_compute_units() -> str:
    return _whisperkit_compute_units(
        "SPOKE_WHISPERKIT_ENCODER_COMPUTE_UNITS",
        _WHISPERKIT_ENCODER_COMPUTE_UNITS,
    )


def _whisperkit_decoder_compute_units() -> str:
    return _whisperkit_compute_units(
        "SPOKE_WHISPERKIT_DECODER_COMPUTE_UNITS",
        _WHISPERKIT_DECODER_COMPUTE_UNITS,
    )


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default


def _whisperkit_resident_enabled() -> bool:
    return _truthy_env("SPOKE_WHISPERKIT_RESIDENT", default=True)


def _whisperkit_timeout_seconds() -> float:
    return _float_env("SPOKE_WHISPERKIT_TIMEOUT_SECONDS", _WHISPERKIT_TIMEOUT_SECONDS)


def _find_available_port() -> int:
    configured = os.environ.get("SPOKE_WHISPERKIT_SERVER_PORT", "").strip()
    if configured:
        try:
            return int(configured)
        except ValueError:
            logger.warning("Invalid SPOKE_WHISPERKIT_SERVER_PORT=%r; using an ephemeral port", configured)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_WHISPERKIT_SERVER_HOST, 0))
        return int(sock.getsockname()[1])


def _tcp_listener_owner_pids(port: int) -> set[int]:
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    try:
        result = subprocess.run(
            [lsof, "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    owners: set[int] = set()
    for raw_pid in result.stdout.splitlines():
        try:
            owners.add(int(raw_pid.strip()))
        except ValueError:
            continue
    return owners


def _wait_for_tcp_port(
    host: str,
    port: int,
    timeout_s: float,
    *,
    expected_pid: int | None = None,
    process: subprocess.Popen | None = None,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if process is not None and process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((host, port))
                if expected_pid is None or expected_pid in _tcp_listener_owner_pids(port):
                    return True
            except OSError:
                pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


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
    mode: str,
    model: str,
    cli_path: str,
    chunking_strategy: str,
    duration_s: float | None,
    write_ms: float,
    postprocess_ms: float,
    total_ms: float,
    attempts: list[dict[str, object]],
    chosen_output: str,
    server_url: str | None = None,
    server_pid: int | None = None,
    requested_route: str | None = None,
    effective_route: str | None = None,
    fallback_reason: str | None = None,
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
            "mode": mode,
            "effective_cli_path": cli_path,
            "effective_server_url": server_url,
            "server_pid": server_pid,
            "requested_route": requested_route or mode,
            "effective_route": effective_route or mode,
            "fallback_reason": fallback_reason,
            "effective_model": model,
            "effective_chunking_strategy": chunking_strategy,
            "audio_encoder_compute_units": _whisperkit_encoder_compute_units(),
            "text_decoder_compute_units": _whisperkit_decoder_compute_units(),
            "timeout_seconds": _whisperkit_timeout_seconds(),
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
            "WhisperKit suspect bundle preserved at %s (status=%s, mode=%s, model=%s, "
            "chunking=%s, duration=%s, bytes=%d)",
            report_path,
            status,
            mode,
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

    def __init__(
        self,
        model: str = DEFAULT_WHISPERKIT_MODEL,
        *,
        prompt_provider: TranscriptionPromptProvider | None = None,
    ) -> None:
        self._model = model
        self._prompt_provider = (
            prompt_provider or TranscriptionPromptProvider.from_environment()
        )
        self._last_prompt_receipt: dict | None = None
        self._cli_path = _find_whisperkit_cli()
        self._server_proc: subprocess.Popen | None = None
        self._external_server_url: str | None = os.environ.get("SPOKE_WHISPERKIT_SERVER_URL") or None
        self._server_url: str | None = self._external_server_url
        self._server_port: int | None = None
        self._server_log_handle = None
        self._server_ready = self._external_server_url is not None
        self._resident_failure_reason: str | None = None
        self._last_route_report: dict[str, object] = {}
        self._server_lifecycle_lock = threading.RLock()
        self._closed = False

    @staticmethod
    def available() -> bool:
        """Whether whisperkit-cli is installed and reachable."""
        return _find_whisperkit_cli() is not None

    def prepare(self) -> None:
        """Seat the selected resident server before the first transcription."""
        if self._cli_path is None:
            self._cli_path = _find_whisperkit_cli()
        if self._cli_path is None:
            logger.warning(
                "whisperkit-cli not found. Install with: brew install whisperkit-cli"
            )
            return
        if _whisperkit_resident_enabled():
            try:
                server_url, _, server_pid = self._ensure_resident_server()
                self._last_route_report = {
                    "requested_route": "resident-server",
                    "effective_route": "resident-server",
                    "fallback_reason": None,
                    "status": "preloaded",
                    "terminal_error": None,
                    "model": self._model,
                    "server_url": server_url,
                    "server_pid": server_pid,
                    "effective_chunking_strategy": _whisperkit_chunking_strategy(),
                    "audio_encoder_compute_units": _whisperkit_encoder_compute_units(),
                    "text_decoder_compute_units": _whisperkit_decoder_compute_units(),
                    "timeout_seconds": _whisperkit_timeout_seconds(),
                }
            except Exception as exc:
                reason = f"preload:{type(exc).__name__}:{exc}"
                self._resident_failure_reason = reason
                self._last_route_report = {
                    "requested_route": "resident-server",
                    "effective_route": None,
                    "fallback_reason": reason,
                    "status": "preload_failed",
                    "model": self._model,
                }
                logger.warning(
                    "WhisperKit resident preload failed; app warmup will continue "
                    "with CLI fallback available (model=%s, fallback_reason=%s)",
                    self._model,
                    reason,
                )

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text."""
        if not wav_bytes:
            return ""

        prompt = self._prompt_provider.resolve()
        prompt_effective = bool(prompt.text)
        self._last_prompt_receipt = prompt.receipt(
            supported=True,
            effective=prompt_effective,
        )
        logger.info(
            "WhisperKit transcription prompt: requested=%s supported=true effective=%s "
            "sha256=%s chars=%d sources=%s",
            self._last_prompt_receipt["requested"],
            prompt_effective,
            prompt.sha256,
            prompt.char_count,
            ",".join(prompt.sources) or "none",
        )

        if _whisperkit_resident_enabled():
            resident_text = self._transcribe_resident(wav_bytes, prompt)
            if resident_text is not None:
                return resident_text
            fallback_reason = self._resident_failure_reason or "unknown resident failure"
            logger.warning(
                "WhisperKit resident server failed; falling back to CLI subprocess "
                "(requested_route=resident-server, effective_route=cli-subprocess, "
                "model=%s, server_url=%s, fallback_reason=%s)",
                self._model,
                self._server_url,
                fallback_reason,
            )
            return self._transcribe_cli(
                wav_bytes,
                prompt,
                fallback_from="resident-server",
                fallback_reason=fallback_reason,
            )

        return self._transcribe_cli(wav_bytes, prompt)

    def _transcribe_cli(
        self,
        wav_bytes: bytes,
        prompt: TranscriptionPrompt,
        *,
        fallback_from: str | None = None,
        fallback_reason: str | None = None,
    ) -> str:
        """Transcribe WAV bytes through one-shot ``whisperkit-cli transcribe``."""
        requested_route = fallback_from or "cli-subprocess"
        encoder_compute_units = _whisperkit_encoder_compute_units()
        decoder_compute_units = _whisperkit_decoder_compute_units()
        chunking_strategy = _whisperkit_chunking_strategy()
        timeout_seconds = _whisperkit_timeout_seconds()
        self._last_route_report = {
            "requested_route": requested_route,
            "effective_route": "cli-subprocess",
            "fallback_reason": fallback_reason,
            "status": "attempting",
            "terminal_error": None,
            "model": self._model,
            "audio_encoder_compute_units": encoder_compute_units,
            "text_decoder_compute_units": decoder_compute_units,
            "effective_chunking_strategy": chunking_strategy,
            "timeout_seconds": timeout_seconds,
            "prompt": dict(self._last_prompt_receipt or {}),
        }
        cli = self._cli_path or _find_whisperkit_cli()
        if cli is None:
            self._last_route_report.update(
                effective_route=None,
                status="failed",
                terminal_error="cli_missing",
            )
            logger.error(
                "whisperkit-cli not found; cannot transcribe "
                "(requested_route=%s, effective_route=None, fallback_reason=%s)",
                requested_route,
                fallback_reason,
            )
            return ""

        total_start = time.perf_counter()
        write_start = total_start
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        write_ms = (time.perf_counter() - write_start) * 1000

        try:
            cmd = [
                cli, "transcribe",
                "--audio-path", tmp_path,
                "--model", self._model,
                "--language", "en",
                "--audio-encoder-compute-units", encoder_compute_units,
                "--text-decoder-compute-units", decoder_compute_units,
                "--chunking-strategy", chunking_strategy,
                "--skip-special-tokens",
                "--without-timestamps",
            ]
            if prompt.text:
                cmd.extend(["--prompt", prompt.text])
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
            for attempt_index in range(_WHISPERKIT_SUSPECT_ATTEMPTS):
                subprocess_start = time.perf_counter()
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                    )
                except Exception as exc:
                    self._last_route_report.update(
                        status="failed",
                        terminal_error=f"cli_exception:{type(exc).__name__}:{exc}",
                    )
                    raise
                subprocess_ms = (time.perf_counter() - subprocess_start) * 1000
                raw_stdout = result.stdout or ""
                raw_stderr = result.stderr or ""
                raw_text = raw_stdout.strip()
                if result.returncode != 0:
                    terminal_error = (
                        f"cli_exit:{result.returncode}:{raw_stderr.strip()}"
                    )
                    self._last_route_report.update(
                        status="failed",
                        terminal_error=terminal_error,
                    )
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
                if suspicion_reason is None:
                    break
            total_ms = (time.perf_counter() - total_start) * 1000
            if any(attempt["suspicious"] for attempt in attempts):
                _record_whisperkit_suspect_bundle(
                    wav_bytes=wav_bytes,
                    status="suspicious_unrecovered",
                    mode="cli-subprocess",
                    model=self._model,
                    cli_path=cli,
                    chunking_strategy=chunking_strategy,
                    duration_s=duration_s,
                    write_ms=write_ms,
                    postprocess_ms=postprocess_ms,
                    total_ms=total_ms,
                    attempts=attempts,
                    chosen_output=text,
                    requested_route=requested_route,
                    effective_route="cli-subprocess",
                    fallback_reason=fallback_reason,
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        logger.info(
            "WhisperKit ANE transcription (%s): %r "
            "(%d bytes audio; chunking=%s; write=%.0fms, subprocess=%.0fms, "
            "postprocess=%.0fms, total=%.0fms; requested_route=%s, "
            "effective_route=cli-subprocess, fallback_reason=%s)",
            self._model,
            text,
            len(wav_bytes),
            chunking_strategy,
            write_ms,
            sum(float(attempt["subprocess_ms"]) for attempt in attempts),
            postprocess_ms,
            total_ms,
            fallback_from or "cli-subprocess",
            fallback_reason,
        )
        self._last_route_report.update(
            status=(
                "suspicious_success"
                if any(attempt["suspicious"] for attempt in attempts)
                else "succeeded"
            ),
            terminal_error=None,
        )
        return text

    def _resident_log_path(self) -> Path:
        configured = os.environ.get("SPOKE_WHISPERKIT_SERVER_LOG")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / "Library" / "Logs" / "Spoke" / "whisperkit-server.log"

    def _ensure_resident_server(self) -> tuple[str, str, int | None]:
        with self._server_lifecycle_lock:
            if self._closed:
                raise RuntimeError("WhisperKit client is closed")
            return self._ensure_resident_server_locked()

    def _ensure_resident_server_locked(self) -> tuple[str, str, int | None]:
        cli = self._cli_path or _find_whisperkit_cli()
        if cli is None:
            raise RuntimeError("whisperkit-cli not found")
        if self._external_server_url:
            return self._external_server_url.rstrip("/"), cli, None
        if (
            self._server_proc is not None
            and self._server_proc.poll() is None
            and self._server_port
            and self._server_ready
        ):
            return f"http://{_WHISPERKIT_SERVER_HOST}:{self._server_port}", cli, self._server_proc.pid
        if self._server_proc is not None and self._server_proc.poll() is not None:
            logger.warning(
                "WhisperKit resident server exited; restarting "
                "(old_pid=%s, returncode=%s, model=%s, url=%s)",
                self._server_proc.pid,
                self._server_proc.poll(),
                self._model,
                self._server_url,
            )
            self._close_owned_server()

        port = _find_available_port()
        chunking_strategy = _whisperkit_chunking_strategy()
        encoder_compute_units = _whisperkit_encoder_compute_units()
        decoder_compute_units = _whisperkit_decoder_compute_units()
        cmd = [
            cli,
            "serve",
            "--host",
            _WHISPERKIT_SERVER_HOST,
            "--port",
            str(port),
            "--model",
            self._model,
            "--language",
            "en",
            "--audio-encoder-compute-units",
            encoder_compute_units,
            "--text-decoder-compute-units",
            decoder_compute_units,
            "--chunking-strategy",
            chunking_strategy,
            "--skip-special-tokens",
            "--without-timestamps",
        ]
        log_path = self._resident_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_log_handle = log_path.open("ab")
        start_time = time.perf_counter()
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
        )
        self._server_port = port
        self._server_url = f"http://{_WHISPERKIT_SERVER_HOST}:{port}"
        start_timeout = _float_env(
            "SPOKE_WHISPERKIT_SERVER_START_TIMEOUT",
            _WHISPERKIT_SERVER_START_TIMEOUT_SECONDS,
        )
        attempted_url = self._server_url
        if not _wait_for_tcp_port(
            _WHISPERKIT_SERVER_HOST,
            port,
            start_timeout,
            expected_pid=self._server_proc.pid,
            process=self._server_proc,
        ):
            self._close_owned_server()
            raise TimeoutError(
                "WhisperKit resident server did not accept its listener before "
                f"timeout (model={self._model}, url={attempted_url}, log={log_path})"
            )
        returncode = self._server_proc.poll()
        if returncode is not None:
            failed_pid = self._server_proc.pid
            self._close_owned_server()
            raise RuntimeError(
                "WhisperKit resident server exited before owning listener "
                f"(pid={failed_pid}, returncode={returncode}, model={self._model}, "
                f"url={attempted_url}, log={log_path})"
            )
        self._server_ready = True
        startup_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "WhisperKit resident server started "
            "(pid=%s, model=%s, url=%s, log=%s, startup=%.0fms, "
            "audio_encoder_compute_units=%s, text_decoder_compute_units=%s)",
            self._server_proc.pid,
            self._model,
            self._server_url,
            log_path,
            startup_ms,
            encoder_compute_units,
            decoder_compute_units,
        )
        return self._server_url, cli, self._server_proc.pid

    @staticmethod
    def _multipart_body(
        *,
        wav_bytes: bytes,
        model: str,
        language: str = "en",
        response_format: str = "json",
        prompt: str = "",
    ) -> tuple[bytes, str]:
        boundary = f"spoke-whisperkit-{hashlib.sha256(wav_bytes).hexdigest()[:16]}"
        parts: list[bytes] = []

        def add_field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\n".encode("utf-8"))
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            parts.append(value.encode("utf-8"))
            parts.append(b"\r\n")

        add_field("model", model)
        add_field("language", language)
        add_field("response_format", response_format)
        add_field("temperature", "0")
        if prompt:
            add_field("prompt", prompt)
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            b'Content-Disposition: form-data; name="file"; filename="spoke-dictation.wav"\r\n'
        )
        parts.append(b"Content-Type: audio/wav\r\n\r\n")
        parts.append(wav_bytes)
        parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(parts), boundary

    @staticmethod
    def _text_from_server_payload(payload: dict[str, object]) -> str:
        text = payload.get("text")
        if isinstance(text, str):
            return text
        segments = payload.get("segments")
        if isinstance(segments, list):
            pieces = [
                segment.get("text", "")
                for segment in segments
                if isinstance(segment, dict) and isinstance(segment.get("text"), str)
            ]
            return " ".join(piece.strip() for piece in pieces if piece.strip())
        return ""

    def _transcribe_resident(
        self,
        wav_bytes: bytes,
        prompt: TranscriptionPrompt,
    ) -> str | None:
        self._resident_failure_reason = None
        try:
            server_url, cli, server_pid = self._ensure_resident_server()
        except Exception as exc:
            logger.warning("WhisperKit resident server unavailable: %s", exc)
            self._resident_failure_reason = f"server_start:{type(exc).__name__}:{exc}"
            return None

        total_start = time.perf_counter()
        chunking_strategy = _whisperkit_chunking_strategy()
        duration_s = _wav_duration_seconds(wav_bytes)
        attempts: list[dict[str, object]] = []
        text = ""
        postprocess_ms = 0.0
        request_timeout = _float_env(
            "SPOKE_WHISPERKIT_SERVER_REQUEST_TIMEOUT",
            _whisperkit_timeout_seconds(),
        )

        for attempt_index in range(_WHISPERKIT_SUSPECT_ATTEMPTS):
            body, boundary = self._multipart_body(
                wav_bytes=wav_bytes,
                model=self._model,
                prompt=prompt.text,
            )
            request = urllib.request.Request(
                f"{server_url.rstrip('/')}/v1/audio/transcriptions",
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json",
                },
                method="POST",
            )
            request_start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    raw_body = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                logger.warning(
                    "WhisperKit resident request failed (model=%s, url=%s, attempt=%d): %s",
                    self._model,
                    server_url,
                    attempt_index + 1,
                    exc,
                )
                self._resident_failure_reason = f"request:{type(exc).__name__}:{exc}"
                return None
            request_ms = (time.perf_counter() - request_start) * 1000
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "WhisperKit resident response was not JSON "
                    "(model=%s, url=%s, bytes=%d): %s",
                    self._model,
                    server_url,
                    len(raw_body),
                    exc,
                )
                self._resident_failure_reason = f"response_decode:{type(exc).__name__}:{exc}"
                return None

            postprocess_start = time.perf_counter()
            attempt_text = self._text_from_server_payload(payload)
            attempt_text = truncate_repetition(attempt_text)
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
                    "mode": "resident-server",
                    "returncode": 0,
                    "response_bytes": len(raw_body),
                    "request_ms": request_ms,
                    "postprocess_ms": attempt_postprocess_ms,
                    "suspicious": suspicion_reason is not None,
                    "suspicion_reason": suspicion_reason,
                }
            )
            text = attempt_text
            if suspicion_reason is None:
                break

        total_ms = (time.perf_counter() - total_start) * 1000
        if any(attempt["suspicious"] for attempt in attempts):
            _record_whisperkit_suspect_bundle(
                wav_bytes=wav_bytes,
                status="suspicious_unrecovered",
                mode="resident-server",
                model=self._model,
                cli_path=cli,
                chunking_strategy=chunking_strategy,
                duration_s=duration_s,
                write_ms=0.0,
                postprocess_ms=postprocess_ms,
                total_ms=total_ms,
                attempts=attempts,
                chosen_output=text,
                server_url=server_url,
                server_pid=server_pid,
                requested_route="resident-server",
                effective_route="resident-server",
            )

        logger.info(
            "WhisperKit resident transcription (%s): %r "
            "(%d bytes audio; url=%s; pid=%s; chunking=%s; request=%.0fms, "
            "postprocess=%.0fms, total=%.0fms; requested_route=resident-server, "
            "effective_route=resident-server)",
            self._model,
            text,
            len(wav_bytes),
            server_url,
            server_pid,
            chunking_strategy,
            sum(float(attempt["request_ms"]) for attempt in attempts),
            postprocess_ms,
            total_ms,
        )
        self._last_route_report = {
            "requested_route": "resident-server",
            "effective_route": "resident-server",
            "fallback_reason": None,
            "status": (
                "suspicious_success"
                if any(attempt["suspicious"] for attempt in attempts)
                else "succeeded"
            ),
            "terminal_error": None,
            "model": self._model,
            "server_url": server_url,
            "server_pid": server_pid,
            "effective_chunking_strategy": chunking_strategy,
            "audio_encoder_compute_units": _whisperkit_encoder_compute_units(),
            "text_decoder_compute_units": _whisperkit_decoder_compute_units(),
            "timeout_seconds": request_timeout,
            "prompt": dict(self._last_prompt_receipt or {}),
        }
        return text

    def unload(self) -> None:
        """Stop the resident server while allowing a later lazy restart."""
        self._close_owned_server()

    @property
    def is_loaded(self) -> bool:
        """Whether this client currently has a live resident server."""
        return (
            self._server_ready
            and self._server_proc is not None
            and self._server_proc.poll() is None
        )

    @property
    def last_route_report(self) -> dict[str, object]:
        """Return the effective route identity for the latest transcription."""
        return dict(self._last_route_report)

    @property
    def last_prompt_receipt(self) -> dict | None:
        """Return source identity and effective use for the latest prompt."""
        if self._last_prompt_receipt is None:
            return None
        return dict(self._last_prompt_receipt)

    def _close_owned_server(self) -> None:
        with self._server_lifecycle_lock:
            self._close_owned_server_locked()

    def _close_owned_server_locked(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        self._server_url = self._external_server_url
        self._server_port = None
        self._server_ready = self._external_server_url is not None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._server_log_handle is not None:
            try:
                self._server_log_handle.close()
            except OSError:
                pass
            self._server_log_handle = None

    def close(self) -> None:
        """Terminate the owned resident server process."""
        with self._server_lifecycle_lock:
            self._closed = True
            self._close_owned_server_locked()
