"""WhisperKit ANE transcription — runs encoder+decoder on Apple Neural Engine.

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
import time
import urllib.error
import urllib.request
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
_WHISPERKIT_SERVER_HOST = "localhost"
_WHISPERKIT_SERVER_START_TIMEOUT_SECONDS = 20.0
_WHISPERKIT_SERVER_REQUEST_TIMEOUT_SECONDS = 120.0


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


def _wait_for_tcp_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((host, port))
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

    def __init__(self, model: str = DEFAULT_WHISPERKIT_MODEL) -> None:
        self._model = model
        self._cli_path = _find_whisperkit_cli()
        self._server_proc: subprocess.Popen | None = None
        self._external_server_url: str | None = os.environ.get("SPOKE_WHISPERKIT_SERVER_URL") or None
        self._server_url: str | None = self._external_server_url
        self._server_port: int | None = None
        self._server_log_handle = None

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

        if _whisperkit_resident_enabled():
            resident_text = self._transcribe_resident(wav_bytes)
            if resident_text is not None:
                return resident_text
            logger.warning(
                "WhisperKit resident server failed; falling back to CLI subprocess "
                "(model=%s, server_url=%s)",
                self._model,
                self._server_url,
            )

        return self._transcribe_cli(wav_bytes)

    def _transcribe_cli(self, wav_bytes: bytes) -> str:
        """Transcribe WAV bytes through one-shot ``whisperkit-cli transcribe``."""
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

    def _resident_log_path(self) -> Path:
        configured = os.environ.get("SPOKE_WHISPERKIT_SERVER_LOG")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / "Library" / "Logs" / "Spoke" / "whisperkit-server.log"

    def _ensure_resident_server(self) -> tuple[str, str, int | None]:
        cli = self._cli_path or _find_whisperkit_cli()
        if cli is None:
            raise RuntimeError("whisperkit-cli not found")
        if self._external_server_url:
            return self._external_server_url.rstrip("/"), cli, None
        if self._server_proc is not None and self._server_proc.poll() is None and self._server_port:
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
            self._server_proc = None
            self._server_url = None
            self._server_port = None

        port = _find_available_port()
        chunking_strategy = _whisperkit_chunking_strategy()
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
            _WHISPERKIT_ENCODER_COMPUTE_UNITS,
            "--text-decoder-compute-units",
            _WHISPERKIT_DECODER_COMPUTE_UNITS,
            "--chunking-strategy",
            chunking_strategy,
            "--skip-special-tokens",
            "--without-timestamps",
        ]
        log_path = self._resident_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._server_log_handle = log_path.open("ab")
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
        if not _wait_for_tcp_port(_WHISPERKIT_SERVER_HOST, port, start_timeout):
            logger.warning(
                "WhisperKit resident server did not accept TCP before timeout "
                "(pid=%s, model=%s, url=%s, log=%s)",
                self._server_proc.pid,
                self._model,
                self._server_url,
                log_path,
            )
        else:
            logger.info(
                "WhisperKit resident server started (pid=%s, model=%s, url=%s, log=%s)",
                self._server_proc.pid,
                self._model,
                self._server_url,
                log_path,
            )
        return self._server_url, cli, self._server_proc.pid

    @staticmethod
    def _multipart_body(
        *,
        wav_bytes: bytes,
        model: str,
        language: str = "en",
        response_format: str = "json",
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

    def _transcribe_resident(self, wav_bytes: bytes) -> str | None:
        try:
            server_url, cli, server_pid = self._ensure_resident_server()
        except Exception as exc:
            logger.warning("WhisperKit resident server unavailable: %s", exc)
            return None

        total_start = time.perf_counter()
        chunking_strategy = _whisperkit_chunking_strategy()
        duration_s = _wav_duration_seconds(wav_bytes)
        attempts: list[dict[str, object]] = []
        text = ""
        postprocess_ms = 0.0
        final_suspicion_reason: str | None = None
        request_timeout = _float_env(
            "SPOKE_WHISPERKIT_SERVER_REQUEST_TIMEOUT",
            _WHISPERKIT_SERVER_REQUEST_TIMEOUT_SECONDS,
        )

        for attempt_index in range(_WHISPERKIT_SUSPECT_RETRY_COUNT + 1):
            body, boundary = self._multipart_body(wav_bytes=wav_bytes, model=self._model)
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
            final_suspicion_reason = suspicion_reason
            if suspicion_reason is None:
                break
            if attempt_index < _WHISPERKIT_SUSPECT_RETRY_COUNT:
                logger.warning(
                    "WhisperKit resident returned suspiciously short success; retrying "
                    "(model=%s, url=%s, duration=%s, response_bytes=%d, text_len=%d)",
                    self._model,
                    server_url,
                    f"{duration_s:.1f}s" if duration_s is not None else "unknown",
                    len(raw_body),
                    len(attempt_text),
                )

        total_ms = (time.perf_counter() - total_start) * 1000
        if any(attempt["suspicious"] for attempt in attempts):
            status = "suspicious_unrecovered" if final_suspicion_reason is not None else "recovered_by_retry"
            _record_whisperkit_suspect_bundle(
                wav_bytes=wav_bytes,
                status=status,
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
            )

        logger.info(
            "WhisperKit resident transcription (%s): %r "
            "(%d bytes audio; url=%s; pid=%s; chunking=%s; request=%.0fms, "
            "postprocess=%.0fms, total=%.0fms)",
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
        return text

    def unload(self) -> None:
        """Stop the resident server if this client owns one."""
        self.close()

    @property
    def is_loaded(self) -> bool:
        """Whether this client currently has a live resident server."""
        return self._server_proc is not None and self._server_proc.poll() is None

    def close(self) -> None:
        """Terminate the owned resident server process."""
        proc = self._server_proc
        self._server_proc = None
        self._server_url = self._external_server_url
        self._server_port = None
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
