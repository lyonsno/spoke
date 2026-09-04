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

from .dedup import repair_ontology_terms
from .transcription_prompt import TranscriptionPromptProvider

logger = logging.getLogger(__name__)

_NEMOTRON_CPU_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b@cpu"
_NEMOTRON_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
_NEMOTRON_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
_NEMOTRON_FILENAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
_NEMOTRON_EXPECTED_SHA256 = (
    "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae"
)
_DEFAULT_BINARY = (
    Path.home()
    / ".local"
    / "share"
    / "spoke"
    / "nemo-speech-cpu"
    / "bin"
    / "nemo-speech"
)
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
_TIMING_NUMBER = r"(?:0|[1-9]\d*)\.\d{2}"
_PHASE_TIMING_PATTERNS = (
    (
        "feature_extraction",
        re.compile(
            rf"^\[timing\] fe path=cpu n_samples=\d+ n_frames=\d+ "
            rf"= {_TIMING_NUMBER} ms$"
        ),
    ),
    (
        "cache_chunk",
        re.compile(
            rf"^\[timing\] cache-chunk enc_frames=\d+ encode={_TIMING_NUMBER} "
            rf"decode={_TIMING_NUMBER} ms$"
        ),
    ),
    (
        "postprocess_dispatch",
        re.compile(
            rf"^\[timing\] postproc-dispatch queue={_TIMING_NUMBER} "
            rf"pnc={_TIMING_NUMBER} total={_TIMING_NUMBER} ms$"
        ),
    ),
    (
        "postprocess_cpu",
        re.compile(
            rf"^\[timing\] postproc-cpu chars=\d+ profanity={_TIMING_NUMBER} "
            rf"itn={_TIMING_NUMBER} ms$"
        ),
    ),
    (
        "offline_decoder",
        re.compile(
            rf"^\[timing\] offline-transducer decode frames=\d+ segments=\d+ "
            rf"= {_TIMING_NUMBER} ms$"
        ),
    ),
    (
        "offline_encoder",
        re.compile(
            rf"^\[timing\] offline-transducer frames=\d+ enc_frames=\d+ "
            rf"fe={_TIMING_NUMBER} encoder\+encproj={_TIMING_NUMBER} ms$"
        ),
    ),
)
_SESSION_TIMING_PATTERNS = (
    re.compile(
        rf"^\[timing\] session out=.+ +nodes=\d+ in={_TIMING_NUMBER} "
        rf"direct compute={_TIMING_NUMBER} out={_TIMING_NUMBER} "
        rf"total={_TIMING_NUMBER} ms$"
    ),
    re.compile(
        rf"^\[timing\] session out=.+ +nodes=\d+ in={_TIMING_NUMBER} "
        rf"alloc={_TIMING_NUMBER} compute={_TIMING_NUMBER} "
        rf"out={_TIMING_NUMBER} total={_TIMING_NUMBER} ms$"
    ),
)
_CACHE_TIMING_FAMILIES = {
    "feature_extraction",
    "cache_chunk",
    "postprocess_cpu",
    "postprocess_dispatch",
}
_OFFLINE_TIMING_FAMILIES = {
    "feature_extraction",
    "offline_encoder",
    "offline_decoder",
    "postprocess_cpu",
    "postprocess_dispatch",
}


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
    return parsed if parsed > 0 else None


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


def _phase_timing_enabled() -> bool:
    value = os.environ.get("SPOKE_NEMOTRON_PHASE_TIMING", "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise NemotronCPUError(
        "SPOKE_NEMOTRON_PHASE_TIMING must be a boolean value, "
        f"got {value!r}"
    )


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


def _subprocess_output_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"subprocess output has unsupported type {type(value).__name__}")


def _decode_subprocess_output(value: bytes | str | None) -> str:
    if isinstance(value, str):
        return value
    return _subprocess_output_bytes(value).decode("utf-8", errors="strict")


def _opaque_output_evidence(
    *, stdout: bytes | str | None, stderr: bytes | str | None
) -> dict[str, int | str]:
    evidence: dict[str, int | str] = {}
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        raw = _subprocess_output_bytes(value)
        evidence[f"{name}_bytes"] = len(raw)
        evidence[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
    return evidence


def _initial_phase_timing_state(
    *, requested: bool, enabled: bool, configuration_valid: bool
) -> dict:
    return {
        "requested": requested,
        "enabled": enabled,
        "configuration_valid": configuration_valid,
        "collection_status": "not_reached" if enabled else "disabled",
        "effective_runner": "unobserved",
        "emitted_lines": 0,
        "observed_families": [],
        "family_counts": {},
        "missing_families": [],
        "contradictory_families": [],
        "rejected_lines": 0,
        "suppressed_detail_lines": 0,
    }


def _parse_phase_timing(
    value: bytes | str | None, *, enabled: bool
) -> tuple[dict, list[str]]:
    if not enabled:
        return (
            _initial_phase_timing_state(
                requested=False, enabled=False, configuration_valid=True
            ),
            [],
        )

    raw = _subprocess_output_bytes(value)
    lines = raw.decode("utf-8", errors="replace").splitlines()
    admitted_lines: list[str] = []
    family_counts: dict[str, int] = {}
    rejected_lines = 0
    suppressed_detail_lines = 0
    timing_prefixed_lines = 0
    for line in lines:
        if not line.startswith("[timing]"):
            continue
        timing_prefixed_lines += 1
        matched_family = next(
            (
                family
                for family, pattern in _PHASE_TIMING_PATTERNS
                if pattern.fullmatch(line)
            ),
            None,
        )
        if matched_family is not None:
            admitted_lines.append(line)
            family_counts[matched_family] = family_counts.get(matched_family, 0) + 1
        elif any(pattern.fullmatch(line) for pattern in _SESSION_TIMING_PATTERNS):
            suppressed_detail_lines += 1
        else:
            rejected_lines += 1

    observed = set(family_counts)
    cache_seen = "cache_chunk" in observed
    offline_seen = "offline_encoder" in observed or "offline_decoder" in observed
    contradictory: list[str] = []
    if cache_seen and offline_seen:
        effective_runner = "contradictory"
        contradictory = ["cache_stream", "offline"]
        expected: set[str] = set()
    elif cache_seen:
        effective_runner = "cache_stream"
        expected = _CACHE_TIMING_FAMILIES
    elif offline_seen:
        effective_runner = "offline"
        expected = _OFFLINE_TIMING_FAMILIES
    else:
        effective_runner = "unobserved"
        expected = set()

    missing = (
        sorted(expected - observed)
        if effective_runner not in {"unobserved", "contradictory"}
        else (["effective_runner"] if effective_runner == "unobserved" else [])
    )
    if contradictory:
        collection_status = "contradictory"
    elif effective_runner == "unobserved" or missing or rejected_lines:
        collection_status = "partial" if timing_prefixed_lines else "absent"
    else:
        collection_status = "complete"

    return (
        {
            "requested": True,
            "enabled": True,
            "configuration_valid": True,
            "collection_status": collection_status,
            "effective_runner": effective_runner,
            "emitted_lines": len(admitted_lines),
            "observed_families": sorted(observed),
            "family_counts": dict(sorted(family_counts.items())),
            "missing_families": missing,
            "contradictory_families": contradictory,
            "rejected_lines": rejected_lines,
            "suppressed_detail_lines": suppressed_detail_lines,
        },
        admitted_lines,
    )


def _controlled_child_environment(
    *, phase_timing: bool = False
) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(
        key
        for key in environment
        if key == "NEMO_SPEECH" or key.startswith("NEMO_SPEECH_")
    )
    for key in removed:
        environment.pop(key, None)
    if phase_timing:
        environment["NEMO_SPEECH_TIMING"] = "1"
    return environment, removed


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
    def availability_error(
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
    ) -> str | None:
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
        missing: list[str] = []
        if not resolved_binary.is_file() or not os.access(resolved_binary, os.X_OK):
            missing.append(f"CPU nemo-speech executable {resolved_binary}")
        if not resolved_model.is_file():
            missing.append(f"pinned Nemotron GGUF {resolved_model}")
        return "; ".join(missing) if missing else None

    @staticmethod
    def available(
        *,
        binary: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        return NemotronCPUClient.availability_error(
            binary=binary,
            model_path=model_path,
        ) is None

    def prepare(self) -> None:
        availability_error = self.availability_error(
            binary=self._binary,
            model_path=self._model_path,
        )
        if availability_error is not None:
            raise NemotronCPUError(
                "Nemotron CPU route is not seated: " + availability_error
            )
        model_stat = self._model_path.stat()
        stat_identity = (
            model_stat.st_dev,
            model_stat.st_ino,
            model_stat.st_size,
            model_stat.st_mtime_ns,
        )
        if stat_identity != self._verified_model_stat:
            actual_sha256 = _sha256_file(self._model_path)
            self._model_sha256_actual = actual_sha256
            if actual_sha256 != self._expected_model_sha256:
                raise NemotronCPUError(
                    "Nemotron GGUF SHA-256 mismatch: "
                    f"expected {self._expected_model_sha256}, got {actual_sha256} "
                    f"for {self._model_path}"
                )
            self._verified_model_stat = stat_identity

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        self._last_receipt = None
        started = time.monotonic()
        route = self._initial_route(wav_bytes)
        timing_value = os.environ.get("SPOKE_NEMOTRON_PHASE_TIMING", "")
        timing_requested = bool(timing_value.strip())
        route["phase_timing"] = _initial_phase_timing_state(
            requested=timing_requested,
            enabled=False,
            configuration_valid=True,
        )
        try:
            phase_timing_enabled = _phase_timing_enabled()
        except NemotronCPUError:
            route["phase_timing"] = _initial_phase_timing_state(
                requested=timing_requested,
                enabled=False,
                configuration_valid=False,
            )
            route["phase_timing"]["collection_status"] = "invalid_configuration"
            self._raise_failure(
                route,
                phase="resolve_phase_timing",
                detail="invalid boolean value",
                operator_message="Nemotron CPU timing configuration is invalid",
                started=started,
            )
        route["phase_timing"] = _initial_phase_timing_state(
            requested=timing_requested,
            enabled=phase_timing_enabled,
            configuration_valid=True,
        )

        try:
            self.prepare()
        except Exception as exc:
            route["model_sha256_actual"] = self._model_sha256_actual
            self._raise_failure(
                route,
                phase="prepare_runtime",
                detail=f"{type(exc).__name__}: {exc}",
                operator_message=f"Nemotron CPU preparation failed: {exc}",
                started=started,
            )
        route["model_sha256_actual"] = self._model_sha256_actual

        try:
            prompt = self._prompt_provider.resolve()
            phrases = _prompt_phrases(prompt.text)
        except Exception as exc:
            self._raise_failure(
                route,
                phase="resolve_prompt",
                detail=type(exc).__name__,
                operator_message="Nemotron CPU prompt resolution failed",
                started=started,
            )

        prompt_receipt = prompt.receipt(
            supported=True,
            payload_constructed=bool(phrases),
            submission_attempted=False,
            runtime_accepted=None,
        )
        route["prompt"] = prompt_receipt
        route["speech_context_count"] = len(phrases)
        route["speech_context_boost"] = (
            self._speech_context_boost if phrases else None
        )
        child_environment, removed_environment_keys = _controlled_child_environment(
            phase_timing=phase_timing_enabled
        )
        recognizer_configuration = {
            "authority": "explicit_cli_with_nemo_environment_cleared",
            "runner_selection": "automatic",
            "streaming_cli_requested": False,
            "endpointing": False,
            "vad": False,
        }
        route["recognizer_configuration"] = recognizer_configuration
        route["cleared_nemo_environment_keys"] = removed_environment_keys
        route.update(
            {
                "endpointing": recognizer_configuration["endpointing"],
                "vad": recognizer_configuration["vad"],
            }
        )

        try:
            with tempfile.TemporaryDirectory(prefix="spoke-nemotron-cpu-") as td:
                wav_path = Path(td) / "input.wav"
                try:
                    wav_path.write_bytes(wav_bytes)
                except Exception as exc:
                    self._raise_failure(
                        route,
                        phase="write_temporary_input",
                        detail=f"{type(exc).__name__}: {exc}",
                        operator_message="Nemotron CPU temporary input write failed",
                        started=started,
                    )
                cmd = self._command(wav_path, phrases)
                if phrases:
                    prompt_receipt["submission_attempted"] = True
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=child_environment,
                        timeout=self._timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    self._collect_phase_timing(
                        route,
                        stderr=exc.stderr,
                        enabled=phase_timing_enabled,
                    )
                    self._raise_failure(
                        route,
                        phase="transcribe_timeout",
                        detail="nemo-speech exceeded the explicit timeout",
                        operator_message="Nemotron CPU transcription timed out",
                        started=started,
                        evidence=_opaque_output_evidence(
                            stdout=exc.output,
                            stderr=exc.stderr,
                        ),
                    )
                except Exception as exc:
                    self._raise_failure(
                        route,
                        phase="process_launch",
                        detail=f"{type(exc).__name__}: {exc}",
                        operator_message="Nemotron CPU process launch failed",
                        started=started,
                    )

                wall_seconds = time.monotonic() - started
                phase_timing_lines = self._collect_phase_timing(
                    route,
                    stderr=result.stderr,
                    enabled=phase_timing_enabled,
                )
                if result.returncode != 0:
                    self._raise_failure(
                        route,
                        phase="transcribe_process",
                        detail=f"nemo-speech exited with status {result.returncode}",
                        operator_message=(
                            f"Nemotron CPU transcription exited {result.returncode}"
                        ),
                        started=started,
                        exit_code=result.returncode,
                        evidence=_opaque_output_evidence(
                            stdout=result.stdout,
                            stderr=result.stderr,
                        ),
                    )
                try:
                    stdout_text = _decode_subprocess_output(result.stdout)
                except Exception as exc:
                    self._raise_failure(
                        route,
                        phase="decode_output",
                        detail=type(exc).__name__,
                        operator_message="Nemotron CPU output decoding failed",
                        started=started,
                        evidence=_opaque_output_evidence(
                            stdout=result.stdout,
                            stderr=result.stderr,
                        ),
                    )
                try:
                    payload = json.loads(stdout_text)
                    text = payload["text"]
                    if not isinstance(text, str):
                        raise TypeError("text is not a string")
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    self._raise_failure(
                        route,
                        phase="parse_output",
                        detail=f"{type(exc).__name__}: {exc}",
                        operator_message="Nemotron CPU returned invalid JSON output",
                        started=started,
                        evidence=_opaque_output_evidence(
                            stdout=result.stdout,
                            stderr=result.stderr,
                        ),
                    )
        except NemotronCPUError:
            raise
        except Exception as exc:
            self._raise_failure(
                route,
                phase="temporary_input",
                detail=f"{type(exc).__name__}: {exc}",
                operator_message="Nemotron CPU temporary input handling failed",
                started=started,
            )

        text = text.strip()
        if not text:
            self._raise_failure(
                route,
                phase="validate_output",
                detail="nemo-speech returned a blank transcript",
                operator_message="Nemotron CPU returned a blank transcript",
                started=started,
            )
        text = repair_ontology_terms(text)
        if phrases:
            prompt_receipt["runtime_accepted"] = True
        self._last_receipt = {
            **route,
            "status": "success",
            "wall_seconds": wall_seconds,
            "reported_duration_seconds": payload.get("duration"),
        }
        logger.info(
            "Nemotron CPU transcription: wall=%.3fs audio=%s bytes=%d prompt=%s "
            "phase_timing=%s/%d",
            wall_seconds,
            route["audio_duration_seconds"],
            len(wav_bytes),
            prompt.sha256,
            phase_timing_enabled,
            len(phase_timing_lines),
        )
        return text

    def _collect_phase_timing(
        self,
        route: dict,
        *,
        stderr: bytes | str | None,
        enabled: bool,
    ) -> list[str]:
        timing, admitted_lines = _parse_phase_timing(stderr, enabled=enabled)
        timing["requested"] = route["phase_timing"]["requested"]
        route["phase_timing"] = timing
        for timing_line in admitted_lines:
            logger.info(
                "Nemotron CPU phase: audio=%s %s",
                route["audio_sha256"][:12],
                timing_line,
            )
        if enabled and timing["collection_status"] != "complete":
            logger.warning(
                "Nemotron CPU phase timing degraded: audio=%s status=%s "
                "runner=%s observed=%s missing=%s contradictory=%s rejected=%d "
                "suppressed=%d stderr_bytes=%d",
                route["audio_sha256"][:12],
                timing["collection_status"],
                timing["effective_runner"],
                ",".join(timing["observed_families"]) or "none",
                ",".join(timing["missing_families"]) or "none",
                ",".join(timing["contradictory_families"]) or "none",
                timing["rejected_lines"],
                timing["suppressed_detail_lines"],
                len(_subprocess_output_bytes(stderr)),
            )
        return admitted_lines

    def _initial_route(self, wav_bytes: bytes) -> dict:
        return {
            "schema": "spoke.nemotron-cpu-transcription.v2",
            "requested_model": _NEMOTRON_CPU_MODEL_ID,
            "effective_model_path": str(self._model_path),
            "model_repo": _NEMOTRON_REPO,
            "model_revision": _NEMOTRON_REVISION,
            "model_sha256_expected": self._expected_model_sha256,
            "model_sha256_actual": None,
            "effective_binary": str(self._binary),
            "effective_device": "cpu",
            "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "audio_bytes": len(wav_bytes),
            "audio_duration_seconds": _wav_duration_seconds(wav_bytes),
            "prompt": None,
            "speech_context_count": 0,
            "speech_context_boost": None,
        }

    def _command(self, wav_path: Path, phrases: list[str]) -> list[str]:
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
            cmd.extend(("--speech-context-boost", str(self._speech_context_boost)))
        return cmd

    def _raise_failure(
        self,
        route: dict,
        *,
        phase: str,
        detail: str,
        operator_message: str,
        started: float,
        exit_code: int | None = None,
        evidence: dict | None = None,
    ) -> None:
        wall_seconds = time.monotonic() - started
        try:
            report = self._write_failure(
                route,
                phase=phase,
                detail=detail,
                wall_seconds=wall_seconds,
                exit_code=exit_code,
                evidence=evidence,
            )
        except Exception as report_exc:
            self._last_receipt = {
                **route,
                "status": "failure",
                "failure_phase": phase,
                "detail": detail,
                "exit_code": exit_code,
                "wall_seconds": wall_seconds,
                "evidence": evidence or {},
                "report_publication": {
                    "status": "failure",
                    "error_type": type(report_exc).__name__,
                    "detail": str(report_exc),
                },
            }
            raise NemotronCPUError(
                f"{operator_message}; failure report publication failed "
                f"({type(report_exc).__name__}: {report_exc})"
            ) from None
        raise NemotronCPUError(f"{operator_message}; report={report}") from None

    def _write_failure(
        self,
        route: dict,
        *,
        phase: str,
        detail: str,
        wall_seconds: float,
        exit_code: int | None = None,
        evidence: dict | None = None,
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
            "evidence": evidence or {},
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
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


def _receipt_proves_cpu_full_buffer(receipt: dict, audio_sha256: str) -> bool:
    configuration = receipt.get("recognizer_configuration")
    return bool(
        receipt.get("status") == "success"
        and receipt.get("requested_model") == _NEMOTRON_CPU_MODEL_ID
        and receipt.get("effective_binary")
        and receipt.get("effective_model_path")
        and receipt.get("effective_device") == "cpu"
        and receipt.get("model_sha256_actual")
        and receipt.get("model_sha256_actual") == receipt.get("model_sha256_expected")
        and receipt.get("audio_sha256") == audio_sha256
        and receipt.get("endpointing") is False
        and receipt.get("vad") is False
        and isinstance(configuration, dict)
        and configuration.get("authority")
        == "explicit_cli_with_nemo_environment_cleared"
        and configuration.get("runner_selection") == "automatic"
        and configuration.get("streaming_cli_requested") is False
        and configuration.get("endpointing") is False
        and configuration.get("vad") is False
    )


def run_replay(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    client: NemotronCPUClient | None = None,
) -> dict:
    """Replay one retained WAV and preserve success or failure evidence."""
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    try:
        wav_bytes = source.read_bytes()
    except Exception as exc:
        _write_replay_report(
            destination,
            {
                "schema": "spoke.nemotron-cpu-replay.v2",
                "status": "failure",
                "failure_phase": "read_input",
                "input_path": str(source),
                "error_type": type(exc).__name__,
            },
        )
        raise NemotronCPUError(
            f"Nemotron replay input is unavailable; report={destination}"
        ) from None
    expected_audio_sha256 = hashlib.sha256(wav_bytes).hexdigest()
    active_client = client or NemotronCPUClient()
    started = time.monotonic()
    try:
        transcript = active_client.transcribe(wav_bytes)
        receipt = active_client._last_receipt
        if not isinstance(receipt, dict):
            raise NemotronCPUError("Replay route identity is missing")
        if not _receipt_proves_cpu_full_buffer(receipt, expected_audio_sha256):
            report = {
                "schema": "spoke.nemotron-cpu-replay.v2",
                "status": "failure",
                "failure_phase": "route_identity",
                "input_path": str(source),
                "expected_audio_sha256": expected_audio_sha256,
                "wall_seconds": time.monotonic() - started,
                "receipt": receipt,
            }
            _write_replay_report(destination, report)
            raise NemotronCPUError(
                "Nemotron replay route identity did not prove CPU/full-buffer use; "
                f"report={destination}"
            )
        report = {
            "schema": "spoke.nemotron-cpu-replay.v2",
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
                "schema": "spoke.nemotron-cpu-replay.v2",
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
