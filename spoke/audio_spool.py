"""Preserved stopped-capture audio and append-only transcription history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
import wave


_DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "Spoke" / "audio-spool"
_DEFAULT_MAX_RECORDINGS = None
_DEFAULT_MAX_BYTES = None
_DEFAULT_MAX_AGE_SECONDS = None


@dataclass(frozen=True)
class AudioSpoolConfig:
    root: Path = _DEFAULT_ROOT
    enabled: bool = True
    max_recordings: int | None = _DEFAULT_MAX_RECORDINGS
    max_bytes: int | None = _DEFAULT_MAX_BYTES
    max_age_seconds: int | None = _DEFAULT_MAX_AGE_SECONDS

    @classmethod
    def from_env(cls) -> "AudioSpoolConfig":
        return cls(
            root=Path(
                os.environ.get("SPOKE_AUDIO_SPOOL_DIR", str(_DEFAULT_ROOT))
            ).expanduser(),
            enabled=_env_enabled("SPOKE_AUDIO_SPOOL_ENABLED", default=True),
            max_recordings=_env_positive_int(
                "SPOKE_AUDIO_SPOOL_MAX_RECORDINGS", _DEFAULT_MAX_RECORDINGS
            ),
            max_bytes=_env_positive_int("SPOKE_AUDIO_SPOOL_MAX_BYTES", _DEFAULT_MAX_BYTES),
            max_age_seconds=_env_positive_int(
                "SPOKE_AUDIO_SPOOL_MAX_AGE_SECONDS", _DEFAULT_MAX_AGE_SECONDS
            ),
        )


@dataclass(frozen=True)
class AudioSpoolRecord:
    capture_id: str
    wav_path: Path
    metadata_path: Path
    byte_count: int
    duration_seconds: float | None
    sha256: str


class AudioSpool:
    def __init__(self, config: AudioSpoolConfig | None = None) -> None:
        self.config = config or AudioSpoolConfig.from_env()
        self._history_lock = threading.RLock()

    @classmethod
    def from_env(cls) -> "AudioSpool":
        return cls(AudioSpoolConfig.from_env())

    def spool_capture(
        self,
        wav_bytes: bytes,
        *,
        metadata: dict | None = None,
    ) -> AudioSpoolRecord | None:
        if not self.config.enabled or not wav_bytes:
            return None

        root = self.config.root
        root.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        capture_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}-{uuid.uuid4().hex[:8]}"
        wav_path = root / f"{capture_id}.wav"
        metadata_path = root / f"{capture_id}.json"
        sha256 = hashlib.sha256(wav_bytes).hexdigest()
        duration_seconds = _wav_duration_seconds(wav_bytes)
        payload = {
            **(metadata or {}),
            "schema": "spoke.audio_spool.capture.v1",
            "capture_id": capture_id,
            "created_at": now.isoformat(),
            "wav_path": str(wav_path),
            "byte_count": len(wav_bytes),
            "duration_seconds": duration_seconds,
            "sha256": sha256,
        }

        _write_atomic(wav_path, wav_bytes)
        _write_atomic(metadata_path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        if not payload.get("preserve_audio"):
            self._prune()
        return AudioSpoolRecord(
            capture_id=capture_id,
            wav_path=wav_path,
            metadata_path=metadata_path,
            byte_count=len(wav_bytes),
            duration_seconds=duration_seconds,
            sha256=sha256,
        )

    def _capture_path(self, capture_id: str) -> Path:
        if not isinstance(capture_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", capture_id):
            raise ValueError("Invalid capture id")
        return self.config.root / capture_id

    def read_recording_audio(self, capture_id: str) -> bytes:
        base = self._capture_path(capture_id)
        payload = _read_json_object(base.with_suffix(".json"))
        if payload.get("capture_id") != capture_id or payload.get("schema") != "spoke.audio_spool.capture.v1":
            raise ValueError("Unverified capture metadata")
        wav_bytes = base.with_suffix(".wav").read_bytes()
        if hashlib.sha256(wav_bytes).hexdigest() != payload.get("sha256"):
            raise ValueError("Recording SHA-256 does not match preserved capture")
        return wav_bytes

    def start_attempt(self, capture_id: str, *, requested: dict, kind: str = "live") -> str:
        base = self._capture_path(capture_id)
        capture = _read_json_object(base.with_suffix(".json"))
        if capture.get("capture_id") != capture_id or capture.get("schema") != "spoke.audio_spool.capture.v1":
            raise ValueError("Capture identity does not match")
        now = datetime.now(timezone.utc)
        attempt_id = f"{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
        payload = {
            "schema": "spoke.recording_attempt.request.v1",
            "capture_id": capture_id, "attempt_id": attempt_id,
            "created_at": now.isoformat(), "requested": requested,
            "kind": kind, "audio_sha256": capture["sha256"], "pid": os.getpid(),
        }
        with self._history_lock:
            directory = self.config.root / "attempts" / capture_id
            directory.mkdir(parents=True, exist_ok=True)
            _write_new_json(directory / f"{attempt_id}.request.json", payload)
        return attempt_id

    def finish_attempt(
        self, capture_id: str, attempt_id: str, *, text: str | None,
        effective: dict, wall_seconds: float, error: str | None = None,
        evidence: dict | None = None,
    ) -> None:
        self._capture_path(capture_id)
        self._capture_path(attempt_id)
        directory = self.config.root / "attempts" / capture_id
        request = _read_json_object(directory / f"{attempt_id}.request.json")
        if request.get("capture_id") != capture_id or request.get("attempt_id") != attempt_id:
            raise ValueError("Attempt identity does not match")
        payload = {
            "schema": "spoke.recording_attempt.result.v1",
            "capture_id": capture_id, "attempt_id": attempt_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if error is not None else "success" if text and text.strip() else "blank",
            "text": text, "error": error, "effective": effective,
            "wall_seconds": wall_seconds, "evidence": evidence or {},
        }
        with self._history_lock:
            _write_new_json(directory / f"{attempt_id}.result.json", payload)

    def record_delivery(self, capture_id: str, attempt_id: str, *, state: str, detail: str = "") -> None:
        self._capture_path(capture_id)
        self._capture_path(attempt_id)
        directory = self.config.root / "attempts" / capture_id
        if not (directory / f"{attempt_id}.request.json").is_file():
            raise ValueError("Delivery has no transcription attempt")
        _write_new_json(directory / f"{attempt_id}.delivery-{uuid.uuid4().hex}.json", {
            "schema": "spoke.recording_delivery.v1", "attempt_id": attempt_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": state, "detail": detail,
        })

    def list_recordings(self) -> list[dict]:
        """Read metadata only; opening history never decodes or loads audio."""
        records = []
        root = self.config.root
        capture_ids = {p.stem for pattern in ("*.json", "*.wav") for p in root.glob(pattern)}
        for capture_id in sorted(capture_ids, reverse=True):
            base = self._capture_path(capture_id)
            try:
                payload = _read_json_object(base.with_suffix(".json"))
                if payload.get("schema") != "spoke.audio_spool.capture.v1" or payload.get("capture_id") != capture_id:
                    raise ValueError("Unverified capture metadata")
                item = dict(payload)
            except (OSError, ValueError) as exc:
                item = {"capture_id": capture_id, "error": str(exc), "status": "unverified"}
            item["wav_path"] = str(base.with_suffix(".wav"))
            item["audio_available"] = base.with_suffix(".wav").is_file()
            attempts = []
            directory = root / "attempts" / capture_id
            for path in sorted(directory.glob("*.request.json")):
                try:
                    attempt = _read_json_object(path)
                    attempt_id = attempt["attempt_id"]
                    if (path.name != f"{attempt_id}.request.json" or attempt.get("capture_id") != capture_id
                            or attempt.get("schema") != "spoke.recording_attempt.request.v1"
                            or not isinstance(attempt.get("requested"), dict)):
                        raise ValueError("Attempt identity mismatch")
                    attempt["status"] = "pending"
                    result_path = directory / f"{attempt_id}.result.json"
                    if result_path.exists():
                        result = _read_json_object(result_path)
                        if (result.get("attempt_id") != attempt_id or result.get("capture_id") != capture_id
                                or result.get("schema") != "spoke.recording_attempt.result.v1"
                                or result.get("status") not in {"success", "blank", "failed"}
                                or not isinstance(result.get("effective"), dict)
                                or (result.get("text") is not None and not isinstance(result.get("text"), str))
                                or (result.get("status") == "success" and not str(result.get("text") or "").strip())):
                            raise ValueError("Result identity mismatch")
                        attempt.update(result)
                    deliveries = [_read_json_object(p) for p in directory.glob(f"{attempt_id}.delivery-*.json")]
                    for event in deliveries:
                        if (event.get("schema") != "spoke.recording_delivery.v1"
                                or event.get("attempt_id") != attempt_id
                                or not isinstance(event.get("state"), str)
                                or not isinstance(event.get("created_at"), str)):
                            raise ValueError("Unverified delivery event")
                    attempt["deliveries"] = sorted(deliveries, key=lambda event: event["created_at"])
                except (OSError, ValueError, KeyError) as exc:
                    attempt = {"attempt_id": path.name.removesuffix(".request.json"),
                               "status": "unverified", "error": str(exc)}
                attempts.append(attempt)
            item["attempts"] = attempts
            item.setdefault("status", attempts[-1]["status"] if attempts else "not_transcribed")
            if not item["audio_available"]:
                item["audio_error"] = "Recording audio is missing"
            records.append(item)
        return records

    def _prune(self) -> None:
        root = self.config.root
        if not root.exists():
            return

        if all(value is None for value in (
            self.config.max_recordings, self.config.max_bytes, self.config.max_age_seconds,
        )):
            return
        # Attempt lineage is a preservation obligation, including failed/blank attempts.
        records = [record for record in _list_records(root)
                   if not record.preserved and not (root / "attempts" / record.metadata_path.stem).exists()]
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self.config.max_age_seconds)
                  if self.config.max_age_seconds is not None else None)
        stale = [record for record in records if cutoff is not None and record.created_at < cutoff]
        for record in stale:
            _unlink_pair(record)

        records = [record for record in records if cutoff is None or record.created_at >= cutoff]
        records.sort(key=lambda record: record.created_at, reverse=True)

        keep: list[_ExistingRecord] = []
        total = 0
        for index, record in enumerate(records):
            total_after = total + record.byte_count
            if ((self.config.max_recordings is not None and index >= self.config.max_recordings)
                    or (self.config.max_bytes is not None and total_after > self.config.max_bytes)):
                _unlink_pair(record)
                continue
            keep.append(record)
            total = total_after


@dataclass(frozen=True)
class _ExistingRecord:
    wav_path: Path
    metadata_path: Path
    created_at: datetime
    byte_count: int
    preserved: bool = False


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_positive_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"0", "off", "none", "unlimited"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _wav_duration_seconds(wav_bytes: bytes) -> float | None:
    try:
        import io

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frame_rate = wf.getframerate()
            if frame_rate <= 0:
                return None
            return wf.getnframes() / float(frame_rate)
    except Exception:
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _write_new_json(path: Path, payload: dict) -> None:
    """Publish a complete immutable event, refusing a second terminal result."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _list_records(root: Path) -> list[_ExistingRecord]:
    records: list[_ExistingRecord] = []
    for metadata_path in root.glob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text())
            wav_path = metadata_path.with_suffix(".wav")
            created_at = datetime.fromisoformat(payload["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            byte_count = int(payload.get("byte_count") or wav_path.stat().st_size)
        except Exception:
            continue
        records.append(
            _ExistingRecord(
                wav_path=wav_path,
                metadata_path=metadata_path,
                created_at=created_at,
                byte_count=byte_count,
                preserved=bool(payload.get("preserve_audio")),
            )
        )
    return records


def _unlink_pair(record: _ExistingRecord) -> None:
    for path in (record.wav_path, record.metadata_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
