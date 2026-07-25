"""Bounded local spool for raw stopped-capture audio.

The spool is a recovery black box: transcription may fail, VAD may trim badly,
and a gesture path may deliberately discard transcription input, but the raw
WAV returned by AudioCapture.stop() should survive long enough for immediate
operator recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import wave


_DEFAULT_ROOT = (
    Path.home() / "Library" / "Application Support" / "Spoke" / "audio-spool"
)
_DEFAULT_MAX_RECORDINGS = 12
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class AudioSpoolConfig:
    root: Path = _DEFAULT_ROOT
    enabled: bool = True
    max_recordings: int = _DEFAULT_MAX_RECORDINGS
    max_bytes: int = _DEFAULT_MAX_BYTES
    max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS

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
            max_bytes=_env_positive_int(
                "SPOKE_AUDIO_SPOOL_MAX_BYTES", _DEFAULT_MAX_BYTES
            ),
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
        capture_id = (
            f"{now.strftime('%Y%m%d-%H%M%S')}-"
            f"{now.microsecond // 1000:03d}-{uuid.uuid4().hex[:8]}"
        )
        wav_path = root / f"{capture_id}.wav"
        metadata_path = root / f"{capture_id}.json"
        sha256 = hashlib.sha256(wav_bytes).hexdigest()
        duration_seconds = _wav_duration_seconds(wav_bytes)
        payload = {
            "schema": "spoke.audio_spool.capture.v1",
            "capture_id": capture_id,
            "created_at": now.isoformat(),
            "wav_path": str(wav_path),
            "byte_count": len(wav_bytes),
            "duration_seconds": duration_seconds,
            "sha256": sha256,
            **(metadata or {}),
        }

        _write_atomic(wav_path, wav_bytes)
        _write_atomic(
            metadata_path,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )
        self._prune()
        return AudioSpoolRecord(
            capture_id=capture_id,
            wav_path=wav_path,
            metadata_path=metadata_path,
            byte_count=len(wav_bytes),
            duration_seconds=duration_seconds,
            sha256=sha256,
        )

    def _prune(self) -> None:
        root = self.config.root
        if not root.exists():
            return

        records = _list_records(root)
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.max_age_seconds
        )
        stale = [record for record in records if record.created_at < cutoff]
        for record in stale:
            _unlink_pair(record)

        records = [record for record in records if record.created_at >= cutoff]
        records.sort(key=lambda record: record.created_at, reverse=True)

        total = 0
        for index, record in enumerate(records):
            total_after = total + record.byte_count
            if (
                index >= self.config.max_recordings
                or total_after > self.config.max_bytes
            ):
                _unlink_pair(record)
                continue
            total = total_after


@dataclass(frozen=True)
class _ExistingRecord:
    wav_path: Path
    metadata_path: Path
    created_at: datetime
    byte_count: int


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
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
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _list_records(root: Path) -> list[_ExistingRecord]:
    records: list[_ExistingRecord] = []
    for metadata_path in root.glob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text())
            wav_path = Path(
                payload.get("wav_path") or metadata_path.with_suffix(".wav")
            )
            created_at = datetime.fromisoformat(payload["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            byte_count = int(
                payload.get("byte_count") or wav_path.stat().st_size
            )
        except Exception:
            continue
        records.append(
            _ExistingRecord(
                wav_path=wav_path,
                metadata_path=metadata_path,
                created_at=created_at,
                byte_count=byte_count,
            )
        )
    return records


def _unlink_pair(record: _ExistingRecord) -> None:
    for path in (record.wav_path, record.metadata_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
