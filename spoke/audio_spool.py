"""Local replay spool for raw stopped-capture audio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import time
import uuid
import wave


logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "Spoke" / "audio-spool"
_DEFAULT_WARN_RECORDINGS = 100
_DEFAULT_WARN_BYTES = 10 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class AudioSpoolConfig:
    root: Path = _DEFAULT_ROOT
    enabled: bool = False
    warn_recordings: int = _DEFAULT_WARN_RECORDINGS
    warn_bytes: int = _DEFAULT_WARN_BYTES

    @classmethod
    def from_env(cls) -> "AudioSpoolConfig":
        return cls(
            root=Path(
                os.environ.get("SPOKE_AUDIO_SPOOL_DIR", str(_DEFAULT_ROOT))
            ).expanduser(),
            enabled=_env_enabled("SPOKE_AUDIO_SPOOL_ENABLED", default=False),
            warn_recordings=_env_positive_int(
                "SPOKE_AUDIO_SPOOL_WARN_RECORDINGS",
                _DEFAULT_WARN_RECORDINGS,
            ),
            warn_bytes=_env_positive_int(
                "SPOKE_AUDIO_SPOOL_WARN_BYTES",
                _DEFAULT_WARN_BYTES,
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
        while True:
            capture_id = _capture_id(now)
            reservation_path = root / f".{capture_id}.reserve"
            try:
                reservation_fd = os.open(
                    reservation_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                continue
            os.close(reservation_fd)
            wav_path = root / f"{capture_id}.wav"
            metadata_path = root / f"{capture_id}.json"
            if not wav_path.exists() and not metadata_path.exists():
                break
            reservation_path.unlink(missing_ok=True)
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

        try:
            _write_atomic(wav_path, wav_bytes)
            _write_atomic(
                metadata_path,
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            )
        finally:
            reservation_path.unlink(missing_ok=True)
        self._warn_if_overloaded()
        return AudioSpoolRecord(
            capture_id=capture_id,
            wav_path=wav_path,
            metadata_path=metadata_path,
            byte_count=len(wav_bytes),
            duration_seconds=duration_seconds,
            sha256=sha256,
        )

    def _warn_if_overloaded(self) -> None:
        wav_paths = list(self.config.root.glob("*.wav"))
        total_bytes = 0
        for path in wav_paths:
            try:
                total_bytes += path.stat().st_size
            except OSError:
                continue
        if (
            len(wav_paths) > self.config.warn_recordings
            or total_bytes > self.config.warn_bytes
        ):
            logger.warning(
                "[audio-spool] *** AUDIO SPOOL OVERLOAD: %s contains %d recordings "
                "and %d bytes; no evidence was deleted. Inspect capture retention "
                "and archive deliberately. ***",
                self.config.root,
                len(wav_paths),
                total_bytes,
            )


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _capture_id(now: datetime) -> str:
    return (
        f"{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}-"
        f"{uuid.uuid4().hex[:8]}"
    )


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
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frame_rate = wf.getframerate()
            if frame_rate <= 0:
                return None
            return wf.getnframes() / float(frame_rate)
    except (wave.Error, EOFError, OSError, ValueError):
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        tmp_path.write_bytes(data)
        os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
