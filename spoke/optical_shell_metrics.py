"""Shared counters for the optical shell compositor campaign.

The surface stays intentionally small: a few counts plus cheap average timing
proxies that can be updated from capture callbacks and overlay timer ticks
without affecting compositor behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any


@dataclass
class OpticalShellMetrics:
    """Cheap counters and timing proxies for compositor-adjacent work."""

    display_link_ticks: int = 0
    presented_frames: int = 0
    capture_ticks: int = 0
    capture_polls: int = 0
    skipped_frames: int = 0
    duplicate_frames: int = 0
    brightness_samples: int = 0
    total_display_tick_ms: float = 0.0
    total_presented_frame_ms: float = 0.0
    total_capture_tick_ms: float = 0.0
    total_capture_poll_ms: float = 0.0
    total_brightness_sample_ms: float = 0.0
    total_capture_tick_interval_ms: float = 0.0
    total_capture_poll_interval_ms: float = 0.0
    total_display_interval_ms: float = 0.0
    total_presented_interval_ms: float = 0.0
    total_brightness_interval_ms: float = 0.0
    _last_capture_tick_at: float | None = field(default=None, init=False, repr=False)
    _last_capture_poll_at: float | None = field(default=None, init=False, repr=False)
    _last_display_tick_at: float | None = field(default=None, init=False, repr=False)
    _last_presented_at: float | None = field(default=None, init=False, repr=False)
    _last_brightness_sample_at: float | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_capture_tick(
        self, *, elapsed_ms: float | None = None, now: float | None = None
    ) -> None:
        with self._lock:
            self.capture_ticks += 1
            if elapsed_ms is not None:
                self.total_capture_tick_ms += max(elapsed_ms, 0.0)
            if now is not None:
                if self._last_capture_tick_at is not None:
                    self.total_capture_tick_interval_ms += max(
                        (now - self._last_capture_tick_at) * 1000.0, 0.0
                    )
                self._last_capture_tick_at = now

    def record_capture_poll(
        self,
        new_chunk_count: int,
        *,
        elapsed_ms: float | None = None,
        now: float | None = None,
    ) -> None:
        with self._lock:
            self.capture_polls += 1
            if elapsed_ms is not None:
                self.total_capture_poll_ms += max(elapsed_ms, 0.0)
            if now is not None:
                if self._last_capture_poll_at is not None:
                    self.total_capture_poll_interval_ms += max(
                        (now - self._last_capture_poll_at) * 1000.0, 0.0
                    )
                self._last_capture_poll_at = now
            if new_chunk_count <= 0:
                self.duplicate_frames += 1
            elif new_chunk_count > 1:
                self.skipped_frames += new_chunk_count - 1

    def record_display_tick(
        self, *, elapsed_ms: float | None = None, now: float | None = None
    ) -> None:
        with self._lock:
            self.display_link_ticks += 1
            if elapsed_ms is not None:
                self.total_display_tick_ms += max(elapsed_ms, 0.0)
            if now is not None:
                if self._last_display_tick_at is not None:
                    self.total_display_interval_ms += max(
                        (now - self._last_display_tick_at) * 1000.0, 0.0
                    )
                self._last_display_tick_at = now

    def record_presented_frame(
        self, *, elapsed_ms: float | None = None, now: float | None = None
    ) -> None:
        with self._lock:
            self.presented_frames += 1
            if elapsed_ms is not None:
                self.total_presented_frame_ms += max(elapsed_ms, 0.0)
            if now is not None:
                if self._last_presented_at is not None:
                    self.total_presented_interval_ms += max(
                        (now - self._last_presented_at) * 1000.0, 0.0
                    )
                self._last_presented_at = now

    def record_brightness_sample(
        self, *, elapsed_ms: float | None = None, now: float | None = None
    ) -> None:
        with self._lock:
            self.brightness_samples += 1
            if elapsed_ms is not None:
                self.total_brightness_sample_ms += max(elapsed_ms, 0.0)
            if now is not None:
                if self._last_brightness_sample_at is not None:
                    self.total_brightness_interval_ms += max(
                        (now - self._last_brightness_sample_at) * 1000.0, 0.0
                    )
                self._last_brightness_sample_at = now

    def snapshot(self) -> dict[str, Any]:
        """Return a stable JSON-friendly view of the current metrics."""
        with self._lock:
            return {
                "display_link_ticks": self.display_link_ticks,
                "presented_frames": self.presented_frames,
                "capture_ticks": self.capture_ticks,
                "capture_polls": self.capture_polls,
                "skipped_frames": self.skipped_frames,
                "duplicate_frames": self.duplicate_frames,
                "brightness_samples": self.brightness_samples,
                "avg_display_tick_ms": self._average(
                    self.total_display_tick_ms, self.display_link_ticks
                ),
                "avg_presented_frame_ms": self._average(
                    self.total_presented_frame_ms, self.presented_frames
                ),
                "avg_capture_tick_ms": self._average(
                    self.total_capture_tick_ms, self.capture_ticks
                ),
                "avg_capture_poll_ms": self._average(
                    self.total_capture_poll_ms, self.capture_polls
                ),
                "avg_brightness_sample_ms": self._average(
                    self.total_brightness_sample_ms, self.brightness_samples
                ),
                "avg_capture_tick_interval_ms": self._average(
                    self.total_capture_tick_interval_ms,
                    max(self.capture_ticks - 1, 0),
                ),
                "avg_capture_poll_interval_ms": self._average(
                    self.total_capture_poll_interval_ms,
                    max(self.capture_polls - 1, 0),
                ),
                "avg_display_interval_ms": self._average(
                    self.total_display_interval_ms,
                    max(self.display_link_ticks - 1, 0),
                ),
                "avg_presented_interval_ms": self._average(
                    self.total_presented_interval_ms,
                    max(self.presented_frames - 1, 0),
                ),
                "avg_brightness_interval_ms": self._average(
                    self.total_brightness_interval_ms,
                    max(self.brightness_samples - 1, 0),
                ),
            }

    @staticmethod
    def _average(total_ms: float, count: int) -> float:
        if count <= 0:
            return 0.0
        return total_ms / count
