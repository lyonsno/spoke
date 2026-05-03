"""Helpers for optical-shell frame-witness capture artifacts.

The GUI smoke itself is still human-visible, but these helpers make the result
agent-inspectable: crop consistently, sample densely, and isolate the pixels
that moved during summon/dismiss animations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence


@dataclass(frozen=True)
class CropRect:
    x: int
    y: int
    width: int
    height: int

    def expanded(self, margin: int, *, bounds: tuple[int, int] | None = None) -> "CropRect":
        if margin < 0:
            raise ValueError("margin must be non-negative")

        x = self.x - margin
        y = self.y - margin
        width = self.width + margin * 2
        height = self.height + margin * 2

        if bounds is None:
            return CropRect(max(0, x), max(0, y), width, height)

        max_width, max_height = bounds
        x = max(0, x)
        y = max(0, y)
        right = min(max_width, self.x + self.width + margin)
        bottom = min(max_height, self.y + self.height + margin)
        return CropRect(x, y, max(1, right - x), max(1, bottom - y))

    def scaled(self, scale_x: float, scale_y: float, *, bounds: tuple[int, int] | None = None) -> "CropRect":
        x = max(0, int(round(self.x * scale_x)))
        y = max(0, int(round(self.y * scale_y)))
        width = max(1, int(round(self.width * scale_x)))
        height = max(1, int(round(self.height * scale_y)))
        if bounds is None:
            return CropRect(x, y, width, height)

        max_width, max_height = bounds
        x = min(x, max_width - 1)
        y = min(y, max_height - 1)
        return CropRect(
            x,
            y,
            min(width, max_width - x),
            min(height, max_height - y),
        )


def parse_crop_rect(value: str) -> CropRect:
    """Parse a crop rectangle as x,y,width,height or x:y:width:height."""
    if not value:
        raise ValueError("crop rect must be non-empty")

    delimiter = "," if "," in value else ":"
    parts = value.split(delimiter)
    if len(parts) != 4:
        raise ValueError("crop rect must have four components: x,y,width,height")

    try:
        x, y, width, height = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise ValueError("crop rect components must be integers") from exc

    if x < 0 or y < 0:
        raise ValueError("crop rect origin must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("crop rect width and height must be positive")
    return CropRect(x, y, width, height)


def ffmpeg_crop_filter(crop: CropRect) -> str:
    return f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}"


def sample_timestamps(
    *,
    duration_s: float,
    fps: float,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int | None = None,
) -> list[float]:
    """Return dense frame timestamps clamped to the requested interval."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when supplied")

    start = max(0.0, start_s)
    end = duration_s if end_s is None else min(duration_s, end_s)
    if end < start:
        raise ValueError("end_s must be greater than or equal to start_s")

    if max_frames is not None:
        if max_frames == 1:
            return [round(start, 6)]
        step = (end - start) / (max_frames - 1)
        return [round(start + step * i, 6) for i in range(max_frames)]

    interval = 1.0 / fps
    count = floor(((end - start) / interval) + 1e-9) + 1
    return [round(start + interval * i, 6) for i in range(count)]


def motion_bbox_from_gray_frames(
    frames: Sequence[bytes],
    width: int,
    height: int,
    *,
    threshold: int = 8,
    margin: int = 0,
) -> CropRect | None:
    """Return the union bbox of pixels that changed between adjacent gray frames."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if len(frames) < 2:
        return None

    expected_len = width * height
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for prev, curr in zip(frames, frames[1:]):
        if len(prev) != expected_len or len(curr) != expected_len:
            raise ValueError("gray frame length does not match dimensions")
        for idx, (before, after) in enumerate(zip(prev, curr)):
            if abs(after - before) < threshold:
                continue
            y, x = divmod(idx, width)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if max_x < 0:
        return None

    return CropRect(
        min_x,
        min_y,
        max_x - min_x + 1,
        max_y - min_y + 1,
    ).expanded(margin, bounds=(width, height))


def build_extract_frames_command(
    movie_path: str,
    output_pattern: str,
    *,
    fps: float,
    crop: CropRect | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    quality: int = 2,
) -> list[str]:
    """Build an ffmpeg command that extracts a dense JPEG frame sequence."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    filters = [f"fps={fps:g}"]
    if crop is not None:
        filters.append(ffmpeg_crop_filter(crop))

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s is not None:
        command.extend(["-ss", f"{start_s:g}"])
    if end_s is not None:
        command.extend(["-to", f"{end_s:g}"])
    command.extend(
        [
            "-i",
            movie_path,
            "-vf",
            ",".join(filters),
            "-q:v",
            str(quality),
            output_pattern,
        ]
    )
    return command


def tile_layout(frame_count: int, *, columns: int) -> str:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")
    return f"{columns}x{ceil(frame_count / columns)}"
