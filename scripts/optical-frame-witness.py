#!/usr/bin/env python3
"""Capture or slice optical-shell summon/dismiss recordings into frame strips.

Typical usage with an existing QuickTime recording:

    uv run python scripts/optical-frame-witness.py --movie /path/to/recording.mov --auto-crop

The tool never launches, kills, or drives Spoke. For live smoke, start a short
timed capture, perform the summon/dismiss manually, then inspect the emitted
cropped frames and contact sheet.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from spoke.optical_frame_witness import (
    CropRect,
    build_extract_frames_command,
    ffmpeg_crop_filter,
    motion_bbox_from_gray_frames,
    parse_crop_rect,
    tile_layout,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "Library/Application Support/Spoke/optical_frame_witness"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool: {name}")


def _video_info(movie: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(movie),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration_s=float(data["format"]["duration"]),
    )


def _record_movie(path: Path, *, seconds: float, delay: float) -> None:
    _require_tool("screencapture")
    print(f"Recording {seconds:g}s to {path}")
    command = ["screencapture", "-v", "-x", f"-V{seconds:g}"]
    if delay > 0:
        print(f"Recording starts in {delay:g}s; prepare the target gesture...")
        command.append(f"-T{delay:g}")
    command.append(str(path))
    subprocess.run(command, check=True)
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit("screencapture did not produce a movie; try --movie with a QuickTime recording")


def _gray_frames(movie: Path, info: VideoInfo, *, fps: float, scale_width: int) -> tuple[list[bytes], int, int]:
    scaled_width = min(scale_width, info.width)
    scaled_height = max(1, round(info.height * (scaled_width / info.width)))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(movie),
        "-vf",
        f"fps={fps:g},scale={scaled_width}:{scaled_height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    frame_len = scaled_width * scaled_height
    if not result.stdout or len(result.stdout) < frame_len:
        return [], scaled_width, scaled_height
    frames = [
        result.stdout[i : i + frame_len]
        for i in range(0, len(result.stdout) - frame_len + 1, frame_len)
    ]
    return frames, scaled_width, scaled_height


def _auto_crop(movie: Path, info: VideoInfo, *, fps: float, margin: int) -> CropRect | None:
    frames, scaled_width, scaled_height = _gray_frames(movie, info, fps=min(fps, 30), scale_width=480)
    if not frames:
        return None
    low_res_crop = motion_bbox_from_gray_frames(
        frames,
        scaled_width,
        scaled_height,
        threshold=10,
        margin=max(2, round(margin * (scaled_width / info.width))),
    )
    if low_res_crop is None:
        return None
    return low_res_crop.scaled(
        info.width / scaled_width,
        info.height / scaled_height,
        bounds=(info.width, info.height),
    ).expanded(margin, bounds=(info.width, info.height))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _build_contact_sheet(frames_glob: str, output: Path, *, columns: int) -> None:
    frame_count = len(glob.glob(frames_glob))
    if frame_count == 0:
        raise SystemExit(f"No frames matched for contact sheet: {frames_glob}")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-pattern_type",
            "glob",
            "-i",
            frames_glob,
            "-vf",
            f"scale=320:-1,tile={tile_layout(frame_count, columns=columns)}",
            "-frames:v",
            "1",
            str(output),
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--movie", type=Path, help="Existing .mov/.mp4 recording to slice")
    source.add_argument("--record-seconds", type=float, help="Record a short manual screen capture")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay before timed recording starts")
    parser.add_argument("--label", default="optical-frame-witness", help="Artifact label")
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Frame extraction FPS")
    parser.add_argument("--crop", help="Crop rect as x,y,width,height")
    parser.add_argument("--auto-crop", action="store_true", help="Infer crop from motion")
    parser.add_argument("--margin", type=int, default=80, help="Pixels to expand inferred/manual crop")
    parser.add_argument("--start", type=float, help="Optional start time for extraction")
    parser.add_argument("--end", type=float, help="Optional end time for extraction")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in args.label).strip("-")
    out_dir = args.out_dir or (DEFAULT_OUTPUT_ROOT / f"{stamp}-{safe_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.movie is not None:
        movie = args.movie.expanduser().resolve()
        if not movie.exists():
            raise SystemExit(f"Movie not found: {movie}")
        movie_for_manifest = movie
    else:
        movie = out_dir / "recording.mov"
        _record_movie(movie, seconds=args.record_seconds, delay=args.delay)
        movie_for_manifest = movie

    info = _video_info(movie)
    crop = parse_crop_rect(args.crop).expanded(args.margin, bounds=(info.width, info.height)) if args.crop else None
    inferred_crop = None
    if args.auto_crop:
        inferred_crop = _auto_crop(movie, info, fps=args.fps, margin=args.margin)
        crop = crop or inferred_crop

    full_dir = out_dir / "full_frames"
    crop_dir = out_dir / "crop_frames"
    full_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    full_pattern = str(full_dir / "frame_%05d.jpg")
    _run(
        build_extract_frames_command(
            str(movie),
            full_pattern,
            fps=args.fps,
            start_s=args.start,
            end_s=args.end,
        )
    )

    crop_pattern = None
    if crop is not None:
        crop_pattern = str(crop_dir / "crop_%05d.jpg")
        _run(
            build_extract_frames_command(
                str(movie),
                crop_pattern,
                fps=args.fps,
                crop=crop,
                start_s=args.start,
                end_s=args.end,
            )
        )
        _build_contact_sheet(str(crop_dir / "*.jpg"), out_dir / "crop_contact_sheet.jpg", columns=8)
    _build_contact_sheet(str(full_dir / "*.jpg"), out_dir / "full_contact_sheet.jpg", columns=6)

    manifest = {
        "label": args.label,
        "source_movie": str(movie_for_manifest),
        "video": asdict(info),
        "extraction": {
            "fps": args.fps,
            "start_s": args.start,
            "end_s": args.end,
            "full_frames": str(full_dir),
            "crop_frames": str(crop_dir) if crop is not None else None,
            "crop_filter": ffmpeg_crop_filter(crop) if crop is not None else None,
        },
        "manual_crop": asdict(parse_crop_rect(args.crop)) if args.crop else None,
        "inferred_crop": asdict(inferred_crop) if inferred_crop is not None else None,
        "effective_crop": asdict(crop) if crop is not None else None,
        "artifacts": {
            "full_contact_sheet": str(out_dir / "full_contact_sheet.jpg"),
            "crop_contact_sheet": str(out_dir / "crop_contact_sheet.jpg") if crop is not None else None,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Optical frame witness written to {out_dir}")
    print(f"  movie: {movie}")
    print(f"  video: {info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration_s:.2f}s")
    print(f"  crop: {crop if crop is not None else 'none'}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
