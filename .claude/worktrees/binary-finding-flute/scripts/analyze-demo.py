#!/usr/bin/env python3
"""Analyze a DontType demo recording for visual review.

Extracts a brightness histogram over time, identifies attack/release
transients, and samples frames densely around those moments. Outputs
a directory of numbered JPEG frames for visual inspection.

Usage:
    python3 scripts/analyze-demo.py <video_path> [output_dir]

Requires: ffmpeg
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path



def get_video_info(path: str) -> dict:
    """Get video duration, fps, and resolution."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration:stream=width,height,r_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": num / den,
        "duration": float(data["format"]["duration"]),
    }


def extract_brightness_curve(path: str, info: dict, sample_fps: int = 10) -> list[tuple[float, float]]:
    """Extract average brightness at sample_fps intervals.

    Returns list of (timestamp, brightness) where brightness is 0.0-1.0.
    """
    duration = info["duration"]
    results = []

    for i in range(int(duration * sample_fps)):
        t = i / sample_fps
        result = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-ss", str(t), "-i", path,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray",
             "-vf", f"scale=320:-1", "pipe:1"],
            capture_output=True,
        )
        if result.stdout:
            pixels = result.stdout
            avg = sum(pixels) / len(pixels) / 255.0
            results.append((t, avg))

    return results


def find_transients(curve: list[tuple[float, float]], threshold: float = 0.02) -> list[tuple[float, str]]:
    """Find attack (brightness rising) and release (brightness falling) moments."""
    transients = []
    for i in range(1, len(curve)):
        t_prev, b_prev = curve[i - 1]
        t_curr, b_curr = curve[i]
        delta = b_curr - b_prev
        if abs(delta) > threshold:
            kind = "attack" if delta > 0 else "release"
            transients.append((t_curr, kind, delta))
    return transients


def extract_frames(path: str, output_dir: Path, timestamps: list[float],
                   info: dict, scale: int = 1728) -> list[str]:
    """Extract JPEG frames at specific timestamps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(timestamps):
        out = output_dir / f"frame_{i:04d}_{t:.2f}s.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-ss", str(t), "-i", path,
             "-frames:v", "1", "-vf", f"scale={scale}:-1",
             "-q:v", "2", str(out)],
        )
        paths.append(str(out))
    return paths


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video_path> [output_dir]", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/demo-analysis")

    print(f"Analyzing: {video_path}")
    info = get_video_info(video_path)
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.0f}fps, {info['duration']:.1f}s")

    # Phase 1: brightness histogram at 10fps
    print("Extracting brightness curve (10fps)...")
    curve = extract_brightness_curve(video_path, info, sample_fps=10)

    # Save curve as TSV
    curve_path = output_dir / "brightness_curve.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(curve_path, "w") as f:
        f.write("time_s\tbrightness\n")
        for t, b in curve:
            f.write(f"{t:.3f}\t{b:.4f}\n")
    print(f"  Saved brightness curve: {curve_path}")

    # Phase 2: find transients
    transients = find_transients(curve)
    print(f"  Found {len(transients)} transients:")
    for t, kind, delta in transients:
        print(f"    {t:.2f}s  {kind:8s}  delta={delta:+.4f}")

    # Phase 3: dense sampling around transients (±0.5s at ~15fps)
    dense_timestamps = set()
    for t, kind, _ in transients:
        for offset in range(-8, 9):  # ±0.5s at ~15fps
            ts = t + offset * (1.0 / 15.0)
            if 0 <= ts <= info["duration"]:
                dense_timestamps.add(round(ts, 3))

    # Also sample uniformly at 2fps for context
    for i in range(int(info["duration"] * 2)):
        dense_timestamps.add(round(i / 2.0, 3))

    timestamps = sorted(dense_timestamps)
    print(f"Extracting {len(timestamps)} frames...")
    paths = extract_frames(video_path, output_dir / "frames", timestamps, info)
    print(f"  Saved {len(paths)} frames to {output_dir / 'frames'}")

    # Summary
    print(f"\nDone. Output in {output_dir}")
    print(f"  brightness_curve.tsv  — brightness over time")
    print(f"  frames/               — {len(paths)} JPEG frames")


if __name__ == "__main__":
    main()
