"""Autonomous visual witness harness for the Perceptasia Throughglass graft."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import struct
import zlib

import numpy as np

from .retina_lasso_witness import (
    DEFAULT_PASSIVE_CAPTURE_PROFILE,
    DEFAULT_STRESS_CAPTURE_PROFILE,
    _capture_profile_from_cli,
    run_autonomous_hammer_witness,
    run_witness_window,
)

DEFAULT_TRACE_PATH = Path("/tmp/spoke-perceptasia-throughglass-graft-command-overlay-trace.jsonl")
DEFAULT_OUTPUT_ROOT = Path("/tmp/spoke-perceptasia-throughglass-autowitnesses")
DEFAULT_LAUNCH_TARGET = "perceptasia_throughglass_graft"
DEFAULT_LANE = "perceptasia-throughglass-graft"
DEFAULT_DIAULOS = "Warpstorm Pit Boss"
DEFAULT_SOURCE_APP = "Spoke"
DEFAULT_SOURCE_WINDOW = "Perceptasia Throughglass / Assistant Overlay"
DEFAULT_LOG_PATHS = (
    Path.home() / "Library" / "Logs" / "spoke-main-launch.log",
    Path.home() / "Library" / "Logs" / "spoke-launch-target.log",
)
VISUAL_CONTENT_CLASSIFIER_VERSION = "throughglass_pixels.v3"
_VISUAL_PASS_THRESHOLDS = {
    "panel_material_fraction": 0.07,
    "saturated_pixel_fraction": 0.04,
    "edge_density": 0.09,
    "graph_saturated_pixel_fraction": 0.04,
    "graph_edge_density": 0.09,
}


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_output_dir(root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(root).expanduser() / f"throughglass-autowitness-{_timestamp_slug()}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a Retina Lasso witness for the Perceptasia Throughglass graft. "
            "By default this is passive and does not relaunch Spoke."
        )
    )
    parser.add_argument("--trace", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", help="Exact output directory; overrides --output-root.")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--capture-profile",
        choices=("low-perturbation", "stress"),
        help="Defaults to low-perturbation unless launch/stimulus mode is enabled.",
    )
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--diaulos", default=DEFAULT_DIAULOS)
    parser.add_argument("--source-app", default=DEFAULT_SOURCE_APP)
    parser.add_argument("--source-window", default=DEFAULT_SOURCE_WINDOW)
    parser.add_argument("--capture-command")
    parser.add_argument(
        "--allow-unproven",
        action="store_true",
        help="Write the witness index but return success even if Throughglass content proof is absent.",
    )
    parser.add_argument(
        "--log-path",
        action="append",
        dest="log_paths",
        help="Spoke runtime log to inspect for Throughglass content-proof receipts.",
    )
    parser.add_argument("--perceptasia-root", help=argparse.SUPPRESS)
    parser.add_argument("--watch-trace", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--watch-timeout", type=float, default=7200.0, help=argparse.SUPPRESS)
    parser.add_argument("--event-capture-duration", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--watch-max-captures", type=int, default=96, help=argparse.SUPPRESS)
    parser.add_argument("--max-trigger-lag", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--open-ready-timeout", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--open-ready-poll-interval", type=float, default=0.025, help=argparse.SUPPRESS)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the Throughglass target before capture; this may replace the current live Spoke process.",
    )
    parser.add_argument("--launch-target", default=DEFAULT_LAUNCH_TARGET)
    parser.add_argument("--launch-wait", type=float, default=3.0)
    parser.add_argument("--hammer-toggles", type=int, default=0)
    parser.add_argument("--toggle-control-path", help=argparse.SUPPRESS)
    parser.add_argument("--toggle-interval", type=float, default=0.18)
    parser.add_argument("--pre-hammer-delay", type=float, default=0.35)
    parser.add_argument("--retarget-during-dismiss-repeats", type=int, default=0)
    parser.add_argument("--open-dwell", type=float, default=0.75)
    parser.add_argument("--dismiss-retarget-delay", type=float, default=0.08)
    parser.add_argument("--reopen-dwell", type=float, default=0.75)
    parser.add_argument("--cycle-pause", type=float, default=0.2)
    return parser


def _load_index(index_path: Path) -> dict:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_index(index_path: Path, payload: dict) -> None:
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_log_contract(log_paths: list[Path]) -> dict:
    for path in log_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        latest_setup = -1
        latest_url = None
        latest_panel_rect = None
        for index, line in enumerate(lines):
            match = re.search(r"Perceptasia Throughglass: setup begin url=([^\s]+)", line)
            if match:
                latest_setup = index
                latest_url = match.group(1)
                latest_panel_rect = None
            rect_match = re.search(
                r"Perceptasia Throughglass: setup complete x=([0-9.]+) y=([0-9.]+) w=([0-9.]+) h=([0-9.]+)",
                line,
            )
            if rect_match and latest_setup >= 0 and index >= latest_setup:
                latest_panel_rect = {
                    "x": float(rect_match.group(1)),
                    "y": float(rect_match.group(2)),
                    "width": float(rect_match.group(3)),
                    "height": float(rect_match.group(4)),
                }
        if latest_setup < 0:
            continue
        scoped = lines[latest_setup:]
        content_verified = any("Perceptasia Throughglass: content verified" in line for line in scoped)
        webview_loaded = any("Perceptasia Throughglass: WKWebView request loaded" in line for line in scoped)
        fallback_seen = any(
            marker in line
            for line in scoped
            for marker in (
                "WKWebView unavailable",
                "provider unavailable",
                "content verification failed",
            )
        )
        return {
            "latest_setup_seen": True,
            "log_path": str(path),
            "provider_url": latest_url,
            "panel_rect_points": latest_panel_rect,
            "webview_loaded": webview_loaded,
            "content_verified": content_verified,
            "fallback_seen": fallback_seen,
            "passed": bool(content_verified and webview_loaded and not fallback_seen),
        }
    return {
        "latest_setup_seen": False,
        "provider_url": None,
        "panel_rect_points": None,
        "webview_loaded": False,
        "content_verified": False,
        "fallback_seen": False,
        "passed": False,
    }


def _paeth_predictor(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _read_png_rgb(path: Path) -> np.ndarray:
    try:
        from Foundation import NSURL
        from Quartz import (
            CGDataProviderCopyData,
            CGImageGetBitsPerPixel,
            CGImageGetBytesPerRow,
            CGImageGetDataProvider,
            CGImageGetHeight,
            CGImageGetWidth,
            CGImageSourceCreateImageAtIndex,
            CGImageSourceCreateWithURL,
        )

        source = CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
        image = CGImageSourceCreateImageAtIndex(source, 0, None) if source is not None else None
        if image is not None:
            width = int(CGImageGetWidth(image))
            height = int(CGImageGetHeight(image))
            row_bytes = int(CGImageGetBytesPerRow(image))
            bits_per_pixel = int(CGImageGetBitsPerPixel(image))
            channels = bits_per_pixel // 8
            if width > 0 and height > 0 and channels in (3, 4):
                data = bytes(CGDataProviderCopyData(CGImageGetDataProvider(image)))
                rows = np.frombuffer(data, dtype=np.uint8).reshape((height, row_bytes))[:, : width * channels]
                return rows.reshape((height, width, channels))[:, :, :3]
    except Exception:
        pass

    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not_png")
    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("unsupported_png_layout")
        elif kind == b"IDAT":
            idat.extend(chunk)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError("unsupported_png_color")
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    row_bytes = width * channels
    raw = zlib.decompress(bytes(idat))
    rows: list[bytearray] = []
    source_offset = 0
    previous = bytearray(row_bytes)
    bytes_per_pixel = channels
    for _ in range(height):
        filter_type = raw[source_offset]
        source_offset += 1
        current = bytearray(raw[source_offset : source_offset + row_bytes])
        source_offset += row_bytes
        for index, value in enumerate(current):
            left = current[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            up_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 1:
                current[index] = (value + left) & 0xFF
            elif filter_type == 2:
                current[index] = (value + up) & 0xFF
            elif filter_type == 3:
                current[index] = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                current[index] = (value + _paeth_predictor(left, up, up_left)) & 0xFF
            elif filter_type != 0:
                raise ValueError("unsupported_png_filter")
        rows.append(current)
        previous = current
    image = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape((height, width, channels))
    if color_type == 0:
        return np.repeat(image[:, :, :1], 3, axis=2)
    return image[:, :, :3]


def _runtime_panel_crop(image: np.ndarray, panel_rect_points: dict | None) -> tuple[np.ndarray, dict] | None:
    if not isinstance(panel_rect_points, dict):
        return None
    try:
        x = float(panel_rect_points["x"])
        y = float(panel_rect_points["y"])
        width = float(panel_rect_points["width"])
        height = float(panel_rect_points["height"])
    except (KeyError, TypeError, ValueError):
        return None
    image_height, image_width, _ = image.shape
    point_screen_width = (x * 2.0) + width
    scale = image_width / point_screen_width if point_screen_width > 0 else 2.0
    if not 0.75 <= scale <= 3.5:
        scale = 2.0
    pad = max(6.0, 10.0 * scale)
    x0 = max(0, int(round(x * scale - pad)))
    x1 = min(image_width, int(round((x + width) * scale + pad)))
    y0 = max(0, int(round(image_height - ((y + height) * scale) - pad)))
    y1 = min(image_height, int(round(image_height - (y * scale) + pad)))
    if (x1 - x0) < image_width * 0.12 or (y1 - y0) < image_height * 0.10:
        return None
    return image[y0:y1, x0:x1, :], {
        "crop_mode": "runtime_panel_rect",
        "crop_bbox": [x0, y0, x1, y1],
        "point_to_pixel_scale": round(scale, 4),
    }


def _candidate_visual_crop(image: np.ndarray, panel_rect_points: dict | None = None) -> tuple[np.ndarray, dict]:
    runtime_crop = _runtime_panel_crop(image, panel_rect_points)
    if runtime_crop is not None:
        return runtime_crop
    height, width, _ = image.shape
    sample_stride = max(1, int(max(height, width) / 420))
    sampled = image[::sample_stride, ::sample_stride, :].astype(np.float32) / 255.0
    brightness = sampled.mean(axis=2)
    max_channel = sampled.max(axis=2)
    min_channel = sampled.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 0.001)
    slab_mask = (brightness > 0.22) & (brightness < 0.78) & (saturation < 0.16)
    ys, xs = np.where(slab_mask)
    mask_coverage = float(slab_mask.mean()) if slab_mask.size else 0.0
    if len(xs) > 0 and mask_coverage > 0.035:
        x0 = max(0, int(xs.min()) * sample_stride - 8)
        x1 = min(width, (int(xs.max()) + 1) * sample_stride + 8)
        y0 = max(0, int(ys.min()) * sample_stride - 8)
        y1 = min(height, (int(ys.max()) + 1) * sample_stride + 8)
        bbox_area = ((x1 - x0) * (y1 - y0)) / max(width * height, 1)
        if (x1 - x0) >= width * 0.18 and (y1 - y0) >= height * 0.12 and bbox_area >= 0.045:
            return image[y0:y1, x0:x1, :], {
                "crop_mode": "detected_frosted_slab",
                "crop_bbox": [x0, y0, x1, y1],
                "slab_mask_coverage": round(mask_coverage, 4),
            }
    x0 = int(width * 0.12)
    x1 = int(width * 0.88)
    y0 = int(height * 0.12)
    y1 = int(height * 0.78)
    return image[y0:y1, x0:x1, :], {
        "crop_mode": "center_fallback",
        "crop_bbox": [x0, y0, x1, y1],
        "slab_mask_coverage": round(mask_coverage, 4),
    }


def _frame_visual_metrics(path: Path, *, panel_rect_points: dict | None = None) -> dict:
    image = _read_png_rgb(path)
    crop, crop_meta = _candidate_visual_crop(image, panel_rect_points)
    if crop.size == 0:
        raise ValueError("empty_visual_crop")
    stride = max(1, int(max(crop.shape[:2]) / 320))
    crop = crop[::stride, ::stride, :].astype(np.float32) / 255.0
    luma = crop[:, :, 0] * 0.2126 + crop[:, :, 1] * 0.7152 + crop[:, :, 2] * 0.0722
    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 0.001)
    panel_material = (luma > 0.22) & (luma < 0.78) & (saturation < 0.16)
    grad_x = np.abs(np.diff(luma, axis=1))
    grad_y = np.abs(np.diff(luma, axis=0))
    edge_density = float(
        (np.count_nonzero(grad_x > 0.055) + np.count_nonzero(grad_y > 0.055))
        / max(grad_x.size + grad_y.size, 1)
    )
    visual = {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "luma_std": round(float(np.std(luma)), 5),
        "edge_density": round(edge_density, 5),
        "panel_material_fraction": round(float(np.mean(panel_material)), 5),
        "saturated_pixel_fraction": round(float(np.mean((saturation > 0.12) & (luma > 0.12))), 5),
        "bright_detail_fraction": round(float(np.mean((luma > 0.68) & (np.std(crop, axis=2) > 0.025))), 5),
        **crop_meta,
    }
    visual["score"] = round(
        visual["luma_std"] * 1.8
        + visual["edge_density"] * 2.4
        + visual["saturated_pixel_fraction"] * 5.0
        + visual["bright_detail_fraction"] * 2.0,
        5,
    )
    material_panel_content = bool(
        visual["panel_material_fraction"] >= _VISUAL_PASS_THRESHOLDS["panel_material_fraction"]
        and visual["saturated_pixel_fraction"] >= _VISUAL_PASS_THRESHOLDS["saturated_pixel_fraction"]
        and visual["edge_density"] >= _VISUAL_PASS_THRESHOLDS["edge_density"]
    )
    colored_graph_content = bool(
        visual["saturated_pixel_fraction"] >= _VISUAL_PASS_THRESHOLDS["graph_saturated_pixel_fraction"]
        and visual["edge_density"] >= _VISUAL_PASS_THRESHOLDS["graph_edge_density"]
    )
    if material_panel_content:
        visual["pass_reason"] = "material_panel_content"
    elif colored_graph_content:
        visual["pass_reason"] = "colored_graph_content"
    else:
        visual["pass_reason"] = None
    visual["passed"] = bool(visual["pass_reason"])
    return visual


def _captured_frame_paths(index_path: Path, payload: dict) -> list[Path]:
    output_dir = index_path.parent
    manifest_path = payload.get("retina_lasso_manifest")
    candidates: list[Path] = []
    if isinstance(manifest_path, str):
        try:
            manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        for key in ("frames", "captures", "images"):
            values = manifest.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        candidates.append(Path(value).expanduser())
                    elif isinstance(value, dict):
                        candidate = value.get("path") or value.get("image") or value.get("file")
                        if isinstance(candidate, str):
                            candidates.append(Path(candidate).expanduser())
    candidates.extend(sorted(output_dir.glob("*.png")))
    seen: set[Path] = set()
    resolved: list[Path] = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate.suffix.lower() == ".png" and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def _visual_content_contract(index_path: Path, payload: dict, *, panel_rect_points: dict | None) -> dict:
    frame_paths = _captured_frame_paths(index_path, payload)
    metrics: list[dict] = []
    errors: list[dict] = []
    for path in frame_paths:
        try:
            metrics.append(_frame_visual_metrics(path, panel_rect_points=panel_rect_points))
        except Exception as exc:
            errors.append({"path": str(path), "error": type(exc).__name__, "detail": str(exc)})
    passed_metrics = [metric for metric in metrics if metric.get("passed") is True]
    tail_count = min(12, max(1, len(metrics) // 4)) if metrics else 0
    settled_tail = metrics[-tail_count:] if tail_count else []
    settled_tail_passed_metrics = [metric for metric in settled_tail if metric.get("passed") is True]
    best = max(metrics, key=lambda metric: metric.get("score", 0.0), default=None)
    passed = bool(settled_tail_passed_metrics)
    if passed:
        failure_reason = None
    elif passed_metrics:
        failure_reason = "settled_tail_lacks_throughglass_content"
    else:
        failure_reason = "captured_pixels_do_not_show_throughglass_content"
    return {
        "classifier_version": VISUAL_CONTENT_CLASSIFIER_VERSION,
        "passed": passed,
        "failure_reason": failure_reason,
        "frame_count": len(frame_paths),
        "frames_analyzed": len(metrics),
        "pass_thresholds": _VISUAL_PASS_THRESHOLDS,
        "settled_tail_policy": "last_min_12_or_quarter_frames_must_contain_content",
        "settled_tail_frame_count": tail_count,
        "settled_tail_pass_count": len(settled_tail_passed_metrics),
        "best_score": best.get("score") if best else None,
        "best_frame": best.get("path") if best else None,
        "best_metrics": best,
        "sample_metrics": metrics[:6],
        "settled_tail_sample_metrics": settled_tail[:6],
        "errors": errors[:6],
    }


def annotate_throughglass_contract(index_path: Path, *, log_paths: list[Path]) -> dict:
    payload = _load_index(index_path)
    existing = payload.get("throughglass_contract")
    if (
        isinstance(existing, dict)
        and existing.get("passed") is True
        and isinstance(existing.get("visual_content"), dict)
        and existing["visual_content"].get("passed") is True
        and existing["visual_content"].get("classifier_version") == VISUAL_CONTENT_CLASSIFIER_VERSION
    ):
        return existing
    contract = _runtime_log_contract(log_paths)
    contract["visual_content"] = _visual_content_contract(
        index_path, payload, panel_rect_points=contract.get("panel_rect_points")
    )
    contract["passed"] = bool(contract.get("passed") is True and contract["visual_content"].get("passed") is True)
    frame_count = payload.get("frame_count")
    if isinstance(frame_count, int):
        contract["frame_count"] = frame_count
    payload["throughglass_contract"] = contract
    _write_index(index_path, payload)
    return contract


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(args.output_root)
    stimulus_mode = bool(args.launch or args.hammer_toggles or args.retarget_during_dismiss_repeats)
    capture_profile = _capture_profile_from_cli(
        args.capture_profile or (DEFAULT_STRESS_CAPTURE_PROFILE if stimulus_mode else DEFAULT_PASSIVE_CAPTURE_PROFILE)
    )

    if stimulus_mode:
        index_path = run_autonomous_hammer_witness(
            trace_path=args.trace,
            output_dir=output_dir,
            repo_root=Path(__file__).resolve().parents[1],
            duration_seconds=args.duration,
            fps=args.fps,
            capture_profile=capture_profile,
            hammer_toggles=args.hammer_toggles,
            toggle_interval_seconds=args.toggle_interval,
            pre_hammer_delay_seconds=args.pre_hammer_delay,
            retarget_during_dismiss_repeats=args.retarget_during_dismiss_repeats,
            open_dwell_seconds=args.open_dwell,
            dismiss_retarget_delay_seconds=args.dismiss_retarget_delay,
            reopen_dwell_seconds=args.reopen_dwell,
            cycle_pause_seconds=args.cycle_pause,
            lane=args.lane,
            diaulos=args.diaulos,
            source_app=args.source_app,
            source_window=args.source_window,
            capture_command=args.capture_command,
            toggle_control_path=args.toggle_control_path,
            launch_target=args.launch_target if args.launch else None,
            launch_wait_seconds=args.launch_wait,
        )
    else:
        index_path = run_witness_window(
            trace_path=args.trace,
            output_dir=output_dir,
            duration_seconds=args.duration,
            fps=args.fps,
            capture_profile=capture_profile,
            lane=args.lane,
            diaulos=args.diaulos,
            source_app=args.source_app,
            source_window=args.source_window,
            capture_command=args.capture_command,
            stimulus={"mode": "passive-throughglass"},
        )

    log_paths = [Path(path).expanduser() for path in args.log_paths] if args.log_paths else list(DEFAULT_LOG_PATHS)
    contract = annotate_throughglass_contract(Path(index_path), log_paths=log_paths)
    print(index_path)
    return 0 if args.allow_unproven or contract.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
