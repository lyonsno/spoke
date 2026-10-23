"""Native Metal parity and SCK capture-age probe; run through Greenroom."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def summarize_capture(rows, expected_size):
    complete = [row for row in rows if row.get("status") == 0]
    if not complete:
        raise ValueError("No complete capture frames")
    if any(row.get("size") != list(expected_size) for row in complete):
        raise ValueError("Effective capture dimensions differ from requested dimensions")
    ages = []
    for row in complete:
        source, received = row.get("source_seconds"), row.get("received_seconds")
        if source is None or received is None or not 0 < source <= received:
            raise ValueError("Capture source clock missing or invalid")
        ages.append((received - source) * 1000)
    ordered = sorted(ages)
    return {"complete_frames": len(complete), "source_to_callback_ms": {
        "min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]},
        "claim": "Observed source-to-callback delay; excludes render, presentation and scanout"}


def _pump_until(predicate):
    from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
    while not predicate():
        NSRunLoop.currentRunLoop().runMode_beforeDate_(
            NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.01))


def _native_parity(root, baseline, out, report):
    import objc
    import Quartz as Q
    from spoke import metal_warp

    spec = importlib.util.spec_from_file_location("spoke._liquid_baseline", baseline / "spoke/metal_warp.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    modules = {"baseline": original, "candidate": metal_warp}
    width, height = 256, 128
    status, pixel_buffer = Q.CVPixelBufferCreate(None, width, height, 1111970369,
        {Q.kCVPixelBufferIOSurfacePropertiesKey: {}}, None)
    if status or pixel_buffer is None:
        raise RuntimeError(f"CVPixelBufferCreate failed: {status}")
    Q.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
    try:
        pitch = Q.CVPixelBufferGetBytesPerRow(pixel_buffer)
        pixels = bytearray(pitch * height)
        for y in range(height):
            for x in range(width):
                offset = y * pitch + x * 4
                pixels[offset:offset + 4] = bytes((x % 256, y * 2 % 256, (x ^ y) % 256, 255))
        address = Q.CVPixelBufferGetBaseAddress(pixel_buffer)
        address.as_buffer(len(pixels))[:] = pixels
    finally:
        Q.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
    (out / "input.bgra").write_bytes(pixels)
    surface = Q.CVPixelBufferGetIOSurface(pixel_buffer)
    config = {"center_x": 90.0, "center_y": 64.0, "content_width_points": 60.0,
              "content_height_points": 30.0, "corner_radius_points": 12.0,
              "core_magnification": 1.1, "temporal_blend": 1.0}
    cases = {"crisp": {**config, "mip_blur_strength": 0.0},
             "blurred": {**config, "mip_blur_strength": 1.0},
             "two_blurred": [{**config, "mip_blur_strength": 1.0},
                             {**config, "center_x": 170.0, "mip_blur_strength": 1.0}],
             "skipped_prefix": [{**config, "center_x": -1000.0, "mip_blur_strength": 1.0},
                                {**config, "mip_blur_strength": 1.0}]}
    results = {}
    for case, shell in cases.items():
        outputs = {}
        for label, module in modules.items():
            pipeline = module.MetalWarpPipeline()
            descriptor = objc.lookUpClass("MTLTextureDescriptor").texture2DDescriptorWithPixelFormat_width_height_mipmapped_(80, width, height, False)
            descriptor.setStorageMode_(0)
            descriptor.setUsage_(3)
            texture = pipeline.device.newTextureWithDescriptor_(descriptor)
            buffers = []
            real_queue = pipeline._command_queue

            class Command:
                def __init__(self):
                    self.native = real_queue.commandBuffer()
                def __getattr__(self, name):
                    return getattr(self.native, name)
                def presentDrawable_(self, drawable):
                    pass
                def commit(self):
                    self.native.commit()
                    buffers.append(self.native)

            class Queue:
                def commandBuffer(self):
                    return Command()

            class Drawable:
                def texture(self):
                    return texture

            pipeline._command_queue = Queue()
            if not pipeline.warp_to_drawable(surface, Drawable(), width=width, height=height, shell_config=shell):
                raise RuntimeError(f"{label}/{case}: no GPU submission")
            for buffer in buffers:
                buffer.waitUntilCompleted()
                if int(buffer.status()) != 4 or buffer.error() is not None:
                    raise RuntimeError(f"{label}/{case}: GPU command failed: {buffer.error()}")
            storage = ctypes.create_string_buffer(width * height * 4)
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(ctypes.addressof(storage), width * 4, ((0, 0, 0), (width, height, 1)), 0)
            data = storage.raw
            (out / f"{case}-{label}.bgra").write_bytes(data)
            outputs[label] = bytes(data)
        results[case] = {"equal": outputs["baseline"] == outputs["candidate"],
                         "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()}}
        report["parity"] = results
        if not results[case]["equal"]:
            raise AssertionError(f"Native output changed: {case}")


def _capture(out, seconds, fps, report):
    import objc
    from AppKit import NSScreen
    from Quartz import CACurrentMediaTime
    from spoke import backdrop_stream as backdrop
    from spoke.presentation_timing import capture_display_seconds

    bridge = backdrop._load_screencapturekit_bridge()
    if bridge is None:
        raise RuntimeError("Installed ScreenCaptureKit bridge unavailable")
    backdrop._build_stream_output_class()
    content_result = []

    def got_content(content, *args):
        content_result.append((content, args[0] if args else None))

    bridge["SCShareableContent"].getShareableContentWithCompletionHandler_(got_content)
    _pump_until(lambda: bool(content_result))
    content, error = content_result[0]
    if content is None or error is not None:
        raise RuntimeError(f"Shareable content failed: {error}")
    screen = NSScreen.mainScreen()
    display_id = backdrop._screen_display_id(screen)
    display = next(d for d in content.displays() if int(d.displayID()) == int(display_id))
    scale = float(screen.backingScaleFactor())
    frame = display.frame()
    runs = []
    report["captures"] = runs
    for name, rect in (("localized", ((0.0, 0.0), (320.0, 180.0))),
                       ("full_display", ((0.0, 0.0), (frame.size.width, frame.size.height)))):
        width, height = int(rect[1][0] * scale), int(rect[1][1] * scale)
        config = bridge["SCStreamConfiguration"].alloc().init()
        config.setSourceRect_(rect)
        config.setWidth_(width)
        config.setHeight_(height)
        config.setPixelFormat_(1111970369)
        config.setQueueDepth_(8)
        config.setShowsCursor_(False)
        config.setMinimumFrameInterval_((1, fps, 0, 0))
        if hasattr(config, "setContentScale_"):
            config.setContentScale_(scale)
        rows, callback_errors = [], []
        run = {"mode": name, "requested_rect": rect, "effective_rect": str(config.sourceRect()),
               "requested_size": [width, height], "fps_requested": fps,
               "effective_interval": str(config.minimumFrameInterval()), "queue_depth": int(config.queueDepth()),
               "display_id": int(display_id), "rows": rows, "callback_errors": callback_errors}
        runs.append(run)

        class Receiver:
            def _consume_sample_buffer(self, sample, output_type):
                received = float(CACurrentMediaTime())
                try:
                    if hasattr(sample, "pointerAsInteger"):
                        sample = objc.objc_object(c_void_p=sample.pointerAsInteger)
                    attachments = bridge["CMSampleBufferGetSampleAttachmentsArray"](sample, False)
                    item = attachments[0] if attachments else {}
                    status = int(item.get(bridge.get("SCStreamFrameInfoStatus"), -1))
                    source = capture_display_seconds(sample, bridge)
                    raw = bridge["_cm_lib"].CMSampleBufferGetImageBuffer(objc.pyobjc_id(sample))
                    size = None
                    if raw:
                        cv = bridge["_cv_lib"]
                        for key in ("CVPixelBufferGetWidth", "CVPixelBufferGetHeight"):
                            fn = getattr(cv, key)
                            fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_size_t
                        size = [int(cv.CVPixelBufferGetWidth(raw)), int(cv.CVPixelBufferGetHeight(raw))]
                    rows.append({"status": status, "size": size, "source_seconds": source,
                                 "received_seconds": received, "output_type": int(output_type),
                                 "attachments": {str(k): str(v) for k, v in item.items()}})
                except BaseException:
                    callback_errors.append(traceback.format_exc())

        receiver = Receiver()
        output = backdrop._ScreenCaptureKitStreamOutput.alloc().initWithRenderer_(receiver)
        filter_ = bridge["SCContentFilter"].alloc().initWithDisplay_excludingWindows_(display, [])
        stream = bridge["SCStream"].alloc().initWithFilter_configuration_delegate_(filter_, config, None)
        result = stream.addStreamOutput_type_sampleHandlerQueue_error_(output, bridge["SCStreamOutputTypeScreen"], backdrop._make_stream_handler_queue("spoke.liquid.capture-age"), None)
        if not (result[0] if isinstance(result, tuple) else result):
            raise RuntimeError(f"Stream output rejected: {result}")
        started, stopped = [], []
        try:
            stream.startCaptureWithCompletionHandler_(lambda *args: started.append(args))
            _pump_until(lambda: bool(started))
            if started[0] and started[0][0] is not None:
                raise RuntimeError(f"Capture start failed: {started}")
            end = time.monotonic() + seconds
            _pump_until(lambda: time.monotonic() >= end)
        finally:
            stream.stopCaptureWithCompletionHandler_(lambda *args: stopped.append(args))
            _pump_until(lambda: bool(stopped))
            run["stop_result"] = str(stopped)
        if callback_errors:
            raise RuntimeError("Capture callback failed")
        if stopped[0] and stopped[0][0] is not None:
            raise RuntimeError("Capture stop failed")
        run["summary"] = summarize_capture(rows, (width, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=120)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--parity-only", action="store_true")
    mode.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"phase": "identity", "passed": False, "argv": sys.argv,
              "capture_instrumentation": "Python callback diagnostic; callback overhead is included",
              "pixel_authority": "Native parity uses deterministic retained BGRA; capture probe judges timestamps only",
              "repo_root": str(args.repo_root.resolve()), "baseline_root": str(args.baseline_root.resolve())}
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        sys.path.insert(0, str(args.repo_root.resolve()))
        from spoke import metal_warp
        if Path(metal_warp.__file__).resolve() != args.repo_root.resolve() / "spoke/metal_warp.py":
            raise RuntimeError("Wrong source route")
        report["sources"] = {label: {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "metal_sha256": hashlib.sha256((root / "spoke/metal_warp.py").read_bytes()).hexdigest()
        } for label, root in (("candidate", args.repo_root), ("baseline", args.baseline_root))}
        if not args.capture_only:
            report["phase"] = "native_metal_parity"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            _native_parity(args.repo_root, args.baseline_root, args.out, report)
        if not args.parity_only:
            report["phase"] = "capture_latency"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            _capture(args.out, args.capture_seconds, args.fps, report)
        report.update(phase="complete", passed=True)
        return 0
    except BaseException:
        report["error"] = traceback.format_exc()
        return 1
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
