#!/usr/bin/env python3
"""Smoke test for scene capture.

Run from Terminal (which has Accessibility permissions):

    cd /private/tmp/spoke-scene-capture
    uv run python scripts/smoke-capture.py

Captures the frontmost app's active window, runs OCR, collects AX hints,
and writes full results to a JSON file. A short summary is printed to
stdout; the full output goes to the file to avoid terminal truncation.

Switch to a browser or other app with visible text before running, then
quickly switch back to Terminal — the script waits 3 seconds before
capturing so you can position the target window.
"""

from __future__ import annotations

import json
import os
import sys
import time


_OUTPUT_DIR = os.path.expanduser(
    "~/Library/Application Support/Spoke/smoke_results"
)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    print("=== Spoke Scene Capture Smoke Test ===")
    print()
    print("You have 3 seconds to switch to the app you want to capture.")
    print("(Or stay in Terminal to capture Terminal itself.)")
    print()

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print()
    print("Capturing...")
    print()

    from spoke.scene_capture import (
        SceneCaptureCache,
        capture_context,
    )
    from spoke.source_ref import RefResolutionError, SourceRef, resolve

    cache = SceneCaptureCache(max_captures=5)

    # ── Capture ──────────────────────────────────────────────
    t0 = time.perf_counter()
    cap = capture_context(scope="active_window", cache=cache)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if cap is None:
        print("FAIL: capture_context returned None")
        sys.exit(1)

    # ── Build result dict ────────────────────────────────────
    result = {
        "capture_ms": round(elapsed_ms),
        "metadata": {
            "scene_ref": cap.scene_ref,
            "scope": cap.scope,
            "app_name": cap.app_name,
            "bundle_id": cap.bundle_id,
            "window_title": cap.window_title,
            "image_size": list(cap.image_size),
            "model_image_size": list(cap.model_image_size),
            "image_path": cap.image_path,
        },
        "ocr": {
            "block_count": len(cap.ocr_blocks),
            "full_text": cap.ocr_text,
            "blocks": [
                {
                    "ref": b.ref,
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": [round(v, 1) for v in b.bbox],
                }
                for b in cap.ocr_blocks
            ],
        },
        "ax_hints": [
            {
                "ref": h.ref,
                "role": h.role,
                "label": h.label,
                "value": h.value,
            }
            for h in cap.ax_hints
        ],
        "verification": {},
    }

    # ── Cache verification ───────────────────────────────────
    cached = cache.get(cap.scene_ref)
    result["verification"]["cache_store_retrieve"] = "OK" if cached is cap else "FAIL"

    if cap.ocr_blocks:
        first_ref = cap.ocr_blocks[0].ref
        resolved = cache.resolve_block(first_ref)
        result["verification"]["block_resolve"] = {
            "ref": first_ref,
            "status": "OK" if resolved == cap.ocr_blocks[0].text else "FAIL",
            "resolved_text": resolved,
        }

    if cap.ax_hints:
        first_hint = cap.ax_hints[0].ref
        resolved = cache.resolve_ax_hint(first_hint)
        expected = cap.ax_hints[0].value or cap.ax_hints[0].label
        result["verification"]["ax_hint_resolve"] = {
            "ref": first_hint,
            "status": "OK" if resolved == expected else "FAIL",
            "resolved_text": resolved,
        }

    # ── Source ref resolution ────────────────────────────────
    ref_checks = {}

    if cap.ocr_blocks:
        ref = SourceRef(kind="scene_block", value=cap.ocr_blocks[0].ref)
        try:
            text = resolve(ref, scene_cache=cache)
            ref_checks["scene_block"] = {"status": "OK", "text": text}
        except RefResolutionError as e:
            ref_checks["scene_block"] = {"status": "FAIL", "error": str(e)}

    try:
        ref = SourceRef(kind="clipboard", value="current")
        text = resolve(ref)
        ref_checks["clipboard"] = {"status": "OK", "text": text}
    except RefResolutionError as e:
        ref_checks["clipboard"] = {"status": "empty_or_unavailable", "error": str(e)}

    ref = SourceRef(kind="literal", value="test literal text")
    text = resolve(ref)
    ref_checks["literal"] = {"status": "OK", "text": text}

    result["verification"]["source_ref_resolution"] = ref_checks

    # ── Write to file ────────────────────────────────────────
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUTPUT_DIR, f"{cap.scene_ref}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ── Print summary to stdout ──────────────────────────────
    print(f"Capture completed in {result['capture_ms']}ms")
    print(f"  app: {cap.app_name} — {cap.window_title}")
    print(f"  size: {cap.image_size[0]}x{cap.image_size[1]} → {cap.model_image_size[0]}x{cap.model_image_size[1]}")
    print(f"  OCR blocks: {len(cap.ocr_blocks)}")
    print(f"  AX hints: {len(cap.ax_hints)}")
    print(f"  cache: {result['verification']['cache_store_retrieve']}")
    print()
    print(f"Full results: {out_path}")
    print(f"Capture image: {cap.image_path}")


if __name__ == "__main__":
    main()
