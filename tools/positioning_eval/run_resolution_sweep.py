"""Sweep image scales to find the sweet spot for flood-grid positioning."""

from __future__ import annotations

import json
import time

from eval_positioning import run_eval


SCREENSHOT = "../../tools/positioning_eval/sample_screenshot.png"
CONSTRAINT = "the text content in the terminal — all the readable text on screen"
SCALES = [1.0, 0.75, 0.50, 0.33]


def main():
    results = []
    for scale in SCALES:
        print(f"\n{'='*60}")
        print(f"  SCALE: {scale} ({int(3456 * scale)}x{int(2234 * scale)})")
        print(f"{'='*60}")

        try:
            result = run_eval(
                SCREENSHOT,
                CONSTRAINT,
                output_dir=f"/tmp/spoke-positioning-eval/scale-{int(scale * 100)}",
                scale=scale,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"status": "error", "error": str(e)}

        result["scale"] = scale
        result["resolution"] = f"{int(3456 * scale)}x{int(2234 * scale)}"
        results.append(result)

        if scale != SCALES[-1]:
            print("\n--- Waiting 3s ---")
            time.sleep(3)

    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}\n")

    for r in results:
        scale = r.get("scale", "?")
        res = r.get("resolution", "?")
        status = r.get("status", "?")
        if status == "ok":
            cells = r.get("cells_claimed", 0)
            rect = r.get("rect_grid", "?")
            area = r["score"]["area_pct"]
            elapsed = r["vlm"]["elapsed_s"]
            print(f"  {scale:.0%} ({res}): {cells} cells -> {rect} ({area}% area) in {elapsed}s")
        else:
            err = r.get("error", r.get("raw", "?"))
            print(f"  {scale:.0%} ({res}): {status} — {str(err)[:60]}")


if __name__ == "__main__":
    main()
