"""Launch three positioning evals with 10-second offsets, report variance."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval_positioning import run_eval


SCREENSHOT = "tools/positioning_eval/sample_screenshot.png"
CONSTRAINT = "the text content in the terminal — all the readable text on screen"
OFFSETS = [0, 10, 20]


def run_one(idx: int) -> dict:
    print(f"\n--- LAUNCHING RUN {idx + 1} ---")
    try:
        result = run_eval(
            SCREENSHOT,
            CONSTRAINT,
            output_dir=f"/tmp/spoke-positioning-eval/run{idx + 1}",
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        result = {"status": "error", "error": str(e)}
    result["run"] = idx + 1
    return result


def main():
    results = []
    for i in range(3):
        if i > 0:
            print(f"\n--- Waiting 5s before next run ---")
            time.sleep(5)
        results.append(run_one(i))

    results.sort(key=lambda r: r.get("run", 0))

    print("\n\n========== SUMMARY ==========\n")
    for r in results:
        run = r.get("run", "?")
        status = r.get("status", "unknown")
        if status == "ok":
            corners = r["corners"]
            elapsed = r["vlm"]["elapsed_s"]
            area_pct = r["score"]["area_pct"]
            print(f"Run {run}: {' '.join(corners)}  |  {area_pct}% area  |  {elapsed}s")
        else:
            raw = r.get("raw", r.get("error", "?"))
            print(f"Run {run}: {status} — {raw[:80]}")

    # Variance analysis
    ok_runs = [r for r in results if r.get("status") == "ok"]
    if len(ok_runs) > 1:
        areas = [r["score"]["area_pct"] for r in ok_runs]
        times = [r["vlm"]["elapsed_s"] for r in ok_runs]
        corners_set = [tuple(r["corners"]) for r in ok_runs]
        unique_placements = len(set(corners_set))
        print(f"\nArea range: {min(areas):.1f}% — {max(areas):.1f}%")
        print(f"Time range: {min(times):.1f}s — {max(times):.1f}s")
        print(f"Unique placements: {unique_placements}/{len(ok_runs)}")
        if unique_placements == 1:
            print("All runs converged to the same placement.")
        else:
            print("Placements diverged — high variance.")


if __name__ == "__main__":
    main()
