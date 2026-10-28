"""Pre-render every stimulus as static WebP files for faster serving.

Reads the active stimulus manifest (the demo set by default), renders each
trial's normal and blur views, and saves them under prerendered/. The Flask
app serves these files when present and falls back to on-the-fly rendering
otherwise.

Usage:
    python prerender_stimuli.py
    python prerender_stimuli.py --force   # regenerate existing files
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from experiment_core import BASE_DIR, STIM_CSV
from stimulus_rendering import render_focus_image

OUTPUT_DIR = BASE_DIR / "prerendered"
RENDER_SIZE = (1280, 1280)
WEBP_QUALITY = 86


def render_one(row: dict, mode: str, *, force: bool = False) -> tuple[str, str, float]:
    """Render a single stimulus and save as WebP. Returns (stim_id, mode, seconds)."""
    stim_id = row["stim_id"]
    source = row["source_image_path"]
    annotation = row["annotation_path"]
    label = row["target_label"]

    out_name = f"{stim_id}_{mode}.webp"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists() and not force:
        return stim_id, mode, 0.0

    start = time.time()
    img = render_focus_image(
        source, annotation, label,
        mode=mode, output_size=RENDER_SIZE,
    )
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=WEBP_QUALITY, method=4)
    out_path.write_bytes(buf.getvalue())
    elapsed = time.time() - start
    return stim_id, mode, elapsed


def render_task(args: tuple[dict, str, bool]) -> tuple[str, str, float]:
    return render_one(args[0], args[1], force=args[2])


def main():
    parser = argparse.ArgumentParser(description="Pre-render stimulus images")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel render workers")
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist")
    args = parser.parse_args()

    if not STIM_CSV.exists():
        print(f"ERROR: {STIM_CSV} not found.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with STIM_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("render_mode") == "labelme_focus"]

    print(f"Found {len(rows)} labelme_focus stimuli to render.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Size: {RENDER_SIZE[0]}x{RENDER_SIZE[1]} WebP q{WEBP_QUALITY}")
    print()

    tasks = []
    for row in rows:
        for mode in ("normal", "blur"):
            tasks.append((row, mode, args.force))

    total = len(tasks)
    done = 0
    skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_task, task): task for task in tasks}
        for future in as_completed(futures):
            stim_id, mode, elapsed = future.result()
            done += 1
            if elapsed == 0:
                skipped += 1
            if done % 100 == 0 or done == total:
                rate = done / (time.time() - t0)
                print(f"  [{done}/{total}] {rate:.1f} img/s (skipped {skipped})")

    elapsed_total = time.time() - t0
    rendered = done - skipped
    print(f"\nDone: {rendered} rendered, {skipped} skipped in {elapsed_total:.1f}s")

    # Verify
    expected = total
    actual = len(list(OUTPUT_DIR.glob("*.webp")))
    print(f"Files on disk: {actual}/{expected}")


if __name__ == "__main__":
    main()
