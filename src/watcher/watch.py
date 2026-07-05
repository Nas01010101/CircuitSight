"""
CircuitSight — Folder Watcher
Monitors an inbox folder for new images and auto-runs inspection, simulating a
factory camera-feed integration:

    - a new image lands in the inbox (dropped by a camera or copied over the network)
    - the watcher waits until the file has finished being written, then runs inference
    - it writes a JSON report + an annotated image to the output folder
    - it moves the original into processed/ (success) or failed/ (error)

Moving handled files out of the inbox is deliberate: it keeps the scan cheap, it
bounds memory (no ever-growing set of seen filenames), it gives an audit trail, and
it means a restart after downtime picks up the backlog instead of dropping it.

Usage:
    python -m src.watcher.watch
    python -m src.watcher.watch --inbox data/inbox --output reports/auto --interval 1.0
    # or: make watch
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
DEFAULT_POLL_INTERVAL = 2.0   # seconds between folder scans
DEFAULT_SETTLE_SECONDS = 1.0  # a file must be untouched this long before we read it


def _unique_dest(dest_dir: Path, name: str) -> Path:
    """A non-clashing path for `name` in dest_dir (append _1, _2, ... if needed)."""
    if not (dest_dir / name).exists():
        return dest_dir / name
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while (dest_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return dest_dir / f"{stem}_{i}{suffix}"


def _archive(src: Path, dest_dir: Path) -> None:
    """Move a handled file into an archive folder without ever clobbering."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(_unique_dest(dest_dir, src.name)))
    except OSError as e:  # e.g. file vanished or a permissions issue
        logger.warning("Could not archive %s: %s", src.name, e)


def process_image(detector, image_path: Path, output_dir: Path) -> dict:
    """Inspect one image and write its report + annotated copy. Returns the result dict.

    Raises on an unreadable image or an inference failure, so the caller can route the
    file to failed/. Kept separate from the loop so it can be unit-tested directly.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("unreadable / not a valid image")

    result = detector.detect(img, annotate=True)
    result.image_path = image_path.name

    output_dir.mkdir(parents=True, exist_ok=True)
    report = result.to_dict()
    with open(output_dir / f"{image_path.stem}_report.json", "w") as rf:
        json.dump(report, rf, indent=2)
    if getattr(result, "annotated_image", None) is not None:
        cv2.imwrite(str(output_dir / f"{image_path.stem}_annotated.png"),
                    result.annotated_image)
    return report


def scan_once(detector, inbox: Path, output: Path, processed_dir: Path,
              failed_dir: Path, settle_seconds: float = DEFAULT_SETTLE_SECONDS):
    """One pass over the inbox. Returns (n_ok, n_failed) for this pass.

    A file still being written (modified within `settle_seconds`) is left for the next
    pass so we never read a half-copied image. Each image is fault-isolated: one bad
    file is logged and moved to failed/, and the scan continues.
    """
    n_ok = n_fail = 0
    for f in sorted(inbox.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if time.time() - f.stat().st_mtime < settle_seconds:
            continue  # still being written; pick it up next pass

        logger.info("New image detected: %s", f.name)
        try:
            report = process_image(detector, f, output)
            logger.info("  Verdict: %s | Defects: %d | Time: %.1f ms",
                        report.get("verdict"), report.get("n_defects", 0),
                        report.get("inference_time_ms", 0.0))
            _archive(f, processed_dir)
            n_ok += 1
        except Exception as e:
            logger.error("Error processing %s: %s", f.name, e)
            _archive(f, failed_dir)
            n_fail += 1
    return n_ok, n_fail


def watch_folder(
    inbox_dir: str = "data/inbox",
    output_dir: str = "reports/auto",
    model_config: str = "configs/model.yaml",
    app_config: str = "configs/app.yaml",
    domain_config: str = "configs/domains/pcb.yaml",
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
):
    """Load the detector once, then scan the inbox forever (polling, cross-platform)."""
    from src.models.detector import CircuitSight_Detector
    from src.utils.weights import find_best_weights

    inbox = Path(inbox_dir)
    output = Path(output_dir)
    processed_dir = inbox.parent / "processed"
    failed_dir = inbox.parent / "failed"
    inbox.mkdir(parents=True, exist_ok=True)

    detector = CircuitSight_Detector.from_config(
        model_config, app_config,
        domain_config=domain_config if Path(domain_config).exists() else None,
    )
    weights = find_best_weights()
    if weights is None:
        logger.error("No model weights found. Train first: make train-pcb")
        return
    detector.load(weights)

    logger.info("=" * 50)
    logger.info("CircuitSight folder watcher started")
    logger.info("  Inbox:     %s", inbox.resolve())
    logger.info("  Output:    %s", output.resolve())
    logger.info("  Processed: %s", processed_dir.resolve())
    logger.info("  Failed:    %s", failed_dir.resolve())
    logger.info("  Watching:  %s", ", ".join(sorted(IMAGE_EXTENSIONS)))
    logger.info("=" * 50)

    total_ok = total_fail = 0
    try:
        while True:
            ok, fail = scan_once(detector, inbox, output, processed_dir,
                                 failed_dir, settle_seconds)
            total_ok += ok
            total_fail += fail
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Watcher stopped. Processed OK: %d | Failed: %d", total_ok, total_fail)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="CircuitSight folder watcher")
    parser.add_argument("--inbox", default="data/inbox", help="Folder to watch")
    parser.add_argument("--output", default="reports/auto", help="Output folder")
    parser.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL,
                        help="Seconds between folder scans")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_SECONDS,
                        help="Seconds a file must be idle before it is read")
    args = parser.parse_args()
    watch_folder(inbox_dir=args.inbox, output_dir=args.output,
                 poll_interval=args.interval, settle_seconds=args.settle)


if __name__ == "__main__":
    main()
