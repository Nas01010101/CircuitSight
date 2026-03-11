"""
AIT Visual Inspector — Folder Watcher
Monitors an inbox folder for new images and auto-runs inspection.

Simulates a factory camera feed integration:
    - New image appears in data/inbox/
    - Watcher detects it, runs inference
    - Writes JSON report + annotated image to reports/auto/

Usage:
    python -m src.watcher.watch
    python -m src.watcher.watch --inbox data/inbox --output reports/auto
    # or: make watch
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
POLL_INTERVAL = 2.0  # seconds


def watch_folder(
    inbox_dir: str = "data/inbox",
    output_dir: str = "reports/auto",
    model_config: str = "configs/model.yaml",
    app_config: str = "configs/app.yaml",
    domain_config: str = "configs/domains/pcb.yaml",
):
    """
    Watch a folder for new images and auto-inspect them.

    Uses polling instead of watchdog for simplicity and cross-platform support.
    """
    from src.models.detector import CircuitSight_Detector

    inbox = Path(inbox_dir)
    output = Path(output_dir)
    inbox.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    # Load detector
    detector = CircuitSight_Detector.from_config(
        model_config, app_config,
        domain_config=domain_config if Path(domain_config).exists() else None,
    )

    # Find weights
    runs_dir = Path("runs")
    loaded = False
    if runs_dir.exists():
        for pt in sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            detector.load(str(pt))
            loaded = True
            break

    if not loaded:
        logger.error("No model weights found. Train first: make train-pcb")
        return

    # Track processed files
    processed = set()

    # Scan for already-existing files
    for f in inbox.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            processed.add(f.name)

    logger.info("=" * 50)
    logger.info("AIT Folder Watcher started")
    logger.info("  Inbox:  %s", inbox.resolve())
    logger.info("  Output: %s", output.resolve())
    logger.info("  Watching for: %s", ", ".join(IMAGE_EXTENSIONS))
    logger.info("=" * 50)

    try:
        while True:
            for f in sorted(inbox.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if f.name in processed:
                    continue

                logger.info("New image detected: %s", f.name)
                processed.add(f.name)

                try:
                    img = cv2.imread(str(f))
                    if img is None:
                        logger.warning("Could not read: %s", f.name)
                        continue

                    result = detector.detect(img, annotate=True)
                    result.image_path = f.name

                    # Save JSON report
                    report_path = output / f"{f.stem}_report.json"
                    with open(report_path, "w") as rf:
                        json.dump(result.to_dict(), rf, indent=2)

                    # Save annotated image
                    if result.annotated_image is not None:
                        ann_path = output / f"{f.stem}_annotated.png"
                        cv2.imwrite(str(ann_path), result.annotated_image)

                    logger.info(
                        "  Verdict: %s | Defects: %d | Time: %.1f ms",
                        result.verdict, result.n_defects, result.inference_time_ms,
                    )

                except Exception as e:
                    logger.error("Error processing %s: %s", f.name, e)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Watcher stopped.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AIT Folder Watcher")
    parser.add_argument("--inbox", default="data/inbox", help="Folder to watch")
    parser.add_argument("--output", default="reports/auto", help="Output folder")
    args = parser.parse_args()

    watch_folder(inbox_dir=args.inbox, output_dir=args.output)


if __name__ == "__main__":
    main()
