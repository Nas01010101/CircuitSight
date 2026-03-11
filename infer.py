#!/usr/bin/env python3
"""
AIT Visual Inspector -- Inference Script
Run defect detection on images, folders, or videos.

Usage:
    python infer.py --source image.png --weights runs/train/best.pt
    python infer.py --source folder/ --save-json
    python infer.py --source video.mp4 --track
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

from src.models.detector import CircuitSight_Detector
from src.utils.viz import save_annotated

logger = logging.getLogger(__name__)


def process_image(detector: CircuitSight_Detector, source: str, output_dir: Path, save_json: bool) -> dict:
    """Process a single image."""
    result = detector.detect(source, annotate=True)

    # Save annotated image
    stem = Path(source).stem
    if result.annotated_image is not None:
        out_path = save_annotated(result.annotated_image, str(output_dir / f"{stem}_annotated.jpg"))

    # Print result
    verdict_icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "NEEDS_REVIEW": "[REVIEW]"}
    icon = verdict_icon.get(result.verdict, "[--]")
    logger.info(
        "  %s %-30s -> %-14s defects=%d  conf_max=%.3f  time=%.1fms",
        icon, Path(source).name, result.verdict, result.n_defects,
        max((d.confidence for d in result.detections), default=0),
        result.inference_time_ms,
    )

    # Save JSON
    if save_json:
        json_path = output_dir / f"{stem}_result.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result.to_dict()


def process_video(
    detector: CircuitSight_Detector,
    source: str,
    output_dir: Path,
    track: bool = False,
    save_json: bool = False,
) -> None:
    """Process a video file or webcam feed."""
    cap = cv2.VideoCapture(source if source != "0" else 0)
    if not cap.isOpened():
        logger.error("Could not open video: %s", source)
        return

    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    stem = Path(source).stem if source != "0" else "webcam"
    out_path = str(output_dir / f"{stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Tracking setup
    tracker = None
    if track:
        try:
            from src.tracking.tracker import AIT_Tracker
            tracker = AIT_Tracker()
            logger.info("Tracking enabled (ByteTrack)")
        except ImportError:
            logger.warning("Tracking module not available, running detection only")

    logger.info("Processing video: %s", source)
    logger.info("  Resolution: %dx%d @ %.1f FPS", w, h, fps)
    if total_frames > 0:
        logger.info("  Total frames: %d", total_frames)

    frame_idx = 0
    event_log = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run detection
        result = detector.detect(frame, annotate=True)

        # Tracking
        if tracker and result.detections:
            tracked_frame, events = tracker.update(
                frame, result.detections, frame_idx, fps
            )
            if result.annotated_image is not None:
                result.annotated_image = tracked_frame
            event_log.extend(events)

        # Write annotated frame
        if result.annotated_image is not None:
            writer.write(result.annotated_image)

        # Progress
        if frame_idx % 100 == 0 and total_frames > 0:
            pct = frame_idx / total_frames * 100
            elapsed = time.time() - start_time
            est_total = elapsed / (frame_idx / total_frames)
            est_remain = est_total - elapsed
            logger.info(
                "  Frame %d/%d (%.1f%%) -- ETA: %.0fs",
                frame_idx, total_frames, pct, est_remain,
            )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0

    logger.info("Processed %d frames in %.1fs (%.1f FPS)", frame_idx, elapsed, avg_fps)
    logger.info("Output: %s", out_path)

    # Save event log
    if event_log and save_json:
        log_path = output_dir / f"{stem}_events.json"
        with open(log_path, "w") as f:
            json.dump(event_log, f, indent=2)
        logger.info("Event log: %s (%d events)", log_path, len(event_log))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run AIT defect detection inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image path, folder, video file, or '0' for webcam")
    parser.add_argument("--weights", type=str, default=None, help="Model weights path")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config")
    parser.add_argument("--app-config", type=str, default="configs/app.yaml", help="App config")
    parser.add_argument("--output-dir", type=str, default="reports/inference", help="Output directory")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--save-json", action="store_true", help="Save JSON results")
    parser.add_argument("--track", action="store_true", help="Enable object tracking (video)")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    # -- Find weights --
    weights_path = args.weights
    if weights_path is None:
        candidates = list(Path("runs").rglob("best.pt"))
        if candidates:
            weights_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
        else:
            logger.error("No weights found. Provide --weights or run training first.")
            sys.exit(1)

    # -- Setup detector --
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = CircuitSight_Detector.from_config(
        model_config=args.config, app_config=args.app_config)
    detector.load(weights_path)

    if args.conf is not None:
        detector.conf_threshold = args.conf

    source = Path(args.source)

    logger.info("=" * 50)
    logger.info("AIT Visual Inspector -- Inference")
    logger.info("=" * 50)
    logger.info("  Source:  %s", args.source)
    logger.info("  Weights: %s", weights_path)

    # -- Determine source type --
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if args.source == "0":
        # Webcam
        process_video(detector, args.source, output_dir, track=args.track, save_json=args.save_json)

    elif source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        # Video file
        process_video(detector, str(source), output_dir, track=args.track, save_json=args.save_json)

    elif source.is_file():
        # Single image
        process_image(detector, str(source), output_dir, args.save_json)

    elif source.is_dir():
        # Folder of images
        images = sorted(
            list(source.glob("*.png"))
            + list(source.glob("*.jpg"))
            + list(source.glob("*.jpeg"))
        )

        if not images:
            logger.error("No images found in %s", source)
            sys.exit(1)

        logger.info("Processing %d images...", len(images))

        all_results = []
        verdicts = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}

        for img_path in images:
            result = process_image(detector, str(img_path), output_dir, args.save_json)
            all_results.append(result)
            verdicts[result["verdict"]] += 1

        # Summary
        logger.info("-" * 50)
        logger.info("Summary: %d images processed", len(images))
        for v, c in verdicts.items():
            pct = c / len(images) * 100
            logger.info("  %-14s %5d (%.1f%%)", v, c, pct)

        # Save batch JSON
        if args.save_json:
            batch_path = output_dir / "batch_results.json"
            with open(batch_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info("Batch results: %s", batch_path)

    else:
        logger.error("Source not found: %s", args.source)
        sys.exit(1)


if __name__ == "__main__":
    main()
