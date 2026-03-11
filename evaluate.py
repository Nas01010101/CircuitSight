#!/usr/bin/env python3
"""
AIT Visual Inspector -- Evaluation Script
Runs comprehensive evaluation: mAP, precision/recall, confusion matrix, latency.

Usage:
    python evaluate.py --weights runs/train/best.pt --data configs/data.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from src.models.detector import CircuitSight_Detector
from src.utils.metrics import benchmark_latency, save_metrics
from src.utils.viz import create_failure_gallery, save_annotated

logger = logging.getLogger(__name__)



def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
) -> str:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate AIT Visual Inspector model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=str, default=None, help="Trained weights path")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config YAML")
    parser.add_argument("--app-config", type=str, default="configs/app.yaml", help="App config YAML")
    parser.add_argument("--output-dir", type=str, default="reports/evaluation", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--n-failures", type=int, default=16, help="Number of failure cases to save")
    args = parser.parse_args()

    # -- Find weights --
    weights_path = args.weights
    if weights_path is None:
        # Auto-discover latest best weights
        candidates = list(Path("runs").rglob("best.pt"))
        if candidates:
            weights_path = str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])
            logger.info("Auto-discovered weights: %s", weights_path)
        else:
            logger.error("No weights found. Provide --weights or run training first.")
            sys.exit(1)

    # -- Setup --
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = CircuitSight_Detector.from_config(args.config, args.app_config)
    detector.load(weights_path)
    detector.conf_threshold = args.conf
    detector.iou_threshold = args.iou

    # -- Load test data --
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg["path"])
    test_img_dir = dataset_root / data_cfg.get("test", "test/images")
    test_lbl_dir = dataset_root / "test" / "labels"

    # Read class names from config
    class_names = data_cfg.get("names", {0: "good", 1: "defect"})
    defect_class_id = None
    for cls_id, cls_name in class_names.items():
        if cls_name == "defect":
            defect_class_id = int(cls_id)
            break
    if defect_class_id is None:
        defect_class_id = 1
        logger.warning("Could not find 'defect' class in data config, defaulting to class_id=1")

    if not test_img_dir.exists():
        logger.error("Test images not found: %s", test_img_dir)
        sys.exit(1)

    test_images = sorted(
        list(test_img_dir.glob("*.png"))
        + list(test_img_dir.glob("*.jpg"))
        + list(test_img_dir.glob("*.jpeg"))
    )

    logger.info("=" * 50)
    logger.info("AIT Visual Inspector -- Evaluation")
    logger.info("=" * 50)
    logger.info("  Weights:     %s", weights_path)
    logger.info("  Test images: %d", len(test_images))
    logger.info("  Conf:        %.2f", args.conf)
    logger.info("  IoU:         %.2f", args.iou)

    # -- Run YOLO validation (built-in mAP) --
    logger.info("Running YOLO validation (mAP, per-class metrics)...")
    from ultralytics import YOLO

    model = YOLO(weights_path)
    val_results = model.val(
        data=str(args.data),
        split="test",
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    # Extract metrics from YOLO val
    yolo_metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
    }

    # Per-class metrics
    if hasattr(val_results.box, "ap_class_index"):
        for i, cls_idx in enumerate(val_results.box.ap_class_index):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            yolo_metrics[f"{cls_name}_AP50"] = round(float(val_results.box.ap50[i]), 4)
            yolo_metrics[f"{cls_name}_precision"] = round(float(val_results.box.p[i]), 4)
            yolo_metrics[f"{cls_name}_recall"] = round(float(val_results.box.r[i]), 4)

    logger.info("  mAP@50:     %.4f", yolo_metrics["mAP50"])
    logger.info("  mAP@50-95:  %.4f", yolo_metrics["mAP50-95"])
    logger.info("  Precision:  %.4f", yolo_metrics["precision"])
    logger.info("  Recall:     %.4f", yolo_metrics["recall"])

    # -- Run custom inference for failure analysis --
    logger.info("Analyzing %d test images for failure cases...", len(test_images))

    failures = []  # (confidence_gap, result)
    all_results = []
    verdicts = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}

    for img_path in test_images:
        result = detector.detect(str(img_path), annotate=True)
        all_results.append(result)
        verdicts[result.verdict] += 1

        # Check if this is a failure case (wrong verdict)
        label_path = test_lbl_dir / f"{img_path.stem}.txt"
        has_gt_defect = False
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == defect_class_id:
                        has_gt_defect = True
                        break

        # Failure: missed defect or false alarm
        is_failure = False
        if has_gt_defect and result.verdict == "PASS":
            is_failure = True  # Missed defect
        elif not has_gt_defect and result.verdict == "FAIL":
            is_failure = True  # False alarm

        if is_failure and result.annotated_image is not None:
            max_conf = max((d.confidence for d in result.detections), default=0)
            failures.append((max_conf, result))

    # -- Save failure gallery --
    failures.sort(key=lambda x: x[0], reverse=True)
    top_failures = failures[: args.n_failures]

    if top_failures:
        failure_images = [f[1].annotated_image for f in top_failures]
        failure_labels = [
            f"{Path(f[1].image_path).stem} ({f[1].verdict})"
            for f in top_failures
        ]

        gallery = create_failure_gallery(failure_images, failure_labels, cols=4)
        gallery_path = save_annotated(gallery, str(output_dir / "failure_gallery.png"))
        logger.info("Failure gallery: %s (%d cases)", gallery_path, len(top_failures))

    # -- Confusion matrix --
    # Build image-level confusion matrix (PASS vs FAIL/REVIEW)
    cm = np.zeros((2, 2), dtype=int)
    for result in all_results:
        label_path = test_lbl_dir / f"{Path(result.image_path).stem}.txt"
        has_gt = False
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == defect_class_id:
                        has_gt = True
                        break

        actual = 1 if has_gt else 0
        predicted = 1 if result.verdict in ["FAIL", "NEEDS_REVIEW"] else 0
        cm[actual, predicted] += 1

    cm_path = plot_confusion_matrix(
        cm, ["Normal", "Defective"], str(output_dir / "confusion_matrix.png")
    )
    logger.info("Confusion matrix: %s", cm_path)

    # -- Latency benchmark --
    latency_metrics = {}
    if args.benchmark and test_images:
        logger.info("Running latency benchmark...")
        sample = cv2.imread(str(test_images[0]))
        latency_metrics = benchmark_latency(model, sample)
        logger.info("Latency: %.1f ms +/- %.1f ms", latency_metrics["mean_ms"], latency_metrics["std_ms"])
        logger.info("FPS: %.0f", latency_metrics["fps"])

    # -- Verdict distribution --
    logger.info("Verdict distribution:")
    for v, c in verdicts.items():
        pct = c / len(all_results) * 100 if all_results else 0
        logger.info("  %-14s %5d (%.1f%%)", v, c, pct)

    # -- Save full report --
    full_report = {
        "model": {
            "weights": weights_path,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
        },
        "dataset": {
            "test_images": len(test_images),
            "dataset_root": str(dataset_root),
        },
        "yolo_metrics": yolo_metrics,
        "verdict_distribution": verdicts,
        "confusion_matrix": cm.tolist(),
        "n_failures": len(failures),
        "latency": latency_metrics,
    }

    report_path = save_metrics(full_report, str(output_dir / "evaluation_report.json"))
    logger.info("Full report: %s", report_path)

    # -- Summary --
    logger.info("=" * 50)
    logger.info("Evaluation Complete")
    logger.info("=" * 50)
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Files generated:")
    for f in sorted(output_dir.glob("*")):
        logger.info("    -> %s", f.name)


if __name__ == "__main__":
    main()
