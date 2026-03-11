"""
Metrics Utilities
Precision/recall helpers, confusion matrix rendering, and latency benchmarking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_detection_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5,
    n_classes: int = 2,
) -> Dict:
    """
    Compute detection metrics from predictions and ground truths.

    Args:
        predictions: List of dicts with 'image', 'boxes' (xyxy), 'scores', 'classes'
        ground_truths: List of dicts with 'image', 'boxes' (xyxy), 'classes'
        iou_threshold: IoU threshold for matching
        n_classes: Number of classes

    Returns:
        Dict with metrics: precision, recall, f1, ap per class
    """
    # Aggregate per-class TP/FP/FN
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(n_classes)}

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = np.array(pred.get("boxes", []))
        pred_scores = np.array(pred.get("scores", []))
        pred_classes = np.array(pred.get("classes", []))
        gt_boxes = np.array(gt.get("boxes", []))
        gt_classes = np.array(gt.get("classes", []))

        for cls in range(n_classes):
            p_mask = pred_classes == cls
            g_mask = gt_classes == cls

            p_boxes = pred_boxes[p_mask] if len(pred_boxes) > 0 and p_mask.any() else np.array([])
            p_confs = pred_scores[p_mask] if len(pred_scores) > 0 and p_mask.any() else np.array([])
            g_boxes = gt_boxes[g_mask] if len(gt_boxes) > 0 and g_mask.any() else np.array([])

            # Sort predictions by confidence (descending) for greedy matching
            if len(p_boxes) > 0 and len(p_confs) > 0:
                sort_idx = np.argsort(-p_confs)
                p_boxes = p_boxes[sort_idx]

            matched_gt = set()

            for pb in p_boxes:
                best_iou = 0.0
                best_idx = -1

                for gi, gb in enumerate(g_boxes):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi

                if best_iou >= iou_threshold and best_idx >= 0:
                    class_stats[cls]["tp"] += 1
                    matched_gt.add(best_idx)
                else:
                    class_stats[cls]["fp"] += 1

            class_stats[cls]["fn"] += len(g_boxes) - len(matched_gt)

    # Compute precision, recall, F1
    metrics = {}
    for cls in range(n_classes):
        tp = class_stats[cls]["tp"]
        fp = class_stats[cls]["fp"]
        fn = class_stats[cls]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"class_{cls}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Overall (macro average)
    avg_precision = np.mean([metrics[f"class_{c}"]["precision"] for c in range(n_classes)])
    avg_recall = np.mean([metrics[f"class_{c}"]["recall"] for c in range(n_classes)])
    avg_f1 = np.mean([metrics[f"class_{c}"]["f1"] for c in range(n_classes)])

    metrics["overall"] = {
        "precision": round(float(avg_precision), 4),
        "recall": round(float(avg_recall), 4),
        "f1": round(float(avg_f1), 4),
    }

    return metrics


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def benchmark_latency(
    model,
    sample_image: np.ndarray,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> Dict:
    """
    Benchmark model inference latency.

    Args:
        model: YOLO model instance
        sample_image: Sample image for benchmarking
        n_warmup: Number of warmup runs
        n_runs: Number of benchmark runs

    Returns:
        Dict with latency stats (mean, std, min, max, fps)
    """
    # Warmup
    for _ in range(n_warmup):
        model(sample_image, verbose=False)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(sample_image, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)

    return {
        "mean_ms": round(float(times.mean()), 2),
        "std_ms": round(float(times.std()), 2),
        "min_ms": round(float(times.min()), 2),
        "max_ms": round(float(times.max()), 2),
        "fps": round(float(1000 / times.mean()), 1),
        "n_runs": n_runs,
    }


def save_metrics(metrics: dict, output_path: str) -> str:
    """Save metrics to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    return str(path)
