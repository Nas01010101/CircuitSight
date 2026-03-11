"""
QC Report Generator — JSON
Creates structured quality control reports from inspection results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.detector import InferenceResult

logger = logging.getLogger(__name__)


def generate_qc_report(
    results: List[InferenceResult],
    batch_name: str = "inspection",
    save_path: Optional[str] = None,
) -> dict:
    """
    Generate a structured QC report from inference results.

    Args:
        results: List of InferenceResult objects
        batch_name: Name for this inspection batch
        save_path: Optional file path to save the JSON report

    Returns:
        dict: Structured QC report
    """
    if not results:
        report = {
            "batch_name": batch_name,
            "timestamp": datetime.now().isoformat(),
            "n_inspected": 0,
            "summary": {
                "pass": 0,
                "fail": 0,
                "needs_review": 0,
                "total_defects": 0,
                "avg_inference_ms": 0.0,
            },
            "inspections": [],
        }
        logger.warning("No results to report")
        if save_path:
            _save_report(report, save_path)
        return report

    # Aggregate statistics
    verdicts = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}
    total_defects = 0
    total_time = 0.0

    inspections = []
    for r in results:
        verdicts[r.verdict] += 1
        total_defects += r.n_defects
        total_time += r.inference_time_ms

        inspections.append(r.to_dict())

    avg_time = total_time / len(results)

    report = {
        "batch_name": batch_name,
        "timestamp": datetime.now().isoformat(),
        "n_inspected": len(results),
        "summary": {
            "pass": verdicts["PASS"],
            "fail": verdicts["FAIL"],
            "needs_review": verdicts["NEEDS_REVIEW"],
            "total_defects": total_defects,
            "avg_inference_ms": round(avg_time, 2),
            "pass_rate": round(verdicts["PASS"] / len(results) * 100, 1),
        },
        "inspections": inspections,
    }

    if save_path:
        _save_report(report, save_path)

    return report


def _save_report(report: dict, save_path: str) -> None:
    """Save report to JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("QC report saved: %s", path)
