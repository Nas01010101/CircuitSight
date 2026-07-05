#!/usr/bin/env python3
"""Autoresearch target: train one detector config, report val mAP@50-95.

The autoresearch harness runs this file under a wall-clock budget and reads the
scalar from $AUTORESEARCH_RESULTS. Only the EXPERIMENT dict (and any model
surgery it references) is the editable surface; data splits and the val
protocol are the judge and stay fixed.
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EXPERIMENT = {
    "name": "e01_baseline_v8s_640",
    "model": "yolov8s.pt",          # pretrained COCO init
    "imgsz": 640,
    "epochs": 60,
    "batch": 8,
    "patience": 30,
    # Repo-recipe augmentation (configs/model.yaml values) = honest baseline.
    "overrides": {
        "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 10.0, "translate": 0.1, "scale": 0.5, "shear": 2.0,
        "flipud": 0.5, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
    },
}

DATA = {
    "path": str(REPO / "data/processed/pcb_yolo"),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 6,
    "names": {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit",
              3: "short", 4: "spur", 5: "spurious_copper"},
}


def main() -> None:
    import yaml
    from ultralytics import YOLO

    exp_dir = Path(__file__).resolve().parent
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    data_yaml = runs_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(DATA, f, default_flow_style=False, sort_keys=False)

    model = YOLO(EXPERIMENT["model"])
    model.train(
        data=str(data_yaml),
        epochs=EXPERIMENT["epochs"],
        batch=EXPERIMENT["batch"],
        imgsz=EXPERIMENT["imgsz"],
        patience=EXPERIMENT["patience"],
        seed=42,
        device="mps",
        project=str(runs_dir),
        name=EXPERIMENT["name"],
        exist_ok=True,
        verbose=False,
        plots=False,
        **EXPERIMENT["overrides"],
    )

    best = runs_dir / EXPERIMENT["name"] / "weights" / "best.pt"
    val = YOLO(str(best)).val(
        data=str(data_yaml), split="val", device="mps", verbose=False, plots=False
    )

    out = {
        "metric": round(float(val.box.map), 5),        # val mAP@50-95 (judged)
        "map50": round(float(val.box.map50), 5),
        "precision": round(float(val.box.mp), 5),
        "recall": round(float(val.box.mr), 5),
        "experiment": EXPERIMENT["name"],
        "config": {k: v for k, v in EXPERIMENT.items() if k != "overrides"},
        "best_weights": str(best),
    }

    results_path = os.environ.get("AUTORESEARCH_RESULTS")
    if not results_path:
        print("AUTORESEARCH_RESULTS not set", file=sys.stderr)
        sys.exit(2)
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    # Sidecar copy so each experiment's full numbers survive for the paper.
    with open(runs_dir / EXPERIMENT["name"] / "result.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
