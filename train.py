#!/usr/bin/env python3
"""
AIT Visual Inspector -- Training Script
Trains YOLOv8 on the prepared dataset with reproducible configs.

Usage:
    python train.py --config configs/model.yaml --data configs/data.yaml
    python train.py --epochs 5 --img 320 --batch 4   # Quick test
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for AIT defect detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Model config YAML")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--img", type=int, default=None, help="Override image size")
    parser.add_argument("--model", type=str, default=None, help="Override model size (e.g., yolov8s)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--name", type=str, default="ait_inspector", help="Run name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, mps)")
    args = parser.parse_args()

    # -- Load config --
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)

    training = model_cfg.get("training", {})
    augmentation = model_cfg.get("augmentation", {})

    # CLI overrides
    model_size = args.model or model_cfg.get("model_size", "yolov8s")
    epochs = args.epochs or training.get("epochs", 100)
    batch_size = args.batch or training.get("batch_size", 16)
    img_size = args.img or training.get("img_size", 640)
    seed = training.get("seed", 42)

    # -- Resolve data config --
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data config not found: %s", args.data)
        logger.info("Run: make prepare-data")
        sys.exit(1)

    # Ensure absolute path in data.yaml
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = data_cfg.get("path", "data/processed/mvtec_yolo")
    if not Path(dataset_root).is_absolute():
        dataset_root = str(Path.cwd() / dataset_root)
        data_cfg["path"] = dataset_root
        # Write back with absolute path
        with open(data_path, "w") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    if not Path(dataset_root).exists():
        logger.error("Dataset not found at: %s", dataset_root)
        logger.info("Run: make prepare-data")
        sys.exit(1)

    # -- Print training config --
    logger.info("=" * 50)
    logger.info("AIT Visual Inspector -- Training")
    logger.info("=" * 50)
    logger.info("  Model:       %s", model_size)
    logger.info("  Epochs:      %d", epochs)
    logger.info("  Batch size:  %d", batch_size)
    logger.info("  Image size:  %d", img_size)
    logger.info("  Seed:        %d", seed)
    logger.info("  Dataset:     %s", dataset_root)
    logger.info("  Device:      %s", args.device or "auto")

    # -- Initialize model --
    from ultralytics import YOLO

    if args.resume:
        logger.info("Resuming from: %s", args.resume)
        model = YOLO(args.resume)
    else:
        # Start from pretrained weights
        weights = f"{model_size}.pt"
        logger.info("Loading pretrained: %s", weights)
        model = YOLO(weights)

    # -- Train --
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        seed=seed,
        name=args.name,
        device=args.device,
        # Learning rate
        lr0=training.get("lr0", 0.01),
        lrf=training.get("lrf", 0.01),
        momentum=training.get("momentum", 0.937),
        weight_decay=training.get("weight_decay", 0.0005),
        warmup_epochs=training.get("warmup_epochs", 3),
        warmup_momentum=training.get("warmup_momentum", 0.8),
        # Early stopping
        patience=training.get("patience", 20),
        # Augmentation
        hsv_h=augmentation.get("hsv_h", 0.015),
        hsv_s=augmentation.get("hsv_s", 0.7),
        hsv_v=augmentation.get("hsv_v", 0.4),
        degrees=augmentation.get("degrees", 10.0),
        translate=augmentation.get("translate", 0.1),
        scale=augmentation.get("scale", 0.5),
        shear=augmentation.get("shear", 2.0),
        flipud=augmentation.get("flipud", 0.5),
        fliplr=augmentation.get("fliplr", 0.5),
        mosaic=augmentation.get("mosaic", 1.0),
        mixup=augmentation.get("mixup", 0.1),
        # Output
        project="runs/train",
        exist_ok=True,
        pretrained=model_cfg.get("pretrained", True),
        verbose=True,
    )

    # -- Summary --
    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else Path("runs/train") / args.name
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    logger.info("=" * 50)
    logger.info("Training Complete")
    logger.info("=" * 50)
    logger.info("  Best weights:  %s", best_weights)
    logger.info("  Last weights:  %s", last_weights)
    logger.info("  Results dir:   %s", save_dir)
    logger.info("  Next steps:")
    logger.info("    python evaluate.py --weights %s --data %s", best_weights, args.data)
    logger.info("    python infer.py --source <image_or_video> --weights %s", best_weights)


if __name__ == "__main__":
    main()
