"""
Dataset to YOLO Format Converter
Supports: MVTec AD (masks → YOLO) and PKU-Market-PCB (VOC XML → YOLO).

Usage:
    python -m src.data.convert_to_yolo --dataset pcb
    python -m src.data.convert_to_yolo --dataset mvtec --config configs/data.yaml
"""

import argparse
import logging
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# PCB defect class mapping (PKU-Market-PCB dataset)
PCB_CLASS_MAP = {
    "missing_hole": 0,
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spur": 4,
    "spurious_copper": 5,
}


# ──────────────────────────────────────────────
# VOC XML → YOLO (for PCB dataset)
# ──────────────────────────────────────────────

def parse_voc_xml(xml_path: str) -> list:
    """
    Parse a Pascal VOC XML annotation file.

    Returns:
        List of (class_name, x1, y1, x2, y2) tuples in pixel coordinates.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError):
        logger.warning("Failed to parse XML: %s", xml_path)
        return []

    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bbox_elem = obj.find("bndbox")

        if name_elem is None or bbox_elem is None:
            continue

        class_name = name_elem.text.strip()

        try:
            x1 = float(bbox_elem.find("xmin").text)
            y1 = float(bbox_elem.find("ymin").text)
            x2 = float(bbox_elem.find("xmax").text)
            y2 = float(bbox_elem.find("ymax").text)
        except (AttributeError, ValueError, TypeError):
            logger.warning("Invalid bbox in %s", xml_path)
            continue

        objects.append((class_name, x1, y1, x2, y2))

    return objects


def voc_to_yolo_bbox(x1: float, y1: float, x2: float, y2: float,
                     img_w: int, img_h: int) -> list:
    """Convert VOC (x1, y1, x2, y2) to YOLO (cx, cy, w, h) normalized."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return [cx, cy, w, h]


def convert_pcb_to_yolo(
    raw_dir: str = "data/raw/pcb",
    output_dir: str = "data/processed/pcb_yolo",
    split_ratios: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """
    Convert PKU-Market-PCB dataset (VOC XML) to YOLO format.

    Args:
        raw_dir: Path to raw PCB dataset (with images/ and Annotations/)
        output_dir: Path to output YOLO directory
        split_ratios: Train/val/test split ratios
        seed: Random seed

    Returns:
        dict with conversion statistics
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Find image and annotation directories (handle nested structures)
    img_dir = None
    ann_dir = None

    for candidate in [raw_path, raw_path / "PCB_DATASET"]:
        if (candidate / "images").exists():
            img_dir = candidate / "images"
        if (candidate / "Annotations").exists():
            ann_dir = candidate / "Annotations"

    if img_dir is None or ann_dir is None:
        logger.error("Could not find images/ and Annotations/ in %s", raw_path)
        return {}

    # Collect all image-annotation pairs
    samples = []  # (image_path, annotations_list)

    images = sorted(list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")))
    logger.info("Found %d images in %s", len(images), img_dir)

    for img_path in images:
        # Find matching XML annotation
        xml_name = img_path.stem + ".xml"
        xml_path = None

        # Search in annotation dir (may be nested)
        for candidate in ann_dir.rglob(xml_name):
            xml_path = candidate
            break

        if xml_path is None:
            continue

        # Parse annotations
        objects = parse_voc_xml(str(xml_path))
        if not objects:
            continue

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Convert to YOLO format
        yolo_labels = []
        for class_name, x1, y1, x2, y2 in objects:
            class_id = PCB_CLASS_MAP.get(class_name)
            if class_id is None:
                logger.warning("Unknown class '%s' in %s", class_name, xml_path)
                continue

            cx, cy, bw, bh = voc_to_yolo_bbox(x1, y1, x2, y2, w, h)

            # Skip tiny boxes
            if bw * w < 3 or bh * h < 3:
                continue

            yolo_labels.append([class_id, cx, cy, bw, bh])

        if yolo_labels:
            samples.append((img_path, yolo_labels, (h, w)))

    logger.info("Valid samples with annotations: %d", len(samples))

    # Shuffle and split
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    # Write YOLO files
    stats = {}
    class_counts = {name: 0 for name in PCB_CLASS_MAP}

    for split_name, split_samples in splits.items():
        img_out = out_path / split_name / "images"
        lbl_out = out_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, labels, (h, w) in split_samples:
            out_name = img_path.stem
            dst_img = img_out / f"{out_name}.jpg"
            dst_lbl = lbl_out / f"{out_name}.txt"

            # Copy image
            shutil.copy2(img_path, dst_img)

            # Write label file
            with open(dst_lbl, "w") as f:
                for label in labels:
                    line = f"{int(label[0])} " + " ".join(f"{v:.6f}" for v in label[1:])
                    f.write(line + "\n")

            # Count classes
            for label in labels:
                for name, idx in PCB_CLASS_MAP.items():
                    if int(label[0]) == idx:
                        class_counts[name] += 1

        stats[split_name] = len(split_samples)

    # Generate data.yaml with absolute path
    data_yaml = {
        "path": str(out_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 6,
        "names": {v: k for k, v in PCB_CLASS_MAP.items()},
        "domain": "pcb",
    }

    yaml_path = Path("configs/data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("Updated %s", yaml_path)
    logger.info("Split: train=%d, val=%d, test=%d", stats.get("train", 0), stats.get("val", 0), stats.get("test", 0))
    logger.info("Class counts: %s", class_counts)

    return {"splits": stats, "classes": class_counts, "total": n}


# ──────────────────────────────────────────────
# MVTec AD Mask → YOLO (kept from original)
# ──────────────────────────────────────────────

def mask_to_bboxes(mask_path: str, min_area: int = 100) -> list:
    """
    Convert a binary mask image to YOLO-format bounding boxes.

    Args:
        mask_path: Path to the binary mask image
        min_area: Minimum contour area to consider (filters noise)

    Returns:
        List of [class_id, x_center, y_center, width, height] (normalized)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        bboxes.append([1, x_center, y_center, norm_w, norm_h])

    return bboxes


def write_yolo_label(label_path: Path, bboxes: list) -> None:
    """Write YOLO-format label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for bbox in bboxes:
            line = " ".join(f"{v:.6f}" if i > 0 else str(int(v)) for i, v in enumerate(bbox))
            f.write(line + "\n")


def convert_category(
    raw_dir: Path,
    output_dir: Path,
    category: str,
    split_ratios: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """Convert a single MVTec AD category to YOLO format."""
    cat_dir = raw_dir / category
    if not cat_dir.exists():
        logger.warning("Category directory not found: %s", cat_dir)
        return {}

    random.seed(seed)
    stats = {"train": {"good": 0, "defect": 0}, "val": {"good": 0, "defect": 0}, "test": {"good": 0, "defect": 0}}

    samples = []

    good_train_dir = cat_dir / "train" / "good"
    if good_train_dir.exists():
        for img_path in sorted(good_train_dir.glob("*.png")):
            samples.append((img_path, "good", []))

    good_test_dir = cat_dir / "test" / "good"
    if good_test_dir.exists():
        for img_path in sorted(good_test_dir.glob("*.png")):
            samples.append((img_path, "good", []))

    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    if test_dir.exists():
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue

            defect_type = defect_dir.name
            mask_dir = gt_dir / defect_type

            for img_path in sorted(defect_dir.glob("*.png")):
                mask_name = img_path.stem + "_mask.png"
                mask_path = mask_dir / mask_name

                bboxes = []
                if mask_path.exists():
                    bboxes = mask_to_bboxes(str(mask_path))

                if bboxes:
                    samples.append((img_path, "defect", bboxes))
                else:
                    samples.append((img_path, "good", []))

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label, bboxes in split_samples:
            out_name = f"{category}_{img_path.stem}"
            out_img = img_dir / f"{out_name}.png"
            out_lbl = lbl_dir / f"{out_name}.txt"

            shutil.copy2(img_path, out_img)
            write_yolo_label(out_lbl, bboxes)
            stats[split_name][label] += 1

    return stats


def validate_labels(data_dir: str) -> None:
    """Validate YOLO label files in a processed directory."""
    data_path = Path(data_dir)
    errors = 0
    total = 0

    for split in ["train", "val", "test"]:
        label_dir = data_path / split / "labels"
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            total += 1
            try:
                with open(label_file) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            logger.error("%s:%d — expected 5 values, got %d", label_file.name, line_num, len(parts))
                            errors += 1
                            continue
                        cls = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                        if any(c < 0 or c > 1 for c in coords):
                            logger.error("%s:%d — coords out of [0, 1]", label_file.name, line_num)
                            errors += 1
            except Exception as e:
                logger.error("%s: %s", label_file.name, e)
                errors += 1

    logger.info("Validation: %d files checked, %d errors", total, errors)
    if errors == 0:
        logger.info("All labels valid")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Convert datasets to YOLO format")
    parser.add_argument("--dataset", type=str, default="pcb",
                        choices=["pcb", "mvtec"],
                        help="Dataset to convert (pcb or mvtec)")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Override raw data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits")
    parser.add_argument("--validate", action="store_true",
                        help="Validate labels after conversion")
    args = parser.parse_args()

    if args.dataset == "pcb":
        raw = args.raw_dir or "data/raw/pcb"
        out = args.output_dir or "data/processed/pcb_yolo"
        logger.info("=" * 50)
        logger.info("PKU-Market-PCB → YOLO Conversion")
        logger.info("=" * 50)
        stats = convert_pcb_to_yolo(raw, out, seed=args.seed)
        if stats:
            logger.info("Conversion complete: %d samples", stats.get("total", 0))

    elif args.dataset == "mvtec":
        raw = args.raw_dir or "data/raw/mvtec_ad"
        out = args.output_dir or "data/processed/mvtec_yolo"
        config_path = "configs/data.yaml"

        config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        categories = config.get("mvtec_categories")
        raw_dir = Path(raw)
        output_dir = Path(out)

        if categories is None:
            if raw_dir.exists():
                categories = [d.name for d in sorted(raw_dir.iterdir()) if d.is_dir()]
            else:
                logger.error("Raw directory not found: %s", raw_dir)
                return

        logger.info("=" * 50)
        logger.info("MVTec AD → YOLO Conversion")
        logger.info("=" * 50)

        for cat in tqdm(categories, desc="Converting"):
            convert_category(raw_dir, output_dir, cat, seed=args.seed)

    if args.validate:
        out = args.output_dir or (
            "data/processed/pcb_yolo" if args.dataset == "pcb"
            else "data/processed/mvtec_yolo"
        )
        validate_labels(out)


if __name__ == "__main__":
    main()
