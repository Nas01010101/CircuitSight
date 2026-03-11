"""
PCB-Specific Image Augmentation & Preprocessing
CLAHE contrast enhancement, green channel emphasis, and synthetic defect generation.

Usage:
    from src.data.augment import augment_image, apply_clahe
"""

import logging
import random
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast to make defects (scratches, traces, holes)
    more visible against the PCB surface.

    Args:
        img: BGR input image
        clip_limit: Contrast limiting threshold
        grid_size: Size of the grid for histogram equalization

    Returns:
        Enhanced BGR image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def enhance_green_channel(img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    """
    Emphasize the green channel for PCB board vs defect separation.

    PCBs are predominantly green — boosting the green channel helps
    distinguish board surface from copper traces and defects.

    Args:
        img: BGR input image
        factor: Green channel boost factor

    Returns:
        Enhanced BGR image
    """
    result = img.copy().astype(np.float32)
    result[:, :, 1] = np.clip(result[:, :, 1] * factor, 0, 255)
    return result.astype(np.uint8)


def preprocess_pcb(
    img: np.ndarray,
    apply_clahe_flag: bool = True,
    apply_green_boost: bool = False,
    clahe_clip: float = 2.0,
) -> np.ndarray:
    """
    Full PCB preprocessing pipeline.

    Args:
        img: BGR input image
        apply_clahe_flag: Whether to apply CLAHE
        apply_green_boost: Whether to boost green channel
        clahe_clip: CLAHE clip limit

    Returns:
        Preprocessed BGR image
    """
    result = img.copy()

    if apply_clahe_flag:
        result = apply_clahe(result, clip_limit=clahe_clip)

    if apply_green_boost:
        result = enhance_green_channel(result)

    return result


# ──────────────────────────────────────────────
# Synthetic defect generation (for training augmentation)
# ──────────────────────────────────────────────

def _add_synthetic_scratch(
    img: np.ndarray,
    min_length: float = 0.1,
    max_length: float = 0.4,
) -> Tuple[np.ndarray, list]:
    """Add a synthetic scratch line to an image."""
    result = img.copy()
    h, w = result.shape[:2]

    # Random start/end points
    x1 = random.randint(int(w * 0.1), int(w * 0.9))
    y1 = random.randint(int(h * 0.1), int(h * 0.9))

    length = random.uniform(min_length, max_length) * min(h, w)
    angle = random.uniform(0, 2 * 3.14159)

    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))

    # Clamp
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    # Draw scratch
    thickness = random.randint(1, 3)
    color = (
        random.randint(180, 255),
        random.randint(180, 255),
        random.randint(180, 255),
    )
    cv2.line(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Compute bounding box
    bx1, by1 = min(x1, x2), min(y1, y2)
    bx2, by2 = max(x1, x2), max(y1, y2)

    # Pad the bbox slightly
    pad = thickness * 2
    bx1 = max(0, bx1 - pad)
    by1 = max(0, by1 - pad)
    bx2 = min(w, bx2 + pad)
    by2 = min(h, by2 + pad)

    cx = ((bx1 + bx2) / 2) / w
    cy = ((by1 + by2) / 2) / h
    bw = (bx2 - bx1) / w
    bh = (by2 - by1) / h

    bbox = [1, cx, cy, bw, bh]  # class_id=1 for generic defect

    return result, [bbox]


def _add_synthetic_stain(img: np.ndarray) -> Tuple[np.ndarray, list]:
    """Add a synthetic contamination stain to an image."""
    result = img.copy()
    h, w = result.shape[:2]

    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    radius = random.randint(int(min(h, w) * 0.03), int(min(h, w) * 0.1))

    # Random dark stain
    overlay = result.copy()
    color = (
        random.randint(40, 100),
        random.randint(40, 100),
        random.randint(40, 100),
    )
    cv2.circle(overlay, (cx, cy), radius, color, -1)
    alpha = random.uniform(0.3, 0.7)
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # Bounding box
    bx1 = max(0, cx - radius)
    by1 = max(0, cy - radius)
    bx2 = min(w, cx + radius)
    by2 = min(h, cy + radius)

    bbox_cx = ((bx1 + bx2) / 2) / w
    bbox_cy = ((by1 + by2) / 2) / h
    bbox_w = (bx2 - bx1) / w
    bbox_h = (by2 - by1) / h

    bbox = [1, bbox_cx, bbox_cy, bbox_w, bbox_h]

    return result, [bbox]


def augment_image(
    img: np.ndarray,
    n_defects: int = None,
) -> Tuple[np.ndarray, List[list]]:
    """
    Apply random synthetic defects to an image for training augmentation.

    Args:
        img: BGR input image
        n_defects: Number of defects to add (None = random 1-3)

    Returns:
        (augmented_image, list_of_yolo_bboxes)
    """
    result = img.copy()
    all_bboxes = []

    if n_defects is None:
        n_defects = random.randint(1, 3)

    for _ in range(n_defects):
        if random.random() < 0.6:
            result, bboxes = _add_synthetic_scratch(result)
        else:
            result, bboxes = _add_synthetic_stain(result)
        all_bboxes.extend(bboxes)

    return result, all_bboxes
