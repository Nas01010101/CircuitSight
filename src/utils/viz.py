"""
Visualization Utilities
Side-by-side comparisons, failure galleries, and annotated image saving.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)



def create_failure_gallery(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    cols: int = 4,
    cell_size: int = 300,
) -> np.ndarray:
    """Create a grid gallery from failure case images."""
    if not images:
        # Return a placeholder image
        placeholder = np.zeros((cell_size, cell_size * cols, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "No failure cases", (10, cell_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA,
        )
        return placeholder

    n = len(images)
    rows = (n + cols - 1) // cols

    gallery = np.zeros((rows * (cell_size + 30), cols * cell_size, 3), dtype=np.uint8)
    gallery[:] = (20, 20, 20)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y0 = r * (cell_size + 30)
        x0 = c * cell_size

        # Resize image to cell
        resized = cv2.resize(img, (cell_size, cell_size))
        gallery[y0 : y0 + cell_size, x0 : x0 + cell_size] = resized

        # Label
        if labels and idx < len(labels):
            label = labels[idx][:35]
            cv2.putText(
                gallery, label, (x0 + 5, y0 + cell_size + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
            )

    return gallery


def save_annotated(image: np.ndarray, output_path: str) -> str:
    """Save an annotated image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), image)
    logger.debug("Saved: %s", path)
    return str(path)
