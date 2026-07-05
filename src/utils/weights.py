"""
Model Weight Discovery
Single source of truth for locating trained weights, shared by the API,
watcher, inference, and evaluation entry points.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Curated release weights take priority over raw training runs.
CURATED_WEIGHTS = [
    "models/pcb_mixed_best.pt",
]


def find_best_weights(root: str = ".") -> Optional[str]:
    """
    Locate the best available model weights.

    Priority:
        1. Curated weights in models/ (the promoted, fine-tuned checkpoint)
        2. The most recently modified runs/**/best.pt training artifact

    Returns:
        Path string, or None if no weights exist.
    """
    root_path = Path(root)

    for rel in CURATED_WEIGHTS:
        candidate = root_path / rel
        if candidate.exists():
            return str(candidate)

    runs_dir = root_path / "runs"
    if runs_dir.exists():
        run_weights = sorted(
            runs_dir.rglob("best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_weights:
            return str(run_weights[0])

    return None
