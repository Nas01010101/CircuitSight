"""
MVTec AD Dataset Downloader
Downloads and extracts the MVTec Anomaly Detection dataset.

Usage:
    python -m src.data.download_mvtec --config configs/data.yaml
"""

import argparse
import hashlib
import logging
import os
import tarfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# MVTec AD download URLs (official mirrors)
MVTEC_BASE_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"

MVTEC_CATEGORIES = {
    "bottle": "420938113-1629952094",
    "cable": "420938133-1629952094",
    "capsule": "420938153-1629952094",
    "carpet": "420938173-1629952094",
    "grid": "420938193-1629952094",
    "hazelnut": "420938213-1629952094",
    "leather": "420938233-1629952094",
    "metal_nut": "420938253-1629952094",
    "pill": "420938273-1629952094",
    "screw": "420938293-1629952094",
    "tile": "420938313-1629952094",
    "toothbrush": "420938333-1629952094",
    "transistor": "420938353-1629952094",
    "wood": "420938373-1629952094",
    "zipper": "420938393-1629952094",
}

# Alternative: single-file download (all categories)
MVTEC_FULL_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938413-1629952094/mvtec_anomaly_detection.tar.xz"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    logger.info("Downloading: %s", desc or dest.name)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    checksum = compute_sha256(dest)
    logger.info("Download complete: %s (SHA256: %s)", dest.name, checksum[:16])
    return dest


def _is_safe_tar_member(member: tarfile.TarInfo, dest: Path) -> bool:
    """Check that a tar member extracts safely within the destination directory."""
    member_path = (dest / member.name).resolve()
    return str(member_path).startswith(str(dest.resolve()))


def extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tar archive with path traversal protection."""
    logger.info("Extracting: %s -> %s", archive.name, dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:*") as tar:
        safe_members = []
        for member in tar.getmembers():
            if _is_safe_tar_member(member, dest):
                safe_members.append(member)
            else:
                logger.warning(
                    "Skipping unsafe tar member (path traversal): %s", member.name
                )
        tar.extractall(path=dest, members=safe_members)


def download_mvtec_ad(
    output_dir: str = "data/raw/mvtec_ad",
    categories: list = None,
) -> Path:
    """
    Download MVTec AD dataset.

    Args:
        output_dir: Directory to save raw data
        categories: List of category names to download (None = all)

    Returns:
        Path to the downloaded dataset root
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = list(MVTEC_CATEGORIES.keys())

    logger.info(
        "MVTec AD Dataset Download -- %d / %d categories",
        len(categories), len(MVTEC_CATEGORIES),
    )

    for cat in categories:
        cat_dir = output_path / cat
        if cat_dir.exists() and any(cat_dir.iterdir()):
            logger.info("%s: already exists, skipping", cat)
            continue

        if cat not in MVTEC_CATEGORIES:
            logger.warning("%s: unknown category, skipping", cat)
            continue

        logger.info("Processing: %s", cat)
        archive_id = MVTEC_CATEGORIES[cat]
        url = f"{MVTEC_BASE_URL}/{archive_id}/{cat}.tar.xz"

        archive_path = output_path / f"{cat}.tar.xz"
        try:
            download_file(url, archive_path, desc=cat)
            extract_tar(archive_path, output_path)
            # Clean up archive after extraction
            archive_path.unlink()
        except Exception as e:
            logger.error("Failed to download %s: %s", cat, e)
            logger.info(
                "Manual download: https://www.mvtec.com/company/research/datasets/mvtec-ad"
            )
            if archive_path.exists():
                archive_path.unlink()

    logger.info("Dataset ready at: %s", output_path)
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to download",
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    output_dir = args.output_dir or "data/raw/mvtec_ad"
    categories = args.categories or config.get("mvtec_categories", None)

    download_mvtec_ad(output_dir=output_dir, categories=categories)


if __name__ == "__main__":
    main()
