"""
PKU-Market-PCB Dataset Downloader
Downloads the PCB defect detection dataset from Kaggle.

Dataset: 693 images with 6 PCB defect types
Source:  https://www.kaggle.com/datasets/akhatova/pcb-defects

Usage:
    python -m src.data.download_pcb
    python -m src.data.download_pcb --output-dir data/raw/pcb
"""

import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "akhatova/pcb-defects"
DEFAULT_OUTPUT = "data/raw/pcb"


def download_from_kaggle(output_dir: str = DEFAULT_OUTPUT) -> Path:
    """
    Download PKU-Market-PCB dataset via Kaggle API.

    Auth: set KAGGLE_API_TOKEN env var with your Kaggle API token.
        export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (output_path / "Annotations").exists() and (output_path / "images").exists():
        logger.info("PCB dataset already exists at %s", output_path)
        return output_path

    # Verify token is set
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        logger.error("KAGGLE_API_TOKEN not set.")
        _print_manual_instructions(output_path)
        raise EnvironmentError(
            "Set KAGGLE_API_TOKEN: export KAGGLE_API_TOKEN=your_token"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading PCB dataset from Kaggle: %s", KAGGLE_DATASET)
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_path),
            unzip=True,
        )
        logger.info("Download complete: %s", output_path)

    except ImportError:
        logger.error(
            "kaggle package not installed. Run: pip install kaggle"
        )
        _print_manual_instructions(output_path)
        raise

    except Exception as e:
        logger.error("Kaggle download failed: %s", e)
        _print_manual_instructions(output_path)
        raise

    return output_path


def _print_manual_instructions(output_path: Path) -> None:
    """Print manual download instructions as fallback."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("Option A: Set your Kaggle token and retry")
    print("  export KAGGLE_API_TOKEN=your_token_here")
    print("  make download-pcb")
    print("")
    print("Option B: Manual download")
    print(f"  1. Go to: https://www.kaggle.com/datasets/akhatova/pcb-defects")
    print(f"  2. Click 'Download' (requires free Kaggle account)")
    print(f"  3. Unzip to: {output_path}")
    print(f"  4. Verify structure:")
    print(f"     {output_path}/Annotations/  (VOC XML files)")
    print(f"     {output_path}/images/       (PCB images)")
    print("=" * 60 + "\n")


def verify_dataset(data_dir: str = DEFAULT_OUTPUT) -> dict:
    """Verify the downloaded PCB dataset structure and count files."""
    data_path = Path(data_dir)

    stats = {
        "valid": False,
        "images": 0,
        "annotations": 0,
        "missing": [],
    }

    img_dir = data_path / "images"
    ann_dir = data_path / "Annotations"

    if not img_dir.exists():
        # Check alternate structures (some Kaggle downloads nest differently)
        for alt in [data_path / "PCB_DATASET" / "images", data_path]:
            if (alt / "images").exists():
                img_dir = alt / "images"
                ann_dir = alt / "Annotations"
                break

    if not img_dir.exists():
        stats["missing"].append("images/")
        logger.error("Image directory not found in %s", data_path)
        return stats

    if not ann_dir.exists():
        stats["missing"].append("Annotations/")
        logger.error("Annotations directory not found in %s", data_path)
        return stats

    images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
    annotations = list(ann_dir.rglob("*.xml"))

    stats["images"] = len(images)
    stats["annotations"] = len(annotations)
    stats["valid"] = len(images) > 0 and len(annotations) > 0

    logger.info(
        "PCB dataset: %d images, %d annotations, valid=%s",
        stats["images"], stats["annotations"], stats["valid"],
    )

    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download PKU-Market-PCB dataset")
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help="Directory to save dataset",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing dataset",
    )
    args = parser.parse_args()

    if args.verify_only:
        stats = verify_dataset(args.output_dir)
        print(f"Dataset valid: {stats['valid']}")
        print(f"Images: {stats['images']}, Annotations: {stats['annotations']}")
    else:
        download_from_kaggle(args.output_dir)
        stats = verify_dataset(args.output_dir)
        if stats["valid"]:
            print(f"\nDataset ready: {stats['images']} images, {stats['annotations']} annotations")
        else:
            print(f"\nDataset verification failed. Missing: {stats['missing']}")


if __name__ == "__main__":
    main()
