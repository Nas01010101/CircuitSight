"""
AIT Visual Inspector — ONNX Model Export
Export YOLOv8 model to ONNX format with inference benchmarking.

Usage:
    python -m src.export.onnx_export --weights runs/pcb/weights/best.pt
    # or: make export-onnx
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def export_to_onnx(
    weights_path: str,
    img_size: int = 640,
    simplify: bool = True,
) -> str:
    """
    Export YOLOv8 model to ONNX format.

    Args:
        weights_path: Path to .pt weights
        img_size: Input image size
        simplify: Whether to simplify the ONNX graph

    Returns:
        Path to exported .onnx file
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)

    logger.info("Exporting to ONNX: %s", weights_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=img_size,
        simplify=simplify,
    )

    logger.info("ONNX exported: %s", onnx_path)
    return str(onnx_path)


def benchmark(
    weights_path: str,
    n_runs: int = 50,
    img_size: int = 640,
) -> dict:
    """
    Benchmark PyTorch vs ONNX inference speed.

    Args:
        weights_path: Path to .pt weights
        n_runs: Number of inference runs for averaging
        img_size: Input image size

    Returns:
        dict with benchmark results
    """
    from ultralytics import YOLO

    # Create dummy image
    dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    results = {}

    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch (%d runs)...", n_runs)
    model_pt = YOLO(weights_path)
    model_pt(dummy, verbose=False)  # warmup

    times_pt = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model_pt(dummy, verbose=False)
        times_pt.append((time.perf_counter() - start) * 1000)

    results["pytorch"] = {
        "mean_ms": round(np.mean(times_pt), 2),
        "std_ms": round(np.std(times_pt), 2),
        "fps": round(1000 / np.mean(times_pt), 1),
    }

    # Export and benchmark ONNX
    onnx_path = Path(weights_path).with_suffix(".onnx")
    if not onnx_path.exists():
        export_to_onnx(weights_path, img_size=img_size)

    if onnx_path.exists():
        logger.info("Benchmarking ONNX (%d runs)...", n_runs)
        model_onnx = YOLO(str(onnx_path))
        model_onnx(dummy, verbose=False)  # warmup

        times_onnx = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model_onnx(dummy, verbose=False)
            times_onnx.append((time.perf_counter() - start) * 1000)

        results["onnx"] = {
            "mean_ms": round(np.mean(times_onnx), 2),
            "std_ms": round(np.std(times_onnx), 2),
            "fps": round(1000 / np.mean(times_onnx), 1),
        }

        speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
        results["speedup"] = round(speedup, 2)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--n-runs", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.weights, img_size=args.img_size)
    print(f"\nExported: {onnx_path}")

    if args.benchmark:
        results = benchmark(args.weights, n_runs=args.n_runs, img_size=args.img_size)
        print(f"\n{'='*40}")
        print("BENCHMARK RESULTS")
        print(f"{'='*40}")
        for fmt, r in results.items():
            if isinstance(r, dict):
                print(f"  {fmt:10s}: {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms  ({r['fps']:.0f} FPS)")
        if "speedup" in results:
            print(f"  ONNX speedup: {results['speedup']}x")


if __name__ == "__main__":
    main()
