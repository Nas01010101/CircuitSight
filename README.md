# CircuitSight

A professional, production-ready computer vision system for automated defect detection in manufacturing environments, focusing primarily on **Printed Circuit Boards (PCBs)**.

## Overview

CircuitSight is designed to replace or augment manual visual inspection on assembly lines. It uses state-of-the-art object detection (YOLOv8) combined with domain-specific feature engineering to identify sub-millimeter defects in real-time.

**Core Capabilities:**
*   **Multi-class Detection**: Specifically trained to detect 6 common PCB defects: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`.
*   **High Precision**: Tuned for automated Quality Control (QC) workflows where false positives must be minimized.
*   **Production Deployment**: Features a FastAPI REST API for integration with manufacturing execution systems (MES), and a directory watcher for continuous camera feeds.
*   **Extensible Architecture**: Uses a plugin-based configuration system to easily add new inspection domains (e.g., metal surfaces, textiles) without altering application code.

## Architecture

*   **Model**: YOLOv8s (Small) trained on the PKU-Market-PCB dataset.
*   **Feature Engineering**: Custom preprocessing pipeline including CLAHE (Contrast Limited Adaptive Histogram Equalization) and green-channel emphasis tailored for PCBs.
*   **Web Dashboard**: A clean, professional Streamlit interface for manual inspection, batch reporting, and metric visualization.
*   **REST API**: FastAPI implementation providing high-throughput inference endpoints.
*   **Optimization**: Built-in ONNX export with benchmarking for CPU/Edge deployment speedups.

## Quick Start (Docker)

The fastest way to run the complete system (Dashboard + API) is via Docker Compose:

```bash
make deploy
```
*   **Dashboard**: `http://localhost:8501`
*   **REST API**: `http://localhost:8000/docs`

## Manual Installation

1.  **Clone & Install Dependencies**
    ```bash
    git clone <repository_url>
    cd circuitsight
    make setup
    ```

2.  **Download & Prepare Dataset**
    Ensure you have set the Kaggle API token environment variable (`export KAGGLE_API_TOKEN=your_token_here`).
    ```bash
    make prepare-pcb
    ```

3.  **Train the Model**
    ```bash
    make train-pcb
    ```

4.  **Run the Applications**
    ```bash
    # Run the interactive dashboard
    make app
    
    # Run the headless REST API
    make api
    
    # Run the automated folder watcher
    make watch
    ```

## Usage

### REST API Example
Test the detection endpoint using curl:
```bash
curl -X POST -F "file=@data/demo_images/missing_hole_sample.jpg" http://localhost:8000/inspect
```

### Folder Watcher
The watcher simulates a camera feed. Drop images into `data/inbox/`; the system automatically detects them, processes the inference, and outputs JSON reports and annotated images to `reports/auto/`.

## Extensibility (New Domains)

To add a new inspection domain (e.g. `metal_surface`), simply create a new configuration file at `configs/domains/metal_surface.yaml`:

```yaml
name: "Metal Surface Defect Detection"
description: "Detects scratches, cracks, and inclusions on steel surfaces."
classes:
  0: "scratch"
  1: "crack"
  2: "inclusion"
class_colors:
  scratch: [0, 0, 255]
  crack: [255, 0, 0]
  inclusion: [0, 255, 0]
preprocessing:
  apply_clahe: true
```
Then train by referencing the new data paths in `data.yaml`.

## License & Acknowledgements
Dataset: [PKU-Market-PCB](https://www.kaggle.com/datasets/akhatova/pcb-defects)
Base Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
