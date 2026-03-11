# CircuitSight

A computer vision system for automated defect detection in Printed Circuit Board (PCB) manufacturing.

## Overview

CircuitSight detects visual defects on PCBs. It utilizes a YOLOv8 object detection model combined with image preprocessing techniques to identify specific defect types.

**Technical Specifications:**
*   **Target Classes**: The model is trained to identify 6 PCB defect categories: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`.
*   **Deployment Architecture**: Includes a FastAPI REST API for integration with manufacturing execution systems and a directory polling script for continuous image processing.
*   **Domain Configuration**: Uses a YAML-based configuration system to specify inspection parameters and class dictionaries.

## Architecture

*   **Model Architecture**: YOLOv8s (Small parameter variant).
*   **Preprocessing**: Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) and green-channel weighting.
*   **User Interface**: A Streamlit dashboard for data visualization and inference testing.
*   **API**: FastAPI server providing HTTP POST endpoints for inference.
*   **Export**: Supports ONNX runtime export for CPU inference.

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
A script monitors the `data/inbox/` directory for new image files. Upon detection, it processes the image through the inference pipeline and writes JSON reports and annotated output images to `reports/auto/`.

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
