# Model Card — AIT Visual Inspector

## Model Overview

| Field | Value |
|-------|-------|
| **Architecture** | YOLOv8s (Small) |
| **Task** | Object Detection (PCB Defect Detection) |
| **Framework** | Ultralytics / PyTorch |
| **Input** | RGB images (640×640) |
| **Output** | Bounding boxes + confidence + defect class |
| **Classes** | 6 (missing_hole, mouse_bite, open_circuit, short_circuit, spur, spurious_copper) |

## Intended Use

- **Primary**: Automated visual inspection of printed circuit boards (PCBs)
- **Scope**: Detecting 6 types of manufacturing defects on bare PCB surfaces
- **Users**: QC engineers, AIT technicians, automated inspection pipelines
- **Deployment**: Docker, REST API, systemd service, or embedded systems

## Training Data

- **Dataset**: PKU-Market-PCB
- **Source**: [Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **Size**: 693 images with Pascal VOC XML annotations
- **Split**: 70% train / 15% val / 15% test (seed=42)
- **Preprocessing**: VOC XML → YOLO format, CLAHE contrast enhancement
- **Augmentation**: Mosaic, mixup, HSV shift, rotation, flip, scale

## Limitations

- Trained on bare PCBs — may not generalize to assembled boards or other components without fine-tuning
- 6 defect types only — does not detect solder defects, component misalignment, etc.
- Performance depends on image quality, lighting, and camera angle
- Small defects (<20px at 640×640) may be missed

## Ethical Considerations

- Not intended for safety-critical decisions without human review
- "NEEDS_REVIEW" verdict requires human judgment
- False negatives should be monitored in production
- Confidence thresholds should be tuned per deployment

## Extensibility

The plugin architecture supports additional domains:
- **Metal Surface** (NEU Surface Defect DB): scratches, cracks, inclusions on steel
- **General Anomaly** (MVTec AD): anomalies across 15 product categories

Add a new domain by creating a YAML config in `configs/domains/` and training on domain-specific data.
