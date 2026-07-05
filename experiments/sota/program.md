# CircuitSight SOTA push — autoresearch program

## Objective
Maximize detection quality on PKU-Market-PCB (6 defect classes) with modern YOLO-family
recipes, closing the gap to 2025-26 published results (mAP@50 97-99.5 on this dataset,
mAP@50-95 up to ~90 on augmented variants). All numbers must be real measured runs.

## Metric (runner-owned)
`val mAP@50-95` on the val split (103 images, seed-42 split from convert_to_yolo).
Written by target.py to `$AUTORESEARCH_RESULTS` as `{"metric": <float>, ...}`.
**Maximize** (`AUTORESEARCH_MAXIMIZE=1`). mAP@50 is logged alongside as auxiliary.

## Editable surface
`experiments/sota/target.py` ONLY (the EXPERIMENT dict + any model-surgery code).

## Judge — OUT OF BOUNDS
- The dataset splits under `data/processed/pcb_yolo/{train,val,test}` (seed-42, fixed).
- `configs/data.yaml`, `src/data/convert_to_yolo.py` split logic.
- The metric extraction: ultralytics `model.val()` on split=val, standard protocol.
- The **test split (105 images) is held out** — never trained on, never used for
  keep/revert. It is evaluated ONCE per champion for the paper.

## Time-box
`AUTORESEARCH_TIMEOUT=14400` (4 h wall-clock per experiment).

## Fixed protocol
seed=42, deterministic split, device=mps, single run per config (note: single-seed —
paper must state this), pretrained COCO init, early-stop patience=30.

## Hypothesis queue (from 2025-26 literature)
1. Baseline anchor: yolov8s @ 640 with repo augs (what the repo shipped).
2. Higher input res (defects are tiny: ~2-3% of a 3034x1586 board image) — imgsz 1024.
3. Newer backbone: yolo11s / yolo12s.
4. Small-object head (P2) via custom model yaml.
5. Aug tuning: mosaic off in last epochs, copy-paste, degrees/scale for PCB geometry.
6. Loss: default CIoU vs alternatives available in ultralytics.
