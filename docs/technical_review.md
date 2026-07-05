# CircuitSight — Technical Review

*Full-codebase audit covering correctness, methodology, deployment, and documentation.*

## Scope

Every Python module (~4,200 LOC), all configs, the test suite, Docker/Compose/systemd
deployment files, the Makefile, and user-facing prose (README, dashboard, model card)
were reviewed. Issues found were fixed in the same change set; each is listed below with
its impact.

## Architecture summary

```
data/inbox ──► watcher ─┐
HTTP upload ─► FastAPI ─┼─► CircuitSight_Detector (YOLOv8s) ─► verdict engine ─► JSON / annotated image
CLI (infer) ────────────┘         │
                                  └─ preprocessing: CLAHE (LAB L-channel) + green-channel boost
```

- **Detection core** (`src/models/detector.py`): YOLOv8s wrapper with domain-aware config
  (`configs/domains/pcb.yaml`), 6 PCB defect classes, PASS / FAIL / NEEDS_REVIEW verdict
  logic driven by confidence thresholds.
- **Serving**: FastAPI REST API (`/inspect`, `/health`, `/model/info`), read-only Streamlit
  analytics dashboard, polling folder-watcher, ONNX export for CPU deployment,
  ByteTrack video tracking.
- **Data pipeline**: Kaggle download → VOC-XML→YOLO conversion with validation →
  optional synthetic scratch/stain augmentation.

## Issues found and fixed

### Correctness

1. **Weight discovery served the *oldest* checkpoint** (`src/api/server.py`).
   Candidates were collected newest-first and then re-inserted at index 0 one by one,
   which reversed the order — the API silently loaded the stalest `runs/**/best.pt`
   instead of the newest, and raw training runs outranked the curated release weights.
   The watcher had a second, divergent copy of this logic that never looked at `models/`
   at all, and `infer.py`/`evaluate.py` had a third. All four call sites now share
   `src/utils/weights.find_best_weights()` with an explicit priority
   (curated `models/pcb_mixed_best.pt` → newest training run), covered by unit tests.

2. **Evaluation assumed a binary good/defect dataset** (`evaluate.py`).
   The failure gallery and image-level confusion matrix decided "does this image contain
   a real defect?" by looking for a class literally named `defect`, falling back to
   class id 1 — which for the 6-class PCB set is `mouse_bite`. Images whose only defects
   were the other five classes were counted as defect-free, corrupting both artifacts.
   Now `resolve_defect_class_ids()` treats every class as a defect unless it is named
   `good`/`normal`/`ok`/`background`, which matches defect-only datasets like
   PKU-Market-PCB while preserving binary-dataset behavior. Unit-tested.

3. **Committed config carried a machine-local absolute path** (`configs/data.yaml`).
   `train.py` rewrote the config in place to absolutize `path:`, leaking the local
   directory layout into version control and breaking the config on any other machine.
   Training now writes a runtime-resolved copy to `runs/resolved_data.yaml` and leaves
   the committed config untouched (it stays relative and portable).

4. **Class-name mismatch: `short` vs `short_circuit`.** The dataset, converter, and
   trained model use `short` (class 3); the domain config and detector defaults said
   `short_circuit`, so color lookups and display names disagreed depending on the code
   path. Unified on `short` (the label the model was actually trained with) everywhere.

5. **Broken package export.** `src/models/__init__.py` declared
   `__all__ = ["AIT_Detector", ...]` for a class that no longer exists (renamed to
   `CircuitSight_Detector`), so `from src.models import *` raised `AttributeError`.

6. **Watcher raced partially-written files.** A file still being copied into
   `data/inbox/` could be read mid-write. The watcher now skips files modified less
   than 1 s ago and picks them up on the next poll.

### Deployment

7. **Docker build failure**: the runtime stage installed `libgl1-mesa-glx`, which no
   longer exists in Debian bookworm (the `python:3.11-slim` base). Replaced with `libgl1`.

8. **Compose never mounted `models/`**, so containers could not see the curated
   weights the loader now prefers. Added read-only `./models` mounts to both services.

9. **Dead Makefile targets**: `download-mvtec`/`convert-mvtec` invoked
   `src.data.download_mvtec`, a module that does not exist. Removed.

10. **Stale dependency**: `watchdog` was listed but the watcher uses stdlib polling by
    design. Removed; `matplotlib`/`seaborn` (imported directly by `evaluate.py`) are now
    declared explicitly instead of arriving transitively via ultralytics.

### API modernization

11. Migrated the deprecated `@app.on_event("startup")` hook to FastAPI's lifespan
    context manager; removed unused imports (`io`, `yaml`, `time`) and hoisted an inline
    `import time` out of the per-call `detect()` hot path.

### Documentation & prose

12. **Dashboard honesty pass**: removed hardcoded "🟢 REST API: Online" status lights
    (the dashboard never checks the API — they now read as endpoint documentation),
    and rewrote marketing-toned copy ("surging from 51% accuracy to a massive 98.1%
    accuracy") as a precise claim: OOD mAP@50 0.510 zero-shot → 0.981 after joint
    fine-tuning.
13. **Branding consistency**: the project rename to CircuitSight had left "AIT Visual
    Inspector" in ~20 files, class names (`AIT_Tracker`), Docker tags, container names,
    and the systemd unit. Unified everywhere.
14. README: fixed the extensibility example to match the real domain-config schema,
    removed a reference to a demo image that isn't in the repo, documented `make test`.

## Methodology assessment

- **Metrics**: primary numbers come from Ultralytics' built-in `model.val()`
  (standard mAP@50 / mAP@50-95 protocol) written to a runner-owned
  `evaluation_report.json` — good practice. The custom greedy IoU matcher in
  `src/utils/metrics.py` sorts predictions by confidence before matching and counts
  per-class TP/FP/FN correctly.
- **Generalization protocol** (dashboard Tab 3) is sound: zero-shot evaluation on an
  unseen second dataset (mAP@50 0.510), then joint fine-tuning on the mixed pool
  (mAP@50 0.981 on the mixed test split). The honest caveat — the post-finetune number
  is no longer strictly out-of-distribution since the second dataset entered training —
  is inherent to the design and now reflected in neutral wording.
- **Known limitation**: dashboard headline metrics are hardcoded snapshots of the
  training runs (the run artifacts are gitignored). Regenerating them from a committed
  `evaluation_report.json` would make the dashboard fully self-verifying; left as
  future work since the underlying `runs/` are not in the repo.

## Test status

`pytest tests/ -v`: **42 passed** (36 pre-existing + 6 new covering weight-discovery
priority and defect-class resolution).
