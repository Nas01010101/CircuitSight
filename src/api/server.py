"""
CircuitSight — FastAPI REST API

Provides HTTP endpoints for defect detection:
    POST /inspect    — Upload image, get JSON detections + verdict
    GET  /health     — Service health check
    GET  /model/info — Model metadata

Usage:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    # or: make api
"""

import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.models.detector import CircuitSight_Detector

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────
app = FastAPI(
    title="CircuitSight API",
    description="PCB defect detection REST API",
    version="2.0.0",
)

# ── Global state ─────────────────────────────
_detector: CircuitSight_Detector = None
_start_time: float = time.time()
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_detector() -> CircuitSight_Detector:
    """Lazy-load the detector on first request."""
    global _detector
    if _detector is not None:
        return _detector

    model_config = "configs/model.yaml"
    app_config = "configs/app.yaml"
    domain_config = "configs/domains/pcb.yaml"

    _detector = CircuitSight_Detector.from_config(
        model_config, app_config,
        domain_config=domain_config if Path(domain_config).exists() else None,
    )

    # Find best available weights
    weight_candidates = [
        "models/pcb_mixed_best.pt",
        "runs/pcb/weights/best.pt",
        "runs/train/weights/best.pt",
        "runs/ait_inspector/weights/best.pt",
    ]

    # Also search runs/ recursively
    runs_dir = Path("runs")
    if runs_dir.exists():
        for pt in sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            weight_candidates.insert(0, str(pt))

    for wp in weight_candidates:
        if Path(wp).exists():
            _detector.load(wp)
            logger.info("API loaded model: %s", wp)
            return _detector

    logger.warning("No model weights found — API will return errors on /inspect")
    return _detector


# ── Endpoints ────────────────────────────────

@app.get("/health")
async def health():
    """Service health check."""
    det = _get_detector()
    return {
        "status": "healthy",
        "model_loaded": det.model is not None,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    det = _get_detector()
    return det.get_model_info()


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    """
    Run defect detection on an uploaded image.

    Returns JSON with detections, verdict, and metrics.
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (PNG, JPEG, BMP)")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    # Run detection
    det = _get_detector()
    if det.model is None:
        raise HTTPException(503, "Model not loaded. Train a model first: make train-pcb")

    result = det.detect(img, annotate=False)
    result.image_path = file.filename or "<upload>"

    return JSONResponse(content={
        "filename": file.filename,
        **result.to_dict(),
    })


@app.on_event("startup")
async def startup():
    """Pre-load model on startup."""
    logger.info("CircuitSight API starting...")
    _get_detector()
