"""
YOLOv8 Detector Wrapper
Multi-class defect detection with domain-aware configuration.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default PCB defect class colors (BGR)
DEFAULT_CLASS_COLORS = {
    "missing_hole": (255, 0, 0),       # Blue
    "mouse_bite": (0, 0, 255),         # Red
    "open_circuit": (0, 165, 255),     # Orange
    "short_circuit": (0, 255, 255),    # Yellow
    "spur": (255, 0, 255),             # Magenta
    "spurious_copper": (0, 255, 0),    # Green
    "defect": (0, 0, 255),             # Red (fallback)
    "good": (0, 200, 0),               # Green
}


@dataclass
class Detection:
    """Single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list  # [x1, y1, x2, y2] pixel coordinates
    bbox_norm: list  # [cx, cy, w, h] normalized YOLO format

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_xyxy": [round(b, 1) for b in self.bbox],
            "bbox_yolo": [round(b, 6) for b in self.bbox_norm],
        }


@dataclass
class InferenceResult:
    """Result of running inference on a single image."""

    image_path: str
    image_shape: tuple  # (h, w, c)
    detections: List[Detection]
    inference_time_ms: float
    verdict: str = "PASS"  # PASS, FAIL, NEEDS_REVIEW
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_defects(self) -> int:
        """Count all detections (all classes are defect types in PCB domain)."""
        return len(self.detections)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "image_shape": list(self.image_shape),
            "n_detections": len(self.detections),
            "n_defects": self.n_defects,
            "verdict": self.verdict,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detections": [d.to_dict() for d in self.detections],
        }


class CircuitSight_Detector:
    """
    YOLOv8-based multi-class defect detector for AIT inspection.

    Supports domain-aware configuration via YAML configs.

    Usage:
        detector = CircuitSight_Detector.from_config("configs/model.yaml")
        detector.load("runs/pcb/weights/best.pt")
        result = detector.detect("image.png")
    """

    def __init__(
        self,
        model_size: str = "yolov8s",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        pass_threshold: float = 0.5,
        review_threshold: float = 0.3,
        max_defects_pass: int = 0,
        class_names: dict = None,
        class_colors: dict = None,
        domain: str = "pcb",
    ):
        self.model_size = model_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.pass_threshold = pass_threshold
        self.review_threshold = review_threshold
        self.max_defects_pass = max_defects_pass
        self.class_names = class_names or {
            0: "missing_hole", 1: "mouse_bite", 2: "open_circuit",
            3: "short_circuit", 4: "spur", 5: "spurious_copper",
        }
        self.class_colors = class_colors or DEFAULT_CLASS_COLORS
        self.domain = domain
        self.model = None
        self._weights_path = None

    @classmethod
    def from_config(cls, model_config: str, app_config: str = None,
                    domain_config: str = None) -> "CircuitSight_Detector":
        """Create detector from YAML config files."""
        with open(model_config) as f:
            mcfg = yaml.safe_load(f)

        kwargs = {
            "model_size": mcfg.get("model_size", "yolov8s"),
            "conf_threshold": mcfg.get("inference", {}).get("conf_threshold", 0.25),
            "iou_threshold": mcfg.get("inference", {}).get("iou_threshold", 0.45),
            "max_det": mcfg.get("inference", {}).get("max_det", 300),
        }

        if app_config:
            with open(app_config) as f:
                acfg = yaml.safe_load(f)
            qc = acfg.get("qc", {})
            kwargs["pass_threshold"] = qc.get("pass_threshold", 0.5)
            kwargs["review_threshold"] = qc.get("review_threshold", 0.3)
            kwargs["max_defects_pass"] = qc.get("max_defects_pass", 0)

        # Load domain config for class names and colors
        if domain_config:
            with open(domain_config) as f:
                dcfg = yaml.safe_load(f)
            kwargs["class_names"] = dcfg.get("names", kwargs.get("class_names"))
            kwargs["domain"] = dcfg.get("domain", {}).get("name", "pcb")

            # Parse class colors
            colors = dcfg.get("class_colors", {})
            if colors:
                kwargs["class_colors"] = {k: tuple(v) for k, v in colors.items()}

        return cls(**kwargs)

    def load(self, weights_path: str) -> None:
        """Load model weights."""
        from ultralytics import YOLO

        self._weights_path = str(weights_path)
        self.model = YOLO(self._weights_path)
        logger.info("Loaded model: %s", weights_path)

    def _determine_verdict(self, detections: List[Detection]) -> str:
        """
        Determine pass/fail/review verdict based on detections.

        In multi-class mode (PCB), ALL classes are defect types.
        Verdict is based on confidence levels, not class names.
        """
        if not detections:
            return "PASS"

        # In PCB domain, all detections are defects
        high_conf = [d for d in detections if d.confidence >= self.pass_threshold]
        mid_conf = [
            d for d in detections
            if self.review_threshold <= d.confidence < self.pass_threshold
        ]

        if len(high_conf) > self.max_defects_pass:
            return "FAIL"

        if mid_conf:
            return "NEEDS_REVIEW"

        return "PASS"

    def detect(
        self,
        source: Union[str, np.ndarray],
        annotate: bool = True,
    ) -> InferenceResult:
        """
        Run detection on a single image.

        Args:
            source: Image path or numpy array (BGR)
            annotate: Whether to draw bounding boxes on the image

        Returns:
            InferenceResult
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load(weights_path) first.")

        import time

        if isinstance(source, str):
            img = cv2.imread(source)
            img_path = source
        else:
            img = source
            img_path = "<array>"

        if img is None:
            raise ValueError(f"Could not load image: {source}")

        h, w = img.shape[:2]

        start = time.perf_counter()
        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                cx = ((xyxy[0] + xyxy[2]) / 2) / w
                cy = ((xyxy[1] + xyxy[3]) / 2) / h
                bw = (xyxy[2] - xyxy[0]) / w
                bh = (xyxy[3] - xyxy[1]) / h

                det = Detection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=xyxy,
                    bbox_norm=[cx, cy, bw, bh],
                )
                detections.append(det)

        verdict = self._determine_verdict(detections)

        annotated = None
        if annotate:
            annotated = img.copy()
            annotated = self._draw_detections(annotated, detections, verdict)

        return InferenceResult(
            image_path=img_path,
            image_shape=img.shape,
            detections=detections,
            inference_time_ms=elapsed_ms,
            verdict=verdict,
            annotated_image=annotated,
        )

    def batch_detect(
        self,
        sources: List[Union[str, np.ndarray]],
        annotate: bool = True,
    ) -> List[InferenceResult]:
        """Run detection on multiple images."""
        return [self.detect(src, annotate=annotate) for src in sources]

    def _draw_detections(
        self,
        img: np.ndarray,
        detections: List[Detection],
        verdict: str,
    ) -> np.ndarray:
        """Draw bounding boxes with per-class colors and verdict banner."""
        VERDICT_COLORS = {
            "PASS": (0, 200, 0),
            "FAIL": (0, 0, 255),
            "NEEDS_REVIEW": (0, 165, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = [int(b) for b in det.bbox]
            color = self.class_colors.get(det.class_name, (0, 0, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Verdict banner
        h, w = img.shape[:2]
        banner_color = VERDICT_COLORS.get(verdict, (128, 128, 128))
        cv2.rectangle(img, (0, 0), (w, 35), banner_color, -1)
        cv2.putText(
            img,
            f"VERDICT: {verdict}  |  Defects: {len(detections)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return img

    def get_model_info(self) -> dict:
        """Return model metadata (for API /model/info endpoint)."""
        return {
            "model_size": self.model_size,
            "domain": self.domain,
            "class_names": self.class_names,
            "conf_threshold": self.conf_threshold,
            "pass_threshold": self.pass_threshold,
            "review_threshold": self.review_threshold,
            "weights": self._weights_path,
            "loaded": self.model is not None,
        }
