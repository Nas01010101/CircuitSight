"""
AIT Visual Inspector -- Object Tracker
ByteTrack-based multi-object tracking for FOD/tool detection in video.

Integrates with the detector to maintain object IDs across frames
and generate alerts for persistent or unexpected objects.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.models.detector import Detection
from src.utils.metrics import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """An object being tracked across frames."""

    track_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    dwell_time: float = 0.0
    bbox: list = field(default_factory=list)
    alerted: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "dwell_time_s": round(self.dwell_time, 2),
            "bbox": self.bbox,
            "alerted": self.alerted,
        }


@dataclass
class TrackingEvent:
    """A tracking alert event."""

    event_type: str  # "dwell_alert", "new_object", "object_lost"
    track_id: int
    class_name: str
    frame_idx: int
    timestamp: float
    message: str

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "message": self.message,
        }


class AIT_Tracker:
    """
    ByteTrack-based tracker for AIT inspection video analysis.

    Wraps supervision's ByteTrack for seamless integration with
    AIT_Detector detections.

    Usage:
        tracker = AIT_Tracker(alert_dwell_seconds=5)
        for frame in video:
            detections = detector.detect(frame)
            annotated, events = tracker.update(frame, detections, frame_idx, fps)
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        alert_dwell_seconds: float = 5.0,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.alert_dwell_seconds = alert_dwell_seconds

        # Initialize ByteTrack via supervision
        try:
            import supervision as sv

            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_thresh,
                frame_rate=30,
            )
            self._sv = sv
            self._use_supervision = True
        except ImportError:
            logger.warning("supervision not installed, using IoU-based tracking fallback")
            self._use_supervision = False
            self._next_id = 0
            self._prev_detections: List[Tuple[int, list]] = []  # (track_id, bbox)

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.start_time = time.time()

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        fps: float = 30.0,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Update tracker with new detections.

        Args:
            frame: Current video frame (BGR)
            detections: List of Detection objects from AIT_Detector
            frame_idx: Current frame index
            fps: Video FPS for dwell time calculation

        Returns:
            (annotated_frame, list_of_event_dicts)
        """
        events = []
        current_time = frame_idx / fps

        if not detections:
            return frame.copy(), events

        if self._use_supervision:
            annotated, events = self._update_supervision(
                frame, detections, frame_idx, current_time
            )
        else:
            annotated, events = self._update_basic(
                frame, detections, frame_idx, current_time
            )

        return annotated, [e.to_dict() for e in events]

    def _update_supervision(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        current_time: float,
    ) -> Tuple[np.ndarray, List[TrackingEvent]]:
        """Update using supervision's ByteTrack."""
        sv = self._sv
        events = []

        # Convert detections to supervision format
        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )

        # Run ByteTrack
        tracked = self.byte_tracker.update_with_detections(sv_detections)

        # Process tracked objects
        annotated = frame.copy()

        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)
                bbox = tracked.xyxy[i].tolist()
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                cls_name = detections[0].class_name if detections else "unknown"

                # For matching class name from original detections
                for det in detections:
                    if det.class_id == cls_id:
                        cls_name = det.class_name
                        break

                # Update or create tracked object
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = TrackedObject(
                        track_id=track_id,
                        class_name=cls_name,
                        first_seen_frame=frame_idx,
                        last_seen_frame=frame_idx,
                        first_seen_time=current_time,
                        bbox=bbox,
                    )
                    events.append(TrackingEvent(
                        event_type="new_object",
                        track_id=track_id,
                        class_name=cls_name,
                        frame_idx=frame_idx,
                        timestamp=current_time,
                        message=f"New object detected: {cls_name} (ID: {track_id})",
                    ))
                else:
                    obj = self.tracked_objects[track_id]
                    obj.last_seen_frame = frame_idx
                    obj.bbox = bbox
                    obj.dwell_time = current_time - obj.first_seen_time

                    # Check dwell time alert
                    if obj.dwell_time >= self.alert_dwell_seconds and not obj.alerted:
                        obj.alerted = True
                        events.append(TrackingEvent(
                            event_type="dwell_alert",
                            track_id=track_id,
                            class_name=cls_name,
                            frame_idx=frame_idx,
                            timestamp=current_time,
                            message=f"ALERT: {cls_name} (ID: {track_id}) dwelling for "
                                    f"{obj.dwell_time:.1f}s (threshold: {self.alert_dwell_seconds}s)",
                        ))

                # Draw track annotation
                x1, y1, x2, y2 = [int(b) for b in bbox]
                color = (0, 165, 255) if self.tracked_objects.get(track_id, TrackedObject(0, "", 0, 0, 0)).alerted else (0, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                dwell = self.tracked_objects[track_id].dwell_time
                label = f"ID:{track_id} {cls_name} {dwell:.1f}s"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated, events



    def _update_basic(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        current_time: float,
    ) -> Tuple[np.ndarray, List[TrackingEvent]]:
        """IoU-based tracking fallback (when supervision not available)."""
        events = []
        annotated = frame.copy()

        # Match current detections to previous tracks via IoU
        assigned = set()
        current_tracks: List[Tuple[int, list]] = []

        for det in detections:
            best_iou = 0.0
            best_prev_idx = -1

            for prev_idx, (prev_id, prev_bbox) in enumerate(self._prev_detections):
                if prev_idx in assigned:
                    continue
                iou = compute_iou(det.bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_idx = prev_idx

            if best_iou >= 0.3 and best_prev_idx >= 0:
                # Matched to existing track
                assigned.add(best_prev_idx)
                track_id = self._prev_detections[best_prev_idx][0]
            else:
                # New track
                track_id = self._next_id
                self._next_id += 1
                self.tracked_objects[track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=det.class_name,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    first_seen_time=current_time,
                    bbox=det.bbox,
                )
                events.append(TrackingEvent(
                    event_type="new_object",
                    track_id=track_id,
                    class_name=det.class_name,
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    message=f"New object detected: {det.class_name} (ID: {track_id})",
                ))

            # Update tracked object state
            obj = self.tracked_objects[track_id]
            obj.last_seen_frame = frame_idx
            obj.bbox = det.bbox
            obj.dwell_time = current_time - obj.first_seen_time

            if obj.dwell_time >= self.alert_dwell_seconds and not obj.alerted:
                obj.alerted = True
                events.append(TrackingEvent(
                    event_type="dwell_alert",
                    track_id=track_id,
                    class_name=det.class_name,
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    message=f"ALERT: {det.class_name} (ID: {track_id}) dwelling for "
                            f"{obj.dwell_time:.1f}s (threshold: {self.alert_dwell_seconds}s)",
                ))

            current_tracks.append((track_id, det.bbox))

            # Draw annotation
            x1, y1, x2, y2 = [int(b) for b in det.bbox]
            color = (0, 165, 255) if obj.alerted else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {det.class_name} {obj.dwell_time:.1f}s"
            cv2.putText(annotated, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        self._prev_detections = current_tracks
        return annotated, events

    def get_active_objects(self, current_frame: int, max_gap: int = 30) -> List[TrackedObject]:
        """Get currently active tracked objects."""
        return [
            obj for obj in self.tracked_objects.values()
            if current_frame - obj.last_seen_frame <= max_gap
        ]

    def get_alerts(self) -> List[TrackedObject]:
        """Get all objects that have triggered dwell alerts."""
        return [obj for obj in self.tracked_objects.values() if obj.alerted]

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_objects.clear()
        if self._use_supervision:
            self.byte_tracker.reset()
        else:
            self._next_id = 0
            self._prev_detections = []
        self.start_time = time.time()
