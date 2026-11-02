"""
SORT/ByteTrack-style Multi-Object Tracker.

Associates detections across frames using IoU overlap and Kalman Filtering
to maintain consistent identity IDs for conveyor products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from linecount.detection.detector import Detection
from linecount.tracking.kalman import KalmanBoxTracker


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area2 = max(0.0, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between N boxes and M boxes."""
    N = len(bboxes1)
    M = len(bboxes2)
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = bbox_iou(bboxes1[i], bboxes2[j])
    return iou_matrix


@dataclass
class Track:
    """Represents an active or confirmed track in the video feed."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        return (float((self.bbox[0] + self.bbox[2]) / 2.0),
                float((self.bbox[1] + self.bbox[3]) / 2.0))


class ObjectTracker:
    """Multi-object tracker using Kalman Filter and Hungarian IoU association.

    Args:
        max_age: Maximum frames a track can miss detections before deletion.
        min_hits: Minimum detection hits before track is confirmed.
        iou_threshold: Minimum IoU overlap to consider an association.
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.meta: Dict[int, Dict[str, object]] = {}
        self.frame_count = 0

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Update tracker with detections from the current frame.

        Args:
            detections: List of current frame detections.

        Returns:
            List of confirmed active Tracks.
        """
        self.frame_count += 1

        # 1. Predict new locations of existing tracks
        predicted_boxes: List[np.ndarray] = []
        for trk in self.trackers:
            p_box = trk.predict()
            predicted_boxes.append(p_box)

        # Remove dead or invalid trackers (NaNs)
        valid_indices = [
            i for i, b in enumerate(predicted_boxes)
            if not np.any(np.isnan(b)) and b[2] > b[0] and b[3] > b[1]
        ]
        self.trackers = [self.trackers[i] for i in valid_indices]
        predicted_boxes = [predicted_boxes[i] for i in valid_indices]

        # 2. Extract bounding boxes from detections
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32) if detections else np.empty((0, 4))

        # 3. Associate detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, np.array(predicted_boxes, dtype=np.float32) if predicted_boxes else np.empty((0, 4))
        )

        # 4. Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(det_boxes[d_idx])
            t_id = self.trackers[t_idx].id
            det = detections[d_idx]
            self.meta[t_id]["class_id"] = det.class_id
            self.meta[t_id]["class_name"] = det.class_name
            self.meta[t_id]["confidence"] = det.confidence

        # 5. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            trk = KalmanBoxTracker(det.bbox)
            self.trackers.append(trk)
            self.meta[trk.id] = {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "trajectory": [],
            }

        # 6. Gather active confirmed tracks and remove aged-out ones
        active_tracks: List[Track] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            d = trk.get_state()
            meta = self.meta.get(trk.id, {
                "class_id": 0, "class_name": "unknown", "confidence": 0.0, "trajectory": []
            })
            center = (float((d[0] + d[2]) / 2.0), float((d[1] + d[3]) / 2.0))
            traj: list = meta.get("trajectory", [])
            traj.append(center)
            # Keep trajectory bounded
            if len(traj) > 60:
                traj.pop(0)
            meta["trajectory"] = traj

            if (trk.time_since_update <= 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                track = Track(
                    track_id=trk.id,
                    bbox=d,
                    class_id=int(meta["class_id"]),
                    class_name=str(meta["class_name"]),
                    confidence=float(meta["confidence"]),
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    trajectory=list(traj),
                )
                active_tracks.append(track)

            if trk.time_since_update <= self.max_age:
                surviving_trackers.append(trk)
            else:
                self.meta.pop(trk.id, None)

        self.trackers = surviving_trackers
        return active_tracks

    def _associate(
        self, det_boxes: np.ndarray, trk_boxes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks via Hungarian matching on IoU matrix."""
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = iou_batch(det_boxes, trk_boxes)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = set(range(len(det_boxes)))
        unmatched_trks = set(range(len(trk_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)

        return matched, sorted(list(unmatched_dets)), sorted(list(unmatched_trks))

    def reset(self) -> None:
        """Reset internal trackers and state."""
        self.trackers.clear()
        self.meta.clear()
        self.frame_count = 0
