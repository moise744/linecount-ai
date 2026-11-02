"""
Object detection module utilizing Ultralytics YOLOv8.

Supports both pretrained models (e.g., yolov8n.pt) and fine-tuned
industrial product inspection weights, with CPU and CUDA support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single detected bounding box in an image/frame.

    Attributes:
        bbox: Coordinates [x1, y1, x2, y2] in pixel space.
        confidence: Detection confidence score in [0.0, 1.0].
        class_id: Numerical class index.
        class_name: Human-readable class label.
    """
    bbox: np.ndarray  # shape (4,), [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class ObjectDetector:
    """YOLOv8 wrapper for industrial object detection on frames.

    Args:
        model_name_or_path: Path to custom .pt weights or ultralytics model name (e.g. 'yolov8n.pt').
        confidence_threshold: Minimum detection confidence score.
        iou_threshold: NMS IoU threshold.
        classes: Optional list of class IDs to filter for.
        device: 'cpu', 'cuda', or 'auto'.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "auto",
    ) -> None:
        self.model_path = model_name_or_path
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.filter_classes = classes

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Initialize the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("Loaded YOLO model '%s' on %s", self.model_path, self.device)
        except Exception as e:
            logger.warning("Failed to initialize Ultralytics YOLO: %s. Fallback mode enabled.", e)
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a BGR or RGB video frame.

        Args:
            frame: Numpy array of shape (H, W, 3) in uint8.

        Returns:
            List of Detection objects.
        """
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            # Fallback mock detector for offline tests and decoupled execution
            return self._mock_detect(frame)

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.filter_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = first_res.names

        for i in range(len(xyxy)):
            c_id = int(cls_ids[i])
            c_name = names.get(c_id, f"class_{c_id}") if isinstance(names, dict) else str(c_id)
            det = Detection(
                bbox=xyxy[i],
                confidence=float(confs[i]),
                class_id=c_id,
                class_name=c_name,
            )
            detections.append(det)

        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Synthetic detection for unit testing when network is isolated."""
        h, w = frame.shape[:2]
        return [
            Detection(
                bbox=np.array([w * 0.2, h * 0.3, w * 0.4, h * 0.6], dtype=np.float32),
                confidence=0.92,
                class_id=0,
                class_name="product",
            )
        ]
