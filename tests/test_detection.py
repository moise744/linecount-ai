"""Unit tests for Detection and ObjectDetector modules."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linecount.detection.detector import Detection, ObjectDetector


class TestDetection:
    def test_detection_properties(self):
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        det = Detection(bbox=bbox, confidence=0.88, class_id=1, class_name="bottle")

        assert det.x1 == 10.0
        assert det.y1 == 20.0
        assert det.x2 == 50.0
        assert det.y2 == 80.0
        assert det.width == 40.0
        assert det.height == 60.0
        assert det.area == 2400.0
        assert det.center == (30.0, 50.0)

    def test_empty_frame_detection(self):
        detector = ObjectDetector(model_name_or_path="yolov8n.pt")
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector.detect(empty) == []

    def test_mock_detection_fallback(self):
        detector = ObjectDetector(model_name_or_path="nonexistent_model.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert len(dets) >= 1
        assert dets[0].confidence > 0.0
