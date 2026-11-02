"""Unit tests for Kalman filtering, IoU association, and ObjectTracker."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linecount.detection.detector import Detection
from linecount.tracking.kalman import KalmanBoxTracker
from linecount.tracking.tracker import ObjectTracker, bbox_iou, iou_batch


class TestIoU:
    def test_exact_overlap(self):
        b1 = np.array([0, 0, 10, 10], dtype=np.float32)
        assert bbox_iou(b1, b1) == 1.0

    def test_no_overlap(self):
        b1 = np.array([0, 0, 10, 10], dtype=np.float32)
        b2 = np.array([20, 20, 30, 30], dtype=np.float32)
        assert bbox_iou(b1, b2) == 0.0

    def test_partial_overlap(self):
        b1 = np.array([0, 0, 10, 10], dtype=np.float32)
        b2 = np.array([5, 0, 15, 10], dtype=np.float32)
        # intersection: 5*10 = 50, union: 100 + 100 - 50 = 150 -> 1/3
        assert pytest.approx(bbox_iou(b1, b2), rel=1e-3) == 1.0 / 3.0


class TestKalmanTracker:
    def test_kalman_predict_and_update(self):
        box = np.array([100.0, 100.0, 150.0, 150.0], dtype=np.float32)
        tracker = KalmanBoxTracker(box)

        # Predict next location
        pred = tracker.predict()
        assert len(pred) == 4
        assert pred[2] > pred[0]
        assert pred[3] > pred[1]

        # Update with new measurement
        new_box = np.array([105.0, 105.0, 155.0, 155.0], dtype=np.float32)
        tracker.update(new_box)
        assert tracker.hits == 2
        assert tracker.hit_streak == 2


class TestObjectTracker:
    def test_track_continuity(self):
        tracker = ObjectTracker(max_age=5, min_hits=2, iou_threshold=0.3)

        # Frame 1: Detection at (100, 100)
        d1 = [Detection(np.array([100, 100, 140, 140], dtype=np.float32), 0.9, 0, "part")]
        tracks1 = tracker.update(d1)
        # In frame 1, age=1, hits=1 (not yet min_hits=2, but frame_count <= min_hits allows track)
        assert len(tracks1) >= 1
        t_id = tracks1[0].track_id

        # Frame 2: Object moves slightly to (105, 105)
        d2 = [Detection(np.array([105, 105, 145, 145], dtype=np.float32), 0.9, 0, "part")]
        tracks2 = tracker.update(d2)

        assert len(tracks2) == 1
        assert tracks2[0].track_id == t_id  # Persistent identity maintained!
        assert len(tracks2[0].trajectory) >= 2

    def test_lost_track_cleanup(self):
        tracker = ObjectTracker(max_age=2, min_hits=1)
        d = [Detection(np.array([50, 50, 90, 90], dtype=np.float32), 0.9, 0, "part")]
        tracker.update(d)

        # Feed empty frames
        tracker.update([])
        tracker.update([])
        tracker.update([])

        assert len(tracker.trackers) == 0
