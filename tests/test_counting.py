"""Unit tests for virtual line crossing logic and anti-double counting."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linecount.counting.line_counter import CountingDirection, CountingLine, LineCounter
from linecount.counting.statistics import ThroughputMonitor
from linecount.tracking.tracker import Track


class TestLineCounter:
    def test_line_crossing_detection(self):
        # Horizontal line at Y=100 from X=0 to X=200
        line = CountingLine(p1=(0.0, 100.0), p2=(200.0, 100.0), name="horizontal_gate")
        counter = LineCounter(line=line, direction=CountingDirection.ANY)

        # Track moving from Y=90 to Y=110 (crosses Y=100)
        track = Track(
            track_id=1,
            bbox=np.array([40, 100, 60, 120], dtype=np.float32),
            class_id=0,
            class_name="bottle",
            confidence=0.95,
            trajectory=[(50.0, 90.0), (50.0, 110.0)],
        )

        events = counter.update([track])
        assert len(events) == 1
        assert counter.total_count == 1
        assert counter.class_counts["bottle"] == 1

    def test_anti_double_counting(self):
        line = CountingLine(p1=(0.0, 100.0), p2=(200.0, 100.0))
        counter = LineCounter(line=line)

        track = Track(
            track_id=42,
            bbox=np.array([40, 100, 60, 120], dtype=np.float32),
            class_id=0,
            class_name="can",
            confidence=0.91,
            trajectory=[(50.0, 90.0), (50.0, 110.0)],
        )

        # First crossing
        counter.update([track])
        assert counter.total_count == 1

        # Subsequent frames with same track continuing past the line
        track.trajectory.append((50.0, 130.0))
        events2 = counter.update([track])
        assert len(events2) == 0
        assert counter.total_count == 1  # Never double-counted!

    def test_no_crossing(self):
        line = CountingLine(p1=(0.0, 100.0), p2=(200.0, 100.0))
        counter = LineCounter(line=line)

        # Track moves parallel to line from Y=50 to Y=60 (never reaches 100)
        track = Track(
            track_id=5,
            bbox=np.array([10, 50, 30, 70], dtype=np.float32),
            class_id=0,
            class_name="bottle",
            confidence=0.88,
            trajectory=[(20.0, 50.0), (20.0, 60.0)],
        )

        events = counter.update([track])
        assert len(events) == 0
        assert counter.total_count == 0


class TestThroughputMonitor:
    def test_throughput_recording(self):
        monitor = ThroughputMonitor(window_seconds=10.0)
        monitor.record_item(5)
        monitor.record_frame()

        snap = monitor.get_snapshot()
        assert snap.total_items == 5
        assert snap.items_per_minute > 0
