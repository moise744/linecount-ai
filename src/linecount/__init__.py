"""LineCount AI — Real-Time Industrial Object Detection, Tracking and Counting.

A computer vision system for automated conveyor belt inspection,
multi-class product tracking, and bidirectional counting.
"""

__version__ = "0.1.0"
__author__ = "Habimana Moise"

from linecount.detection.detector import ObjectDetector, Detection
from linecount.tracking.tracker import ObjectTracker, Track
from linecount.counting.line_counter import LineCounter, CountingLine, CountingDirection
from linecount.counting.statistics import ThroughputMonitor

__all__ = [
    "ObjectDetector",
    "Detection",
    "ObjectTracker",
    "Track",
    "LineCounter",
    "CountingLine",
    "CountingDirection",
    "ThroughputMonitor",
]
