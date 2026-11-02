"""
Industrial Throughput Monitor & Analytics.

Calculates real-time conveyor metrics:
- Items per minute (PPM)
- Items per hour (PPH)
- Moving average throughput
- Instantaneous and average FPS
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ThroughputSnapshot:
    """Snapshot of production rate at a point in time."""
    timestamp: float
    total_items: int
    items_per_minute: float
    items_per_hour: float
    fps: float


class ThroughputMonitor:
    """Computes real-time production throughput and processing speeds.

    Args:
        window_seconds: Rolling time window over which to compute rate.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self.start_time = time.time()
        self.event_timestamps: deque[float] = deque()
        self.frame_timestamps: deque[float] = deque(maxlen=120)
        self.total_count = 0

    def record_item(self, count: int = 1) -> None:
        """Record newly counted item(s)."""
        now = time.time()
        for _ in range(count):
            self.event_timestamps.append(now)
            self.total_count += 1
        self._prune_old_events(now)

    def record_frame(self) -> None:
        """Record video processing frame tick for FPS calculation."""
        self.frame_timestamps.append(time.time())

    def _prune_old_events(self, current_time: float) -> None:
        """Discard event timestamps outside the moving time window."""
        cutoff = current_time - self.window
        while self.event_timestamps and self.event_timestamps[0] < cutoff:
            self.event_timestamps.popleft()

    def get_items_per_minute(self) -> float:
        """Calculate items per minute within the current rolling window."""
        now = time.time()
        self._prune_old_events(now)
        active_duration = min(self.window, max(1.0, now - self.start_time))
        return (len(self.event_timestamps) / active_duration) * 60.0

    def get_items_per_hour(self) -> float:
        """Calculate projected items per hour."""
        return self.get_items_per_minute() * 60.0

    def get_current_fps(self) -> float:
        """Calculate instantaneous processing FPS based on recent frames."""
        if len(self.frame_timestamps) < 2:
            return 0.0
        delta = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if delta <= 0:
            return 0.0
        return (len(self.frame_timestamps) - 1) / delta

    def get_snapshot(self) -> ThroughputSnapshot:
        """Get complete snapshot of current conveyor throughput metrics."""
        return ThroughputSnapshot(
            timestamp=time.time(),
            total_items=self.total_count,
            items_per_minute=round(self.get_items_per_minute(), 1),
            items_per_hour=round(self.get_items_per_hour(), 0),
            fps=round(self.get_current_fps(), 1),
        )

    def reset(self) -> None:
        """Reset the monitor."""
        self.start_time = time.time()
        self.event_timestamps.clear()
        self.frame_timestamps.clear()
        self.total_count = 0
