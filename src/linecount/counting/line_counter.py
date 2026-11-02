"""
Virtual Line Crossing Counter with Anti-Double Counting Protection.

Detects when an object track crosses a defined line segment on a conveyor
belt and maintains per-class, directional, and aggregate counts.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from linecount.tracking.tracker import Track


class CountingDirection(str, enum.Enum):
    """Direction of object motion relative to the oriented line."""
    INCOMING = "incoming"      # Left-to-right or top-to-bottom
    OUTGOING = "outgoing"      # Right-to-left or bottom-to-top
    ANY = "any"                # Cross in either direction


@dataclass
class CountingLine:
    """Represents a virtual line defined across a conveyor belt.

    Points p1=(x1, y1) and p2=(x2, y2) define an oriented line segment.
    """
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    name: str = "main_line"

    @property
    def x1(self) -> float:
        return self.p1[0]

    @property
    def y1(self) -> float:
        return self.p1[1]

    @property
    def x2(self) -> float:
        return self.p2[0]

    @property
    def y2(self) -> float:
        return self.p2[1]

    def signed_distance(self, pt: Tuple[float, float]) -> float:
        """Compute the signed 2D cross product distance from point pt to oriented line p1->p2.

        Positive if pt is on one side, negative if on the other.
        """
        return (self.x2 - self.x1) * (pt[1] - self.y1) - (self.y2 - self.y1) * (pt[0] - self.x1)

    def intersects_segment(
        self, seg_start: Tuple[float, float], seg_end: Tuple[float, float]
    ) -> bool:
        """Check if segment (seg_start -> seg_end) intersects line segment (p1 -> p2)."""
        def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = self.p1, self.p2
        C, D = seg_start, seg_end
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class LineCounter:
    """Tracks and counts objects passing through virtual lines.

    Includes robust anti-double-counting mechanisms, trajectory lookback,
    and per-class statistics.
    """

    def __init__(
        self,
        line: CountingLine,
        direction: CountingDirection = CountingDirection.ANY,
    ) -> None:
        self.line = line
        self.direction = direction

        # Persistent counts
        self.total_count = 0
        self.class_counts: Dict[str, int] = {}
        self.in_count = 0
        self.out_count = 0

        # Anti-double count registry: track IDs that have crossed
        self._counted_ids: Set[int] = set()

        # Detailed event log
        self.crossing_events: List[Dict[str, object]] = []

    def update(self, tracks: List[Track]) -> List[Dict[str, object]]:
        """Process active tracks and increment counts for newly crossed items.

        Args:
            tracks: List of active tracks in the current frame.

        Returns:
            List of crossing events triggered in this frame.
        """
        new_events: List[Dict[str, object]] = []

        for track in tracks:
            t_id = track.track_id
            if t_id in self._counted_ids:
                continue

            traj = track.trajectory
            if len(traj) < 2:
                continue

            # Check crossing between previous position and current position
            p_prev = traj[-2]
            p_curr = traj[-1]

            crossed = self.line.intersects_segment(p_prev, p_curr)
            if not crossed:
                # Also check slightly longer history (up to 3 steps) for fast moving items
                if len(traj) >= 4:
                    crossed = self.line.intersects_segment(traj[-4], p_curr)

            if crossed:
                # Determine crossing direction
                d_prev = self.line.signed_distance(p_prev)
                d_curr = self.line.signed_distance(p_curr)

                is_incoming = d_prev < 0 and d_curr >= 0
                is_outgoing = d_prev > 0 and d_curr <= 0

                event_dir = CountingDirection.INCOMING if is_incoming else CountingDirection.OUTGOING

                # Check direction filter
                if self.direction != CountingDirection.ANY and self.direction != event_dir:
                    continue

                # Register crossing event
                self._counted_ids.add(t_id)
                self.total_count += 1

                c_name = track.class_name
                self.class_counts[c_name] = self.class_counts.get(c_name, 0) + 1

                if event_dir == CountingDirection.INCOMING:
                    self.in_count += 1
                else:
                    self.out_count += 1

                event_data = {
                    "track_id": t_id,
                    "class_name": c_name,
                    "class_id": track.class_id,
                    "direction": event_dir.value,
                    "position": p_curr,
                    "total_count": self.total_count,
                }
                self.crossing_events.append(event_data)
                new_events.append(event_data)

        return new_events

    def reset(self) -> None:
        """Reset all counters and memory."""
        self.total_count = 0
        self.class_counts.clear()
        self.in_count = 0
        self.out_count = 0
        self._counted_ids.clear()
        self.crossing_events.clear()

    def get_summary(self) -> Dict[str, object]:
        """Return aggregated counting summary."""
        return {
            "total_count": self.total_count,
            "in_count": self.in_count,
            "out_count": self.out_count,
            "class_breakdown": dict(self.class_counts),
            "line_name": self.line.name,
        }
