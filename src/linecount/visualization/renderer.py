"""
Conveyor Belt Video & Frame Overlay Renderer.

Annotates frames with:
- Persistent Track IDs, bounding boxes, and confidence scores
- Trajectory path breadcrumbs
- Virtual counting lines with status indicators
- Production metrics HUD (total count, throughput, FPS)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from linecount.counting.line_counter import CountingLine, LineCounter
from linecount.counting.statistics import ThroughputSnapshot
from linecount.tracking.tracker import Track

# Distinct aesthetic palette for tracks
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Get stable RGB/BGR color tuple for a given integer track ID."""
    return PALETTE[track_id % len(PALETTE)]


class ConveyorRenderer:
    """Renders visual inspection overlays on video frames.

    Args:
        line_thickness: Bounding box line thickness.
        font_scale: Text font scale.
    """

    def __init__(self, line_thickness: int = 2, font_scale: float = 0.55) -> None:
        self.thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        counter: Optional[LineCounter] = None,
        snapshot: Optional[ThroughputSnapshot] = None,
    ) -> np.ndarray:
        """Render complete conveyor inspection HUD onto a copy of the frame.

        Args:
            frame: Base BGR frame (H, W, 3).
            tracks: List of active Track instances.
            counter: LineCounter instance holding virtual line and counts.
            snapshot: ThroughputSnapshot holding real-time production rates.

        Returns:
            Annotated BGR frame copy.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Virtual Counting Line
        if counter is not None:
            line = counter.line
            pt1 = (int(line.x1), int(line.y1))
            pt2 = (int(line.x2), int(line.y2))

            # Draw outer glow line
            cv2.line(canvas, pt1, pt2, (0, 255, 255), self.thickness + 2, cv2.LINE_AA)
            # Draw core line
            cv2.line(canvas, pt1, pt2, (0, 200, 0), self.thickness, cv2.LINE_AA)

            # Label on the line
            mid_x = int((line.x1 + line.x2) / 2.0)
            mid_y = int((line.y1 + line.y2) / 2.0) - 10
            cv2.putText(
                canvas,
                f"COUNT LINE: {line.name.upper()}",
                (mid_x, mid_y),
                self.font,
                self.font_scale * 0.9,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Active Tracks & Trajectories
        for trk in tracks:
            color = get_color_for_id(trk.track_id)
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.thickness, cv2.LINE_AA)

            # Trajectory breadcrumbs
            if len(trk.trajectory) > 1:
                pts = np.array(trk.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)

            # Badge banner
            label = f"#{trk.track_id} {trk.class_name} {trk.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, self.font, self.font_scale, 1)

            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1
            cv2.rectangle(canvas, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, badge_y2 - baseline - 2),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Render Top Dashboard HUD
        self._render_hud(canvas, counter, snapshot)

        return canvas

    def _render_hud(
        self,
        canvas: np.ndarray,
        counter: Optional[LineCounter],
        snapshot: Optional[ThroughputSnapshot],
    ) -> None:
        """Render telemetry banner at the top of the video canvas."""
        w = canvas.shape[1]
        hud_h = 52

        # Semi-transparent dark banner overlay
        sub = canvas[0:hud_h, 0:w]
        dark_rect = np.zeros_like(sub)
        canvas[0:hud_h, 0:w] = cv2.addWeighted(sub, 0.35, dark_rect, 0.65, 0)

        # Bottom accent border line
        cv2.line(canvas, (0, hud_h), (w, hud_h), (0, 180, 240), 2)

        # Left: Project Brand
        cv2.putText(
            canvas,
            "LINECOUNT AI | CONVEYOR TELEMETRY",
            (16, 22),
            self.font,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Counters & Rates
        total_items = counter.total_count if counter else 0
        ppm = snapshot.items_per_minute if snapshot else 0.0
        fps = snapshot.fps if snapshot else 0.0

        metrics_text = f"TOTAL: {total_items}   |   PPM: {ppm:.1f}   |   FPS: {fps:.1f}"
        cv2.putText(
            canvas,
            metrics_text,
            (16, 42),
            self.font,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
