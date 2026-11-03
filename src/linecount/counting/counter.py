"""
Compatibility shims for linecount modules.
Provides alias names used by notebooks and documentation.

Real class names:
  LineCounter / CountingLine  → in linecount.counting.line_counter
  ObjectTracker               → in linecount.tracking.tracker
  compute_counting_metrics    → in linecount.evaluation.metrics
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# ── Re-export real classes under alias names ──────────────────────────────────
from linecount.counting.line_counter import (
    LineCounter,
    CountingLine,
    CountingDirection,
)
from linecount.tracking.tracker import ObjectTracker, Track
from linecount.evaluation.metrics import compute_counting_metrics


# ── High-level façade: LineCrossingCounter ────────────────────────────────────
class LineCrossingCounter:
    """
    High-level façade wrapping LineCounter + CountingLine.

    API used by notebooks and app/api.py:
        counter = LineCrossingCounter(line_start=(x1,y1), line_end=(x2,y2))
        counter.update(tracks)
        count = counter.count
        counts_dict = counter.get_counts()
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        name: str = "main_line",
    ) -> None:
        self.p1 = (float(line_start[0]), float(line_start[1]))
        self.p2 = (float(line_end[0]), float(line_end[1]))
        self._line = CountingLine(p1=self.p1, p2=self.p2, name=name)
        self._counter = LineCounter(line=self._line, direction=CountingDirection.ANY)
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

    def update(self, tracks: Union[List[Track], List[Dict[str, Any]]]) -> None:
        """Update counter with current frame tracks (supports Track objects or dicts)."""
        valid_tracks: List[Track] = []
        for t in tracks:
            if isinstance(t, Track):
                valid_tracks.append(t)
            elif isinstance(t, dict):
                tid = int(t.get("track_id", t.get("id", 0)))
                bbox = t.get("bbox", [0, 0, 10, 10])
                if isinstance(bbox, (list, tuple)):
                    bbox_np = np.array(bbox, dtype=np.float32)
                else:
                    bbox_np = np.asarray(bbox, dtype=np.float32)
                cx = float((bbox_np[0] + bbox_np[2]) / 2.0)
                cy = float((bbox_np[1] + bbox_np[3]) / 2.0)

                # Maintain history for dicts
                prev_c = self._prev_centroids.get(tid, (cx, cy))
                self._prev_centroids[tid] = (cx, cy)
                traj = [prev_c, (cx, cy)]

                track_obj = Track(
                    track_id=tid,
                    bbox=bbox_np,
                    class_id=int(t.get("class_id", 0)),
                    class_name=str(t.get("class_name", t.get("name", "object"))),
                    confidence=float(t.get("confidence", t.get("conf", 1.0))),
                    trajectory=traj,
                )
                valid_tracks.append(track_obj)

        if valid_tracks:
            self._counter.update(valid_tracks)

    @property
    def count(self) -> int:
        return self._counter.total_count

    @property
    def total(self) -> int:
        return self._counter.total_count

    def get_counts(self) -> Dict[str, int]:
        """Return per-class crossing counts."""
        return dict(self._counter.class_counts)


# ── ByteTracker alias ────────────────────────────────────────────────────────
class ByteTracker(ObjectTracker):
    """Alias for ObjectTracker, used by notebooks."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
        )


# ── CountingEvaluator alias ──────────────────────────────────────────────────
class CountingEvaluator:
    """
    Evaluates counting accuracy against ground-truth counts.
    Used in notebooks and scripts.
    """

    def evaluate(
        self,
        predicted: List[int],
        ground_truth: List[int],
    ) -> dict:
        """Compute counting error metrics."""
        pred = [max(0, p) for p in predicted]
        gt = [max(0, g) for g in ground_truth]
        errors = [abs(p - g) for p, g in zip(pred, gt)]
        rel_errors = [e / max(g, 1) for e, g in zip(errors, gt)]
        return {
            "mae": round(float(np.mean(errors)), 4) if errors else 0.0,
            "max_error": int(max(errors)) if errors else 0,
            "mean_relative_error": round(float(np.mean(rel_errors)), 4) if rel_errors else 0.0,
            "exact_match_rate": round(sum(1 for e in errors if e == 0) / len(errors), 4) if errors else 1.0,
            "n_samples": len(pred),
        }
