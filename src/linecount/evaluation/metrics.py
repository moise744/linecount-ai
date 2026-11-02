"""
Evaluation Metrics for Industrial Tracking and Conveyor Counting.

Provides standard quantitative metrics:
- Counting Accuracy & Error Rates (MAE, MAPE)
- Tracking Consistency & Identity Preservation
- Detection performance stats
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def compute_counting_metrics(
    ground_truth_counts: Dict[str, int],
    predicted_counts: Dict[str, int],
) -> Dict[str, float]:
    """Compute precision metrics comparing predicted line counts with verified ground truth.

    Args:
        ground_truth_counts: Mapping of {class_or_total: count}.
        predicted_counts: Mapping of {class_or_total: count}.

    Returns:
        Dictionary of evaluation metrics:
        - accuracy (0.0 to 1.0)
        - absolute_error
        - percentage_error (MAPE)
        - undercount
        - overcount
    """
    total_gt = sum(ground_truth_counts.values()) if ground_truth_counts else 0
    total_pred = sum(predicted_counts.values()) if predicted_counts else 0

    abs_err = abs(total_pred - total_gt)
    mape = (abs_err / total_gt) * 100.0 if total_gt > 0 else 0.0
    accuracy = max(0.0, 1.0 - (abs_err / total_gt)) if total_gt > 0 else 1.0

    undercount = max(0, total_gt - total_pred)
    overcount = max(0, total_pred - total_gt)

    return {
        "ground_truth_total": total_gt,
        "predicted_total": total_pred,
        "absolute_error": abs_err,
        "percentage_error": round(mape, 2),
        "accuracy": round(accuracy, 4),
        "undercount": undercount,
        "overcount": overcount,
    }


def compute_tracking_summary(
    total_tracks_instantiated: int,
    total_counted: int,
    mean_track_length: float,
    id_switches: int = 0,
) -> Dict[str, object]:
    """Compile summary of tracking behavior."""
    return {
        "total_unique_tracks": total_tracks_instantiated,
        "total_items_counted": total_counted,
        "mean_track_length_frames": round(mean_track_length, 1),
        "id_switches": id_switches,
    }
