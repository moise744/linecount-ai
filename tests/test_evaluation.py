"""Unit tests for LineCount AI evaluation metrics."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linecount.evaluation.metrics import compute_counting_metrics, compute_tracking_summary


class TestCountingMetrics:
    def test_perfect_count(self):
        gt = {"product_A": 100, "product_B": 50}
        pred = {"product_A": 100, "product_B": 50}
        result = compute_counting_metrics(gt, pred)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["absolute_error"] == 0
        assert result["undercount"] == 0
        assert result["overcount"] == 0

    def test_overcount(self):
        gt = {"total": 100}
        pred = {"total": 110}
        result = compute_counting_metrics(gt, pred)
        assert result["overcount"] == 10
        assert result["undercount"] == 0
        assert result["absolute_error"] == 10

    def test_undercount(self):
        gt = {"total": 100}
        pred = {"total": 90}
        result = compute_counting_metrics(gt, pred)
        assert result["undercount"] == 10
        assert result["overcount"] == 0

    def test_empty_predictions(self):
        gt = {"product": 50}
        pred = {}
        result = compute_counting_metrics(gt, pred)
        assert result["predicted_total"] == 0
        assert result["accuracy"] == pytest.approx(0.0)

    def test_mape_calculation(self):
        gt = {"a": 100}
        pred = {"a": 90}
        result = compute_counting_metrics(gt, pred)
        assert result["percentage_error"] == pytest.approx(10.0)

    def test_zero_ground_truth(self):
        # Should not divide by zero
        gt = {}
        pred = {}
        result = compute_counting_metrics(gt, pred)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["percentage_error"] == pytest.approx(0.0)


class TestTrackingSummary:
    def test_basic_summary(self):
        result = compute_tracking_summary(
            total_tracks_instantiated=42,
            total_counted=38,
            mean_track_length=14.5,
            id_switches=2,
        )
        assert result["total_unique_tracks"] == 42
        assert result["total_items_counted"] == 38
        assert result["mean_track_length_frames"] == pytest.approx(14.5)
        assert result["id_switches"] == 2

    def test_zero_id_switches_default(self):
        result = compute_tracking_summary(10, 9, 8.0)
        assert result["id_switches"] == 0
