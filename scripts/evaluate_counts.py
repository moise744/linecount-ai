"""
LineCount AI — Evaluation Script.

Compares predicted counts vs. manually verified ground truth counts,
and computes counting accuracy, MAE, and MAPE metrics.

Usage:
    python scripts/evaluate_counts.py --ground-truth data/ground_truth.json \\
                                      --predictions outputs/predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linecount.evaluation.metrics import compute_counting_metrics, compute_tracking_summary


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LineCount AI counting accuracy")
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="JSON file with ground truth counts {class: count}"
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="JSON file with predicted counts {class: count}"
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data to verify the metric pipeline"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo:
        print("[INFO] Running in demo mode with synthetic counts.")
        ground_truth = {"product_A": 120, "product_B": 85, "product_C": 45}
        predictions = {"product_A": 118, "product_B": 87, "product_C": 44}
    else:
        if args.ground_truth is None or args.predictions is None:
            print("[ERROR] Provide --ground-truth and --predictions, or use --demo")
            sys.exit(1)

        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)
        with open(args.predictions, "r") as f:
            predictions = json.load(f)

    # Count metrics
    metrics = compute_counting_metrics(ground_truth, predictions)

    print(f"\n{'='*55}")
    print(f"  LINECOUNT AI — COUNTING ACCURACY REPORT")
    print(f"{'='*55}")
    print(f"  Ground Truth Total  : {metrics['ground_truth_total']}")
    print(f"  Predicted Total     : {metrics['predicted_total']}")
    print(f"  Absolute Error      : {metrics['absolute_error']}")
    print(f"  MAPE (%)            : {metrics['percentage_error']:.2f}%")
    print(f"  Counting Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']:.1%})")
    print(f"  Undercount          : {metrics['undercount']}")
    print(f"  Overcount           : {metrics['overcount']}")
    print(f"{'='*55}")

    # Per-class breakdown
    print(f"\n  Per-Class Comparison:")
    all_keys = set(ground_truth) | set(predictions)
    for k in sorted(all_keys):
        gt = ground_truth.get(k, 0)
        pr = predictions.get(k, 0)
        err = abs(pr - gt)
        print(f"    {k:<20}  GT={gt:4d}  Pred={pr:4d}  Err={err:3d}")

    print()


if __name__ == "__main__":
    main()
