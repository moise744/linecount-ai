"""
LineCount AI — Latency and FPS Benchmark Suite.

Benchmarks detection latency, Kalman tracking association latency,
and end-to-end processing throughput on the current hardware.

Usage:
    python scripts/benchmark_fps.py --frames 100 --resolution 1280 720
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from linecount.counting.line_counter import CountingLine, LineCounter
from linecount.detection.detector import Detection, ObjectDetector
from linecount.tracking.tracker import ObjectTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LineCount AI FPS & Latency")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720], help="Width Height")
    parser.add_argument("--num-objects", type=int, default=8, help="Simulated objects per frame")
    args = parser.parse_args()

    width, height = args.resolution
    n_frames = args.frames
    n_objs = args.num_objects

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: LineCount AI Processing Pipeline")
    print(f"Resolution: {width}x{height} | Frames: {n_frames} | Objects/Frame: {n_objs}")
    print(f"{'=' * 60}")

    tracker = ObjectTracker()
    line = CountingLine(p1=(0, height // 2), p2=(width, height // 2))
    counter = LineCounter(line=line)

    tracking_latencies = []
    counting_latencies = []

    # Generate synthetic frames and detections for tracking benchmark
    for f in range(n_frames):
        # Move objects down the frame
        detections = []
        for i in range(n_objs):
            y_pos = (f * 10 + i * 80) % height
            x_pos = 100 + i * 120
            det = Detection(
                bbox=np.array([x_pos - 30, y_pos - 30, x_pos + 30, y_pos + 30], dtype=np.float32),
                confidence=0.95,
                class_id=i % 3,
                class_name=f"item_{i % 3}",
            )
            detections.append(det)

        # 1. Benchmark Tracker
        t0 = time.perf_counter()
        tracks = tracker.update(detections)
        t_track = (time.perf_counter() - t0) * 1000.0
        tracking_latencies.append(t_track)

        # 2. Benchmark Line Counter
        t1 = time.perf_counter()
        counter.update(tracks)
        t_count = (time.perf_counter() - t1) * 1000.0
        counting_latencies.append(t_count)

    mean_track = np.mean(tracking_latencies)
    mean_count = np.mean(counting_latencies)
    total_pipeline_ms = mean_track + mean_count
    pure_tracking_fps = 1000.0 / total_pipeline_ms

    print("\nRESULTS:")
    print(f"  Tracker Mean Latency:  {mean_track:.3f} ms (± {np.std(tracking_latencies):.3f} ms)")
    print(f"  Counter Mean Latency:  {mean_count:.3f} ms (± {np.std(counting_latencies):.3f} ms)")
    print(f"  Combined Latency:      {total_pipeline_ms:.3f} ms")
    print(f"  Tracking Throughput:   {pure_tracking_fps:.1f} FPS (without detector overhead)")
    print(f"  Total Items Counted:   {counter.total_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
