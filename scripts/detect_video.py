"""
LineCount AI — Video Processing & Conveyor Counting CLI.

Processes an input video file or live RTSP camera stream, runs real-time
object detection and multi-object tracking, records virtual line crossings,
and optionally saves the rendered output video with HUD.

Usage:
    python scripts/detect_video.py --video path/to/conveyor.mp4 --output outputs/tracked.mp4
    python scripts/detect_video.py --config configs/conveyor.yaml --video 0
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import yaml

from linecount.counting.line_counter import CountingDirection, CountingLine, LineCounter
from linecount.counting.statistics import ThroughputMonitor
from linecount.detection.detector import ObjectDetector
from linecount.tracking.tracker import ObjectTracker
from linecount.visualization.renderer import ConveyorRenderer


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LineCount AI on a video source")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or webcam index (e.g. 0)")
    parser.add_argument("--config", type=Path, default=Path("configs/tracking.yaml"), help="YAML config")
    parser.add_argument("--output", type=Path, default=None, help="Path to save annotated MP4 video")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--line-y", type=float, default=None, help="Override Y coord for horizontal line")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # 1. Initialize Components
    detector = ObjectDetector(
        model_name_or_path=args.weights,
        confidence_threshold=args.conf,
    )
    tracker = ObjectTracker(
        max_age=cfg.get("tracker", {}).get("max_age", 20),
        min_hits=cfg.get("tracker", {}).get("min_hits", 2),
        iou_threshold=cfg.get("tracker", {}).get("iou_threshold", 0.3),
    )
    monitor = ThroughputMonitor()
    renderer = ConveyorRenderer()

    # Video Source
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.video}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Line setup
    line_y = args.line_y if args.line_y is not None else float(height // 2)
    counting_line = CountingLine(p1=(0.0, line_y), p2=(float(width), line_y), name="conveyor_gate")
    counter = LineCounter(line=counting_line, direction=CountingDirection.ANY)

    # Video Writer
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    print(f"\n[INFO] Starting processing video: {args.video} ({width}x{height} @ {fps:.1f} FPS)")
    frame_idx = 0
    t_start = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            monitor.record_frame()

            # Detection & Tracking
            detections = detector.detect(frame)
            active_tracks = tracker.update(detections)

            # Line Counting
            new_events = counter.update(active_tracks)
            if new_events:
                monitor.record_item(len(new_events))

            # Visualization
            annotated = renderer.render(frame, active_tracks, counter, monitor.get_snapshot())

            if writer:
                writer.write(annotated)

            if frame_idx % 30 == 0:
                snap = monitor.get_snapshot()
                print(
                    f"Frame {frame_idx:5d} | Count: {counter.total_count:4d} | "
                    f"Rate: {snap.items_per_minute:5.1f} PPM | FPS: {snap.fps:4.1f}"
                )

    finally:
        cap.release()
        if writer:
            writer.release()

    elapsed = time.time() - t_start
    summary = counter.get_summary()
    print("\n" + "=" * 60)
    print("CONVEYOR INSPECTION RUN SUMMARY")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Time Elapsed:     {elapsed:.2f} s")
    print(f"Average Throughput FPS: {frame_idx / max(0.001, elapsed):.1f}")
    print(f"Total Items Counted:    {summary['total_count']}")
    print(f"Class Breakdown:        {summary['class_breakdown']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
