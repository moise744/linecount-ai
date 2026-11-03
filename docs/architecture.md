# LineCount AI — System Architecture

## Overview

LineCount AI is a real-time industrial object detection, tracking, and counting system. It chains a YOLO-based detector, a ByteTrack multi-object tracker, and a virtual line-crossing counter into a low-latency pipeline suitable for conveyor belt monitoring.

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Input Layer                                  │
│     Single image  ─────┐                                             │
│     Video file    ─────┤──► Frame iterator (OpenCV VideoCapture)     │
│     RTSP stream   ─────┘                                             │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ frame (H×W×3 BGR)
                      ┌─────────▼─────────┐
                      │  ObjectDetector   │
                      │  YOLOv8 / YOLOv5  │
                      │  (Ultralytics)    │
                      └─────────┬─────────┘
                                │ detections: [{bbox, conf, class_id}]
                      ┌─────────▼─────────┐
                      │   ByteTracker     │
                      │  Kalman filter +  │
                      │  Hungarian assign │
                      └─────────┬─────────┘
                                │ tracks: [{track_id, bbox, class_id}]
                      ┌─────────▼─────────┐
                      │  LineCrossing     │
                      │  Counter          │
                      │  (virtual line)   │
                      └─────────┬─────────┘
                                │ counts per class + crossing events
               ┌────────────────┼────────────────┐
               │                │                │
      ┌────────▼───────┐ ┌──────▼──────┐ ┌──────▼──────────┐
      │ Frame annotator│ │  REST API   │ │ Streamlit App   │
      │ (OpenCV draw)  │ │  FastAPI    │ │ live dashboard  │
      └────────────────┘ └─────────────┘ └─────────────────┘
```

---

## Component Descriptions

| Component | Location | Responsibility |
|-----------|----------|---------------|
| `ObjectDetector` | `src/linecount/detection/detector.py` | Wraps Ultralytics YOLO; provides `detect(frame)` → list of detection dicts with `{bbox, conf, class_id, class_name}` |
| `ByteTracker` | `src/linecount/tracking/tracker.py` | Multi-object tracker using Kalman filter for motion prediction and Hungarian algorithm for assignment; maintains persistent track IDs across frames |
| `KalmanFilter` | `src/linecount/tracking/kalman.py` | 8-state Kalman filter (`[x,y,w,h,vx,vy,vw,vh]`) for smooth bounding box prediction during brief occlusions |
| `TrackManager` | `src/linecount/tracking/track_manager.py` | Lifecycle management (tentative → confirmed → lost → removed); configurable `max_age` and `min_hits` |
| `LineCrossingCounter` | `src/linecount/counting/counter.py` | Defines virtual line segments; detects centroid crossing direction using sign-change of signed perpendicular distance; prevents double-counting with per-track crossing log |
| `CountingEvaluator` | `src/linecount/evaluation/evaluator.py` | Computes counting accuracy (MAE, relative error) against ground-truth count files |
| `api.py` | `app/api.py` | FastAPI endpoints: `POST /detect` (image → detections), `POST /count/video` (video → count summary), `GET /health` |
| `streamlit_app.py` | `app/streamlit_app.py` | Real-time dashboard: upload image/video, configure virtual line, visualize counts and tracking IDs |

---

## Data Flow

```
Video frame
     │
     ▼
ObjectDetector.detect(frame)
     │ [{bbox:[x1,y1,x2,y2], conf, class_id}]
     ▼
ByteTracker.update(detections)
     │ [{track_id, bbox, class_id, state}]
     ▼
LineCrossingCounter.update(tracks)
     │ {class_name: count, crossing_events: [...]}
     ▼
Annotated frame + count overlay
     │
     ▼
Streamlit dashboard / FastAPI response / saved video
```

---

## Tracking Algorithm Detail

### ByteTrack Association Strategy

ByteTrack handles high- and low-confidence detections separately to reduce track fragmentation:

```
High-conf detections (conf ≥ thresh)
    │
    ├──► IoU matching → confirmed tracks        (first association)
    │
    └──► Remaining detections
             │
Low-conf detections (conf < thresh)
    │
    └──► IoU matching → lost tracks            (second association)

Unmatched high-conf detections → new tentative tracks
Unmatched tracks (age > max_age) → removed
```

### Line Crossing Logic

For a virtual line defined by points P₁ and P₂, and a track centroid C:

```
signed_dist = (P2-P1) × (C-P1)   # cross product (z-component)
crossing = sign change between consecutive frames
direction = "up" if sign goes negative→positive else "down"
```

Each track ID is recorded after its first crossing to prevent double-counting.

---

## Key Design Decisions

### 1. ByteTrack over DeepSORT
ByteTrack achieves competitive tracking accuracy without requiring a re-identification (ReID) feature extractor. This keeps the system CPU-friendly and avoids the need for a separate embedding model.

### 2. Virtual line as a line segment
Counting is triggered only when a centroid crosses the configured line segment (not an infinite line). This prevents spurious counts from objects moving parallel to the line near its endpoints.

### 3. Kalman-based occlusion handling
When a detected object is briefly occluded (e.g., by another object or frame edge), Kalman prediction keeps the track alive for up to `max_age` frames. The track resumes if the object reappears.

### 4. Separation of detector and tracker
The detector and tracker are independent modules. The detector can be swapped (YOLOv8n → YOLOv8m → custom ONNX) without changing the tracking or counting logic.

---

## Performance Characteristics

| Component | Typical latency (CPU, 640×640) |
|-----------|-------------------------------|
| YOLOv8n detection | ~150–400 ms |
| ByteTrack update | ~2–5 ms |
| Line crossing check | < 1 ms |
| Full pipeline | ~150–410 ms (~2–6 FPS) |

> [!TIP]
> GPU inference with YOLOv8n achieves 30+ FPS at 640×640. Use `device="cuda"` in `ObjectDetector`.

---

## Dataset

**COCO** — for general object detection evaluation.
- 80 object categories.
- YOLOv8n pretrained weights available via Ultralytics.

For conveyor belt counting evaluation, the system uses ground-truth count files (manually verified) alongside `scripts/evaluate_counts.py`.

See `data/README.md` for dataset preparation details.
