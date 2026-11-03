# LineCount AI — API Reference

Base URL: `http://localhost:8001`

Interactive docs: `http://localhost:8001/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "linecount-ai",
  "fastapi": "True",
  "opencv": "True",
  "linecount_modules": "True"
}
```

---

### `POST /detect`

Detect objects in a single image.

**Request:** `multipart/form-data`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | ✅ | — | Image (JPEG/PNG) |
| `conf_threshold` | float | — | `0.25` | Minimum detection confidence |

**Response:**
```json
{
  "filename": "frame_001.jpg",
  "image_shape": {"height": 480, "width": 640},
  "detection_count": 3,
  "detections": [
    {
      "class_id": 39,
      "class_name": "bottle",
      "confidence": 0.87,
      "bbox": {"x1": 120.0, "y1": 80.0, "x2": 200.0, "y2": 220.0}
    }
  ]
}
```

---

### `POST /count/image`

Count objects in a single frame (no tracking, no line crossing).

**Request:** `multipart/form-data`

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `file` | file | ✅ | — |
| `conf_threshold` | float | — | `0.25` |

**Response:**
```json
{
  "filename": "frame.jpg",
  "total_objects": 5,
  "counts_by_class": {"bottle": 3, "can": 2},
  "note": "Frame-level count only. For line-crossing counts, process a video."
}
```

---

### `POST /count/video`

Process a video: detect → track → count line crossings.

**Request:** `multipart/form-data`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | — | Video (MP4/AVI) |
| `line_x1` | float | `0.2` | Line start X (0–1 fraction) |
| `line_y1` | float | `0.5` | Line start Y (0–1 fraction) |
| `line_x2` | float | `0.8` | Line end X (0–1 fraction) |
| `line_y2` | float | `0.5` | Line end Y (0–1 fraction) |
| `conf_threshold` | float | `0.25` | Detection confidence |
| `max_frames` | int | `500` | Max frames to process |

**Response:**
```json
{
  "filename": "conveyor.mp4",
  "video_info": {"width": 1280, "height": 720, "fps": 30.0, "total_frames": 900, "frames_processed": 500},
  "virtual_line": {"start": {"x": 256, "y": 360}, "end": {"x": 1024, "y": 360}},
  "counts_by_class": {"bottle": 12, "can": 8},
  "total_crossings": 20
}
```

---

## Error Responses

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Invalid image or video file |
| `503` | LineCount modules or OpenCV not available |

---

## Python Client Example

```python
import requests

# Count objects in a video
with open("conveyor.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8001/count/video",
        files={"file": f},
        params={"line_y1": 0.5, "line_y2": 0.5, "max_frames": 300},
    )

result = response.json()
print("Total crossings:", result["total_crossings"])
print("By class:", result["counts_by_class"])
```
