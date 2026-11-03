"""
LineCount AI — FastAPI REST API
Endpoints for object detection, tracking, and counting.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Conditional import of LineCount modules
# ---------------------------------------------------------------------------
try:
    from linecount.detection.detector import ObjectDetector, Detection
    from linecount.tracking.tracker import ByteTracker, Track
    from linecount.counting.counter import LineCrossingCounter
    _LINECOUNT_AVAILABLE = True
except ImportError:
    _LINECOUNT_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="LineCount AI API",
        description=(
            "Real-time industrial object detection, tracking, and counting. "
            "Detects products on conveyor belts, tracks across frames, and counts "
            "objects crossing a virtual line."
        ),
        version="1.0.0",
        contact={
            "name": "Habimana Moise",
            "email": "habimanamoise77@gmail.com",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Global model state
    # -----------------------------------------------------------------------
    _detector: Optional[Any] = None

    def _get_detector() -> Any:
        global _detector
        if _detector is None:
            if not _LINECOUNT_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="LineCount modules not installed. Run `pip install -e .`",
                )
            _detector = ObjectDetector(model_name_or_path="yolov8n.pt", confidence_threshold=0.25)
        return _detector

    def _decode_image(upload_bytes: bytes) -> np.ndarray:
        """Decode uploaded image bytes to a BGR numpy array."""
        if not _CV2_AVAILABLE:
            raise HTTPException(status_code=503, detail="OpenCV not available.")
        arr = np.frombuffer(upload_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        return frame

    def _detection_to_dict(d: Any) -> Dict[str, Any]:
        """Convert a Detection dataclass or dict to a uniform dictionary."""
        if isinstance(d, dict):
            bbox = d.get("bbox", [0, 0, 0, 0])
            return {
                "class_id": int(d.get("class_id", -1)),
                "class_name": str(d.get("class_name", "unknown")),
                "confidence": round(float(d.get("conf", d.get("confidence", 0.0))), 4),
                "bbox": {
                    "x1": round(float(bbox[0]), 2),
                    "y1": round(float(bbox[1]), 2),
                    "x2": round(float(bbox[2]), 2),
                    "y2": round(float(bbox[3]), 2),
                },
            }
        # Detection object
        return {
            "class_id": int(getattr(d, "class_id", -1)),
            "class_name": str(getattr(d, "class_name", "unknown")),
            "confidence": round(float(getattr(d, "confidence", 0.0)), 4),
            "bbox": {
                "x1": round(float(d.x1 if hasattr(d, "x1") else d.bbox[0]), 2),
                "y1": round(float(d.y1 if hasattr(d, "y1") else d.bbox[1]), 2),
                "x2": round(float(d.x2 if hasattr(d, "x2") else d.bbox[2]), 2),
                "y2": round(float(d.y2 if hasattr(d, "y2") else d.bbox[3]), 2),
            },
        }

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health", summary="Health check")
    async def health() -> Dict[str, str]:
        """Returns service health and dependency status."""
        return {
            "status": "ok",
            "service": "linecount-ai",
            "fastapi": str(_FASTAPI_AVAILABLE),
            "opencv": str(_CV2_AVAILABLE),
            "linecount_modules": str(_LINECOUNT_AVAILABLE),
        }

    @app.post("/detect", summary="Detect objects in a single image")
    async def detect(
        file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
        conf_threshold: float = Query(0.25, ge=0.01, le=1.0, description="Detection confidence threshold"),
    ) -> JSONResponse:
        """
        Run object detection on a single uploaded image.

        Returns a list of detected objects with bounding boxes, class names, and confidence scores.
        """
        detector = _get_detector()
        detector.conf_thresh = conf_threshold
        image_bytes = await file.read()
        frame = _decode_image(image_bytes)

        detections = detector.detect(frame)

        return JSONResponse(content={
            "filename": file.filename,
            "image_shape": {"height": frame.shape[0], "width": frame.shape[1]},
            "detection_count": len(detections),
            "detections": [_detection_to_dict(d) for d in detections],
        })

    @app.post("/count/image", summary="Count objects in a single image")
    async def count_image(
        file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
        conf_threshold: float = Query(0.25, ge=0.01, le=1.0),
    ) -> JSONResponse:
        """
        Detect objects in an image and return per-class counts.

        Note: line-crossing counting requires video (see /count/video).
        This endpoint returns simple frame-level object counts.
        """
        detector = _get_detector()
        detector.conf_thresh = conf_threshold
        image_bytes = await file.read()
        frame = _decode_image(image_bytes)

        detections = detector.detect(frame)

        # Count per class
        class_counts: Dict[str, int] = {}
        for d in detections:
            name = str(d.get("class_name", "unknown") if isinstance(d, dict) else getattr(d, "class_name", "unknown"))
            class_counts[name] = class_counts.get(name, 0) + 1

        return JSONResponse(content={
            "filename": file.filename,
            "total_objects": len(detections),
            "counts_by_class": class_counts,
            "note": (
                "Frame-level count only. "
                "For line-crossing counts with anti-double-counting, process a video."
            ),
        })

    @app.post("/count/video", summary="Count objects crossing a virtual line in a video")
    async def count_video(
        file: UploadFile = File(..., description="Video file (MP4/AVI)"),
        line_x1: float = Query(0.2, description="Line start X (fraction of frame width)"),
        line_y1: float = Query(0.5, description="Line start Y (fraction of frame height)"),
        line_x2: float = Query(0.8, description="Line end X (fraction of frame width)"),
        line_y2: float = Query(0.5, description="Line end Y (fraction of frame height)"),
        conf_threshold: float = Query(0.25, ge=0.01, le=1.0),
        max_frames: int = Query(500, ge=1, le=10000, description="Maximum frames to process"),
    ) -> JSONResponse:
        """
        Process a video file: detect objects, track them across frames,
        and count how many cross the virtual line defined by (line_x1,line_y1)→(line_x2,line_y2).

        Coordinates are expressed as fractions of frame dimensions (0.0–1.0).
        """
        if not _LINECOUNT_AVAILABLE:
            raise HTTPException(status_code=503, detail="LineCount modules not available.")
        if not _CV2_AVAILABLE:
            raise HTTPException(status_code=503, detail="OpenCV not available.")

        # Save upload to temp file
        video_bytes = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Convert fractional line coords to pixel coords
        p1 = (int(line_x1 * width), int(line_y1 * height))
        p2 = (int(line_x2 * width), int(line_y2 * height))

        detector = _get_detector()
        detector.conf_thresh = conf_threshold
        tracker = ByteTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        counter = LineCrossingCounter(line_start=p1, line_end=p2)

        frames_processed = 0
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            counter.update(tracks)
            frames_processed += 1

        cap.release()
        Path(tmp_path).unlink(missing_ok=True)

        counts = counter.get_counts()
        total_count = sum(counts.values()) if counts else (counter.count if hasattr(counter, "count") else 0)

        return JSONResponse(content={
            "filename": file.filename,
            "video_info": {
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "frames_processed": frames_processed,
            },
            "virtual_line": {
                "start": {"x": p1[0], "y": p1[1]},
                "end": {"x": p2[0], "y": p2[1]},
            },
            "counts_by_class": counts if isinstance(counts, dict) else {"objects": total_count},
            "total_crossings": total_count,
        })

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, str]:
        return {"message": "LineCount AI API. Visit /docs for the interactive API reference."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
