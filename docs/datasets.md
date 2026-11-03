# LineCount AI — Dataset Guide

## Primary Use Case

LineCount AI is designed for **conveyor belt monitoring** and general industrial counting.
It does not require a purpose-built counting dataset — it uses:

1. **COCO-pretrained YOLO weights** for out-of-the-box object detection.
2. **Your own video footage** from conveyor cameras for counting evaluation.
3. **Ground-truth count files** (manually verified) for accuracy evaluation.

---

## Using COCO Pretrained Weights

YOLOv8n pretrained on COCO (80 classes) is downloaded automatically on first use:

```python
from linecount.detection.detector import ObjectDetector
detector = ObjectDetector(model_name="yolov8n.pt")
# Downloads ~6 MB on first run
```

For custom industrial objects, fine-tune on your own annotated data (see below).

---

## Ground-Truth Count Files (for evaluation)

Create a JSON file with ground-truth counts per video:

```json
{
  "video_001.mp4": {"bottle": 42, "can": 15},
  "video_002.mp4": {"bottle": 38, "can": 22},
  "video_003.mp4": {"bottle": 55}
}
```

Then run:
```bash
python scripts/evaluate_counts.py \
  --ground_truth data/ground_truth_counts.json \
  --predictions outputs/predicted_counts.json
```

---

## Fine-Tuning on Custom Data

To detect your own product types:

1. Annotate images in **YOLO format** (`.txt` per image, one box per line):
   ```
   class_id  cx  cy  w  h   (all normalised 0–1)
   ```

2. Create a `data.yaml`:
   ```yaml
   path: data/custom_products
   train: images/train
   val: images/val
   nc: 3
   names: ["bottle", "can", "box"]
   ```

3. Fine-tune:
   ```bash
   yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
   ```

4. Update `ObjectDetector` to use the new weights:
   ```python
   detector = ObjectDetector(model_name="runs/detect/train/weights/best.pt")
   ```

---

## Recommended Annotation Tools

| Tool | Format | Notes |
|------|--------|-------|
| [Label Studio](https://labelstud.io/) | YOLO/COCO | Free, self-hosted |
| [CVAT](https://www.cvat.ai/) | YOLO/COCO | Open source |
| [Roboflow](https://roboflow.com/) | YOLO | Free tier available |

---

## Expected Directory Structure

```
data/
├── custom_products/        ← your annotated dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── ground_truth_counts.json
└── README.md
```

Do not commit video files or large datasets to git.
