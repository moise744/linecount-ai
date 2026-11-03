# LineCount AI — Quick Start Guide

Get up and running in under 5 minutes.

## Prerequisites

- Python 3.9+
- pip

## 1. Clone and Install

```bash
git clone https://github.com/moise744/linecount-ai.git
cd linecount-ai
pip install -e ".[dev]"
```

## 2. Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 21 passed, 1 skipped
```

## 3. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
# Opens at http://localhost:8502
```

Upload an image or video. Configure the virtual counting line in the sidebar. View live counts and tracking IDs.

## 4. Run the FastAPI Server

```bash
pip install fastapi uvicorn[standard]
uvicorn app.api:app --host 0.0.0.0 --port 8001 --reload
# API docs at http://localhost:8001/docs
```

## 5. Detect on a Single Image

```bash
python scripts/detect_video.py --source path/to/image.jpg
```

## 6. Process a Video with Line Counting

```bash
python scripts/detect_video.py \
  --source path/to/conveyor.mp4 \
  --line 0.2,0.5,0.8,0.5 \
  --output outputs/counted.mp4
```

The `--line` argument is `x1,y1,x2,y2` as fractions of frame size (0.0–1.0).

## 7. Evaluate Counting Accuracy

```bash
python scripts/evaluate_counts.py \
  --predictions outputs/predicted_counts.json \
  --ground_truth data/ground_truth_counts.json
```

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `make install` | Install production deps |
| `make test` | Run all tests |
| `make run-app` | Launch Streamlit dashboard |
| `make run-api` | Launch FastAPI server |
| `make docker-build` | Build Docker image |
