<div align="center">

# 📊 LineCount AI

### Real-Time Industrial Object Detection, Tracking & Counting

Automated conveyor inspection system featuring multi-object tracking, bidirectional virtual line counting, anti-double-counting mechanisms, and live MES throughput telemetry.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

In modern manufacturing, accurate unit counting and throughput monitoring are vital for production line balancing, yield tracking, and bottleneck identification. Traditional photoelectric sensors suffer from blind spots, occlusions, and lack category intelligence.

**LineCount AI** is an end-to-end computer vision solution engineered for industrial conveyor belts:
- **Detection**: YOLOv8 neural network detecting multi-class products at high speed.
- **Tracking**: Kalman Filter state estimation + Hungarian IoU data association (SORT/ByteTrack paradigm) to assign persistent tracking IDs across occlusions and motion jitter.
- **Counting**: Vector cross-product line crossing algorithm with anti-double-counting protection.
- **Throughput Analytics**: Real-time parts-per-minute (PPM), projected hourly rate (PPH), and processing FPS.
- **Factory Integration**: REST API endpoints for direct communication with factory SCADA / MES systems.

---

## System Architecture

```
                                VIDEO STREAM
                        (Conveyor Belt Camera Feed)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    YOLOv8 Object Detector   │
                      │  BBoxes + Class Confidence  │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Kalman Filter Tracker    │
                      │   State Estimation [x,y,s,r]│
                      │    Hungarian IoU Matching   │
                      │   Persistent Object IDs     │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   Virtual Line Gate Engine  │
                      │  Segment Intersection Check │
                      │  Signed Directional Analysis│
                      │  Anti-Double Count Registry │
                      └──────────────┬──────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
      ┌─────────────────────────────┐ ┌─────────────────────────────┐
      │   MES Telemetry REST API    │ │   Streamlit Live Dashboard  │
      │  /status, /line, /reset     │ │ Real-Time Video HUD + PPM   │
      └─────────────────────────────┘ └─────────────────────────────┘
```

---

## Key Capabilities

- **Bidirectional Virtual Lines**: Define arbitrary line segments across conveyor lanes to count incoming, outgoing, or bidirectional items.
- **Anti-Double-Count Protection**: Once a track ID successfully crosses a virtual line boundary, it is registered and locked against subsequent re-triggers.
- **Occlusion Resilience**: Kalman filter velocity prediction preserves tracking continuity when products temporarily overlap or undergo partial conveyor occlusions.
- **Telemetry HUD**: Automatically renders real-time tracking badges, breadcrumb paths, active line gates, and KPI banners directly onto video streams.
- **SCADA / MES API**: REST interface exposing production counters, PPM throughput, and remote calibration endpoints.

---

## Project Structure

```
linecount-ai/
├── configs/
│   ├── tracking.yaml              # Core detector & tracker parameters
│   └── conveyor.yaml              # Production line specific overrides
├── data/
│   └── README.md                  # Video ingestion format guide
├── scripts/
│   ├── detect_video.py            # Video stream processor CLI
│   └── benchmark_fps.py           # Latency and throughput benchmark
├── src/linecount/
│   ├── detection/                 # YOLOv8 detection wrapper
│   │   └── detector.py
│   ├── tracking/                  # Kalman filtering & Hungarian association
│   │   ├── kalman.py
│   │   └── tracker.py
│   ├── counting/                  # Line crossing & throughput math
│   │   ├── line_counter.py
│   │   └── statistics.py
│   ├── visualization/             # Video overlay renderer & HUD
│   │   └── renderer.py
│   ├── evaluation/                # Ground truth accuracy verification
│   │   └── metrics.py
│   └── api/                       # Factory MES FastAPI service
│       └── app.py
├── app/
│   └── streamlit_app.py           # Interactive conveyor telemetry dashboard
├── tests/                         # Full automated pytest suite
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_counting.py
│   └── test_api.py
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE
```

---

## Quick Start

### 1. Installation

```powershell
git clone https://github.com/moise744/linecount-ai.git
cd linecount-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Test Suite

```powershell
python -m pytest tests/ -v
```

### 3. Launch Interactive Conveyor Dashboard

```powershell
streamlit run app/streamlit_app.py
```

### 4. Process an Industrial Video Stream

```powershell
python scripts/detect_video.py --video data/sample_conveyor.mp4 --output outputs/tracked_output.mp4
```

### 5. Launch Factory Telemetry API

```powershell
uvicorn linecount.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Quantitative Verification Metrics

LineCount AI is evaluated using:
- **Counting Accuracy**: $1 - \frac{|Count_{pred} - Count_{gt}|}{Count_{gt}}$
- **Mean Absolute Percentage Error (MAPE)**
- **Mean Track Lifetime (frames)**
- **Processing Latency (ms) & Sustained FPS**

---

## Author

**Habimana Moise** — Computer Vision & Machine Learning Engineer

- GitHub: [@moise744](https://github.com/moise744)
- LinkedIn: [habimana-moise](https://www.linkedin.com/in/habimana-moise/)
- Email: habimanamoise77@gmail.com
