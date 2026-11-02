"""
LineCount AI — Interactive Industrial Conveyor Dashboard.

Streamlit application providing real-time video inspection, interactive
virtual counting line calibration, throughput monitoring, and event logging.

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import streamlit as st

from linecount.counting.line_counter import CountingDirection, CountingLine, LineCounter
from linecount.counting.statistics import ThroughputMonitor
from linecount.detection.detector import Detection, ObjectDetector
from linecount.tracking.tracker import ObjectTracker
from linecount.visualization.renderer import ConveyorRenderer

# Page Setup
st.set_page_config(
    page_title="LineCount AI — Conveyor Inspection & Tracking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styles
st.markdown("""
<style>
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00adb5;
        margin-bottom: 0.1rem;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #1e2022;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #00adb5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Calibration & Settings")
    st.markdown("---")

    st.markdown("### Virtual Counting Line")
    line_y = st.slider("Line Vertical Position (Y)", 50, 650, 320, 10)
    line_x1 = st.slider("Line Start (X1)", 0, 400, 50, 10)
    line_x2 = st.slider("Line End (X2)", 400, 1000, 750, 10)
    direction_choice = st.selectbox(
        "Counting Direction",
        ["any", "incoming", "outgoing"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Tracker Hyperparameters")
    min_hits = st.slider("Min Hits to Confirm Track", 1, 5, 2)
    max_age = st.slider("Max Track Age (Frames)", 5, 45, 20)
    iou_thresh = st.slider("Association IoU Threshold", 0.1, 0.7, 0.3, 0.05)

    st.markdown("---")
    sim_speed = st.slider("Simulation Speed (FPS)", 5, 60, 24)


# Main Header
st.markdown('<div class="brand-title">📊 LineCount AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">Real-Time Industrial Object Detection, Multi-Object Tracking & Throughput Telemetry</div>',
    unsafe_allow_html=True,
)

# Initialize Session State
if "counter" not in st.session_state:
    counting_line = CountingLine(p1=(float(line_x1), float(line_y)), p2=(float(line_x2), float(line_y)))
    st.session_state.counter = LineCounter(line=counting_line, direction=CountingDirection(direction_choice))
    st.session_state.tracker = ObjectTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
    st.session_state.monitor = ThroughputMonitor(window_seconds=60.0)
    st.session_state.renderer = ConveyorRenderer()
    st.session_state.sim_step = 0
    st.session_state.sim_objects = [
        {"id": 1, "x": 120.0, "y": 80.0, "vx": 0.0, "vy": 7.0, "cls": "bottle"},
        {"id": 2, "x": 380.0, "y": 40.0, "vx": 0.0, "vy": 6.2, "cls": "can"},
        {"id": 3, "x": 600.0, "y": 140.0, "vx": 0.0, "vy": 7.5, "cls": "package"},
    ]

# Update Line Position from Sliders
st.session_state.counter.line = CountingLine(p1=(float(line_x1), float(line_y)), p2=(float(line_x2), float(line_y)))
st.session_state.counter.direction = CountingDirection(direction_choice)

# Telemetry KPI Display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
summary = st.session_state.counter.get_summary()
snapshot = st.session_state.monitor.get_snapshot()

kpi1.metric("Total Counted", f"{summary['total_count']} units")
kpi2.metric("Production Rate", f"{snapshot.items_per_minute:.1f} PPM")
kpi3.metric("Projected Hourly", f"{int(snapshot.items_per_hour)} PPH")
kpi4.metric("Engine FPS", f"{snapshot.fps:.1f}")

st.markdown("---")

col_feed, col_stats = st.columns([1.6, 1.0])

with col_feed:
    st.markdown("#### Live Conveyor Video Stream")
    feed_placeholder = st.empty()

with col_stats:
    st.markdown("#### Production Analytics")
    class_breakdown = summary["class_breakdown"]
    if class_breakdown:
        st.bar_chart(class_breakdown)
    else:
        st.info("Awaiting items crossing the virtual counting gate...")

    st.markdown("#### Recent Crossing Log")
    events = st.session_state.counter.crossing_events[-8:]
    if events:
        table_data = [
            {"Track": f"#{e['track_id']}", "Class": e["class_name"], "Dir": e["direction"]}
            for e in reversed(events)
        ]
        st.table(table_data)
    else:
        st.caption("No crossing events registered yet.")

# Control Buttons
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
with btn_col1:
    step_btn = st.button("Advance 1 Frame")
with btn_col2:
    if st.button("Reset Counters"):
        st.session_state.counter.reset()
        st.session_state.monitor.reset()
        st.session_state.tracker.reset()
        st.rerun()

# Advance Simulation Step
h, w = 600, 800
frame = np.full((h, w, 3), 42, dtype=np.uint8)

# Draw subtle conveyor belt textures
for y_line in range(0, h, 60):
    cv2.line(frame, (0, y_line), (w, y_line), (55, 55, 55), 1)

# Move simulated objects down the conveyor
detections: list[Detection] = []
for obj in st.session_state.sim_objects:
    obj["y"] += obj["vy"]
    if obj["y"] > h:
        obj["y"] = -40.0
        obj["x"] = float(np.random.randint(100, 700))

    x1, y1 = obj["x"] - 35, obj["y"] - 35
    x2, y2 = obj["x"] + 35, obj["y"] + 35
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    detections.append(
        Detection(
            bbox=box,
            confidence=float(np.random.uniform(0.92, 0.98)),
            class_id=0,
            class_name=obj["cls"],
        )
    )

# Process tracking and counting
active_tracks = st.session_state.tracker.update(detections)
new_events = st.session_state.counter.update(active_tracks)
if new_events:
    st.session_state.monitor.record_item(len(new_events))
st.session_state.monitor.record_frame()

# Render overlays
annotated = st.session_state.renderer.render(
    frame, active_tracks, st.session_state.counter, st.session_state.monitor.get_snapshot()
)

# Convert BGR to RGB for Streamlit display
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
feed_placeholder.image(annotated_rgb, use_container_width=True)
