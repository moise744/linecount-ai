"""
FastAPI REST Service for Industrial LineCount AI.

Enables automated communication with factory MES/SCADA systems:
- Query real-time production counts
- Retrieve throughput (PPM, PPH) and performance metrics
- Dynamically reconfigure virtual counting lines
- Batch production reset endpoints
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = Exception
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

from linecount.counting.line_counter import CountingDirection, CountingLine, LineCounter
from linecount.counting.statistics import ThroughputMonitor


class LineConfigRequest(BaseModel):
    """Payload to configure or update virtual conveyor counting line."""
    p1: Tuple[float, float] = Field(..., description="Start point (x1, y1)")
    p2: Tuple[float, float] = Field(..., description="End point (x2, y2)")
    name: str = Field(default="conveyor_gate_1")
    direction: str = Field(default="any", description="'any', 'incoming', or 'outgoing'")


class StatusResponse(BaseModel):
    """Response containing live conveyor statistics."""
    line_name: str
    total_count: int
    in_count: int
    out_count: int
    class_breakdown: Dict[str, int]
    items_per_minute: float
    items_per_hour: float
    fps: float


def create_app(
    counter: Optional[LineCounter] = None,
    monitor: Optional[ThroughputMonitor] = None,
):
    """Factory creating and wiring the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        return None
    app = FastAPI(
        title="LineCount AI — Industrial Telemetry API",
        description="Real-Time Object Detection, Tracking & Counting Telemetry Service",
        version="0.1.0",
    )

    # State containers
    if counter is None:
        default_line = CountingLine(p1=(0.0, 300.0), p2=(800.0, 300.0), name="conveyor_main")
        counter = LineCounter(line=default_line)

    if monitor is None:
        monitor = ThroughputMonitor()

    app.state.counter = counter
    app.state.monitor = monitor

    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, str]:
        return {"status": "healthy", "service": "LineCount AI"}

    @app.get("/status", response_model=StatusResponse, tags=["Telemetry"])
    async def get_status() -> StatusResponse:
        c: LineCounter = app.state.counter
        m: ThroughputMonitor = app.state.monitor
        snap = m.get_snapshot()

        return StatusResponse(
            line_name=c.line.name,
            total_count=c.total_count,
            in_count=c.in_count,
            out_count=c.out_count,
            class_breakdown=dict(c.class_counts),
            items_per_minute=snap.items_per_minute,
            items_per_hour=snap.items_per_hour,
            fps=snap.fps,
        )

    @app.post("/line", tags=["Configuration"])
    async def set_line(config: LineConfigRequest) -> Dict[str, object]:
        try:
            dir_enum = CountingDirection(config.direction.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid direction: {config.direction}")

        new_line = CountingLine(p1=config.p1, p2=config.p2, name=config.name)
        app.state.counter.line = new_line
        app.state.counter.direction = dir_enum
        return {"status": "success", "configured_line": config.dict()}

    @app.post("/reset", tags=["Production Control"])
    async def reset_counts() -> Dict[str, str]:
        app.state.counter.reset()
        app.state.monitor.reset()
        return {"status": "success", "message": "Production counts and telemetry reset."}

    return app


app = create_app() if FASTAPI_AVAILABLE else None
