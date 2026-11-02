"""Unit tests for FastAPI endpoints in LineCount AI."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from linecount.api.app import create_app
from linecount.counting.line_counter import CountingLine, LineCounter
from linecount.counting.statistics import ThroughputMonitor


class TestAPIEndpoints:
    def setup_method(self):
        line = CountingLine(p1=(0, 200), p2=(500, 200), name="test_gate")
        self.counter = LineCounter(line=line)
        self.monitor = ThroughputMonitor()
        self.app = create_app(counter=self.counter, monitor=self.monitor)
        self.client = TestClient(self.app)

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_status_endpoint(self):
        self.counter.total_count = 12
        self.counter.class_counts["bottle"] = 12
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 12
        assert data["class_breakdown"]["bottle"] == 12
        assert data["line_name"] == "test_gate"

    def test_set_line_endpoint(self):
        payload = {
            "p1": [10.0, 300.0],
            "p2": [700.0, 300.0],
            "name": "updated_gate",
            "direction": "incoming",
        }
        response = self.client.post("/line", json=payload)
        assert response.status_code == 200
        assert self.counter.line.name == "updated_gate"
        assert self.counter.line.y1 == 300.0

    def test_reset_endpoint(self):
        self.counter.total_count = 50
        response = self.client.post("/reset")
        assert response.status_code == 200
        assert self.counter.total_count == 0
