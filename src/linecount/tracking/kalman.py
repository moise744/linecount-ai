"""
Kalman Filter implementation for bounding box tracking.

State vector: [x_center, y_center, area, aspect_ratio, dx, dy, da]
Measurement vector: [x_center, y_center, area, aspect_ratio]
"""

from __future__ import annotations

import numpy as np


class KalmanBoxTracker:
    """Tracks a single bounding box state using a constant velocity Kalman Filter."""

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """Initialize state from initial bounding box [x1, y1, x2, y2]."""
        # 7-state: x, y, s (area), r (ratio), vx, vy, vs
        self.dim_x = 7
        self.dim_z = 4

        # Transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # x + vx*dt, y + vy*dt, s + vs*dt

        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        for i in range(self.dim_z):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty in initial velocities

        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        # State vector
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.x[:4, 0] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: list[np.ndarray] = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x_center, y_center, s, r]."""
        w = max(1e-4, bbox[2] - bbox[0])
        h = max(1e-4, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r], dtype=np.float32)

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, s, r] to [x1, y1, x2, y2]."""
        w = np.sqrt(max(1e-4, x[2] * x[3]))
        h = max(1e-4, x[2] / w)
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0,
        ], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Advance state vector by one time step and return predicted bbox."""
        # Check area constraint
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred_box = self._x_to_bbox(self.x[:4, 0])
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox: np.ndarray) -> None:
        """Update the state with observed bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox).reshape(-1, 1)
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate."""
        return self._x_to_bbox(self.x[:4, 0])
