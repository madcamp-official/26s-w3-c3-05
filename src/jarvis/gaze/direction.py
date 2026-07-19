"""Human-readable yaw/pitch representation of calibrated gaze directions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jarvis.gaze.features import Vector3


@dataclass(frozen=True, slots=True)
class CalibratedGaze:
    yaw: float
    pitch: float
    confidence: float
    timestamp_ms: int

    def __post_init__(self) -> None:
        values = (self.yaw, self.pitch, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calibrated gaze values must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")


def direction_to_yaw_pitch(direction: Vector3) -> tuple[float, float]:
    """Convert the camera-space unit vector to yaw(+right), pitch(+up)."""
    x, y, z = (float(value) for value in direction)
    yaw = math.degrees(math.atan2(x, z))
    pitch = math.degrees(math.atan2(-y, math.hypot(x, z)))
    return yaw, pitch


def yaw_pitch_to_direction(yaw: float, pitch: float) -> Vector3:
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise ValueError("yaw and pitch must be finite")
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    return np.array(
        [
            math.sin(yaw_rad) * math.cos(pitch_rad),
            -math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
        ],
        dtype=np.float64,
    )
