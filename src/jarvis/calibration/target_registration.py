"""Robust two-second collection for look-to-register targets."""

from __future__ import annotations

import math

import numpy as np

from jarvis.calibration.registry import TargetDirection, TargetRecord, TargetSpread
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.smoothing import SmoothedGaze


class TargetRegistrationSession:
    def __init__(
        self,
        target_id: str,
        name: str,
        device_type: str,
        device_id: str,
        *,
        duration_ms: int = 2_000,
        minimum_valid_frames: int = 30,
        minimum_confidence: float = 0.5,
        maximum_jump_deg: float = 12.0,
    ) -> None:
        if duration_ms <= 0 or minimum_valid_frames <= 0:
            raise ValueError("duration and frame count must be positive")
        self.target_id, self.name = target_id, name
        self.device_type, self.device_id = device_type, device_id
        self.duration_ms, self.minimum_valid_frames = duration_ms, minimum_valid_frames
        self.minimum_confidence, self.maximum_jump_deg = minimum_confidence, maximum_jump_deg
        self.started_at_ms: int | None = None
        self._samples: list[tuple[float, float]] = []

    @property
    def valid_frame_count(self) -> int:
        return len(self._samples)

    def add(self, gaze: SmoothedGaze | None, confidence: float, *, eyes_open: bool = True) -> bool:
        if gaze is None or not eyes_open or confidence < self.minimum_confidence:
            return False
        if self.started_at_ms is None:
            self.started_at_ms = gaze.timestamp_ms
        yaw, pitch = direction_to_yaw_pitch(gaze.direction)
        if self._samples:
            previous_yaw, previous_pitch = self._samples[-1]
            if math.hypot(yaw - previous_yaw, pitch - previous_pitch) > self.maximum_jump_deg:
                return False
        self._samples.append((yaw, pitch))
        return True

    def is_elapsed(self, timestamp_ms: int) -> bool:
        return (
            self.started_at_ms is not None and timestamp_ms - self.started_at_ms >= self.duration_ms
        )

    def finalize(self) -> TargetRecord:
        if len(self._samples) < self.minimum_valid_frames:
            raise ValueError(
                f"not enough valid registration frames: {len(self._samples)}/{self.minimum_valid_frames}"
            )
        samples = np.asarray(self._samples, dtype=np.float64)
        center = np.median(samples, axis=0)
        deviations = np.abs(samples - center)
        spread = np.percentile(deviations, 90, axis=0)
        return TargetRecord(
            target_id=self.target_id,
            name=self.name,
            device_type=self.device_type,
            direction=TargetDirection(float(center[0]), float(center[1])),
            spread=TargetSpread(max(4.0, float(spread[0])), max(4.0, float(spread[1]))),
            device_id=self.device_id,
        )
