"""Robust ten-second collection for look-to-register targets.

사용자가 물체를 다양한 각도·자세로 바라보는 동안(README 5.1) 각도 기반
direction+spread(오늘까지의 동작, 항상 계산됨)와 3D 위치(가능할 때만) 둘 다를
시도한다. 3D는 머리 이동(parallax)으로 얻은 시선 광선들을 삼각측량해 품질
기준을 만족할 때만 채택되고, 그렇지 않으면 조용히 각도 기반으로 대체된다
(calibration/triangulation.py, documents/decisions.md).
"""

from __future__ import annotations

import math

import numpy as np

from jarvis.calibration.registry import (
    TargetDirection,
    TargetGeometry3DRecord,
    TargetRecord,
    TargetSpread,
)
from jarvis.calibration.triangulation import TriangulationResult, triangulate_rays
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.features import Vector3
from jarvis.gaze.smoothing import SmoothedGaze


class TargetRegistrationSession:
    def __init__(
        self,
        target_id: str,
        name: str,
        device_type: str,
        device_id: str,
        *,
        duration_ms: int = 10_000,
        minimum_valid_frames: int = 30,
        minimum_confidence: float = 0.5,
        maximum_jump_deg: float = 12.0,
        config: GazeConfig = GazeConfig(),
    ) -> None:
        if duration_ms <= 0 or minimum_valid_frames <= 0:
            raise ValueError("duration and frame count must be positive")
        self.target_id, self.name = target_id, name
        self.device_type, self.device_id = device_type, device_id
        self.duration_ms, self.minimum_valid_frames = duration_ms, minimum_valid_frames
        self.minimum_confidence, self.maximum_jump_deg = minimum_confidence, maximum_jump_deg
        self.config = config
        self.started_at_ms: int | None = None
        self._samples: list[tuple[float, float]] = []
        self._rays: list[tuple[Vector3, Vector3]] = []
        self.triangulation_result: TriangulationResult | None = None
        """가장 최근 `finalize()` 호출이 시도한 삼각측량 결과 — 품질 기준을
        만족하지 못해 각도 모드로 대체된 경우에도 진단을 위해 남는다(값을
        숨기지 않는다). 광선이 아예 부족해 시도조차 못 했으면 None이다."""

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
        if gaze.origin is not None:
            self._rays.append((gaze.origin, gaze.direction))
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
            position_3d=self._try_triangulate(),
        )

    def _try_triangulate(self) -> TargetGeometry3DRecord | None:
        """가능하면 3D 위치를 추정하고, 품질 기준을 만족할 때만 반환한다.

        기준 미달(머리 이동 부족, 광선이 거의 평행, 잔차 과다)이면 조용히
        None을 반환해 각도 기반 등록으로 대체되게 한다 — 대신 진단 결과는
        `self.triangulation_result`에 남겨 호출자가 왜 대체됐는지 로그로 보여줄
        수 있게 한다(지어낸 성공을 반환하지 않는다).
        """
        if len(self._rays) < self.config.minimum_triangulation_frames:
            self.triangulation_result = None
            return None
        origins = [origin for origin, _ in self._rays]
        directions = [direction for _, direction in self._rays]
        result = triangulate_rays(origins, directions)
        self.triangulation_result = result
        if not result.passes_quality_gates(self.config):
            return None
        radius_mm = max(result.residual_rms_mm, self.config.target_radius_floor_mm)
        center = result.center_mm
        return TargetGeometry3DRecord(
            center_mm=(float(center[0]), float(center[1]), float(center[2])),
            radius_mm=radius_mm,
        )
