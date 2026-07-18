"""Per-device gaze calibration session (README 5.1 "초기 기기 등록").

사용자가 몇 초간 기기를 바라보는 동안 합성된 시선 방향 벡터를 모았다가
`DeviceGazeProfile`(평균 방향 + 각도 분산)로 축약한다. Raw per-frame 벡터는
`finalize()` 이후 버린다(development-principles.md 1절 5:
"raw 프레임은 필요한 계산이 끝나면 기본적으로 폐기한다").
"""

from __future__ import annotations

import numpy as np

from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation, Vector3, compose_gaze_vector


class CalibrationSession:
    """한 기기에 대한 calibration 프레임을 모아 `DeviceGazeProfile`로 축약한다."""

    def __init__(self, device_id: str, config: GazeConfig = GazeConfig()) -> None:
        self._device_id = device_id
        self._config = config
        self._directions: list[Vector3] = []

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def frame_count(self) -> int:
        """지금까지 반영된(추적 성공) 프레임 수."""
        return len(self._directions)

    def add_observation(self, observation: FaceObservation) -> bool:
        """한 프레임을 반영한다. 추적 손실 프레임은 무시하고 False를 반환한다."""
        gaze_vector = compose_gaze_vector(observation, self._config)
        if gaze_vector is None:
            return False
        self._directions.append(gaze_vector.direction)
        return True

    def finalize(self) -> DeviceGazeProfile:
        """수집한 방향 벡터를 평균 방향 + 각도 분산으로 축약하고 raw 프레임은 버린다."""
        if not self._directions:
            raise ValueError(
                f"No usable observations collected for device '{self._device_id}'; "
                "cannot calibrate from zero valid frames."
            )

        stacked = np.stack(self._directions)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            raise ValueError(
                f"Calibration observations for device '{self._device_id}' cancel out to a "
                "zero vector; recollect calibration frames."
            )
        mean_direction = mean / norm

        cosine_similarities = np.clip(stacked @ mean_direction, -1.0, 1.0)
        angular_distances = np.arccos(cosine_similarities)
        variance = float(np.mean(angular_distances**2))

        self._directions = []  # raw 프레임 폐기 — 이후 재사용하지 않는다.
        return DeviceGazeProfile(
            device_id=self._device_id,
            mean_direction=mean_direction,
            variance=variance,
        )
