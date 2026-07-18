"""Device target classifier: cosine similarity baseline + variance normalization + UNKNOWN rejection.

README 7장 "Target 추정"을 구현한다.

Baseline: 현재 시선 방향 벡터를 각 기기 prototype 방향 벡터와 코사인 유사도로 비교해
가장 높은 기기를 고른다.

최종 방식: 유사도를 등록 시 저장한 분산(variance)으로 정규화한 뒤 기기 간 확률로
softmax 정규화하고, 최고 확률이 임계값 미만이면 `UNKNOWN`으로 거부한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import Vector3

_MINIMUM_VARIANCE = 1e-6
"""분산이 0에 가까울 때 나눗셈이 발산하지 않도록 하는 하한값."""


@dataclass(frozen=True, slots=True)
class DeviceGazeProfile:
    """기기별 calibration 결과 (README 7장 "기기 등록 방식").

    `mean_direction`은 단위 벡터, `variance`는 등록 시 관측한 방향 벡터의
    각도 분산(라디안^2)이다.
    """

    device_id: str
    mean_direction: Vector3
    variance: float

    def __post_init__(self) -> None:
        norm = float(np.linalg.norm(self.mean_direction))
        if not math.isclose(norm, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"DeviceGazeProfile.mean_direction must be a unit vector, got norm={norm}"
            )
        if self.variance < 0:
            raise ValueError(f"DeviceGazeProfile.variance must be >= 0, got {self.variance}")


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Target 추정 결과 — TargetEstimate로 변환되기 전 단계.

    `target`은 임계값 미만이면 `config.UNKNOWN_TARGET`이다(README 7장 "Unknown
    rejection"). `probability`·`second_best_probability`는 거부 여부와 무관하게
    실제 계산값을 그대로 담는다 — 값을 숨기거나 지어내지 않는다.
    """

    target: str
    probability: float
    second_best_probability: float


def cosine_similarity(a: Vector3, b: Vector3) -> float:
    """두 단위 벡터의 코사인 유사도(내적)를 [-1, 1]로 clip해 반환한다."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


class TargetClassifier:
    """등록된 기기 gaze profile을 바탕으로 현재 시선 방향의 대상 기기를 추정한다."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._profiles: dict[str, DeviceGazeProfile] = {}

    def register_profile(self, profile: DeviceGazeProfile) -> None:
        """기기 gaze profile을 등록하거나 갱신한다."""
        self._profiles[profile.device_id] = profile

    def unregister_profile(self, device_id: str) -> None:
        self._profiles.pop(device_id, None)

    @property
    def profiles(self) -> dict[str, DeviceGazeProfile]:
        return dict(self._profiles)

    def classify(self, direction: Vector3) -> ClassificationResult:
        """합성된 시선 방향 단위 벡터로부터 대상 기기를 추정한다.

        등록된 기기가 없으면 항상 UNKNOWN을 반환한다(지어낸 대상을 반환하지 않는다).
        """
        if not self._profiles:
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )

        device_ids = list(self._profiles.keys())
        scores = np.empty(len(device_ids), dtype=np.float64)
        for i, device_id in enumerate(device_ids):
            profile = self._profiles[device_id]
            similarity = cosine_similarity(direction, profile.mean_direction)
            angular_distance = math.acos(similarity)
            variance = max(profile.variance, _MINIMUM_VARIANCE)
            scores[i] = math.exp(-(angular_distance**2) / (2.0 * variance))

        score_sum = float(scores.sum())
        if score_sum <= 0.0 or not math.isfinite(score_sum):
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )
        probabilities = scores / score_sum

        order = np.argsort(probabilities)[::-1]
        best_probability = float(probabilities[order[0]])
        second_best_probability = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        best_device_id = device_ids[order[0]]

        if best_probability < self._config.unknown_probability_threshold:
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=best_probability,
                second_best_probability=second_best_probability,
            )

        return ClassificationResult(
            target=best_device_id,
            probability=best_probability,
            second_best_probability=second_best_probability,
        )
