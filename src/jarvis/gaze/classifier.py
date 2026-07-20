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
from jarvis.gaze.feature_profile import (
    TargetAreaProfile,
    TargetFeatureProfile,
    TargetFeatureSample,
)
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
    reference_face_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("DeviceGazeProfile.device_id must not be empty")
        if self.mean_direction.shape != (3,) or not np.all(np.isfinite(self.mean_direction)):
            raise ValueError("DeviceGazeProfile.mean_direction must contain three finite values")
        norm = float(np.linalg.norm(self.mean_direction))
        if not math.isclose(norm, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"DeviceGazeProfile.mean_direction must be a unit vector, got norm={norm}"
            )
        if not math.isfinite(self.variance) or self.variance < 0:
            raise ValueError(f"DeviceGazeProfile.variance must be >= 0, got {self.variance}")
        if self.reference_face_scale is not None and (
            not math.isfinite(self.reference_face_scale) or self.reference_face_scale <= 0.0
        ):
            raise ValueError("reference_face_scale must be finite and positive")


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


@dataclass(frozen=True, slots=True)
class TargetGeometry3D:
    """3D 삼각측량(calibration/triangulation.py)으로 얻은 물체의 카메라 기준
    위치와 유효 반경.

    `radius_mm`은 실제로 측정한 물체 크기가 아니라 삼각측량 잔차에서 유도한
    판정 허용 오차다. 이 타입은 런타임 전용이며 JSON으로 직접 저장되지 않는다 —
    영속화 형태는 `calibration.registry`의 평범한 tuple 기반 레코드를 쓴다.
    """

    center_mm: Vector3
    radius_mm: float

    def __post_init__(self) -> None:
        if self.center_mm.shape != (3,) or not np.all(np.isfinite(self.center_mm)):
            raise ValueError("TargetGeometry3D.center_mm must contain three finite values")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0:
            raise ValueError(
                f"TargetGeometry3D.radius_mm must be finite and positive, got {self.radius_mm}"
            )


_MINIMUM_DEPTH_MM = 1.0
"""이보다 가까운(또는 광선 뒤쪽) 깊이는 원점 바로 앞/뒤로 퇴화한 것으로 보고
각도 기반 폴백을 쓴다 — 나눗셈 발산과 "뒤에 있는 것을 보고 있다"는 물리적으로
말이 안 되는 결과를 막는다."""


def effective_distance_and_variance(
    direction: Vector3,
    origin: Vector3 | None,
    profile: DeviceGazeProfile,
    geometry: TargetGeometry3D | None,
    config: GazeConfig,
    current_face_scale: float | None = None,
) -> tuple[float, float]:
    """등록 물체 하나에 대한 (각도 거리, 분산)을 계산한다.

    `origin`과 `geometry`가 모두 있고 깊이가 유효하면, 현재 머리 위치에서 물체
    중심을 향하는 방향을 매 프레임 새로 계산해 비교한다(3D 모드) — 분산은
    `atan(radius_mm / depth)`로 얻은 각도 반경의 제곱이며, 등록 시 각도 모드의
    최소 퍼짐(`target_minimum_angular_variance_deg`)보다 작아지지 않도록
    바닥을 둔다. 이 바닥이 없으면 작거나 먼 물체의 물리적 분산이 실제 추적
    잡음보다 좁아져, 3D 모드가 각도 모드보다 더 쉽게 UNKNOWN으로 거부해 버린다.

    그 외의 경우(geometry 없음, origin 없음, 깊이 퇴화)는 등록 시 저장한 고정
    `mean_direction` + `variance`로 비교한다 — 오늘까지의 각도 모드와 동일하다.
    """
    if origin is not None and geometry is not None:
        to_target = geometry.center_mm - origin
        depth = float(np.linalg.norm(to_target))
        if depth > _MINIMUM_DEPTH_MM:
            direction_to_target = to_target / depth
            similarity = cosine_similarity(direction, direction_to_target)
            angular_distance = math.acos(similarity)
            variance = max(
                math.atan(geometry.radius_mm / depth) ** 2,
                math.radians(config.target_minimum_angular_variance_deg) ** 2,
            )
            return angular_distance, variance

    similarity = cosine_similarity(direction, profile.mean_direction)
    angular_distance = math.acos(similarity)
    variance = profile.variance
    if (
        profile.reference_face_scale is not None
        and current_face_scale is not None
        and math.isfinite(current_face_scale)
        and current_face_scale > 0.0
    ):
        ratio = current_face_scale / profile.reference_face_scale
        spread_scale = min(2.0, max(1.0, 1.0 + 1.5 * abs(math.log(ratio))))
        variance *= spread_scale**2
    return angular_distance, variance


class TargetClassifier:
    """등록된 기기 gaze profile을 바탕으로 현재 시선 방향의 대상 기기를 추정한다."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._profiles: dict[str, DeviceGazeProfile] = {}
        self._geometries: dict[str, TargetGeometry3D] = {}
        self._feature_profiles: dict[str, TargetFeatureProfile] = {}
        self._area_profiles: dict[str, TargetAreaProfile] = {}

    def register_profile(
        self,
        profile: DeviceGazeProfile,
        geometry_3d: TargetGeometry3D | None = None,
        feature_profile: TargetFeatureProfile | None = None,
        area_profile: TargetAreaProfile | None = None,
    ) -> None:
        """기기 gaze profile을 등록하거나 갱신한다.

        `geometry_3d`가 있으면 이후 `classify()`가 `origin`과 함께 호출될 때 이
        기기는 깊이로 정규화한 3D 매칭을 우선 시도한다(`effective_distance_and_variance`
        참고). `geometry_3d=None`으로 다시 등록하면 3D geometry가 제거된다.
        """
        self._profiles[profile.device_id] = profile
        if geometry_3d is not None:
            self._geometries[profile.device_id] = geometry_3d
        else:
            self._geometries.pop(profile.device_id, None)
        if feature_profile is not None:
            self._feature_profiles[profile.device_id] = feature_profile
        else:
            self._feature_profiles.pop(profile.device_id, None)
        if area_profile is not None:
            self._area_profiles[profile.device_id] = area_profile
        else:
            self._area_profiles.pop(profile.device_id, None)

    def unregister_profile(self, device_id: str) -> None:
        self._profiles.pop(device_id, None)
        self._geometries.pop(device_id, None)
        self._feature_profiles.pop(device_id, None)
        self._area_profiles.pop(device_id, None)

    @property
    def profiles(self) -> dict[str, DeviceGazeProfile]:
        return dict(self._profiles)

    @property
    def geometries(self) -> dict[str, TargetGeometry3D]:
        """3D geometry가 등록된 기기만 반환한다(각도 전용 기기는 제외).

        모니터링 UI(`gaze_probe.py`의 `_device_details`)가 `classify()`와 같은
        깊이 보정 거리를 재계산할 때 쓴다 — 디버그 패널이 오래된 고정 각도를
        보여주지 않도록 한다.
        """
        return dict(self._geometries)

    @property
    def feature_profiles(self) -> dict[str, TargetFeatureProfile]:
        return dict(self._feature_profiles)

    @property
    def area_profiles(self) -> dict[str, TargetAreaProfile]:
        return dict(self._area_profiles)

    def classify(
        self,
        direction: Vector3,
        origin: Vector3 | None = None,
        *,
        current_face_scale: float | None = None,
        feature_sample: TargetFeatureSample | None = None,
    ) -> ClassificationResult:
        """합성된 시선 방향 단위 벡터로부터 대상 기기를 추정한다.

        `origin`이 주어지고 어떤 기기에 3D geometry가 등록되어 있으면 그 기기는
        현재 머리 위치 기준으로 새로 계산한 거리로 비교된다(3D 모드). 나머지
        기기, 또는 이번 프레임에 `origin`이 없는 경우는 등록 시 저장한 고정
        방향(각도 모드, 이전과 동일)으로 비교된다 — 3D·각도 혼합 등록도 이 하나의
        루프에서 그대로 동작한다.

        등록된 기기가 없으면 항상 UNKNOWN을 반환한다(지어낸 대상을 반환하지 않는다).
        """
        if not self._profiles:
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )

        if feature_sample is not None and (self._feature_profiles or self._area_profiles):
            return self._classify_by_feature_profile(feature_sample, direction, origin)

        device_ids = list(self._profiles.keys())
        scores = np.empty(len(device_ids), dtype=np.float64)
        angular_distances = np.empty(len(device_ids), dtype=np.float64)
        variances = np.empty(len(device_ids), dtype=np.float64)
        normalized_distances = np.empty(len(device_ids), dtype=np.float64)
        for i, device_id in enumerate(device_ids):
            profile = self._profiles[device_id]
            geometry = (
                self._geometries.get(device_id)
                if self._config.enable_3d_target_matching
                else None
            )
            angular_distance, variance = effective_distance_and_variance(
                direction, origin, profile, geometry, self._config, current_face_scale
            )
            angular_distances[i] = angular_distance
            variance = max(variance, _MINIMUM_VARIANCE)
            variances[i] = variance
            normalized_distances[i] = angular_distance / math.sqrt(variance)
            scores[i] = math.exp(-(angular_distance**2) / (2.0 * variance))

        score_sum = float(scores.sum())
        if score_sum <= 0.0 or not math.isfinite(score_sum):
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )
        probabilities = scores / score_sum

        order = np.lexsort((normalized_distances, angular_distances))
        best_probability = float(probabilities[order[0]])
        second_best_probability = float(probabilities[order[1]]) if len(order) > 1 else 0.0
        best_device_id = device_ids[order[0]]

        best_angle = float(angular_distances[order[0]])
        best_angle_deg = math.degrees(best_angle)
        best_normalized_distance = float(normalized_distances[order[0]])
        if (
            best_normalized_distance > 1.0
            or best_angle_deg > self._config.unknown_max_angle_deg
        ):
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

    def _classify_by_feature_profile(
        self,
        feature_sample: TargetFeatureSample,
        direction: Vector3 | None = None,
        origin: Vector3 | None = None,
    ) -> ClassificationResult:
        area_result = self._classify_by_area_profile(feature_sample, direction, origin)
        if area_result is not None:
            return area_result
        if not self._feature_profiles:
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )
        device_ids = list(self._feature_profiles.keys())
        distances = np.asarray(
            [
                self._feature_profiles[device_id].mahalanobis_distance(feature_sample)
                for device_id in device_ids
            ],
            dtype=np.float64,
        )
        thresholds = np.asarray(
            [self._feature_profiles[device_id].threshold for device_id in device_ids],
            dtype=np.float64,
        )
        normalized = distances / thresholds
        scores = np.exp(-0.5 * normalized**2)
        score_sum = float(scores.sum())
        if score_sum <= 0.0 or not math.isfinite(score_sum):
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=0.0,
                second_best_probability=0.0,
            )
        probabilities = scores / score_sum
        order = np.argsort(normalized)
        best_index = int(order[0])
        second_index = int(order[1]) if len(order) > 1 else None
        if (
            second_index is not None
            and float(normalized[best_index]) <= 1.0
            and float(normalized[second_index]) <= 1.0
            and float(normalized[second_index] - normalized[best_index]) <= 0.15
        ):
            best_index = self._prefer_closer_3d_target(
                device_ids,
                int(best_index),
                int(second_index),
                direction,
                origin,
            )
        best_probability = float(probabilities[best_index])
        remaining = [i for i in order if int(i) != best_index]
        second_best_probability = float(probabilities[int(remaining[0])]) if remaining else 0.0
        if float(normalized[best_index]) > 1.0:
            return ClassificationResult(
                target=self._config.UNKNOWN_TARGET,
                probability=best_probability,
                second_best_probability=second_best_probability,
            )
        return ClassificationResult(
            target=device_ids[best_index],
            probability=best_probability,
            second_best_probability=second_best_probability,
        )

    def _prefer_closer_3d_target(
        self,
        device_ids: list[str],
        first_index: int,
        second_index: int,
        direction: Vector3 | None,
        origin: Vector3 | None,
    ) -> int:
        if direction is None or origin is None:
            return first_index
        candidates = []
        for index in (first_index, second_index):
            geometry = self._geometries.get(device_ids[index])
            if geometry is None:
                continue
            to_target = geometry.center_mm - origin
            depth = float(np.linalg.norm(to_target))
            if depth <= _MINIMUM_DEPTH_MM:
                continue
            target_direction = to_target / depth
            candidates.append((math.acos(cosine_similarity(direction, target_direction)), index))
        if len(candidates) < 2:
            return first_index
        candidates.sort(key=lambda item: item[0])
        return int(candidates[0][1])

    def _classify_by_area_profile(
        self,
        feature_sample: TargetFeatureSample,
        direction: Vector3 | None,
        origin: Vector3 | None,
    ) -> ClassificationResult | None:
        if not self._area_profiles:
            return None
        candidates = [
            (
                profile.normalized_distance(
                    feature_sample.gaze_yaw,
                    feature_sample.gaze_pitch,
                    self._config.registration_max_area_radius_deg,
                ),
                device_id,
            )
            for device_id, profile in self._area_profiles.items()
            if profile.contains(
                feature_sample.gaze_yaw,
                feature_sample.gaze_pitch,
                self._config.registration_max_area_radius_deg,
            )
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        best_device_id = candidates[0][1]
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 0.15:
            device_ids = [candidates[0][1], candidates[1][1]]
            preferred = self._prefer_closer_3d_target(device_ids, 0, 1, direction, origin)
            best_device_id = device_ids[preferred]
        second = math.exp(-0.5 * candidates[1][0] ** 2) if len(candidates) > 1 else 0.0
        best = math.exp(-0.5 * candidates[0][0] ** 2)
        total = best + second
        return ClassificationResult(
            target=best_device_id,
            probability=best / total if total > 0.0 else 1.0,
            second_best_probability=second / total if total > 0.0 else 0.0,
        )
