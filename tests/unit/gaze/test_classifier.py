"""README 7장 "Target 추정"(코사인 유사도 baseline + 분산 정규화 + UNKNOWN rejection)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.gaze.classifier import (
    DeviceGazeProfile,
    TargetClassifier,
    TargetGeometry3D,
    effective_distance_and_variance,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import (
    TargetAreaProfile,
    TargetFeatureProfile,
    TargetFeatureSample,
    build_feature_profile,
)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def test_unknown_when_no_devices_registered() -> None:
    classifier = TargetClassifier()
    result = classifier.classify(_unit([0.0, 0.0, 1.0]))
    assert result.target == GazeConfig().UNKNOWN_TARGET
    assert result.probability == 0.0
    assert result.second_best_probability == 0.0


def _feature_profile(gaze_yaw: float, head_yaw: float) -> TargetFeatureProfile:
    return TargetFeatureProfile(
        mean=(gaze_yaw, 5.0, head_yaw, 10.0, 0.0, 0.10),
        covariance=(
            (4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 9.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 9.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 9.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.01),
        ),
        sample_count=20,
        threshold=2.5,
    )


def test_feature_profile_selects_nearest_distribution() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(
        DeviceGazeProfile("monitor", _unit([0.0, 0.0, 1.0]), variance=0.05),
        feature_profile=_feature_profile(0.0, 0.0),
    )
    classifier.register_profile(
        DeviceGazeProfile("speaker", _unit([0.3, 0.0, 0.95]), variance=0.05),
        feature_profile=_feature_profile(-20.0, -20.0),
    )

    result = classifier.classify(
        _unit([0.0, 0.0, 1.0]),
        feature_sample=TargetFeatureSample(0.5, 5.0, 1.0, 10.0, 0.0, 0.10),
    )

    assert result.target == "monitor"


def test_feature_profile_rejects_far_distribution() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(
        DeviceGazeProfile("monitor", _unit([0.0, 0.0, 1.0]), variance=0.05),
        feature_profile=_feature_profile(0.0, 0.0),
    )

    result = classifier.classify(
        _unit([0.0, 0.0, 1.0]),
        feature_sample=TargetFeatureSample(30.0, 40.0, 50.0, 60.0, 20.0, 0.30),
    )

    assert result.target == GazeConfig().UNKNOWN_TARGET


def test_area_profile_accepts_gaze_inside_registered_object_boundary() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(
        DeviceGazeProfile("monitor", _unit([0.0, 0.0, 1.0]), variance=0.01),
        feature_profile=_feature_profile(0.0, 0.0),
        area_profile=TargetAreaProfile(
            center_yaw=0.0,
            center_pitch=5.0,
            radius_yaw=8.0,
            radius_pitch=4.0,
            sample_count=80,
        ),
    )

    result = classifier.classify(
        _unit([0.0, 0.0, 1.0]),
        feature_sample=TargetFeatureSample(5.0, 5.0, 40.0, 30.0, 0.0, 0.2),
    )

    assert result.target == "monitor"


def test_area_profile_runtime_cap_rejects_overwide_registered_area() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(
        DeviceGazeProfile("monitor", _unit([0.0, 0.0, 1.0]), variance=0.01),
        area_profile=TargetAreaProfile(
            center_yaw=0.0,
            center_pitch=0.0,
            radius_yaw=20.0,
            radius_pitch=20.0,
            sample_count=80,
        ),
    )

    result = classifier.classify(
        _unit([0.0, 0.0, 1.0]),
        feature_sample=TargetFeatureSample(8.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    )

    assert result.target == GazeConfig().UNKNOWN_TARGET


def test_area_profile_accepts_near_boundary_with_tolerance() -> None:
    classifier = TargetClassifier(
        GazeConfig(
            unknown_probability_threshold=0.0,
            registration_max_area_radius_deg=8.0,
        )
    )
    classifier.register_profile(
        DeviceGazeProfile("monitor", _unit([0.0, 0.0, 1.0]), variance=0.01),
        area_profile=TargetAreaProfile(
            center_yaw=0.0,
            center_pitch=0.0,
            radius_yaw=8.0,
            radius_pitch=8.0,
            sample_count=80,
        ),
    )

    result = classifier.classify(
        _unit([0.0, 0.0, 1.0]),
        feature_sample=TargetFeatureSample(8.5, 0.0, 0.0, 0.0, 0.0, 0.1),
    )

    assert result.target == "monitor"


def test_built_feature_profile_tolerates_head_pose_when_gaze_matches() -> None:
    profile = build_feature_profile(
        [
            TargetFeatureSample(1.0, 4.0, -5.0, 2.0, 0.0, 0.10),
            TargetFeatureSample(1.2, 4.1, -4.0, 2.5, 0.5, 0.10),
            TargetFeatureSample(0.8, 3.9, -6.0, 1.5, -0.5, 0.10),
            TargetFeatureSample(1.1, 4.0, -5.5, 2.2, 0.2, 0.10),
        ]
    ).profile

    same_gaze_different_head = TargetFeatureSample(1.0, 4.0, -15.0, 3.5, -0.6, 0.10)

    assert profile.mahalanobis_distance(same_gaze_different_head) <= profile.threshold


def test_selects_closest_device_by_cosine_similarity() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(
        DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.05)
    )
    classifier.register_profile(
        DeviceGazeProfile("room.bulb", _unit([1.0, 0.0, 0.2]), variance=0.05)
    )

    result = classifier.classify(_unit([0.0, 0.0, 1.0]))
    assert result.target == "laptop"
    assert result.probability > result.second_best_probability


def test_rejects_as_unknown_below_probability_threshold() -> None:
    config = GazeConfig(unknown_probability_threshold=0.95)
    classifier = TargetClassifier(config)
    laptop = DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.05)
    bulb = DeviceGazeProfile("room.bulb", _unit([1.0, 0.0, 0.2]), variance=0.05)
    classifier.register_profile(laptop)
    classifier.register_profile(bulb)

    # 두 기기 방향의 정확히 중간(각 프로토타입과 등거리)을 보면 두 기기의 확률이
    # 비슷해져 top-1 확률이 임계값 아래로 떨어져야 한다.
    ambiguous = _unit((laptop.mean_direction + bulb.mean_direction).tolist())
    result = classifier.classify(ambiguous)
    assert result.target == config.UNKNOWN_TARGET
    assert result.probability < config.unknown_probability_threshold


def test_accepts_nearest_profile_when_inside_registered_range_despite_low_probability() -> None:
    """Overlapping demo targets should still surface the closest in-range target.

    Probability can be below 0.8 simply because another registered target is
    nearby.  The more useful live rule is: if the nearest target is inside its
    own registered radius, show that target and let diagnostics expose overlap.
    """
    config = GazeConfig(unknown_probability_threshold=0.8, minimum_margin=0.2)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "left_speaker",
            _unit([math.sin(math.radians(-16.0)), 0.0, math.cos(math.radians(-16.0))]),
            variance=math.radians(20.0) ** 2,
        )
    )
    classifier.register_profile(
        DeviceGazeProfile(
            "monitor",
            _unit([math.sin(math.radians(2.8)), 0.0, math.cos(math.radians(2.8))]),
            variance=math.radians(15.0) ** 2,
        )
    )

    result = classifier.classify(
        _unit([math.sin(math.radians(-21.8)), 0.0, math.cos(math.radians(-21.8))])
    )

    assert result.target == "left_speaker"
    assert result.probability < config.unknown_probability_threshold


def test_closer_target_wins_over_looser_higher_probability_profile() -> None:
    config = GazeConfig(unknown_probability_threshold=0.0)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "close_tight",
            _unit([math.sin(math.radians(4.0)), 0.0, math.cos(math.radians(4.0))]),
            variance=math.radians(4.5) ** 2,
        )
    )
    classifier.register_profile(
        DeviceGazeProfile(
            "far_loose",
            _unit([math.sin(math.radians(-10.0)), 0.0, math.cos(math.radians(-10.0))]),
            variance=math.radians(30.0) ** 2,
        )
    )

    result = classifier.classify(_unit([0.0, 0.0, 1.0]))

    assert result.target == "close_tight"


def test_single_registered_device_rejects_gaze_far_from_profile() -> None:
    """기기가 하나여도 상대확률 1.0만으로 먼 시선을 선택하면 안 된다."""
    config = GazeConfig(unknown_probability_threshold=0.8, unknown_max_angle_deg=25.0)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.05)
    )

    result = classifier.classify(_unit([1.0, 0.0, 0.0]))

    assert result.target == config.UNKNOWN_TARGET
    assert result.probability == pytest.approx(1.0)


def test_absolute_angle_threshold_accepts_nearby_single_device() -> None:
    config = GazeConfig(unknown_probability_threshold=0.8, unknown_max_angle_deg=25.0)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.05)
    )

    result = classifier.classify(_unit([math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))]))

    assert result.target == "laptop"


def test_two_devices_probabilities_sum_to_one() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(DeviceGazeProfile("a", _unit([0.0, 0.0, 1.0]), variance=0.02))
    classifier.register_profile(DeviceGazeProfile("b", _unit([0.3, 0.0, 0.95]), variance=0.02))

    result = classifier.classify(_unit([0.05, 0.0, 0.99]))
    assert result.probability + result.second_best_probability == pytest.approx(1.0)


def test_larger_registered_variance_is_more_tolerant() -> None:
    """같은 각도 오차라도 등록 시 분산이 큰(느슨한) 기기가 더 높은 확률을 받아야 한다."""
    offset_rad = math.radians(10.0)
    tight = DeviceGazeProfile(
        "tight", _unit([math.sin(offset_rad), 0.0, math.cos(offset_rad)]), variance=0.002
    )
    loose = DeviceGazeProfile(
        "loose", _unit([-math.sin(offset_rad), 0.0, math.cos(offset_rad)]), variance=0.2
    )
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(tight)
    classifier.register_profile(loose)

    # 두 기기 모두 query와 정확히 10도 떨어져 있어 raw 코사인 유사도는 동일하다.
    result = classifier.classify(_unit([0.0, 0.0, 1.0]))
    assert result.target == "loose"


def test_unregister_profile_removes_device() -> None:
    classifier = TargetClassifier(GazeConfig(unknown_probability_threshold=0.0))
    classifier.register_profile(DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.05))
    classifier.unregister_profile("laptop")
    result = classifier.classify(_unit([0.0, 0.0, 1.0]))
    assert result.target == GazeConfig().UNKNOWN_TARGET


def test_device_gaze_profile_rejects_non_unit_direction() -> None:
    with pytest.raises(ValueError):
        DeviceGazeProfile("bad", np.array([1.0, 1.0, 1.0]), variance=0.01)


def test_device_gaze_profile_rejects_negative_variance() -> None:
    with pytest.raises(ValueError):
        DeviceGazeProfile("bad", _unit([0.0, 0.0, 1.0]), variance=-0.1)


def test_device_gaze_profile_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        DeviceGazeProfile("bad", np.array([0.0, 0.0, float("nan")]), variance=0.01)
    with pytest.raises(ValueError):
        DeviceGazeProfile("bad", _unit([0.0, 0.0, 1.0]), variance=float("nan"))


# --- 3D geometry (documents/decisions.md: "3D 시도 + 신뢰도 낮으면 각도 모델로 폴백") ---


def test_target_geometry_3d_rejects_bad_center_or_radius() -> None:
    with pytest.raises(ValueError):
        TargetGeometry3D(np.array([0.0, 0.0, float("nan")]), radius_mm=20.0)
    with pytest.raises(ValueError):
        TargetGeometry3D(np.array([0.0, 0.0, 1.0]), radius_mm=0.0)
    with pytest.raises(ValueError):
        TargetGeometry3D(np.array([0.0, 0.0, 1.0]), radius_mm=-5.0)


def test_3d_geometry_corrects_for_head_movement_since_registration() -> None:
    """등록 후 머리가 크게 이동하면 고정 각도(mean_direction)는 더 이상 맞지 않는다.

    3D geometry가 있으면 현재 머리 위치(origin) 기준으로 매 프레임 새로 방향을
    계산하므로, 각도 전용 모델이 오답을 내는 상황에서도 올바른 기기를 고른다.
    """
    config = GazeConfig(unknown_probability_threshold=0.5, enable_3d_target_matching=True)
    classifier = TargetClassifier(config)

    bulb_position = np.array([300.0, 0.0, 2000.0])
    registration_origin = np.array([0.0, 0.0, 0.0])
    bulb_registered_direction = _unit((bulb_position - registration_origin).tolist())

    classifier.register_profile(DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.02))
    classifier.register_profile(
        DeviceGazeProfile("bulb", bulb_registered_direction, variance=0.02),
        geometry_3d=TargetGeometry3D(center_mm=bulb_position, radius_mm=50.0),
    )

    runtime_origin = np.array([800.0, 0.0, 0.0])
    true_direction_to_bulb = _unit((bulb_position - runtime_origin).tolist())

    # 각도 전용(=origin 없이 classify)이면 등록된 어떤 target 반경에도 안정적으로
    # 들어오지 않으므로 UNKNOWN으로 거부한다.
    stale_result = classifier.classify(true_direction_to_bulb)
    assert stale_result.target == config.UNKNOWN_TARGET

    # origin이 주어지면 bulb의 3D geometry로 매 프레임 새로 계산해 올바르게 고른다.
    corrected_result = classifier.classify(true_direction_to_bulb, origin=runtime_origin)
    assert corrected_result.target == "bulb"


def test_3d_geometry_is_diagnostic_only_by_default() -> None:
    """Webcam demo stability: geometry may be stored, but live matching uses the
    angle profile unless `enable_3d_target_matching` is explicitly enabled."""
    config = GazeConfig(unknown_probability_threshold=0.5)
    classifier = TargetClassifier(config)

    bulb_position = np.array([300.0, 0.0, 2000.0])
    bulb_registered_direction = _unit(bulb_position.tolist())
    classifier.register_profile(DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.02))
    classifier.register_profile(
        DeviceGazeProfile("bulb", bulb_registered_direction, variance=0.02),
        geometry_3d=TargetGeometry3D(center_mm=bulb_position, radius_mm=50.0),
    )

    runtime_origin = np.array([800.0, 0.0, 0.0])
    true_direction_to_bulb = _unit((bulb_position - runtime_origin).tolist())

    result = classifier.classify(true_direction_to_bulb, origin=runtime_origin)

    assert result.target == config.UNKNOWN_TARGET


def test_classify_without_origin_ignores_registered_geometry() -> None:
    """origin을 넘기지 않으면 3D geometry가 등록되어 있어도 각도 모드와 동일하다
    (하위 호환 — 기존 호출부는 그대로 동작해야 한다)."""
    config = GazeConfig(unknown_probability_threshold=0.0)
    with_geometry = TargetClassifier(config)
    without_geometry = TargetClassifier(config)

    profile = DeviceGazeProfile("bulb", _unit([0.1, 0.0, 0.99]), variance=0.02)
    with_geometry.register_profile(
        profile, geometry_3d=TargetGeometry3D(np.array([300.0, 0.0, 2000.0]), radius_mm=50.0)
    )
    without_geometry.register_profile(profile)

    direction = _unit([0.05, 0.0, 0.99])
    result_with = with_geometry.classify(direction)
    result_without = without_geometry.classify(direction)

    assert result_with.target == result_without.target
    assert result_with.probability == pytest.approx(result_without.probability)


def test_degenerate_depth_falls_back_to_angular() -> None:
    """물체 위치가 현재 머리 위치와 거의 같으면(깊이 퇴화) 각도 모드로 대체된다."""
    profile = DeviceGazeProfile("bulb", _unit([0.0, 0.0, 1.0]), variance=0.02)
    origin = np.array([100.0, 50.0, 10.0])
    geometry = TargetGeometry3D(center_mm=origin.copy(), radius_mm=20.0)

    angular_distance, variance = effective_distance_and_variance(
        _unit([0.0, 0.0, 1.0]), origin, profile, geometry, GazeConfig()
    )

    assert angular_distance == pytest.approx(0.0, abs=1e-9)
    assert variance == pytest.approx(profile.variance)


def test_variance_floor_prevents_3d_mode_being_stricter_than_angular() -> None:
    """작거나 먼 물체의 물리적 각도 분산이 각도 모드 최소 퍼짐보다 좁아지지
    않아야 한다 — 그렇지 않으면 3D 모드가 각도 모드보다 더 쉽게 UNKNOWN을 낸다."""
    config = GazeConfig(target_minimum_angular_variance_deg=4.0)
    profile = DeviceGazeProfile("bulb", _unit([0.0, 0.0, 1.0]), variance=0.02)
    origin = np.array([0.0, 0.0, 0.0])
    # 작은 반경(10mm)·먼 거리(5000mm) -> 물리적 각도 분산은 매우 작다.
    geometry = TargetGeometry3D(center_mm=np.array([0.0, 0.0, 5000.0]), radius_mm=10.0)

    _angular_distance, variance = effective_distance_and_variance(
        _unit([0.0, 0.0, 1.0]), origin, profile, geometry, config
    )

    floor = math.radians(config.target_minimum_angular_variance_deg) ** 2
    assert variance == pytest.approx(floor)


def test_geometries_property_and_unregister_removes_geometry() -> None:
    classifier = TargetClassifier()
    profile = DeviceGazeProfile("bulb", _unit([0.0, 0.0, 1.0]), variance=0.02)
    geometry = TargetGeometry3D(np.array([0.0, 0.0, 2000.0]), radius_mm=30.0)
    classifier.register_profile(profile, geometry_3d=geometry)

    assert "bulb" in classifier.geometries

    classifier.unregister_profile("bulb")
    assert classifier.geometries == {}
    assert classifier.profiles == {}


def test_reregistering_without_geometry_clears_previous_geometry() -> None:
    classifier = TargetClassifier()
    profile = DeviceGazeProfile("bulb", _unit([0.0, 0.0, 1.0]), variance=0.02)
    classifier.register_profile(
        profile, geometry_3d=TargetGeometry3D(np.array([0.0, 0.0, 2000.0]), radius_mm=30.0)
    )
    assert "bulb" in classifier.geometries

    classifier.register_profile(profile)
    assert "bulb" not in classifier.geometries
