"""README 7장 "Target 추정"(코사인 유사도 baseline + 분산 정규화 + UNKNOWN rejection)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import GazeConfig


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def test_unknown_when_no_devices_registered() -> None:
    classifier = TargetClassifier()
    result = classifier.classify(_unit([0.0, 0.0, 1.0]))
    assert result.target == GazeConfig().UNKNOWN_TARGET
    assert result.probability == 0.0
    assert result.second_best_probability == 0.0


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
