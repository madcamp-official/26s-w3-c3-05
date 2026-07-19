"""README 8장 "속도·가속도·관절 각도 생성" 단계를 검증한다.

핵심 회귀 대상: (1) causal 차분(과거만 사용), (2) monotonic timestamp 기반 dt,
(3) 추적 손실·프레임 공백에서 허위 속도를 만들지 않는 리셋, (4) config로 feature
그룹을 켜고 끄는 교체 가능성.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.gesture_fusion.config import (
    HAND_LANDMARK_COUNT,
    JOINT_ANGLE_TRIPLETS,
    GestureConfig,
)
from jarvis.gesture_fusion.features import (
    HandFeatureExtractor,
    compute_joint_angles,
    feature_dimension,
)
from jarvis.gesture_fusion.landmarks import HandObservation

_POSITION_DIMS = HAND_LANDMARK_COUNT * 3


def _obs(
    landmarks: np.ndarray,
    *,
    timestamp_ms: int,
    frame_id: int,
    hand_detected: bool = True,
) -> HandObservation:
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=landmarks.astype(np.float64),
        handedness="Right",
        palm_scale=0.2,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=hand_detected,
    )


def _zeros() -> np.ndarray:
    return np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float64)


# --- 관절 각도 ---


def test_straight_finger_angle_is_pi() -> None:
    """일직선으로 뻗은 세 점의 꼭짓점 각은 π(180도)."""
    landmarks = _zeros()
    a, b, c = JOINT_ANGLE_TRIPLETS[0]
    landmarks[a] = [0.0, 0.0, 0.0]
    landmarks[b] = [1.0, 0.0, 0.0]
    landmarks[c] = [2.0, 0.0, 0.0]
    angles = compute_joint_angles(landmarks)
    assert angles[0] == pytest.approx(math.pi)


def test_right_angle_joint() -> None:
    landmarks = _zeros()
    a, b, c = JOINT_ANGLE_TRIPLETS[0]
    landmarks[a] = [1.0, 0.0, 0.0]
    landmarks[b] = [0.0, 0.0, 0.0]
    landmarks[c] = [0.0, 1.0, 0.0]
    angles = compute_joint_angles(landmarks)
    assert angles[0] == pytest.approx(math.pi / 2)


def test_degenerate_joint_angle_is_zero_not_nan() -> None:
    landmarks = _zeros()  # 모든 점이 겹침 → 각 정의 불가
    angles = compute_joint_angles(landmarks)
    assert np.all(np.isfinite(angles))
    assert np.all(angles == 0.0)


# --- 속도·가속도 (causal) ---


def test_first_frame_has_zero_velocity_and_acceleration() -> None:
    extractor = HandFeatureExtractor()
    features = extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    assert features.hand_detected
    # 위치·각도 뒤의 속도·가속도 블록이 모두 0이어야 한다.
    velocity = features.vector[_POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS):
                               _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS) + _POSITION_DIMS]
    assert np.all(velocity == 0.0)


def test_velocity_is_per_second_causal_difference() -> None:
    # 이 테스트는 raw 차분 수학을 검증하므로 평활화를 끈다(평활화 동작은 별도 테스트).
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    first = _zeros()
    second = _zeros()
    second[0] = [0.1, 0.0, 0.0]  # 손목이 0.1만큼 이동
    extractor.push(_obs(first, timestamp_ms=1000, frame_id=1))
    features = extractor.push(_obs(second, timestamp_ms=1100, frame_id=2))  # dt=100ms
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = features.vector[offset:offset + _POSITION_DIMS].reshape(HAND_LANDMARK_COUNT, 3)
    # 0.1 이동 / 0.1초 = 1.0/초
    assert velocity[0, 0] == pytest.approx(1.0)


def test_acceleration_from_velocity_change() -> None:
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    f0 = _zeros()
    f1 = _zeros()
    f1[0] = [0.1, 0.0, 0.0]
    f2 = _zeros()
    f2[0] = [0.3, 0.0, 0.0]  # 속도 증가
    extractor.push(_obs(f0, timestamp_ms=1000, frame_id=1))
    extractor.push(_obs(f1, timestamp_ms=1100, frame_id=2))  # v=1.0
    features = extractor.push(_obs(f2, timestamp_ms=1200, frame_id=3))  # v=2.0, a=(2-1)/0.1=10
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS) + _POSITION_DIMS
    accel = features.vector[offset:offset + _POSITION_DIMS].reshape(HAND_LANDMARK_COUNT, 3)
    assert accel[0, 0] == pytest.approx(10.0)


def test_lost_tracking_resets_history_and_zeros_features() -> None:
    extractor = HandFeatureExtractor()
    moving = _zeros()
    moving[0] = [0.1, 0.0, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    lost = extractor.push(_obs(_zeros(), timestamp_ms=1050, frame_id=2, hand_detected=False))
    assert not lost.hand_detected
    assert np.all(lost.vector == 0.0)
    # 손실 뒤 첫 프레임은 history가 리셋되어 속도 0이어야 한다(공백 넘는 허위 속도 금지).
    after = extractor.push(_obs(moving, timestamp_ms=1100, frame_id=3))
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = after.vector[offset:offset + _POSITION_DIMS]
    assert np.all(velocity == 0.0)


def test_large_frame_gap_resets_history() -> None:
    config = GestureConfig(max_frame_gap_ms=200)
    extractor = HandFeatureExtractor(config)
    moving = _zeros()
    moving[0] = [0.1, 0.0, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    # 500ms 공백 > 200ms → 리셋, 이 프레임 속도 0
    features = extractor.push(_obs(moving, timestamp_ms=1500, frame_id=2))
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = features.vector[offset:offset + _POSITION_DIMS]
    assert np.all(velocity == 0.0)


def test_out_of_order_timestamp_does_not_crash_or_fabricate() -> None:
    extractor = HandFeatureExtractor()
    moving = _zeros()
    moving[0] = [0.1, 0.0, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    # timestamp 역전 → dt<=0, 리셋되어 속도 0
    features = extractor.push(_obs(moving, timestamp_ms=900, frame_id=2))
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = features.vector[offset:offset + _POSITION_DIMS]
    assert np.all(np.isfinite(features.vector))
    assert np.all(velocity == 0.0)


# --- feature 그룹 교체 가능성 ---


def test_feature_dimension_matches_vector_length() -> None:
    config = GestureConfig()
    extractor = HandFeatureExtractor(config)
    features = extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    assert features.vector.shape[0] == feature_dimension(config) == extractor.dimension


def test_disabling_groups_shrinks_vector() -> None:
    config = GestureConfig(
        include_positions=True,
        include_joint_angles=True,
        include_velocity=False,
        include_acceleration=False,
    )
    assert feature_dimension(config) == _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    extractor = HandFeatureExtractor(config)
    features = extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    assert features.vector.shape[0] == _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)


def test_angles_only_config() -> None:
    config = GestureConfig(
        include_positions=False,
        include_joint_angles=True,
        include_velocity=False,
        include_acceleration=False,
    )
    assert feature_dimension(config) == len(JOINT_ANGLE_TRIPLETS)


# --- landmark 평활화 (One-Euro, 미분 전 노이즈 제거) ---


def test_smoothing_is_on_by_default() -> None:
    assert GestureConfig().smooth_landmarks is True


def _velocity_block(features: object) -> np.ndarray:
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    return features.vector[offset:offset + _POSITION_DIMS]  # type: ignore[attr-defined]


def test_smoothing_reduces_velocity_noise_on_a_still_hand() -> None:
    """정지한 손 + 고주파 지터를 넣으면, 평활화가 속도 feature의 노이즈를 줄여야 한다.

    같은 노이즈 시퀀스를 평활화 on/off 두 추출기에 흘려 속도 블록의 에너지를 비교한다.
    """
    rng = np.random.default_rng(0)
    base = _zeros()
    base[:] = 0.5  # 정지한 손(모든 점 고정)
    noisy = []
    for i in range(30):
        frame = base + rng.normal(0.0, 0.01, size=base.shape)  # ±0.01 지터
        noisy.append(_obs(frame, timestamp_ms=1000 + i * 33, frame_id=i))

    raw = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    smooth = HandFeatureExtractor(GestureConfig(smooth_landmarks=True))
    raw_energy = 0.0
    smooth_energy = 0.0
    for obs in noisy:
        raw_energy += float(np.sum(_velocity_block(raw.push(obs)) ** 2))
        smooth_energy += float(np.sum(_velocity_block(smooth.push(obs)) ** 2))

    # 정지 신호이므로 이상적 속도는 0. 평활화가 미분 노이즈를 확실히 낮춰야 한다.
    assert smooth_energy < raw_energy * 0.5


def test_last_landmarks_exposes_the_model_input() -> None:
    """last_landmarks는 모델에 실제로 들어간 (평활화된) 정점과 같아야 한다."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=True))
    assert extractor.last_landmarks is None  # 첫 push 전
    lm = _zeros()
    lm[0] = [0.2, 0.1, 0.0]
    features = extractor.push(_obs(lm, timestamp_ms=1000, frame_id=1))
    exposed = extractor.last_landmarks
    assert exposed is not None and exposed.shape == (21, 3)
    # feature 벡터의 위치 블록(모델 입력)과 동일해야 한다.
    np.testing.assert_allclose(exposed.reshape(-1), features.vector[:_POSITION_DIMS])
    # 추적 손실 뒤에는 다시 None.
    extractor.push(_obs(_zeros(), timestamp_ms=1033, frame_id=2, hand_detected=False))
    assert extractor.last_landmarks is None


def test_smoothing_resets_on_tracking_loss() -> None:
    """추적 손실 뒤 첫 프레임은 평활화 상태가 리셋되어 속도 0(공백 미연결)."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=True))
    moving = _zeros()
    moving[0] = [0.3, 0.0, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    extractor.push(_obs(_zeros(), timestamp_ms=1033, frame_id=2, hand_detected=False))
    after = extractor.push(_obs(moving, timestamp_ms=1066, frame_id=3))
    assert np.all(_velocity_block(after) == 0.0)
