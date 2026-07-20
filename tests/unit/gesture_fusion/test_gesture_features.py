"""README 8장 "속도·관절 각도 생성" 단계를 검증한다.

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
    LANDMARK_DIMS,
    GestureConfig,
)
from jarvis.gesture_fusion.features import (
    HandFeatureExtractor,
    compute_joint_angles,
    feature_dimension,
)
from jarvis.gesture_fusion.landmarks import HandObservation

_POSITION_DIMS = HAND_LANDMARK_COUNT * LANDMARK_DIMS


def _obs(
    landmarks: np.ndarray,
    *,
    timestamp_ms: int,
    frame_id: int,
    hand_detected: bool = True,
    wrist_position: object = None,
    palm_scale: float = 0.2,
) -> HandObservation:
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=landmarks.astype(np.float64),
        handedness="Right",
        palm_scale=palm_scale,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=hand_detected,
        wrist_position=(
            np.zeros(LANDMARK_DIMS, dtype=np.float64)
            if wrist_position is None
            else np.asarray(wrist_position, dtype=np.float64)
        ),
    )


def _zeros() -> np.ndarray:
    return np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)


# --- 관절 각도 ---


def test_straight_finger_angle_is_pi() -> None:
    """일직선으로 뻗은 세 점의 꼭짓점 각은 π(180도)."""
    landmarks = _zeros()
    a, b, c = JOINT_ANGLE_TRIPLETS[0]
    landmarks[a] = [0.0, 0.0]
    landmarks[b] = [1.0, 0.0]
    landmarks[c] = [2.0, 0.0]
    angles = compute_joint_angles(landmarks)
    assert angles[0] == pytest.approx(math.pi)


def test_right_angle_joint() -> None:
    landmarks = _zeros()
    a, b, c = JOINT_ANGLE_TRIPLETS[0]
    landmarks[a] = [1.0, 0.0]
    landmarks[b] = [0.0, 0.0]
    landmarks[c] = [0.0, 1.0]
    angles = compute_joint_angles(landmarks)
    assert angles[0] == pytest.approx(math.pi / 2)


def test_degenerate_joint_angle_is_zero_not_nan() -> None:
    landmarks = _zeros()  # 모든 점이 겹침 → 각 정의 불가
    angles = compute_joint_angles(landmarks)
    assert np.all(np.isfinite(angles))
    assert np.all(angles == 0.0)


# --- 속도 (causal) ---


def test_first_frame_has_zero_velocity() -> None:
    extractor = HandFeatureExtractor()
    features = extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    assert features.hand_detected
    # 위치·각도 뒤의 속도 블록이 모두 0이어야 한다.
    velocity = features.vector[_POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS):
                               _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS) + _POSITION_DIMS]
    assert np.all(velocity == 0.0)


def test_velocity_is_per_second_causal_difference() -> None:
    # 이 테스트는 raw 차분 수학을 검증하므로 평활화를 끈다(평활화 동작은 별도 테스트).
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    first = _zeros()
    second = _zeros()
    second[0] = [0.1, 0.0]  # 손목이 0.1만큼 이동
    extractor.push(_obs(first, timestamp_ms=1000, frame_id=1))
    features = extractor.push(_obs(second, timestamp_ms=1100, frame_id=2))  # dt=100ms
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = features.vector[offset:offset + _POSITION_DIMS].reshape(HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    # 0.1 이동 / 0.1초 = 1.0/초
    assert velocity[0, 0] == pytest.approx(1.0)


def test_lost_tracking_resets_history_and_zeros_features() -> None:
    extractor = HandFeatureExtractor()
    moving = _zeros()
    moving[0] = [0.1, 0.0]
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
    moving[0] = [0.1, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    # 500ms 공백 > 200ms → 리셋, 이 프레임 속도 0
    features = extractor.push(_obs(moving, timestamp_ms=1500, frame_id=2))
    offset = _POSITION_DIMS + len(JOINT_ANGLE_TRIPLETS)
    velocity = features.vector[offset:offset + _POSITION_DIMS]
    assert np.all(velocity == 0.0)


def test_out_of_order_timestamp_does_not_crash_or_fabricate() -> None:
    extractor = HandFeatureExtractor()
    moving = _zeros()
    moving[0] = [0.1, 0.0]
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
        include_wrist_translation=False,
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
        include_wrist_translation=False,
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
    lm[0] = [0.2, 0.1]
    features = extractor.push(_obs(lm, timestamp_ms=1000, frame_id=1))
    exposed = extractor.last_landmarks
    assert exposed is not None and exposed.shape == (21, LANDMARK_DIMS)
    # feature 벡터의 위치 블록(모델 입력)과 동일해야 한다.
    np.testing.assert_allclose(exposed.reshape(-1), features.vector[:_POSITION_DIMS])
    # 추적 손실 뒤에는 다시 None.
    extractor.push(_obs(_zeros(), timestamp_ms=1033, frame_id=2, hand_detected=False))
    assert extractor.last_landmarks is None


def test_smoothing_resets_on_tracking_loss() -> None:
    """추적 손실 뒤 첫 프레임은 평활화 상태가 리셋되어 속도 0(공백 미연결)."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=True))
    moving = _zeros()
    moving[0] = [0.3, 0.0]
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1))
    extractor.push(_obs(_zeros(), timestamp_ms=1033, frame_id=2, hand_detected=False))
    after = extractor.push(_obs(moving, timestamp_ms=1066, frame_id=3))
    assert np.all(_velocity_block(after) == 0.0)


# --- 손목 평행이동 (swipe 신호 복원, decisions.md 2026-07-19) ---

_WRIST_DIMS = LANDMARK_DIMS


def _wrist_velocity_block(features: object) -> np.ndarray:
    # 손목 그룹은 벡터 맨 뒤에 속도(2)+가속도(2) 순으로 붙는다.
    return features.vector[-2 * _WRIST_DIMS:][:_WRIST_DIMS]  # type: ignore[attr-defined]


def _wrist_acceleration_block(features: object) -> np.ndarray:
    return features.vector[-2 * _WRIST_DIMS:][_WRIST_DIMS:]  # type: ignore[attr-defined]


def test_wrist_translation_adds_six_velocity_accel_dims() -> None:
    without = GestureConfig(include_wrist_translation=False)
    with_wrist = GestureConfig(include_wrist_translation=True)
    assert feature_dimension(with_wrist) - feature_dimension(without) == 2 * _WRIST_DIMS


def test_wrist_translation_is_on_by_default() -> None:
    assert GestureConfig().include_wrist_translation is True


def test_wrist_velocity_is_per_second_causal_difference() -> None:
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0]))
    features = extractor.push(
        _obs(_zeros(), timestamp_ms=1100, frame_id=2, wrist_position=[0.2, 0.0])
    )
    # 손목 0.2 이동 / 0.1초 = 2.0/초.
    velocity = _wrist_velocity_block(features)
    assert velocity[0] == pytest.approx(2.0)
    assert velocity[1] == pytest.approx(0.0)


def test_pure_translation_invisible_in_landmarks_but_visible_in_wrist() -> None:
    """이 그룹의 존재 이유: 손 모양이 그대로인 순수 평행이동에서 landmark 속도는 0이지만
    손목 이동 속도는 살아 있어야 한다 — swipe를 구분할 유일한 신호."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    shape = _zeros()
    shape[8] = [0.5, 0.5]  # 고정된 손 모양(정규화 좌표는 매 프레임 동일)
    extractor.push(_obs(shape, timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0]))
    features = extractor.push(
        _obs(shape, timestamp_ms=1100, frame_id=2, wrist_position=[0.3, 0.0])
    )
    # landmark(위치·속도) 블록은 이동을 전혀 못 본다.
    assert np.all(_velocity_block(features) == 0.0)
    # 손목 평행이동 속도만 신호를 담는다: 0.3/0.1s = 3.0.
    assert _wrist_velocity_block(features)[0] == pytest.approx(3.0)


def test_wrist_acceleration_from_velocity_change() -> None:
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0]))
    extractor.push(_obs(_zeros(), timestamp_ms=1100, frame_id=2, wrist_position=[0.1, 0.0]))  # v=1.0
    features = extractor.push(
        _obs(_zeros(), timestamp_ms=1200, frame_id=3, wrist_position=[0.3, 0.0])
    )  # v=2.0, a=(2-1)/0.1=10
    assert _wrist_acceleration_block(features)[0] == pytest.approx(10.0)


def test_last_wrist_vectors_expose_model_input() -> None:
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    assert extractor.last_wrist_velocity is None
    assert extractor.last_wrist_acceleration is None
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0]))
    features = extractor.push(
        _obs(_zeros(), timestamp_ms=1100, frame_id=2, wrist_position=[0.2, 0.0])
    )
    velocity = extractor.last_wrist_velocity
    acceleration = extractor.last_wrist_acceleration
    assert velocity is not None and acceleration is not None
    np.testing.assert_allclose(velocity, _wrist_velocity_block(features))
    np.testing.assert_allclose(acceleration, _wrist_acceleration_block(features))
    # 추적 손실 뒤에는 다시 None.
    extractor.push(_obs(_zeros(), timestamp_ms=1133, frame_id=3, hand_detected=False))
    assert extractor.last_wrist_velocity is None
    assert extractor.last_wrist_acceleration is None


def test_wrist_translation_resets_on_tracking_loss() -> None:
    """손실 뒤 첫 프레임은 손목 히스토리도 리셋되어 속도 0(공백 넘는 허위 이동 금지)."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False))
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0]))
    extractor.push(_obs(_zeros(), timestamp_ms=1033, frame_id=2, hand_detected=False))
    after = extractor.push(
        _obs(_zeros(), timestamp_ms=1066, frame_id=3, wrist_position=[0.5, 0.0])
    )
    assert np.all(_wrist_velocity_block(after) == 0.0)


# --- palm_scale 평활화 (2026-07-19, 손목 잡음 수정) ---
#
# 원인: wrist_position = origin / palm_scale에서 분자(화면상 절대 위치)가 일반
# landmark의 분자(손 안에서의 상대적 차이)보다 훨씬 커서, 매 프레임 새로 계산되는
# palm_scale의 잡음이 나눗셈을 타고 크게 증폭된다. 정지한 손 시뮬레이션 실측으로
# 확인: 이 증폭 때문에 손목 속도 잡음이 손가락 끝 속도 잡음보다 약 2.5배 컸다(기존
# _wrist_smoother는 나눗셈 *이후* 값만 다뤄 이 증폭을 못 잡음). palm_scale 자체를
# 별도로 평활화해 나눗셈에 쓰면 정지 시 잡음이 약 3.85배 줄어든다(documents/decisions.md).


def test_palm_scale_smoothing_is_on_by_default() -> None:
    assert GestureConfig().smooth_palm_scale is True


def test_palm_scale_smoothing_reduces_wrist_velocity_noise_on_a_still_hand() -> None:
    """손목이 실제로 정지해 있어도(원점 고정) palm_scale 측정값 자체가 프레임마다
    떨리면 wrist_position=origin/palm_scale의 나눗셈이 그 잡음을 증폭한다.
    palm_scale 평활화가 이 증폭을 줄여야 한다(landmark 평활화 검증과 같은 패턴 —
    같은 노이즈 시퀀스를 on/off 두 추출기에 흘려 속도 에너지를 비교)."""
    rng = np.random.default_rng(1)
    true_origin = np.array([1.0, 0.0])  # 실제로는 고정된 손목(원점)
    base_scale = 0.2
    noisy_scales = base_scale + rng.normal(0.0, 0.006, size=40)  # palm_scale 자체의 프레임별 잡음(~3%)

    raw = HandFeatureExtractor(GestureConfig(smooth_landmarks=True, smooth_palm_scale=False))
    smooth = HandFeatureExtractor(GestureConfig(smooth_landmarks=True, smooth_palm_scale=True))
    raw_energy = 0.0
    smooth_energy = 0.0
    for i, scale in enumerate(noisy_scales):
        scale = float(scale)
        wrist_position = (true_origin / scale).tolist()
        obs = _obs(
            _zeros(), timestamp_ms=1000 + i * 33, frame_id=i,
            wrist_position=wrist_position, palm_scale=scale,
        )
        raw_energy += float(np.sum(_wrist_velocity_block(raw.push(obs)) ** 2))
        smooth_energy += float(np.sum(_wrist_velocity_block(smooth.push(obs)) ** 2))

    # 손목은 실제로 정지해 있으므로 이상적 속도는 0. palm_scale 평활화가 이 증폭된
    # 미분 노이즈를 확실히 낮춰야 한다.
    assert smooth_energy < raw_energy * 0.5


def test_palm_scale_smoothing_is_noop_when_scale_never_changes() -> None:
    """palm_scale이 프레임마다 똑같으면(잡음 없음) 재조정 비율이 항상 1이라
    평활화 on/off 결과가 완전히 같아야 한다 — 진짜 신호를 왜곡하지 않는지 확인."""

    def _run(smooth_scale: bool) -> np.ndarray:
        extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=True, smooth_palm_scale=smooth_scale))
        extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0], palm_scale=0.2))
        features = extractor.push(
            _obs(_zeros(), timestamp_ms=1033, frame_id=2, wrist_position=[0.2, 0.0], palm_scale=0.2)
        )
        return _wrist_velocity_block(features)

    np.testing.assert_allclose(_run(smooth_scale=True), _run(smooth_scale=False), atol=1e-9)


def test_disabling_palm_scale_smoothing_keeps_old_behavior_available() -> None:
    """smooth_palm_scale=False는 이 수정 이전 동작(재조정 없음)과 동일해야 한다."""
    extractor = HandFeatureExtractor(GestureConfig(smooth_landmarks=False, smooth_palm_scale=False))
    extractor.push(_obs(_zeros(), timestamp_ms=1000, frame_id=1, wrist_position=[0.0, 0.0], palm_scale=0.2))
    features = extractor.push(
        _obs(_zeros(), timestamp_ms=1100, frame_id=2, wrist_position=[0.2, 0.0], palm_scale=0.5)
    )
    # 재조정이 없으므로 raw wrist_position 그대로 차분: 0.2/0.1s = 2.0 (palm_scale 변화와 무관).
    velocity = _wrist_velocity_block(features)
    assert velocity[0] == pytest.approx(2.0)
