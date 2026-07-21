"""Augmentation(training/augment.py) — 좌우반전+라벨스왑, 시간축 속도 변형을 검증한다."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation
from training.augment import flip_landmarks, resample_clip_to_fps, time_warp
from training.data.clip_cache import observations_to_cached_clip


def _observation(frame_id: int, timestamp_ms: int) -> HandObservation:
    landmarks = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    landmarks[:, 0] = 0.1 + 0.01 * frame_id  # x
    landmarks[:, 1] = 0.2 - 0.02 * frame_id  # y
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=landmarks,
        handedness="Right",
        palm_scale=0.2,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=True,
        wrist_position=np.array([0.05 + 0.01 * frame_id, -0.03], dtype=np.float64),
    )


def _clip(gesture_label: str, length: int = 10, fps_interval_ms: int = 33):
    observations = [_observation(i, 1000 + i * fps_interval_ms) for i in range(length)]
    return observations_to_cached_clip(observations, gesture_label, "clip-flip")


def test_flip_negates_x_only() -> None:
    clip = _clip("swipe_left")
    flipped = flip_landmarks(clip)

    np.testing.assert_allclose(flipped.landmarks[..., 0], -clip.landmarks[..., 0])
    np.testing.assert_allclose(flipped.landmarks[..., 1], clip.landmarks[..., 1])
    np.testing.assert_allclose(flipped.wrist_position[..., 0], -clip.wrist_position[..., 0])
    np.testing.assert_allclose(flipped.wrist_position[..., 1], clip.wrist_position[..., 1])


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("slide_two_fingers_left", "slide_two_fingers_right"),
        ("slide_two_fingers_right", "slide_two_fingers_left"),
        ("rotate_clockwise", "rotate_counter_clockwise"),
        ("none", "none"),
    ],
)
def test_flip_swaps_label(label: str, expected: str) -> None:
    clip = _clip(label)
    flipped = flip_landmarks(clip)
    assert flipped.gesture_label == expected


def test_flip_does_not_mutate_original() -> None:
    clip = _clip("swipe_up")
    original_x = clip.landmarks[..., 0].copy()
    flip_landmarks(clip)
    np.testing.assert_allclose(clip.landmarks[..., 0], original_x)


def test_time_warp_identity_rate_is_noop() -> None:
    clip = _clip("swipe_down", length=10)
    warped = time_warp(clip, rate=1.0)
    assert len(warped) == len(clip)
    np.testing.assert_allclose(warped.landmarks, clip.landmarks)


def test_time_warp_speeds_up_reduces_frame_count() -> None:
    clip = _clip("swipe_down", length=20)
    warped = time_warp(clip, rate=2.0)
    assert len(warped) < len(clip)
    assert len(warped) >= 2


def test_time_warp_slows_down_increases_frame_count() -> None:
    clip = _clip("swipe_down", length=20)
    warped = time_warp(clip, rate=0.5)
    assert len(warped) > len(clip)


def test_time_warp_never_drops_below_two_frames() -> None:
    clip = _clip("swipe_down", length=3)
    warped = time_warp(clip, rate=100.0)
    assert len(warped) >= 2


def test_time_warp_regenerates_evenly_spaced_timestamps() -> None:
    clip = _clip("swipe_down", length=20, fps_interval_ms=33)
    warped = time_warp(clip, rate=2.0)
    diffs = np.diff(warped.timestamp_ms)
    assert np.all(diffs > 0)  # monotonic 증가 유지


def test_time_warp_marks_all_frames_as_detected_when_source_is_fully_detected() -> None:
    clip = _clip("swipe_down", length=15)
    warped = time_warp(clip, rate=1.5)
    assert np.all(warped.hand_detected)


def test_time_warp_preserves_missing_frame_region_after_resample() -> None:
    """미검출 구간(2026-07-20부터 클립에 남을 수 있음)이 리샘플 후에도 살아남아야 한다.

    이걸 다시 전부 True로 덮어쓰면 dataset.py의 IGNORE_INDEX 마스킹이 깨져서
    0벡터(신호 없음) 프레임에 실제 제스처 라벨이 붙어버린다.
    """
    clip = _clip("swipe_down", length=20)
    hand_detected = np.ones(20, dtype=np.bool_)
    hand_detected[8:14] = False  # 클립 중간에 미검출 구간
    clip = replace(clip, hand_detected=hand_detected)

    warped = time_warp(clip, rate=2.0)
    assert not np.all(warped.hand_detected)  # 미검출 구간이 완전히 사라지지 않음
    assert np.any(warped.hand_detected)  # 검출 구간도 남아 있음


def test_time_warp_rejects_non_positive_rate() -> None:
    clip = _clip("swipe_down")
    with pytest.raises(ValueError):
        time_warp(clip, rate=0.0)
    with pytest.raises(ValueError):
        time_warp(clip, rate=-1.0)


# --- resample_clip_to_fps: fps 정규화 (augmentation 아님) ---
#
# 웹캠 30fps 클립을 pretrain(Jester 12fps)에 정합시키는 저장 직전 변환. time_warp와
# 달리 duration을 보존하고 프레임 간격만 target fps로 바꾼다.

_MEAN_X_VEL_30FPS = 0.01 / (33 / 1000.0)  # _observation의 프레임당 0.01, 33ms 간격


def _mean_x_velocity(clip: object) -> float:
    """랜드마크 x의 평균 velocity(단위/초) — 실제 dt로 나눈 causal 미분."""
    xs = clip.landmarks[:, 0, 0]  # type: ignore[attr-defined]
    ts_s = clip.timestamp_ms.astype(np.float64) / 1000.0  # type: ignore[attr-defined]
    return float(np.mean(np.diff(xs) / np.diff(ts_s)))


def test_resample_downsamples_30fps_to_12fps_frame_count() -> None:
    clip = _clip("slide_two_fingers_up", length=31, fps_interval_ms=33)  # ~1초, 30fps
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    # ~1초를 12fps로 → 약 12~13프레임
    assert 11 <= len(resampled) <= 14
    assert len(resampled) < len(clip)


def test_resample_uses_target_interval_timestamps_not_original() -> None:
    """이 설계의 정의적 테스트 — 간격이 원본(33ms)이 아니라 target(83.3ms)이어야 한다.

    time_warp를 그대로 쓰면 원본 간격이 남아 velocity가 부풀려진다. 이 테스트가 그
    회귀를 막는다.
    """
    clip = _clip("slide_two_fingers_up", length=31, fps_interval_ms=33)
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    mean_interval = float(np.mean(np.diff(resampled.timestamp_ms)))
    assert abs(mean_interval - 1000.0 / 12.0) < 2.0  # ≈83.3ms

    # 같은 프레임 수로 줄여도 time_warp는 원본 간격(~33ms)을 유지한다(다른 의미).
    warped = time_warp(clip, rate=len(clip) / len(resampled))
    warped_interval = float(np.mean(np.diff(warped.timestamp_ms)))
    assert abs(warped_interval - 33.0) < 5.0
    assert mean_interval > warped_interval * 2  # 확실히 다른 스케일


def test_resample_preserves_velocity() -> None:
    """duration 보존 + target 간격 timestamp라 units/초 velocity가 유지돼야 한다."""
    clip = _clip("slide_two_fingers_up", length=31, fps_interval_ms=33)
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    assert abs(_mean_x_velocity(resampled) - _MEAN_X_VEL_30FPS) < 0.02 * _MEAN_X_VEL_30FPS


def test_resample_preserves_duration() -> None:
    clip = _clip("slide_two_fingers_up", length=31, fps_interval_ms=33)
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    original_duration = int(clip.timestamp_ms[-1] - clip.timestamp_ms[0])
    new_duration = int(resampled.timestamp_ms[-1] - resampled.timestamp_ms[0])
    assert abs(new_duration - original_duration) < 1000.0 / 12.0  # 1프레임 이내


def test_resample_preserves_missing_frame_region() -> None:
    clip = _clip("slide_two_fingers_up", length=31)
    hand_detected = np.ones(31, dtype=np.bool_)
    hand_detected[12:20] = False
    clip = replace(clip, hand_detected=hand_detected)
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    assert not np.all(resampled.hand_detected)
    assert np.any(resampled.hand_detected)


def test_resample_noop_when_already_target_fps() -> None:
    clip = _clip("slide_two_fingers_up", length=13, fps_interval_ms=83)  # 이미 ~12fps
    resampled = resample_clip_to_fps(clip, target_fps=12.0)
    assert len(resampled) == len(clip)


def test_resample_rejects_non_positive_fps() -> None:
    clip = _clip("slide_two_fingers_up", length=10)
    with pytest.raises(ValueError):
        resample_clip_to_fps(clip, target_fps=0.0)
    with pytest.raises(ValueError):
        resample_clip_to_fps(clip, target_fps=-12.0)
