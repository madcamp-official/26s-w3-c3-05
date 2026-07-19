"""Augmentation(training/augment.py) — 좌우반전+라벨스왑, 시간축 속도 변형을 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation
from training.augment import flip_landmarks, time_warp
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
        ("swipe_left", "swipe_right"),
        ("swipe_right", "swipe_left"),
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


def test_time_warp_marks_all_frames_as_detected() -> None:
    clip = _clip("swipe_down", length=15)
    warped = time_warp(clip, rate=1.5)
    assert np.all(warped.hand_detected)


def test_time_warp_rejects_non_positive_rate() -> None:
    clip = _clip("swipe_down")
    with pytest.raises(ValueError):
        time_warp(clip, rate=0.0)
    with pytest.raises(ValueError):
        time_warp(clip, rate=-1.0)
