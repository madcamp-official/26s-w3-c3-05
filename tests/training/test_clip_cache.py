"""클립 캐시 포맷(training/data/clip_cache.py) 저장·로드 왕복을 검증한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation
from training.data.clip_cache import CachedClip, load_clip, observations_to_cached_clip, save_clip


def _observation(frame_id: int, timestamp_ms: int) -> HandObservation:
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=np.full((HAND_LANDMARK_COUNT, LANDMARK_DIMS), 0.1 * frame_id, dtype=np.float64),
        handedness="Right",
        palm_scale=0.2,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=True,
        wrist_position=np.array([0.01 * frame_id, -0.02 * frame_id], dtype=np.float64),
    )


def test_observations_to_cached_clip_stacks_fields() -> None:
    observations = [_observation(i, 1000 + i * 33) for i in range(5)]
    clip = observations_to_cached_clip(observations, "swipe_down", "clip-001")
    assert len(clip) == 5
    assert clip.landmarks.shape == (5, HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    assert clip.wrist_position.shape == (5, LANDMARK_DIMS)
    np.testing.assert_allclose(clip.landmarks[2], observations[2].landmarks)
    assert clip.gesture_label == "swipe_down"
    assert clip.clip_id == "clip-001"


def test_observations_to_cached_clip_rejects_empty() -> None:
    with pytest.raises(ValueError):
        observations_to_cached_clip([], "swipe_down", "clip-empty")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    observations = [_observation(i, 1000 + i * 33) for i in range(7)]
    clip = observations_to_cached_clip(observations, "rotate_clockwise", "clip-002")
    out_path = tmp_path / "clip-002.npz"

    save_clip(out_path, clip)
    assert out_path.is_file()
    assert not out_path.with_name(out_path.name + ".tmp").exists()  # 임시 파일이 남지 않음

    loaded = load_clip(out_path)
    assert loaded.clip_id == clip.clip_id
    assert loaded.gesture_label == clip.gesture_label
    np.testing.assert_allclose(loaded.landmarks, clip.landmarks)
    np.testing.assert_allclose(loaded.wrist_position, clip.wrist_position)
    np.testing.assert_allclose(loaded.palm_scale, clip.palm_scale)
    np.testing.assert_array_equal(loaded.hand_detected, clip.hand_detected)
    np.testing.assert_array_equal(loaded.timestamp_ms, clip.timestamp_ms)


def test_cached_clip_rejects_mismatched_lengths() -> None:
    observations = [_observation(i, 1000 + i * 33) for i in range(3)]
    clip = observations_to_cached_clip(observations, "swipe_down", "clip-003")
    with pytest.raises(ValueError):
        CachedClip(
            clip_id=clip.clip_id,
            gesture_label=clip.gesture_label,
            landmarks=clip.landmarks,
            wrist_position=clip.wrist_position[:2],  # 길이 불일치
            palm_scale=clip.palm_scale,
            detection_confidence=clip.detection_confidence,
            handedness_score=clip.handedness_score,
            hand_detected=clip.hand_detected,
            timestamp_ms=clip.timestamp_ms,
        )
