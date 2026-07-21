"""ClipRecorder(training/clip_recorder.py) — 카메라 없이 합성 관측값으로 녹화 규칙을 검증.

CLI(record_webcam_clips)와 모니터 GUI가 공유하는 순수 코어라, 여기서 저장·폐기·카운터·
fps 리샘플·미검출 게이트를 전부 단위 테스트한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation, _lost_tracking_observation
from training.clip_recorder import ClipRecorder
from training.data.clip_cache import load_clip


def _observation(frame_id: int, timestamp_ms: int) -> HandObservation:
    landmarks = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    landmarks[:, 0] = 0.1 + 0.01 * frame_id
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=landmarks,
        handedness="Right",
        palm_scale=0.2,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=True,
        wrist_position=np.array([0.05, -0.03], dtype=np.float64),
    )


def _record_clip(
    recorder: ClipRecorder,
    *,
    person: str,
    gesture: str,
    frames: int,
    interval_ms: int = 33,
    detected: list[bool] | None = None,
) -> object:
    recorder.start(person, gesture)
    for i in range(frames):
        if detected is not None and not detected[i]:
            recorder.add(_lost_tracking_observation(i, 1000 + i * interval_ms))
        else:
            recorder.add(_observation(i, 1000 + i * interval_ms))
    return recorder.stop()


def test_saves_a_clip_with_expected_fields(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    result = _record_clip(recorder, person="me-s1", gesture="slide_two_fingers_up", frames=20)

    assert result.saved is True
    saved_path = tmp_path / "webcam" / "me-s1" / "me-s1-slide_two_fingers_up-0000.npz"
    assert saved_path.is_file()
    clip = load_clip(saved_path)
    assert clip.gesture_label == "slide_two_fingers_up"
    assert clip.landmarks.shape == (20, HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    assert clip.wrist_position.shape == (20, LANDMARK_DIMS)
    assert np.all(np.diff(clip.timestamp_ms) > 0)  # 엄격 증가


def test_target_fps_resamples_to_12fps(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3, target_fps=12.0)
    result = _record_clip(
        recorder, person="me-s1", gesture="rotate_clockwise", frames=31, interval_ms=33
    )  # ~1초 30fps

    assert result.saved is True
    clip = load_clip(tmp_path / "webcam" / "me-s1" / f"{result.clip_id}.npz")
    mean_interval = float(np.mean(np.diff(clip.timestamp_ms)))
    assert abs(mean_interval - 1000.0 / 12.0) < 2.0  # 저장 시 12fps로 정규화됨
    assert len(clip) < 31


def test_discards_empty_clip(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    recorder.start("me-s1", "none")
    result = recorder.stop()
    assert result.saved is False
    assert not (tmp_path / "webcam" / "me-s1").exists() or not list(
        (tmp_path / "webcam" / "me-s1").glob("*.npz")
    )


def test_discards_clip_with_too_many_missing_frames(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    detected = [i >= 6 for i in range(10)]  # 앞 6/10 미검출 = 60% > 30%
    result = _record_clip(
        recorder, person="me-s1", gesture="none", frames=10, detected=detected
    )
    assert result.saved is False
    assert "미검출" in result.detail


def test_keeps_clip_when_missing_within_tolerance(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    detected = [i >= 2 for i in range(10)]  # 2/10 미검출 = 20% ≤ 30%
    result = _record_clip(
        recorder, person="me-s1", gesture="none", frames=10, detected=detected
    )
    assert result.saved is True
    clip = load_clip(tmp_path / "webcam" / "me-s1" / f"{result.clip_id}.npz")
    assert not np.all(clip.hand_detected)  # 미검출 프레임이 클립에 남아 있음


def test_counter_continues_across_clips(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    r0 = _record_clip(recorder, person="me-s1", gesture="none", frames=5)
    r1 = _record_clip(recorder, person="me-s1", gesture="rotate_clockwise", frames=5)
    assert r0.clip_id == "me-s1-none-0000"
    assert r1.clip_id == "me-s1-rotate_clockwise-0001"  # 사람 폴더 전체 기준으로 증가


def test_add_ignored_when_not_recording(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    recorder.add(_observation(0, 1000))  # start 전 → 무시
    assert recorder.frame_count == 0
    assert recorder.stop().saved is False  # 녹화 중이 아니었음


def test_start_rejects_empty_person_or_gesture(tmp_path: Path) -> None:
    recorder = ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3)
    with pytest.raises(ValueError, match="person_id"):
        recorder.start("  ", "none")
    with pytest.raises(ValueError, match="gesture_label"):
        recorder.start("me-s1", "")


def test_rejects_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_missing_frame_fraction"):
        ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=1.5)
    with pytest.raises(ValueError, match="target_fps"):
        ClipRecorder(cache_dir=tmp_path, max_missing_frame_fraction=0.3, target_fps=0.0)
