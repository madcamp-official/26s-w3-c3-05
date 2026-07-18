"""README 5.1 "초기 기기 등록": 여러 프레임 → mean_direction + variance로 축약."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.calibration.session import CalibrationSession
from jarvis.gaze.features import FaceObservation


def _observation(frame_id: int, head_yaw_deg: float, face_detected: bool = True) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=1_000 + frame_id,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=head_yaw_deg,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=face_detected,
    )


def test_constant_gaze_yields_zero_variance_and_matching_mean() -> None:
    session = CalibrationSession("room.bulb")
    for i in range(30):
        assert session.add_observation(_observation(i, head_yaw_deg=0.0))

    profile = session.finalize()
    assert profile.device_id == "room.bulb"
    np.testing.assert_allclose(profile.mean_direction, [0.0, 0.0, 1.0], atol=1e-9)
    assert profile.variance == pytest.approx(0.0, abs=1e-12)


def test_noisy_gaze_yields_positive_variance() -> None:
    session = CalibrationSession("room.bulb")
    for i in range(20):
        yaw = 5.0 if i % 2 == 0 else -5.0
        session.add_observation(_observation(i, head_yaw_deg=yaw))

    profile = session.finalize()
    assert profile.variance > 0.0


def test_tracking_loss_frames_are_ignored() -> None:
    session = CalibrationSession("room.bulb")
    session.add_observation(_observation(0, head_yaw_deg=0.0))
    ignored = session.add_observation(_observation(1, head_yaw_deg=0.0, face_detected=False))
    assert ignored is False
    assert session.frame_count == 1


def test_finalize_raises_without_any_valid_frames() -> None:
    session = CalibrationSession("room.bulb")
    session.add_observation(_observation(0, head_yaw_deg=0.0, face_detected=False))
    with pytest.raises(ValueError):
        session.finalize()


def test_finalize_discards_raw_frames() -> None:
    session = CalibrationSession("room.bulb")
    session.add_observation(_observation(0, head_yaw_deg=0.0))
    session.finalize()
    assert session.frame_count == 0
