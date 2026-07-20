from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.gaze.calibration_model import (
    GazeCalibrationSample,
    GazeCalibrationStore,
    observation_features,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.features import FaceObservation, compose_gaze_vector


def _observation(*, head_yaw: float, iris_x: float) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=0,
        frame_id=0,
        left_iris_relative=(iris_x, 0.0),
        right_iris_relative=(iris_x, 0.0),
        head_yaw_deg=head_yaw,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
        left_eye_center_normalized=(0.4, 0.3),
        right_eye_center_normalized=(0.6, 0.3),
    )


def test_calibration_model_corrects_head_pose_bias(tmp_path: Path) -> None:
    config = GazeConfig()
    store = GazeCalibrationStore(tmp_path / "gaze_regressor.json", regularization=0.01)
    observations = [
        _observation(head_yaw=-20.0, iris_x=0.65),
        _observation(head_yaw=0.0, iris_x=0.0),
        _observation(head_yaw=20.0, iris_x=-0.65),
        _observation(head_yaw=30.0, iris_x=-0.95),
    ]
    samples = []
    for observation in observations:
        raw = compose_gaze_vector(observation, config)
        assert raw is not None
        samples.append(
            GazeCalibrationSample(
                features=observation_features(observation, raw),
                target_yaw=0.0,
                target_pitch=8.0,
            )
        )

    model = store.add_samples(samples)
    raw_test = compose_gaze_vector(_observation(head_yaw=25.0, iris_x=-0.8), config)
    assert raw_test is not None
    corrected = model.correct(_observation(head_yaw=25.0, iris_x=-0.8), raw_test)

    yaw, pitch = direction_to_yaw_pitch(corrected.direction)
    assert yaw == pytest.approx(0.0, abs=2.0)
    assert pitch == pytest.approx(8.0, abs=2.0)


def test_calibration_store_persists_samples_and_model(tmp_path: Path) -> None:
    path = tmp_path / "gaze_regressor.json"
    sample = GazeCalibrationSample(
        features=tuple(float(i) for i in range(13)),
        target_yaw=1.0,
        target_pitch=2.0,
    )

    store = GazeCalibrationStore(path)
    store.add_samples([sample])
    reloaded = GazeCalibrationStore(path)

    assert len(reloaded.samples) == 1
    assert reloaded.model.fitted is True
    assert np.asarray(reloaded.model.to_dict()["coefficients"]).shape == (13, 2)
