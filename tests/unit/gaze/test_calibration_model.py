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
    for observation in [
        _observation(head_yaw=-30.0, iris_x=-0.6),
        _observation(head_yaw=30.0, iris_x=0.6),
    ]:
        raw = compose_gaze_vector(observation, config)
        assert raw is not None
        samples.append(
            GazeCalibrationSample(
                features=observation_features(observation, raw),
                target_yaw=25.0,
                target_pitch=0.0,
            )
        )

    model = store.add_samples(samples)
    test_observation = _observation(head_yaw=30.0, iris_x=-0.5)
    raw_test = compose_gaze_vector(test_observation, config)
    assert raw_test is not None
    raw_yaw, _raw_pitch = direction_to_yaw_pitch(raw_test.direction)
    corrected = model.correct(test_observation, raw_test)

    yaw, pitch = direction_to_yaw_pitch(corrected.direction)
    assert abs(yaw) < abs(raw_yaw)
    assert pitch == pytest.approx(8.0, abs=3.0)


def test_single_target_calibration_does_not_force_every_gaze_to_that_target(
    tmp_path: Path,
) -> None:
    config = GazeConfig()
    observation = _observation(head_yaw=0.0, iris_x=0.0)
    raw = compose_gaze_vector(observation, config)
    assert raw is not None
    store = GazeCalibrationStore(tmp_path / "gaze_regressor.json")
    model = store.add_samples(
        [
            GazeCalibrationSample(
                features=observation_features(observation, raw),
                target_yaw=2.8,
                target_pitch=7.8,
            )
        ]
    )

    assert model.fitted is False
    corrected = model.correct(observation, raw)
    assert corrected == raw


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
    assert reloaded.model.fitted is False
    assert np.asarray(reloaded.model.to_dict()["coefficients"]).shape == (13, 2)


def test_preview_model_replaces_target_in_memory_without_persisting(tmp_path: Path) -> None:
    path = tmp_path / "gaze_regressor.json"
    store = GazeCalibrationStore(path)
    original = [
        GazeCalibrationSample(
            features=tuple(float(index) for index in range(13)),
            target_yaw=target_yaw,
            target_pitch=0.0,
            target_id=target_id,
        )
        for target_id, target_yaw in (("left", -10.0), ("right", 10.0))
    ]
    store.add_samples(original)
    before = path.read_text(encoding="utf-8")
    replacement = GazeCalibrationSample(
        features=tuple(float(index + 1) for index in range(13)),
        target_yaw=-12.0,
        target_pitch=2.0,
        target_id="left",
    )

    preview = store.preview_model([replacement], replace_target_id="left")

    assert preview is not None
    assert preview.fitted
    assert preview.sample_count == 2
    assert len(store.samples) == 2
    assert path.read_text(encoding="utf-8") == before
