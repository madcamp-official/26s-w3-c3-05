from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jarvis.gaze.calibration_model import GazeCalibrationSample, GazeCalibrationStore
from jarvis.gaze.mlp_calibration import GazeMLPCalibrationModel


def _samples(frames_per_target: int = 30) -> list[GazeCalibrationSample]:
    rows: list[GazeCalibrationSample] = []
    targets = (
        ("left", -22.0, 3.0),
        ("center", 0.0, 9.0),
        ("right", 24.0, -4.0),
    )
    for target_id, target_yaw, target_pitch in targets:
        for frame in range(frames_per_target):
            head_yaw = -24.0 + 48.0 * frame / max(1, frames_per_target - 1)
            head_pitch = 12.0 * np.sin(frame * 0.35)
            raw_yaw = target_yaw + 0.32 * head_yaw + 0.006 * head_yaw * abs(head_yaw)
            raw_pitch = target_pitch + 0.28 * head_pitch
            iris_x = (target_yaw - 0.25 * head_yaw) / 45.0
            iris_y = (target_pitch - 0.40 * head_pitch) / 45.0
            rows.append(
                GazeCalibrationSample(
                    features=(
                        1.0,
                        raw_yaw,
                        raw_pitch,
                        iris_x,
                        iris_y,
                        iris_x,
                        iris_y,
                        head_yaw,
                        head_pitch,
                        0.0,
                        0.5 + head_yaw * 0.001,
                        0.5 + head_pitch * 0.001,
                        0.1,
                    ),
                    target_yaw=target_yaw,
                    target_pitch=target_pitch,
                    target_id=target_id,
                )
            )
    return rows


def test_residual_mlp_improves_held_out_pose_error() -> None:
    model = GazeMLPCalibrationModel.fit(_samples())

    assert model.fitted
    assert model.validation_raw_error_deg is not None
    assert model.validation_mlp_error_deg is not None
    assert model.validation_mlp_error_deg < model.validation_raw_error_deg * 0.5

    sample = _samples()[17]
    prediction = model.predict_yaw_pitch(sample.features)
    assert prediction is not None
    raw_error = abs(sample.features[1] - sample.target_yaw)
    corrected_error = abs(prediction[0] - sample.target_yaw)
    assert corrected_error < raw_error


def test_store_persists_mlp_and_replaces_one_targets_online_samples(tmp_path: Path) -> None:
    path = tmp_path / "gaze_regressor.json"
    store = GazeCalibrationStore(path)
    store.add_samples(_samples())

    assert store.mlp_model.fitted
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mlp_model"]["architecture"] == [12, 24, 12, 2]

    replacement = [sample for sample in _samples(20) if sample.target_id == "left"]
    original_non_left = sum(sample.target_id != "left" for sample in store.samples)
    store.add_samples(replacement, replace_target_id="left")
    assert len(store.samples) == original_non_left + len(replacement)

    reloaded = GazeCalibrationStore(path)
    assert reloaded.mlp_model.fitted
    assert reloaded.preferred_model is reloaded.mlp_model
