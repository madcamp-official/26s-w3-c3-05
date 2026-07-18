"""Synthetic end-to-end replay for calibration, targeting, UNKNOWN, and evaluation."""

from __future__ import annotations

from jarvis.calibration.session import CalibrationSession
from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.evaluation import LabeledFrame, compute_target_selection_accuracy
from jarvis.gaze.features import FaceObservation


def _observation(frame_id: int, yaw_deg: float) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=frame_id * 33,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=yaw_deg,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
    )


def _calibrate(device_id: str, yaw_deg: float) -> DeviceGazeProfile:
    session = CalibrationSession(device_id)
    for frame_id in range(30):
        session.add_observation(_observation(frame_id, yaw_deg))
    return session.finalize()


def test_synthetic_replay_exercises_complete_gaze_path() -> None:
    config = GazeConfig(smoothing_window_frames=1, unknown_max_angle_deg=25.0)
    engine = GazeTargetingEngine(config)
    engine.register_device(_calibrate("laptop", 0.0))
    engine.register_device(_calibrate("room.bulb", 40.0))

    labeled_frames = []
    scenarios = [(0.0, "laptop"), (40.0, "room.bulb"), (90.0, "UNKNOWN")]
    for frame_id, (yaw_deg, expected) in enumerate(scenarios, start=100):
        estimate = engine.process(_observation(frame_id, yaw_deg))
        labeled_frames.append(LabeledFrame(frame_id, frame_id * 33, estimate.target, expected))

    result = compute_target_selection_accuracy(
        labeled_frames,
        dataset_id="synthetic-gaze-smoke-v1",
        conditions="synthetic exact yaw; not a real-camera performance result",
    )

    assert result.accuracy == 1.0
    assert result.total_frames == 3
