"""GazeTargetingEngine: FaceObservation → TargetEstimate 전체 파이프라인."""

from __future__ import annotations

import numpy as np

from jarvis.contracts.messages import TargetEstimate
from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.lock import GazeLockState

UNKNOWN = GazeConfig().UNKNOWN_TARGET


def _observation(
    frame_id: int,
    timestamp_ms: int,
    *,
    face_detected: bool = True,
    head_yaw_deg: float = 0.0,
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=timestamp_ms,
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


def test_process_returns_target_estimate_type() -> None:
    engine = GazeTargetingEngine()
    estimate = engine.process(_observation(0, 1_000))
    assert isinstance(estimate, TargetEstimate)


def test_no_devices_registered_yields_unknown() -> None:
    engine = GazeTargetingEngine()
    estimate = engine.process(_observation(0, 1_000))
    assert estimate.target == UNKNOWN
    assert estimate.probability == 0.0


def test_tracking_loss_yields_unknown_with_zero_stability() -> None:
    engine = GazeTargetingEngine()
    estimate = engine.process(_observation(0, 1_000, face_detected=False))
    assert estimate.target == UNKNOWN
    assert estimate.stability == 0.0
    assert engine.lock_state == GazeLockState.SEARCHING
    assert engine.last_smoothed_gaze is None


def test_dwell_leads_to_target_locked_and_estimate_matches() -> None:
    config = GazeConfig(dwell_time_ms=200, unknown_probability_threshold=0.5)
    engine = GazeTargetingEngine(config)
    engine.register_device(
        DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05)
    )

    estimate = None
    for i in range(6):
        estimate = engine.process(_observation(i, i * 50))

    assert estimate is not None
    assert estimate.target == "laptop"
    assert engine.lock_state == GazeLockState.TARGET_LOCKED
    assert engine.is_gaze_locked_to("laptop")
    assert not engine.is_gaze_locked_to("room.bulb")


def test_gesture_and_commit_hooks_delegate_to_lock() -> None:
    config = GazeConfig(dwell_time_ms=0, unknown_probability_threshold=0.5)
    engine = GazeTargetingEngine(config)
    engine.register_device(
        DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05)
    )
    engine.process(_observation(0, 1_000))
    assert engine.lock_state == GazeLockState.TARGET_LOCKED

    assert engine.notify_gesture_started(1_001) == GazeLockState.GESTURE_WAIT
    assert engine.notify_committed(1_002) == GazeLockState.COMMITTED


def test_unregister_device_falls_back_to_unknown() -> None:
    engine = GazeTargetingEngine(GazeConfig(unknown_probability_threshold=0.5))
    engine.register_device(
        DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05)
    )
    engine.unregister_device("laptop")
    estimate = engine.process(_observation(0, 1_000))
    assert estimate.target == UNKNOWN
