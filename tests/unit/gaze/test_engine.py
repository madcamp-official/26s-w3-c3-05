"""GazeTargetingEngine: FaceObservation → TargetEstimate 전체 파이프라인."""

from __future__ import annotations

import numpy as np

from jarvis.contracts.messages import TargetEstimate
from jarvis.gaze.classifier import DeviceGazeProfile, TargetGeometry3D
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
    eyes_open: bool = True,
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
        eyes_open=eyes_open,
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


def test_short_blink_holds_last_smoothed_gaze() -> None:
    engine = GazeTargetingEngine(GazeConfig(blink_hold_ms=200))
    first = engine.process(_observation(0, 1_000, head_yaw_deg=0.0))
    held = engine.process(_observation(1, 1_100, head_yaw_deg=35.0, eyes_open=False))

    assert engine.last_smoothed_gaze is not None
    assert held.frame_id == 1
    assert held.target == first.target
    assert held.stability == first.stability


def test_long_eye_unavailable_interval_switches_to_head_only() -> None:
    engine = GazeTargetingEngine(GazeConfig(blink_hold_ms=200))
    engine.process(_observation(0, 1_000, head_yaw_deg=0.0))
    engine.process(_observation(1, 1_100, head_yaw_deg=20.0, eyes_open=False))
    engine.process(_observation(2, 1_250, head_yaw_deg=30.0, eyes_open=False))

    assert engine.last_smoothed_gaze is not None
    assert engine.last_smoothed_gaze.source == "head-only"


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


def test_3d_registered_device_resolves_correctly_through_full_pipeline() -> None:
    """head_position_mm이 프레임마다 주어지면 smoothing이 origin을 평균 내
    engine 전체 파이프라인을 통해 3D geometry 매칭까지 이어져야 한다."""
    config = GazeConfig(unknown_probability_threshold=0.5, enable_3d_target_matching=True)
    engine = GazeTargetingEngine(config)
    engine.register_device(
        DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05),
        geometry_3d=TargetGeometry3D(np.array([0.0, 0.0, 500.0]), radius_mm=50.0),
    )

    head_position = np.array([0.0, 0.0, 0.0])
    estimate = None
    for i in range(config.smoothing_window_frames + 2):
        observation = FaceObservation(
            timestamp_ms=i * 30,
            frame_id=i,
            left_iris_relative=(0.0, 0.0),
            right_iris_relative=(0.0, 0.0),
            head_yaw_deg=0.0,
            head_pitch_deg=0.0,
            head_roll_deg=0.0,
            eye_tracking_confidence=1.0,
            face_tracking_confidence=1.0,
            face_detected=True,
            head_position_mm=head_position,
        )
        estimate = engine.process(observation)

    assert estimate is not None
    assert estimate.target == "laptop"
    assert engine.last_smoothed_gaze is not None
    assert engine.last_smoothed_gaze.origin is not None
