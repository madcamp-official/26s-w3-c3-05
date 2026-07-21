"""Tests for the gaze probe's pure evaluator (no mediapipe / camera needed).

Feeds synthetic ``FaceObservation``s through ``evaluate`` and checks that each
pipeline stage's value is captured, that tracking loss and the UNKNOWN reject
reasons surface honestly, and — crucially — that the ``TargetEstimate`` matches
what ``GazeTargetingEngine.process`` produces for the same input.
"""

from __future__ import annotations

import math

import numpy as np

from jarvis.gaze.classifier import DeviceGazeProfile, TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.lock import GazeLockState, GazeLockStateMachine
from jarvis.gaze.smoothing import GazeSmoother
from jarvis.monitoring.gaze_probe import GazeProbe, evaluate


def _observation(
    *,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    detected: bool = True,
    confidence: float = 1.0,
    eyes_open: bool = True,
    iris_relative: tuple[float, float] = (0.0, 0.0),
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        left_iris_relative=iris_relative,
        right_iris_relative=iris_relative,
        head_yaw_deg=yaw,
        head_pitch_deg=pitch,
        head_roll_deg=0.0,
        eye_tracking_confidence=confidence,
        face_tracking_confidence=confidence,
        face_detected=detected,
        eyes_open=eyes_open,
    )


def _fresh() -> tuple[GazeSmoother, TargetClassifier, GazeLockStateMachine, GazeConfig]:
    config = GazeConfig()
    return GazeSmoother(config), TargetClassifier(config), GazeLockStateMachine(config), config


# --- tracking loss ------------------------------------------------------------

def test_tracking_loss_is_surfaced_not_faked() -> None:
    smoother, classifier, lock, config = _fresh()
    snapshot = evaluate(
        _observation(detected=False),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    assert snapshot.tracking_lost is True
    assert snapshot.gaze_direction is None
    assert snapshot.gaze_confidence is None
    assert snapshot.smoothed_stability is None
    assert snapshot.target == config.UNKNOWN_TARGET
    assert snapshot.target_estimate.stability == 0.0


def test_camera_pose_warning_flags_large_roll() -> None:
    smoother, classifier, lock, config = _fresh()

    snapshot = evaluate(
        _observation(yaw=0.0, pitch=0.0, detected=True),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    rolled = FaceObservation(
        timestamp_ms=33,
        frame_id=1,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=0.0,
        head_pitch_deg=0.0,
        head_roll_deg=-44.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
        eyes_open=True,
    )
    rolled_snapshot = evaluate(
        rolled,
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )

    assert snapshot.camera_pose_warning is None
    assert rolled_snapshot.camera_pose_warning is not None
    assert "head roll -44deg" in rolled_snapshot.camera_pose_warning


def test_short_face_dropout_uses_recovering_hold() -> None:
    smoother, classifier, lock, config = _fresh()
    first = evaluate(
        _observation(timestamp_ms=1_000),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    held = evaluate(
        _observation(timestamp_ms=1_100, detected=False),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )

    assert first.smoothed_gaze_direction is not None
    assert held.tracking_lost is False
    assert held.tracking_recovering is True
    assert held.smoothed_gaze_direction == first.smoothed_gaze_direction


def test_short_blink_holds_last_gaze_without_composing_jumpy_raw_vector() -> None:
    smoother, classifier, lock, config = _fresh()
    first = evaluate(
        _observation(timestamp_ms=1_000, yaw=0.0),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    blink = evaluate(
        _observation(timestamp_ms=1_100, yaw=35.0, eyes_open=False),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )

    assert first.smoothed_gaze_direction is not None
    assert blink.gaze_direction is None
    assert blink.gaze_confidence is None
    assert blink.smoothed_gaze_direction == first.smoothed_gaze_direction


def test_iris_jump_keeps_live_vector_with_low_confidence() -> None:
    smoother, classifier, lock, config = _fresh()
    first = evaluate(
        _observation(timestamp_ms=1_000, iris_relative=(0.0, 0.0)),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    jumped = evaluate(
        _observation(timestamp_ms=1_050, iris_relative=(0.5, 0.0)),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
        previous_iris_offset=(0.0, 0.0),
    )

    assert first.smoothed_gaze_direction is not None
    assert jumped.gaze_direction is not None
    assert jumped.gaze_confidence == config.ema_min_alpha
    assert jumped.smoothed_gaze_direction is not None
    assert jumped.smoothed_gaze_direction != first.smoothed_gaze_direction


def test_blink_recovery_holds_until_iris_settles() -> None:
    smoother, classifier, lock, config = _fresh()
    first = evaluate(
        _observation(timestamp_ms=1_000),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    recovered = evaluate(
        _observation(timestamp_ms=1_080, eyes_open=True, iris_relative=(0.2, 0.0)),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
        last_closed_eye_ms=1_000,
    )

    assert first.smoothed_gaze_direction is not None
    assert recovered.gaze_direction is None
    assert recovered.smoothed_gaze_direction == first.smoothed_gaze_direction


# --- no calibration profiles --------------------------------------------------

def test_no_profiles_yields_unknown_with_reason() -> None:
    smoother, classifier, lock, config = _fresh()
    snapshot = evaluate(
        _observation(),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    assert snapshot.gaze_direction is not None  # gaze vector still composed
    assert snapshot.target == config.UNKNOWN_TARGET
    assert snapshot.reject_reason is not None
    assert "프로파일" in snapshot.reject_reason
    assert snapshot.device_details == ()
    assert snapshot.buffer_fill == 1


# --- a registered, well-aligned device ---------------------------------------

def test_aligned_device_is_selected_with_small_angle() -> None:
    smoother, classifier, lock, config = _fresh()
    # Frontal gaze composes to (0, 0, 1); register a device pointing the same way.
    classifier.register_profile(
        DeviceGazeProfile(
            device_id="laptop",
            mean_direction=np.array([0.0, 0.0, 1.0]),
            variance=0.05,
        )
    )
    snapshot = evaluate(
        _observation(),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )
    assert snapshot.target == "laptop"
    assert snapshot.reject_reason is None
    assert snapshot.is_confident is True
    (detail,) = snapshot.device_details
    assert detail.device_id == "laptop"
    assert detail.is_selected is True
    assert detail.angular_distance_deg < 1.0
    assert math.isclose(detail.allowed_radius_deg, math.degrees(math.sqrt(0.05)))
    assert detail.normalized_distance < 1.0
    assert detail.within_profile_radius is True
    assert detail.range_status == "IN"


def test_device_detail_reports_outside_registered_range() -> None:
    smoother, classifier, lock, config = _fresh()
    classifier.register_profile(
        DeviceGazeProfile(
            device_id="small_target",
            mean_direction=np.array([0.0, 0.0, 1.0]),
            variance=math.radians(2.0) ** 2,
        )
    )

    snapshot = evaluate(
        _observation(yaw=12.0),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
    )

    (detail,) = snapshot.device_details
    assert detail.device_id == "small_target"
    assert detail.allowed_radius_deg == 2.0
    assert detail.normalized_distance > 1.0
    assert detail.within_profile_radius is False
    assert detail.range_status == "OUT"


def test_target_label_uses_registered_display_name() -> None:
    smoother, classifier, lock, config = _fresh()
    classifier.register_profile(
        DeviceGazeProfile(
            device_id="target_001",
            mean_direction=np.array([0.0, 0.0, 1.0]),
            variance=0.05,
        )
    )

    snapshot = evaluate(
        _observation(),
        smoother=smoother,
        classifier=classifier,
        lock=lock,
        config=config,
        target_labels={"target_001": "스피커"},
    )

    assert snapshot.target == "target_001"
    assert snapshot.target_label == "스피커"


def test_lock_reaches_target_locked_after_dwell() -> None:
    smoother, classifier, lock, config = _fresh()
    classifier.register_profile(
        DeviceGazeProfile(
            device_id="laptop", mean_direction=np.array([0.0, 0.0, 1.0]), variance=0.05
        )
    )
    last = None
    for i in range(6):
        last = evaluate(
            _observation(frame_id=i, timestamp_ms=i * 200),
            smoother=smoother,
            classifier=classifier,
            lock=lock,
            config=config,
        )
    assert last is not None
    assert last.lock_state == GazeLockState.TARGET_LOCKED
    assert last.locked_device == "laptop"


# --- honesty: evaluate() mirrors the engine ----------------------------------

def test_target_estimate_matches_engine() -> None:
    config = GazeConfig()
    profile = DeviceGazeProfile(
        device_id="laptop", mean_direction=np.array([0.0, 0.0, 1.0]), variance=0.05
    )

    engine = GazeTargetingEngine(config)
    engine.register_device(profile)

    smoother, classifier, lock, _ = _fresh()
    classifier.register_profile(profile)

    for i in range(5):
        obs = _observation(frame_id=i, timestamp_ms=i * 200, yaw=float(i))
        engine_estimate = engine.process(obs)
        snapshot = evaluate(
            obs, smoother=smoother, classifier=classifier, lock=lock, config=config
        )
        assert snapshot.target_estimate == engine_estimate


# --- probe liveness degrades honestly without a model -------------------------

def test_probe_without_model_is_unavailable() -> None:
    probe = GazeProbe(model_path=None)
    assert probe.start() is False
    assert probe.available is False
    assert "모델" in probe.status_text
    assert probe.process_bgr(object(), 0, 0) is None
