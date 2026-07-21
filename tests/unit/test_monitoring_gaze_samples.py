"""User-triggered gaze sample persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.gaze.classifier import TargetClassifier
from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.feature_profile import TargetAreaProfile
from jarvis.gaze.lock import GazeLockStateMachine
from jarvis.gaze.smoothing import GazeSmoother
from jarvis.monitoring.gaze_probe import GazeSnapshot, evaluate
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample

import numpy as np


def _snapshot(frame_id: int = 1) -> GazeSnapshot:
    observation = FaceObservation(
        timestamp_ms=frame_id * 33,
        frame_id=frame_id,
        left_iris_relative=(0.1, -0.2),
        right_iris_relative=(0.2, -0.1),
        head_yaw_deg=3.0,
        head_pitch_deg=-4.0,
        head_roll_deg=1.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=True,
        left_eye_center_normalized=(0.4, 0.3),
        right_eye_center_normalized=(0.6, 0.3),
    )
    config = GazeConfig()
    return evaluate(
        observation,
        smoother=GazeSmoother(config),
        classifier=TargetClassifier(config),
        lock=GazeLockStateMachine(config),
        config=config,
    )


def _snapshot_with_target(frame_id: int = 1) -> GazeSnapshot:
    config = GazeConfig(unknown_probability_threshold=0.0)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "speaker",
            np.array([-0.1270695, 0.1472359, 0.9809052], dtype=np.float64),
            variance=0.1,
        )
    )
    return evaluate(
        FaceObservation(
            timestamp_ms=frame_id * 33,
            frame_id=frame_id,
            left_iris_relative=(0.1, -0.2),
            right_iris_relative=(0.2, -0.1),
            head_yaw_deg=3.0,
            head_pitch_deg=-4.0,
            head_roll_deg=1.0,
            eye_tracking_confidence=1.0,
            face_tracking_confidence=1.0,
            face_detected=True,
            left_eye_center_normalized=(0.4, 0.3),
            right_eye_center_normalized=(0.6, 0.3),
        ),
        smoother=GazeSmoother(config),
        classifier=classifier,
        lock=GazeLockStateMachine(config),
        config=config,
    )


def _snapshot_with_traced_area(frame_id: int = 1) -> GazeSnapshot:
    config = GazeConfig(unknown_probability_threshold=0.0)
    classifier = TargetClassifier(config)
    classifier.register_profile(
        DeviceGazeProfile(
            "speaker",
            np.array([-0.1270695, 0.1472359, 0.9809052], dtype=np.float64),
            variance=0.1,
        ),
        area_profile=TargetAreaProfile(
            center_yaw=-7.4,
            center_pitch=-8.5,
            radius_yaw=4.0,
            radius_pitch=4.0,
            sample_count=20,
            boundary_polygon=(
                (-11.4, -12.5),
                (-3.4, -12.5),
                (-3.4, -4.5),
                (-11.4, -4.5),
            ),
        ),
    )
    return evaluate(
        FaceObservation(
            timestamp_ms=frame_id * 33,
            frame_id=frame_id,
            left_iris_relative=(0.1, -0.2),
            right_iris_relative=(0.2, -0.1),
            head_yaw_deg=3.0,
            head_pitch_deg=-4.0,
            head_roll_deg=1.0,
            eye_tracking_confidence=1.0,
            face_tracking_confidence=1.0,
            face_detected=True,
            left_eye_center_normalized=(0.4, 0.3),
            right_eye_center_normalized=(0.6, 0.3),
        ),
        smoother=GazeSmoother(config),
        classifier=classifier,
        lock=GazeLockStateMachine(config),
        config=config,
    )


def test_store_persists_snapshot_as_json(tmp_path: Path) -> None:
    path = tmp_path / "gaze_samples.json"
    store = GazeSampleStore(path)

    sample = store.add(_snapshot())

    assert sample["gaze_direction"] == pytest.approx(
        [-0.1270695, 0.1472359, 0.9809052], abs=1e-3
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["head_pose_deg"]["pitch"] == -4.0
    assert payload[0]["left_iris_relative"] == [0.1, -0.2]
    assert payload[0]["gaze_yaw_pitch_deg"]["yaw"] == pytest.approx(-7.4, abs=0.1)
    assert payload[0]["target_label"] == "UNKNOWN"
    assert payload[0]["confirmed_target"] is None
    assert payload[0]["dwell_required_ms"] == 3000
    assert payload[0]["unknown_elapsed_ms"] == 0
    assert payload[0]["unknown_required_ms"] == 2000
    assert payload[0]["gaze_velocity_deg_s"] is None
    assert payload[0]["gaze_acceleration_deg_s2"] is None
    assert payload[0]["gaze_motion_history_valid"] is True
    assert "personal_feature_weights" not in payload[0]
    assert payload[0]["nearest_target_range"] is None


def test_store_enforces_capacity_across_reloads(tmp_path: Path) -> None:
    path = tmp_path / "gaze_samples.json"
    store = GazeSampleStore(path, capacity=2)
    store.add(_snapshot(1))
    store.add(_snapshot(2))

    reloaded = GazeSampleStore(path, capacity=2)
    assert reloaded.full
    with pytest.raises(ValueError, match="capacity"):
        reloaded.add(_snapshot(3))


def test_format_sample_shows_vector_head_and_target(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")
    rendered = format_gaze_sample(store.add(_snapshot()))

    assert "#1 [1f]" in rendered
    assert "gaze=(-0.127, +0.147, +0.981)" in rendered
    assert "raw_y/p=(-7.4, -8.5)" in rendered
    assert "final_y/p=(-7.4, -8.5)" in rendered
    assert "head=(+3.0, -4.0, +1.0)" in rendered
    assert "실시간=응시대상 없음 P=0.00" in rendered
    assert "확정=없음 dwell=0.0/3.0s" in rendered


def test_format_sample_shows_raw_nearest_target(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")

    sample = store.add(_snapshot_with_target())
    rendered = format_gaze_sample(sample)

    assert sample["raw_nearest_target_range"]["device_id"] == "speaker"
    assert "raw_angle=speaker" in rendered


def test_format_sample_prioritizes_authoritative_traced_area(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")

    sample = store.add(_snapshot_with_traced_area())
    rendered = format_gaze_sample(sample)

    area = sample["nearest_traced_area"]
    assert area["device_id"] == "speaker"
    assert area["status"] == "IN"
    assert area["hull_vertex_count"] == 4
    assert "area=speaker" in rendered
    assert "IN hull=4" in rendered
    assert rendered.index("area=speaker") < rendered.index("angle=speaker")


def test_window_averages_multiple_smoothed_frames(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")
    snapshots = [_snapshot(frame_id) for frame_id in range(1, 6)]

    sample = store.add_window(snapshots)

    assert sample["window_frame_count"] == 5
    assert sample["window_duration_ms"] == 132
    assert sample["gaze_direction"] == pytest.approx(
        [-0.1270695, 0.1472359, 0.9809052], abs=1e-3
    )
    assert sample["raw_gaze_yaw_pitch_deg"] == pytest.approx(
        {"yaw": -7.4, "pitch": -8.5}, abs=0.1
    )
    assert "calibration_applied" not in sample
    assert sample["face_metrics"] == pytest.approx(
        {"center": [0.5, 0.3], "scale": 0.2}, abs=1e-6
    )


def test_not_enough_frames_error_includes_diagnostic_counts(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")
    snapshots = [_snapshot(1), _snapshot(2)]

    with pytest.raises(ValueError) as exc_info:
        store.add_window(snapshots, minimum_frames=3)

    message = str(exc_info.value)
    assert "not enough valid gaze frames: 2/3" in message
    assert "history=2" in message
    assert "face=2" in message
    assert "smoothed=2" in message
    assert "eyes_open=2" in message


def test_clear_empties_memory_and_persisted_file(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    store = GazeSampleStore(path)
    store.add(_snapshot())

    store.clear()

    assert store.count == 0
    assert json.loads(path.read_text(encoding="utf-8")) == []
