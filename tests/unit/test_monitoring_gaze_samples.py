"""User-triggered gaze sample persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.gaze.classifier import TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.lock import GazeLockStateMachine
from jarvis.gaze.smoothing import GazeSmoother
from jarvis.monitoring.gaze_probe import GazeSnapshot, evaluate
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample


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


def test_store_persists_snapshot_as_json(tmp_path: Path) -> None:
    path = tmp_path / "gaze_samples.json"
    store = GazeSampleStore(path)

    sample = store.add(_snapshot())

    assert sample["gaze_direction"] == pytest.approx(
        [-0.1372622, 0.1506876, 0.9790058], abs=1e-3
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["head_pose_deg"]["pitch"] == -4.0
    assert payload[0]["left_iris_relative"] == [0.1, -0.2]
    assert payload[0]["gaze_yaw_pitch_deg"]["yaw"] == pytest.approx(-8.0, abs=0.1)
    assert payload[0]["target_label"] == "UNKNOWN"
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
    assert "gaze=(-0.137, +0.151, +0.979)" in rendered
    assert "gaze_y/p=(-8.0, -8.7)" in rendered
    assert "head=(+3.0, -4.0, +1.0)" in rendered
    assert "판단=응시대상 없음 P=0.00" in rendered


def test_window_averages_multiple_smoothed_frames(tmp_path: Path) -> None:
    store = GazeSampleStore(tmp_path / "samples.json")
    snapshots = [_snapshot(frame_id) for frame_id in range(1, 6)]

    sample = store.add_window(snapshots)

    assert sample["window_frame_count"] == 5
    assert sample["window_duration_ms"] == 132
    assert sample["gaze_direction"] == pytest.approx(
        [-0.1372622, 0.1506876, 0.9790058], abs=1e-3
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
