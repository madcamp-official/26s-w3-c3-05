"""Gaze → Fusion 계약(documents/interface-contract.md 1번)에 대한 producer 쪽 검증.

`jarvis.gaze.engine.GazeTargetingEngine`이 만드는 값이 항상 유효한
`jarvis.contracts.messages.TargetEstimate`인지 확인한다. 이 계약이 바뀌면
이 테스트와 documents/interface-contract.md를 같은 변경 단위에서 갱신한다
(development-principles.md 4절).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from jarvis.contracts.messages import TargetEstimate
from jarvis.gaze.classifier import DeviceGazeProfile, TargetGeometry3D
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.engine import GazeTargetingEngine
from jarvis.gaze.features import FaceObservation


def _observation(
    frame_id: int,
    timestamp_ms: int,
    face_detected: bool = True,
    head_position_mm: np.ndarray | None = None,
) -> FaceObservation:
    return FaceObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        left_iris_relative=(0.0, 0.0),
        right_iris_relative=(0.0, 0.0),
        head_yaw_deg=0.0,
        head_pitch_deg=0.0,
        head_roll_deg=0.0,
        eye_tracking_confidence=1.0,
        face_tracking_confidence=1.0,
        face_detected=face_detected,
        head_position_mm=head_position_mm,
    )


def _assert_valid_target_estimate(estimate: TargetEstimate, expected_frame_id: int) -> None:
    assert isinstance(estimate, TargetEstimate)
    field_names = {f.name for f in dataclasses.fields(TargetEstimate)}
    assert field_names == {
        "timestamp_ms",
        "frame_id",
        "target",
        "probability",
        "second_best_probability",
        "stability",
    }
    assert isinstance(estimate.timestamp_ms, int)
    assert isinstance(estimate.frame_id, int)
    assert estimate.frame_id == expected_frame_id
    assert isinstance(estimate.target, str)
    assert 0.0 <= estimate.probability <= 1.0
    assert 0.0 <= estimate.second_best_probability <= 1.0
    assert 0.0 <= estimate.stability <= 1.0
    assert estimate.probability >= estimate.second_best_probability


def test_estimate_is_valid_with_registered_devices() -> None:
    engine = GazeTargetingEngine(GazeConfig(unknown_probability_threshold=0.5))
    engine.register_device(DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05))

    for i in range(5):
        estimate = engine.process(_observation(i, i * 30))
    _assert_valid_target_estimate(estimate, expected_frame_id=4)


def test_estimate_is_valid_with_no_devices_registered() -> None:
    engine = GazeTargetingEngine()
    estimate = engine.process(_observation(0, 1_000))
    _assert_valid_target_estimate(estimate, expected_frame_id=0)
    assert estimate.target == GazeConfig().UNKNOWN_TARGET


def test_estimate_is_valid_on_tracking_loss() -> None:
    engine = GazeTargetingEngine()
    estimate = engine.process(_observation(0, 1_000, face_detected=False))
    _assert_valid_target_estimate(estimate, expected_frame_id=0)
    assert estimate.stability == 0.0


def test_estimate_is_valid_with_3d_registered_device() -> None:
    """3D geometry로 등록된 기기가 있어도 TargetEstimate 계약은 그대로다
    (documents/interface-contract.md 1번 — 모양이 바뀌면 이 테스트와 계약 문서를
    같은 변경 단위에서 갱신한다)."""
    engine = GazeTargetingEngine(GazeConfig(unknown_probability_threshold=0.5))
    engine.register_device(
        DeviceGazeProfile("laptop", np.array([0.0, 0.0, 1.0]), variance=0.05),
        geometry_3d=TargetGeometry3D(np.array([0.0, 0.0, 500.0]), radius_mm=50.0),
    )

    for i in range(5):
        estimate = engine.process(
            _observation(i, i * 30, head_position_mm=np.array([0.0, 0.0, 0.0]))
        )
    _assert_valid_target_estimate(estimate, expected_frame_id=4)
    assert estimate.target == "laptop"
