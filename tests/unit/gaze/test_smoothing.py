"""README 7장 "Temporal smoothing"과 stability(안정성) 계산을 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import GazeVector
from jarvis.gaze.smoothing import GazeSmoother


def _vector(direction: np.ndarray, confidence: float = 1.0, frame_id: int = 0) -> GazeVector:
    unit = direction / np.linalg.norm(direction)
    return GazeVector(
        direction=unit, confidence=confidence, timestamp_ms=1_000 + frame_id, frame_id=frame_id
    )


def test_constant_input_yields_stability_near_one() -> None:
    smoother = GazeSmoother()
    direction = np.array([0.0, 0.0, 1.0])
    result = None
    for i in range(10):
        result = smoother.update(_vector(direction, frame_id=i))
    assert result is not None
    assert result.stability == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(result.direction, direction, atol=1e-9)


def test_noisy_input_lowers_stability_relative_to_constant() -> None:
    smoother = GazeSmoother()
    a = np.array([1.0, 0.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    result = None
    for i in range(10):
        result = smoother.update(_vector(a if i % 2 == 0 else b, frame_id=i))
    assert result is not None
    assert result.stability < 0.9


def test_tracking_loss_resets_buffer() -> None:
    smoother = GazeSmoother()
    direction = np.array([0.0, 0.0, 1.0])
    for i in range(5):
        smoother.update(_vector(direction, frame_id=i))

    assert smoother.update(None) is None

    # 리셋 직후 첫 프레임은 버퍼에 하나뿐이므로 완벽히 안정된 것으로 취급된다.
    result = smoother.update(_vector(np.array([1.0, 0.0, 0.0]), frame_id=99))
    assert result is not None
    assert result.stability == pytest.approx(1.0)


def test_window_only_retains_recent_frames() -> None:
    config = GazeConfig(smoothing_window_frames=3)
    smoother = GazeSmoother(config)
    stale = np.array([-1.0, 0.0, 0.0])
    fresh = np.array([0.0, 0.0, 1.0])

    smoother.update(_vector(stale, frame_id=0))
    smoother.update(_vector(stale, frame_id=1))
    result = None
    for i in range(2, 6):
        result = smoother.update(_vector(fresh, frame_id=i))
    assert result is not None
    np.testing.assert_allclose(result.direction, fresh, atol=1e-6)
    assert result.stability == pytest.approx(1.0, abs=1e-6)


def test_zero_confidence_sample_yields_no_result() -> None:
    smoother = GazeSmoother()
    zero_confidence_vector = GazeVector(
        direction=np.array([0.0, 0.0, 1.0]), confidence=0.0, timestamp_ms=1_000, frame_id=0
    )
    assert smoother.update(zero_confidence_vector) is None
