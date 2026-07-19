"""GestureModel 경계의 torch-무의존 부분(엔트로피·스트리밍 윈도우)을 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.model_protocol import (
    ModelPrediction,
    SlidingFeatureWindow,
    normalized_entropy,
)


def test_confident_prediction_has_low_uncertainty() -> None:
    probs = np.array([0.97, 0.01, 0.01, 0.01])
    assert normalized_entropy(probs) < 0.2


def test_uniform_prediction_has_max_uncertainty() -> None:
    probs = np.full(4, 0.25)
    assert normalized_entropy(probs) == pytest.approx(1.0, abs=1e-6)


def test_single_class_has_zero_uncertainty() -> None:
    assert normalized_entropy(np.array([1.0])) == 0.0


def test_model_prediction_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="gesture_confidence"):
        ModelPrediction(
            gesture="swipe_down",
            gesture_confidence=1.5,
            phase="ENDING",  # type: ignore[arg-type]
            phase_confidence=0.8,
            uncertainty=0.1,
        )


def test_sliding_window_pads_before_full() -> None:
    window = SlidingFeatureWindow(window_size=3, feature_dim=2)
    snapshot = window.push(np.array([1.0, 2.0]))
    assert snapshot.shape == (1, 2)


def test_sliding_window_drops_oldest_when_full() -> None:
    window = SlidingFeatureWindow(window_size=2, feature_dim=1)
    window.push(np.array([1.0]))
    window.push(np.array([2.0]))
    snapshot = window.push(np.array([3.0]))
    np.testing.assert_array_equal(snapshot, [[2.0], [3.0]])


def test_sliding_window_reset_on_none() -> None:
    window = SlidingFeatureWindow(window_size=2, feature_dim=1)
    window.push(np.array([1.0]))
    snapshot = window.push(None)
    np.testing.assert_array_equal(snapshot, np.zeros((2, 1)))
    # 리셋 후 다음 push는 새 시퀀스로 시작한다(과거 프레임이 섞이지 않음).
    next_snapshot = window.push(np.array([9.0]))
    assert next_snapshot.shape == (1, 1)
    np.testing.assert_array_equal(next_snapshot, [[9.0]])


def test_sliding_window_rejects_wrong_shape() -> None:
    window = SlidingFeatureWindow(window_size=2, feature_dim=3)
    with pytest.raises(ValueError, match="shape"):
        window.push(np.array([1.0, 2.0]))
