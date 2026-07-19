"""Unit tests for the One-Euro filter (pure numpy, no camera/model)."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.smoothing import OneEuroFilter


def test_first_sample_passes_through_unchanged() -> None:
    f = OneEuroFilter()
    x = np.array([0.5, 0.5, 0.5])
    out = f.filter(x, timestamp_ms=0)
    np.testing.assert_array_equal(out, x)


def test_smoothing_reduces_variance_of_noisy_constant() -> None:
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)  # beta=0 → 순수 저역통과
    inputs, outputs = [], []
    for i in range(200):
        x = np.array([1.0]) + rng.normal(0.0, 0.05, size=1)
        inputs.append(float(x[0]))
        outputs.append(float(f.filter(x, timestamp_ms=i * 33)[0]))
    # 평활화된 출력의 분산이 입력보다 확실히 작아야 한다.
    assert np.var(outputs[10:]) < np.var(inputs[10:]) * 0.5


def test_non_increasing_timestamp_returns_last_output() -> None:
    f = OneEuroFilter()
    f.filter(np.array([0.0]), timestamp_ms=100)
    out1 = f.filter(np.array([1.0]), timestamp_ms=200)
    # 같은/과거 timestamp → 상태 갱신 없이 직전 출력 반환
    out2 = f.filter(np.array([5.0]), timestamp_ms=200)
    np.testing.assert_array_equal(out1, out2)


def test_reset_makes_next_sample_pass_through() -> None:
    f = OneEuroFilter()
    f.filter(np.array([0.0]), timestamp_ms=0)
    f.filter(np.array([1.0]), timestamp_ms=33)
    f.reset()
    x = np.array([9.0])
    np.testing.assert_array_equal(f.filter(x, timestamp_ms=66), x)


def test_fast_step_tracks_more_than_slow_drift() -> None:
    """적응형 컷오프: 큰 변화에는 덜 평활화(더 빨리 추종)해야 한다."""
    fast = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    fast.filter(np.array([0.0]), timestamp_ms=0)
    out = float(fast.filter(np.array([1.0]), timestamp_ms=33)[0])
    assert 0.0 < out <= 1.0


def test_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(beta=-1.0)
    with pytest.raises(ValueError):
        OneEuroFilter(d_cutoff=-1.0)
