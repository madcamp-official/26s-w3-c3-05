"""engine_port 비교 지표 검증.

랜드마크 엔진 자체는 mediapipe가 있어야 돌아가므로 여기서는 다루지 않는다(엔진
어댑터는 `test_solutions_hands.py`). 대신 **엔진 없이도 깨질 수 있는 부분**을
덮는다: 편차·검출 일치율·지터 집계가 맞는가. 이게 틀리면 A/B 수치 전체가 무의미하다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.engine_port.metrics import (
    LANDMARK_COUNT,
    ComparisonAccumulator,
    JitterTracker,
    landmark_deviation,
)


def _points(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((LANDMARK_COUNT, 2))


def test_landmark_deviation_is_zero_for_identical_points() -> None:
    pts = _points(1)
    np.testing.assert_allclose(landmark_deviation(pts, pts), np.zeros(LANDMARK_COUNT), atol=1e-12)


def test_landmark_deviation_matches_known_offset() -> None:
    pts = _points(2)
    shifted = pts + np.array([0.03, 0.04])  # 각 점이 정확히 0.05만큼 이동
    np.testing.assert_allclose(landmark_deviation(pts, shifted), np.full(LANDMARK_COUNT, 0.05))


def test_landmark_deviation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="랜드마크"):
        landmark_deviation(np.zeros((2, 2)), np.zeros((2, 2)))


def test_accumulator_counts_detection_agreement() -> None:
    acc = ComparisonAccumulator()
    a, b = _points(3), _points(4)
    acc.update(a, b)        # 둘 다 검출
    acc.update(a, None)     # A만
    acc.update(None, b)     # B만
    acc.update(None, None)  # 둘 다 미검출

    summary = acc.summary()
    assert (summary.frames, summary.both_detected) == (4, 1)
    assert (summary.only_a, summary.only_b, summary.neither) == (1, 1, 1)
    assert summary.agreement_rate == pytest.approx(0.5)  # both + neither = 2/4


def test_accumulator_deviation_stats_use_known_offsets() -> None:
    acc = ComparisonAccumulator()
    base = _points(5)
    acc.update(base, base + np.array([0.03, 0.04]))  # 편차 0.05
    acc.update(base, base + np.array([0.06, 0.08]))  # 편차 0.10

    summary = acc.summary()
    assert summary.mean_deviation == pytest.approx(0.075)
    assert summary.max_deviation == pytest.approx(0.10)
    assert summary.per_landmark_mean is not None
    np.testing.assert_allclose(summary.per_landmark_mean, np.full(LANDMARK_COUNT, 0.075))


def test_accumulator_summary_is_safe_when_never_detected() -> None:
    acc = ComparisonAccumulator()
    acc.update(None, None)
    summary = acc.summary()
    assert summary.mean_deviation is None
    assert summary.per_landmark_mean is None
    # 수치가 없어도 리포트는 렌더링돼야 한다(n/a 표기).
    assert "n/a" in summary.format_report(label_a="A", label_b="B")


def test_jitter_tracker_ignores_first_frame_and_detection_gaps() -> None:
    tracker = JitterTracker()
    base = _points(6)
    assert tracker.update(base) is None  # 첫 프레임은 비교 대상 없음
    assert tracker.update(base + np.array([0.03, 0.04])) == pytest.approx(0.05)

    assert tracker.update(None) is None  # 추적 끊김
    # 끊긴 뒤 첫 프레임은 이전 손과 이어 붙이지 않는다(가짜 큰 이동량 방지).
    assert tracker.update(_points(7)) is None
    assert tracker.mean == pytest.approx(0.05)
