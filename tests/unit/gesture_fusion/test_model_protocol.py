"""GestureModel 경계의 torch-무의존 부분(엔트로피·스트리밍 윈도우)을 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_BACKGROUND_LABELS,
    DEFAULT_GESTURE_LABELS,
    ModelPrediction,
    SlidingFeatureWindow,
    collapse_background_probabilities,
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


# --- 배경 클래스 합산 결정 규칙 ---
#
# 배경(가만히 있는 손·손가락 두드리기·아무 동작)은 **각각 다른 클래스로 학습**하고
# 합치는 것은 결정 단계에서만 한다. 아래 테스트가 그 규칙을 고정한다 — 특히 첫 번째가
# 이 설계의 존재 이유(표 분산)를 증명하는 정의적 테스트다.

_BG = (0, 1)  # 배경 2개
_FG = (2, 3)  # 제스처 2개


def test_summing_background_beats_a_gesture_that_would_win_naive_argmax() -> None:
    """이 설계의 정의적 테스트 — 합산하지 않으면 배경 구간에서 제스처가 이긴다.

    배경 확률 합계 0.60이 제스처 0.30을 압도하는데도, 배경이 두 클래스로 쪼개져
    각각 0.30/0.30이라 전체 argmax는 (동점 tie-break에 따라) 제스처를 고를 수 있다.
    """
    probs = np.array([0.30, 0.30, 0.30, 0.10])
    index, confidence, collapsed = collapse_background_probabilities(probs, _BG, _FG)

    assert index == _BG[0]  # 대표 배경
    assert confidence == pytest.approx(0.60)
    np.testing.assert_allclose(collapsed, [0.60, 0.30, 0.10])


def test_confident_gesture_still_wins() -> None:
    """합산이 제스처를 무조건 이기게 만드는 것은 아니다 — 과잉 억제 회귀 방지."""
    probs = np.array([0.10, 0.10, 0.75, 0.05])
    index, confidence, _ = collapse_background_probabilities(probs, _BG, _FG)
    assert index == 2
    assert confidence == pytest.approx(0.75)


def test_tie_goes_to_background() -> None:
    """동점이면 액션을 일으키지 않는다(알 수 없으면 거부)."""
    probs = np.array([0.25, 0.25, 0.50, 0.00])
    index, _, _ = collapse_background_probabilities(probs, _BG, _FG)
    assert index == _BG[0]


def test_collapsed_distribution_is_a_valid_distribution() -> None:
    probs = np.array([0.30, 0.30, 0.30, 0.10])
    _, _, collapsed = collapse_background_probabilities(probs, _BG, _FG)
    assert collapsed.shape == (1 + len(_FG),)
    assert float(collapsed.sum()) == pytest.approx(1.0)


def test_uncertainty_is_measured_on_collapsed_distribution() -> None:
    """어느 배경인지 모호한 것이 불확실성으로 새면 안 된다.

    배경 두 클래스에 정확히 반씩 쏠린 상황은 "배경임이 확실"한 것이지 모호한 게
    아니다. 원본 분포에서 엔트로피를 재면 이걸 높은 불확실성으로 잘못 읽는다.
    """
    probs = np.array([0.50, 0.50, 0.00, 0.00])
    _, _, collapsed = collapse_background_probabilities(probs, _BG, _FG)
    assert normalized_entropy(collapsed) < 0.05
    assert normalized_entropy(probs) > 0.4  # 원본 기준이면 모호해 보인다


def test_collapse_rejects_empty_index_groups() -> None:
    probs = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="non-empty"):
        collapse_background_probabilities(probs, (), (0, 1))
    with pytest.raises(ValueError, match="non-empty"):
        collapse_background_probabilities(probs, (0, 1), ())


def test_default_background_labels_are_real_labels() -> None:
    """배경 이름이 라벨 집합에 실제로 있어야 한다 — 개명 시 조용히 전경이 되는 것 방지."""
    assert DEFAULT_BACKGROUND_LABELS <= set(DEFAULT_GESTURE_LABELS)


def test_representative_background_is_label_index_zero() -> None:
    """spotting.py가 배경 판정에 `DEFAULT_GESTURE_LABELS[0]`을 쓰는 규약과 맞아야 한다."""
    assert DEFAULT_GESTURE_LABELS[0] in DEFAULT_BACKGROUND_LABELS


# --- FrameRateLimiter: 추론 feed를 학습 cadence로 솎기 ---


def test_frame_rate_limiter_accepts_first_frame() -> None:
    from jarvis.gesture_fusion.model_protocol import FrameRateLimiter

    limiter = FrameRateLimiter(target_fps=12.0)
    assert limiter.should_accept(0) is True


def test_frame_rate_limiter_skips_frames_closer_than_interval() -> None:
    """30fps(33ms) 스트림을 12fps(83ms)로 솎으면 일부 프레임이 skip돼야 한다."""
    from jarvis.gesture_fusion.model_protocol import FrameRateLimiter

    limiter = FrameRateLimiter(target_fps=12.0)
    accepted = [ts for ts in range(0, 1000, 33) if limiter.should_accept(ts)]
    # 1초 스트림 → 약 12프레임 채택(30fps 30프레임에서 솎임)
    assert 11 <= len(accepted) <= 13
    # 채택된 프레임 간격은 target(83ms) 이상
    assert all(b - a >= 83 for a, b in zip(accepted, accepted[1:], strict=False))


def test_frame_rate_limiter_reset_reaccepts() -> None:
    from jarvis.gesture_fusion.model_protocol import FrameRateLimiter

    limiter = FrameRateLimiter(target_fps=12.0)
    assert limiter.should_accept(0) is True
    assert limiter.should_accept(10) is False  # 10ms < 83ms
    limiter.reset()
    assert limiter.should_accept(20) is True  # reset 후 무조건 채택


def test_frame_rate_limiter_rejects_invalid_fps() -> None:
    from jarvis.gesture_fusion.model_protocol import FrameRateLimiter

    with pytest.raises(ValueError, match="target_fps"):
        FrameRateLimiter(target_fps=0.0)
    with pytest.raises(ValueError, match="target_fps"):
        FrameRateLimiter(target_fps=float("nan"))
