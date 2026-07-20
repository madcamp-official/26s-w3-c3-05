"""프레임 단위 분류 리포트(training/metrics.py)를 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from training.metrics import collapse_class_indices, compute_classification_report


def test_perfect_predictions_give_macro_f1_of_one() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    predictions = targets.copy()
    report = compute_classification_report(predictions, targets, num_classes=3)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.confusion.sum() == 6
    assert np.trace(report.confusion) == 6


def test_all_wrong_gives_low_macro_f1() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    predictions = np.array([1, 1, 0, 0], dtype=np.int64)
    report = compute_classification_report(predictions, targets, num_classes=2)
    assert report.macro_f1 == pytest.approx(0.0)


def test_ignore_index_excludes_padded_frames() -> None:
    targets = np.array([0, 0, -100, -100], dtype=np.int64)
    predictions = np.array([0, 0, 5, 5], dtype=np.int64)  # 패딩 위치의 엉터리 예측
    report = compute_classification_report(predictions, targets, num_classes=6, ignore_index=-100)
    assert report.confusion.sum() == 2  # 패딩 프레임 2개는 제외됨
    assert report.macro_f1 == pytest.approx(1.0)


def test_class_absent_from_targets_excluded_from_macro_average() -> None:
    """평가 셋에 아예 등장하지 않는 클래스는 F1을 0으로 지어내지 않고 평균에서 뺀다."""
    targets = np.array([0, 0, 0], dtype=np.int64)
    predictions = np.array([0, 0, 0], dtype=np.int64)
    report = compute_classification_report(predictions, targets, num_classes=3)
    assert set(report.per_class_f1.keys()) == {0}
    assert report.macro_f1 == pytest.approx(1.0)


def test_rejects_empty_valid_frames() -> None:
    targets = np.array([-100, -100], dtype=np.int64)
    predictions = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        compute_classification_report(predictions, targets, num_classes=2, ignore_index=-100)


def test_accepts_multidimensional_arrays() -> None:
    """배치 (B, T) 형태를 flatten 없이 그대로 넘겨도 동작해야 한다."""
    targets = np.array([[0, 1], [1, 0]], dtype=np.int64)
    predictions = np.array([[0, 1], [1, 0]], dtype=np.int64)
    report = compute_classification_report(predictions, targets, num_classes=2)
    assert report.macro_f1 == pytest.approx(1.0)


# --- 배경 클래스 접기 (모델 선택 지표) ---


def test_collapse_maps_background_to_zero_and_renumbers_gestures() -> None:
    # 배경 = {0, 4}, 제스처 = {1, 2, 3} → 접은 공간에서 1, 2, 3
    indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    collapsed = collapse_class_indices(indices, (0, 4), (1, 2, 3))
    np.testing.assert_array_equal(collapsed, [0, 1, 2, 3, 0])


def test_collapse_preserves_ignore_index() -> None:
    """배치 패딩 프레임은 접기 대상이 아니다 — 평가에서 제외되어야 한다."""
    indices = np.array([0, -100, 2, -100], dtype=np.int64)
    collapsed = collapse_class_indices(indices, (0, 4), (1, 2, 3), ignore_index=-100)
    np.testing.assert_array_equal(collapsed, [0, -100, 2, -100])


def test_collapse_rejects_ignore_index_that_is_a_real_class() -> None:
    indices = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="ignore_index"):
        collapse_class_indices(indices, (0,), (1,), ignore_index=1)


def test_collapsed_macro_f1_ignores_confusion_among_background_classes() -> None:
    """배경끼리 서로 틀려도 접은 기준 점수는 만점이어야 한다 — 이 지표의 존재 이유다.

    원본 기준으로는 배경 0과 4를 뒤바꿔 맞혀 점수가 깎이지만, 우리는 그 구분에
    관심이 없다. 이 차이가 곧 잘못된 epoch을 고르게 만드는 원인이다.
    """
    targets = np.array([0, 4, 1, 2], dtype=np.int64)
    predictions = np.array([4, 0, 1, 2], dtype=np.int64)  # 배경끼리만 뒤바뀜

    raw = compute_classification_report(predictions, targets, num_classes=5)
    assert raw.macro_f1 < 1.0

    collapsed = compute_classification_report(
        collapse_class_indices(predictions, (0, 4), (1, 2, 3)),
        collapse_class_indices(targets, (0, 4), (1, 2, 3)),
        num_classes=4,
    )
    assert collapsed.macro_f1 == pytest.approx(1.0)
