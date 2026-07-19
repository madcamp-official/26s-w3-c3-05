"""프레임 단위 분류 리포트(training/metrics.py)를 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from training.metrics import compute_classification_report


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
