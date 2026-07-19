"""참고 레포 방법론 이식(reference_port) 검증.

핵심은 (1) max-abs 정규화가 참고 레포 규약(손목 원점·max-abs)을 따르는지,
(2) 추출한 학습 가중치 + numpy forward가 참고 레포 학습 데이터를 높은 정확도로
재현하는지다 — 후자가 가중치 추출·forward 재구현의 정확성을 보장한다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from jarvis.reference_port import (
    ReferenceKeyPointClassifier,
    preprocess_landmark_max_abs,
)

# 참고 레포 학습 데이터(있으면 재현 정확도 검증에 사용). 저장소 밖 경로라 없으면 skip.
_REF_KEYPOINT_CSV = (
    Path.home()
    / "Projects/hand-gesture-recognition-using-mediapipe/model/keypoint_classifier/keypoint.csv"
)


def test_preprocess_wrist_becomes_origin() -> None:
    pts = np.random.default_rng(0).random((21, 2)) + 0.5
    feat = preprocess_landmark_max_abs(pts)
    # 손목(첫 점) 상대좌표라 flatten의 앞 두 성분(손목 x,y)은 0이다.
    assert feat[0] == pytest.approx(0.0)
    assert feat[1] == pytest.approx(0.0)


def test_preprocess_is_scale_invariant() -> None:
    pts = np.random.default_rng(1).random((21, 2))
    a = preprocess_landmark_max_abs(pts)
    b = preprocess_landmark_max_abs(pts * 5.0)  # 스케일만 다른 같은 손모양
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_preprocess_range_is_within_unit() -> None:
    pts = np.random.default_rng(2).random((21, 2)) * 100
    feat = preprocess_landmark_max_abs(pts)
    assert np.max(np.abs(feat)) == pytest.approx(1.0)  # max-abs 정규화 → 최댓값 1


def test_degenerate_all_same_point_is_zero_vector() -> None:
    pts = np.full((21, 2), 0.3)
    feat = preprocess_landmark_max_abs(pts)
    assert np.all(feat == 0.0)  # 0 나눗셈 대신 0벡터


def test_classifier_outputs_valid_probability_distribution() -> None:
    clf = ReferenceKeyPointClassifier()
    pred = clf.classify_landmarks(np.random.default_rng(3).random((21, 2)))
    assert pred.label in clf.labels
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.probabilities.shape == (len(clf.labels),)
    assert np.sum(pred.probabilities) == pytest.approx(1.0)
    assert pred.probabilities[pred.class_id] == pytest.approx(pred.confidence)


@pytest.mark.skipif(not _REF_KEYPOINT_CSV.is_file(), reason="참고 레포 keypoint.csv 없음")
def test_reproduces_reference_training_accuracy() -> None:
    """추출 가중치 + numpy forward가 참고 학습 데이터를 높은 정확도로 재현한다.

    잘못 추출/재구현했다면 학습 데이터에서조차 정확도가 무너진다. 3-class 모델이라
    참고 데이터의 클래스 0~2만 평가한다(클래스 3은 모델이 표현 불가).
    """
    clf = ReferenceKeyPointClassifier()
    rows = list(csv.reader(_REF_KEYPOINT_CSV.open()))
    correct = total = 0
    for r in rows:
        y = int(r[0])
        if y > 2:
            continue
        feat = np.array(r[1:], dtype=np.float64)
        if feat.shape != (42,):
            continue
        if clf.classify_vector(feat).class_id == y:
            correct += 1
        total += 1
    assert total > 1000
    assert correct / total > 0.90  # 실측 0.965
