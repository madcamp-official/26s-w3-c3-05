"""프레임 단위 평가지표 — macro-F1을 모델 선택·early stopping 기준으로 쓴다
(학습 파이프라인 인터뷰 결정: `none` 클래스가 절대다수라 단순 accuracy는
"항상 none"만 찍어도 높게 나오는 함정이 있음).

torch에 의존하지 않는 순수 numpy 구현이다 — `train.py`가 예측·정답 텐서를
numpy로 변환해 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    """평가 결과. `per_class_f1`은 평가 셋에 실제로 등장한 클래스만 담는다."""

    macro_f1: float
    per_class_f1: dict[int, float]
    confusion: IntArray  # (num_classes, num_classes), 행=정답, 열=예측


def collapse_class_indices(
    indices: IntArray,
    background_indices: tuple[int, ...],
    foreground_indices: tuple[int, ...],
    ignore_index: int = -100,
) -> IntArray:
    """원본 클래스 index를 "배경 1개 + 제스처 N개" 공간으로 접는다(0=배경, k+1=제스처k).

    배경을 여러 클래스로 나눠 학습하므로(`DEFAULT_BACKGROUND_LABELS`), 원본
    클래스 수 그대로 macro-F1을 재면 **서로 구분할 필요가 없는 배경 클래스들을 얼마나
    잘 가르는지**까지 평균에 섞인다. 그 점수로 early stopping을 하면 우리가 원하지
    않는 능력을 최적화한 epoch이 선택된다 — 그래서 선택 지표는 접은 공간에서 잰다.

    `ignore_index`(배치 패딩)는 그대로 보존한다.
    """
    if ignore_index in background_indices or ignore_index in foreground_indices:
        raise ValueError("ignore_index must not be a valid class index")

    lookup = {index: 0 for index in background_indices}
    for position, index in enumerate(foreground_indices):
        lookup[index] = position + 1

    collapsed = np.full_like(indices, ignore_index)
    for original, target in lookup.items():
        collapsed[indices == original] = target
    return collapsed


def compute_classification_report(
    predictions: IntArray,
    targets: IntArray,
    num_classes: int,
    ignore_index: int = -100,
) -> ClassificationReport:
    """`predictions`/`targets`는 같은 shape의 정수 배열(flatten해도 무방).

    `ignore_index`인 위치(배치 패딩 프레임)는 평가에서 제외한다.
    """
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    preds = predictions[valid]
    trues = targets[valid]
    if trues.size == 0:
        raise ValueError("no valid (non-ignored) frames to evaluate")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(trues.tolist(), preds.tolist()):
        confusion[t, p] += 1

    per_class_f1: dict[int, float] = {}
    for cls in range(num_classes):
        support = int(confusion[cls, :].sum())
        if support == 0:
            # 이 평가 셋에 아예 등장하지 않은 클래스는 F1이 정의되지 않으므로
            # macro 평균에서 제외한다(0으로 지어내지 않음).
            continue
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum()) - tp
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        per_class_f1[cls] = float(f1)

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    return ClassificationReport(macro_f1=macro_f1, per_class_f1=per_class_f1, confusion=confusion)
