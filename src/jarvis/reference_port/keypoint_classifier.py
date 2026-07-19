"""참고 레포 KeyPoint 분류기 이식 — max-abs 정규화 + numpy MLP forward.

참고 레포 `app.py::pre_process_landmark`와 `model/keypoint_classifier`를 그대로
옮긴다. 학습 가중치는 참고 레포의 hdf5에서 추출한 `keypoint_weights.npz`를 쓰고,
추론(Dropout은 항등)은 numpy로 재구현했다 — TensorFlow/LiteRT 없이 돈다.

좌표 규약: 입력은 참고 레포와 동일하게 **이미지 좌표(픽셀 또는 [0,1] 어느 쪽이든
무방)** 의 (21, 2) 배열이다. max-abs 정규화가 평행이동·스케일을 제거하므로 절대
단위는 상관없다. 이 프로젝트의 정규화(손목 원점·palm_scale)와는 **다른** 방식이라,
이식 비교의 핵심 변수다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_WEIGHTS_PATH = Path(__file__).with_name("keypoint_weights.npz")
_LABELS_PATH = Path(__file__).with_name("keypoint_labels.csv")


def preprocess_landmark_max_abs(points_xy: npt.ArrayLike) -> FloatArray:
    """참고 레포 `pre_process_landmark` 이식: 손목 상대좌표 → flatten → max-abs 정규화.

    1) 손목(index 0)을 원점으로 빼 상대좌표로 만든다.
    2) (21, 2)를 42차원으로 flatten한다.
    3) 전체 성분의 **최대 절댓값**으로 나눠 [-1, 1]로 스케일한다(손 크기 정규화).

    이 프로젝트의 palm_scale 정규화와 달리 특정 두 점 거리가 아니라 전체 bounding
    범위로 정규화한다. 손목과 나머지가 모두 겹쳐 max=0이면 0벡터를 돌려준다(0 나눗셈
    회피 — 참고 레포에는 없는 방어이나 퇴화 입력에서 NaN을 막는다).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape != (21, 2):
        raise ValueError(f"points_xy must have shape (21, 2), got {pts.shape}")
    relative = pts - pts[0]
    flat = relative.reshape(-1)
    max_abs = float(np.max(np.abs(flat)))
    if max_abs == 0.0:
        return np.zeros_like(flat)
    return cast("FloatArray", flat / max_abs)


def _load_labels() -> tuple[str, ...]:
    text = _LABELS_PATH.read_text(encoding="utf-8-sig")
    return tuple(line.strip() for line in text.splitlines() if line.strip())


@dataclass(frozen=True, slots=True)
class KeyPointPrediction:
    """한 프레임의 정적 손모양 분류 결과."""

    label: str
    class_id: int
    confidence: float  # softmax 최대값
    probabilities: FloatArray  # 클래스별 softmax 확률


class ReferenceKeyPointClassifier:
    """참고 레포의 학습된 KeyPoint MLP(42→20→10→3)를 numpy로 재현한 분류기.

    Dropout은 추론 시 항등이라 무시한다. 가중치는 `keypoint_weights.npz`(참고 레포
    hdf5에서 추출)에서 로드하며, forward는 relu 2층 + softmax다.
    """

    def __init__(self, weights_path: Path = _WEIGHTS_PATH) -> None:
        w = np.load(weights_path)
        # Keras Dense: y = x @ kernel + bias. kernel shape (in, out).
        self._W0 = w["W0"].astype(np.float64)  # (42, 20)
        self._b0 = w["b0"].astype(np.float64)
        self._W1 = w["W1"].astype(np.float64)  # (20, 10)
        self._b1 = w["b1"].astype(np.float64)
        self._W2 = w["W2"].astype(np.float64)  # (10, 3)
        self._b2 = w["b2"].astype(np.float64)
        self._labels = _load_labels()

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify_vector(self, feature_42: npt.ArrayLike) -> KeyPointPrediction:
        """42차원 max-abs 정규화 feature를 분류한다."""
        x = np.asarray(feature_42, dtype=np.float64)
        if x.shape != (42,):
            raise ValueError(f"feature must have shape (42,), got {x.shape}")
        h0 = np.maximum(0.0, x @ self._W0 + self._b0)
        h1 = np.maximum(0.0, h0 @ self._W1 + self._b1)
        logits = h1 @ self._W2 + self._b2
        probs = _softmax(logits)
        class_id = int(np.argmax(probs))
        label = self._labels[class_id] if class_id < len(self._labels) else str(class_id)
        return KeyPointPrediction(
            label=label,
            class_id=class_id,
            confidence=float(probs[class_id]),
            probabilities=probs,
        )

    def classify_landmarks(self, points_xy: npt.ArrayLike) -> KeyPointPrediction:
        """이미지 좌표 (21, 2) 랜드마크를 참고 레포 방식으로 정규화 후 분류한다."""
        return self.classify_vector(preprocess_landmark_max_abs(points_xy))


def _softmax(x: FloatArray) -> FloatArray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return cast("FloatArray", exp / np.sum(exp))
