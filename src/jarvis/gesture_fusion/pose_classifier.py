"""정적 손 자세 분류기의 torch 구현 — `training/train_pose.py`가 만든 가중치를 읽는다.

`pose_protocol.py`(순수)와의 관계는 `model.py`↔`model_protocol.py`와 같다.

**전처리 대조가 이 모듈의 핵심 안전장치다.** 학습과 추론의 전처리가 어긋나면 예외 없이
정확도만 조용히 떨어진다 — 가장 찾기 어려운 종류의 고장이다. 그래서 저장 파일의
전처리 설정을 현재 `GestureConfig`와 대조하고, 다르면 로드를 거부한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import (
    DEFAULT_GESTURE_CONFIG,
    LANDMARK_DIMS,
    GestureConfig,
)
from jarvis.gesture_fusion.pose_protocol import PosePrediction, is_pose_trusted, pose_features

FloatArray = npt.NDArray[np.float64]

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only hit without the `ml` extra
    raise ImportError(
        "torch is required for jarvis.gesture_fusion.pose_classifier; install with "
        "`pip install -e '.[ml]'`"
    ) from exc


class PreprocessingMismatch(ValueError):
    """저장된 모델의 전처리와 현재 설정이 다르다 — 조용히 쓰면 정확도만 떨어진다."""


def _check_preprocessing(saved: dict[str, object], config: GestureConfig) -> None:
    expected = {
        "landmark_dims": LANDMARK_DIMS,
        "origin_index": config.origin_index,
        "palm_scale_root_index": config.palm_scale_root_index,
        "palm_scale_tip_index": config.palm_scale_tip_index,
        "smoothing_min_cutoff": config.smoothing_min_cutoff,
        "smoothing_beta": config.smoothing_beta,
        "smoothing_d_cutoff": config.smoothing_d_cutoff,
    }
    differences = [
        f"{key}: 모델 {saved.get(key)!r} != 현재 {value!r}"
        for key, value in expected.items()
        if saved.get(key) != value
    ]
    if differences:
        raise PreprocessingMismatch(
            "학습과 추론의 전처리가 다르다(모델을 다시 학습하거나 설정을 되돌려라):\n  "
            + "\n  ".join(differences)
        )


class TorchPoseClassifier:
    """학습된 MLP로 한 프레임의 자세를 판정한다(`PoseClassifier` Protocol 구현)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: GestureConfig = DEFAULT_GESTURE_CONFIG,
        tilt_limits: dict[str, float] | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"자세 분류 모델이 없다: {path}. "
                "`python -m training.train_pose --data <수집 npz>`로 먼저 학습하라."
            )
        blob = torch.load(path, weights_only=False, map_location="cpu")
        _check_preprocessing(blob.get("preprocessing", {}), config)

        self._labels: tuple[str, ...] = tuple(blob["label_names"])
        self._mean = np.asarray(blob["feature_mean"], dtype=np.float64)
        self._std = np.asarray(blob["feature_std"], dtype=np.float64)
        self._tilt_limits = tilt_limits
        self._min_confidence = min_confidence
        input_dim = int(blob["input_dim"])
        self._model = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, len(self._labels)),
        )
        self._model.load_state_dict(blob["state_dict"])
        self._model.eval()
        self.metrics: dict[str, object] = blob.get("metrics", {})

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify(
        self, landmarks: FloatArray, palm_tilt_degrees: float | None
    ) -> PosePrediction:
        """정규화·평활된 좌표 한 프레임을 판정한다(학습과 동일 전처리를 전제).

        판정을 거부할 때도 `label`·`confidence`는 채워 돌려준다 — 무엇이 왜 거부됐는지
        보여줄 수 있어야 사용자가 자세를 고칠 수 있다.
        """
        points = np.asarray(landmarks, dtype=np.float64)
        flat = (
            pose_features(points)
            if points.ndim == 2 and np.all(np.isfinite(points))
            else np.zeros(0)
        )
        if flat.shape != self._mean.shape or not np.all(np.isfinite(flat)):
            return PosePrediction(
                label="", confidence=0.0, trusted=False,
                reason="랜드마크가 없거나 형태가 맞지 않음",
                palm_tilt_degrees=palm_tilt_degrees,
            )
        x = torch.tensor((flat - self._mean) / self._std, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(x), dim=1).numpy()[0]
        index = int(probabilities.argmax())
        label, confidence = self._labels[index], float(probabilities[index])

        if confidence < self._min_confidence:
            return PosePrediction(
                label=label, confidence=confidence, trusted=False,
                reason=f"신뢰도 {confidence:.0%} < 임계 {self._min_confidence:.0%}",
                palm_tilt_degrees=palm_tilt_degrees,
            )
        trusted, reason = is_pose_trusted(label, palm_tilt_degrees, self._tilt_limits)
        return PosePrediction(
            label=label, confidence=confidence, trusted=trusted,
            reason=reason, palm_tilt_degrees=palm_tilt_degrees,
        )
