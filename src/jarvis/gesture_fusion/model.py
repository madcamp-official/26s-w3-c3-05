"""Causal TCN gesture·phase 분류기 — `GestureModel` Protocol의 torch 구현.

README 8장 처리 과정의 마지막 단계를 구현한다:

    속도·가속도·관절 각도 생성 → Causal TCN/GRU → Gesture·Phase 출력

`jarvis.gesture_fusion` 패키지에서 torch를 직접 import하는 유일한 모듈이다 —
`mediapipe_hands.py`와 같은 격리 원칙이다(pyproject.toml의 `ml` extra는 이 모듈에서만
필요하다). Protocol·입출력 타입은 `model_protocol.py`(torch 무의존)에 있으므로,
downstream(gesture spotting 등)은 이 모듈을 import하지 않고도 타입을 다룰 수 있다.

**주의(development-principles.md 1·7절)**: `CausalTCNGestureModel`은 기본적으로
무작위 초기화 가중치를 갖는다. 실제 학습 데이터로 학습한 가중치를 `load_weights`로
불러오기 전까지는 이 모델의 출력을 실제 제스처 인식 결과로 신뢰해서는 안 된다 —
프로덕션 경로에서 성공을 가장하지 않는다는 원칙에 따라, 학습되지 않은 모델임을
`ModelMetadata.trained`로 명시적으로 드러낸다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_GESTURE_LABELS,
    PHASE_LABELS,
    ModelMetadata,
    ModelPrediction,
    normalized_entropy,
)

FloatArray = npt.NDArray[np.float64]

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only hit without the `ml` extra
    raise ImportError(
        "torch is required for jarvis.gesture_fusion.model; install with "
        "`pip install -e '.[ml]'`"
    ) from exc


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Causal TCN 아키텍처 파라미터.

    `GestureConfig`(전처리 임계값)와 분리한다 — 이 값들은 모델 가중치의 shape을
    결정하므로, 저장된 가중치와 함께 버전 관리해야 하는 별개의 관심사다
    (development-principles.md 7.3).
    """

    feature_dim: int
    """입력 feature 벡터 차원. `features.feature_dimension(GestureConfig)`와 일치해야 한다."""

    gesture_labels: tuple[str, ...] = DEFAULT_GESTURE_LABELS
    """gesture 분류 head의 출력 클래스 순서. 열린 문자열 키(interface-contract.md)."""

    channels: tuple[int, ...] = (32, 32, 32)
    """각 temporal block의 채널 수. 층이 늘수록(dilation이 커질수록) 더 긴 과거를 본다."""

    kernel_size: int = 3
    """각 causal conv의 시간축 커널 크기."""

    dropout: float = 0.2
    """temporal block 내부 dropout 비율 (0=off, 학습 시에만 적용)."""

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if len(self.gesture_labels) < 2:
            raise ValueError("gesture_labels must contain at least two classes")
        if len(set(self.gesture_labels)) != len(self.gesture_labels):
            raise ValueError("gesture_labels must not contain duplicates")
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError("channels must be a non-empty tuple of positive ints")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2 for causal padding to be meaningful")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")

    @property
    def receptive_field(self) -> int:
        """이 아키텍처가 인과적으로 볼 수 있는 최대 과거 프레임 수(현재 프레임 포함).

        각 temporal block은 동일 dilation의 causal conv 두 개를 직렬로 쓰므로,
        block i(0-indexed, dilation=2**i)가 늘리는 시야는 `2 * (kernel_size-1) * 2**i`.
        스트리밍 추론에 필요한 최소 window 길이로 쓰인다.
        """
        span = 0
        for i in range(len(self.channels)):
            dilation = 2**i
            span += 2 * (self.kernel_size - 1) * dilation
        return span + 1


class _CausalConv1d(nn.Module):
    """왼쪽만 패딩하는 1D conv — 출력의 각 시점이 그 시점까지의 입력만 본다."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self._left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = nn.functional.pad(x, (self._left_pad, 0))
        return cast("torch.Tensor", self.conv(x))


class _TemporalBlock(nn.Module):
    """causal conv 두 개 + residual. WaveNet/TCN 표준 구조의 최소 형태."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        residual = x if self.downsample is None else self.downsample(x)
        return cast("torch.Tensor", self.relu(out + residual))


class CausalTCN(nn.Module):
    """전체 시퀀스에 대해 gesture·phase logits을 내는 raw 아키텍처(테스트·학습용).

    스트리밍 추론에는 `CausalTCNGestureModel`(마지막 시점만 반환·검증)을 쓴다.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_channels = config.feature_dim
        for i, out_channels in enumerate(config.channels):
            blocks.append(
                _TemporalBlock(
                    in_channels, out_channels, config.kernel_size, dilation=2**i, dropout=config.dropout
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.gesture_head = nn.Conv1d(in_channels, len(config.gesture_labels), 1)
        self.phase_head = nn.Conv1d(in_channels, len(PHASE_LABELS), 1)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """x: (batch, feature_dim, time) → (gesture_logits, phase_logits), 각 (batch, classes, time)."""
        hidden = self.blocks(x)
        return self.gesture_head(hidden), self.phase_head(hidden)


class CausalTCNGestureModel:
    """`GestureModel` Protocol 구현체 — window(과거~현재) → 마지막 시점 예측 하나.

    torch 추론(`torch.no_grad()`, `eval()`)만 노출해 학습 세부사항을 호출자에게서
    감춘다. 출력은 항상 `ModelPrediction`의 range 검증을 통과한 값이다(생성자가
    통과 못 하면 예외를 낸다 — NaN이나 범위 밖 값을 조용히 흘려보내지 않는다,
    development-principles.md 7.2).
    """

    def __init__(self, config: ModelConfig, metadata: ModelMetadata | None = None) -> None:
        self._config = config
        self._net = CausalTCN(config)
        self._net.eval()
        self.metadata = metadata or ModelMetadata()

    @property
    def labels(self) -> tuple[str, ...]:
        return self._config.gesture_labels

    @property
    def window_size(self) -> int:
        return self._config.receptive_field

    def load_weights(self, state_dict_path: str, metadata: ModelMetadata) -> None:
        """학습된 가중치를 불러온다. 불러온 뒤에만 `metadata.trained=True`를 인정한다."""
        state_dict = torch.load(state_dict_path, map_location="cpu")
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self.metadata = metadata

    def predict(self, window: FloatArray) -> ModelPrediction:
        """window: (window_size, feature_dim), 시간순(가장 오래된 것이 index 0).

        길이가 `window_size`보다 짧으면 앞쪽을 0으로 패딩한다(스트리밍 시작 직후
        history가 아직 안 쌓인 구간을 표현 — 진짜 과거를 지어내지 않고 0으로 명시).
        """
        if window.ndim != 2 or window.shape[1] != self._config.feature_dim:
            raise ValueError(
                f"window must have shape (T, {self._config.feature_dim}), got {window.shape}"
            )
        if not np.all(np.isfinite(window)):
            raise ValueError("window must contain only finite values")

        padded = self._pad_to_window(window)
        tensor = torch.as_tensor(padded, dtype=torch.float32).T.unsqueeze(0)  # (1, feature_dim, T)

        with torch.no_grad():
            gesture_logits, phase_logits = self._net(tensor)

        gesture_probs = torch.softmax(gesture_logits[0, :, -1], dim=0).numpy()
        phase_probs = torch.softmax(phase_logits[0, :, -1], dim=0).numpy()

        gesture_index = int(np.argmax(gesture_probs))
        phase_index = int(np.argmax(phase_probs))

        return ModelPrediction(
            gesture=self._config.gesture_labels[gesture_index],
            gesture_confidence=float(gesture_probs[gesture_index]),
            phase=PHASE_LABELS[phase_index],
            phase_confidence=float(phase_probs[phase_index]),
            uncertainty=normalized_entropy(gesture_probs),
        )

    def _pad_to_window(self, window: FloatArray) -> FloatArray:
        length = window.shape[0]
        target = self._config.receptive_field
        if length == target:
            return window
        if length > target:
            return window[-target:]
        pad = np.zeros((target - length, window.shape[1]), dtype=window.dtype)
        return np.concatenate([pad, window], axis=0)
