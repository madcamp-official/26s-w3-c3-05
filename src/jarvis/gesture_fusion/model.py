"""Causal TCN gesture·phase 분류기 — `GestureModel` Protocol의 torch 구현.

README 8장 처리 과정의 마지막 단계를 구현한다:

    속도·관절 각도 생성 → Causal TCN/GRU → Gesture·Phase 출력

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

from typing import cast

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.model_protocol import (
    PHASE_LABELS,
    ModelConfig,
    ModelMetadata,
    ModelPrediction,
    collapse_background_probabilities,
    normalized_entropy,
)

# ModelConfig는 torch 무의존이라 model_protocol(torch-free 경계)에 산다. 기존
# `from jarvis.gesture_fusion.model import ModelConfig` 경로가 깨지지 않도록 여기서
# 다시 노출한다.
__all__ = ["CausalTCN", "CausalTCNGestureModel", "ModelConfig"]

FloatArray = npt.NDArray[np.float64]

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only hit without the `ml` extra
    raise ImportError(
        "torch is required for jarvis.gesture_fusion.model; install with "
        "`pip install -e '.[ml]'`"
    ) from exc


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
        # 입력 표준화 통계(학습셋에서 1회 산출해 `set_input_normalization`으로 주입).
        # BatchNorm을 쓰지 않는 이유: BatchNorm은 train 모드에서 배치·**시간축 전체**
        # 통계로 정규화해 t 시점 출력에 t 이후 프레임 정보가 섞인다(causal 위반 —
        # eval 모드만 보는 회귀 테스트로는 안 잡힌다). 고정 통계 affine은 프레임마다
        # 독립이라 학습·추론·스트리밍 전부에서 엄격히 causal하다.
        # 기본값(mean=0, std=1)은 항등변환이라 통계를 주입하지 않으면 기존 동작과 같다.
        # buffer라 state_dict에 함께 저장돼 체크포인트와 통계가 분리되지 않는다.
        self.register_buffer("input_mean", torch.zeros(config.feature_dim))
        self.register_buffer("input_std", torch.ones(config.feature_dim))
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

    def set_input_normalization(self, mean: "torch.Tensor", std: "torch.Tensor") -> None:
        """학습셋에서 구한 차원별 평균·표준편차를 주입한다(feature 그룹 간 스케일 격차 보정).

        std가 0에 가까운(사실상 상수인) 차원은 1.0으로 둔다 — 0으로 나눠 inf/NaN을
        만들지 않는다(development-principles.md 7.2: 비정상값은 안전값으로).
        """
        if mean.shape != (self.config.feature_dim,) or std.shape != (self.config.feature_dim,):
            raise ValueError(f"mean·std must both have shape ({self.config.feature_dim},)")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("mean·std must contain only finite values")
        safe_std = torch.where(std > 1e-6, std, torch.ones_like(std))
        self.input_mean.copy_(mean.to(self.input_mean.dtype))
        self.input_std.copy_(safe_std.to(self.input_std.dtype))

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """x: (batch, feature_dim, time) → (gesture_logits, phase_logits), 각 (batch, classes, time)."""
        # (feature_dim,) → (1, feature_dim, 1)로 broadcast: 시간축과 무관한 프레임별 affine.
        x = (x - self.input_mean.view(1, -1, 1)) / self.input_std.view(1, -1, 1)
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
        """학습된 가중치를 불러온다. 불러온 뒤에만 `metadata.trained=True`를 인정한다.

        `weights_only=True`로 로드해 pickle 임의 코드 실행 경로를 차단한다 — torch<2.6은
        이 기본값이 False라 명시한다(state_dict만 담긴 신뢰 파일이라도 방어적으로).
        """
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
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

        # 배경 클래스들(가만히 있는 손·손가락 두드리기·아무 동작)은 각각 다른 클래스로
        # 학습하지만 런타임 결정에서는 그 구분이 의미가 없다. 전체 argmax 대신 배경
        # 확률을 합산해 비교한다 — 이유는 `collapse_background_probabilities` 참고.
        gesture_index, gesture_confidence, collapsed_probs = collapse_background_probabilities(
            gesture_probs, self._config.background_indices, self._config.foreground_indices
        )
        phase_index = int(np.argmax(phase_probs))

        return ModelPrediction(
            gesture=self._config.gesture_labels[gesture_index],
            gesture_confidence=gesture_confidence,
            phase=PHASE_LABELS[phase_index],
            phase_confidence=float(phase_probs[phase_index]),
            uncertainty=normalized_entropy(collapsed_probs),
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
