"""제스처·phase 멀티태스크 loss (학습 파이프라인 인터뷰 결정).

- gesture loss: 클래스 빈도 역수로 가중한 cross-entropy — `none`이 절대다수라
  가중치 없이 학습하면 드문 제스처 클래스가 gradient에 거의 기여하지 못한다.
- phase loss: 동등 가중이 아니라 `phase_loss_weight`(기본 0.3)만큼만 반영한다 —
  phase 라벨이 클립 내 상대 위치로 근사한 휴리스틱(`training/phase_labels.py`)이라
  실제 프레임 경계보다 노이즈가 있어, gesture 표현 학습에 나쁜 영향을 주지 않도록
  낮은 가중치로 보조 신호처럼 쓴다.
- 둘 다 `ignore_index=training.dataset.IGNORE_INDEX`로 배치 패딩 프레임을 제외한다.
"""

from __future__ import annotations

from collections import Counter

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only hit without the `ml`/`training` extra
    raise ImportError(
        "torch is required for training.losses; install with `pip install -e '.[training]'`"
    ) from exc

from training.dataset import IGNORE_INDEX


def compute_class_weights(gesture_indices: list[int], num_classes: int) -> "torch.Tensor":
    """클래스 빈도 역수로 cross-entropy 가중치를 만든다.

    이 학습셋에 아예 등장하지 않는 클래스(빈도 0)는 가중치 0으로 둔다 — 등장하지
    않는 클래스에 무한대 가중치를 지어내지 않는다. 가중치 합의 평균이 1이 되도록
    정규화해 학습률 스케일과 독립적으로 만든다.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    counts = Counter(gesture_indices)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        weights[cls] = 0.0 if count == 0 else 1.0 / count
    total = float(weights.sum())
    if total > 0.0:
        weights = weights * (num_classes / total)
    return weights


class GesturePhaseLoss(nn.Module):
    """`CausalTCN.forward`가 낸 (gesture_logits, phase_logits)에 대한 결합 loss.

    두 logits 텐서 모두 `(batch, classes, time)` 축 순서를 그대로 받는다 —
    `nn.CrossEntropyLoss`는 이 "K-차원" 형태를 추가 reshape 없이 지원한다
    (target은 대응하는 `(batch, time)`).
    """

    def __init__(self, gesture_class_weights: "torch.Tensor", phase_loss_weight: float = 0.3) -> None:
        super().__init__()
        if phase_loss_weight < 0.0:
            raise ValueError("phase_loss_weight must be non-negative")
        self._gesture_loss = nn.CrossEntropyLoss(weight=gesture_class_weights, ignore_index=IGNORE_INDEX)
        self._phase_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._phase_loss_weight = phase_loss_weight

    def forward(
        self,
        gesture_logits: "torch.Tensor",
        phase_logits: "torch.Tensor",
        gesture_targets: "torch.Tensor",
        phase_targets: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """`(총 loss, gesture loss, phase loss)`를 반환한다 — 셋 다 TensorBoard에 기록한다."""
        gesture_loss = self._gesture_loss(gesture_logits, gesture_targets)
        phase_loss = self._phase_loss(phase_logits, phase_targets)
        total = gesture_loss + self._phase_loss_weight * phase_loss
        return total, gesture_loss, phase_loss
