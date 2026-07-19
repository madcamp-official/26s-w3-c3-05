"""멀티태스크 loss(training/losses.py)를 검증한다. torch(`ml`/`training` extra) 필요."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip 뒤에 와야 함)

from training.dataset import IGNORE_INDEX  # noqa: E402
from training.losses import GesturePhaseLoss, compute_class_weights  # noqa: E402


def test_compute_class_weights_is_inverse_frequency() -> None:
    # 클래스 0이 2번, 클래스 1이 1번 등장 → 0의 가중치가 1보다 작아야 한다.
    weights = compute_class_weights([0, 0, 1], num_classes=3)
    assert weights[0] < weights[1]
    assert weights[2] == 0.0  # 등장하지 않은 클래스는 가중치 0


def test_compute_class_weights_mean_is_normalized() -> None:
    weights = compute_class_weights([0, 0, 1, 2, 2, 2], num_classes=3)
    assert torch.mean(weights).item() == pytest.approx(1.0, abs=1e-5)


def test_compute_class_weights_all_absent_is_all_zero() -> None:
    weights = compute_class_weights([], num_classes=4)
    assert torch.all(weights == 0.0)


def test_gesture_phase_loss_ignores_padded_frames() -> None:
    num_classes = 3
    weights = torch.ones(num_classes)
    loss_fn = GesturePhaseLoss(weights, phase_loss_weight=0.3)

    batch, time = 2, 5
    gesture_logits = torch.randn(batch, num_classes, time)
    phase_logits = torch.randn(batch, 4, time)
    gesture_targets = torch.zeros(batch, time, dtype=torch.long)
    phase_targets = torch.zeros(batch, time, dtype=torch.long)
    # 마지막 두 프레임을 패딩으로 마킹
    gesture_targets[:, -2:] = IGNORE_INDEX
    phase_targets[:, -2:] = IGNORE_INDEX

    total, gesture_loss, phase_loss = loss_fn(gesture_logits, phase_logits, gesture_targets, phase_targets)
    assert torch.isfinite(total)
    assert total.item() == pytest.approx((gesture_loss + 0.3 * phase_loss).item())


def test_gesture_phase_loss_weight_scales_phase_contribution() -> None:
    num_classes = 3
    weights = torch.ones(num_classes)
    torch.manual_seed(0)
    gesture_logits = torch.randn(2, num_classes, 4)
    phase_logits = torch.randn(2, 4, 4)
    gesture_targets = torch.randint(0, num_classes, (2, 4))
    phase_targets = torch.randint(0, 4, (2, 4))

    low_weight = GesturePhaseLoss(weights, phase_loss_weight=0.0)
    high_weight = GesturePhaseLoss(weights, phase_loss_weight=1.0)

    total_low, gesture_loss_low, _ = low_weight(gesture_logits, phase_logits, gesture_targets, phase_targets)
    total_high, gesture_loss_high, phase_loss_high = high_weight(
        gesture_logits, phase_logits, gesture_targets, phase_targets
    )

    assert total_low.item() == pytest.approx(gesture_loss_low.item())
    assert total_high.item() == pytest.approx((gesture_loss_high + phase_loss_high).item())


def test_gesture_phase_loss_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        GesturePhaseLoss(torch.ones(3), phase_loss_weight=-0.1)
