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


def test_compute_class_weights_accepts_count_mapping() -> None:
    """빈도를 직접 넘기는 형태 — 클립 수가 아니라 유효 프레임 수로 세기 위함.

    2026-07-20: cross-entropy가 프레임마다 걸리므로 빈도도 프레임 단위여야 한다
    (`ClipDataset.valid_frames_per_label` 참조). index 리스트를 세는 형태와 같은
    결과를 내야 한다.
    """
    from_list = compute_class_weights([0, 0, 1], num_classes=3)
    from_mapping = compute_class_weights({0: 2, 1: 1}, num_classes=3)
    assert torch.allclose(from_list, from_mapping)


def test_compute_class_weights_equalizes_loss_share_across_classes() -> None:
    """빈도 역수 가중치는 클래스별 총 loss 기여(가중치 x 빈도)를 같게 만든다.

    train.py가 이 성질에 기대어 "각 클래스 균등"을 얻으므로, 빈도를 어떤 단위로
    세느냐가 곧 어떤 단위가 균등해지느냐를 결정한다.
    """
    counts = {0: 100, 1: 400, 2: 25}
    weights = compute_class_weights(counts, num_classes=3)
    shares = [float(weights[cls]) * count for cls, count in counts.items()]
    assert shares[0] == pytest.approx(shares[1])
    assert shares[1] == pytest.approx(shares[2])


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
