"""Causal TCN 구현(model.py)을 검증한다.

핵심 회귀 대상: (1) 진짜 인과성(미래 프레임이 과거 시점 출력에 영향을 주면 안 됨),
(2) window 패딩·트리밍, (3) 예측 출력이 항상 유효 범위, (4) 아키텍처 파라미터로
모델을 교체 가능한지(설계 목표).
"""

from __future__ import annotations

import numpy as np
import pytest

# model.py는 torch(`ml` extra)를 필요로 한다. 다른 optional extra(mediapipe) 테스트와
# 같이, extra 미설치 환경에서도 나머지 스위트 수집이 깨지지 않도록 여기서 건너뛴다.
pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip 뒤에 와야 함)

from jarvis.contracts.messages import GesturePhase  # noqa: E402
from jarvis.gesture_fusion.model import (  # noqa: E402
    CausalTCN,
    CausalTCNGestureModel,
    ModelConfig,
)


def _config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = dict(feature_dim=4, channels=(8, 8), kernel_size=3, gesture_labels=("none", "swipe_down"))
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def test_receptive_field_matches_two_causal_convs_per_block() -> None:
    config = _config(channels=(8, 8), kernel_size=3)
    # block0: dilation=1 → 2*(3-1)*1=4, block1: dilation=2 → 2*(3-1)*2=8, +1
    assert config.receptive_field == 4 + 8 + 1


def test_output_is_truly_causal() -> None:
    """t 시점 이후 입력을 바꿔도 t 시점 출력은 변하지 않아야 한다."""
    torch.manual_seed(0)
    config = _config(channels=(8, 8), kernel_size=3)
    net = CausalTCN(config)
    net.eval()

    seq_len = 20
    t = 10
    base = torch.randn(1, config.feature_dim, seq_len)
    perturbed = base.clone()
    perturbed[:, :, t + 1 :] += 100.0  # t 이후만 크게 바꾼다

    with torch.no_grad():
        gesture_base, phase_base = net(base)
        gesture_pert, phase_pert = net(perturbed)

    torch.testing.assert_close(gesture_base[:, :, : t + 1], gesture_pert[:, :, : t + 1])
    torch.testing.assert_close(phase_base[:, :, : t + 1], phase_pert[:, :, : t + 1])


def test_past_perturbation_changes_current_output() -> None:
    """수용영역 안의 과거를 바꾸면 현재 출력이 바뀔 수 있어야 한다(항상 0이 되는 퇴화 방지)."""
    torch.manual_seed(1)
    config = _config(channels=(8, 8), kernel_size=3)
    net = CausalTCN(config)
    net.eval()

    seq_len = 20
    assert config.receptive_field < seq_len  # 마지막 시점의 수용영역이 시퀀스 안에 들어오는지 확인
    base = torch.randn(1, config.feature_dim, seq_len)
    perturbed = base.clone()
    # 마지막 시점(index -1)의 수용영역 안(끝에서 receptive_field 프레임 이내)을 건드린다.
    edge = seq_len - config.receptive_field
    perturbed[:, :, edge:] += 50.0

    with torch.no_grad():
        gesture_base, _ = net(base)
        gesture_pert, _ = net(perturbed)

    assert not torch.allclose(gesture_base[:, :, -1], gesture_pert[:, :, -1])


def test_predict_output_is_within_valid_ranges() -> None:
    torch.manual_seed(2)
    config = _config()
    model = CausalTCNGestureModel(config)
    window = np.random.default_rng(0).normal(size=(config.receptive_field, config.feature_dim))

    prediction = model.predict(window)

    assert prediction.gesture in config.gesture_labels
    assert isinstance(prediction.phase, GesturePhase)
    assert 0.0 <= prediction.gesture_confidence <= 1.0
    assert 0.0 <= prediction.phase_confidence <= 1.0
    assert 0.0 <= prediction.uncertainty <= 1.0


def test_predict_pads_short_window() -> None:
    config = _config()
    model = CausalTCNGestureModel(config)
    short_window = np.zeros((1, config.feature_dim))
    prediction = model.predict(short_window)  # 예외 없이 0-패딩되어 처리됨
    assert prediction.gesture in config.gesture_labels


def test_predict_truncates_long_window_to_receptive_field() -> None:
    config = _config()
    model = CausalTCNGestureModel(config)
    long_window = np.zeros((config.receptive_field + 50, config.feature_dim))
    prediction = model.predict(long_window)
    assert prediction.gesture in config.gesture_labels


def test_predict_rejects_wrong_feature_dim() -> None:
    config = _config()
    model = CausalTCNGestureModel(config)
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((config.receptive_field, config.feature_dim + 1)))


def test_predict_rejects_non_finite_window() -> None:
    config = _config()
    model = CausalTCNGestureModel(config)
    window = np.zeros((config.receptive_field, config.feature_dim))
    window[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(window)


def test_swappable_architecture_via_config() -> None:
    """채널·kernel_size를 바꾸면 receptive_field와 window_size가 함께 바뀐다."""
    small = CausalTCNGestureModel(_config(channels=(4,), kernel_size=2))
    large = CausalTCNGestureModel(_config(channels=(8, 8, 8), kernel_size=5))
    assert small.window_size < large.window_size


def test_swappable_labels_via_config() -> None:
    model = CausalTCNGestureModel(
        _config(gesture_labels=("none", "swipe_up", "swipe_down", "swipe_left"))
    )
    assert model.labels == ("none", "swipe_up", "swipe_down", "swipe_left")
