"""TrainingConfig(training/config.py) 검증 규칙을 확인한다."""

from __future__ import annotations

import pytest

from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig


def test_default_config_is_valid() -> None:
    TrainingConfig()  # 예외 없이 생성돼야 한다.


def test_default_training_config_singleton_matches_defaults() -> None:
    assert DEFAULT_TRAINING_CONFIG.phase_loss_weight == pytest.approx(0.3)
    assert DEFAULT_TRAINING_CONFIG.onset_fraction == pytest.approx(0.15)
    assert DEFAULT_TRAINING_CONFIG.ending_fraction == pytest.approx(0.15)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"max_epochs": 0},
        {"early_stopping_patience": 0},
        {"phase_loss_weight": -0.1},
        {"flip_probability": 1.5},
        {"time_warp_probability": -0.1},
        {"time_warp_rate_range": (1.0, 0.5)},
        {"onset_fraction": 0.6},
        {"ending_fraction": 0.0},
        {"onset_fraction": 0.5, "ending_fraction": 0.5},
        {"lr_min_factor": -0.1},
        {"lr_min_factor": 1.1},
    ],
)
def test_rejects_invalid_overrides(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)  # type: ignore[arg-type]


def test_lr_min_factor_accepts_boundary_values() -> None:
    TrainingConfig(lr_min_factor=0.0)  # 완전히 0까지 감쇠 — 허용.
    TrainingConfig(lr_min_factor=1.0)  # 감쇠 없음(상수 LR)과 동일 — 허용.
