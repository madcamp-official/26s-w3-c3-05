"""ModelConfig 검증 규칙 테스트."""

from __future__ import annotations

import pytest

from jarvis.gesture_fusion.model import ModelConfig


def test_default_labels_include_none_background_class() -> None:
    config = ModelConfig(feature_dim=10)
    assert "none" in config.gesture_labels


def test_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ModelConfig(feature_dim=10, gesture_labels=("swipe_down", "swipe_down"))


def test_rejects_single_label() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ModelConfig(feature_dim=10, gesture_labels=("only_one",))


def test_rejects_non_positive_feature_dim() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        ModelConfig(feature_dim=0)


def test_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        ModelConfig(feature_dim=10, channels=())


def test_rejects_kernel_size_below_two() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ModelConfig(feature_dim=10, kernel_size=1)
