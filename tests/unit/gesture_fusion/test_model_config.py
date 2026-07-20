"""ModelConfig 검증 규칙 테스트."""

from __future__ import annotations

import pytest

# ModelConfig는 torch 무의존이라 model_protocol(torch-free)에서 가져온다 — `ml` extra
# 없이도 이 순수 검증 규칙을 CI에서 돌린다(importorskip 불필요).
from jarvis.gesture_fusion.model_protocol import ModelConfig


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


# --- 배경 클래스 집합 ---


def test_default_background_indices_cover_the_three_background_classes() -> None:
    config = ModelConfig(feature_dim=10)
    background = {config.gesture_labels[i] for i in config.background_indices}
    assert background == {"none", "drumming_fingers", "doing_other_things"}


def test_foreground_indices_are_the_actionable_gestures() -> None:
    config = ModelConfig(feature_dim=10)
    foreground = {config.gesture_labels[i] for i in config.foreground_indices}
    assert foreground == {
        "rotate_clockwise",
        "rotate_counter_clockwise",
        "slide_two_fingers_up",
        "slide_two_fingers_down",
        "slide_two_fingers_left",
        "slide_two_fingers_right",
    }


def test_background_and_foreground_indices_partition_the_labels() -> None:
    config = ModelConfig(feature_dim=10)
    combined = set(config.background_indices) | set(config.foreground_indices)
    assert combined == set(range(len(config.gesture_labels)))
    assert not set(config.background_indices) & set(config.foreground_indices)


def test_representative_background_is_index_zero() -> None:
    config = ModelConfig(feature_dim=10)
    assert config.background_indices[0] == 0


def test_rejects_background_label_that_is_not_a_known_label() -> None:
    """라벨을 개명하고 배경 집합을 안 고치면 그 동작이 조용히 전경이 되는 것을 막는다."""
    with pytest.raises(ValueError, match="unknown label"):
        ModelConfig(feature_dim=10, background_labels=frozenset({"none", "typo_gesture"}))


def test_rejects_when_first_label_is_not_background() -> None:
    with pytest.raises(ValueError, match="background label"):
        ModelConfig(
            feature_dim=10,
            gesture_labels=("rotate_clockwise", "none"),
            background_labels=frozenset({"none"}),
        )


def test_rejects_when_every_label_is_background() -> None:
    with pytest.raises(ValueError, match="non-background"):
        ModelConfig(
            feature_dim=10,
            gesture_labels=("none", "drumming_fingers"),
            background_labels=frozenset({"none", "drumming_fingers"}),
        )


def test_reduced_label_sets_may_reuse_the_default_background_set() -> None:
    """축소된 label 튜플(테스트·실험)에서 없는 배경 이름은 무시된다."""
    config = ModelConfig(feature_dim=10, gesture_labels=("none", "rotate_clockwise"))
    assert config.background_indices == (0,)
    assert config.foreground_indices == (1,)
