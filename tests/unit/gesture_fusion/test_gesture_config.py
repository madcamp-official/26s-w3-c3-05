"""GestureConfig 검증 규칙 테스트 — 잘못된 파라미터를 조용히 통과시키지 않는다."""

from __future__ import annotations

import pytest

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig


def test_default_config_is_valid() -> None:
    assert DEFAULT_GESTURE_CONFIG.num_hands == 1


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="min_hand_detection_confidence"):
        GestureConfig(min_hand_detection_confidence=1.5)


def test_num_hands_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_hands"):
        GestureConfig(num_hands=0)


def test_palm_reference_indices_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        GestureConfig(palm_scale_root_index=5, palm_scale_tip_index=5)


def test_palm_reference_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="valid hand landmark index"):
        GestureConfig(palm_scale_tip_index=99)


def test_origin_index_out_of_range() -> None:
    with pytest.raises(ValueError, match="origin_index"):
        GestureConfig(origin_index=99)
