"""Jester 라벨 매핑 테이블(training/data/jester_labels.py)의 불변식을 검증한다."""

from __future__ import annotations

import pytest

from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS
from training.data.jester_labels import (
    FLIP_LABEL_SWAP,
    JESTER_TO_OUR_LABEL,
    swap_label_for_flip,
    validate_mapping,
)


def test_validate_mapping_passes_for_current_table() -> None:
    validate_mapping()  # 예외 없이 통과해야 한다.


def test_every_default_gesture_label_is_reachable() -> None:
    mapped = {v for v in JESTER_TO_OUR_LABEL.values() if v is not None}
    assert mapped == set(DEFAULT_GESTURE_LABELS)


def test_confirmed_mappings() -> None:
    assert JESTER_TO_OUR_LABEL["Swiping Up"] == "swipe_up"
    assert JESTER_TO_OUR_LABEL["Swiping Down"] == "swipe_down"
    assert JESTER_TO_OUR_LABEL["Swiping Left"] == "swipe_left"
    assert JESTER_TO_OUR_LABEL["Swiping Right"] == "swipe_right"
    assert JESTER_TO_OUR_LABEL["Turning Hand Clockwise"] == "rotate_clockwise"
    assert JESTER_TO_OUR_LABEL["Turning Hand Counterclockwise"] == "rotate_counter_clockwise"
    assert JESTER_TO_OUR_LABEL["No gesture"] == "none"


def test_flip_swap_is_symmetric() -> None:
    for a, b in FLIP_LABEL_SWAP.items():
        assert FLIP_LABEL_SWAP[b] == a


def test_swap_label_for_flip_round_trips() -> None:
    assert swap_label_for_flip("swipe_left") == "swipe_right"
    assert swap_label_for_flip("swipe_right") == "swipe_left"
    assert swap_label_for_flip("rotate_clockwise") == "rotate_counter_clockwise"
    assert swap_label_for_flip("rotate_counter_clockwise") == "rotate_clockwise"


def test_swap_label_for_flip_leaves_unmapped_label_unchanged() -> None:
    assert swap_label_for_flip("none") == "none"


def test_validate_mapping_rejects_unreachable_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(JESTER_TO_OUR_LABEL, "No gesture", None)
    with pytest.raises(ValueError, match="no Jester source"):
        validate_mapping()
