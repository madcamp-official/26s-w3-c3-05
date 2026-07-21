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
    assert JESTER_TO_OUR_LABEL["No gesture"] == "none"
    assert JESTER_TO_OUR_LABEL["Turning Hand Clockwise"] == "rotate_clockwise"
    assert JESTER_TO_OUR_LABEL["Turning Hand Counterclockwise"] == "rotate_counter_clockwise"
    assert JESTER_TO_OUR_LABEL["Sliding Two Fingers Up"] == "slide_two_fingers_up"
    assert JESTER_TO_OUR_LABEL["Sliding Two Fingers Down"] == "slide_two_fingers_down"
    assert JESTER_TO_OUR_LABEL["Sliding Two Fingers Left"] == "slide_two_fingers_left"
    assert JESTER_TO_OUR_LABEL["Sliding Two Fingers Right"] == "slide_two_fingers_right"
    assert JESTER_TO_OUR_LABEL["Drumming Fingers"] == "drumming_fingers"
    assert JESTER_TO_OUR_LABEL["Doing other things"] == "doing_other_things"


def test_excluded_mappings_are_none() -> None:
    # 2026-07-20: swipe를 포함한 17개는 이번 라운드에서 제외(학습 대상 아님).
    # (Stop Sign은 2026-07-20 stop_sign 전경 제스처로 추가돼 제외 목록에서 빠졌다.)
    for jester_label in ("Swiping Up", "Swiping Down", "Swiping Left", "Swiping Right"):
        assert JESTER_TO_OUR_LABEL[jester_label] is None


def test_flip_swap_is_symmetric() -> None:
    for a, b in FLIP_LABEL_SWAP.items():
        assert FLIP_LABEL_SWAP[b] == a


def test_swap_label_for_flip_round_trips() -> None:
    assert swap_label_for_flip("slide_two_fingers_left") == "slide_two_fingers_right"
    assert swap_label_for_flip("slide_two_fingers_right") == "slide_two_fingers_left"
    assert swap_label_for_flip("rotate_clockwise") == "rotate_counter_clockwise"
    assert swap_label_for_flip("rotate_counter_clockwise") == "rotate_clockwise"


def test_swap_label_for_flip_leaves_unmapped_label_unchanged() -> None:
    assert swap_label_for_flip("none") == "none"


def test_validate_mapping_rejects_unreachable_label(monkeypatch: pytest.MonkeyPatch) -> None:
    # 출처가 하나뿐인 라벨을 골라 끊는다.
    monkeypatch.setitem(JESTER_TO_OUR_LABEL, "Turning Hand Clockwise", None)
    with pytest.raises(ValueError, match="no Jester source"):
        validate_mapping()
