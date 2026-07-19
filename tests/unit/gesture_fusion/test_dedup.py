"""Duplicate intent 방지를 검증한다 (README 9장 Commit 조건 7, intent_id 생성)."""

from __future__ import annotations

import pytest

from jarvis.gesture_fusion.dedup import IntentDeduplicator, generate_intent_id


def test_generate_intent_id_is_deterministic() -> None:
    assert generate_intent_id(1042) == generate_intent_id(1042)


def test_generate_intent_id_differs_by_frame() -> None:
    assert generate_intent_id(1) != generate_intent_id(2)


def test_first_registration_returns_intent_id() -> None:
    dedup = IntentDeduplicator()
    intent_id = dedup.register(100)
    assert intent_id == generate_intent_id(100)


def test_second_registration_of_same_frame_returns_none() -> None:
    dedup = IntentDeduplicator()
    dedup.register(100)
    assert dedup.register(100) is None


def test_different_frames_both_register() -> None:
    dedup = IntentDeduplicator()
    first = dedup.register(1)
    second = dedup.register(2)
    assert first is not None
    assert second is not None
    assert first != second


def test_contains_reflects_registration() -> None:
    dedup = IntentDeduplicator()
    assert 5 not in dedup
    dedup.register(5)
    assert 5 in dedup


def test_reset_clears_history() -> None:
    dedup = IntentDeduplicator()
    dedup.register(1)
    dedup.reset()
    assert 1 not in dedup
    assert dedup.register(1) is not None


def test_bounded_history_evicts_oldest() -> None:
    dedup = IntentDeduplicator(max_tracked=2)
    dedup.register(1)
    dedup.register(2)
    dedup.register(3)  # frame 1이 밀려남
    assert 1 not in dedup
    assert 2 in dedup
    assert 3 in dedup
    # 밀려난 frame_id는 다시 신규로 등록될 수 있다(더 이상 기억하지 않으므로).
    assert dedup.register(1) is not None


def test_max_tracked_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_tracked"):
        IntentDeduplicator(max_tracked=0)
