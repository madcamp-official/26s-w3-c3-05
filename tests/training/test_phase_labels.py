"""클립 내 상대 위치 phase 라벨링 휴리스틱(training/phase_labels.py)을 검증한다."""

from __future__ import annotations

import pytest

from jarvis.contracts.messages import GesturePhase
from jarvis.gesture_fusion.model_protocol import DEFAULT_BACKGROUND_LABELS
from training.phase_labels import label_phases


def test_background_label_is_all_idle() -> None:
    phases = label_phases(30, "none", background_labels=frozenset({"none"}))
    assert phases == tuple(GesturePhase.IDLE for _ in range(30))


@pytest.mark.parametrize("background_label", sorted(DEFAULT_BACKGROUND_LABELS))
def test_every_default_background_label_is_all_idle(background_label: str) -> None:
    """배경 클래스는 전부 IDLE이어야 한다 — "none" 하나만이 아니다.

    2026-07-20 발견: `background_label`이 문자열 하나("none")뿐이라
    `drumming_fingers`·`doing_other_things`에도 ONSET→ACTIVE→ENDING이라는 가짜
    제스처 구조가 라벨링됐다. phase head가 "배경에서도 ONSET이 일어난다"고 배우면
    런타임 `GestureSpotter`가 배경 구간을 제스처 시작 후보로 받는다.
    """
    phases = label_phases(30, background_label)
    assert phases == tuple(GesturePhase.IDLE for _ in range(30))


def test_gesture_clip_has_onset_active_ending_in_order() -> None:
    phases = label_phases(20, "slide_two_fingers_down", onset_fraction=0.2, ending_fraction=0.2)
    assert phases[0] == GesturePhase.ONSET
    assert phases[-1] == GesturePhase.ENDING
    assert GesturePhase.ACTIVE in phases
    # ONSET 구간 뒤에 ACTIVE, 그 뒤에 ENDING만 오고 역행하지 않는다.
    seen_active = False
    seen_ending = False
    for phase in phases:
        if phase == GesturePhase.ACTIVE:
            seen_active = True
        if phase == GesturePhase.ENDING:
            seen_ending = True
            continue
        if seen_ending:
            pytest.fail("ENDING 이후 다른 phase가 다시 나타남")
        if phase == GesturePhase.ONSET and seen_active:
            pytest.fail("ACTIVE 이후 ONSET으로 되돌아감")


def test_onset_and_ending_fraction_sizes() -> None:
    phases = label_phases(100, "slide_two_fingers_up", onset_fraction=0.15, ending_fraction=0.15)
    onset_count = sum(1 for p in phases if p == GesturePhase.ONSET)
    ending_count = sum(1 for p in phases if p == GesturePhase.ENDING)
    assert onset_count == 15
    assert ending_count == 15
    assert sum(1 for p in phases if p == GesturePhase.ACTIVE) == 70


def test_short_clip_has_no_fabricated_active_segment() -> None:
    """onset+ending이 전체 길이를 넘는 아주 짧은 클립은 ACTIVE 없이 반씩 나뉜다."""
    phases = label_phases(2, "slide_two_fingers_down", onset_fraction=0.15, ending_fraction=0.15)
    assert GesturePhase.ACTIVE not in phases
    assert phases[0] == GesturePhase.ONSET
    assert phases[-1] == GesturePhase.ENDING


def test_rejects_non_positive_frame_count() -> None:
    with pytest.raises(ValueError):
        label_phases(0, "slide_two_fingers_down")


def test_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        label_phases(10, "slide_two_fingers_down", onset_fraction=0.6)
    with pytest.raises(ValueError):
        label_phases(10, "slide_two_fingers_down", ending_fraction=0.0)
