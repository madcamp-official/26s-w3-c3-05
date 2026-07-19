"""Duplicate intent 방지 — README 9장 Commit 조건 7("동일 이벤트가 이전에 실행되지
않음"), `intent_id` 결정적 생성.

development-principles.md 2.3: "한 gesture event는 최대 하나의 intent를 만든다."
Task 4(spotting.py)가 제스처당 `ENDING`을 정확히 한 번만 내고, task 6(fusion.py)의
COOLDOWN이 시간적으로 가까운 재커밋을 막지만, 그것만으로는 같은 이벤트가
재전송·재생(retry, replay)되는 경우까지 막지 못한다 — 예: 배선 계층이 같은 프레임
구간을 두 번 흘려보내는 장애. 이 모듈은 커밋을 트리거한 원본 프레임(`frame_id`)을
기억해, 같은 프레임에서 나온 커밋이 다시 들어와도 두 번째 Intent를 만들지 않는다.

`intent_id`는 `frame_id`로부터 결정적으로 생성한다(2026-07-18 결정: Protocol이
`command_id`를 `cmd-{intent_id}`로 결정적 생성하는 것과 대칭 — 재시도가 항상 같은
`intent_id`로 수렴해야 두 dedup 계층이 함께 idempotency를 보장한다).
"""

from __future__ import annotations

from collections import OrderedDict


def generate_intent_id(frame_id: int) -> str:
    """커밋을 트리거한 `ENDING` 프레임의 `frame_id`로부터 결정적 `intent_id`를 만든다.

    `frame_id`는 캡처 파이프라인이 부여하는 단조 증가 값(interface-contract.md
    공통 규칙)이라, 같은 프레임에서 나온 재시도는 항상 같은 `intent_id`로 수렴한다.
    """
    return f"intent-{frame_id}"


class IntentDeduplicator:
    """이미 커밋된 `frame_id`를 기억해 Commit 조건 7을 판정한다.

    bounded LRU로 최근 `max_tracked`개만 기억한다 — 무한정 쌓이면 장시간 세션에서
    메모리가 누수된다(README 5장 bounded queue와 같은 원칙: 무한정 오래된 상태를
    붙들지 않는다). `frame_id`가 단조 증가하는 한, 밀려난 오래된 항목이 다시
    재생될 일은 없다.
    """

    def __init__(self, max_tracked: int = 256) -> None:
        if max_tracked < 1:
            raise ValueError("max_tracked must be at least 1")
        self._max_tracked = max_tracked
        self._seen: OrderedDict[int, str] = OrderedDict()

    def register(self, frame_id: int) -> str | None:
        """이 `frame_id`의 커밋을 신규 등록한다.

        이미 등록된 적이 있으면 `None`(중복, Commit 조건 7 위반)을 반환한다.
        신규면 결정적으로 생성한 `intent_id`를 반환한다.
        """
        if frame_id in self._seen:
            return None
        intent_id = generate_intent_id(frame_id)
        self._seen[frame_id] = intent_id
        if len(self._seen) > self._max_tracked:
            self._seen.popitem(last=False)
        return intent_id

    def __contains__(self, frame_id: int) -> bool:
        return frame_id in self._seen

    def reset(self) -> None:
        self._seen.clear()
