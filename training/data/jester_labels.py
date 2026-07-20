"""Jester 27개 클래스 → 우리 gesture 라벨 매핑.

`jarvis.gesture_fusion.model_protocol.DEFAULT_GESTURE_LABELS`(열린 문자열 키,
interface-contract.md 공통 규칙)만 참조한다 — 라벨 문자열을 여기 다시 하드코딩하지
않아, 라벨 집합이 바뀌면 이 파일이 자동으로 따라간다.

2026-07-19 결정: 학습 제스처 목록 전부를 개별 클래스로 학습한다. 27개 Jester
클래스를 각각 고유 라벨로 매핑한다 — "No gesture"만 배경 클래스 "none"에 대응하고
"Doing other things"는 배경과 구분되는 별도 라벨("doing_other_things")로 둔다.
따라서 `None`(학습 제외)은 더 이상 없다. 라벨 문자열·순서는 `DEFAULT_GESTURE_LABELS`가
정의하고, 이 테이블은 Jester 클래스명 ↔ 그 라벨의 대응만 담는다.
"""

from __future__ import annotations

from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS

# Jester 27개 클래스 전체를 매핑한다 — None(학습 제외)은 없다. 인식 대상 12종은
# 각자 고유 라벨을 갖고, 대상 외 15종(밀기/당기기·굴리기·줌·엄지·손 흔들기·딴짓)은
# 전부 배경 "none"으로 모은다(DEFAULT_GESTURE_LABELS 주석의 선정·통합 근거 참조).
# 버리지 않고 모으는 이유: 데이터를 살리면서 모델이 "동작 아님"을 배우게 해 런타임
# 오탐(원치 않는 기기 동작)을 줄인다.
JESTER_TO_OUR_LABEL: dict[str, str | None] = {
    # --- 인식 대상 (12종) ---
    "Swiping Up": "swipe_up",
    "Swiping Down": "swipe_down",
    "Swiping Left": "swipe_left",
    "Swiping Right": "swipe_right",
    "Turning Hand Clockwise": "rotate_clockwise",
    "Turning Hand Counterclockwise": "rotate_counter_clockwise",
    "Sliding Two Fingers Up": "slide_two_fingers_up",
    "Sliding Two Fingers Down": "slide_two_fingers_down",
    "Sliding Two Fingers Left": "slide_two_fingers_left",
    "Sliding Two Fingers Right": "slide_two_fingers_right",
    "Stop Sign": "stop_sign",
    "Drumming Fingers": "drumming_fingers",
    # --- 배경: 런타임에서 "동작 없음"인 것 전부 (깊이 의존 동작 + 비대상 동작) ---
    "No gesture": "none",
    "Doing other things": "none",
    "Pushing Hand Away": "none",
    "Pulling Hand In": "none",
    "Pushing Two Fingers Away": "none",
    "Pulling Two Fingers In": "none",
    "Rolling Hand Forward": "none",
    "Rolling Hand Backward": "none",
    "Zooming In With Full Hand": "none",
    "Zooming Out With Full Hand": "none",
    "Zooming In With Two Fingers": "none",
    "Zooming Out With Two Fingers": "none",
    "Thumb Up": "none",
    "Thumb Down": "none",
    "Shaking Hand": "none",
}

# 좌우반전 augmentation이 라벨을 스왑해야 하는 쌍(양방향). 좌우가 그대로 뒤바뀌는
# 제스처만 스왑한다: swipe left/right, 두 손가락 슬라이드 left/right, 그리고 회전
# (카메라 기준 시계방향 → 거울상에서 반시계방향). 위/아래·앞/뒤·줌·엄지 등은 좌우
# 거울상에서 라벨이 바뀌지 않으므로 스왑하지 않는다. 매핑에 없는 라벨은 그대로다.
FLIP_LABEL_SWAP: dict[str, str] = {
    "swipe_left": "swipe_right",
    "swipe_right": "swipe_left",
    "slide_two_fingers_left": "slide_two_fingers_right",
    "slide_two_fingers_right": "slide_two_fingers_left",
    "rotate_clockwise": "rotate_counter_clockwise",
    "rotate_counter_clockwise": "rotate_clockwise",
}


def swap_label_for_flip(label: str) -> str:
    """좌우반전 augmentation 적용 시 라벨을 대응 라벨로 바꾼다(없으면 그대로)."""
    return FLIP_LABEL_SWAP.get(label, label)


def validate_mapping() -> None:
    """매핑 테이블의 불변식을 검사한다 — import 시가 아니라 명시적으로 호출한다.

    (1) 매핑 대상 우리 라벨은 전부 `DEFAULT_GESTURE_LABELS`에 있어야 한다(오타 방지).
    (2) `DEFAULT_GESTURE_LABELS`의 모든 라벨이 최소 하나의 Jester 클래스에서
        도달 가능해야 한다(학습 데이터에 아예 없는 클래스가 조용히 생기는 것 방지 —
        development-principles.md "no silent caps"와 같은 취지).
    """
    reachable: set[str] = set()
    for jester_label, our_label in JESTER_TO_OUR_LABEL.items():
        if our_label is None:
            continue
        if our_label not in DEFAULT_GESTURE_LABELS:
            raise ValueError(
                f"JESTER_TO_OUR_LABEL['{jester_label}'] = '{our_label}' is not in "
                f"DEFAULT_GESTURE_LABELS {DEFAULT_GESTURE_LABELS}"
            )
        reachable.add(our_label)

    unreachable = set(DEFAULT_GESTURE_LABELS) - reachable
    if unreachable:
        raise ValueError(
            f"DEFAULT_GESTURE_LABELS entries with no Jester source: {sorted(unreachable)} — "
            "학습 데이터에 이 클래스가 전혀 없다. JESTER_TO_OUR_LABEL을 편집하거나, "
            "웹캠 파인튜닝 단계에서 보충할 계획이면 이 함수 호출부에서 명시적으로 허용하라."
        )

    for swap_a, swap_b in FLIP_LABEL_SWAP.items():
        if FLIP_LABEL_SWAP.get(swap_b) != swap_a:
            raise ValueError(f"FLIP_LABEL_SWAP is not symmetric for '{swap_a}' <-> '{swap_b}'")
