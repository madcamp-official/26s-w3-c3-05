"""Jester 27개 클래스 → 우리 gesture 라벨 매핑.

`jarvis.gesture_fusion.model_protocol.DEFAULT_GESTURE_LABELS`(열린 문자열 키,
interface-contract.md 공통 규칙)만 참조한다 — 라벨 문자열을 여기 다시 하드코딩하지
않아, 라벨 집합이 바뀌면 이 파일이 자동으로 따라간다.

`None`으로 매핑된 클래스는 학습셋에서 **제외**한다(포함하지 않음, "none"으로
접지 않음) — 어느 쪽이 맞는지는 아직 결정되지 않았다(학습 파이프라인 인터뷰 기록).
사용자가 나중에 이 테이블만 편집해 확정하면 된다: 코드·다른 파일은 손댈 필요 없다.
"""

from __future__ import annotations

from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS

# 확정된 매핑만 채운다(README 8장 "지원 제스처"와 직접 대응). 나머지 19개는 아직
# 미정이라 전부 제외(None)로 둔다 — TODO: 사용자가 "none"으로 접을지/뺄지 결정.
JESTER_TO_OUR_LABEL: dict[str, str | None] = {
    "Swiping Up": "swipe_up",
    "Swiping Down": "swipe_down",
    "Swiping Left": "swipe_left",
    "Swiping Right": "swipe_right",
    "Turning Hand Clockwise": "rotate_clockwise",
    "Turning Hand Counterclockwise": "rotate_counter_clockwise",
    "No gesture": "none",
    # --- TODO(사용자 결정 대기): 아래는 "none"으로 접을지 제외할지 미정 ---
    "Doing other things": None,
    "Pushing Hand Away": None,
    "Pulling Hand In": None,
    "Sliding Two Fingers Left": None,
    "Sliding Two Fingers Right": None,
    "Sliding Two Fingers Down": None,
    "Sliding Two Fingers Up": None,
    "Pushing Two Fingers Away": None,
    "Pulling Two Fingers In": None,
    "Rolling Hand Forward": None,
    "Rolling Hand Backward": None,
    "Zooming In With Full Hand": None,
    "Zooming Out With Full Hand": None,
    "Zooming In With Two Fingers": None,
    "Zooming Out With Two Fingers": None,
    "Thumb Up": None,
    "Thumb Down": None,
    "Shaking Hand": None,
    "Stop Sign": None,
    "Drumming Fingers": None,
}

# 좌우반전 augmentation이 라벨을 스왑해야 하는 쌍(양방향). 회전은 거울상에서
# 방향이 반대로 보이고(카메라 기준 시계방향 → 거울상에서는 반시계방향), swipe는
# 좌우가 그대로 뒤바뀐다. 매핑에 없는 라벨(예: "none")은 반전해도 그대로다.
FLIP_LABEL_SWAP: dict[str, str] = {
    "swipe_left": "swipe_right",
    "swipe_right": "swipe_left",
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
