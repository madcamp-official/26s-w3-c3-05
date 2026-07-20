"""Jester 27개 클래스 → 우리 gesture 라벨 매핑.

`jarvis.gesture_fusion.model_protocol.DEFAULT_GESTURE_LABELS`(열린 문자열 키,
interface-contract.md 공통 규칙)만 참조한다 — 라벨 문자열을 여기 다시 하드코딩하지
않아, 라벨 집합이 바뀌면 이 파일이 자동으로 따라간다.

2026-07-20 결정: 사용자가 지정한 8개 제스처(+배경 "none")만 학습한다 —
swipe를 포함한 나머지 18개 Jester 클래스는 이번 라운드에서 제외("none"으로도
뭉치지 않고 학습셋에서 아예 빠짐, `None`으로 매핑). README 8장 "지원 제스처"
(swipe 4종)와는 다른 목록이라는 점을 인지하고 있음 — README는 이후 갱신 예정.
라벨 문자열·순서는 `DEFAULT_GESTURE_LABELS`가 정의하고, 이 테이블은 Jester
클래스명 ↔ 그 라벨(또는 제외)의 대응만 담는다.

제외된 18개는 `training/extract/extract_jester.py`의 `build_manifest()`가
애초에 추출 대상에서 뺀다 — 캐시에 이미 있는 클립(이전 27클래스 라운드에서
저장됨)은 그대로 남지만 더는 학습에 쓰이지 않는다(ClipDataset이 라벨 불일치를
잡아내지 않도록, 학습 전 `training.relabel_cache`로 캐시를 갱신해야 한다 —
train/README.md 참고).
"""

from __future__ import annotations

from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS

# 지정된 8종 + 배경만 매핑한다. 나머지 18개는 None(학습 제외) — 캐시에서도,
# 추출 대상에서도 빠진다.
JESTER_TO_OUR_LABEL: dict[str, str | None] = {
    "No gesture": "none",
    "Turning Hand Clockwise": "rotate_clockwise",
    "Turning Hand Counterclockwise": "rotate_counter_clockwise",
    "Sliding Two Fingers Up": "slide_two_fingers_up",
    "Sliding Two Fingers Down": "slide_two_fingers_down",
    "Sliding Two Fingers Left": "slide_two_fingers_left",
    "Sliding Two Fingers Right": "slide_two_fingers_right",
    "Drumming Fingers": "drumming_fingers",
    "Doing other things": "doing_other_things",
    # --- 이번 라운드 제외 (README 공식 목록의 swipe 포함) ---
    "Swiping Up": None,
    "Swiping Down": None,
    "Swiping Left": None,
    "Swiping Right": None,
    "Pushing Hand Away": None,
    "Pulling Hand In": None,
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
}

# 좌우반전 augmentation이 라벨을 스왑해야 하는 쌍(양방향). 좌우가 그대로 뒤바뀌는
# 제스처만 스왑한다: 두 손가락 슬라이드 left/right, 회전(카메라 기준 시계방향 →
# 거울상에서 반시계방향). 매핑에 없는 라벨은 그대로다.
FLIP_LABEL_SWAP: dict[str, str] = {
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
