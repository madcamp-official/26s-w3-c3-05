"""클립 내 상대적 위치로 프레임별 phase(IDLE/ONSET/ACTIVE/ENDING)를 근사한다.

Jester·웹캠 파인튜닝 클립 모두 "클립 전체가 제스처 하나(또는 무제스처)"라는
클립 단위 라벨만 있고, 실제 ONSET/ENDING 프레임 경계 정보는 없다. 완벽하진 않지만
grounding 없이 phase head를 같이 학습시킬 수 있는 유일한 실용적 방법으로, 클립
앞부분을 ONSET, 뒷부분을 ENDING, 중간을 ACTIVE로 근사한다(학습 파이프라인 인터뷰
결정). 배경(`"none"`) 클래스 클립은 전부 IDLE — 애초에 제스처가 없으므로 시작·끝
경계가 없다.

`jarvis.contracts.messages.GesturePhase`를 그대로 재사용한다(자체 enum 재정의 없음,
런타임 계약과 동일한 값 집합을 보장).
"""

from __future__ import annotations

from jarvis.contracts.messages import GesturePhase


def label_phases(
    frame_count: int,
    gesture_label: str,
    *,
    background_label: str = "none",
    onset_fraction: float = 0.15,
    ending_fraction: float = 0.15,
) -> tuple[GesturePhase, ...]:
    """프레임 수와 클립 라벨로부터 프레임별 phase 시퀀스를 근사한다.

    클립이 너무 짧아 onset+ending 구간이 전체 길이를 넘으면(예: 4프레임짜리 클립에
    15%씩 반올림해도 각 최소 1프레임은 필요) ACTIVE 없이 앞뒤 절반씩 ONSET/ENDING으로
    나눈다 — 짧은 클립에서 존재하지 않는 ACTIVE 구간을 지어내지 않기 위함.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if not 0.0 < onset_fraction < 0.5 or not 0.0 < ending_fraction < 0.5:
        raise ValueError("onset_fraction and ending_fraction must each be within (0, 0.5)")

    if gesture_label == background_label:
        return tuple(GesturePhase.IDLE for _ in range(frame_count))

    onset_frames = max(1, round(frame_count * onset_fraction))
    ending_frames = max(1, round(frame_count * ending_fraction))

    if onset_frames + ending_frames > frame_count:
        onset_frames = frame_count // 2
        ending_frames = frame_count - onset_frames
        return tuple(
            GesturePhase.ONSET if i < onset_frames else GesturePhase.ENDING
            for i in range(frame_count)
        )

    return tuple(
        GesturePhase.ONSET
        if i < onset_frames
        else GesturePhase.ENDING
        if i >= frame_count - ending_frames
        else GesturePhase.ACTIVE
        for i in range(frame_count)
    )
