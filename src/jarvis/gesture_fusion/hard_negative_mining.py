"""Hard-negative mining — README 12장 담당 범위 "hard-negative mining", 13장
Wrong Actuation Rate(≤1%) 평가.

development-principles.md 1절 4: "Wrong Actuation Rate... 계산할 때 데이터셋,
실행 조건, 분모와 측정 구간을 함께 기록한다. 수동으로 만든 숫자나 재현할 수 없는
결과를 사용하지 않는다." `jarvis.gaze.evaluation`(Target Selection Accuracy)과
같은 패턴으로, 결과값에 `dataset_id`·`conditions`를 강제로 담아 숫자만 따로
떠도는 것을 막는다.

"hard negative"는 두 종류로 나눈다:

1. **실제 오발(wrong_actuation)** — 라벨상 커밋되면 안 됐는데 `FusionEngine`
   (task 6)이 실제로 커밋한 경우. Wrong Actuation Rate 분자에 직접 들어간다.
2. **near-miss** — 라벨대로 정상 거부됐지만 결합 점수가 `commit_threshold`에
   아주 가까웠던 경우. 지금은 우연히 안전한 쪽으로 떨어졌을 뿐이라, 모델·
   threshold를 재보정할 때 우선적으로 다시 봐야 할 사례다.

두 종류 모두 재학습·threshold 재보정 파이프라인의 입력으로 쓰도록 리스트로
낸다. raw feature window는 이 계층에 없다(모델 추론 이후 정보만 다룸) — 원본
시퀀스가 필요하면 replay trace의 `frame_id`로 캡처 로그에서 다시 찾는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.gesture_fusion.fusion import CommitDecision


@dataclass(frozen=True, slots=True)
class LabeledCommitAttempt:
    """정답이 있는 replay trace의 한 제스처 완결(`ENDING`) 시도.

    `decision`은 그 시도에 대해 `FusionEngine`이 실제로 낸 `CommitDecision`이다.
    `ground_truth_should_commit=False`인데 `decision.committed=True`면 실제
    Wrong Actuation이다. README 13장이 나열한 5가지 오발 유형("잘못된 기기
    선택"·"잘못된 제스처 실행"·"시선만으로 실행"·"Target Lock 만료 후 실행"·
    "동일 명령 중복 실행")은 시나리오 라벨링 시 전부 이 하나의 불리언으로
    환원한다 — 무엇이 왜 틀렸는지는 `ground_truth_reason`에 자유 텍스트로 남긴다.
    """

    ground_truth_should_commit: bool
    decision: CommitDecision
    ground_truth_reason: str = ""


@dataclass(frozen=True, slots=True)
class WrongActuationRateResult:
    """측정 결과 — 비율과 재현에 필요한 맥락을 함께 담는다."""

    dataset_id: str
    conditions: str
    total_attempts: int
    wrong_actuations: int

    @property
    def rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.wrong_actuations / self.total_attempts


def compute_wrong_actuation_rate(
    attempts: list[LabeledCommitAttempt], dataset_id: str, conditions: str
) -> WrongActuationRateResult:
    """README 13장: WAR = 잘못 실행된 명령 수 / 전체 명령 시도 수.

    분모는 라벨된 전체 제스처 완결 시도 수(커밋됐든 안 됐든)다. 분자는 라벨상
    커밋되면 안 됐는데 실제로 커밋된 시도 수다 — 정상적으로 거부된 시도나,
    라벨상 커밋돼야 했는데 놓친 시도(recall 문제, WAR과는 다른 지표)는 분자에
    포함하지 않는다.

    `dataset_id`·`conditions`는 어떤 데이터셋·시나리오에서 측정했는지를 결과에
    강제로 남기기 위한 필수 인자다(development-principles.md 1절 4).
    """
    if not attempts:
        raise ValueError("Cannot compute Wrong Actuation Rate over zero attempts.")
    wrong = sum(
        1 for a in attempts if a.decision.committed and not a.ground_truth_should_commit
    )
    return WrongActuationRateResult(
        dataset_id=dataset_id,
        conditions=conditions,
        total_attempts=len(attempts),
        wrong_actuations=wrong,
    )


@dataclass(frozen=True, slots=True)
class HardNegativeConfig:
    """hard-negative 채집 파라미터."""

    near_miss_margin: float = 0.05
    """정답대로 거부됐지만 결합 점수가 `commit_threshold`에서 이 값 이내면
    near-miss로 채집한다."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.near_miss_margin) or not 0.0 < self.near_miss_margin <= 1.0:
            raise ValueError("near_miss_margin must be finite and within (0, 1]")


DEFAULT_HARD_NEGATIVE_CONFIG = HardNegativeConfig()


@dataclass(frozen=True, slots=True)
class HardNegative:
    """재학습·threshold 재보정 파이프라인에 우선적으로 넘길 사례 하나."""

    kind: str
    """`"wrong_actuation"` 또는 `"near_miss"`."""

    attempt: LabeledCommitAttempt


def mine_hard_negatives(
    attempts: list[LabeledCommitAttempt],
    commit_threshold: float,
    config: HardNegativeConfig = DEFAULT_HARD_NEGATIVE_CONFIG,
) -> list[HardNegative]:
    """실제 오발(`wrong_actuation`)과 근접 오발(`near_miss`) 사례를 모은다.

    `commit_threshold`는 이 배치를 만든 `FusionEngine`의 `FusionConfig.
    commit_threshold`와 같아야 한다 — near-miss 판정이 그 값을 기준으로 하기
    때문이다. 라벨상 커밋됐어야 하는데 놓친 경우(false negative)는 다루지
    않는다 — 이 함수의 대상은 "negative"(거부돼야 할 사례) 쪽 채굴이며, 놓친
    긍정 사례는 Gesture Event Recall이 다루는 별개 지표다.
    """
    hard_negatives: list[HardNegative] = []
    for attempt in attempts:
        if attempt.decision.committed and not attempt.ground_truth_should_commit:
            hard_negatives.append(HardNegative("wrong_actuation", attempt))
            continue
        if attempt.decision.committed or attempt.ground_truth_should_commit:
            continue
        score = attempt.decision.score
        if score is None:
            continue
        if abs(score.value - commit_threshold) <= config.near_miss_margin:
            hard_negatives.append(HardNegative("near_miss", attempt))
    return hard_negatives
