"""Target Selection Accuracy evaluation (README 13장 "필수 제약" ≥ 90%).

development-principles.md 1절 4: "Target Selection Accuracy... 계산할 때
데이터셋, 실행 조건, 분모와 측정 구간을 함께 기록한다. 수동으로 만든 숫자나
재현할 수 없는 결과를 사용하지 않는다." 이 모듈은 그 맥락(`dataset_id`,
`conditions`)을 결과 값에 강제로 포함시켜, 정확도 숫자만 따로 떠도는 것을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledFrame:
    """정답이 있는 replay trace의 한 프레임."""

    frame_id: int
    timestamp_ms: int
    predicted_target: str
    ground_truth_target: str


@dataclass(frozen=True, slots=True)
class TargetSelectionAccuracyResult:
    """측정 결과 — 정확도 숫자와 재현에 필요한 맥락을 함께 담는다."""

    dataset_id: str
    conditions: str
    total_frames: int
    correct_frames: int

    @property
    def accuracy(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.correct_frames / self.total_frames


def compute_target_selection_accuracy(
    frames: list[LabeledFrame], dataset_id: str, conditions: str
) -> TargetSelectionAccuracyResult:
    """프레임 단위 Target Selection Accuracy를 계산한다.

    정의: 분모는 ground truth가 있는 전체 프레임 수, 분자는
    `predicted_target == ground_truth_target`인 프레임 수다(정답이 UNKNOWN인
    프레임에서 UNKNOWN을 맞히는 것도 정답으로 센다). 이 정의를 바꾸면
    documents/gaze.md와 documents/decisions.md에 기록한다.

    `dataset_id`·`conditions`는 어떤 데이터셋·환경(조명·안경 착용·카메라 거리 등,
    README 15장 "환경 변화")에서 측정했는지를 결과에 강제로 남기기 위한 필수
    인자다.
    """
    if not frames:
        raise ValueError("Cannot compute Target Selection Accuracy over zero frames.")
    correct = sum(1 for f in frames if f.predicted_target == f.ground_truth_target)
    return TargetSelectionAccuracyResult(
        dataset_id=dataset_id,
        conditions=conditions,
        total_frames=len(frames),
        correct_frames=correct,
    )
