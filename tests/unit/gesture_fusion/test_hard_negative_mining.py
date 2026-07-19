"""Hard-negative mining·Wrong Actuation Rate 평가를 검증한다 (README 13장).

여기 쓰인 라벨은 합성(synthetic) 시나리오다 — 실제 캡처 데이터가 아니라
development-principles.md 1절 3에 따라 테스트 전용 fixture임을 이 docstring과
파일 위치(tests/)로 명시한다.
"""

from __future__ import annotations

import pytest

from jarvis.gesture_fusion.fusion import CommitDecision, FusionScore
from jarvis.gesture_fusion.hard_negative_mining import (
    HardNegativeConfig,
    LabeledCommitAttempt,
    compute_wrong_actuation_rate,
    mine_hard_negatives,
)


def _score(value: float) -> FusionScore:
    # value가 정확히 재현되도록 나머지 항은 1.0으로 고정한다.
    return FusionScore(
        target_confidence=1.0, gesture_confidence=1.0, gaze_stability=1.0,
        uncertainty=1.0 - value, value=value,
    )


def _decision(*, committed: bool, score_value: float | None = None) -> CommitDecision:
    return CommitDecision(
        committed=committed,
        reason="committed" if committed else "rejected",
        target="room.bulb",
        gesture="swipe_down",
        score=_score(score_value) if score_value is not None else None,
        timestamp_ms=100,
        frame_id=1,
        intent_id="intent-1" if committed else None,
    )


def _attempt(
    *, should_commit: bool, committed: bool, score_value: float | None = None
) -> LabeledCommitAttempt:
    return LabeledCommitAttempt(
        ground_truth_should_commit=should_commit,
        decision=_decision(committed=committed, score_value=score_value),
    )


# --- compute_wrong_actuation_rate ---


def test_correct_commit_is_not_wrong_actuation() -> None:
    attempts = [_attempt(should_commit=True, committed=True, score_value=0.8)]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="unit test")
    assert result.wrong_actuations == 0
    assert result.rate == 0.0


def test_correct_rejection_is_not_wrong_actuation() -> None:
    attempts = [_attempt(should_commit=False, committed=False, score_value=0.1)]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="unit test")
    assert result.wrong_actuations == 0


def test_false_positive_counts_as_wrong_actuation() -> None:
    attempts = [_attempt(should_commit=False, committed=True, score_value=0.6)]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="unit test")
    assert result.wrong_actuations == 1
    assert result.rate == 1.0


def test_missed_commit_is_not_wrong_actuation() -> None:
    """라벨상 커밋됐어야 하는데 놓친 경우는 WAR이 아니라 recall 문제다."""
    attempts = [_attempt(should_commit=True, committed=False, score_value=0.1)]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="unit test")
    assert result.wrong_actuations == 0


def test_rate_is_fraction_of_total() -> None:
    attempts = [
        _attempt(should_commit=True, committed=True, score_value=0.8),
        _attempt(should_commit=False, committed=True, score_value=0.6),
        _attempt(should_commit=False, committed=False, score_value=0.1),
        _attempt(should_commit=False, committed=False, score_value=0.1),
    ]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="unit test")
    assert result.total_attempts == 4
    assert result.wrong_actuations == 1
    assert result.rate == pytest.approx(0.25)


def test_empty_attempts_raises() -> None:
    with pytest.raises(ValueError, match="zero attempts"):
        compute_wrong_actuation_rate([], dataset_id="synthetic-1", conditions="unit test")


def test_result_carries_dataset_context() -> None:
    attempts = [_attempt(should_commit=True, committed=True, score_value=0.8)]
    result = compute_wrong_actuation_rate(attempts, dataset_id="synthetic-1", conditions="조명 어두움")
    assert result.dataset_id == "synthetic-1"
    assert result.conditions == "조명 어두움"


# --- mine_hard_negatives ---


def test_wrong_actuation_is_mined() -> None:
    attempts = [_attempt(should_commit=False, committed=True, score_value=0.6)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5)
    assert len(mined) == 1
    assert mined[0].kind == "wrong_actuation"


def test_near_miss_is_mined_within_margin() -> None:
    attempts = [_attempt(should_commit=False, committed=False, score_value=0.47)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5, config=HardNegativeConfig(near_miss_margin=0.05))
    assert len(mined) == 1
    assert mined[0].kind == "near_miss"


def test_far_from_threshold_is_not_mined() -> None:
    attempts = [_attempt(should_commit=False, committed=False, score_value=0.05)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5, config=HardNegativeConfig(near_miss_margin=0.05))
    assert mined == []


def test_correct_commit_is_not_mined() -> None:
    attempts = [_attempt(should_commit=True, committed=True, score_value=0.9)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5)
    assert mined == []


def test_missed_commit_is_not_mined_as_negative() -> None:
    """놓친 긍정 사례는 hard negative가 아니다 — recall이 다루는 별개 지표."""
    attempts = [_attempt(should_commit=True, committed=False, score_value=0.49)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5, config=HardNegativeConfig(near_miss_margin=0.05))
    assert mined == []


def test_rejected_without_score_is_not_mined() -> None:
    """정렬 실패 등으로 score 자체가 없는 거부는 near-miss 판정을 할 수 없다."""
    attempts = [_attempt(should_commit=False, committed=False, score_value=None)]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5)
    assert mined == []


def test_mining_preserves_multiple_cases() -> None:
    attempts = [
        _attempt(should_commit=False, committed=True, score_value=0.6),  # wrong actuation
        _attempt(should_commit=False, committed=False, score_value=0.48),  # near miss
        _attempt(should_commit=True, committed=True, score_value=0.9),  # 정상 커밋
        _attempt(should_commit=False, committed=False, score_value=0.01),  # 확실한 거부
    ]
    mined = mine_hard_negatives(attempts, commit_threshold=0.5, config=HardNegativeConfig(near_miss_margin=0.05))
    kinds = sorted(m.kind for m in mined)
    assert kinds == ["near_miss", "wrong_actuation"]


def test_hard_negative_config_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="near_miss_margin"):
        HardNegativeConfig(near_miss_margin=0.0)
    with pytest.raises(ValueError, match="near_miss_margin"):
        HardNegativeConfig(near_miss_margin=1.5)
