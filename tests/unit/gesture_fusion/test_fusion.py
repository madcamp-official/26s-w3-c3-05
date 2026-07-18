"""Fusion confidence·safe commit을 검증한다 (README 9장 결합 점수, Commit 조건
4·5, COOLDOWN).
"""

from __future__ import annotations

import pytest

from jarvis.contracts.messages import GestureEstimate, GesturePhase, TargetEstimate
from jarvis.gesture_fusion.alignment import AlignmentConfig
from jarvis.gesture_fusion.fusion import (
    FusionConfig,
    FusionEngine,
    IntentPhase,
    compute_fusion_score,
)


def _target(
    timestamp_ms: int,
    *,
    target: str = "room.bulb",
    probability: float = 0.9,
    second_best_probability: float = 0.05,
    stability: float = 0.9,
) -> TargetEstimate:
    return TargetEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=0,
        target=target,
        probability=probability,
        second_best_probability=second_best_probability,
        stability=stability,
    )


def _gesture(
    timestamp_ms: int,
    phase: GesturePhase,
    *,
    gesture: str = "swipe_down",
    gesture_confidence: float = 0.9,
    uncertainty: float = 0.05,
) -> GestureEstimate:
    return GestureEstimate(
        timestamp_ms=timestamp_ms,
        frame_id=0,
        gesture=gesture,
        gesture_confidence=gesture_confidence,
        phase=phase,
        phase_confidence=0.9,
        uncertainty=uncertainty,
    )


def _alignment_config(**overrides: object) -> AlignmentConfig:
    defaults: dict[str, object] = dict(target_dwell_ms=0, target_lock_ttl_ms=2000)
    defaults.update(overrides)
    return AlignmentConfig(**defaults)  # type: ignore[arg-type]


def _fusion_config(**overrides: object) -> FusionConfig:
    defaults: dict[str, object] = dict(
        commit_threshold=0.3, min_target_confidence=0.5, min_gesture_confidence=0.5, cooldown_ms=500
    )
    defaults.update(overrides)
    return FusionConfig(**defaults)  # type: ignore[arg-type]


def _locked_engine(fusion_config: FusionConfig | None = None) -> FusionEngine:
    engine = FusionEngine(fusion_config or _fusion_config(), _alignment_config())
    engine.push_target(_target(0))
    engine.push_target(_target(0))
    return engine


# --- compute_fusion_score ---


def test_fusion_score_multiplies_four_terms() -> None:
    score = compute_fusion_score(
        target_confidence=0.9, gesture_confidence=0.8, gaze_stability=0.9, uncertainty=0.1
    )
    assert score.value == pytest.approx(0.9 * 0.8 * 0.9 * 0.9)


def test_fusion_score_rejects_out_of_range_input() -> None:
    with pytest.raises(ValueError, match="target_confidence"):
        compute_fusion_score(1.5, 0.8, 0.9, 0.1)


def test_high_uncertainty_drives_score_toward_zero() -> None:
    score = compute_fusion_score(1.0, 1.0, 1.0, uncertainty=1.0)
    assert score.value == 0.0


# --- FusionEngine: 정상 커밋 경로 ---


def test_full_commit_path() -> None:
    engine = _locked_engine()
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    decision = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert decision is not None
    assert decision.committed
    assert decision.target == "room.bulb"
    assert decision.gesture == "swipe_down"
    assert decision.score is not None


def test_no_decision_before_gesture_ends() -> None:
    engine = _locked_engine()
    assert engine.push_gesture(_gesture(100, GesturePhase.ONSET)) is None
    assert engine.push_gesture(_gesture(150, GesturePhase.ACTIVE)) is None


# --- Commit 조건 4·5: 개별 confidence 하한 ---


def test_rejects_low_target_confidence() -> None:
    engine = FusionEngine(
        _fusion_config(min_target_confidence=0.95), _alignment_config()
    )
    engine.push_target(_target(0, probability=0.85, second_best_probability=0.05))
    engine.push_target(_target(0, probability=0.85, second_best_probability=0.05))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    decision = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert decision is not None
    assert not decision.committed
    assert "target confidence" in decision.reason


def test_rejects_low_gesture_confidence() -> None:
    engine = _locked_engine(_fusion_config(min_gesture_confidence=0.95))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET, gesture_confidence=0.6))
    decision = engine.push_gesture(
        _gesture(300, GesturePhase.ENDING, gesture_confidence=0.6)
    )
    assert decision is not None
    assert not decision.committed
    assert "gesture confidence" in decision.reason


# --- threshold ---


def test_rejects_below_commit_threshold() -> None:
    engine = _locked_engine(_fusion_config(commit_threshold=0.99))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    decision = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert decision is not None
    assert not decision.committed
    assert "threshold" in decision.reason


# --- 정렬 실패(task 5)도 그대로 전파 ---


def test_rejects_when_not_aligned() -> None:
    engine = FusionEngine(_fusion_config(), _alignment_config())  # lock 안 함
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    decision = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert decision is not None
    assert not decision.committed
    assert "not locked" in decision.reason


# --- COOLDOWN ---


def test_cooldown_blocks_immediate_second_commit() -> None:
    engine = _locked_engine(_fusion_config(cooldown_ms=1000))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    first = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert first is not None and first.committed

    engine.push_gesture(_gesture(400, GesturePhase.ONSET))
    second = engine.push_gesture(_gesture(500, GesturePhase.ENDING))  # 여전히 cooldown 안(< 300+1000)
    assert second is not None
    assert not second.committed
    assert "cooldown" in second.reason


def test_cooldown_expires_after_configured_duration() -> None:
    engine = _locked_engine(_fusion_config(cooldown_ms=200))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    first = engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert first is not None and first.committed  # cooldown until 500

    engine.push_target(_target(600))  # cooldown 경과 후 프레임이 만료를 트리거
    engine.push_gesture(_gesture(700, GesturePhase.ONSET))
    second = engine.push_gesture(_gesture(900, GesturePhase.ENDING))
    assert second is not None
    assert second.committed


# --- phase 프로퍼티 ---


def test_phase_idle_when_nothing_locked() -> None:
    engine = FusionEngine(_fusion_config(), _alignment_config())
    assert engine.phase == IntentPhase.IDLE


def test_phase_target_candidate_before_dwell() -> None:
    engine = FusionEngine(_fusion_config(), _alignment_config(target_dwell_ms=500))
    engine.push_target(_target(0))
    assert engine.phase == IntentPhase.TARGET_CANDIDATE


def test_phase_target_locked_without_gesture() -> None:
    engine = _locked_engine()
    assert engine.phase == IntentPhase.TARGET_LOCKED


def test_phase_gesture_tracking_during_onset() -> None:
    engine = _locked_engine()
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    assert engine.phase == IntentPhase.GESTURE_TRACKING


def test_phase_cooldown_after_commit() -> None:
    engine = _locked_engine(_fusion_config(cooldown_ms=1000))
    engine.push_gesture(_gesture(100, GesturePhase.ONSET))
    engine.push_gesture(_gesture(300, GesturePhase.ENDING))
    assert engine.phase == IntentPhase.COOLDOWN
