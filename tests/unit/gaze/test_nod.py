"""고개 끄덕임 검출(nod.py): dip→recovery 패턴, baseline 추적, 지속적 자세 변화와의 구분."""

from __future__ import annotations

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.nod import NodDetector


def test_first_frame_only_seeds_baseline() -> None:
    detector = NodDetector()
    assert detector.update(0.0, 0) is False
    assert detector.baseline_pitch_deg == 0.0


def test_simple_dip_and_recovery_detects_nod() -> None:
    detector = NodDetector(GazeConfig(nod_dip_threshold_deg=8.0, nod_recovery_deg=5.0))
    detector.update(0.0, 0)  # baseline = 0
    assert detector.update(-10.0, 100) is False  # dip starts
    assert detector.update(-12.0, 150) is False  # deeper
    assert detector.update(-4.0, 250) is True  # recovered 8 from extremum -12


def test_shallow_dip_does_not_trigger() -> None:
    detector = NodDetector(GazeConfig(nod_dip_threshold_deg=8.0))
    detector.update(0.0, 0)
    assert detector.update(-3.0, 100) is False
    assert detector.update(1.0, 200) is False  # never entered dip state


def test_sustained_head_down_times_out_without_false_nod() -> None:
    """빠른 끄덕임이 아니라 계속 고개를 숙이고 있는 상태는 끄덕임으로 세지 않는다."""
    config = GazeConfig(nod_dip_threshold_deg=8.0, nod_max_duration_ms=900)
    detector = NodDetector(config)
    detector.update(0.0, 0)
    assert detector.update(-10.0, 100) is False
    # 900ms를 넘겨도 계속 낮게 유지 — 자세 변화로 보고 포기, 새 baseline 재기준.
    assert detector.update(-10.0, 1200) is False
    assert detector.baseline_pitch_deg == -10.0
    # 재기준 이후 그대로 있으면(더 이상 dip이 아님) 계속 False.
    assert detector.update(-10.0, 1300) is False


def test_baseline_slowly_tracks_gradual_posture_shift() -> None:
    config = GazeConfig(nod_baseline_decay=0.5)
    detector = NodDetector(config)
    detector.update(0.0, 0)
    detector.update(2.0, 100)  # below dip threshold, baseline moves toward 2.0
    assert detector.baseline_pitch_deg == 1.0


def test_baseline_frozen_during_dip() -> None:
    """dip 중에는 baseline을 갱신하지 않아 끄덕임 자체가 새 기준으로 흡수되지 않는다."""
    config = GazeConfig(nod_dip_threshold_deg=8.0, nod_baseline_decay=0.5)
    detector = NodDetector(config)
    detector.update(0.0, 0)
    detector.update(-10.0, 100)  # dip started; baseline must stay 0 internally
    detector.update(-6.0, 200)  # partial recovery, not yet enough (needs +5 from -10 extremum)
    # recovered only 4 from extremum(-10) -> not yet True, still no baseline change
    assert detector.baseline_pitch_deg == 0.0


def test_deepening_dip_tracks_extremum_for_recovery_threshold() -> None:
    config = GazeConfig(nod_dip_threshold_deg=8.0, nod_recovery_deg=5.0)
    detector = NodDetector(config)
    detector.update(0.0, 0)
    detector.update(-10.0, 100)
    detector.update(-20.0, 150)  # extremum deepens to -20
    # recovering to -10 is only +10 from -20 extremum... wait +10 >= 5, should trigger
    assert detector.update(-10.0, 200) is True


def test_reset_clears_baseline_and_dip_state() -> None:
    detector = NodDetector()
    detector.update(0.0, 0)
    detector.update(-10.0, 100)
    detector.reset()
    assert detector.baseline_pitch_deg is None
    assert detector.update(-10.0, 200) is False  # re-seeds baseline instead of dipping
    assert detector.baseline_pitch_deg == -10.0
