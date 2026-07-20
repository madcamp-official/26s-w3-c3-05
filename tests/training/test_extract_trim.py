"""`training.extract.extract_jester._trim_and_classify`의 가장자리 트리밍 정책을 검증한다.

MediaPipe·실제 프레임 없이 `HandObservation` 시퀀스를 직접 구성해 분류 규칙만 본다
(2026-07-19 검출 수율 완화: 앞뒤 연속 미검출만 잘라내고 내부 전검출 클립만 채택).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation
from training.data.jester_manifest import JesterClipRef
from training.extract.extract_jester import _trim_and_classify


def _obs(frame_id: int, detected: bool) -> HandObservation:
    landmarks = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    if detected:
        landmarks[:, 0] = 0.1 + 0.01 * frame_id
    return HandObservation(
        timestamp_ms=1000 + frame_id * 83,
        frame_id=frame_id,
        landmarks=landmarks,
        handedness="Right" if detected else "",
        palm_scale=0.2 if detected else 0.0,
        detection_confidence=0.9 if detected else 0.0,
        handedness_score=0.9 if detected else 0.0,
        hand_detected=detected,
        wrist_position=np.array([0.05, -0.03], dtype=np.float64),
    )


def _ref() -> JesterClipRef:
    return JesterClipRef(
        clip_id="c1", frames_dir=Path("unused"), our_label="swipe_up", split="train"
    )


def _seq(pattern: str) -> list[HandObservation]:
    """'.'=미검출, 'x'=검출 패턴으로 관측값 시퀀스를 만든다."""
    return [_obs(i, ch == "x") for i, ch in enumerate(pattern)]


def test_trims_leading_and_trailing_and_keeps_clean_core() -> None:
    result, core = _trim_and_classify(_seq("..xxxxxx.."), _ref(), min_frames=4)
    assert result.status == "ok"
    assert result.frame_count == 10
    assert result.kept_frames == 6
    assert len(core) == 6
    assert all(o.hand_detected for o in core)  # 유지된 core는 전부 검출


def test_all_detected_clip_is_kept_whole() -> None:
    result, core = _trim_and_classify(_seq("xxxxx"), _ref(), min_frames=4)
    assert result.status == "ok"
    assert result.kept_frames == 5


def test_interior_gap_is_rejected_not_filled() -> None:
    # 트리밍 후에도 가운데가 뚫려 있으면 좌표를 지어내지 않고 제외한다.
    result, core = _trim_and_classify(_seq("xx..xx"), _ref(), min_frames=2)
    assert result.status == "interior_gap"
    assert core == []


def test_fully_undetected_clip_is_no_hand() -> None:
    result, core = _trim_and_classify(_seq("....."), _ref(), min_frames=2)
    assert result.status == "no_hand"
    assert core == []


def test_too_short_after_trim_is_rejected() -> None:
    result, core = _trim_and_classify(_seq(".xxx."), _ref(), min_frames=8)
    assert result.status == "too_short"
    assert result.frame_count == 5
    assert core == []
