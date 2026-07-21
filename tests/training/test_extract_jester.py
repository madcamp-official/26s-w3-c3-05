"""extract_jester.py의 클립 단위 미검출 허용 로직(_extract_clip)을 검증한다.

실제 MediaPipe 모델은 쓰지 않는다 — `_extract_clip`은 `.process(rgb, ts, frame_id)`를
가진 아무 객체나 받으므로(덕 타이핑), frame_id로 검출 성공/실패를 제어하는 스텁을
쓴다. 프레임 파일은 `cv2.imread`가 실제로 읽어야 하므로 작은 더미 JPG를 저장한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS  # noqa: E402
from jarvis.gesture_fusion.landmarks import HandObservation  # noqa: E402
from training.data.jester_manifest import JesterClipRef  # noqa: E402
from training.extract.extract_jester import _extract_clip  # noqa: E402


class _StubLandmarker:
    """frame_id 집합으로 검출 성공/실패를 제어하는 가짜 랜드마커."""

    def __init__(self, missing_frame_ids: set[int]) -> None:
        self._missing_frame_ids = missing_frame_ids

    def process(self, rgb_frame: object, timestamp_ms: int, frame_id: int) -> HandObservation:
        detected = frame_id not in self._missing_frame_ids
        landmarks = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
        return HandObservation(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            landmarks=landmarks,
            handedness="Right" if detected else "",
            palm_scale=0.2 if detected else 0.0,
            detection_confidence=0.9 if detected else 0.0,
            handedness_score=0.9 if detected else 0.0,
            hand_detected=detected,
            wrist_position=np.zeros(LANDMARK_DIMS, dtype=np.float64),
        )


def _write_dummy_frames(frames_dir: Path, count: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(count):
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), image)


def _ref(frames_dir: Path, clip_id: str = "clip-a", our_label: str = "swipe_down") -> JesterClipRef:
    return JesterClipRef(clip_id=clip_id, frames_dir=frames_dir, our_label=our_label, split="train")


def test_clip_kept_when_missing_fraction_within_tolerance(tmp_path: Path) -> None:
    _write_dummy_frames(tmp_path, count=10)
    landmarker = _StubLandmarker(missing_frame_ids={2})  # 10개 중 1개만 미검출 = 10%
    result, observations, _ = _extract_clip(landmarker, _ref(tmp_path), 0.0, max_missing_frame_fraction=0.3)

    assert result.status == "ok"
    assert result.missing_frame_count == 1
    assert len(observations) == 10
    assert observations[2].hand_detected is False
    assert all(o.hand_detected for i, o in enumerate(observations) if i != 2)


def test_clip_rejected_when_missing_fraction_exceeds_tolerance(tmp_path: Path) -> None:
    _write_dummy_frames(tmp_path, count=10)
    landmarker = _StubLandmarker(missing_frame_ids={0, 1, 2, 3, 4})  # 50% 미검출
    result, observations, _ = _extract_clip(landmarker, _ref(tmp_path), 0.0, max_missing_frame_fraction=0.3)

    assert result.status == "too_many_missed_frames"
    assert result.missing_frame_count == 5
    assert observations == []


def test_fully_detected_clip_has_zero_missing_frames(tmp_path: Path) -> None:
    _write_dummy_frames(tmp_path, count=6)
    landmarker = _StubLandmarker(missing_frame_ids=set())
    result, observations, _ = _extract_clip(landmarker, _ref(tmp_path), 0.0, max_missing_frame_fraction=0.3)

    assert result.status == "ok"
    assert result.missing_frame_count == 0
    assert len(observations) == 6


def test_fully_missing_clip_is_rejected_even_at_high_tolerance(tmp_path: Path) -> None:
    """미검출 비율이 정확히 1.0이면 max_missing_frame_fraction<1.0인 한 항상 제외된다."""
    _write_dummy_frames(tmp_path, count=5)
    landmarker = _StubLandmarker(missing_frame_ids={0, 1, 2, 3, 4})
    result, observations, _ = _extract_clip(landmarker, _ref(tmp_path), 0.0, max_missing_frame_fraction=0.9)

    assert result.status == "too_many_missed_frames"
    assert observations == []


def test_zero_tolerance_reproduces_old_any_failure_excludes_clip_behavior(tmp_path: Path) -> None:
    _write_dummy_frames(tmp_path, count=8)
    landmarker = _StubLandmarker(missing_frame_ids={7})  # 마지막 프레임 하나만 실패
    result, observations, _ = _extract_clip(landmarker, _ref(tmp_path), 0.0, max_missing_frame_fraction=0.0)

    assert result.status == "too_many_missed_frames"
    assert observations == []


@pytest.mark.parametrize("exempt_label", ["none", "doing_other_things"])
def test_exempt_labels_ignore_missing_fraction_limit(tmp_path: Path, exempt_label: str) -> None:
    """"none"·"doing_other_things"는 미검출 비율이 100%여도 제외되지 않는다.

    둘 다 정의상 손이 안 보이는 장면이 많아 다른 클래스와 같은 기준을 적용하면
    대부분 걸러진다(2026-07-20 실측: "none"은 validation 533개 중 17개=3.2%,
    "doing_other_things"는 train 9592개 중 4454개=46.4%만 생존) — 미검출 프레임
    자체는 IGNORE_INDEX로 마스킹되니 배제할 필요가 없다.
    """
    _write_dummy_frames(tmp_path, count=10)
    landmarker = _StubLandmarker(missing_frame_ids=set(range(10)))  # 100% 미검출
    result, observations, _ = _extract_clip(
        landmarker, _ref(tmp_path, our_label=exempt_label), 0.0, max_missing_frame_fraction=0.3
    )

    assert result.status == "ok"
    assert result.missing_frame_count == 10
    assert len(observations) == 10


def test_no_frames_short_circuits_before_missing_fraction_check(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    landmarker = _StubLandmarker(missing_frame_ids=set())
    result, observations, _ = _extract_clip(
        landmarker, _ref(empty_dir), 0.0, max_missing_frame_fraction=0.3
    )

    assert result.status == "no_frames"
    assert observations == []
