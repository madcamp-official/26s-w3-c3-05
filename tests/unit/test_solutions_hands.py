"""참고 레포 방식 랜드마크 어댑터(`solutions_hands`) 검증.

`vision` extra(레거시 solutions가 살아있는 mediapipe)가 없으면 통째로 skip한다 —
`mediapipe_hands`와 같은 optional-extra 경계 처리다.

여기서 확인하는 것은 **어댑터가 downstream 계약을 지키는가**다: 손이 없으면 추적
손실을 정직하게 보고하는가, 검출되면 21점 이미지 좌표와 정규화된 HandObservation을
같은 규약으로 내는가. mediapipe 모델의 정확도 자체는 검증 대상이 아니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe", reason="vision extra 미설치")

mp = pytest.importorskip("mediapipe")
if not hasattr(mp, "solutions"):  # pragma: no cover - 설치된 휠에 따라 달라진다
    pytest.skip("설치된 mediapipe에 레거시 solutions가 없음", allow_module_level=True)

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS  # noqa: E402
from jarvis.gesture_fusion.solutions_hands import (  # noqa: E402
    SolutionsHandDetector,
    SolutionsHandLandmarker,
)


#: 실제 손이 찍힌 샘플 이미지 경로. 저작권 있는 사진을 저장소에 넣지 않으려고 커밋
#: 대신 환경변수로 받는다 — 없으면 손 검출이 필요한 테스트만 skip된다(손 없는
#: 프레임으로 하는 계약 검증은 샘플 없이도 전부 돈다).
_HAND_IMAGE_ENV = "JARVIS_TEST_HAND_IMAGE"


def _blank_rgb() -> np.ndarray:
    """손이 없는 프레임 — 검출은 반드시 실패해야 한다."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def hand_rgb() -> np.ndarray:
    """샘플 손 사진을 RGB로 읽어온다. 경로가 없으면 skip."""
    import os

    path = os.environ.get(_HAND_IMAGE_ENV)
    if not path or not Path(path).is_file():
        pytest.skip(f"{_HAND_IMAGE_ENV} 에 손 샘플 이미지 경로를 지정하면 실행된다")
    cv2 = pytest.importorskip("cv2", reason="vision extra 미설치")
    bgr = cv2.imread(path)
    if bgr is None:
        pytest.skip(f"이미지를 읽지 못했다: {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def test_detector_reports_no_hand_on_blank_frame() -> None:
    with SolutionsHandDetector() as detector:
        assert detector.detect(_blank_rgb()) is None


def test_landmarker_reports_tracking_loss_instead_of_inventing_pose() -> None:
    """손이 없으면 지어낸 좌표가 아니라 hand_detected=False를 돌려준다."""
    with SolutionsHandLandmarker() as landmarker:
        observation = landmarker.process(_blank_rgb(), timestamp_ms=100, frame_id=7)
    assert observation.hand_detected is False
    assert observation.timestamp_ms == 100
    assert observation.frame_id == 7


def test_landmarker_passes_timestamp_and_frame_id_through() -> None:
    """Solutions는 타임스탬프를 요구하지 않지만 관측값에는 그대로 실려야 한다."""
    with SolutionsHandLandmarker() as landmarker:
        first = landmarker.process(_blank_rgb(), timestamp_ms=0, frame_id=0)
        second = landmarker.process(_blank_rgb(), timestamp_ms=33, frame_id=1)
    assert (first.timestamp_ms, first.frame_id) == (0, 0)
    assert (second.timestamp_ms, second.frame_id) == (33, 1)


def test_detector_returns_21_normalized_points_for_a_real_hand(hand_rgb: np.ndarray) -> None:
    """실제 손 이미지에서 21점을 이미지 정규화 좌표 [0, 1]로 낸다."""
    with SolutionsHandDetector() as detector:
        detection = detector.detect(hand_rgb)

    assert detection is not None, "샘플 손 이미지에서 검출에 실패했다"
    assert detection.points.shape == (HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    assert np.all((detection.points >= 0.0) & (detection.points <= 1.0))
    assert detection.handedness in {"Left", "Right"}
    assert 0.0 < detection.handedness_score <= 1.0


def test_landmarker_normalizes_a_real_hand(hand_rgb: np.ndarray) -> None:
    """검출된 손은 손목 원점·palm_scale 정규화를 거쳐 나온다(공용 normalize_hand 재사용)."""
    with SolutionsHandLandmarker() as landmarker:
        observation = landmarker.process(hand_rgb, timestamp_ms=0, frame_id=0)

    assert observation.hand_detected is True
    assert observation.landmarks.shape == (HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    # 손목(0번)이 원점으로 옮겨진 것이 정규화 규약이다.
    np.testing.assert_allclose(observation.landmarks[0], np.zeros(LANDMARK_DIMS), atol=1e-9)
    assert observation.palm_scale > 0.0
