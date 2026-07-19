"""기존 방식 vs 이식(참고) 방식 랜드마크 A/B 비교 — 좌우 나란히 카메라 표시.

`python -m jarvis.reference_port.compare`

같은 카메라 프레임을 두 파이프라인에 흘려 각각 검출한 손 랜드마크를 좌/우 패널에
그린다. 관심사는 **랜드마크 검출/표시 품질**뿐이다(손모양 분류는 대상 아님). 두
방식 모두 랜드마크 **엔진**은 이 프로젝트의 Tasks API HandLandmarker로 동일하다
(참고 레포의 레거시 엔진은 이식 불가). 실제로 다른 것은:

- 왼쪽 "기존 방식": 검출 신뢰도 0.5 + 우리 파이프라인의 **One-Euro 평활화** 적용.
- 오른쪽 "참고 방식": 검출 신뢰도 0.7 + **raw**(평활화 없음, 참고 레포처럼 그대로).

즉 "검출 엔진 우열"이 아니라 **설정(신뢰도)·평활화가 랜드마크 표시에 어떻게
보이는가**를 비교하는 도구다. 표시는 둘 다 거울상(좌우 반전), ESC로 종료.

의존성은 이 프로젝트 것만 쓴다(cv2·mediapipe·numpy + jarvis 내부). 다른 프로젝트
파일은 수정하지 않는다 — 이 모듈은 gesture_fusion의 OneEuroFilter를 import만 한다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision import HandLandmarker

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG
from jarvis.gesture_fusion.smoothing import OneEuroFilter

_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _find_model(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for c in [
        Path("models/hand_landmarker.task"),
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
    ]:
        if c.is_file():
            return c
    return None


def _make_landmarker(model_path: Path, min_detection: float) -> "HandLandmarker":
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _detect_points(landmarker: object, rgb: npt.NDArray[np.uint8], ts_ms: int) -> npt.NDArray[np.float64] | None:
    """한 프레임 검출 → (21, 2) 이미지 좌표[0,1] 또는 손 없으면 None."""
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat as MpImageFormat

    result = landmarker.detect_for_video(  # type: ignore[attr-defined]
        MpImage(image_format=MpImageFormat.SRGB, data=rgb), ts_ms
    )
    if not result.hand_landmarks:
        return None
    lms = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y] for lm in lms], dtype=np.float64)


def _draw_panel(
    frame_bgr: npt.NDArray[np.uint8],
    points01: npt.NDArray[np.float64] | None,
    *,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    label: str | None,
) -> npt.NDArray[np.uint8]:
    """프레임 복사본을 거울상으로 뒤집고 랜드마크·헤더를 그려 반환한다(표시 전용)."""
    disp = cast("npt.NDArray[np.uint8]", cv2.flip(frame_bgr, 1))
    h, w = disp.shape[:2]
    if points01 is not None:
        px = [(int((1.0 - x) * w), int(y * h)) for x, y in points01]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(disp, px[a], px[b], color, 2, cv2.LINE_AA)
        for x, y in px:
            cv2.circle(disp, (x, y), 3, (240, 240, 240), -1)
    # 헤더 배너
    cv2.rectangle(disp, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(disp, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(disp, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    status = label if label is not None else ("hand" if points01 is not None else "no hand")
    cv2.putText(disp, status, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return disp


def run(device: int, model_path: Path, *, selftest_frames: int = 0) -> int:
    left_lm = _make_landmarker(model_path, min_detection=0.5)   # 기존 방식 설정
    right_lm = _make_landmarker(model_path, min_detection=0.7)  # 참고 방식 설정
    # 기존 방식의 표시 평활화(우리 파이프라인과 동일 파라미터). 이미지 좌표 (21,2)에 적용.
    smoother = OneEuroFilter(
        min_cutoff=DEFAULT_GESTURE_CONFIG.smoothing_min_cutoff,
        beta=DEFAULT_GESTURE_CONFIG.smoothing_beta,
        d_cutoff=DEFAULT_GESTURE_CONFIG.smoothing_d_cutoff,
    )

    if selftest_frames > 0:
        # 헤드리스 자가검증: 합성(빈) 프레임으로 두 파이프라인이 크래시 없이 도는지만 확인.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
        for i in range(selftest_frames):
            lp = _detect_points(left_lm, rgb, i * 40 + 1)
            rp = _detect_points(right_lm, rgb, i * 40 + 1)
            left = _draw_panel(blank, lp, title="a", subtitle="b", color=(80, 200, 80), label=None)
            right = _draw_panel(blank, rp, title="a", subtitle="b", color=(80, 180, 240), label="x")
            combined = cv2.hconcat([left, right])
        print(f"[selftest] {selftest_frames} 프레임 OK · 합성 크기 {combined.shape} · 손검출(빈프레임) 좌:{lp is not None} 우:{rp is not None}")
        left_lm.close()
        right_lm.close()
        return 0

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"카메라 {device}번을 열 수 없습니다.")
        return 1
    start = time.monotonic()
    last_ts = -1
    print("A/B 비교 실행 중 (좌: 기존 0.5+One-Euro / 우: 참고 0.7+raw) — ESC 종료")
    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok:
                break
            frame = cast("npt.NDArray[np.uint8]", frame_raw)
            rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts

            left_raw = _detect_points(left_lm, rgb, ts)
            left_pts = None
            if left_raw is not None:
                left_pts = smoother.filter(left_raw, ts)  # 우리 방식: 평활화
            else:
                smoother.reset()

            right_pts = _detect_points(right_lm, rgb, ts)  # 참고 방식: raw (평활화 없음)

            left_panel = _draw_panel(
                frame, left_pts, title="기존 방식", subtitle="Tasks API 0.5 + One-Euro 평활화",
                color=(80, 200, 80), label=None,
            )
            right_panel = _draw_panel(
                frame, right_pts, title="참고 방식", subtitle="Tasks API 0.7 + raw (평활화 없음)",
                color=(80, 180, 240), label=None,
            )
            cv2.imshow("landmark A/B  (left=기존 / right=참고)  ESC=quit", cv2.hconcat([left_panel, right_panel]))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        left_lm.close()
        right_lm.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="랜드마크 A/B 비교 (기존 vs 이식)")
    parser.add_argument("--device", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--model", type=str, default=None, help="hand_landmarker.task 경로")
    parser.add_argument("--selftest", type=int, default=0, help="N>0이면 카메라 없이 헤드리스 자가검증")
    args = parser.parse_args()
    model_path = _find_model(args.model)
    if model_path is None:
        print("hand_landmarker.task 모델을 찾지 못했습니다. --model 로 경로를 지정하세요.")
        return 1
    return run(args.device, model_path, selftest_frames=args.selftest)


if __name__ == "__main__":
    raise SystemExit(main())
