"""참고 레포 방법론 이식 라이브 데모 — `python -m jarvis.reference_port.demo`.

이 프로젝트의 **Tasks API HandLandmarker**로 랜드마크를 뽑고, 참고 레포의
**max-abs 정규화 + 학습된 KeyPoint 분류기**로 정적 손모양(Open/Close/Pointer)을
매 프레임 분류해 웹캠 위에 오버레이한다. 목적은 "참고 레포 방법론이 우리 스택
(같은 엔진) 위에서도 잘 인식하는가"를 눈으로 A/B 확인하는 것이다.

표시는 디버그 툴과 같은 거울상(좌우 반전)이며, 랜드마크·분류 입력은 반전하지 않은
원본을 쓴다(표시 전용 반전). ESC로 종료.

주의: 이건 실험 데모다. 분류 대상은 참고 레포가 학습한 3개 정적 손모양뿐이며,
이 프로젝트의 동적 제스처(swipe/rotate) 파이프라인과는 별개다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from jarvis.reference_port import ReferenceKeyPointClassifier, preprocess_landmark_max_abs

# 참고 레포 기본 검출 신뢰도(0.7) — 우리 기본(0.5)보다 높다. 이식 비교 변수 중 하나.
_REF_MIN_DETECTION_CONFIDENCE = 0.7
_REF_MIN_TRACKING_CONFIDENCE = 0.5

_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _find_model() -> Path | None:
    """hand_landmarker.task를 흔한 위치에서 찾는다."""
    candidates = [
        Path("models/hand_landmarker.task"),
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="참고 레포 방법론 이식 라이브 데모")
    parser.add_argument("--device", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--model", type=str, default=None, help="hand_landmarker.task 경로")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _find_model()
    if model_path is None or not model_path.is_file():
        print("hand_landmarker.task 모델을 찾지 못했습니다. --model 로 경로를 지정하세요.")
        return 1

    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat as MpImageFormat
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
        min_hand_detection_confidence=_REF_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=_REF_MIN_TRACKING_CONFIDENCE,
    )
    landmarker = HandLandmarker.create_from_options(options)
    classifier = ReferenceKeyPointClassifier()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"카메라 {args.device}번을 열 수 없습니다.")
        landmarker.close()
        return 1

    start = time.monotonic()
    last_ts = -1
    print("참고 방법론 이식 데모 실행 중 — ESC로 종료.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(
                MpImage(image_format=MpImageFormat.SRGB, data=rgb), ts
            )

            display = cv2.flip(frame, 1)  # 거울상(표시 전용)
            h, w = display.shape[:2]
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pts = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float64)  # (21,2) [0,1]
                pred = classifier.classify_vector(preprocess_landmark_max_abs(pts))
                _draw_hand(display, pts, mirror=True)
                _draw_label(display, f"{pred.label}  {pred.confidence:.0%}")
            else:
                _draw_label(display, "no hand")

            cv2.imshow("reference-port demo (ESC=quit)", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


def _draw_hand(frame: np.ndarray, pts01: np.ndarray, *, mirror: bool) -> None:
    h, w = frame.shape[:2]
    px = [(int((1.0 - x) * w) if mirror else int(x * w), int(y * h)) for x, y in pts01]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (80, 200, 80), 2, cv2.LINE_AA)
    for x, y in px:
        cv2.circle(frame, (x, y), 3, (235, 235, 235), -1)


def _draw_label(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, f"[ref-method] {text}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 120), 2, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
