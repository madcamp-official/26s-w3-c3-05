"""웹캠에서 라벨된 제스처 클립을 녹화해 Jester와 같은 캐시 포맷으로 저장한다.

파인튜닝 단계(Phase 5)의 데이터 수집 도구다. `gaze_samples.py`(atomic-write·저장
전 검증 패턴)와 같은 원칙: 클립 하나가 완결될 때만(SPACE로 시작→SPACE로 종료)
캐시에 쓴다. 손 검출 실패 프레임 허용 정책은 `extract_jester.py`와 동일하다
(2026-07-20): 미검출 프레임 비율이 `TrainingConfig.max_missing_frame_fraction`을
넘을 때만 클립을 버린다 — 웹캠 클립은 사람이 직접 다시 녹화해야 하는 만큼
Jester보다 한 프레임 실패로 통째로 버리는 비용이 더 크다.

저장 경로는 `<cache_dir>/webcam/<person_id>/*.npz`다 — `training/train.py --stage
finetune`이 `--train-persons`/`--val-persons`로 사람 단위 split을 할 때 이 폴더
구조를 그대로 쓴다.

키 조작: SPACE=녹화 시작/종료, g=제스처 변경, q 또는 ESC=종료.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS
from training.clip_recorder import ClipRecorder
from training.config import DEFAULT_TRAINING_CONFIG


def _prompt_gesture_label() -> str:
    print("녹화할 제스처를 선택하세요:")
    for i, label in enumerate(DEFAULT_GESTURE_LABELS):
        print(f"  {i}: {label}")
    while True:
        raw = input("번호 입력: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(DEFAULT_GESTURE_LABELS):
            return DEFAULT_GESTURE_LABELS[int(raw)]
        print("잘못된 입력입니다.")


def run(
    person_id: str,
    model_path: Path,
    cache_dir: Path,
    camera_index: int = 0,
    max_missing_frame_fraction: float = DEFAULT_TRAINING_CONFIG.max_missing_frame_fraction,
    target_fps: float | None = 12.0,
) -> None:
    # 녹화 규칙(누적·미검출 게이트·fps 정규화·저장)은 GUI와 공유하는 ClipRecorder에 있다.
    recorder = ClipRecorder(
        cache_dir=cache_dir,
        max_missing_frame_fraction=max_missing_frame_fraction,
        target_fps=target_fps,
    )

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"camera index {camera_index} could not be opened")

    # 세션 전체(여러 클립에 걸쳐)에서 단조 증가하는 timestamp — extract_jester.py와
    # 같은 이유(detect_for_video의 monotonic 제약, 랜드마커 인스턴스 재사용).
    session_start = time.monotonic()
    frame_id = 0
    gesture_label = _prompt_gesture_label()

    print("SPACE=녹화 시작/종료, g=제스처 변경, q/ESC=종료")

    try:
        with MediaPipeHandLandmarker(model_path) as landmarker:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    continue
                timestamp_ms = int((time.monotonic() - session_start) * 1000)
                # mediapipe_hands.py 색상 규약: process()는 RGB를 기대한다.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                display = bgr.copy()
                status = (
                    f"[{gesture_label}] {'REC' if recorder.is_recording else 'idle'} "
                    f"person={person_id} frames={recorder.frame_count}"
                )
                color = (0, 0, 255) if recorder.is_recording else (200, 200, 200)
                cv2.putText(display, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("record_webcam_clips", display)

                if recorder.is_recording:
                    observation = landmarker.process(rgb, timestamp_ms, frame_id)
                    frame_id += 1
                    recorder.add(observation)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if not recorder.is_recording:
                        recorder.start(person_id, gesture_label)
                        frame_id = 0
                        print("녹화 시작")
                    else:
                        print(recorder.stop().detail)
                elif key == ord("g") and not recorder.is_recording:
                    gesture_label = _prompt_gesture_label()
                elif key in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-id", required=True)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_TRAINING_CONFIG.models_dir / "hand_landmarker.task"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.cache_dir)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--max-missing-frame-fraction",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.max_missing_frame_fraction,
        help="클립 내 미검출 프레임 비율이 이 값을 넘으면 클립을 버린다(기본 %(default)s)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=12.0,
        help=(
            "저장 직전 클립을 이 fps로 리샘플해 pretrain(Jester 12fps)과 정합시킨다. "
            "0 이하면 리샘플하지 않고 캡처 fps 그대로 저장(기본 %(default)s)."
        ),
    )
    args = parser.parse_args(argv)

    run(
        args.person_id,
        args.model,
        args.cache_dir,
        args.camera_index,
        max_missing_frame_fraction=args.max_missing_frame_fraction,
        target_fps=args.target_fps if args.target_fps > 0.0 else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
