"""웹캠에서 라벨된 제스처 클립을 녹화해 Jester와 같은 캐시 포맷으로 저장한다.

파인튜닝 단계(Phase 5)의 데이터 수집 도구다. `gaze_samples.py`(atomic-write·저장
전 검증 패턴)와 같은 원칙: 클립 하나가 완결될 때만(SPACE로 시작→SPACE로 종료)
캐시에 쓰고, 녹화 도중 손 검출이 실패한 프레임이 하나라도 있으면 저장하지 않는다
— `extract_jester.py`와 동일한 정책(학습 파이프라인 인터뷰 결정: 검출 실패
클립 전체 제외).

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

from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS
from training.config import DEFAULT_TRAINING_CONFIG
from training.data.clip_cache import observations_to_cached_clip, save_clip


def _prompt_gesture_label() -> str:
    print("녹화할 제스처를 선택하세요:")
    for i, label in enumerate(DEFAULT_GESTURE_LABELS):
        print(f"  {i}: {label}")
    while True:
        raw = input("번호 입력: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(DEFAULT_GESTURE_LABELS):
            return DEFAULT_GESTURE_LABELS[int(raw)]
        print("잘못된 입력입니다.")


def run(person_id: str, model_path: Path, cache_dir: Path, camera_index: int = 0) -> None:
    out_dir = cache_dir / "webcam" / person_id
    clip_counter = len(sorted(out_dir.glob("*.npz"))) if out_dir.exists() else 0

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"camera index {camera_index} could not be opened")

    # 세션 전체(여러 클립에 걸쳐)에서 단조 증가하는 timestamp — extract_jester.py와
    # 같은 이유(detect_for_video의 monotonic 제약, 랜드마커 인스턴스 재사용).
    session_start = time.monotonic()
    recording = False
    observations: list[HandObservation] = []
    frame_id = 0
    dropped_this_clip = False
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
                    f"[{gesture_label}] {'REC' if recording else 'idle'} "
                    f"person={person_id} clip#{clip_counter}"
                )
                color = (0, 0, 255) if recording else (200, 200, 200)
                cv2.putText(display, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("record_webcam_clips", display)

                if recording:
                    observation = landmarker.process(rgb, timestamp_ms, frame_id)
                    frame_id += 1
                    if not observation.hand_detected:
                        dropped_this_clip = True
                    observations.append(observation)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if not recording:
                        recording = True
                        observations = []
                        frame_id = 0
                        dropped_this_clip = False
                        print("녹화 시작")
                    else:
                        recording = False
                        if dropped_this_clip:
                            print(f"손 미검출 프레임이 있어 이 클립을 버립니다 ({len(observations)}프레임 중).")
                        elif not observations:
                            print("빈 클립 — 저장하지 않음")
                        else:
                            clip_id = f"{person_id}-{gesture_label}-{clip_counter:04d}"
                            clip = observations_to_cached_clip(observations, gesture_label, clip_id)
                            save_clip(out_dir / f"{clip_id}.npz", clip)
                            print(f"저장: {clip_id} ({len(observations)}프레임)")
                            clip_counter += 1
                elif key == ord("g") and not recording:
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
    args = parser.parse_args(argv)

    run(args.person_id, args.model, args.cache_dir, args.camera_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
