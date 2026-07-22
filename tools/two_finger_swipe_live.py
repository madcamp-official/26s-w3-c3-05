"""두 손가락 좌↔우 전이 스와이프 감지를 라이브로 관찰하는 툴(실행 없음).

정적 two_fingers 방향 전이로 데스크톱 전환을 인식하는 규칙(`pose_state._track_swipe`:
수평 방향 부호가 한쪽에서 반대쪽으로 넘어가면 즉시 스와이프)이 실제 손동작에서 잘 잡히는지
눈으로 확인하기 위한 관찰 전용 툴이다.

`HandProbe`가 내부에 이미 `PoseStateMachine`을 돌려 `snapshot.pose_events`로 이벤트를
내보내므로(스크롤·볼륨·좌우 스와이프 포함), 이 툴은 그 이벤트를 **읽어서 표시만** 한다 —
`PoseControlBridge`를 붙이지 않으므로 실제 데스크톱은 전환되지 않는다(안전하게 튜닝 가능).

실행:
  python tools/two_finger_swipe_live.py \
    --hand-model <repo>/models/hand_landmarker.task \
    --pose-model <repo>/models/hand_pose_classifier.pt
  q 또는 ESC로 종료.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jarvis.gesture_fusion.pose_state import MIN_VERTICALITY, pointing_direction


def _draw(frame: np.ndarray, *, pose: str, trusted: bool,
          direction: tuple[float, float] | None,
          counts: Counter, last_swipe: str, flash: bool) -> None:
    import cv2

    y = 26
    def line(txt: str, color: tuple[int, int, int], size: float = 0.6) -> None:
        nonlocal y
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
        y += 28

    line("좌우 스와이프 라이브 관찰 (실행 안 함)  q/ESC:종료", (60, 220, 120), 0.5)
    if direction is None:
        line("두 손가락 미검출", (120, 120, 240))
    else:
        dx, dy = direction
        # 판별 기준은 스크롤과 동일: |dy|≥MIN_VERTICALITY면 '수직(좌우 판별 안 함)'.
        if abs(dy) >= MIN_VERTICALITY:
            committed = "수직(스크롤)"
        else:
            committed = "왼쪽" if dx > 0 else "오른쪽"
        pose_txt = f"{pose or '-'}{'' if trusted else '(거부)'}"
        line(f"pose={pose_txt}  dx={dx:+.2f} dy={dy:+.2f}  판정={committed}", (80, 240, 240))
    line(
        f"스와이프  이전(prev)={counts.get('desktop_prev', 0)}  다음(next)={counts.get('desktop_next', 0)}",
        (230, 200, 90),
    )
    if last_swipe:
        color = (60, 120, 255) if flash else (200, 200, 200)
        line(f"최근: {last_swipe}", color, 0.7)


def run(camera: int, hand_model: Path, pose_model: Path) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print('opencv 미설치: pip install -e ".[ui]"')
        return 2
    from jarvis.monitoring.hand_probe import HandProbe

    probe = HandProbe(model_path=hand_model, pose_model_path=pose_model)
    if not probe.start():
        print(f"hand 프로브 비활성: {probe.status_text}")
        return 2
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"카메라 {camera}번을 열 수 없습니다.")
        probe.close()
        return 2

    counts: Counter = Counter()
    last_swipe = ""
    flash_until = 0.0
    start = time.monotonic()
    last_ts = -1
    frame_id = 0
    print("관찰 시작 — 두 손가락을 좌↔우로 넘겨 보세요(데스크톱은 바뀌지 않음). q로 종료.")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue
            now = time.monotonic()
            ts = max(int((now - start) * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            snapshot = probe.process_rgb(rgb, ts, frame_id)
            frame_id += 1

            pose_label, trusted, direction = "", True, None
            if snapshot is not None and snapshot.hand_detected and snapshot.observation is not None:
                if snapshot.pose is not None:
                    pose_label, trusted = snapshot.pose.label, bool(snapshot.pose.trusted)
                direction = pointing_direction(np.asarray(snapshot.observation.landmarks, dtype=np.float64))
                for event in snapshot.pose_events:
                    if event.kind in ("desktop_prev", "desktop_next"):
                        counts[event.kind] += 1
                        last_swipe = "◀ 이전 데스크톱" if event.kind == "desktop_prev" else "다음 데스크톱 ▶"
                        flash_until = now + 0.5
                        print(f"[{ts:>6}ms] SWIPE {event.kind}  누적 {dict(counts)}")

            view = cv2.flip(bgr, 1)
            _draw(view, pose=pose_label, trusted=trusted, direction=direction, counts=counts,
                  last_swipe=last_swipe, flash=now < flash_until)
            cv2.imshow("two-finger swipe live", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        probe.close()
    print(f"\n종료 — 누적 스와이프: {dict(counts)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two-finger-swipe-live", description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 인덱스 (기본 0)")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="hand_landmarker.task 경로")
    parser.add_argument("--pose-model", default="models/hand_pose_classifier.pt", help="자세 분류 모델 경로")
    args = parser.parse_args(argv)
    return run(args.camera, Path(args.hand_model), Path(args.pose_model))


if __name__ == "__main__":
    raise SystemExit(main())
