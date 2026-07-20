"""참고 레포 **레거시 엔진**(mp.solutions.hands) 실행 데모 — 진짜 엔진 이식.

이 스크립트만은 이 프로젝트의 Tasks API가 아니라 참고 레포가 쓰는 **레거시
Solutions API**(`mp.solutions.hands`)로 랜드마크를 검출한다. 참고 레포 `app.py`의
랜드마크 경로를 그대로 옮긴 것이다(cv.flip 미러 → RGB → hands.process → 랜드마크).

**중요: 이 스크립트는 별도 격리 venv로만 돈다.**
- 프로젝트 메인 venv는 mediapipe 0.10.35(slim, solutions 없음)라 여기서 실행 불가.
- solutions가 살아있는 구버전(예: mediapipe==0.10.14)을 설치한 격리 venv 필요.
- 그래서 이 모듈은 프로젝트 패키지의 정상 import/테스트/린트 대상이 **아니다**
  (`reference_port/__init__.py`에서 import하지 않는다). mediapipe는 main()
  안에서 지연 import한다.

실행 예:
    # 격리 venv(레거시 mediapipe + opencv)로:
    <legacy-venv>/bin/python .../legacy_engine_demo.py --device 0

Tasks API 쪽(우리 엔진)은 `compare.py`(메인 venv)로 띄워 눈으로 A/B 비교한다.
두 엔진은 서로 다른 mediapipe 버전이라 한 프로세스에 공존할 수 없다.
"""

from __future__ import annotations

import argparse

_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="참고 레거시 엔진(mp.solutions.hands) 데모")
    parser.add_argument("--device", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--min-detection", type=float, default=0.7, help="검출 신뢰도(참고 기본 0.7)")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="추적 신뢰도(참고 기본 0.5)")
    args = parser.parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        print(f"레거시 엔진 데모는 격리 venv(구버전 mediapipe + opencv)가 필요합니다: {exc}")
        return 1

    if not hasattr(mp, "solutions"):
        print(
            f"설치된 mediapipe {mp.__version__}에 레거시 solutions가 없습니다. "
            "solutions 포함 구버전(예: mediapipe==0.10.14)을 격리 venv에 설치하세요."
        )
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
    )
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"카메라 {args.device}번을 열 수 없습니다.")
        hands.close()
        return 1

    print(f"참고 레거시 엔진 데모 실행 중 (mediapipe {mp.__version__}, mp.solutions.hands) — ESC 종료")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # 참고 레포 app.py 그대로: 미러 → RGB → process
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            h, w = frame.shape[:2]
            handed = "no hand"
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                px = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                for a, b in _HAND_CONNECTIONS:
                    cv2.line(frame, px[a], px[b], (200, 120, 240), 2, cv2.LINE_AA)
                for x, y in px:
                    cv2.circle(frame, (x, y), 3, (240, 240, 240), -1)
                if results.multi_handedness:
                    c = results.multi_handedness[0].classification[0]
                    handed = f"{c.label} {c.score:.0%}"

            cv2.rectangle(frame, (0, 0), (w, 56), (0, 0, 0), -1)
            cv2.putText(frame, "참고 레거시 엔진 (mp.solutions.hands)", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 120, 240), 2, cv2.LINE_AA)
            cv2.putText(frame, f"mediapipe {mp.__version__}  ·  {handed}", (12, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("legacy engine (mp.solutions.hands)  ESC=quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
