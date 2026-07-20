"""참고 레포 **랜드마크 엔진 자체**(`mp.solutions.hands`)를 돌리는 워커 프로세스.

이 스크립트만은 이 프로젝트의 Tasks API가 아니라 참고 레포
(Kazuhito00/hand-gesture-recognition-using-mediapipe)가 쓰는 **레거시 Solutions
API**로 랜드마크를 검출한다. 참고 `app.py`의 검출 경로를 그대로 옮겼다:

    RGB 변환 → `hands.process(rgb)` → `multi_hand_landmarks[0]`

**메인 venv에서는 실행되지 않는다.** 프로젝트 런타임의 mediapipe 0.10.35(slim)에는
`solutions`가 없으므로, solutions가 살아있는 구버전(0.10.14)을 설치한 **격리 venv**의
인터프리터로만 돈다. 그래서 이 모듈은 `jarvis.engine_port.__init__`에서 import하지
않으며 mediapipe도 `main()` 안에서 지연 import한다 — 메인 환경의 테스트·타입체크가
mediapipe 구버전을 요구하지 않도록.

직접 실행할 일은 없다. `jarvis.engine_port.client.LegacyEngineClient`가 격리 venv의
파이썬으로 이 파일을 자식 프로세스로 띄우고 stdin/stdout으로 대화한다. 규약은
`protocol.py` 참고.

    <legacy-venv>/bin/python src/jarvis/engine_port/legacy_worker.py

stdout은 결과 JSON 전용이다. 진단 메시지는 전부 stderr로 보낸다 — stdout에 한 줄만
잘못 섞여도 메인 쪽 파싱이 깨진다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 이 워커는 jarvis가 설치되지 않은 격리 venv에서 돈다. 규약 모듈만 쓰면 되므로
# src/를 sys.path에 얹어 소스 트리에서 직접 import한다(jarvis/__init__은 의존성 없음).
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from jarvis.engine_port.protocol import (  # noqa: E402 - 위 sys.path 설정 이후여야 한다
    FRAME_HEADER_SIZE,
    LandmarkResult,
    decode_frame_header,
    encode_result,
    read_exactly,
)


def _log(message: str) -> None:
    """진단 출력은 stdout(결과 전용)을 오염시키지 않도록 stderr로만."""
    print(f"[legacy_worker] {message}", file=sys.stderr, flush=True)


def _extract(results: Any, timestamp_ms: int) -> LandmarkResult:  # noqa: ANN401 - mediapipe 결과 타입
    """참고 레포와 동일하게 첫 번째 손의 정규화 랜드마크만 뽑는다."""
    import numpy as np

    if not results.multi_hand_landmarks:
        return LandmarkResult(timestamp_ms=timestamp_ms, points=None)

    landmarks = results.multi_hand_landmarks[0]
    points = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float64)

    handedness: str | None = None
    score: float | None = None
    if results.multi_handedness:
        classification = results.multi_handedness[0].classification[0]
        handedness = str(classification.label)
        score = float(classification.score)

    return LandmarkResult(
        timestamp_ms=timestamp_ms,
        points=points,
        handedness=handedness,
        score=score,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="레거시 mp.solutions.hands 워커")
    parser.add_argument("--min-detection", type=float, default=0.7, help="검출 신뢰도(참고 기본 0.7)")
    parser.add_argument("--min-tracking", type=float, default=0.5, help="추적 신뢰도(참고 기본 0.5)")
    parser.add_argument("--max-hands", type=int, default=1, help="최대 손 개수(참고 기본 1)")
    args = parser.parse_args()

    try:
        import numpy as np
        import mediapipe as mp
    except ImportError as exc:
        _log(f"import 실패 — 격리 venv(구버전 mediapipe)로 실행해야 합니다: {exc}")
        return 1

    if not hasattr(mp, "solutions"):
        _log(
            f"설치된 mediapipe {mp.__version__}에 레거시 solutions가 없습니다. "
            "solutions 포함 구버전(예: mediapipe==0.10.14)을 격리 venv에 설치하세요."
        )
        return 1

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection,
        min_tracking_confidence=args.min_tracking,
    )
    _log(f"레거시 엔진 준비 완료 (mediapipe {mp.__version__}, mp.solutions.hands)")

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        while True:
            raw_header = read_exactly(stdin, FRAME_HEADER_SIZE)
            if raw_header is None:
                break  # 메인이 stdin을 닫음 = 정상 종료
            header = decode_frame_header(raw_header)

            payload = read_exactly(stdin, header.payload_size)
            if payload is None:
                _log("픽셀 블록 도중 EOF — 종료합니다")
                break

            bgr = np.frombuffer(payload, dtype=np.uint8).reshape(header.height, header.width, 3)
            # 참고 app.py와 동일하게 RGB를 먹인다. cv2 없이 채널만 뒤집고 연속 배열로
            # 만든다(mediapipe는 C-contiguous 버퍼를 요구한다).
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])

            result = _extract(hands.process(rgb), header.timestamp_ms)
            stdout.write(encode_result(result))
            stdout.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass  # 메인이 먼저 사라진 경우 — 조용히 종료한다
    finally:
        hands.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
