"""기존 엔진 vs 이식한 레거시 엔진 랜드마크 A/B — `python -m jarvis.engine_port.compare`.

**같은 카메라 프레임**을 두 개의 서로 다른 랜드마크 엔진에 그대로 흘려 좌/우 패널에
그린다. 이전 실험(`reference_port`)의 A/B와 달리 여기서 다른 것은 설정이 아니라
**엔진 자체**다:

- 왼쪽 "기존 엔진"  : 이 프로젝트의 mediapipe Tasks API `HandLandmarker` (같은 프로세스)
- 오른쪽 "이식 엔진": 참고 레포의 레거시 `mp.solutions.hands` (격리 venv의 자식 프로세스)

두 mediapipe 버전은 한 프로세스에 공존할 수 없어 오른쪽은 서브프로세스로 돈다.
프레임은 **동기 요청/응답**으로 주고받으므로 좌우 패널은 항상 동일한 프레임이다 —
그래야 편차 수치가 의미를 갖는다.

공정한 엔진 비교를 위해 기본적으로 **양쪽에 같은 검출 신뢰도**를 주고 **평활화는
양쪽 다 끈다**(One-Euro는 우리 파이프라인의 후처리이지 엔진의 능력이 아니다).

    python -m jarvis.engine_port.setup_legacy_env   # 최초 1회: 격리 venv 생성
    python -m jarvis.engine_port.compare            # 웹캠 A/B
    python -m jarvis.engine_port.compare --dump ab.csv   # 프레임별 수치를 CSV로

ESC로 종료하면 편차·검출 일치율·지터 요약이 출력된다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

import cv2
import numpy as np
import numpy.typing as npt

from jarvis.engine_port.client import LegacyEngineClient, LegacyEngineError, resolve_legacy_python
from jarvis.engine_port.metrics import ComparisonAccumulator
from jarvis.engine_port.protocol import FloatArray

if TYPE_CHECKING:
    import _csv

    from mediapipe.tasks.python.vision import HandLandmarker

#: 좌우 패널을 합친 창의 기본 가로 크기(px). 640x480 웹캠이면 패널당 800px이 된다.
DEFAULT_WINDOW_WIDTH = 1600

_WINDOW_NAME = "engine A/B  (left=기존 / right=이식)  ESC=quit"

_LABEL_A = "기존 엔진"
_LABEL_B = "이식 엔진"
_COLOR_A = (80, 200, 80)
_COLOR_B = (80, 180, 240)

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
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in (
        Path("models/hand_landmarker.task"),
        Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task",
    ):
        if candidate.is_file():
            return candidate
    return None


def _make_landmarker(model_path: Path, min_detection: float) -> "HandLandmarker":
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _detect_tasks_api(
    landmarker: "HandLandmarker", rgb: npt.NDArray[np.uint8], timestamp_ms: int
) -> FloatArray | None:
    """Tasks API로 한 프레임 검출 → (21, 2) 정규화 좌표 또는 미검출 시 None."""
    from mediapipe import Image as MpImage
    from mediapipe import ImageFormat as MpImageFormat

    result = landmarker.detect_for_video(
        MpImage(image_format=MpImageFormat.SRGB, data=rgb), timestamp_ms
    )
    if not result.hand_landmarks:
        return None
    return np.array([[lm.x, lm.y] for lm in result.hand_landmarks[0]], dtype=np.float64)


def _draw_panel(
    frame_bgr: npt.NDArray[np.uint8],
    points: FloatArray | None,
    *,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
    status: str,
) -> npt.NDArray[np.uint8]:
    """프레임 복사본을 거울상으로 뒤집고 랜드마크·헤더를 그려 반환한다(표시 전용).

    선 두께·글자 크기는 패널 폭에 비례해 키운다 — 창을 키웠는데 UI만 작게 남는 것을
    막는다(640px 패널 기준으로 환산).
    """
    display = cast("npt.NDArray[np.uint8]", cv2.flip(frame_bgr, 1))
    height, width = display.shape[:2]
    ui = max(1.0, width / 640)
    line_width = max(2, round(2 * ui))

    if points is not None:
        pixels = [(int((1.0 - x) * width), int(y * height)) for x, y in points]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(display, pixels[a], pixels[b], color, line_width, cv2.LINE_AA)
        for x, y in pixels:
            cv2.circle(display, (x, y), max(3, round(3 * ui)), (240, 240, 240), -1)

    header_height = round(60 * ui)
    cv2.rectangle(display, (0, 0), (width, header_height), (0, 0, 0), -1)
    cv2.putText(
        display, title, (round(12 * ui), round(24 * ui)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui, color, line_width, cv2.LINE_AA,
    )
    cv2.putText(
        display, subtitle, (round(12 * ui), round(48 * ui)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui, (200, 200, 200), max(1, round(ui)), cv2.LINE_AA,
    )
    cv2.putText(
        display, status, (round(12 * ui), height - round(16 * ui)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ui, color, line_width, cv2.LINE_AA,
    )
    return display


def _scale_for_display(
    frame_bgr: npt.NDArray[np.uint8], panel_width: int
) -> npt.NDArray[np.uint8]:
    """표시용으로만 프레임을 확대한다 — 엔진에는 원본 해상도를 그대로 넣는다.

    그리기 **전에** 키워야 랜드마크 선과 글자가 또렷하다. 합친 뒤 확대하면 전부
    뭉개진다. 랜드마크는 정규화 좌표라 해상도가 바뀌어도 그대로 쓸 수 있다.
    """
    height, width = frame_bgr.shape[:2]
    if panel_width <= 0 or panel_width == width:
        return frame_bgr
    panel_height = max(1, round(height * panel_width / width))
    interpolation = cv2.INTER_LINEAR if panel_width > width else cv2.INTER_AREA
    return cast(
        "npt.NDArray[np.uint8]",
        cv2.resize(frame_bgr, (panel_width, panel_height), interpolation=interpolation),
    )


def _draw_deviation_banner(
    combined: npt.NDArray[np.uint8], frame_deviation: float | None, frame_width: int
) -> None:
    """합쳐진 화면 아래에 이번 프레임의 두 엔진 편차를 표시한다."""
    height, width = combined.shape[:2]
    ui = max(1.0, width / 1280)  # 합친 화면 기준(패널 2장)
    banner_height = round(40 * ui)
    cv2.rectangle(combined, (0, height - banner_height), (width, height), (0, 0, 0), -1)
    if frame_deviation is None:
        text = "편차: 두 엔진이 동시에 검출한 프레임에서만 측정됨"
    else:
        text = f"이번 프레임 평균 편차: {frame_deviation:.4f} ({frame_deviation * frame_width:.1f}px)"
    cv2.putText(
        combined, text, (round(12 * ui), height - round(14 * ui)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (230, 230, 230), max(1, round(ui)), cv2.LINE_AA,
    )


def run(
    *,
    source: int | str,
    model_path: Path,
    legacy_python: Path,
    min_detection: float,
    dump_path: Path | None,
    max_frames: int,
    headless: bool,
    window_width: int,
) -> int:
    landmarker = _make_landmarker(model_path, min_detection)
    accumulator = ComparisonAccumulator()
    dump_file: TextIO | None = None
    dump_writer: "_csv._writer | None" = None
    if dump_path is not None:
        dump_file = dump_path.open("w", newline="", encoding="utf-8")
        dump_writer = csv.writer(dump_file)
        dump_writer.writerow(["frame", "timestamp_ms", "detected_a", "detected_b", "mean_deviation"])

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"입력을 열 수 없습니다: {source}", file=sys.stderr)
        landmarker.close()
        if dump_file is not None:
            dump_file.close()
        return 1

    frame_width = 0
    started = time.monotonic()
    last_timestamp = -1
    frame_index = 0
    panel_width = max(320, window_width // 2)  # 좌우 두 장이 합쳐지므로 절반씩
    window_ready = False

    print(
        f"엔진 A/B 실행 중 — 좌: {_LABEL_A}(Tasks API) / 우: {_LABEL_B}(mp.solutions.hands)\n"
        f"양쪽 검출 신뢰도 {min_detection}, 평활화 없음. ESC로 종료하면 요약이 출력됩니다."
    )
    try:
        with LegacyEngineClient(legacy_python, min_detection=min_detection) as legacy:
            while True:
                ok, frame_raw = capture.read()
                if not ok:
                    break
                frame = cast("npt.NDArray[np.uint8]", frame_raw)
                frame_width = int(frame.shape[1])
                rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = max(int((time.monotonic() - started) * 1000), last_timestamp + 1)
                last_timestamp = timestamp

                points_a = _detect_tasks_api(landmarker, rgb, timestamp)
                points_b = legacy.detect(frame, timestamp).points

                frame_deviation = accumulator.update(points_a, points_b)
                frame_index += 1
                if dump_writer is not None:
                    dump_writer.writerow([
                        frame_index,
                        timestamp,
                        int(points_a is not None),
                        int(points_b is not None),
                        "" if frame_deviation is None else f"{frame_deviation:.6f}",
                    ])

                if not headless:
                    if not window_ready:
                        # WINDOW_NORMAL이라야 사용자가 창을 자유롭게 늘릴 수 있다.
                        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
                        panel_height = round(frame.shape[0] * panel_width / frame.shape[1])
                        cv2.resizeWindow(_WINDOW_NAME, panel_width * 2, panel_height)
                        window_ready = True
                    display_frame = _scale_for_display(frame, panel_width)
                    panel_a = _draw_panel(
                        display_frame, points_a,
                        title=_LABEL_A, subtitle="mediapipe Tasks API HandLandmarker",
                        color=_COLOR_A, status="hand" if points_a is not None else "no hand",
                    )
                    panel_b = _draw_panel(
                        display_frame, points_b,
                        title=_LABEL_B, subtitle="참고 레포 레거시 mp.solutions.hands",
                        color=_COLOR_B, status="hand" if points_b is not None else "no hand",
                    )
                    combined = cast("npt.NDArray[np.uint8]", cv2.hconcat([panel_a, panel_b]))
                    _draw_deviation_banner(combined, frame_deviation, frame_width)
                    cv2.imshow(_WINDOW_NAME, combined)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if 0 < max_frames <= frame_index:
                    break
    except LegacyEngineError as exc:
        print(f"\n레거시 엔진 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        with suppress(cv2.error):
            cv2.destroyAllWindows()
        landmarker.close()
        if dump_file is not None:
            dump_file.close()
            print(f"\n프레임별 수치를 저장했습니다: {dump_path}")

    print("\n" + accumulator.summary().format_report(
        label_a=_LABEL_A, label_b=_LABEL_B, frame_width=frame_width or None
    ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="랜드마크 엔진 A/B (기존 Tasks API vs 이식 레거시)")
    parser.add_argument("--device", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument(
        "--video", type=str, default=None,
        help="웹캠 대신 쓸 녹화 파일 경로 — 같은 영상으로 비교를 재현할 수 있다",
    )
    parser.add_argument("--model", type=str, default=None, help="hand_landmarker.task 경로")
    parser.add_argument(
        "--legacy-python", type=str, default=None,
        help="격리 venv 파이썬 경로 (기본: 저장소 루트 .venv-legacy)",
    )
    parser.add_argument(
        "--min-detection", type=float, default=0.5,
        help="양쪽 엔진에 동일하게 주는 검출 신뢰도 (기본 0.5 — 우리 파이프라인 기본값)",
    )
    parser.add_argument("--dump", type=str, default=None, help="프레임별 수치를 쓸 CSV 경로")
    parser.add_argument("--max-frames", type=int, default=0, help="N>0이면 N 프레임 후 자동 종료")
    parser.add_argument(
        "--window-width", type=int, default=DEFAULT_WINDOW_WIDTH,
        help=f"좌우 패널을 합친 창의 가로 크기 px (기본 {DEFAULT_WINDOW_WIDTH}). 창은 드래그로도 조절 가능",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="창을 띄우지 않고 수치만 집계 (--video/--dump 와 함께 배치 비교용)",
    )
    args = parser.parse_args()

    model_path = _find_model(args.model)
    if model_path is None:
        print("hand_landmarker.task 모델을 찾지 못했습니다. --model 로 경로를 지정하세요.", file=sys.stderr)
        return 1
    try:
        legacy_python = resolve_legacy_python(args.legacy_python)
    except LegacyEngineError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return run(
        source=args.video if args.video else args.device,
        model_path=model_path,
        legacy_python=legacy_python,
        min_detection=args.min_detection,
        dump_path=Path(args.dump) if args.dump else None,
        max_frames=args.max_frames,
        headless=args.headless,
        window_width=args.window_width,
    )


if __name__ == "__main__":
    raise SystemExit(main())
