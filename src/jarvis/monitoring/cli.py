"""``jarvis-monitor`` — launch the real-time desktop monitor."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jarvis-monitor",
        description="JARVIS 실시간 파이프라인 모니터 (데스크탑 앱)",
    )
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 인덱스 (기본 0)")
    parser.add_argument(
        "--model",
        default="models/face_landmarker.task",
        help="face_landmarker.task 경로 (gaze 파이프라인 라이브에 필요)",
    )
    parser.add_argument(
        "--profiles",
        default="data/calibration/profiles.json",
        help="calibration 프로파일 JSON 경로 (Target 분류에 필요)",
    )
    args = parser.parse_args(argv)

    try:
        from jarvis.monitoring.app import MainWindow
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        parser.exit(
            2,
            f"UI 의존성이 없습니다 ({exc.name}). 설치: pip install -e \".[ui]\"\n",
        )

    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        device_index=args.camera,
        model_path=Path(args.model),
        profiles_path=Path(args.profiles),
    )
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
