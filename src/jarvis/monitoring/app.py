"""JARVIS real-time monitor — desktop application (PySide6).

Tab 1 (실시간): live webcam in the center, a recognized-gesture sidebar on the
right, and a system-message panel along the bottom.
Tab 2 (파이프라인): a card per pipeline stage showing its real availability.

The window wires the parts that exist today (webcam capture, adapter/config
detection, system messages) and honestly marks the parts that do not (Gesture and
Fusion — dev-2). No detection is faked.
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.gesture_source import GestureSource, NullGestureSource
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.overlay import draw_hud, placeholder_frame
from jarvis.monitoring.pipeline_status import StageState, StageStatus, detect_pipeline_status
from jarvis.runtime_protocol.config import read_env_file

_STATE_COLOR = {
    StageState.LIVE: "#3fb950",
    StageState.DEGRADED: "#d29922",
    StageState.UNAVAILABLE: "#8b949e",
    StageState.ERROR: "#f85149",
}
_LEVEL_COLOR = {
    MessageLevel.INFO: "#8b949e",
    MessageLevel.WARN: "#d29922",
    MessageLevel.ERROR: "#f85149",
}


class VideoView(QLabel):
    """Displays webcam frames scaled to the widget, with a HUD."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#0b0e13;")
        self._fps_times: deque[float] = deque(maxlen=30)
        self._frame_count = 0
        self._show_placeholder("카메라 시작 중…")

    def _show_placeholder(self, text: str) -> None:
        self._render(placeholder_frame(text=text))

    def show_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        self._fps_times.append(now)
        self._frame_count += 1
        fps = self._current_fps()
        h, w = frame.shape[:2]
        draw_hud(
            frame,
            [f"{w}x{h}  {fps:4.1f} FPS", f"frame #{self._frame_count}"],
        )
        self._render(frame)

    def _current_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 0 else 0.0

    def _render(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class GestureSidebar(QWidget):
    """Recognized-gesture list bound to a GestureSource."""

    def __init__(self, source: GestureSource) -> None:
        super().__init__()
        self._source = source
        layout = QVBoxLayout(self)
        title = QLabel("인식된 제스처")
        title.setStyleSheet("font-weight:600; color:#58a6ff; padding:4px 0;")
        self._status = QLabel(source.status_text)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            "color:#8b949e;" if source.available else "color:#d29922;"
        )
        self._list = QListWidget()
        layout.addWidget(title)
        layout.addWidget(self._status)
        layout.addWidget(self._list, 1)
        self.setMinimumWidth(220)

    def poll(self) -> None:
        for g in self._source.poll():
            self._list.insertItem(0, f"{g.gesture}  {g.confidence:.0%}  [{g.phase}]")
            while self._list.count() > 100:
                self._list.takeItem(self._list.count() - 1)


class MessagePanel(QListWidget):
    """Bottom panel rendering the most recent system messages."""

    def __init__(self, log: MessageLog) -> None:
        super().__init__()
        self._log = log
        self.setMaximumHeight(150)
        self.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")

    def refresh(self) -> None:
        self.clear()
        for m in self._log.recent(50):
            item_text = f"[{m.timestamp_ms:>8} ms] {m.level}  {m.text}"
            self.addItem(item_text)
            item = self.item(self.count() - 1)
            item.setForeground(QColor(_LEVEL_COLOR[m.level]))
        self.scrollToBottom()


class StageCard(QFrame):
    """One pipeline stage's availability card."""

    def __init__(self, status: StageStatus) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame{background:#161b22; border:1px solid #30363d; border-radius:8px;}")
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        name = QLabel(status.name)
        name.setStyleSheet("font-weight:600; color:#e6e6e6; border:none;")
        chip = QLabel(status.state)
        chip.setStyleSheet(
            f"color:{_STATE_COLOR[status.state]}; font-weight:700; border:none;"
        )
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(chip)
        detail = QLabel(status.detail)
        detail.setWordWrap(True)
        detail.setStyleSheet("color:#8b949e; border:none;")
        layout.addLayout(header)
        layout.addWidget(detail)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        device_index: int = 0,
        env: dict[str, str] | None = None,
        model_path: Path | None = None,
        start_camera: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS Pipeline Monitor")
        self.resize(1100, 760)
        self._log = MessageLog()
        self._gesture_source: GestureSource = NullGestureSource()
        self._env = env if env is not None else _load_env()
        self._model_path = model_path

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "실시간")
        tabs.addTab(self._build_pipeline_tab(), "파이프라인")
        self.setCentralWidget(tabs)

        self._log.info("모니터 시작")
        if not self._gesture_source.available:
            self._log.warn(self._gesture_source.status_text)

        self._camera: CameraWorker | None = None
        if start_camera:
            self._start_camera(device_index)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(150)

    def _build_live_tab(self) -> QWidget:
        self._video = VideoView()
        self._sidebar = GestureSidebar(self._gesture_source)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._video)
        top.addWidget(self._sidebar)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 0)

        self._messages = MessagePanel(self._log)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self._messages)
        split.setStretchFactor(0, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(split)
        return container

    def _build_pipeline_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        for status in detect_pipeline_status(self._env, self._model_path):
            layout.addWidget(StageCard(status))
        layout.addStretch(1)
        return container

    def _start_camera(self, device_index: int) -> None:
        worker = CameraWorker(device_index)
        worker.frame_ready.connect(self._on_frame)
        worker.failed.connect(self._on_camera_failed)
        self._camera = worker
        worker.start()
        self._log.info(f"카메라 {device_index}번 시작")

    def _on_frame(self, frame: np.ndarray) -> None:
        self._video.show_frame(frame)

    def _on_camera_failed(self, message: str) -> None:
        self._log.error(message)
        self._video._show_placeholder("NO CAMERA")

    def _on_tick(self) -> None:
        self._sidebar.poll()
        self._messages.refresh()

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        if self._camera is not None:
            self._camera.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(read_env_file(Path(".env")))
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    return int(app.exec())
