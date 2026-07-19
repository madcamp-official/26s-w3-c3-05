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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QInputDialog,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.gaze_source import GazeSnapshot
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample
from jarvis.monitoring.gesture_source import GestureSource, NullGestureSource
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.overlay import Frame, draw_gaze_overlay, draw_hud, placeholder_frame
from jarvis.monitoring.pipeline_status import StageState, StageStatus, detect_pipeline_status
from jarvis.calibration.registry import TargetRecord, TargetRegistry
from jarvis.calibration.target_registration import TargetRegistrationSession
from jarvis.gaze.direction import direction_to_yaw_pitch

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
        self._gaze_snapshot: GazeSnapshot | None = None
        self._targets: list[TargetRecord] = []
        self._registration_samples: list[tuple[float, float]] = []
        self._show_placeholder("카메라 시작 중…")

    def _show_placeholder(self, text: str) -> None:
        self._render(placeholder_frame(text=text))

    def show_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        self._fps_times.append(now)
        self._frame_count += 1
        fps = self._current_fps()
        h, w = frame.shape[:2]
        if self._gaze_snapshot is not None:
            draw_gaze_overlay(frame, self._gaze_snapshot, self._targets, self._registration_samples)
        draw_hud(
            frame,
            [f"{w}x{h}  {fps:4.1f} FPS", f"frame #{self._frame_count}"],
        )
        self._render(frame)

    def update_gaze(self, snapshot: GazeSnapshot) -> None:
        self._gaze_snapshot = snapshot

    def update_targets(self, targets: list[TargetRecord]) -> None:
        self._targets = targets

    def update_registration_samples(self, samples: list[tuple[float, float]]) -> None:
        self._registration_samples = samples

    def _current_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 0 else 0.0

    def _render(self, frame: Frame) -> None:
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
        self._status.setStyleSheet("color:#8b949e;" if source.available else "color:#d29922;")
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
        self.setStyleSheet(
            "QFrame{background:#161b22; border:1px solid #30363d; border-radius:8px;}"
        )
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        name = QLabel(status.name)
        name.setStyleSheet("font-weight:600; color:#e6e6e6; border:none;")
        chip = QLabel(status.state)
        chip.setStyleSheet(f"color:{_STATE_COLOR[status.state]}; font-weight:700; border:none;")
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
        profiles_path: Path | None = None,
        samples_path: Path | None = None,
        start_camera: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS Pipeline Monitor")
        self.resize(1100, 760)
        self._log = MessageLog()
        self._gesture_source: GestureSource = NullGestureSource()
        self._env = env if env is not None else _load_env()
        self._model_path = model_path or Path("models/face_landmarker.task")
        self._profiles_path = profiles_path or Path("data/calibration/profiles.json")
        self._latest_gaze: GazeSnapshot | None = None
        self._gaze_history: deque[GazeSnapshot] = deque()
        self._sample_store = GazeSampleStore(
            samples_path or Path("data/evaluation/gaze_samples.json")
        )
        self._target_registry = TargetRegistry(self._profiles_path)
        self._registration: TargetRegistrationSession | None = None
        self._registration_points: list[tuple[float, float]] = []

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
        self._sample_button = QPushButton()
        self._sample_button.clicked.connect(self._save_gaze_sample)
        self._clear_samples_button = QPushButton("샘플 초기화")
        self._clear_samples_button.clicked.connect(self._clear_gaze_samples)
        self._refresh_sample_button()
        sample_controls = QHBoxLayout()
        sample_controls.addWidget(self._sample_button, 1)
        sample_controls.addWidget(self._clear_samples_button)
        layout.addLayout(sample_controls)
        self._sample_list = QListWidget()
        self._sample_list.setMaximumHeight(130)
        self._sample_list.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")
        for sample in self._sample_store.samples:
            self._sample_list.addItem(format_gaze_sample(sample))
        layout.addWidget(self._sample_list)
        target_controls = QHBoxLayout()
        self._register_target_button = QPushButton("기기 바라보고 등록")
        self._register_target_button.clicked.connect(self._start_target_registration)
        self._reregister_target_button = QPushButton("위치 다시 등록")
        self._reregister_target_button.clicked.connect(self._reregister_selected_target)
        self._rename_target_button = QPushButton("이름 변경")
        self._rename_target_button.clicked.connect(self._rename_selected_target)
        self._delete_target_button = QPushButton("기기 삭제")
        self._delete_target_button.clicked.connect(self._delete_selected_target)
        for button in (
            self._register_target_button,
            self._reregister_target_button,
            self._rename_target_button,
            self._delete_target_button,
        ):
            target_controls.addWidget(button)
        layout.addLayout(target_controls)
        self._target_list = QListWidget()
        self._target_list.setMaximumHeight(100)
        layout.addWidget(self._target_list)
        self._refresh_targets()
        return container

    def _build_pipeline_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        for status in detect_pipeline_status(self._env, self._model_path):
            layout.addWidget(StageCard(status))
        layout.addStretch(1)
        return container

    def _start_camera(self, device_index: int) -> None:
        worker = CameraWorker(
            device_index,
            model_path=self._model_path,
            profiles_path=self._profiles_path,
        )
        worker.frame_ready.connect(self._on_frame)
        worker.gaze_ready.connect(self._on_gaze)
        worker.failed.connect(self._on_camera_failed)
        worker.gaze_failed.connect(self._on_gaze_failed)
        self._camera = worker
        worker.start()
        self._log.info(f"카메라 {device_index}번 시작")

    def _on_frame(self, frame: Frame) -> None:
        self._video.show_frame(frame)

    def _on_gaze(self, snapshot: GazeSnapshot) -> None:
        self._latest_gaze = snapshot
        self._gaze_history.append(snapshot)
        cutoff_ms = snapshot.observation.timestamp_ms - 500
        while self._gaze_history and self._gaze_history[0].observation.timestamp_ms < cutoff_ms:
            self._gaze_history.popleft()
        self._video.update_gaze(snapshot)
        if self._registration is not None:
            confidence = min(
                snapshot.observation.eye_tracking_confidence,
                snapshot.observation.face_tracking_confidence,
            )
            if (
                self._registration.add(
                    snapshot.gaze_vector,
                    confidence,
                    eyes_open=snapshot.observation.eyes_open,
                )
                and snapshot.gaze_vector is not None
            ):
                self._registration_points.append(
                    direction_to_yaw_pitch(snapshot.gaze_vector.direction)
                )
                self._video.update_registration_samples(self._registration_points)
            if self._registration.is_elapsed(snapshot.observation.timestamp_ms):
                self._finish_target_registration()

    def _selected_target(self) -> TargetRecord | None:
        row = self._target_list.currentRow()
        records = self._target_registry.records
        return records[row] if 0 <= row < len(records) else None

    def _start_target_registration(self) -> None:
        target_id, ok = QInputDialog.getText(self, "기기 등록", "Target ID")
        if not ok or not target_id.strip():
            return
        name, ok = QInputDialog.getText(self, "기기 등록", "표시 이름", text=target_id.strip())
        if not ok or not name.strip():
            return
        device_type, ok = QInputDialog.getText(self, "기기 등록", "기기 종류", text="LIGHT")
        if not ok or not device_type.strip():
            return
        device_id, ok = QInputDialog.getText(
            self, "기기 등록", "Adapter Device ID", text=target_id.strip()
        )
        if not ok or not device_id.strip():
            return
        self._begin_registration(
            target_id.strip(), name.strip(), device_type.strip(), device_id.strip()
        )

    def _begin_registration(
        self, target_id: str, name: str, device_type: str, device_id: str
    ) -> None:
        if self._registration is not None:
            self._log.warn("이미 기기 등록이 진행 중입니다")
            return
        self._registration = TargetRegistrationSession(target_id, name, device_type, device_id)
        self._registration_points = []
        self._register_target_button.setEnabled(False)
        self._log.info(f"'{name}'을 2초 동안 바라보며 고개를 천천히 움직여 주세요")

    def _finish_target_registration(self) -> None:
        assert self._registration is not None
        try:
            record = self._registration.finalize()
            nearby = [
                item
                for item in self._target_registry.nearby(
                    record.direction.yaw, record.direction.pitch
                )
                if item.target_id != record.target_id
            ]
            if nearby:
                names = ", ".join(item.name for item in nearby)
                self._log.warn(f"등록 방향이 기존 기기와 가깝습니다: {names}")
            self._target_registry.upsert(record)
            if self._camera is not None:
                self._camera.register_profile(record.to_profile())
            self._log.info(
                f"'{record.name}' 방향 등록 완료 ({self._registration.valid_frame_count} frames)"
            )
        except ValueError as exc:
            self._log.warn(f"기기 등록 실패: {exc}")
        finally:
            self._registration = None
            self._registration_points = []
            self._video.update_registration_samples([])
            self._register_target_button.setEnabled(True)
            self._refresh_targets()

    def _reregister_selected_target(self) -> None:
        record = self._selected_target()
        if record is None:
            self._log.warn("위치를 다시 등록할 기기를 선택하세요")
            return
        self._begin_registration(
            record.target_id, record.name, record.device_type, record.device_id
        )

    def _rename_selected_target(self) -> None:
        record = self._selected_target()
        if record is None:
            return
        name, ok = QInputDialog.getText(self, "이름 변경", "표시 이름", text=record.name)
        if ok and name.strip():
            self._target_registry.rename(record.target_id, name.strip())
            self._refresh_targets()

    def _delete_selected_target(self) -> None:
        record = self._selected_target()
        if record is None:
            return
        if (
            QMessageBox.question(self, "기기 삭제", f"'{record.name}'을 삭제할까요?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._target_registry.remove(record.target_id)
        if self._camera is not None:
            self._camera.unregister_profile(record.target_id)
        self._refresh_targets()

    def _refresh_targets(self) -> None:
        self._target_list.clear()
        for record in self._target_registry.records:
            self._target_list.addItem(
                f"{record.name} [{record.target_id}]  yaw={record.direction.yaw:+.1f} pitch={record.direction.pitch:+.1f}"
            )
        self._video.update_targets(self._target_registry.records)

    def _save_gaze_sample(self) -> None:
        if self._latest_gaze is None:
            self._log.warn("저장할 Gaze 값이 아직 없습니다")
            return
        try:
            sample = self._sample_store.add_window(list(self._gaze_history))
        except ValueError as exc:
            self._log.warn(f"Gaze 샘플 저장 실패: {exc}")
            return
        self._log.info(f"Gaze 샘플 {sample['sample_index']}/{self._sample_store.capacity} 저장")
        self._sample_list.addItem(format_gaze_sample(sample))
        self._sample_list.scrollToBottom()
        self._refresh_sample_button()

    def _refresh_sample_button(self) -> None:
        self._sample_button.setText(
            f"시선 샘플 저장 ({self._sample_store.count}/{self._sample_store.capacity})"
        )
        self._sample_button.setEnabled(not self._sample_store.full)

    def _clear_gaze_samples(self) -> None:
        self._sample_store.clear()
        self._sample_list.clear()
        self._refresh_sample_button()
        self._log.info("저장된 Gaze 샘플 초기화")

    def _on_gaze_failed(self, message: str) -> None:
        self._log.error(message)

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
    env_file = Path(".env")
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    return int(app.exec())
