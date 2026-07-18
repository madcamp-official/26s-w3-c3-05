"""JARVIS real-time monitor — desktop application (PySide6).

Tabs:
- 실시간: live webcam (with gaze overlay) + recognized-gesture sidebar + messages.
- Gaze 파이프라인: every intermediate value of the real gaze engine, per frame
  (landmarks → vector → smoothing → classifier → lock → TargetEstimate).
- 파이프라인: a card per stage's real availability + the message contracts that
  the not-yet-implemented stages (Gesture/Fusion/Command) will carry.
- 지연·어댑터: measured per-stage latency and device-adapter readiness.

The window wires the parts that exist today (webcam capture, the gaze pipeline
when mediapipe+model+calibration are present, adapter/config detection) and
honestly marks the parts that do not. No detection is faked.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.contracts.messages import Command, GestureEstimate, Intent
from jarvis.gaze.lock import GazeLockState
from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.gaze_probe import GazeProbe, GazeSnapshot
from jarvis.monitoring.gesture_source import GestureSource, NullGestureSource
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.overlay import draw_gaze_overlay, draw_hud, placeholder_frame
from jarvis.monitoring.pipeline_status import StageState, StageStatus, detect_pipeline_status
from jarvis.runtime_protocol.config import read_env_file
from jarvis.runtime_protocol.telemetry.latency import LatencyAggregator, LatencyStage

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
_LOCK_COLOR = {
    GazeLockState.SEARCHING: "#8b949e",
    GazeLockState.CANDIDATE: "#58a6ff",
    GazeLockState.TARGET_LOCKED: "#3fb950",
    GazeLockState.GESTURE_WAIT: "#d29922",
    GazeLockState.EXPIRED: "#f85149",
    GazeLockState.COMMITTED: "#2ea043",
}
_MONO = "font-family:Consolas,monospace; font-size:12px; color:#c9d1d9;"


def _header(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight:600; color:#58a6ff; padding:6px 0 2px 0;")
    return label


class LabeledBar(QWidget):
    """A name + 0..1 progress bar + value text, with an optional threshold note."""

    def __init__(self, name: str, threshold: float | None = None) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        label_text = name if threshold is None else f"{name} (≥{threshold:.2f})"
        self._name = QLabel(label_text)
        self._name.setMinimumWidth(150)
        self._name.setStyleSheet("color:#8b949e;")
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)
        self._value = QLabel("--")
        self._value.setMinimumWidth(64)
        self._value.setStyleSheet(_MONO)
        layout.addWidget(self._name)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._value)

    def set_value(self, value: float | None, *, color: str = "#3fb950") -> None:
        if value is None:
            self._bar.setValue(0)
            self._value.setText("--")
            return
        clamped = max(0.0, min(1.0, value))
        self._bar.setValue(int(clamped * 1000))
        self._bar.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}")
        self._value.setText(f"{value:.3f}")


class LockStateStrip(QWidget):
    """The six gaze-lock states as chips, highlighting the current one."""

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._chips: dict[GazeLockState, QLabel] = {}
        for state in GazeLockState:
            chip = QLabel(str(state))
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setStyleSheet(self._chip_style(state, active=False))
            self._chips[state] = chip
            layout.addWidget(chip)

    @staticmethod
    def _chip_style(state: GazeLockState, *, active: bool) -> str:
        color = _LOCK_COLOR[state]
        if active:
            return (
                f"background:{color}; color:#0b0e13; font-weight:700; "
                "border-radius:4px; padding:4px 6px;"
            )
        return "color:#6e7681; border:1px solid #30363d; border-radius:4px; padding:4px 6px;"

    def set_state(self, active: GazeLockState) -> None:
        for state, chip in self._chips.items():
            chip.setStyleSheet(self._chip_style(state, active=(state == active)))


class GazePanel(QScrollArea):
    """Live view of every gaze-pipeline stage for the current frame."""

    def __init__(self, probe_status: str) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)

        self._status = QLabel(probe_status)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#d29922; padding:2px 0;")
        layout.addWidget(self._status)

        # 2e — lock state (put on top: it is the headline signal)
        layout.addWidget(_header("Gaze Lock 상태 (2e)"))
        self._lock_strip = LockStateStrip()
        layout.addWidget(self._lock_strip)
        self._locked = QLabel("locked device: --")
        self._locked.setStyleSheet(_MONO)
        layout.addWidget(self._locked)

        # 2d — classifier
        layout.addWidget(_header("Target 분류 (2d)"))
        self._prob = LabeledBar("top-1 확률", threshold=0.80)
        self._margin = LabeledBar("margin", threshold=0.20)
        layout.addWidget(self._prob)
        layout.addWidget(self._margin)
        self._reject = QLabel("")
        self._reject.setWordWrap(True)
        self._reject.setStyleSheet("color:#d29922;")
        layout.addWidget(self._reject)
        self._devices = QListWidget()
        self._devices.setMaximumHeight(96)
        self._devices.setStyleSheet(_MONO)
        layout.addWidget(self._devices)

        # 2b/2c — vector + smoothing gauges
        layout.addWidget(_header("Gaze Vector · Smoothing (2b·2c)"))
        self._track_conf = LabeledBar("tracking confidence", threshold=0.50)
        self._gaze_conf = LabeledBar("gaze confidence")
        self._stability = LabeledBar("smoothing stability")
        layout.addWidget(self._track_conf)
        layout.addWidget(self._gaze_conf)
        layout.addWidget(self._stability)

        # 2a/2f — raw numbers + contract message
        layout.addWidget(_header("Landmarks (2a) · TargetEstimate → Fusion (2f)"))
        self._numeric = QLabel("웹캠·mediapipe·모델이 준비되면 실시간 값이 표시됩니다.")
        self._numeric.setStyleSheet(_MONO)
        self._numeric.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._numeric)

        layout.addStretch(1)
        self.setWidget(body)

    def update_snapshot(self, s: GazeSnapshot) -> None:
        self._status.setText(
            f"frame #{s.frame_id} · {s.inference_ms:.0f} ms/frame"
            + ("  ·  얼굴 추적 손실" if s.tracking_lost else "")
        )
        self._status.setStyleSheet("color:#f85149;" if s.tracking_lost else "color:#3fb950;")

        self._lock_strip.set_state(s.lock_state)
        self._locked.setText(f"locked device: {s.locked_device or '--'}   confident: {s.is_confident}")

        prob_color = "#3fb950" if s.probability >= 0.80 else "#d29922"
        self._prob.set_value(s.probability, color=prob_color)
        self._margin.set_value(s.margin, color="#3fb950" if s.margin >= 0.20 else "#d29922")
        self._reject.setText(f"UNKNOWN 사유: {s.reject_reason}" if s.reject_reason else "")

        self._devices.clear()
        if not s.device_details:
            self._devices.addItem("등록된 기기 프로파일 없음 — jarvis-gaze calibrate 필요")
        for d in s.device_details:
            angle = "--" if np.isnan(d.angular_distance_deg) else f"{d.angular_distance_deg:6.1f}°"
            mark = "◀ 선택" if d.is_selected else ""
            self._devices.addItem(f"{d.device_id:<16} {angle}  {mark}")

        self._track_conf.set_value(s.tracking_confidence, color="#58a6ff")
        self._gaze_conf.set_value(s.gaze_confidence, color="#58a6ff")
        self._stability.set_value(s.smoothed_stability, color="#58a6ff")

        direction = (
            "  ".join(f"{v:+.3f}" for v in s.gaze_direction)
            if s.gaze_direction is not None
            else "추적 손실 (None)"
        )
        est = s.target_estimate
        self._numeric.setText(
            f"face_detected : {s.face_detected}\n"
            f"head (deg)    : yaw {s.head_yaw_deg:+7.2f}  pitch {s.head_pitch_deg:+7.2f}  "
            f"roll {s.head_roll_deg:+7.2f}\n"
            f"iris L / R    : {s.left_iris_relative}  /  {s.right_iris_relative}\n"
            f"gaze vector   : {direction}\n"
            f"smoothing buf : {s.buffer_fill}/{s.buffer_capacity} frames\n"
            "── TargetEstimate (contract) ──────────────\n"
            f"target={est.target}  p={est.probability:.3f}  "
            f"p2={est.second_best_probability:.3f}  stability={est.stability:.3f}\n"
            f"frame_id={est.frame_id}  timestamp_ms={est.timestamp_ms}"
        )


class ContractPanel(QFrame):
    """Shows the message contract a not-yet-implemented stage will emit."""

    def __init__(self, title: str, message_type: Any, note: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame{background:#161b22; border:1px solid #30363d; border-radius:8px;}")
        layout = QVBoxLayout(self)
        head = QLabel(title)
        head.setStyleSheet("font-weight:600; color:#e6e6e6; border:none;")
        layout.addWidget(head)
        note_label = QLabel(note)
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color:#8b949e; border:none;")
        layout.addWidget(note_label)
        fields = "\n".join(
            f"  {f.name}: {f.type}" for f in dataclasses.fields(message_type)
        )
        shape = QLabel(f"{message_type.__name__}\n{fields}")
        shape.setStyleSheet(_MONO + " border:none;")
        layout.addWidget(shape)


class LatencyPanel(QScrollArea):
    """Measured per-stage latency, refreshed from a shared aggregator."""

    _COLUMNS = ("stage", "n", "p50", "p95", "p99", "max", "mean")

    def __init__(self, aggregator: LatencyAggregator) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._aggregator = aggregator
        body = QWidget()
        self._layout = QGridLayout(body)
        self._layout.setContentsMargins(8, 8, 8, 8)
        header = _header("End-to-end 지연 (실측, ms)")
        self._layout.addWidget(header, 0, 0, 1, len(self._COLUMNS))
        for col, name in enumerate(self._COLUMNS):
            cell = QLabel(name)
            cell.setStyleSheet("color:#8b949e; font-weight:600;")
            self._layout.addWidget(cell, 1, col)
        note = QLabel(
            "p95 SLO: 노트북 ≤ 150ms · 전구 ≤ 1000ms (end-to-end).\n"
            "지금은 capture→inference(gaze 연산)만 실측된다 — Fusion·Command 단계는 "
            "2인 파트가 붙어야 채워진다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6e7681; padding-top:8px;")
        self._note_row = 2
        self._note = note
        self._value_labels: dict[str, list[QLabel]] = {}
        self.setWidget(body)
        self._layout.addWidget(self._note, 20, 0, 1, len(self._COLUMNS))

    def refresh(self) -> None:
        summaries = self._aggregator.summaries()
        row = 2
        for stage in LatencyStage:
            summary = summaries.get(stage)
            if summary is None:
                continue
            values = [
                str(stage),
                str(summary.count),
                f"{summary.p50:.0f}",
                f"{summary.p95:.0f}",
                f"{summary.p99:.0f}",
                f"{summary.maximum:.0f}",
                f"{summary.mean:.0f}",
            ]
            labels = self._value_labels.get(str(stage))
            if labels is None:
                labels = []
                for col in range(len(self._COLUMNS)):
                    cell = QLabel("")
                    cell.setStyleSheet(_MONO)
                    self._layout.addWidget(cell, row, col)
                    labels.append(cell)
                self._value_labels[str(stage)] = labels
            for cell, text in zip(labels, values, strict=True):
                cell.setText(text)
            row += 1


class AdapterPanel(QScrollArea):
    """Device-adapter readiness and configured targets (secrets redacted)."""

    def __init__(self, env: dict[str, str]) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(_header("Adapters / Command (5·6)"))

        windows_ok = os.name == "nt"
        token_present = bool(env.get("SMARTTHINGS_TOKEN", "").strip())
        targets_raw = env.get("SMARTTHINGS_DEVICE_TARGETS", "").strip()
        target_names = _target_names(targets_raw)

        lines = [
            ("Windows 입력 어댑터", "준비됨" if windows_ok else "Windows 아님", windows_ok),
            (
                "SmartThings 토큰",
                "설정됨 (값 비노출)" if token_present else "UNCONFIGURED",
                token_present,
            ),
            (
                "SmartThings 대상 기기",
                ", ".join(target_names) if target_names else "없음",
                bool(target_names),
            ),
        ]
        for name, value, ok in lines:
            row = QLabel(f"{name:<22} : {value}")
            row.setStyleSheet(_MONO + (" color:#3fb950;" if ok else " color:#d29922;"))
            layout.addWidget(row)

        safe = QLabel(
            "safe-default: intent/command 입력원이 없으므로 지금은 아무 명령도 "
            "디스패치되지 않는다(비실행이 안전한 기본값). 토큰 등 비밀값은 화면·로그에 "
            "노출하지 않는다."
        )
        safe.setWordWrap(True)
        safe.setStyleSheet("color:#8b949e; padding-top:8px;")
        layout.addWidget(safe)
        layout.addStretch(1)
        self.setWidget(body)


def _target_names(raw: str) -> list[str]:
    """Device-target *names* only — never values (they can hold ids/secrets)."""
    if not raw:
        return []
    try:
        import json

        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, dict):
        return [str(k) for k in parsed]
    return []


class VideoView(QLabel):
    """Displays webcam frames scaled to the widget, with a HUD and gaze overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#0b0e13;")
        self._fps_times: deque[float] = deque(maxlen=30)
        self._frame_count = 0
        self._gaze: GazeSnapshot | None = None
        self._show_placeholder("카메라 시작 중…")

    def set_gaze(self, snapshot: GazeSnapshot) -> None:
        self._gaze = snapshot

    def _show_placeholder(self, text: str) -> None:
        self._render(placeholder_frame(text=text))

    def show_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        self._fps_times.append(now)
        self._frame_count += 1
        fps = self._current_fps()
        h, w = frame.shape[:2]
        draw_hud(frame, [f"{w}x{h}  {fps:4.1f} FPS", f"frame #{self._frame_count}"])
        if self._gaze is not None:
            draw_gaze_overlay(frame, self._gaze)
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
        self.setStyleSheet(_MONO)

    def refresh(self) -> None:
        self.clear()
        for m in self._log.recent(50):
            self.addItem(f"[{m.timestamp_ms:>8} ms] {m.level}  {m.text}")
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
        start_camera: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS Pipeline Monitor")
        self.resize(1180, 820)
        self._log = MessageLog()
        self._gesture_source: GestureSource = NullGestureSource()
        self._env = env if env is not None else _load_env()
        self._model_path = model_path if model_path is not None else _default_model_path()
        self._profiles_path = profiles_path if profiles_path is not None else _default_profiles_path()
        self._latency = LatencyAggregator()

        self._probe = GazeProbe(
            model_path=self._model_path, profiles_path=self._profiles_path
        )

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "실시간")
        tabs.addTab(self._build_gaze_tab(), "Gaze 파이프라인")
        tabs.addTab(self._build_pipeline_tab(), "파이프라인")
        tabs.addTab(self._build_latency_tab(), "지연·어댑터")
        self.setCentralWidget(tabs)

        self._log.info("모니터 시작")
        if not self._gesture_source.available:
            self._log.warn(self._gesture_source.status_text)

        self._camera: CameraWorker | None = None
        if start_camera:
            if self._probe.start():
                self._log.info(f"gaze 프로브: {self._probe.status_text}")
            else:
                self._log.warn(f"gaze 프로브 비활성: {self._probe.status_text}")
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

    def _build_gaze_tab(self) -> QWidget:
        self._gaze_panel = GazePanel(self._probe.status_text)
        return self._gaze_panel

    def _build_pipeline_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        for status in detect_pipeline_status(self._env, self._model_path):
            layout.addWidget(StageCard(status))
        layout.addWidget(_header("아직 흐르지 않는 메시지 계약 (2인 파트 완성 시 채워짐)"))
        layout.addWidget(
            ContractPanel(
                "Gesture Spotter → Fusion", GestureEstimate, "손 랜드마크 기반 제스처·구간 추정."
            )
        )
        layout.addWidget(
            ContractPanel("Fusion → Protocol", Intent, "시선 타겟 + 제스처를 합쳐 만든 의도.")
        )
        layout.addWidget(
            ContractPanel("Protocol → Adapters", Command, "TTL·dedup을 거친 기기 실행 명령.")
        )
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        return scroll

    def _build_latency_tab(self) -> QWidget:
        self._latency_panel = LatencyPanel(self._latency)
        self._adapter_panel = AdapterPanel(self._env)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._latency_panel)
        split.addWidget(self._adapter_panel)
        return split

    def _start_camera(self, device_index: int) -> None:
        worker = CameraWorker(device_index, probe=self._probe)
        worker.frame_ready.connect(self._on_frame)
        worker.gaze_ready.connect(self._on_gaze)
        worker.failed.connect(self._on_camera_failed)
        self._camera = worker
        worker.start()
        self._log.info(f"카메라 {device_index}번 시작")

    def _on_frame(self, frame: np.ndarray) -> None:
        self._video.show_frame(frame)

    def _on_gaze(self, snapshot: object) -> None:
        assert isinstance(snapshot, GazeSnapshot)
        self._video.set_gaze(snapshot)
        self._gaze_panel.update_snapshot(snapshot)
        self._latency.record(LatencyStage.CAPTURE_TO_INFERENCE, snapshot.inference_ms)

    def _on_camera_failed(self, message: str) -> None:
        self._log.error(message)
        self._video._show_placeholder("NO CAMERA")

    def _on_tick(self) -> None:
        self._sidebar.poll()
        self._messages.refresh()
        self._latency_panel.refresh()

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        if self._camera is not None:
            self._camera.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


def _default_model_path() -> Path | None:
    # Always return the conventional path; GazeProbe/pipeline status explain its
    # absence honestly rather than silently disabling the stage.
    return Path("models/face_landmarker.task")


def _default_profiles_path() -> Path | None:
    return Path("data/calibration/profiles.json")


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(read_env_file(Path(".env")))
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    return int(app.exec())
