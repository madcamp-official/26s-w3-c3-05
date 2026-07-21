"""JARVIS real-time monitor — desktop application (PySide6).

Tabs:
- 실시간: live webcam (gaze + hand overlays) + gesture sidebar + messages.
- Gaze 파이프라인: every intermediate value of the real gaze engine, per frame
  (landmarks → vector → smoothing → classifier → lock → TargetEstimate).
- 손 추적: real MediaPipe hand landmarks per frame. Hand *tracking* is live;
  gesture *recognition* is honestly marked off (the classifier is untrained).
- 파이프라인: a card per stage's real availability + the message contracts.
- 지연·어댑터: measured per-stage latency and device-adapter readiness.

Display convention — the 실시간 and 손 추적 tabs render a horizontally-flipped
(selfie/거울상) view so the hand mirrors the user's real hand. This is a **display
concern only**: the frames and landmarks fed to MediaPipe, the model, and any
training-data logging are the un-flipped originals. Only the drawing is mirrored
(``cv2.flip`` on a display copy; overlays drawn with ``mirror=True``).

The window wires the parts that exist today (webcam capture, the gaze pipeline
when mediapipe+model+calibration are present, hand tracking when the hand model is
present, adapter/config detection) and honestly marks the parts that do not. No
detection is faked, and the untrained gesture model is never shown as recognizing.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QInputDialog,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.calibration.registry import TargetRecord, TargetRegistry
from jarvis.calibration.target_registration import TargetRegistrationSession
from jarvis.contracts.messages import Command, GestureEstimate, Intent
from jarvis.gaze.calibration_model import GazeCalibrationSample, GazeCalibrationStore
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_BACKGROUND_LABELS,
    DEFAULT_GESTURE_LABELS,
    EXPECTED_INPUT_FPS,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.lock import GazeLockState
from jarvis.gaze.smoothing import SmoothedGaze
from jarvis.gesture_fusion.pose_protocol import DEFAULT_POSE_TILT_LIMITS
from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.gaze_probe import GazeProbe, GazeSnapshot
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample
from jarvis.monitoring.gesture_probe import (
    GestureProbe,
    GestureSnapshot,
    ProbeGestureSource,
    load_trained_gesture_model,
)
from jarvis.monitoring.gesture_source import GestureSource, UntrainedGestureSource
from jarvis.monitoring.hand_probe import HandProbe, HandSnapshot
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.overlay import (
    Frame,
    draw_target_heatmap,
    draw_gaze_overlay,
    draw_hand_overlay,
    draw_hud,
    placeholder_frame,
    render_normalized_hand,
    render_vector,
)
from jarvis.monitoring.pipeline_status import StageState, StageStatus, detect_pipeline_status
from jarvis.monitoring.pose_control import PoseControlBridge, default_input_sink
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
_REGISTRATION_GUIDANCE_PHASES: tuple[tuple[int, str], ...] = (
    (3_000, "1/5 move face to LEFT-UP, keep looking at target"),
    (6_000, "2/5 move face to RIGHT-DOWN, keep looking at target"),
    (9_000, "3/5 move face to LEFT-DOWN, keep looking at target"),
    (12_000, "4/5 move face to RIGHT-UP, keep looking at target"),
    (15_000, "5/5 move slightly NEAR/FAR, keep looking at target"),
)
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
        if s.area_details:
            self._devices.addItem("[target area / edge loop]")
            for area_detail in s.area_details:
                mark = "selected" if area_detail.is_selected else ""
                self._devices.addItem(
                    f"{area_detail.device_id:<16} x{area_detail.normalized_distance:4.2f} "
                    f"{area_detail.range_status}  {mark}"
                )
        if s.feature_details:
            self._devices.addItem("[feature profile / Mahalanobis]")
            for feature_detail in s.feature_details:
                mark = "selected" if feature_detail.is_selected else ""
                self._devices.addItem(
                    f"{feature_detail.device_id:<16} dist {feature_detail.distance:5.2f} / thr {feature_detail.threshold:4.2f}  "
                    f"x{feature_detail.normalized_distance:4.2f} {feature_detail.range_status}  {mark}"
                )
            self._devices.addItem("[angle fallback]")
        if not s.device_details and not s.feature_details and not s.area_details:
            self._devices.addItem("no registered target profile")
        for device_detail in s.device_details:
            if np.isnan(device_detail.angular_distance_deg):
                angle = "err -- / radius --"
                ratio = "x--"
            else:
                angle = f"err {device_detail.angular_distance_deg:5.1f}deg / radius {device_detail.allowed_radius_deg:4.1f}deg"
                ratio = f"x{device_detail.normalized_distance:4.2f}"
            mark = "selected" if device_detail.is_selected else ""
            self._devices.addItem(
                f"{device_detail.device_id:<16} {angle}  {ratio} {device_detail.range_status}  {mark}"
            )
        self._track_conf.set_value(s.tracking_confidence, color="#58a6ff")
        self._gaze_conf.set_value(s.gaze_confidence, color="#58a6ff")
        self._stability.set_value(s.smoothed_stability, color="#58a6ff")

        direction = (
            "  ".join(f"{v:+.3f}" for v in s.gaze_direction)
            if s.gaze_direction is not None
            else "추적 손실 (None)"
        )
        feature = (
            "  ".join(
                (
                    f"gy={s.feature_sample.gaze_yaw:+.1f}",
                    f"gp={s.feature_sample.gaze_pitch:+.1f}",
                    f"hy={s.feature_sample.head_yaw:+.1f}",
                    f"hp={s.feature_sample.head_pitch:+.1f}",
                    f"hr={s.feature_sample.head_roll:+.1f}",
                    f"scale={s.feature_sample.face_scale:.3f}",
                )
            )
            if s.feature_sample is not None
            else "None"
        )
        motion = (
            f"dyaw={s.gaze_motion_delta_deg[0]:+.2f}  dpitch={s.gaze_motion_delta_deg[1]:+.2f}"
            if s.gaze_motion_delta_deg is not None
            else "None"
        )
        est = s.target_estimate
        self._numeric.setText(
            f"face_detected : {s.face_detected}\n"
            f"head (deg)    : yaw {s.head_yaw_deg:+7.2f}  pitch {s.head_pitch_deg:+7.2f}  "
            f"roll {s.head_roll_deg:+7.2f}\n"
            f"iris L / R    : {s.left_iris_relative}  /  {s.right_iris_relative}\n"
            f"face_scale    : {s.face_scale if s.face_scale is not None else 'None'}\n"
            f"gaze vector   : {direction}\n"
            f"feature       : {feature}\n"
            f"gaze motion   : {motion}\n"
            f"smoothing buf : {s.buffer_fill}/{s.buffer_capacity} frames\n"
            "── TargetEstimate (contract) ──────────────\n"
            f"target={est.target}  p={est.probability:.3f}  "
            f"p2={est.second_best_probability:.3f}  stability={est.stability:.3f}\n"
            f"frame_id={est.frame_id}  timestamp_ms={est.timestamp_ms}"
        )


class _AxisBar(QWidget):
    """One vector component: axis name + |value|/scale bar (signed color) + value."""

    def __init__(self, axis: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        name = QLabel(axis)
        name.setFixedWidth(14)
        name.setStyleSheet("color:#8b949e;")
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._value = QLabel("   --")
        self._value.setMinimumWidth(76)
        self._value.setStyleSheet(_MONO)
        layout.addWidget(name)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._value)

    def set_value(self, value: float | None, scale: float) -> None:
        if value is None:
            self._bar.setValue(0)
            self._value.setText("   --")
            return
        frac = min(1.0, abs(value) / scale) if scale > 0 else 0.0
        self._bar.setValue(int(frac * 1000))
        # green = 양(+) 방향, 주황 = 음(−) 방향 — 부호(=이동 방향)를 색으로도 드러낸다.
        color = "#3fb950" if value >= 0 else "#f0883e"
        self._bar.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}")
        self._value.setText(f"{value:+9.3f}")


class VectorView(QWidget):
    """A titled 2-component (x/y) vector display — one model-input signal.

    z(depth) is dropped from the pipeline (config.LANDMARK_DIMS), so wrist
    translation is an in-plane (x, y) signal. Bars are scaled by a running maximum
    magnitude (per view), so both the small velocities of a still hand and the large
    ones of a fast swipe stay readable without hardcoding a scale that only fits the demo.
    """

    _AXES = ("x", "y")

    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        self.setMinimumWidth(196)
        self.setMaximumWidth(268)
        layout = QVBoxLayout(self)
        layout.addWidget(_header(title))
        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#6e7681; font-size:11px;")
        layout.addWidget(sub)
        self._axes: dict[str, _AxisBar] = {}
        for axis in self._AXES:
            bar = _AxisBar(axis)
            self._axes[axis] = bar
            layout.addWidget(bar)
        self._magnitude = QLabel("‖·‖    --")
        self._magnitude.setStyleSheet(_MONO + " color:#58a6ff;")
        layout.addWidget(self._magnitude)
        layout.addStretch(1)
        self._scale = 1e-6  # running max component magnitude, floors bar scaling

    def set_vector(self, vec: tuple[float, float] | None) -> None:
        if vec is None:
            for bar in self._axes.values():
                bar.set_value(None, self._scale)
            self._magnitude.setText("‖·‖    --   (히스토리 없음)")
            return
        self._scale = max(self._scale, abs(vec[0]), abs(vec[1]))
        for axis, value in zip(self._AXES, vec, strict=True):
            self._axes[axis].set_value(value, self._scale)
        magnitude = math.sqrt(sum(component * component for component in vec))
        self._magnitude.setText(f"‖·‖ {magnitude:9.3f}")


class VectorArrowView(QWidget):
    """A titled 2D vector shown as a direction+magnitude arrow, not per-axis bars.

    Same auto-scaling idea as ``VectorView`` (running max magnitude, so a still
    hand's tiny jitter and a fast swipe both stay readable) but drawn as an actual
    arrow via ``overlay.render_vector`` — closer to "seeing" the motion than
    reading two independent x/y bars.
    """

    def __init__(self, title: str, subtitle: str, *, mirror: bool = False) -> None:
        super().__init__()
        self.setMinimumWidth(196)
        self.setMaximumWidth(268)
        layout = QVBoxLayout(self)
        layout.addWidget(_header(title))
        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#6e7681; font-size:11px;")
        layout.addWidget(sub)
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet("background:#12141a; border:1px solid #30363d;")
        layout.addWidget(self._canvas)
        layout.addStretch(1)
        self._scale = 1e-6  # running max magnitude, floors arrow scaling
        self._mirror = mirror

    def set_vector(self, vec: tuple[float, float] | None) -> None:
        if vec is not None:
            self._scale = max(self._scale, math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]))
        frame = render_vector(vec, scale=self._scale, mirror=self._mirror)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._canvas.setPixmap(QPixmap.fromImage(image))


class HandPanel(QScrollArea):
    """Live MediaPipe hand-tracking view + honest gesture-recognition status."""

    def __init__(
        self,
        probe_status: str,
        gesture_status: str,
        *,
        smoothing: bool = True,
        on_smoothing_toggled: Callable[[bool], None] | None = None,
        on_control_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        body = QWidget()
        outer = QVBoxLayout(body)

        self._status = QLabel(probe_status)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#d29922; padding:2px 0;")
        outer.addWidget(self._status)

        # Three columns: the wide empty space either side of the (square) model-input
        # canvas now shows the two wrist-translation vectors the model consumes.
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._velocity_view = VectorArrowView(
            "손목 이동 속도 (모델 입력)",
            "정규화 손목 평행이동의 속도 (palm-width/s · x·y). 손목 원점 정규화로 "
            "landmark 블록에서 지워지는 swipe 신호를 이 벡터가 담아 모델에 넣는다. "
            "화살표는 웹캠과 맞춘 거울상(좌우 반전)이며 표시 전용 — 모델 입력은 원본 그대로다.",
            mirror=True,
        )
        columns.addWidget(self._velocity_view, 0)

        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)

        # The faithful debug view: the exact normalized landmarks the model consumes.
        layout.addWidget(_header("모델 입력 정점 (실제 · 정규화 좌표)"))
        self._model_canvas = QLabel()
        self._model_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_canvas.setStyleSheet("background:#12141a; border:1px solid #30363d;")
        layout.addWidget(self._model_canvas)

        # Toggle: show the model input smoothed (real, default) or raw, to compare.
        self._smooth_toggle = QCheckBox("스무딩 적용 (모델이 실제로 쓰는 입력 · 끄면 raw 정규화 정점)")
        self._smooth_toggle.setChecked(smoothing)
        if on_smoothing_toggled is not None:
            self._smooth_toggle.toggled.connect(on_smoothing_toggled)
        layout.addWidget(self._smooth_toggle)

        # 실제 OS 제어 스위치. 이 경로는 사용자의 진짜 데스크톱을 클릭하고 스크롤한다.
        self._control_toggle = QCheckBox("🖱 손동작으로 컴퓨터 제어 (실제 클릭·스크롤이 실행됩니다)")
        self._control_toggle.setChecked(True)
        self._control_toggle.setStyleSheet("color:#f0b429; font-weight:600;")
        if on_control_toggled is not None:
            self._control_toggle.toggled.connect(on_control_toggled)
        layout.addWidget(self._control_toggle)
        self._control_status = QLabel("제어 꺼짐")
        self._control_status.setStyleSheet(_MONO)
        layout.addWidget(self._control_status)

        # The one thing that must not be misread: recognition is OFF (untrained).
        banner = QLabel("⚠ 제스처 인식 비활성\n" + gesture_status)
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background:#3d2a12; color:#f0b429; border:1px solid #7a5a1e;"
            " border-radius:6px; padding:8px; font-weight:600;"
        )
        layout.addWidget(banner)

        layout.addWidget(_header("손 랜드마크 추적 (MediaPipe Hands, 실제)"))
        self._detected = QLabel("hand detected : —")
        self._detected.setStyleSheet(_MONO)
        layout.addWidget(self._detected)
        self._det_conf = LabeledBar("detection confidence", threshold=0.50)
        layout.addWidget(self._det_conf)
        self._numeric = QLabel("웹캠·mediapipe·hand 모델이 준비되면 실시간 값이 표시됩니다.")
        self._numeric.setStyleSheet(_MONO)
        layout.addWidget(self._numeric)

        note = QLabel(
            "위 캔버스는 모델이 실제로 소비하는 정점(정규화·손목 원점)을 그대로 그린다 — "
            "웹캠 스켈레톤(이미지 좌표)과 달리 이게 모델 입력이다. 손목이 원점(파란 점)이라 "
            "이 캔버스에서 빠지는 손 전체의 평행이동(swipe)은 양옆의 손목 이동 속도·가속도 "
            "벡터가 대신 담아 모델에 함께 들어간다. 그래서 웹캠이 흔들려도 이 캔버스는 안정적이다. "
            "표시는 웹캠과 맞춘 거울상(좌우 반전)이지만 이는 화면 표시 전용이며 "
            "모델 입력·학습 데이터는 반전하지 않은 원본이다. "
            "제스처 이름·phase는 분류 모델 학습 후에만 의미가 있어 지금은 표시하지 않는다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6e7681; padding-top:8px;")
        layout.addWidget(note)
        layout.addStretch(1)
        columns.addWidget(center, 1)

        self._accel_view = VectorView(
            "손목 이동 가속도 (모델 입력)",
            "손목 이동 속도의 프레임 간 변화 (palm-width/s² · x·y). swipe의 시작·끝 "
            "(가속·감속) 순간을 드러내 onset/ending 판정을 돕는다.",
        )
        columns.addWidget(self._accel_view, 0)

        outer.addLayout(columns)
        outer.addStretch(1)
        self.setWidget(body)

    def update_snapshot(self, s: HandSnapshot, control_action: str = "") -> None:
        self._status.setText(f"frame #{s.frame_id} · {s.inference_ms:.0f} ms/frame")
        self._status.setStyleSheet("color:#3fb950;" if s.hand_detected else "color:#8b949e;")
        self._detected.setText(
            f"hand detected : {s.hand_detected}"
            + (f"   handedness : {s.handedness} ({s.handedness_score:.0%})" if s.hand_detected else "")
        )
        self._det_conf.set_value(
            s.detection_confidence if s.hand_detected else 0.0, color="#58a6ff"
        )

        # Render the actual model input: smoothed (real) or raw normalized per toggle.
        points = s.model_points if s.smoothed else s.model_points_raw
        # Mirror to match the selfie (거울상) webcam view — display only; points unchanged.
        canvas = render_normalized_hand(points, smoothed=s.smoothed, mirror=True)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self._model_canvas.setPixmap(QPixmap.fromImage(image))

        enabled = self._control_toggle.isChecked()
        self._control_status.setText(
            ("제어 켜짐 · " if enabled else "제어 꺼짐 · ") + (control_action or "대기")
        )
        self._control_status.setStyleSheet(_MONO + ("color:#3fb950;" if enabled else ""))
        if s.hand_detected:
            mode = "스무딩됨 (모델 실제 입력)" if s.smoothed else "raw (스무딩 꺼짐)"
            # 기울기 판정도 자세별 한계를 따른다 — 전역 한계(20°)를 보여주면 40°까지
            # 허용되는 two_fingers에 "손을 세우세요"가 떠 오버레이와 어긋난다.
            if s.palm_tilt_degrees is None:
                tilt = "?  (소스가 z를 내지 않아 게이트 없음)"
            elif s.pose is not None and s.pose.label:
                limit = DEFAULT_POSE_TILT_LIMITS.get(s.pose.label)
                allowed = "허용 ?" if limit is None else f"허용 {limit:.0f}°"
                verdict = "정상" if s.pose.trusted else "판정 거부 — 손을 세우세요"
                tilt = f"{s.palm_tilt_degrees:5.1f}°  / {allowed} ({s.pose.label})  {verdict}"
            else:
                tilt = f"{s.palm_tilt_degrees:5.1f}°  " + (
                    "판정 거부 — 손을 세우세요" if s.palm_tilted else "정상"
                )
            if s.pose is None:
                pose_line = "-"
            elif s.pose.trusted:
                pose_line = f"{s.pose.label}  ({s.pose.confidence:.0%})"
            else:
                pose_line = f"거부 — {s.pose.reason}" + (
                    f"  [{s.pose.label} {s.pose.confidence:.0%}]" if s.pose.label else ""
                )
            self._numeric.setText(
                f"모델 입력   : {mode}\n"
                f"자세 판정   : {pose_line}\n"
                f"확정 상태   : {s.pose_state or '—'}"
                + (f"   → {', '.join(e.kind for e in s.pose_events)}" if s.pose_events else "")
                + "\n"
                f"손 기울기   : {tilt}\n"
                f"palm scale  : {s.palm_scale:.4f}\n"
                f"landmarks   : {s.landmark_count} points (정규화·손목 원점)\n"
                f"handedness  : {s.handedness}  score {s.handedness_score:.3f}"
            )
        else:
            self._numeric.setText("손 없음 (추적 손실)")

        # 양옆 벡터 뷰: 모델에 들어가는 손목 평행이동 속도·가속도(추적 손실·첫 프레임 None).
        self._velocity_view.set_vector(s.wrist_velocity)
        self._accel_view.set_vector(s.wrist_acceleration)


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
        self._hand: HandSnapshot | None = None
        self._control_action = ""
        self._control_enabled = False
        self._show_target_heatmap = False
        self._show_placeholder("카메라 시작 중…")

    def set_gaze(self, snapshot: GazeSnapshot) -> None:
        self._gaze = snapshot

    def set_target_heatmap_visible(self, visible: bool) -> None:
        self._show_target_heatmap = visible

    def set_hand(self, snapshot: HandSnapshot) -> None:
        self._hand = snapshot

    def set_control(self, action: str, enabled: bool) -> None:
        """실시간 탭에도 제어 상태를 띄운다 — 3번 탭에만 있으면 자세를 보며 못 고친다."""
        self._control_action = action
        self._control_enabled = enabled

    def _show_placeholder(self, text: str) -> None:
        self._render(placeholder_frame(text=text))

    def show_frame(self, frame: Frame) -> None:
        now = time.monotonic()
        self._fps_times.append(now)
        self._frame_count += 1
        fps = self._current_fps()
        # Mirror to a selfie (거울상) view for display only. cv2.flip returns a new
        # array, so the original frame handed to MediaPipe / training stays un-flipped;
        # overlays are drawn with mirror=True to line up, and text stays readable since
        # it is drawn onto the already-flipped frame at un-mirrored positions.
        display = cast("Frame", cv2.flip(frame, 1))
        h, w = display.shape[:2]
        draw_hud(display, [f"{w}x{h}  {fps:4.1f} FPS", f"frame #{self._frame_count}"])
        if self._gaze is not None:
            if self._show_target_heatmap:
                draw_target_heatmap(display, self._gaze, mirror=True)
            draw_gaze_overlay(display, self._gaze, mirror=True)
        if self._hand is not None:
            draw_hand_overlay(
                display,
                self._hand,
                mirror=True,
                control_action=self._control_action,
                control_enabled=self._control_enabled,
            )
        self._render(display)

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
    """Terminal-style live readout of per-inference recognition + a hand-track line.

    The terminal streams one line per model inference tick (decimated to the training
    cadence, ``EXPECTED_INPUT_FPS``) — gesture · confidence · phase — like a log. It
    stays empty while the classifier is untrained/absent (nothing is fed) — no
    fabricated detections; the status header says why. The hand line reflects real
    MediaPipe tracking so the sidebar shows live signal even with recognition off.
    """

    _TERMINAL_STYLE = (
        "QListWidget{background:#0a0d12; border:1px solid #1f2630;"
        " font-family:Consolas,monospace; font-size:11px; color:#c9d1d9;}"
    )
    _MAX_LINES = 300

    def __init__(self, source: GestureSource) -> None:
        super().__init__()
        self._source = source
        layout = QVBoxLayout(self)
        title = QLabel(f"인식 스트림 ({int(EXPECTED_INPUT_FPS)}fps · frame · gesture · conf · phase)")
        title.setStyleSheet("font-weight:600; color:#58a6ff; padding:4px 0;")
        self._status = QLabel(source.status_text)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#8b949e;" if source.available else "color:#d29922;")
        self._hand_line = QLabel("손 추적: —")
        self._hand_line.setWordWrap(True)
        self._hand_line.setStyleSheet(_MONO)
        self._terminal = QListWidget()
        self._terminal.setStyleSheet(self._TERMINAL_STYLE)
        layout.addWidget(title)
        layout.addWidget(self._status)
        layout.addWidget(self._hand_line)
        layout.addWidget(self._terminal, 1)
        self.setMinimumWidth(240)

    def set_hand_status(self, snapshot: HandSnapshot) -> None:
        if snapshot.hand_detected:
            label = snapshot.handedness or "?"
            self._hand_line.setText(f"손 추적: {label} 검출 (det {snapshot.detection_confidence:.0%})")
            self._hand_line.setStyleSheet(_MONO + " color:#3fb950;")
        else:
            self._hand_line.setText("손 추적: 손 없음")
            self._hand_line.setStyleSheet(_MONO + " color:#8b949e;")

    def append_tick(self, snapshot: GestureSnapshot) -> None:
        """Append one recognition line for a processed inference tick (~12fps)."""
        estimate = snapshot.estimate
        phase = getattr(estimate.phase, "name", str(estimate.phase))
        if not snapshot.hand_detected:
            text = f"{snapshot.frame_id:>6}  {'— 손 없음':<18} {'':>4}  {phase}"
            color = "#6e7681"
        else:
            text = (
                f"{snapshot.frame_id:>6}  {estimate.gesture:<18} "
                f"{estimate.gesture_confidence:>4.0%}  {phase}"
            )
            # 배경(none·drumming·doing_other)은 흐리게, 액션 제스처는 초록으로 강조.
            color = "#6e7681" if estimate.gesture in DEFAULT_BACKGROUND_LABELS else "#3fb950"
        self._terminal.addItem(text)
        item = self._terminal.item(self._terminal.count() - 1)
        if item is not None:
            item.setForeground(QColor(color))
        while self._terminal.count() > self._MAX_LINES:
            self._terminal.takeItem(0)
        self._terminal.scrollToBottom()


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


class _RecordResult(Protocol):
    """`ClipRecorder.stop()` 반환값의 구조적 타입(training 패키지 static import 회피)."""

    saved: bool
    detail: str
    frame_count: int


class _ClipRecorder(Protocol):
    """`training.clip_recorder.ClipRecorder`의 구조적 타입.

    `training/`은 런타임 패키지(jarvis)와 분리돼 있어(pyproject 39행) static import
    하지 않는다 — 이 Protocol로 타입만 잡고 실제 클래스는 importlib로 런타임에 붙인다.
    """

    @property
    def is_recording(self) -> bool: ...

    @property
    def frame_count(self) -> int: ...

    def start(self, person_id: str, gesture_label: str) -> None: ...

    def add(self, observation: HandObservation) -> None: ...

    def stop(self) -> _RecordResult: ...


def _build_clip_recorder() -> _ClipRecorder | None:
    """training.ClipRecorder를 동적으로 만든다 — 실패하면 None(녹화 정직하게 비활성).

    training 패키지가 없거나(런타임 전용 배포) import에 실패하면 녹화를 비활성화하고
    UI가 그 사실을 드러낸다(성공을 가장하지 않는다). 저장은 학습 cadence(12fps)로
    리샘플되도록 `target_fps=EXPECTED_INPUT_FPS`를 준다.
    """
    try:
        recorder_mod = importlib.import_module("training.clip_recorder")
        config_mod = importlib.import_module("training.config")
    except Exception:  # noqa: BLE001 - training 부재는 정상(런타임 전용 실행)일 수 있다
        return None
    cfg = config_mod.DEFAULT_TRAINING_CONFIG
    recorder = recorder_mod.ClipRecorder(
        cache_dir=cfg.cache_dir,
        max_missing_frame_fraction=cfg.max_missing_frame_fraction,
        target_fps=EXPECTED_INPUT_FPS,
    )
    return cast("_ClipRecorder", recorder)


class FinetuneRecordingPanel(QWidget):
    """파인튜닝 클립 녹화 탭 — 웹캠+스켈레톤 뷰 + 동작 선택 + person_id + REC 버튼.

    표시 계층이라 저장소를 직접 모른다(HandPanel의 `on_smoothing_toggled`과 같은 규약):
    REC 버튼은 `on_record_toggled` 콜백만 부르고, 실제 녹화 상태·저장은 MainWindow가
    ClipRecorder로 처리한 뒤 `set_recording`/`set_status`로 결과를 돌려준다.
    """

    def __init__(
        self,
        *,
        gesture_labels: tuple[str, ...],
        recording_available: bool,
        on_record_toggled: Callable[[], None],
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        # 웹캠+스켈레톤 뷰(실시간 탭과 같은 VideoView) — 녹화 중 손 프레이밍 확인용.
        # gaze는 설정하지 않으므로 손 오버레이만 그려진다.
        self.video = VideoView()
        layout.addWidget(self.video, 1)

        note = QLabel(
            f"웹캠에서 직접 파인튜닝 클립을 녹화한다. 저장 직전 {EXPECTED_INPUT_FPS:.0f}fps로 "
            "리샘플해 사전학습(Jester 12fps)과 정합시킨다. person_id는 세션 단위 split을 위해 "
            "'<이름>-s<세션>' 형식을 권장한다(예: me-s1). 저장 위치: "
            "training/cache/webcam/<person_id>/."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6e7681; font-size:11px;")
        layout.addWidget(note)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("person_id"))
        self._person_edit = QLineEdit()
        self._person_edit.setPlaceholderText("예: me-s1")
        controls.addWidget(self._person_edit, 1)
        controls.addWidget(QLabel("동작"))
        self._gesture_combo = QComboBox()
        for label in gesture_labels:
            self._gesture_combo.addItem(label)
        controls.addWidget(self._gesture_combo, 1)
        self._record_button = QPushButton("녹화 시작")
        self._record_button.clicked.connect(lambda: on_record_toggled())
        controls.addWidget(self._record_button)
        layout.addLayout(controls)

        self._status = QLabel(
            "녹화 대기"
            if recording_available
            else "녹화 비활성 — training 패키지를 불러올 수 없습니다(런타임 전용 실행)"
        )
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#8b949e; padding:4px 0;")
        layout.addWidget(self._status)

        if not recording_available:
            self._record_button.setEnabled(False)
            self._person_edit.setEnabled(False)
            self._gesture_combo.setEnabled(False)

    @property
    def person_id(self) -> str:
        return self._person_edit.text().strip()

    @property
    def gesture(self) -> str:
        return self._gesture_combo.currentText()

    def set_recording(self, is_recording: bool) -> None:
        self._record_button.setText("녹화 종료" if is_recording else "녹화 시작")
        # 녹화 중엔 person/동작 잠금 — 한 클립 도중 라벨이 바뀌면 안 된다.
        self._person_edit.setEnabled(not is_recording)
        self._gesture_combo.setEnabled(not is_recording)

    def set_status(self, text: str, *, ok: bool = True) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(("color:#3fb950;" if ok else "color:#d29922;") + " padding:4px 0;")


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        device_index: int = 0,
        env: dict[str, str] | None = None,
        model_path: Path | None = None,
        profiles_path: Path | None = None,
        hand_model_path: Path | None = None,
        samples_path: Path | None = None,
        gesture_models_dir: Path | None = None,
        start_camera: bool = True,
        gaze_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS Pipeline Monitor")
        self.resize(1180, 820)
        self._log = MessageLog()
        # 학습된 체크포인트가 models/에 있으면 실시간 탭에 인식을 표시한다(finetuned 우선,
        # 없으면 Jester 사전학습). 없거나 torch 미설치면 정직하게 off(UntrainedGestureSource).
        # 인식은 HandProbe의 관측값을 재사용하므로 두 번째 landmarker를 돌리지 않는다.
        # models_dir은 테스트에서 주입 가능(기본은 models/) — 앰비언트 체크포인트 유무에
        # 좌우되지 않게 한다.
        self._gesture_models_dir = (
            gesture_models_dir if gesture_models_dir is not None else _default_models_dir()
        )
        self._recognizer: GestureProbe | None = None
        self._gesture_source: GestureSource = self._make_gesture_source()
        self._env = env if env is not None else _load_env()
        self._model_path = model_path if model_path is not None else _default_model_path()
        self._profiles_path = profiles_path if profiles_path is not None else _default_profiles_path()
        self._hand_model_path = (
            hand_model_path if hand_model_path is not None else _default_hand_model_path()
        )
        self._latency = LatencyAggregator()
        self._latest_gaze: GazeSnapshot | None = None
        self._gaze_history: deque[GazeSnapshot] = deque()
        self._sample_store = GazeSampleStore(
            samples_path or Path("data/evaluation/gaze_samples.json")
        )
        self._gaze_config = GazeConfig(
            enable_3d_target_matching=False,
            require_3d_target_registration=False,
        )
        self._calibration_store = GazeCalibrationStore(_default_calibration_model_path())
        self._target_registry = TargetRegistry(self._profiles_path)
        # Diagnostic experiment #1: keep learned gaze calibration OFF so raw/head
        # composition can be inspected without a regression model shifting final y/p.
        self._active_calibration_model = None
        self._registration: TargetRegistrationSession | None = None
        self._registration_points: list[tuple[float, float]] = []
        self._registration_calibration_features: list[tuple[float, ...]] = []
        self._registration_phase_index: int | None = None
        self._target_list = QListWidget()
        self._register_target_button = QPushButton()
        self._probe = GazeProbe(
            model_path=self._model_path,
            profiles_path=self._profiles_path,
            config=self._gaze_config,
            calibration_model=self._active_calibration_model,
        )
        self._hand_probe = HandProbe(
            model_path=self._hand_model_path, pose_model_path=_default_pose_model_path()
        )
        # 판정과 실행의 분리: 상태기계는 순수 로직이고, 실제 OS 입력은 이 브리지만 한다.
        self._pose_control = PoseControlBridge(sink=default_input_sink(), enabled=True)

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "실시간")
        tabs.addTab(self._build_gaze_tab(), "Gaze 파이프라인")
        tabs.addTab(self._build_hand_tab(), "손 추적")
        tabs.addTab(self._build_pipeline_tab(), "파이프라인")
        tabs.addTab(self._build_latency_tab(), "지연·어댑터")
        tabs.addTab(self._build_finetune_tab(), "파인튜닝")
        self.setCentralWidget(tabs)

        self._log.info("모니터 시작")
        if self._gesture_source.available:
            self._log.info(f"제스처 인식: {self._gesture_source.status_text}")
        else:
            self._log.warn(self._gesture_source.status_text)

        self._camera: CameraWorker | None = None
        if start_camera:
            if not gaze_enabled:
                # 손 추적 지연 진단용: gaze(FaceLandmarker) 추론을 아예 돌리지 않아
                # CameraWorker가 프레임당 hand 모델 하나만 실행하게 한다(2026-07-20).
                self._log.info("gaze 프로브: --no-gaze로 비활성화됨")
            elif self._probe.start():
                self._log.info(f"gaze 프로브: {self._probe.status_text}")
            else:
                self._log.warn(f"gaze 프로브 비활성: {self._probe.status_text}")
            if self._hand_probe.start():
                self._log.info(f"hand 프로브: {self._hand_probe.status_text}")
            else:
                self._log.warn(f"hand 프로브 비활성: {self._hand_probe.status_text}")
            self._start_camera(device_index)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(150)

    def _make_gesture_source(self) -> GestureSource:
        """학습된 체크포인트가 있으면 관측값 재사용 인식 소스를, 없으면 정직한 off 소스를 만든다.

        인식은 HandProbe의 관측값을 재사용하므로(두 번째 landmarker 없음) 여기서 만든
        `GestureProbe`는 landmarker를 시작하지 않고 `activate_headless()`로 모델·윈도우만
        준비한다. 프레임 구동은 `_on_hand`가 `advance(observation)`로 한다.
        """
        model = load_trained_gesture_model(self._gesture_models_dir)
        if model is None:
            return UntrainedGestureSource()
        probe = GestureProbe(model_asset_path=None, model=model)
        if not probe.activate_headless():
            return UntrainedGestureSource()
        self._recognizer = probe
        return ProbeGestureSource(probe)

    def _build_live_tab(self) -> QWidget:
        self._video = VideoView()
        self._sidebar = GestureSidebar(self._gesture_source)
        # 손 추적 탭과 상태를 공유하는 제어 토글 — 자세를 취하며 보는 화면이 실시간
        # 탭이라 여기서도 켜고 끌 수 있어야 한다.
        self._control_toggle_live = QCheckBox("🖱 손동작으로 컴퓨터 제어 (실제 클릭·스크롤이 실행됩니다)")
        self._control_toggle_live.setChecked(True)
        self._control_toggle_live.setStyleSheet("color:#f0b429; font-weight:600;")
        self._control_toggle_live.toggled.connect(self._on_control_toggled_live)
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
        layout.addWidget(self._control_toggle_live)
        layout.addWidget(split)
        self._sample_button = QPushButton()
        self._sample_button.clicked.connect(self._save_gaze_sample)
        self._clear_samples_button = QPushButton("샘플 초기화")
        self._clear_samples_button.clicked.connect(self._clear_gaze_samples)
        self._target_heatmap_toggle = QCheckBox("Target heatmap / 물체 영역 표시")
        self._target_heatmap_toggle.toggled.connect(self._video.set_target_heatmap_visible)
        sample_controls = QHBoxLayout()
        sample_controls.addWidget(self._sample_button, 1)
        sample_controls.addWidget(self._clear_samples_button)
        sample_controls.addWidget(self._target_heatmap_toggle)
        layout.addLayout(sample_controls)
        self._sample_list = QListWidget()
        self._sample_list.setMaximumHeight(130)
        self._sample_list.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")
        for sample in self._sample_store.samples:
            self._sample_list.addItem(format_gaze_sample(sample))
        layout.addWidget(self._sample_list)
        self._refresh_sample_button()
        self._registration_status = QLabel("등록 대기")
        self._registration_status.setStyleSheet(
            "background:#161b22; color:#8b949e; border:1px solid #30363d;"
            " border-radius:6px; padding:8px; font-weight:600;"
        )
        layout.addWidget(self._registration_status)

        target_controls = QHBoxLayout()
        self._register_target_button = QPushButton("물체 등록")
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

    def _build_gaze_tab(self) -> QWidget:
        self._gaze_panel = GazePanel(self._probe.status_text)
        return self._gaze_panel

    def _build_hand_tab(self) -> QWidget:
        self._hand_panel = HandPanel(
            self._hand_probe.status_text,
            self._hand_probe.gesture_recognition_status,
            smoothing=self._hand_probe.smoothing,
            on_smoothing_toggled=self._hand_probe.set_smoothing,
            on_control_toggled=self._set_control_enabled,
        )
        return self._hand_panel

    def _build_finetune_tab(self) -> QWidget:
        # 녹화 코어는 training 패키지에 있다(런타임 패키지와 분리) — 동적으로 붙이고,
        # 없으면 패널이 녹화 비활성 상태로 뜬다.
        self._recorder: _ClipRecorder | None = _build_clip_recorder()
        self._finetune_panel = FinetuneRecordingPanel(
            gesture_labels=DEFAULT_GESTURE_LABELS,
            recording_available=self._recorder is not None,
            on_record_toggled=self._toggle_recording,
        )
        if self._recorder is None:
            self._log.warn("파인튜닝 녹화 비활성: training 패키지를 불러올 수 없습니다")
        return self._finetune_panel

    def _toggle_recording(self) -> None:
        if self._recorder is None:
            return
        if self._recorder.is_recording:
            result = self._recorder.stop()
            self._finetune_panel.set_recording(False)
            self._finetune_panel.set_status(result.detail, ok=result.saved)
            self._log.info(f"파인튜닝 녹화: {result.detail}")
            return
        person = self._finetune_panel.person_id
        if not person:
            self._finetune_panel.set_status("person_id를 입력하세요 (예: me-s1)", ok=False)
            return
        gesture = self._finetune_panel.gesture
        self._recorder.start(person, gesture)
        self._finetune_panel.set_recording(True)
        self._finetune_panel.set_status(f"녹화 중… [{gesture}] 0프레임")
        self._log.info(f"파인튜닝 녹화 시작: person={person} 동작={gesture}")

    def _on_control_toggled_live(self, enabled: bool) -> None:
        """실시간 탭 토글 → 손 추적 탭 토글과 상태를 맞춘다(무한 재귀 없이)."""
        panel = self._hand_panel._control_toggle
        if panel.isChecked() != enabled:
            panel.blockSignals(True)
            panel.setChecked(enabled)
            panel.blockSignals(False)
        self._set_control_enabled(enabled)

    def _set_control_enabled(self, enabled: bool) -> None:
        """실제 OS 제어를 켜고 끈다. 끌 때는 눌린 버튼을 반드시 놓는다."""
        # 손 추적 탭에서 토글되면 실시간 탭 토글도 맞춘다.
        if hasattr(self, "_control_toggle_live") and self._control_toggle_live.isChecked() != enabled:
            self._control_toggle_live.blockSignals(True)
            self._control_toggle_live.setChecked(enabled)
            self._control_toggle_live.blockSignals(False)
        if not enabled:
            self._pose_control.release()  # 드래그 중 껐을 때 버튼이 눌린 채 남지 않게
        self._pose_control.enabled = enabled
        if enabled and self._pose_control.sink is None:
            self._log.warn("이 플랫폼에는 입력 어댑터가 없어 제어를 실행할 수 없습니다")
        else:
            self._log.info(f"손동작 제어 {'켜짐' if enabled else '꺼짐'}")

    def _build_pipeline_tab(self) -> QWidget:
        body = QWidget()
        layout = QVBoxLayout(body)
        for status in detect_pipeline_status(self._env, self._model_path):
            layout.addWidget(StageCard(status))
        layout.addWidget(_header("모듈 경계 메시지 계약"))
        layout.addWidget(
            ContractPanel(
                "Gesture Spotter → Fusion",
                GestureEstimate,
                "손 랜드마크 기반 제스처·구간 추정. 코드는 구현됨 — 단 분류 모델 미학습이라 "
                "아직 실제 제스처로 흐르지 않음.",
            )
        )
        layout.addWidget(
            ContractPanel(
                "Fusion → Protocol", Intent, "시선 타겟 + 제스처를 합쳐 만든 의도 (모델 학습 후 활성)."
            )
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
        worker = CameraWorker(device_index, probe=self._probe, hand_probe=self._hand_probe)
        worker.frame_ready.connect(self._on_frame)
        worker.gaze_ready.connect(self._on_gaze)
        worker.hand_ready.connect(self._on_hand)
        worker.failed.connect(self._on_camera_failed)
        self._camera = worker
        worker.start()
        self._log.info(f"카메라 {device_index}번 시작")

    def _on_frame(self, frame: Frame) -> None:
        self._video.show_frame(frame)
        # 파인튜닝 탭의 웹캠 뷰에도 같은 프레임을 뿌린다(Qt 시그널 fan-out — 각 VideoView가
        # 자기 복사본에 그리므로 서로 간섭하지 않는다).
        self._finetune_panel.video.show_frame(frame)

    def _on_gaze(self, snapshot: object) -> None:
        assert isinstance(snapshot, GazeSnapshot)
        self._latest_gaze = snapshot
        self._gaze_history.append(snapshot)
        cutoff_ms = snapshot.timestamp_ms - 500
        while self._gaze_history and self._gaze_history[0].timestamp_ms < cutoff_ms:
            self._gaze_history.popleft()
        self._video.set_gaze(snapshot)
        self._gaze_panel.update_snapshot(snapshot)
        self._latency.record(LatencyStage.CAPTURE_TO_INFERENCE, snapshot.inference_ms)
        if self._registration is not None:
            smoothed = self._smoothed_from_snapshot(snapshot)
            if (
                self._registration.add(
                    smoothed,
                    snapshot.tracking_confidence,
                    eyes_open=snapshot.eyes_open,
                    face_scale=snapshot.face_scale,
                    feature_sample=snapshot.feature_sample,
                )
                and smoothed is not None
            ):
                self._registration_points.append(direction_to_yaw_pitch(smoothed.direction))
                if snapshot.calibration_features is not None:
                    self._registration_calibration_features.append(snapshot.calibration_features)
            self._update_registration_guidance(snapshot.timestamp_ms)
            if self._registration.is_elapsed(snapshot.timestamp_ms):
                self._finish_target_registration()

    @staticmethod
    def _smoothed_from_snapshot(snapshot: GazeSnapshot) -> SmoothedGaze | None:
        if snapshot.smoothed_gaze_direction is None or snapshot.smoothed_stability is None:
            return None
        origin = (
            np.array(snapshot.smoothed_gaze_origin, dtype=np.float64)
            if snapshot.smoothed_gaze_origin is not None
            else None
        )
        return SmoothedGaze(
            direction=np.array(snapshot.smoothed_gaze_direction, dtype=np.float64),
            stability=snapshot.smoothed_stability,
            timestamp_ms=snapshot.timestamp_ms,
            frame_id=snapshot.frame_id,
            origin=origin,
        )

    def _selected_target(self) -> TargetRecord | None:
        row = self._target_list.currentRow()
        records = self._target_registry.records
        return records[row] if 0 <= row < len(records) else None

    def _start_target_registration(self) -> None:
        name, ok = QInputDialog.getText(self, "물체 등록", "물체 이름")
        if not ok or not name.strip():
            return
        target_id = self._next_target_id()
        self._begin_registration(target_id, name.strip(), "UNKNOWN", target_id)

    def _next_target_id(self) -> str:
        existing = {record.target_id for record in self._target_registry.records}
        index = 1
        while True:
            target_id = f"target_{index:03d}"
            if target_id not in existing:
                return target_id
            index += 1

    def _begin_registration(
        self, target_id: str, name: str, device_type: str, device_id: str
    ) -> None:
        if self._registration is not None:
            self._log.warn("이미 기기 등록이 진행 중입니다")
            return
        self._registration = TargetRegistrationSession(
            target_id, name, device_type, device_id, config=self._gaze_config
        )
        self._registration_points = []
        self._registration_calibration_features = []
        self._registration_phase_index = None
        self._register_target_button.setEnabled(False)
        self._registration_status.setText(
            f"'{name}' registration ready: keep looking at the target; "
            "move face center diagonally and near/far for 15s"
        )
        self._registration_status.setStyleSheet(
            "background:#3d2a12; color:#f0b429; border:1px solid #7a5a1e;"
            " border-radius:6px; padding:8px; font-weight:700;"
        )
        self._log.info(
            f"'{name}' registration start: do not only rotate your head; "
            "move face/body LEFT-UP -> RIGHT-DOWN -> LEFT-DOWN -> RIGHT-UP -> NEAR/FAR "
            "while continuously looking at the target"
        )

    def _update_registration_guidance(self, timestamp_ms: int) -> None:
        assert self._registration is not None
        if self._registration.started_at_ms is None:
            self._registration_status.setText("등록 대기: 얼굴과 시선이 안정적으로 잡히길 기다리는 중")
            return
        elapsed_ms = max(0, timestamp_ms - self._registration.started_at_ms)
        phase_index = 0
        for index, (end_ms, _label) in enumerate(_REGISTRATION_GUIDANCE_PHASES):
            if elapsed_ms < end_ms:
                phase_index = index
                break
        else:
            phase_index = len(_REGISTRATION_GUIDANCE_PHASES) - 1
        phase_end_ms, label = _REGISTRATION_GUIDANCE_PHASES[phase_index]
        remaining_s = max(0.0, (phase_end_ms - elapsed_ms) / 1000.0)
        status = (
            f"등록 중: {label}  |  남은 {remaining_s:0.1f}s  |  "
            f"valid {self._registration.valid_frame_count}/{self._registration.minimum_valid_frames}"
        )
        self._registration_status.setText(status)
        if phase_index != self._registration_phase_index:
            self._registration_phase_index = phase_index
            self._log.info(status)

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
            self._probe.register_profile(
                record.to_profile(),
                geometry_3d=record.to_geometry_3d(),
                feature_profile=record.feature_profile,
                area_profile=record.area_profile,
                label=record.name,
            )
            calibration_samples = [
                GazeCalibrationSample(
                    features=features,
                    target_yaw=record.direction.yaw,
                    target_pitch=record.direction.pitch,
                )
                for features in self._registration_calibration_features
            ]
            model = self._calibration_store.add_samples(calibration_samples)
            self._active_calibration_model = None
            self._probe.set_calibration_model(self._active_calibration_model)
            self._log.info(
                f"'{record.name}' 방향 등록 완료 ({self._registration.valid_frame_count} frames) "
                f"— {self._describe_triangulation_outcome(record)} | "
                f"{self._registration.diagnostic_summary()} | "
                f"gaze calibration samples={model.sample_count}"
            )
        except ValueError as exc:
            self._log.warn(f"기기 등록 실패: {exc} | {self._registration.diagnostic_summary()}")
        finally:
            self._registration = None
            self._registration_points = []
            self._registration_calibration_features = []
            self._registration_phase_index = None
            self._register_target_button.setEnabled(True)
            self._registration_status.setText("등록 대기")
            self._registration_status.setStyleSheet(
                "background:#161b22; color:#8b949e; border:1px solid #30363d;"
                " border-radius:6px; padding:8px; font-weight:600;"
            )
            self._refresh_targets()

    def _describe_triangulation_outcome(self, record: TargetRecord) -> str:
        """3D 위치 추정이 성공했는지, 실패했다면 왜 각도 모드로 대체됐는지를
        그대로 보여준다 — 성공을 지어내지 않는다(development-principles.md 1절)."""
        assert self._registration is not None
        triangulation = self._registration.triangulation_result
        if record.position_3d is not None:
            if triangulation is None:
                return "3D 위치 추정 성공"
            return f"3D 위치 추정 성공 (baseline {triangulation.baseline_mm:.0f}mm)"
        if triangulation is None:
            return "각도 모드로 대체 (머리 움직임 데이터 부족)"
        return (
            "각도 모드로 대체 (baseline "
            f"{triangulation.baseline_mm:.0f}mm, 잔차 {triangulation.residual_rms_mm:.0f}mm, "
            f"고유값 {triangulation.min_eigenvalue:.4f} — 머리를 더 크게 움직여 보세요)"
        )

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
            updated = self._target_registry.rename(record.target_id, name.strip())
            self._probe.register_profile(
                updated.to_profile(),
                geometry_3d=updated.to_geometry_3d(),
                feature_profile=updated.feature_profile,
                area_profile=updated.area_profile,
                label=updated.name,
            )
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
        self._probe.unregister_profile(record.target_id)
        self._refresh_targets()

    def _refresh_targets(self) -> None:
        self._target_list.clear()
        for record in self._target_registry.records:
            scale_label = (
                f" scale={record.reference_face_scale:.3f}"
                if record.reference_face_scale is not None
                else ""
            )
            feature_label = (
                f" feature={record.feature_profile.sample_count}/thr{record.feature_profile.threshold:.2f}"
                if record.feature_profile is not None
                else " feature=none"
            )
            area_label = (
                f" area={record.area_profile.radius_yaw:.1f}/{record.area_profile.radius_pitch:.1f}"
                f" cap={self._gaze_config.registration_max_area_radius_deg:.1f}"
                if record.area_profile is not None
                else " area=none"
            )
            self._target_list.addItem(
                f"{record.name} [{record.target_id}]  "
                f"yaw={record.direction.yaw:+.1f} pitch={record.direction.pitch:+.1f}"
                f" spread={record.spread.yaw:.1f}/{record.spread.pitch:.1f}"
                f"{scale_label}{feature_label}{area_label}"
            )

    def _save_gaze_sample(self) -> None:
        if self._latest_gaze is None:
            self._log.warn("저장할 Gaze 값이 아직 없습니다")
            return
        try:
            sample = self._sample_store.add_window(list(self._gaze_history), minimum_frames=3)
        except ValueError as exc:
            self._log.warn(f"Gaze 샘플 저장 실패: {exc}")
            return
        self._sample_list.addItem(format_gaze_sample(sample))
        self._sample_list.scrollToBottom()
        self._log.info(
            f"Gaze 샘플 {sample['sample_index']}/{self._sample_store.capacity} 저장"
        )
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

    def _on_hand(self, snapshot: object) -> None:
        assert isinstance(snapshot, HandSnapshot)
        self._video.set_hand(snapshot)
        # 손을 놓치면 드래그가 눌린 채 남지 않게 먼저 정리한다.
        if not snapshot.hand_detected:
            self._pose_control.release()
        self._pose_control.apply(list(snapshot.pose_events))
        self._video.set_control(self._pose_control.last_action, self._pose_control.enabled)
        self._hand_panel.update_snapshot(snapshot, self._pose_control.last_action)
        self._sidebar.set_hand_status(snapshot)
        self._finetune_panel.video.set_hand(snapshot)
        # 학습된 모델이 로드됐으면 HandProbe의 관측값을 그대로 인식 파이프라인에 흘린다
        # (두 번째 landmarker 없이). advance는 12fps로 솎아 처리한 tick에서만 스냅샷을
        # 반환하므로(그 외엔 None), 그때마다 실시간 탭 터미널에 한 줄씩 찍는다.
        if self._recognizer is not None and snapshot.observation is not None:
            tick = self._recognizer.advance(snapshot.observation)
            # 검출된 tick만 스트림에 찍는다 — 추적 손실은 위의 손 추적 라인이 보여주므로
            # 터미널을 손실 프레임(솎이지 않아 30fps)으로 도배하지 않는다.
            if tick is not None and tick.hand_detected:
                self._sidebar.append_tick(tick)
        # 녹화 중이면 이 프레임의 관측값(스무딩 전 원본)을 클립 버퍼에 넣는다. observation은
        # 검출·손실 프레임 둘 다 존재하므로(hand_probe A단계) 미검출도 클립에 남아 미검출
        # 게이트가 CLI와 동일하게 동작한다.
        if (
            self._recorder is not None
            and self._recorder.is_recording
            and snapshot.observation is not None
        ):
            self._recorder.add(snapshot.observation)
            self._finetune_panel.set_status(
                f"녹화 중… [{self._finetune_panel.gesture}] {self._recorder.frame_count}프레임"
            )

    def _on_camera_failed(self, message: str) -> None:
        self._log.error(message)
        self._video._show_placeholder("NO CAMERA")

    def _on_tick(self) -> None:
        # 인식 스트림은 _on_hand에서 프레임마다(12fps로 솎아) 직접 찍는다 — 여기선 안 건드림.
        self._messages.refresh()
        self._latency_panel.refresh()

    def keyPressEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        """ESC로 창을 닫는다 — 카메라 정리는 closeEvent가 맡는다.

        디버깅 툴은 자세를 바꿔가며 반복해서 띄웠다 닫는 도구라, 창 버튼까지 마우스를
        옮기지 않고 닫을 수 있어야 한다.
        """
        if event.key() == Qt.Key.Key_Escape:  # type: ignore[attr-defined]
            self.close()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        if self._camera is not None:
            self._camera.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


def _default_model_path() -> Path | None:
    # Always return the conventional path; GazeProbe/pipeline status explain its
    # absence honestly rather than silently disabling the stage.
    return Path("models/face_landmarker.task")


def _default_hand_model_path() -> Path | None:
    return Path("models/hand_landmarker.task")


def _default_models_dir() -> Path:
    return Path("models")


def _default_pose_model_path() -> Path:
    """`training/train_pose.py`의 기본 산출물 경로. 없으면 프로브가 사유를 표시한다."""
    return Path("models/hand_pose_classifier.pt")


def _default_profiles_path() -> Path:
    return Path("data/calibration/profiles.json")


def _default_calibration_model_path() -> Path:
    return Path("data/calibration/gaze_regressor.json")


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(read_env_file(Path(".env")))
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    return int(app.exec())
