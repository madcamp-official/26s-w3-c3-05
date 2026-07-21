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
import math
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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
from jarvis.calibration.target_registration import RegistrationPhase, TargetRegistrationSession
from jarvis.contracts.messages import Command, GestureEstimate, Intent
from jarvis.gaze.calibration_model import (
    GazeCalibrationCorrector,
    GazeCalibrationSample,
    GazeCalibrationStore,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.lock import GazeLockState
from jarvis.gaze.personal_classifier import PersonalTargetStore
from jarvis.gaze.smoothing import SmoothedGaze
from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.gaze_probe import GazeProbe, GazeSnapshot
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample
from jarvis.monitoring.gesture_source import GestureSource, UntrainedGestureSource
from jarvis.monitoring.hand_probe import HandProbe, HandSnapshot
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.overlay import (
    Frame,
    draw_target_heatmap,
    draw_gaze_overlay,
    draw_hand_overlay,
    draw_hud,
    draw_registration_guidance,
    placeholder_frame,
    render_normalized_hand,
    render_vector,
)
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
_CENTER_GUIDANCE_PHASES: tuple[tuple[int, str, str], ...] = (
    (4_000, "중앙점을 보면서 얼굴·몸을 왼쪽 위로 천천히 이동", "EYES CENTER - FACE LEFT-UP"),
    (8_000, "중앙점을 보면서 얼굴·몸을 오른쪽 아래로 천천히 이동", "EYES CENTER - FACE RIGHT-DOWN"),
    (12_000, "중앙점을 보면서 얼굴·몸을 왼쪽 아래로 천천히 이동", "EYES CENTER - FACE LEFT-DOWN"),
    (16_000, "중앙점을 보면서 얼굴·몸을 오른쪽 위로 천천히 이동", "EYES CENTER - FACE RIGHT-UP"),
    (20_000, "중앙점을 보면서 카메라에 조금 가까이·멀리 이동", "EYES CENTER - MOVE NEAR / FAR"),
)
_BOUNDARY_GUIDANCE_PHASES: tuple[tuple[int, str, str], ...] = (
    (2_000, "고개를 고정하고 시선을 물체의 왼쪽 위 모서리로 이동", "HEAD STILL - LOOK TOP-LEFT"),
    (5_500, "윗변을 따라 왼쪽 위에서 오른쪽 위까지 천천히 응시", "TRACE TOP EDGE  LEFT -> RIGHT"),
    (9_000, "오른쪽 변을 따라 오른쪽 위에서 오른쪽 아래까지 응시", "TRACE RIGHT EDGE  TOP -> BOTTOM"),
    (12_500, "아랫변을 따라 오른쪽 아래에서 왼쪽 아래까지 응시", "TRACE BOTTOM EDGE  RIGHT -> LEFT"),
    (16_000, "왼쪽 변을 따라 왼쪽 아래에서 왼쪽 위까지 응시", "TRACE LEFT EDGE  BOTTOM -> TOP"),
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
        self._engine_target = QLabel("실시간 엔진 판단: UNKNOWN")
        self._engine_target.setStyleSheet(_MONO)
        layout.addWidget(self._engine_target)
        self._dwell = LabeledBar("3초 연속 응시 진행")
        layout.addWidget(self._dwell)
        self._confirmed_target = QLabel("3초 확정 응시 대상: --")
        self._confirmed_target.setStyleSheet(_MONO)
        layout.addWidget(self._confirmed_target)

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
        self._engine_target.setText(
            f"실시간 엔진 판단: {s.target_label} [{s.target}]  "
            f"P={s.probability:.2f}  confident={s.is_confident}"
        )
        dwell_color = "#3fb950" if s.dwell_progress >= 1.0 else "#58a6ff"
        self._dwell.set_value(s.dwell_progress, color=dwell_color)
        if s.candidate_label is not None:
            dwell_seconds = s.dwell_elapsed_ms / 1000.0
            required_seconds = s.dwell_required_ms / 1000.0
            confirmed = s.locked_target_label or "--"
            confirmed_id = s.locked_device or "--"
            self._confirmed_target.setText(
                f"3초 확정 응시 대상: {confirmed} [{confirmed_id}]  | 교체 후보 {s.candidate_label} "
                f"{dwell_seconds:.1f}/{required_seconds:.1f}s"
            )
            self._confirmed_target.setStyleSheet(_MONO + " color:#58a6ff;")
        elif s.locked_device is not None and s.unknown_elapsed_ms > 0:
            elapsed_seconds = s.unknown_elapsed_ms / 1000.0
            required_seconds = s.unknown_required_ms / 1000.0
            self._confirmed_target.setText(
                f"3초 확정 응시 대상: {s.locked_target_label} [{s.locked_device}]  | "
                f"UNKNOWN 해제 대기 {elapsed_seconds:.1f}/{required_seconds:.1f}s"
            )
            self._confirmed_target.setStyleSheet(_MONO + " color:#d29922;")
        else:
            self._confirmed_target.setText(
                f"3초 확정 응시 대상: {s.locked_target_label or '--'} "
                f"[{s.locked_device or '--'}]"
            )
            color = "#3fb950" if s.locked_device is not None else "#8b949e"
            self._confirmed_target.setStyleSheet(_MONO + f" color:{color};")

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
        velocity = (
            f"yaw={s.gaze_motion_velocity_deg_s[0]:+.1f}  "
            f"pitch={s.gaze_motion_velocity_deg_s[1]:+.1f} deg/s"
            if s.gaze_motion_velocity_deg_s is not None
            else "None (blink/hold/jump/gap)"
        )
        acceleration = (
            f"yaw={s.gaze_motion_acceleration_deg_s2[0]:+.1f}  "
            f"pitch={s.gaze_motion_acceleration_deg_s2[1]:+.1f} deg/s2"
            if s.gaze_motion_acceleration_deg_s2 is not None
            else "None"
        )
        feature_weights = (
            f"gaze x{s.personal_feature_weights[0]:.2f}  "
            f"head x{s.personal_feature_weights[2]:.2f}  "
            f"scale x{s.personal_feature_weights[5]:.2f}  head-cap 20%"
            if s.personal_feature_weights is not None
            else "None"
        )
        ml = (
            f"{s.personal_prediction.target_id}  p={s.personal_prediction.confidence:.3f}  "
            f"p2={s.personal_prediction.second_best_confidence:.3f}  "
            f"{'USED' if s.personal_prediction.confidence >= s.personal_confidence_threshold else 'fallback'} "
            f"(thr={s.personal_confidence_threshold:.2f})"
            if s.personal_prediction is not None
            else f"None (needs 2+ registered targets, thr={s.personal_confidence_threshold:.2f})"
        )
        est = s.target_estimate
        self._numeric.setText(
            f"face_detected : {s.face_detected}\n"
            f"head (deg)    : yaw {s.head_yaw_deg:+7.2f}  pitch {s.head_pitch_deg:+7.2f}  "
            f"roll {s.head_roll_deg:+7.2f}\n"
            f"pose warning  : {s.camera_pose_warning or 'None'}\n"
            f"iris L / R    : {s.left_iris_relative}  /  {s.right_iris_relative}\n"
            f"face_scale    : {s.face_scale if s.face_scale is not None else 'None'}\n"
            f"gaze vector   : {direction}\n"
            f"vector model  : {s.calibration_model_kind or 'geometric'} "
            f"({'APPLIED' if s.calibration_applied else 'raw'})\n"
            f"feature       : {feature}\n"
            f"gaze delta    : {motion}\n"
            f"gaze velocity : {velocity}\n"
            f"gaze accel    : {acceleration}\n"
            f"ML priority   : {feature_weights}\n"
            f"ml score      : {ml}\n"
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

    def update_snapshot(self, s: HandSnapshot) -> None:
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

        if s.hand_detected:
            mode = "스무딩됨 (모델 실제 입력)" if s.smoothed else "raw (스무딩 꺼짐)"
            self._numeric.setText(
                f"모델 입력   : {mode}\n"
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
        self._show_target_heatmap = False
        self._registration_guide: tuple[str, str, float] | None = None
        self._show_placeholder("카메라 시작 중…")

    def set_gaze(self, snapshot: GazeSnapshot) -> None:
        self._gaze = snapshot

    def set_target_heatmap_visible(self, visible: bool) -> None:
        self._show_target_heatmap = visible

    def set_hand(self, snapshot: HandSnapshot) -> None:
        self._hand = snapshot

    def set_registration_guidance(
        self, title: str, instruction: str, progress: float
    ) -> None:
        self._registration_guide = (title, instruction, progress)

    def clear_registration_guidance(self) -> None:
        self._registration_guide = None

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
            draw_hand_overlay(display, self._hand, mirror=True)
        if self._registration_guide is not None:
            title, instruction, progress = self._registration_guide
            draw_registration_guidance(
                display,
                title=title,
                instruction=instruction,
                progress=progress,
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
    """Recognized-gesture list bound to a GestureSource, plus a live hand-track line.

    The recognized-gesture list stays empty while the classifier is untrained
    (the source yields nothing) — no fabricated detections. The separate hand line
    reflects real MediaPipe hand tracking so the sidebar still shows live signal.
    """

    def __init__(self, source: GestureSource) -> None:
        super().__init__()
        self._source = source
        layout = QVBoxLayout(self)
        title = QLabel("인식된 제스처")
        title.setStyleSheet("font-weight:600; color:#58a6ff; padding:4px 0;")
        self._status = QLabel(source.status_text)
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#8b949e;" if source.available else "color:#d29922;")
        self._hand_line = QLabel("손 추적: —")
        self._hand_line.setWordWrap(True)
        self._hand_line.setStyleSheet(_MONO)
        self._list = QListWidget()
        layout.addWidget(title)
        layout.addWidget(self._status)
        layout.addWidget(self._hand_line)
        layout.addWidget(self._list, 1)
        self.setMinimumWidth(220)

    def set_hand_status(self, snapshot: HandSnapshot) -> None:
        if snapshot.hand_detected:
            label = snapshot.handedness or "?"
            self._hand_line.setText(f"손 추적: {label} 검출 (det {snapshot.detection_confidence:.0%})")
            self._hand_line.setStyleSheet(_MONO + " color:#3fb950;")
        else:
            self._hand_line.setText("손 추적: 손 없음")
            self._hand_line.setStyleSheet(_MONO + " color:#8b949e;")

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
        hand_model_path: Path | None = None,
        samples_path: Path | None = None,
        calibration_model_path: Path | None = None,
        personal_classifier_path: Path | None = None,
        start_camera: bool = True,
        gaze_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("JARVIS Pipeline Monitor")
        self.resize(1180, 820)
        self._log = MessageLog()
        self._gesture_source: GestureSource = UntrainedGestureSource()
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
        self._calibration_store = GazeCalibrationStore(
            calibration_model_path or _default_calibration_model_path()
        )
        self._personal_target_store = PersonalTargetStore(
            personal_classifier_path or _default_personal_classifier_path(),
            feature_weights=self._gaze_config.personal_feature_weights,
        )
        self._target_registry = TargetRegistry(self._profiles_path)
        # Prefer the validated residual MLP; Ridge remains a fallback when the
        # personal dataset is still too small for an MLP.
        self._active_calibration_model: GazeCalibrationCorrector | None = (
            self._calibration_store.preferred_model
        )
        self._gaze_regression_user_disabled = False
        self._registration: TargetRegistrationSession | None = None
        self._registration_phase_marker: tuple[RegistrationPhase, int] | None = None
        self._last_camera_pose_warning: str | None = None
        self._target_list = QListWidget()
        self._register_target_button = QPushButton()
        self._probe = GazeProbe(
            model_path=self._model_path,
            profiles_path=self._profiles_path,
            config=self._gaze_config,
            calibration_model=self._active_calibration_model,
        )
        self._probe.set_personal_classifier(
            self._personal_target_store.model,
            confidence_threshold=self._personal_target_store.confidence_threshold,
        )
        self._hand_probe = HandProbe(model_path=self._hand_model_path)

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "실시간")
        tabs.addTab(self._build_gaze_tab(), "Gaze 파이프라인")
        tabs.addTab(self._build_hand_tab(), "손 추적")
        tabs.addTab(self._build_pipeline_tab(), "파이프라인")
        tabs.addTab(self._build_latency_tab(), "지연·어댑터")
        self.setCentralWidget(tabs)

        self._log.info("모니터 시작")
        if not self._gesture_source.available:
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
        self._target_heatmap_toggle = QCheckBox("Target heatmap / 물체 영역 표시")
        self._target_heatmap_toggle.toggled.connect(self._video.set_target_heatmap_visible)
        self._gaze_regression_toggle = QCheckBox("Use MLP gaze-vector calibration")
        self._gaze_regression_toggle.setToolTip(
            "등록 데이터로 학습된 residual MLP를 final yaw/pitch에 적용합니다. "
            "MLP 데이터가 부족하면 Ridge를 사용합니다."
        )
        self._gaze_regression_toggle.setChecked(self._active_calibration_model is not None)
        self._gaze_regression_toggle.toggled.connect(self._set_gaze_regression_enabled)
        sample_controls = QHBoxLayout()
        sample_controls.addWidget(self._sample_button, 1)
        sample_controls.addWidget(self._clear_samples_button)
        sample_controls.addWidget(self._target_heatmap_toggle)
        sample_controls.addWidget(self._gaze_regression_toggle)
        layout.addLayout(sample_controls)
        self._sample_list = QListWidget()
        self._sample_list.setMaximumHeight(130)
        self._sample_list.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")
        for sample in self._sample_store.samples:
            self._sample_list.addItem(format_gaze_sample(sample))
        layout.addWidget(self._sample_list)
        self._refresh_sample_button()
        self._registration_step = QLabel("2단계 물체 등록: 대기")
        self._registration_step.setStyleSheet("font-weight:700; color:#8b949e;")
        layout.addWidget(self._registration_step)
        self._registration_progress = QProgressBar()
        self._registration_progress.setRange(0, 1000)
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("등록 대기")
        layout.addWidget(self._registration_progress)
        self._registration_status = QLabel(
            "1단계에서는 중앙점만 보며 자세를 바꾸고, "
            "2단계에서는 고개를 고정한 채 눈으로 물체 테두리를 따라갑니다."
        )
        self._registration_status.setWordWrap(True)
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
        self._cancel_registration_button = QPushButton("등록 취소")
        self._cancel_registration_button.setEnabled(False)
        self._cancel_registration_button.clicked.connect(self._cancel_target_registration)
        for button in (
            self._register_target_button,
            self._reregister_target_button,
            self._rename_target_button,
            self._delete_target_button,
            self._cancel_registration_button,
        ):
            target_controls.addWidget(button)
        layout.addLayout(target_controls)
        self._target_list = QListWidget()
        self._target_list.setMaximumHeight(100)
        layout.addWidget(self._target_list)
        self._refresh_targets()
        return container

    def _set_gaze_regression_enabled(self, enabled: bool) -> None:
        self._gaze_regression_user_disabled = not enabled
        preferred = getattr(self._calibration_store, "preferred_model", None)
        if preferred is None:
            fallback = getattr(self._calibration_store, "model", None)
            preferred = fallback if fallback is not None and fallback.fitted else None
        if enabled and preferred is not None:
            self._active_calibration_model = preferred
            self._probe.set_calibration_model(self._active_calibration_model)
            self._log.info(
                f"gaze {preferred.kind} 벡터 보정 ON "
                f"(samples={preferred.sample_count}, "
                f"targets={preferred.target_count})"
            )
            return
        self._active_calibration_model = None
        self._probe.set_calibration_model(None)
        if enabled:
            # The user asked to enable correction, but no model exists yet.
            # Auto-enable it after enough target registrations have trained one.
            self._gaze_regression_user_disabled = False
            self._gaze_regression_toggle.blockSignals(True)
            self._gaze_regression_toggle.setChecked(False)
            self._gaze_regression_toggle.blockSignals(False)
            self._log.warn(
                "gaze 벡터 보정 사용 불가: 최소 2개 target의 calibration sample이 필요합니다"
            )
        else:
            self._log.info("gaze 벡터 보정 OFF")

    def _build_gaze_tab(self) -> QWidget:
        self._gaze_panel = GazePanel(self._probe.status_text)
        return self._gaze_panel

    def _build_hand_tab(self) -> QWidget:
        self._hand_panel = HandPanel(
            self._hand_probe.status_text,
            self._hand_probe.gesture_recognition_status,
            smoothing=self._hand_probe.smoothing,
            on_smoothing_toggled=self._hand_probe.set_smoothing,
        )
        return self._hand_panel

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

    def _on_gaze(self, snapshot: object) -> None:
        assert isinstance(snapshot, GazeSnapshot)
        self._latest_gaze = snapshot
        self._gaze_history.append(snapshot)
        cutoff_ms = snapshot.timestamp_ms - 500
        while self._gaze_history and self._gaze_history[0].timestamp_ms < cutoff_ms:
            self._gaze_history.popleft()
        self._video.set_gaze(snapshot)
        self._gaze_panel.update_snapshot(snapshot)
        if (
            snapshot.camera_pose_warning
            and snapshot.camera_pose_warning != self._last_camera_pose_warning
        ):
            self._log.warn(snapshot.camera_pose_warning)
            self._last_camera_pose_warning = snapshot.camera_pose_warning
        elif snapshot.camera_pose_warning is None:
            self._last_camera_pose_warning = None
        self._latency.record(LatencyStage.CAPTURE_TO_INFERENCE, snapshot.inference_ms)
        if self._registration is not None:
            smoothed = self._smoothed_from_snapshot(snapshot)
            self._registration.add(
                smoothed,
                snapshot.tracking_confidence,
                eyes_open=snapshot.eyes_open,
                face_scale=snapshot.face_scale,
                feature_sample=snapshot.feature_sample,
                calibration_features=snapshot.calibration_features,
            )
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
        self._registration_phase_marker = None
        self._set_registration_controls(active=True)
        self._registration_step.setText("1/2 중앙점 MLP 보정 · 20초")
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("1/2 중앙점 보정  %p%")
        self._registration_status.setText(
            f"'{name}' 중앙의 한 점을 계속 바라보세요. 눈은 그 점에 고정하고, "
            "안내에 따라 얼굴·몸의 위치와 거리를 바꿉니다. 물체 테두리는 아직 보지 마세요."
        )
        self._registration_status.setStyleSheet(
            "background:#3d2a12; color:#f0b429; border:1px solid #7a5a1e;"
            " border-radius:6px; padding:8px; font-weight:700;"
        )
        self._log.info(
            f"'{name}' 2단계 등록 시작 — 1/2 중앙점 MLP 보정: "
            "물체 중앙의 한 점만 보면서 얼굴/몸을 좌상·우하·좌하·우상·근거리/원거리로 이동"
        )
        self._video.set_registration_guidance(
            "REGISTRATION 1/2 - CENTER CALIBRATION",
            "KEEP EYES ON ONE CENTER POINT",
            0.0,
        )

    def _update_registration_guidance(self, timestamp_ms: int) -> None:
        assert self._registration is not None
        if self._registration.started_at_ms is None:
            self._registration_status.setText("등록 대기: 얼굴과 시선이 안정적으로 잡히길 기다리는 중")
            return
        phase = self._registration.phase
        if phase == RegistrationPhase.COMPLETE:
            self._registration_step.setText("2/2 물체 영역 확정 완료")
            self._registration_progress.setValue(1000)
            self._registration_progress.setFormat("등록 데이터 처리 중  %p%")
            self._video.set_registration_guidance(
                "REGISTRATION COMPLETE", "BUILDING TARGET PROFILE", 1.0
            )
            return
        elapsed_ms = self._registration.phase_elapsed_ms(timestamp_ms)
        phases = (
            _CENTER_GUIDANCE_PHASES
            if phase == RegistrationPhase.CENTER
            else _BOUNDARY_GUIDANCE_PHASES
        )
        phase_index = 0
        for index, (end_ms, _label, _video_label) in enumerate(phases):
            if elapsed_ms < end_ms:
                phase_index = index
                break
        else:
            phase_index = len(phases) - 1
        phase_end_ms, label, video_label = phases[phase_index]
        remaining_s = max(0.0, (phase_end_ms - elapsed_ms) / 1000.0)
        progress = self._registration.phase_progress(timestamp_ms)
        if phase == RegistrationPhase.CENTER:
            step = "1/2 중앙점 MLP 보정"
            count = self._registration.center_valid_frame_count
            required = self._registration.minimum_valid_frames
            title = "REGISTRATION 1/2 - CENTER CALIBRATION"
        else:
            step = "2/2 물체 영역 확정"
            count = self._registration.boundary_valid_frame_count
            required = self._registration.minimum_boundary_frames
            title = "REGISTRATION 2/2 - TRACE BOUNDARY"
        if elapsed_ms >= self._registration.phase_duration_ms() and count < required:
            label = "시간은 완료됐지만 유효 프레임이 부족합니다. 안내 자세를 유지해 주세요."
            video_label = "HOLD POSITION - NEED MORE VALID FRAMES"
        status = (
            f"{label}\n현재 구간 남은 시간 {remaining_s:0.1f}s · 유효 프레임 {count}/{required} · "
            f"중앙 {self._registration.center_valid_frame_count} / 경계 "
            f"{self._registration.boundary_valid_frame_count}"
        )
        self._registration_step.setText(step)
        self._registration_progress.setValue(round(progress * 1000))
        self._registration_progress.setFormat(f"{step}  %p%")
        self._registration_status.setText(status)
        self._video.set_registration_guidance(title, video_label, progress)
        marker = (phase, phase_index)
        if marker != self._registration_phase_marker:
            previous_phase = (
                self._registration_phase_marker[0]
                if self._registration_phase_marker is not None
                else None
            )
            if phase == RegistrationPhase.BOUNDARY and previous_phase != phase:
                self._activate_registration_preview_calibration()
                self._log.info(
                    "1/2 중앙점 보정 완료 — 2/2 영역 확정 시작: "
                    "이제 고개·몸을 고정하고 눈으로만 네 모서리와 테두리를 따라가세요"
                )
            self._registration_phase_marker = marker
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
                    target_id=record.target_id,
                )
                for features in self._registration.calibration_features
            ]
            if calibration_samples:
                model = self._calibration_store.add_samples(
                    calibration_samples,
                    replace_target_id=record.target_id,
                )
            else:
                model = self._calibration_store.model
                self._log.warn(
                    f"'{record.name}' 중앙점 calibration feature가 없어 MLP 재학습을 건너뜁니다"
                )
            preferred_model = self._calibration_store.preferred_model
            auto_enable = (
                preferred_model is not None and not self._gaze_regression_user_disabled
            )
            if auto_enable:
                self._gaze_regression_toggle.blockSignals(True)
                self._gaze_regression_toggle.setChecked(True)
                self._gaze_regression_toggle.blockSignals(False)
            self._active_calibration_model = (
                preferred_model
                if auto_enable
                else None
            )
            self._probe.set_calibration_model(self._active_calibration_model)
            personal_model = self._personal_target_store.add_samples(
                record.target_id,
                list(self._registration.feature_samples),
                replace_target=True,
            )
            self._probe.set_personal_classifier(
                personal_model,
                confidence_threshold=self._personal_target_store.confidence_threshold,
            )
            personal_label = (
                f"personal classifier samples={personal_model.sample_count}"
                if personal_model is not None
                else "personal classifier waiting for 2+ targets"
            )
            mlp_model = self._calibration_store.mlp_model
            mlp_label = (
                f"MLP samples={mlp_model.sample_count}, "
                f"validation={mlp_model.validation_raw_error_deg:.2f}→"
                f"{mlp_model.validation_mlp_error_deg:.2f}deg"
                if mlp_model.fitted
                and mlp_model.validation_raw_error_deg is not None
                and mlp_model.validation_mlp_error_deg is not None
                else "MLP waiting for 3+ target directions / validation improvement"
            )
            self._log.info(
                f"'{record.name}' 2단계 등록 완료 "
                f"(중앙 {self._registration.center_valid_frame_count}, "
                f"경계 {self._registration.boundary_valid_frame_count} frames) "
                f"— {self._describe_triangulation_outcome(record)} | "
                f"{self._registration.diagnostic_summary()} | "
                f"gaze calibration samples={model.sample_count} | {mlp_label} | {personal_label}"
            )
        except ValueError as exc:
            self._log.warn(f"기기 등록 실패: {exc} | {self._registration.diagnostic_summary()}")
        finally:
            self._clear_registration_state()
            self._refresh_targets()

    def _activate_registration_preview_calibration(self) -> None:
        """Apply phase-1 calibration to phase 2 without persisting partial data."""
        assert self._registration is not None
        center = self._registration.center_yaw_pitch
        if center is None:
            return
        center_yaw, center_pitch = center
        samples = [
            GazeCalibrationSample(
                features=features,
                target_yaw=center_yaw,
                target_pitch=center_pitch,
                target_id=self._registration.target_id,
            )
            for features in self._registration.calibration_features
        ]
        preview = self._calibration_store.preview_model(
            samples,
            replace_target_id=self._registration.target_id,
        )
        if preview is None:
            self._log.warn(
                "1/2 중앙점 데이터는 수집됐지만 아직 보정 모델에 필요한 target 수가 부족합니다"
            )
            return
        if self._gaze_regression_user_disabled:
            self._log.info(
                f"1/2 중앙점 임시 {preview.kind} 학습 완료 — 사용자가 보정을 꺼서 2단계에는 미적용"
            )
            return
        self._active_calibration_model = preview
        self._probe.set_calibration_model(preview)
        self._gaze_regression_toggle.blockSignals(True)
        self._gaze_regression_toggle.setChecked(True)
        self._gaze_regression_toggle.blockSignals(False)
        self._log.info(
            f"1/2 중앙점 임시 {preview.kind} 학습 완료·2단계 적용 "
            f"(samples={preview.sample_count}, targets={preview.target_count}, 저장은 2단계 완료 후)"
        )

    def _cancel_target_registration(self) -> None:
        if self._registration is None:
            return
        name = self._registration.name
        self._clear_registration_state()
        self._log.warn(f"'{name}' 물체 등록을 사용자가 취소했습니다")

    def _set_registration_controls(self, *, active: bool) -> None:
        for button in (
            self._register_target_button,
            self._reregister_target_button,
            self._rename_target_button,
            self._delete_target_button,
        ):
            button.setEnabled(not active)
        self._cancel_registration_button.setEnabled(active)

    def _clear_registration_state(self) -> None:
        self._registration = None
        self._registration_phase_marker = None
        persistent_model = self._calibration_store.preferred_model
        self._active_calibration_model = (
            persistent_model
            if persistent_model is not None and not self._gaze_regression_user_disabled
            else None
        )
        self._probe.set_calibration_model(self._active_calibration_model)
        self._set_registration_controls(active=False)
        self._registration_step.setText("2단계 물체 등록: 대기")
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("등록 대기")
        self._registration_status.setText(
            "1단계에서는 중앙점만 보며 자세를 바꾸고, "
            "2단계에서는 고개를 고정한 채 눈으로 물체 테두리를 따라갑니다."
        )
        self._registration_status.setStyleSheet(
            "background:#161b22; color:#8b949e; border:1px solid #30363d;"
            " border-radius:6px; padding:8px; font-weight:600;"
        )
        self._video.clear_registration_guidance()

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
        self._personal_target_store.remove_target(record.target_id)
        self._probe.set_personal_classifier(
            self._personal_target_store.model,
            confidence_threshold=self._personal_target_store.confidence_threshold,
        )
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
        self._hand_panel.update_snapshot(snapshot)
        self._sidebar.set_hand_status(snapshot)

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


def _default_hand_model_path() -> Path | None:
    return Path("models/hand_landmarker.task")


def _default_profiles_path() -> Path:
    return Path("data/calibration/profiles.json")


def _default_calibration_model_path() -> Path:
    return Path("data/calibration/gaze_regressor.json")


def _default_personal_classifier_path() -> Path:
    return Path("data/calibration/personal_target_classifier.json")


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(read_env_file(Path(".env")))
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    return int(app.exec())
