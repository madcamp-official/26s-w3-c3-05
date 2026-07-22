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
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeySequence, QPixmap, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
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
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jarvis.calibration.registry import TargetRecord, TargetRegistry
from jarvis.calibration.target_registration import RegistrationPhase, TargetRegistrationSession
from jarvis.contracts.messages import Command, GestureEstimate, Intent
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.lock import GazeLockState
from jarvis.gaze.registration_lint import lint_target_record
from jarvis.gaze.session_report import build_report, format_report, load_session
from jarvis.gaze.smoothing import SmoothedGaze
from jarvis.gesture_fusion.fusion import CommitDecision
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.model_protocol import (
    DEFAULT_BACKGROUND_LABELS,
    DEFAULT_GESTURE_LABELS,
    EXPECTED_INPUT_FPS,
)
from jarvis.gesture_fusion.pose_protocol import DEFAULT_POSE_TILT_LIMITS
from jarvis.gesture_fusion.pose_state import MIN_FINGER_EXTENSION
from jarvis.monitoring.camera_worker import CameraWorker
from jarvis.monitoring.console import ConsoleLog, StderrCapture
from jarvis.monitoring.gaze_probe import GazeProbe, GazeSnapshot
from jarvis.monitoring.gaze_samples import GazeSampleStore, format_gaze_sample
from jarvis.monitoring.gesture_probe import (
    GestureProbe,
    GestureSnapshot,
    ProbeGestureSource,
    available_gesture_checkpoints,
    load_gesture_model_from,
    load_trained_gesture_model,
)
from jarvis.monitoring.session_recorder import GazeSessionRecorder, NO_TARGET_LABEL
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
from jarvis.monitoring.demo_bridge import (
    BULB_DEVICE_ID,
    DEVICE_TYPE_TO_RUNTIME_ID,
    LAPTOP_DEVICE_ID,
    UNKNOWN_TARGET,
    DemoBridge,
    DemoPreset,
    DeviceMappingStore,
    describe_decision,
    describe_outcome,
)
from jarvis.monitoring.demo_panel import DemoPanel, TargetChoice
from jarvis.monitoring.execute_worker import ExecuteWorker
from jarvis.monitoring.pipeline_status import StageState, StageStatus, detect_pipeline_status
from jarvis.monitoring.pose_control import PoseControlBridge, default_input_sink
from jarvis.monitoring.virtual_bulb import VirtualBulbState
from jarvis.runtime.devices import (
    build_default_adapters,
    build_default_capability_map,
    build_default_registry,
)
from jarvis.runtime.executor import ExecutionOutcome, ExecutionStage, IntentExecutor
from jarvis.runtime_protocol.adapters.wiz import WizConfig
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.config import read_env_file
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine
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

_TARGET_DEVICE_TYPES: tuple[str, ...] = ("computer", "electric bulb")
"""시선 등록 UI에서 선택 가능한 데모 기종. 저장값과 표시값을 동일하게 쓴다."""

# 1단계는 "중앙 한 점 응시 + 고개 스윕"이다. 테두리를 훑으며 고개를 돌리면
# 사람은 보는 방향으로 고개를 돌리므로 head-yaw bin이 센서 편향이 아니라
# "그 자세에서 보던 테두리 위치"를 학습해 pose 보정의 부호가 뒤집힌다
# (2026-07-22 실측, documents/gaze.md).
_CENTER_GUIDANCE_PHASES: tuple[tuple[int, str, str], ...] = (
    (5_000, "물체 중앙 한 점을 응시한 채 고개를 왼쪽으로 천천히 끝까지", "FIX CENTER - TURN HEAD LEFT"),
    (10_000, "중앙 응시 유지, 고개를 오른쪽으로 천천히 끝까지", "FIX CENTER - TURN HEAD RIGHT"),
    (14_000, "중앙 응시 유지, 고개를 정면으로 되돌리고 위·아래로 천천히", "FIX CENTER - HEAD UP / DOWN"),
    (17_000, "중앙 응시 유지, 카메라에 조금 가까이·멀리 이동", "FIX CENTER - MOVE NEAR / FAR"),
    (20_000, "중앙 응시 유지, 편한 자세로 되돌아오기", "FIX CENTER - RETURN NEUTRAL"),
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
        if s.tracking_lost:
            source_status = "  ·  얼굴 추적 손실"
            status_color = "#f85149"
        elif s.gaze_source == "head-only":
            reason = f": {s.gaze_source_reason}" if s.gaze_source_reason else ""
            source_status = f"  ·  HEAD ONLY{reason}"
            status_color = "#d29922"
        elif s.gaze_source in {"held", "tracking-hold"}:
            source_status = "  ·  이전 gaze 유지"
            status_color = "#d29922"
        else:
            source_status = ""
            status_color = "#3fb950"
        self._status.setText(
            f"frame #{s.frame_id} · {s.inference_ms:.0f} ms/frame"
            + source_status
        )
        self._status.setStyleSheet(f"color:{status_color};")

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
            else "사용 불가 (None)"
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
                    f"face=({s.feature_sample.face_center_x:.3f},"
                    f"{s.feature_sample.face_center_y:.3f})",
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
        settle = (
            f"yaw={s.gaze_settle_velocity_deg_s[0]:+.1f}  "
            f"pitch={s.gaze_settle_velocity_deg_s[1]:+.1f} deg/s  "
            f"age={s.gaze_settle_age_ms}ms"
            if s.gaze_settle_velocity_deg_s is not None
            else "None (no completed eye movement)"
        )
        est = s.target_estimate
        eye_opening = (
            f"L {s.left_eye_open_ratio:.3f}/{s.left_eye_open_baseline:.3f}  "
            f"R {s.right_eye_open_ratio:.3f}/{s.right_eye_open_baseline:.3f}  "
            f"{'OPEN' if s.eyes_open else 'CLOSED'}"
            if (
                s.left_eye_open_ratio is not None
                and s.right_eye_open_ratio is not None
                and s.left_eye_open_baseline is not None
                and s.right_eye_open_baseline is not None
            )
            else "unavailable"
        )
        self._numeric.setText(
            f"face_detected : {s.face_detected}\n"
            f"head (deg)    : yaw {s.head_yaw_deg:+7.2f}  pitch {s.head_pitch_deg:+7.2f}  "
            f"roll {s.head_roll_deg:+7.2f}\n"
            f"pose warning  : {s.camera_pose_warning or 'None'}\n"
            f"iris L / R    : {s.left_iris_relative}  /  {s.right_iris_relative}\n"
            f"eye ratio/base: {eye_opening}\n"
            f"face_scale    : {s.face_scale if s.face_scale is not None else 'None'}\n"
            f"gaze vector   : {direction}\n"
            f"gaze source   : {s.gaze_source}\n"
            f"source reason : {s.gaze_source_reason or 'None'}\n"
            "vector model  : geometric head + iris, head-only fallback\n"
            f"feature       : {feature}\n"
            f"gaze delta    : {motion}\n"
            f"gaze velocity : {velocity}\n"
            f"gaze accel    : {acceleration}\n"
            f"settle intent : {settle}\n"
            "profile model : deterministic area + Mahalanobis\n"
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
            if s.finger_extension is None:
                ext_line = "—"
            else:
                gate = "스크롤 가능" if s.finger_extension >= MIN_FINGER_EXTENSION else "게이트 차단"
                ext_line = f"{s.finger_extension:.3f}  / {MIN_FINGER_EXTENSION:g} ({gate})"
            self._numeric.setText(
                f"모델 입력   : {mode}\n"
                f"자세 판정   : {pose_line}\n"
                f"확정 상태   : {s.pose_state or '—'}"
                + (f"   → {', '.join(e.kind for e in s.pose_events)}" if s.pose_events else "")
                + "\n"
                f"손 기울기   : {tilt}\n"
                f"손가락 폄   : {ext_line}\n"
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
    """Displays webcam frames scaled to the widget, with a HUD and gaze overlay.

    `show_overlay=False`는 관객이 보는 화면(시연 탭)용이다 — HUD·gaze 벡터·등록
    가이드를 끈다. 다만 시연에서 손 검출과 제스처 입력을 확인해야 하므로
    `show_hand_overlay=True`를 별도로 줄 수 있다. 실시간·손 추적 탭은 기본값으로
    모든 디버그 정보를 계속 보여준다.
    """

    def __init__(
        self, *, show_overlay: bool = True, show_hand_overlay: bool | None = None
    ) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#0b0e13;")
        self._show_overlay = show_overlay
        self._show_hand_overlay = (
            show_overlay if show_hand_overlay is None else show_hand_overlay
        )
        self._fps_times: deque[float] = deque(maxlen=30)
        self._frame_count = 0
        self._gaze: GazeSnapshot | None = None
        self._hand: HandSnapshot | None = None
        self._control_action = ""
        self._control_enabled = False
        self._show_target_heatmap = False
        self._registration_guide: tuple[str, str, float] | None = None
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
        if self._show_overlay:
            draw_hud(display, [f"{w}x{h}  {fps:4.1f} FPS", f"frame #{self._frame_count}"])
            if self._gaze is not None:
                if self._show_target_heatmap:
                    draw_target_heatmap(display, self._gaze, mirror=True)
                draw_gaze_overlay(display, self._gaze, mirror=True)
            if self._registration_guide is not None:
                title, instruction, progress = self._registration_guide
                draw_registration_guidance(
                    display,
                    title=title,
                    instruction=instruction,
                    progress=progress,
                )
        if self._show_hand_overlay and self._hand is not None:
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

    def set_source(self, source: GestureSource) -> None:
        """활성 인식 소스를 바꾼다(가중치 전환 시) — 상태 헤더를 새 소스로 갱신한다."""
        self._source = source
        self._status.setText(source.status_text)
        self._status.setStyleSheet("color:#8b949e;" if source.available else "color:#d29922;")

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


class ConsolePanel(QListWidget):
    """앱 하단 콘솔 독 — 가로챈 프로세스 stderr(네이티브 로그)를 표시한다.

    터미널로 새던 C++ stderr(MediaPipe clearcut 등)를 여기로 모은다. 시스템 메시지
    패널(:class:`MessagePanel`)과 분리해, 노이즈가 실제 INFO/WARN을 덮지 않게 한다.
    """

    def __init__(self, console: ConsoleLog) -> None:
        super().__init__()
        self._console = console
        self.setStyleSheet(
            "QListWidget{background:#0a0d12; border:none;"
            " font-family:Consolas,monospace; font-size:11px; color:#8b949e;}"
        )

    def refresh(self) -> None:
        self.clear()
        for line in self._console.recent(200):
            suffix = f"  (x{line.count})" if line.count > 1 else ""
            self.addItem(f"{line.text}{suffix}")
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


def _build_demo_executor(env: dict[str, str]) -> IntentExecutor | None:
    """시연용 `IntentExecutor` — Fusion 커밋을 실제 기기 명령까지 잇는 실행기.

    `build_laptop_only_executor()`는 전구 설정을 받지 않으므로 공개 빌더로 직접
    조립한다(`jarvis.runtime.devices`는 다른 작업 중이라 건드리지 않는다). 전구
    설정(`WIZ_DEVICE_TARGETS`)이 없으면 WiZ adapter가 UNCONFIGURED를 돌려준다 —
    배선은 완성되고 실행만 안전하게 실패한다.

    이 OS에 입력 어댑터가 없으면(`default_input_sink()`가 raise) None을 돌려
    시연 실행 자체를 끈다 — 제어를 흉내내지 않는다.
    """
    try:
        adapters = build_default_adapters(wiz_config=WizConfig.from_env(env))
    except Exception:  # noqa: BLE001 - 미지원 OS·어댑터 부재는 정직하게 off
        return None
    registry = build_default_registry()
    return IntentExecutor(
        engine=ProtocolEngine(registry, RuntimeClock()),
        registry=registry,
        adapters=adapters,
        capability_map=build_default_capability_map(),
    )


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
            "training/cache/webcam/<person_id>/. 이 탭이 보이는 동안 스페이스바로도 "
            "녹화를 시작/종료할 수 있다(버튼과 동일)."
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
        self._on_record_toggled = on_record_toggled
        self._record_button.clicked.connect(lambda: on_record_toggled())
        controls.addWidget(self._record_button)
        layout.addLayout(controls)

        # 스페이스바는 이 탭에서 무조건 녹화 토글이어야 한다. QComboBox·QPushButton은
        # 포커스를 쥐면 스페이스를 자체 소비해(콤보는 팝업을 열고, 버튼은 클릭) 이벤트
        # 필터 없이는 그 동작이 녹화 토글과 함께 겹쳐 발동한다(2026-07-21 발견). 이
        # 두 위젯에 이벤트 필터를 걸어 스페이스를 위젯 자신에게 전달되기 전에 가로채
        # 녹화 토글만 실행하고 소비한다 — 팝업이 열리거나 버튼이 별도로 클릭되는
        # 일 자체가 없어진다.
        self._gesture_combo.installEventFilter(self)
        self._record_button.installEventFilter(self)

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

    def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802 - Qt override name
        """동작 콤보·REC 버튼이 스페이스를 받기 전에 가로채 녹화 토글로만 처리한다.

        `installEventFilter`로 이 두 위젯에 걸려 있다 — 필터가 True를 반환하면 Qt가
        그 이벤트를 대상 위젯에 아예 전달하지 않으므로, 콤보 팝업이 열리거나 버튼이
        자체 클릭되는 일이 애초에 일어나지 않는다(단순히 신호 후 무시하는 게 아니라
        위젯의 기본 동작 자체를 막는다).
        """
        if (
            event.type() == QEvent.Type.KeyPress  # type: ignore[attr-defined]
            and event.key() == Qt.Key.Key_Space  # type: ignore[attr-defined]
            and watched in (self._gesture_combo, self._record_button)
        ):
            self._on_record_toggled()
            return True
        return super().eventFilter(watched, event)  # type: ignore[arg-type]

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
        diagnostics_dir: Path | None = None,
        start_camera: bool = True,
        gaze_enabled: bool = True,
        start_tab: str | None = None,
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
        self._session_recorder = GazeSessionRecorder()
        self._session_label: str | None = None
        self._diagnostics_dir = diagnostics_dir or Path("data/diagnostics")
        self._gaze_config = GazeConfig(
            enable_3d_target_matching=False,
            require_3d_target_registration=False,
        )
        self._target_registry = TargetRegistry(self._profiles_path)
        self._registration: TargetRegistrationSession | None = None
        self._registration_phase_marker: tuple[RegistrationPhase, int] | None = None
        self._last_camera_pose_warning: str | None = None
        self._target_list = QListWidget()
        self._register_target_button = QPushButton()
        self._probe = GazeProbe(
            model_path=self._model_path,
            profiles_path=self._profiles_path,
            config=self._gaze_config,
        )
        self._hand_probe = HandProbe(
            model_path=self._hand_model_path, pose_model_path=_default_pose_model_path()
        )
        # 판정과 실행의 분리: 상태기계는 순수 로직이고, 실제 OS 입력은 이 브리지만 한다.
        self._pose_control = PoseControlBridge(sink=default_input_sink(), enabled=True)

        # 시연 배선: Gaze·Gesture 실시간 스트림 → Fusion 판정 → (워커 스레드) 기기 명령.
        # 물체 등록이 발급하는 target_001…을 런타임 기기 id로 잇는 매핑은 등록 프로파일
        # 옆에 따로 둔다 — 등록 흐름 자체는 건드리지 않는다.
        self._device_mapping = DeviceMappingStore(
            self._profiles_path.with_name("demo_device_map.json")
        )
        self._ensure_demo_mappings()
        self._demo_bridge = DemoBridge(mapping_store=self._device_mapping)
        self._last_demo_gesture = "-"
        # 억제 전이를 기억한다 — release()를 매 프레임이 아니라 진입할 때 한 번만 부르려고.
        self._pose_suppressed = False
        self._virtual_bulb = VirtualBulbState()
        self._last_bulb_badge = ("미설정", False)
        executor = _build_demo_executor(self._env)
        self._execute_worker: ExecuteWorker | None = None
        if executor is not None:
            self._execute_worker = ExecuteWorker(executor)
            self._execute_worker.outcome_ready.connect(self._on_execution_outcome)
            self._execute_worker.failed.connect(self._on_execution_failed)
            self._execute_worker.dropped.connect(self._on_execution_failed)
            self._execute_worker.start()

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "실시간")
        tabs.addTab(self._build_gaze_recognition_tab(), "시선 인식")
        tabs.addTab(self._build_gaze_tab(), "Gaze 파이프라인")
        tabs.addTab(self._build_hand_tab(), "손 추적")
        tabs.addTab(self._build_pipeline_tab(), "파이프라인")
        tabs.addTab(self._build_latency_tab(), "지연·어댑터")
        tabs.addTab(self._build_finetune_tab(), "파인튜닝")
        tabs.addTab(self._build_demo_tab(), "시연")
        self.setCentralWidget(tabs)
        self._tabs = tabs
        if start_tab is not None:
            for index in range(tabs.count()):
                if tabs.tabText(index) == start_tab:
                    tabs.setCurrentIndex(index)
                    break
        # keyPressEvent에서 "지금 파인튜닝 탭이 보이는가"를 판정하는 데 쓴다(스페이스바
        # 녹화 단축키가 다른 탭에서 실수로 발동하지 않도록).
        self._tabs = tabs

        # 하단 콘솔 독: 터미널로 새던 네이티브 stderr(MediaPipe clearcut 등)를 앱 안에
        # 모은다. 모든 탭 아래에 걸쳐 항상 보인다. 실제 캡처(fd 2 리다이렉트)는 run()이
        # start_console_capture()로 시작한다 — __init__에서 하면 테스트의 stderr까지 삼킨다.
        self._console_log = ConsoleLog()
        self._console_panel = ConsolePanel(self._console_log)
        self._stderr_capture: StderrCapture | None = None
        console_dock = QDockWidget("콘솔 (stderr)", self)
        console_dock.setObjectName("console_dock")
        console_dock.setWidget(self._console_panel)
        console_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        console_dock.setMaximumHeight(160)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, console_dock)

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

    def _build_recognition_panel(self) -> QWidget:
        """실시간 탭 오른쪽: 사용할 가중치 선택 콤보 + 인식 스트림 사이드바."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        selector_row = QHBoxLayout()
        label = QLabel("가중치:")
        label.setStyleSheet("font-weight:600; color:#58a6ff;")
        self._weight_selector = QComboBox()
        self._populate_weight_selector()
        self._weight_selector.currentIndexChanged.connect(self._on_weight_selected)
        selector_row.addWidget(label)
        selector_row.addWidget(self._weight_selector, 1)
        layout.addLayout(selector_row)
        layout.addWidget(self._sidebar, 1)
        return panel

    def _populate_weight_selector(self) -> None:
        """models_dir의 학습된 체크포인트로 콤보를 채운다.

        선호 순서(finetuned > jester)라 index 0이 시작 시 로드된 활성 모델과 같다 —
        별도 선택 없이 기본값(0)이 곧 현재 상태다. 학습된 체크포인트가 없으면 콤보를
        비활성으로 두어 실제 상태(인식 off)를 정직하게 드러낸다.
        """
        self._weight_selector.blockSignals(True)
        self._weight_selector.clear()
        for checkpoint in available_gesture_checkpoints(self._gesture_models_dir):
            self._weight_selector.addItem(_weight_label(checkpoint.name), userData=str(checkpoint))
        if self._weight_selector.count() == 0:
            self._weight_selector.addItem("학습된 가중치 없음", userData=None)
            self._weight_selector.setEnabled(False)
        self._weight_selector.blockSignals(False)

    def _on_weight_selected(self, index: int) -> None:
        data = self._weight_selector.itemData(index)
        if data is None:
            return  # "학습된 가중치 없음" placeholder — 전환할 대상이 없다
        self._switch_gesture_model(Path(str(data)))

    def _switch_gesture_model(self, checkpoint: Path) -> None:
        """선택한 체크포인트로 인식기를 실시간 교체한다(실패 시 정직하게 off).

        새 `GestureProbe`를 만들어 내부 상태(슬라이딩 윈도우·decimation)를 초기화하고,
        `_recognizer`(실제 인식 구동)와 `_gesture_source`(상태 표시)를 함께 바꾼다.
        """
        model = load_gesture_model_from(checkpoint)
        probe = GestureProbe(model_asset_path=None, model=model) if model is not None else None
        if probe is None or not probe.activate_headless():
            self._recognizer = None
            self._gesture_source = UntrainedGestureSource()
            self._sidebar.set_source(self._gesture_source)
            self._log.warn(f"가중치 전환 실패: {checkpoint.name} — 인식 off")
            return
        self._recognizer = probe
        self._gesture_source = ProbeGestureSource(probe)
        self._sidebar.set_source(self._gesture_source)
        self._log.info(f"제스처 인식 가중치 전환: {checkpoint.name}")

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
        top.addWidget(self._build_recognition_panel())
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
        return container

    def _build_gaze_recognition_tab(self) -> QWidget:
        """시선 샘플 저장 + 물체(Target) 등록 UI를 모은 탭.

        자체 카메라 뷰(`_gaze_video`)에 gaze·등록 가이드 오버레이를 그린다 — 실시간
        탭의 손/제스처 뷰(`_video`)와 분리해, 등록 중 물체 테두리를 따라가는 화면을
        이 탭에서 본다. 프레임은 `_on_frame`이 두 뷰에 fan-out한다.
        """
        self._gaze_video = VideoView()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._gaze_video, 1)
        self._sample_button = QPushButton()
        self._sample_button.clicked.connect(self._save_gaze_sample)
        self._clear_samples_button = QPushButton("샘플 초기화")
        self._clear_samples_button.clicked.connect(self._clear_gaze_samples)
        self._target_heatmap_toggle = QCheckBox("Target heatmap / 물체 영역 표시")
        self._target_heatmap_toggle.toggled.connect(self._gaze_video.set_target_heatmap_visible)
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

        # 라벨된 디버깅 세션 녹화: F9 시작/종료, 숫자키 0=아무것도 안 봄,
        # 1..9=등록 target n번을 정답 라벨로 표시. jarvis-gaze report가 집계한다.
        session_controls = QHBoxLayout()
        self._session_record_button = QPushButton("세션 녹화 시작 (F9)")
        self._session_record_button.clicked.connect(self._toggle_session_recording)
        self._session_label_combo = QComboBox()
        self._session_label_combo.currentIndexChanged.connect(self._on_session_label_changed)
        self._session_status = QLabel("세션 녹화 대기")
        self._session_status.setStyleSheet(_MONO)
        self._session_report_button = QPushButton("최근 세션 분석")
        self._session_report_button.clicked.connect(self._analyze_latest_session)
        self._session_report_button.setEnabled(self._latest_session_path() is not None)
        session_controls.addWidget(self._session_record_button)
        session_controls.addWidget(self._session_label_combo, 1)
        session_controls.addWidget(self._session_status, 1)
        session_controls.addWidget(self._session_report_button)
        layout.addLayout(session_controls)
        self._session_report_view = QPlainTextEdit()
        self._session_report_view.setReadOnly(True)
        self._session_report_view.setMaximumHeight(240)
        self._session_report_view.setPlaceholderText(
            "세션을 녹화하면 target별 정확도, 자세·거리·얼굴 위치 구간과 "
            "대표 실패 프레임이 여기에 표시됩니다."
        )
        self._session_report_view.setStyleSheet(
            "font-family:Consolas,monospace; font-size:11px;"
        )
        layout.addWidget(self._session_report_view)
        QShortcut(QKeySequence("F9"), self, self._toggle_session_recording)
        for digit in range(10):
            QShortcut(
                QKeySequence(str(digit)),
                self,
                lambda index=digit: self._select_session_label_by_digit(index),
            )
        self._registration_step = QLabel("2단계 물체 등록: 대기")
        self._registration_step.setStyleSheet("font-weight:700; color:#8b949e;")
        layout.addWidget(self._registration_step)
        self._registration_progress = QProgressBar()
        self._registration_progress.setRange(0, 1000)
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("등록 대기")
        layout.addWidget(self._registration_progress)
        self._registration_status = QLabel(
            "1단계에서는 물체 중앙 한 점을 응시한 채 고개·거리만 바꾸고, "
            "2단계에서는 고개를 고정한 채 눈으로 테두리를 정밀하게 따라갑니다."
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

    # --- 시연 탭 -----------------------------------------------------------

    def _build_demo_tab(self) -> QWidget:
        """웹캠 뷰 + 시연 패널. `VideoView`는 app.py에 있어 패널이 직접 못 만든다
        (순환 import) — `_build_live_tab`이 인식 패널을 붙이는 방식과 같이 여기서 잇는다.
        """
        # 관객용 화면에서는 gaze/HUD 디버그는 숨기되, 실제 MediaPipe 손 입력과
        # 제스처 판정을 사용자가 확인할 수 있도록 손 스켈레톤만 남긴다.
        self._demo_video = VideoView(show_overlay=False, show_hand_overlay=True)
        self._demo_panel = DemoPanel(
            on_mapping_changed=self._on_demo_mapping_changed,
            on_fallback_changed=self._on_demo_fallback_changed,
            on_preset_changed=self._on_demo_preset_changed,
            on_execution_toggled=self._on_demo_execution_toggled,
            on_open_registration=self._open_gaze_registration_tab,
        )
        self._refresh_demo_targets()
        self._refresh_bulb_badge()
        # 패널의 초기 상태(실행 켜짐·타깃 고정 laptop)를 브릿지에 명시 동기화한다.
        # 체크박스는 시그널 연결 전에 setChecked돼 생성 중 toggled를 발화하지 않으므로
        # (발화하면 아직 대입 전인 self._demo_panel을 콜백이 참조해 깨진다), 대입이
        # 끝난 지금 패널 값을 단일 소스로 삼아 브릿지에 반영한다.
        self._on_demo_fallback_changed(self._demo_panel.fallback_device)
        self._on_demo_execution_toggled(self._demo_panel.execution_enabled)
        if self._execute_worker is None:
            self._demo_panel.append_line(
                "실행기 없음 — 이 플랫폼에 입력 어댑터가 없어 기기 명령을 실행할 수 없습니다",
                ok=False,
            )
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._demo_video)
        split.addWidget(self._demo_panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(split)
        return container

    def _refresh_demo_targets(self) -> None:
        """등록 물체 목록을 매핑 표에 반영한다(등록·삭제·이름 변경 후)."""
        if not hasattr(self, "_demo_panel"):
            return  # 시연 탭이 아직 만들어지기 전(등록 탭이 먼저 빌드된다)
        choices = [
            TargetChoice(
                target_id=record.target_id,
                name=record.name,
                device_type=record.device_type,
            )
            for record in self._target_registry.records
        ]
        self._demo_panel.set_targets(choices, self._device_mapping.mapping)

    def _ensure_demo_mappings(self) -> None:
        """등록 시 선택한 기종을 실제 런타임 기기에 한 번만 자동 연결한다.

        사용자가 시연 탭에서 명시적으로 "연결 안 함"을 선택한 경우에는 None 선택이
        저장돼 있으므로 다시 덮지 않는다. 기존 프로필처럼 매핑 파일이 아예 없는 경우만
        device_type을 기준으로 안전하게 마이그레이션한다.
        """
        for record in self._target_registry.records:
            if self._device_mapping.has_selection(record.target_id):
                continue
            device_id = DEVICE_TYPE_TO_RUNTIME_ID.get(record.device_type)
            if device_id is not None:
                self._device_mapping.set_default(record.target_id, device_id)

    def _open_gaze_registration_tab(self) -> None:
        """시연 탭에서 복잡한 등록 UI를 복제하지 않고 원본 탭으로 이동한다."""
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index) == "시선 인식":
                self._tabs.setCurrentIndex(index)
                self._register_target_button.setFocus()
                return

    def _refresh_bulb_badge(self) -> None:
        configured = WizConfig.from_env(self._env) is not None
        badge = "설정됨 · 명령 대기" if configured else "미설정 (WIZ_DEVICE_TARGETS 없음)"
        self._last_bulb_badge = (badge, configured)
        self._demo_panel.set_bulb(self._virtual_bulb, badge=badge, ok=configured)

    def _on_demo_mapping_changed(self, target_id: str, device_id: str | None) -> None:
        self._device_mapping.set(target_id, device_id)
        # 매핑이 바뀌면 이전 대상으로 쌓인 lock을 이어받으면 안 된다.
        self._demo_bridge.reset()
        label = device_id or "연결 안 함"
        self._log.info(f"시연 기기 매핑: {target_id} → {label}")
        self._demo_panel.append_line(f"매핑 변경 {target_id} → {label}", ok=device_id is not None)

    def _on_demo_fallback_changed(self, device_id: str | None) -> None:
        self._demo_bridge.set_fallback(device_id)
        if device_id is None:
            self._log.info("시연 타깃 고정 해제 — 실제 시선 추정을 사용합니다")
            self._demo_panel.append_line("타깃 고정 해제", ok=False)
        else:
            self._log.warn(f"시연 타깃 고정: {device_id} (시선 판정을 우회합니다)")
            self._demo_panel.append_line(f"타깃 고정 → {device_id}", ok=True)
        self._update_demo_state(self._last_demo_gesture)

    def _on_demo_preset_changed(self, preset: DemoPreset) -> None:
        self._demo_bridge.reconfigure(preset)
        self._log.info(f"시연 임계값 프리셋: {preset.label} (target lock 초기화됨)")
        self._demo_panel.append_line(
            f"임계값 {preset.label} — dwell {preset.alignment.target_dwell_ms}ms, "
            f"commit {preset.fusion.commit_threshold:.2f} (lock 초기화)",
            ok=True,
        )

    def _on_demo_execution_toggled(self, enabled: bool) -> None:
        if enabled and self._execute_worker is None:
            self._demo_bridge.execution_enabled = False
            self._log.warn("실행기가 없어 기기 명령을 실행할 수 없습니다")
            self._demo_panel.append_line("실행기 없음 — 켜도 실행되지 않습니다", ok=False)
            self._demo_panel.set_execution_enabled(False)
            return
        self._demo_bridge.execution_enabled = enabled
        self._log.info(f"시연 기기 명령 실행 {'켜짐' if enabled else '꺼짐'}")
        self._demo_panel.append_line(
            "실제 기기 명령 실행 켜짐" if enabled else "판정 전용 모드 — 명령 실행 꺼짐",
            ok=enabled,
        )

    def _on_execution_outcome(self, outcome: object) -> None:
        """워커 스레드가 돌려준 실행 결과(GUI 스레드에서 수신)."""
        assert isinstance(outcome, ExecutionOutcome)
        line = describe_outcome(outcome)
        self._demo_panel.append_line(line, ok=outcome.executed)
        self._demo_panel.set_last_action(line, ok=outcome.executed)
        intent = outcome.intent
        if intent is None or intent.target != BULB_DEVICE_ID:
            return
        # 가상 전구는 **보낸 명령**을 누적한다. 실물의 성공 여부는 배지로 따로 말한다 —
        # dispatch가 실패했는데 그림만 밝아지면 성공을 지어내는 것이다.
        self._virtual_bulb.apply(intent)
        if outcome.executed:
            badge = "OK · " + outcome.detail
        elif outcome.stage is ExecutionStage.DISPATCHED:
            badge = "실패: " + outcome.detail
        else:
            badge = "미전달: " + outcome.detail
        self._last_bulb_badge = (badge, outcome.executed)
        self._demo_panel.set_bulb(self._virtual_bulb, badge=badge, ok=outcome.executed)

    def _on_execution_failed(self, message: str) -> None:
        self._log.error(message)
        self._demo_panel.append_line(message, ok=False)

    def _update_demo_state(self, gesture: str) -> None:
        self._last_demo_gesture = gesture
        # "실시간 시선"은 이번 gaze 프레임의 원시 classifier 결과를 그대로
        # 기기 id로 치환한 값이다 — Fusion dwell을 거치지 않아 프레임마다
        # 흔들릴 수 있다("바라보는 기기"는 그 이후 확정/후보 상태). 타깃
        # 고정(fallback)이 켜져 있으면 원시 시선 대신 고정된 기기를 그대로
        # 보여준다 — push_target()이 실제로 쓰는 값과 화면이 어긋나지 않게.
        raw_target: str | None = None
        if self._demo_bridge.fallback_device is not None:
            raw_target = self._demo_bridge.fallback_device
        elif self._latest_gaze is not None:
            resolved = self._demo_bridge.resolve_target(self._latest_gaze.target_estimate.target)
            raw_target = resolved if resolved != UNKNOWN_TARGET else None
        self._demo_panel.set_state(
            locked=self._demo_bridge.locked_device,
            candidate=self._demo_bridge.candidate_device,
            phase=str(self._demo_bridge.intent_phase),
            gesture=gesture,
            suppressed=self._demo_bridge.should_suppress_pose,
            raw_target=raw_target,
        )

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

    def _demo_tab_active(self) -> bool:
        """시연 탭에서는 정적 pose 제어 대신 TCN→Fusion 명령 경로만 사용한다."""
        return self._tabs.tabText(self._tabs.currentIndex()) == "시연"

    def _handle_commit_decision(self, decision: CommitDecision) -> None:
        """Fusion 커밋 판정 하나를 화면에 남기고, 통과하면 실행 워커로 넘긴다.

        `_on_hand`에서 분리해 둔다 — 실행 여부를 가르는 게이트가 두 개(아래)라
        카메라 프레임 없이 단위 테스트로 고정할 수 있어야 하기 때문이다.
        """
        self._demo_panel.append_line(describe_decision(decision), ok=decision.committed)
        if not decision.committed:
            return
        # 노트북(내 PC) 대상 명령은 "손동작으로 컴퓨터 제어" 토글도 지켜야 한다. 이
        # 경로(TCN→Fusion→Intent)는 정적 pose 제어와 배선이 달라 예전에는 그 토글을
        # 통과하지 않았고, 체크를 꺼도 두 손가락 slide가 데스크톱을 전환했다(2026-07-22
        # 발견). 전구는 OS 입력이 아니라 네트워크 명령이라 이 토글의 대상이 아니다 —
        # 시연 실행 토글만 본다.
        if decision.target == LAPTOP_DEVICE_ID and not self._pose_control.enabled:
            self._demo_panel.set_last_action(
                f"{decision.gesture} → {decision.target} "
                "(손동작으로 컴퓨터 제어 꺼짐 — 실행 안 함)",
                ok=False,
            )
        elif self._demo_bridge.execution_enabled and self._execute_worker is not None:
            self._execute_worker.submit(decision)
        else:
            self._demo_panel.set_last_action(
                f"{decision.gesture} → {decision.target} (실행 안 함)", ok=False
            )

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
        # 시선 인식 탭·파인튜닝 탭의 웹캠 뷰에도 같은 프레임을 뿌린다(Qt 시그널 fan-out —
        # 각 VideoView가 자기 복사본에 그리므로 서로 간섭하지 않는다).
        self._gaze_video.show_frame(frame)
        self._finetune_panel.video.show_frame(frame)
        self._demo_video.show_frame(frame)

    def _on_gaze(self, snapshot: object) -> None:
        assert isinstance(snapshot, GazeSnapshot)
        self._latest_gaze = snapshot
        # Gaze→Fusion 스트림(계약 §1). 브리지가 등록 물체 id를 런타임 기기 id로
        # 치환하고(매핑 없으면 UNKNOWN), 타깃 고정이 켜져 있으면 합성 추정치로 대체한다.
        self._demo_bridge.push_target(snapshot.target_estimate)
        self._update_demo_state(self._last_demo_gesture)
        self._demo_video.set_gaze(snapshot)
        self._gaze_history.append(snapshot)
        cutoff_ms = snapshot.timestamp_ms - 500
        while self._gaze_history and self._gaze_history[0].timestamp_ms < cutoff_ms:
            self._gaze_history.popleft()
        self._gaze_video.set_gaze(snapshot)
        self._gaze_panel.update_snapshot(snapshot)
        if self._session_recorder.recording:
            self._session_recorder.record(snapshot, self._session_label)
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
            accepted = self._registration.add(
                smoothed,
                snapshot.tracking_confidence,
                eyes_open=snapshot.eyes_open,
                face_scale=snapshot.face_scale,
                feature_sample=snapshot.feature_sample,
            )
            if (
                not accepted
                and not snapshot.eyes_open
                and snapshot.left_eye_open_ratio is not None
                and snapshot.right_eye_open_ratio is not None
                and snapshot.left_eye_open_baseline is not None
                and snapshot.right_eye_open_baseline is not None
            ):
                self._registration.last_rejection_reason = (
                    "eyes classified closed "
                    f"(L {snapshot.left_eye_open_ratio:.3f}/"
                    f"{snapshot.left_eye_open_baseline:.3f}, "
                    f"R {snapshot.right_eye_open_ratio:.3f}/"
                    f"{snapshot.right_eye_open_baseline:.3f})"
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
        device_type, ok = QInputDialog.getItem(
            self,
            "물체 등록",
            "기종",
            list(_TARGET_DEVICE_TYPES),
            0,
            False,
        )
        if not ok or device_type not in _TARGET_DEVICE_TYPES:
            return
        # 카메라 정면 근처에 두는 computer(노트북) 기종은 "가만히 있을 때의
        # 기본 시선 방향"과 겹쳐 다른 물체를 안 보고 있을 뿐인 순간에도
        # 오확정되기 쉽다(2026-07-22). 각도가 확실히 갈라진 electric bulb는
        # 이 문제가 없어 묻지 않는다 — 다른 물체에서 computer로 돌아올 때만
        # 끄덕임 확인을 요구할지 등록 시점에 정한다.
        requires_nod_gate = device_type == "computer" and (
            QMessageBox.question(
                self,
                "끄덕임 확인 게이트",
                f"'{name.strip()}'로 돌아올 때 고개 끄덕임 확인을 요구할까요?\n"
                "(다른 물체를 보다가 이 물체로 돌아올 때만 적용 — 카메라 정면"
                " 근처 물체의 오확정 방지용)",
            )
            == QMessageBox.StandardButton.Yes
        )
        target_id = self._next_target_id()
        self._begin_registration(
            target_id, name.strip(), device_type, target_id, requires_nod_gate=requires_nod_gate
        )

    def _next_target_id(self) -> str:
        existing = {record.target_id for record in self._target_registry.records}
        index = 1
        while True:
            target_id = f"target_{index:03d}"
            if target_id not in existing:
                return target_id
            index += 1

    def _begin_registration(
        self,
        target_id: str,
        name: str,
        device_type: str,
        device_id: str,
        *,
        requires_nod_gate: bool = False,
    ) -> None:
        if self._registration is not None:
            self._log.warn("이미 기기 등록이 진행 중입니다")
            return
        self._registration = TargetRegistrationSession(
            target_id, name, device_type, device_id, config=self._gaze_config,
            coverage_min_frames=self._gaze_config.registration_coverage_min_frames,
            raw_sample_dir=Path("data/calibration/raw_samples"),
            requires_nod_gate=requires_nod_gate,
        )
        self._registration_phase_marker = None
        self._set_registration_controls(active=True)
        self._registration_step.setText("1/2 중앙 응시 + 고개 스윕 · 조건 충족까지")
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("1/2 중앙 응시 + 고개 스윕  %p%")
        self._registration_status.setText(
            f"'{name}' ({device_type}) 중앙 한 점에서 눈을 떼지 마세요. 안내에 따라 고개를 "
            "좌우 끝까지·위아래로 돌리고 카메라 거리를 바꿔 자세별 편향을 수집합니다."
        )
        self._registration_status.setStyleSheet(
            "background:#3d2a12; color:#f0b429; border:1px solid #7a5a1e;"
            " border-radius:6px; padding:8px; font-weight:700;"
        )
        self._log.info(
            f"'{name}' ({device_type}) 2단계 등록 시작 — 1/2 중앙 응시 + 고개 스윕: "
            "물체 중앙 한 점을 계속 응시하면서 고개를 좌우 끝까지·위아래로 돌리고 거리 변경"
        )
        self._gaze_video.set_registration_guidance(
            "REGISTRATION 1/2 - POSE SWEEP",
            "FIX CENTER, TURN HEAD",
            0.0,
        )

    def _update_registration_guidance(self, timestamp_ms: int) -> None:
        assert self._registration is not None
        if self._registration.started_at_ms is None:
            reason = self._registration.last_rejection_reason
            reason_text = reason or "waiting for the first valid face + iris frame"
            self._registration_status.setText(
                "등록 대기: 첫 유효 시선 프레임을 기다리는 중 · "
                f"현재 차단 원인: {reason_text} · "
                f"{self._registration.diagnostic_summary()}"
            )
            self._gaze_video.set_registration_guidance(
                "REGISTRATION WAITING",
                f"BLOCKED: {reason_text.upper()}",
                0.0,
            )
            return
        phase = self._registration.phase
        if phase == RegistrationPhase.COMPLETE:
            self._registration_step.setText("2/2 물체 영역 확정 완료")
            self._registration_progress.setValue(1000)
            self._registration_progress.setFormat("등록 데이터 처리 중  %p%")
            self._gaze_video.set_registration_guidance(
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
            step = "1/2 중앙 응시 + 고개 스윕"
            count = self._registration.center_valid_frame_count
            required = self._registration.minimum_valid_frames
            title = "REGISTRATION 1/2 - POSE SWEEP"
        else:
            step = "2/2 물체 영역 확정"
            count = self._registration.boundary_valid_frame_count
            required = self._registration.minimum_boundary_frames
            title = "REGISTRATION 2/2 - TRACE BOUNDARY"
        coverage = self._registration.coverage
        coverage_active = phase == RegistrationPhase.CENTER and coverage is not None
        if (
            not coverage_active
            and elapsed_ms >= self._registration.phase_duration_ms()
            and count < required
        ):
            label = "시간은 완료됐지만 유효 프레임이 부족합니다. 안내 자세를 유지해 주세요."
            video_label = "HOLD POSITION - NEED MORE VALID FRAMES"
        if coverage_active:
            assert coverage is not None
            # 조건 충족식 1단계: 시간 대신 구간별 수집 현황을 안내한다.
            rows = "  ".join(
                f"{item.label} {item.count}/{item.required}{'✓' if item.met else ''}"
                for item in coverage.report()
            )
            missing = coverage.missing_labels()
            hint = f"다음 구간을 채워주세요: {', '.join(missing)}" if missing else "모든 구간 완료!"
            status = f"물체 중앙을 계속 바라보세요 — {hint}\n{rows}"
        else:
            status = (
                f"{label}\n현재 구간 남은 시간 {remaining_s:0.1f}s · 유효 프레임 {count}/{required} · "
                f"자세별 {self._registration.center_valid_frame_count} / 정밀 "
                f"{self._registration.boundary_valid_frame_count}"
            )
        if self._registration.last_rejection_reason is not None:
            status += f" · 현재 제외: {self._registration.last_rejection_reason}"
        self._registration_step.setText(step)
        self._registration_progress.setValue(round(progress * 1000))
        self._registration_progress.setFormat(f"{step}  %p%")
        self._registration_status.setText(status)
        self._gaze_video.set_registration_guidance(title, video_label, progress)
        marker = (phase, phase_index)
        if marker != self._registration_phase_marker:
            previous_phase = (
                self._registration_phase_marker[0]
                if self._registration_phase_marker is not None
                else None
            )
            if phase == RegistrationPhase.BOUNDARY and previous_phase != phase:
                self._log.info(
                    "1/2 자세·거리 문맥 수집 완료 — 2/2 정밀 영역 확정 시작: "
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
            if not self._device_mapping.has_selection(record.target_id):
                device_id = DEVICE_TYPE_TO_RUNTIME_ID.get(record.device_type)
                if device_id is not None:
                    self._device_mapping.set(record.target_id, device_id)
            self._probe.register_profile(
                record.to_profile(),
                geometry_3d=record.to_geometry_3d(),
                feature_profile=record.feature_profile,
                area_profile=record.area_profile,
                pose_correction=record.pose_correction,
                requires_nod_gate=record.requires_nod_gate,
                label=record.name,
            )
            self._log.info(
                f"'{record.name}' 2단계 등록 완료 "
                f"(중앙 응시 스윕 {self._registration.center_valid_frame_count}, "
                f"정밀 테두리 {self._registration.boundary_valid_frame_count} frames) | "
                f"{self._registration.diagnostic_summary()}"
            )
            # 등록 품질 린트: 스윕 커버리지·clamp·희박 bin 문제를 지금 잡아야
            # 나중에 "왜 안 되지"로 돌아오지 않는다.
            for warning in lint_target_record(record, self._gaze_config):
                self._log.warn(f"등록 린트 [{record.name}]: {warning}")
        except ValueError as exc:
            self._log.warn(f"기기 등록 실패: {exc} | {self._registration.diagnostic_summary()}")
        finally:
            self._clear_registration_state()
            self._refresh_targets()

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
        self._set_registration_controls(active=False)
        self._registration_step.setText("2단계 물체 등록: 대기")
        self._registration_progress.setValue(0)
        self._registration_progress.setFormat("등록 대기")
        self._registration_status.setText(
            "1단계에서는 물체 중앙 한 점을 응시한 채 고개·거리만 바꾸고, "
            "2단계에서는 고개를 고정한 채 눈으로 테두리를 정밀하게 따라갑니다."
        )
        self._registration_status.setStyleSheet(
            "background:#161b22; color:#8b949e; border:1px solid #30363d;"
            " border-radius:6px; padding:8px; font-weight:600;"
        )
        self._gaze_video.clear_registration_guidance()

    def _reregister_selected_target(self) -> None:
        record = self._selected_target()
        if record is None:
            self._log.warn("위치를 다시 등록할 기기를 선택하세요")
            return
        self._begin_registration(
            record.target_id,
            record.name,
            record.device_type,
            record.device_id,
            requires_nod_gate=record.requires_nod_gate,
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
                pose_correction=updated.pose_correction,
                requires_nod_gate=updated.requires_nod_gate,
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
        self._device_mapping.remove(record.target_id)
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
                f" hull={len(record.area_profile.boundary_polygon)}"
                f" cap={self._gaze_config.registration_max_area_radius_deg:.1f}"
                if record.area_profile is not None
                else " area=none"
            )
            nod_label = " 🔁끄덕임필요" if record.requires_nod_gate else ""
            self._target_list.addItem(
                f"{record.name} [{record.target_id}]  type={record.device_type}  "
                f"yaw={record.direction.yaw:+.1f} pitch={record.direction.pitch:+.1f}"
                f" spread={record.spread.yaw:.1f}/{record.spread.pitch:.1f}"
                f"{scale_label}{feature_label}{area_label}{nod_label}"
            )
        self._refresh_session_labels()
        # 시연 탭의 "물체 → 기기" 매핑 표도 같은 목록에서 다시 그린다.
        self._refresh_demo_targets()

    def _refresh_session_labels(self) -> None:
        """세션 라벨 콤보를 등록 target 목록과 동기화한다(선택 유지)."""
        current = self._session_label
        combo = self._session_label_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("라벨 없음 (집계 제외)", None)
        combo.addItem("0: 아무것도 안 봄", NO_TARGET_LABEL)
        for index, record in enumerate(self._target_registry.records, start=1):
            prefix = f"{index}: " if index <= 9 else ""
            combo.addItem(f"{prefix}{record.name} [{record.target_id}]", record.target_id)
        restored = combo.findData(current)
        combo.setCurrentIndex(restored if restored >= 0 else 0)
        combo.blockSignals(False)
        self._session_label = combo.currentData()

    def _on_session_label_changed(self) -> None:
        self._session_label = self._session_label_combo.currentData()
        self._update_session_status()

    def _select_session_label_by_digit(self, digit: int) -> None:
        """숫자키 라벨 선택: 0=아무것도 안 봄, 1..9=등록 target n번."""
        target_value = (
            NO_TARGET_LABEL
            if digit == 0
            else (
                self._target_registry.records[digit - 1].target_id
                if digit - 1 < len(self._target_registry.records)
                else None
            )
        )
        if target_value is None:
            return
        index = self._session_label_combo.findData(target_value)
        if index >= 0:
            self._session_label_combo.setCurrentIndex(index)

    def _toggle_session_recording(self) -> None:
        if self._session_recorder.recording:
            summary = self._session_recorder.stop()
            path = self._session_recorder.path
            self._log.info(
                f"세션 녹화 종료: {path} — {summary['frames']} frames, "
                f"labels {summary['labels']}"
            )
            self._session_record_button.setText("세션 녹화 시작 (F9)")
            self._update_session_status()
            if path is not None:
                self._show_session_report(path)
            return
        path = self._diagnostics_dir / time.strftime("session_%Y%m%d_%H%M%S.jsonl")
        self._session_recorder.start(
            path,
            config=self._gaze_config,
            targets=self._target_registry.records,
        )
        self._log.info(f"세션 녹화 시작: {path} (숫자키로 정답 라벨 표시)")
        self._session_record_button.setText("세션 녹화 종료 (F9)")
        self._update_session_status()

    def _latest_session_path(self) -> Path | None:
        if not self._diagnostics_dir.is_dir():
            return None
        paths = list(self._diagnostics_dir.glob("session_*.jsonl"))
        return max(paths, key=lambda item: item.stat().st_mtime_ns) if paths else None

    def _analyze_latest_session(self) -> None:
        path = self._latest_session_path()
        if path is None:
            self._log.warn("분석할 gaze 세션 파일이 없습니다")
            return
        self._show_session_report(path)

    def _show_session_report(self, path: Path) -> None:
        try:
            report = build_report(load_session(path), path=path)
            rendered = format_report(report)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._session_report_view.setPlainText(f"세션 분석 실패: {error}")
            self._log.warn(f"gaze 세션 분석 실패: {error}")
            return
        self._session_report_view.setPlainText(rendered)
        self._session_report_view.moveCursor(QTextCursor.MoveOperation.Start)
        self._log.info(f"gaze 세션 분석 완료: {path}")

    def _update_session_status(self) -> None:
        label = self._session_label if self._session_label is not None else "(없음)"
        if self._session_recorder.recording:
            self._session_status.setText(
                f"REC {self._session_recorder.frame_count} frames | label: {label}"
            )
            self._session_status.setStyleSheet(
                "font-family:Consolas,monospace; font-size:12px; color:#f85149;"
            )
        else:
            self._session_status.setText(f"세션 녹화 대기 | label: {label}")
            self._session_status.setStyleSheet(_MONO)
        self._session_report_button.setEnabled(self._latest_session_path() is not None)

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
        self._demo_video.set_hand(snapshot)
        # 손을 놓치면 드래그가 눌린 채 남지 않게 먼저 정리한다.
        if not snapshot.hand_detected:
            self._pose_control.release()
        # 시선 lock에 의한 경로 중재: 노트북이 아닌 기기(전구)를 보는 동안에는 pose
        # 제어를 멈춘다 — 전구를 보며 손을 움직일 때 커서까지 따라가면 안 된다. 사용자의
        # 제어 토글(`_pose_control.enabled`)은 건드리지 않고 억제만 얹으므로, 전구에서
        # 시선을 떼면 커서가 곧바로 되살아난다. pose_control.py는 수정하지 않는다.
        #
        # release()는 **억제로 들어가는 전이에서 한 번만** 부른다. 매 프레임 부르면
        # macOS sink의 restore_dock()이 초당 30번 호출된다(Windows sink엔 없어 무해하지만
        # 플랫폼에 기대지 않는다).
        # 시연 탭은 동적 TCN 명령 하나만 실제 실행 경로로 쓴다. 여기서 정적 pose
        # 제어까지 함께 실행하면 노트북 대상 slide가 pose scroll + Intent scroll로
        # 중복 작동한다. 손 스켈레톤·pose 판정은 계속 그리되 OS 입력만 막는다.
        suppressed = self._demo_tab_active() or self._demo_bridge.should_suppress_pose
        if suppressed:
            if not self._pose_suppressed:
                self._pose_control.release()
        else:
            self._pose_control.apply(list(snapshot.pose_events))
        self._pose_suppressed = suppressed
        self._video.set_control(self._pose_control.last_action, self._pose_control.enabled)
        self._demo_video.set_control(
            "TCN 판정 대기",
            self._demo_bridge.execution_enabled,
        )
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
            if tick is not None:
                # Gesture→Fusion 스트림(계약 §2). 솎인 프레임(None)에는 아무것도 밀지
                # 않는다 — 12fps로 학습된 모델의 cadence를 그대로 유지한다. 판정은
                # 제스처가 완결된 프레임에서만 나온다.
                decision = self._demo_bridge.push_gesture(tick.estimate)
                if decision is not None:
                    self._handle_commit_decision(decision)
                self._update_demo_state(tick.estimate.gesture)
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
        self._gaze_video._show_placeholder("NO CAMERA")

    def start_console_capture(self) -> None:
        """프로세스 stderr(fd 2)를 하단 콘솔 독으로 가로챈다. run()이 창을 띄운 뒤 호출한다.

        __init__이 아니라 여기서 하는 이유: __init__은 테스트가 창을 만들 때도 돌아,
        거기서 fd 2를 리다이렉트하면 pytest의 stderr까지 삼켜버린다.
        """
        if self._stderr_capture is not None:
            return
        self._stderr_capture = StderrCapture(self._console_log.add)
        self._stderr_capture.start()

    def _on_tick(self) -> None:
        # 인식 스트림은 _on_hand에서 프레임마다(12fps로 솎아) 직접 찍는다 — 여기선 안 건드림.
        self._messages.refresh()
        self._console_panel.refresh()
        self._latency_panel.refresh()
        self._update_session_status()

    def keyPressEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        """ESC로 창을 닫고, 파인튜닝 탭이 보이는 동안엔 스페이스바로 녹화를 토글한다.

        디버깅 툴은 자세를 바꿔가며 반복해서 띄웠다 닫는 도구라, 창 버튼까지 마우스를
        옮기지 않고 닫을 수 있어야 한다. 녹화도 같은 이유(카메라 앞에서 자세를 잡은 채
        마우스로 REC 버튼을 찾아 누르기 번거로움)로 단축키를 둔다 — person_id 입력칸이
        포커스를 쥐고 있으면 Qt가 스페이스를 그 위젯에서 먼저 소비하므로(텍스트 입력)
        여기까지 전달되지 않아 타이핑과 충돌하지 않는다. 동작 콤보·REC 버튼에 포커스가
        있는 경우는 `FinetuneRecordingPanel`의 이벤트 필터가 위젯 자신에게 전달되기
        전에 가로채므로(팝업 열기/클릭 자체가 안 일어남) 여기까지 오지 않는다 — 그 외
        모든 경우(포커스 없음·비디오 뷰 등)만 이 핸들러가 처리한다.
        """
        if event.key() == Qt.Key.Key_Escape:  # type: ignore[attr-defined]
            self.close()
            return
        if (
            event.key() == Qt.Key.Key_Space  # type: ignore[attr-defined]
            and self._tabs.currentWidget() is self._finetune_panel
        ):
            self._toggle_recording()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        if self._session_recorder.recording:
            self._session_recorder.stop()
        if self._camera is not None:
            self._camera.stop()
        # 실행 워커를 확실히 접는다 — 진행 중인 명령 하나는 끝까지 보내고 종료한다.
        if self._execute_worker is not None:
            self._execute_worker.stop()
        # stderr를 원래 콘솔로 복원한다 — 프로세스 종료 후에도 stderr가 깨지지 않게.
        if self._stderr_capture is not None:
            self._stderr_capture.stop()
        super().closeEvent(event)  # type: ignore[arg-type]


def _default_model_path() -> Path | None:
    # Always return the conventional path; GazeProbe/pipeline status explain its
    # absence honestly rather than silently disabling the stage.
    return Path("models/face_landmarker.task")


def _default_hand_model_path() -> Path | None:
    return Path("models/hand_landmarker.task")


def _default_models_dir() -> Path:
    return Path("models")


def _weight_label(checkpoint_name: str) -> str:
    """체크포인트 파일명 → 가중치 셀렉터에 표시할 사람이 읽는 라벨."""
    return {
        "gesture_tcn_finetuned.pt": "파인튜닝 (finetuned)",
        "gesture_tcn_jester.pt": "사전학습 (jester)",
    }.get(checkpoint_name, checkpoint_name)


def _default_pose_model_path() -> Path:
    """`training/train_pose.py`의 기본 산출물 경로. 없으면 프로브가 사유를 표시한다."""
    return Path("models/hand_pose_classifier.pt")


def _default_profiles_path() -> Path:
    return Path("data/calibration/profiles.json")


def _load_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(read_env_file(Path(".env")))
    return env


def run(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.show()
    # 창을 띄운 뒤 stderr 캡처를 켠다 — 이후 네이티브 로그는 터미널이 아니라 하단 콘솔 독으로.
    window.start_console_capture()
    return int(app.exec())
