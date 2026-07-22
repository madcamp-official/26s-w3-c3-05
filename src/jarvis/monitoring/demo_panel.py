"""'시연' 탭의 표시·조작 계층.

`FinetuneRecordingPanel`과 같은 규약을 따른다 — 이 위젯은 저장소도 엔진도 직접
모른다. 조작은 주입받은 콜백만 부르고, 결과는 MainWindow가 `set_*`로 돌려준다.
그래서 카메라·Fusion 없이도 위젯 트리만 따로 검증할 수 있다.

웹캠 뷰는 여기 두지 않는다. `VideoView`는 `app.py`에 있어 여기서 import하면 순환
참조가 되므로, MainWindow가 `_build_demo_tab()`에서 `VideoView`와 이 패널을
나란히 배치한다(`_build_live_tab`이 `_build_recognition_panel()`을 붙이는 방식과
같다).

화면이 지켜야 할 정직성 경계 하나: **가상 전구는 '명령 기준' 상태이고 실물의
응답이 아니다.** 그래서 전구 그림 옆에 실물 결과 배지를 따로 둔다 — 실물이
실패했는데 그림만 밝아지면 그건 성공을 지어내는 것이다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.monitoring.demo_bridge import (
    DEMO_PRESETS,
    LAPTOP_DEVICE_ID,
    RUNTIME_DEVICE_IDS,
    RUNTIME_DEVICE_LABELS,
    DemoPreset,
)
from jarvis.monitoring.virtual_bulb import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    COLOR_TEMPERATURE_MAX,
    COLOR_TEMPERATURE_MIN,
    VirtualBulbState,
)
if TYPE_CHECKING:
    from jarvis.monitoring.hand_probe import HandSnapshot

# 이 앱에는 전역 다크 테마가 없다 — 어두운 외관은 위젯마다 stylesheet로 직접 만든다
# (`app.py`의 `_MONO`가 색을 항상 함께 지정하는 것과 같은 규약). 그래서 `background`만
# 주고 `color`를 빼면 글자가 시스템 기본색으로 떨어져 **검은 배경에 검은 글씨**가 된다.
# 배경을 지정하는 스타일은 반드시 색도 함께 지정한다. 상태별 색은 이 기본값 뒤에 덧붙여
# 덮어쓴다(뒤에 온 선언이 이긴다).
_STATUS_TEXT = "#c9d1d9"
_STATUS_STYLE = (
    "background:#161b22; border:1px solid #30363d; border-radius:6px;"
    f" padding:6px 10px; font-weight:700; color:{_STATUS_TEXT};"
)


@dataclass(frozen=True, slots=True)
class TargetChoice:
    """물체 관리 목록 한 줄 — 등록된 물체 하나."""

    target_id: str
    name: str
    device_type: str = ""


def _pilot_number(pilot: Mapping[str, object], key: str, default: float) -> float:
    """getPilot 응답 한 필드를 안전하게 float로. 없거나 비수치면 default."""
    value = pilot.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


class BulbView(QWidget):
    """실물 전구 — 주기적으로 읽어온 실측 ``getPilot`` 상태를 원 하나로 그린다.

    아직 한 번도 조회에 성공하지 못했으면(시작 직후·설정 없음·통신 실패) 꺼짐과
    같은 회색으로 그린다 — 마지막으로 **보낸** 명령을 지어내 "실물이 이렇다"고
    보여주지 않는다(development-principles 1.1, 정직한 상태).
    """

    def __init__(self) -> None:
        super().__init__()
        # 시연 상태를 보조하는 아이콘일 뿐 핵심 화면이 아니다. 레이아웃이 남는 세로
        # 공간을 모두 줘 원이 200px 이상 커지지 않도록 작은 고정 크기로 제한한다.
        self.setFixedSize(56, 56)
        self._pilot: Mapping[str, object] | None = None

    def set_live_state(self, pilot: Mapping[str, object] | None) -> None:
        """실물에서 막 읽어온 ``getPilot`` 결과. 조회 실패·미설정이면 ``None``."""
        self._pilot = pilot
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height()) - 8
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        painter.setPen(QColor("#30363d"))
        painter.setBrush(self._bulb_color())
        painter.drawEllipse(x, y, size, size)
        painter.end()

    def _bulb_color(self) -> QColor:
        pilot = self._pilot
        if pilot is None or not bool(pilot.get("state", False)):
            return QColor("#21262d")
        red, green, blue = self._tint(pilot)
        # 밝기는 명도로. 하한이 10이라 완전히 검어지지는 않는다(꺼짐과 구분).
        dimming = _pilot_number(pilot, "dimming", BRIGHTNESS_MIN)
        level = (dimming - BRIGHTNESS_MIN) / (BRIGHTNESS_MAX - BRIGHTNESS_MIN)
        scale = 0.35 + 0.65 * max(0.0, min(1.0, level))
        return QColor(int(red * scale), int(green * scale), int(blue * scale))

    def _tint(self, pilot: Mapping[str, object]) -> tuple[int, int, int]:
        """색조. 실물 WiZ와 같이 색상 모드와 색온도 모드 중 하나를 따른다.

        ``temp`` 필드 존재로 모드를 가른다 — CCT 모드에서는 WiZ가 ``temp``를
        주고 r/g/b는 아예 안 주거나 0으로 준다(wiz.py 모듈 docstring과 같은 관찰).
        """
        if "temp" in pilot:
            span = COLOR_TEMPERATURE_MAX - COLOR_TEMPERATURE_MIN
            temp = _pilot_number(pilot, "temp", COLOR_TEMPERATURE_MIN)
            warmth = max(0.0, min(1.0, (temp - COLOR_TEMPERATURE_MIN) / span))
            return 255, int(170 + 70 * warmth), int(90 + 155 * warmth)
        # 색상 모드: 기기가 보고한 r/g/b를 그대로 쓴다(재채도 변환 없음) — WiZ 앱
        # 등 다른 경로로 파스텔 색을 넣었을 때도 화면이 실물과 같은 색을 낸다.
        red = int(max(0, min(255, _pilot_number(pilot, "r", 0))))
        green = int(max(0, min(255, _pilot_number(pilot, "g", 0))))
        blue = int(max(0, min(255, _pilot_number(pilot, "b", 0))))
        return red, green, blue


class DemoPanel(QWidget):
    """시연 탭 우측 본문 — 상태 스트립 + 가상 전구 + 설정 + 물체 관리.

    판정·실행 로그(`log_widget`)는 이 패널에 배치하지 않는다 — MainWindow가 웹캠
    바로 밑에 둔다. 위젯 자체는 여기서 만들어 `append_line`이 계속 이 패널을 통해
    쓰지만, 부모 레이아웃은 밖에서 정해진다.
    """

    def __init__(
        self,
        *,
        on_fallback_changed: Callable[[str | None], None],
        on_preset_changed: Callable[[DemoPreset], None],
        on_execution_toggled: Callable[[bool], None],
        on_register_target: Callable[[], None] | None = None,
        on_reregister_target: Callable[[str], None] | None = None,
        on_rename_target: Callable[[str], None] | None = None,
        on_delete_target: Callable[[str], None] | None = None,
        on_cancel_registration: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_fallback_changed = on_fallback_changed
        self._on_preset_changed = on_preset_changed
        self._on_execution_toggled = on_execution_toggled
        self._on_reregister_target = on_reregister_target
        self._on_rename_target = on_rename_target
        self._on_delete_target = on_delete_target
        self._target_ids: list[str] = []

        layout = QVBoxLayout(self)

        # --- 상태 스트립 -------------------------------------------------
        # 실시간 시선(raw_target)과 바라보는 기기(locked/candidate)는 서로 다른
        # 신호다 — raw는 이번 프레임 classifier 결과 그대로(매 프레임 흔들릴 수
        # 있음), "바라보는 기기"는 Fusion이 dwell로 확정/후보 판정한 값이다.
        # 시연에서 "왜 아직 안 잡히지"를 raw로 바로 보여주기 위해 분리한다.
        self._raw_target_label = QLabel("실시간 시선: -")
        self._target_label = QLabel("바라보는 기기: -")
        self._gesture_label = QLabel("제스처: -")
        self._phase_label = QLabel("Intent: IDLE")
        for label in (
            self._raw_target_label,
            self._target_label,
            self._gesture_label,
            self._phase_label,
        ):
            label.setStyleSheet(_STATUS_STYLE)
            layout.addWidget(label)

        self._last_action = QLabel("마지막 실행: 없음")
        self._last_action.setWordWrap(True)
        self._last_action.setStyleSheet(_STATUS_STYLE + " color:#8b949e;")
        layout.addWidget(self._last_action)

        # --- 전구 ----------------------------------------------------------
        bulb_row = QHBoxLayout()
        self.bulb_view = BulbView()
        bulb_row.addWidget(self.bulb_view)
        bulb_text = QVBoxLayout()
        self._bulb_state_label = QLabel("밝기 60% · 색온도 4000K")
        self._bulb_state_label.setStyleSheet(f"font-weight:700; color:{_STATUS_TEXT};")
        bulb_text.addWidget(self._bulb_state_label)
        bulb_note = QLabel("원은 전구에서 직접 읽어온 실측 색입니다. 위 문구는 보낸 명령 기준입니다.")
        bulb_note.setWordWrap(True)
        bulb_note.setStyleSheet("color:#6e7681; font-size:11px;")
        bulb_text.addWidget(bulb_note)
        self._bulb_badge = QLabel("실물: 미설정")
        self._bulb_badge.setWordWrap(True)
        self._bulb_badge.setStyleSheet("color:#8b949e; font-weight:700;")
        bulb_text.addWidget(self._bulb_badge)
        bulb_text.addStretch(1)
        bulb_row.addLayout(bulb_text, 1)
        layout.addLayout(bulb_row)

        # --- 실행 스위치 --------------------------------------------------
        # 기본 켜짐: 동적 제스처로 노트북을 바로 제어할 수 있게 한다(사용자 지시,
        # 2026-07-22). setChecked는 아래 connect보다 먼저라 생성 중 toggled를
        # 발화하지 않으므로, 상태 라벨은 초기 체크 상태를 직접 반영해 만든다 —
        # 초기 bridge 상태는 MainWindow가 패널 값을 읽어 명시 동기화한다
        # (app.py `_build_demo_tab`).
        self._execution_toggle = QCheckBox("기기 명령 실행")
        self._execution_toggle.setChecked(True)
        self._execution_toggle.toggled.connect(self._emit_execution)
        layout.addWidget(self._execution_toggle)
        self._execution_status = QLabel(
            "실행 활성 · 확정된 제스처를 실제 컴퓨터/전구에 전달"
            if self._execution_toggle.isChecked()
            else "판정 전용 · 실제 컴퓨터/전구 명령은 실행하지 않음"
        )
        self._execution_status.setStyleSheet(
            ("color:#3fb950;" if self._execution_toggle.isChecked() else "color:#8b949e;")
            + " font-weight:600;"
        )
        layout.addWidget(self._execution_status)

        # --- 폴백(타깃 고정) ----------------------------------------------
        # 기본 켜짐 + laptop 고정: 시선 lock 없이도 동적 제스처가 노트북으로 간다
        # (사용자 지시). 전구 시연 때는 이 토글을 끄거나 콤보를 room.bulb로 바꾼다.
        fallback_row = QHBoxLayout()
        self._fallback_toggle = QCheckBox("타깃 고정")
        self._fallback_toggle.setToolTip(
            "시선 판정을 우회하고 아래 기기에 항상 lock한다. 등록·조명 조건이 나빠 "
            "lock이 안 걸릴 때의 안전 폴백."
        )
        self._fallback_toggle.setChecked(True)
        self._fallback_toggle.toggled.connect(self._emit_fallback)
        fallback_row.addWidget(self._fallback_toggle)
        self._fallback_combo = QComboBox()
        for device_id in RUNTIME_DEVICE_IDS:
            self._fallback_combo.addItem(RUNTIME_DEVICE_LABELS[device_id], userData=device_id)
        # 기본 laptop 고정(사용자 지시, 2026-07-22) — 표시는 라벨이라 인덱스로 고른다.
        self._fallback_combo.setCurrentIndex(RUNTIME_DEVICE_IDS.index(LAPTOP_DEVICE_ID))
        self._fallback_combo.currentIndexChanged.connect(lambda _: self._emit_fallback())
        fallback_row.addWidget(self._fallback_combo, 1)
        layout.addLayout(fallback_row)

        # --- 임계값 프리셋 ------------------------------------------------
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("임계값"))
        self._preset_combo = QComboBox()
        for preset in DEMO_PRESETS:
            self._preset_combo.addItem(preset.label)
        self._preset_combo.setToolTip(
            "프리셋을 바꾸면 Fusion 엔진이 재생성되어 현재 target lock이 초기화된다."
        )
        self._preset_combo.currentIndexChanged.connect(self._emit_preset)
        preset_row.addWidget(self._preset_combo, 1)
        layout.addLayout(preset_row)

        # --- 물체 등록·관리 -------------------------------------------------
        # '시선 인식' 탭과 같은 등록 엔진을 그대로 쓴다(MainWindow가 콜백으로 잇는다) —
        # 시연 탭을 떠나지 않고 등록·재등록·이름변경·삭제까지 끝낼 수 있게.
        layout.addWidget(_section("물체 관리"))
        # MainWindow._clear_registration_state가 쓰는 대기 문구와 똑같이 시작한다 —
        # "2단계 물체 등록: 대기" 같은 내부 단계 이름만 있으면 맥락 없이 뜬금없어 보인다.
        self._registration_status_label = QLabel(
            "1단계에서는 물체 중앙 한 점을 응시한 채 고개·거리만 바꾸고, "
            "2단계에서는 고개를 고정한 채 눈으로 테두리를 정밀하게 따라갑니다."
        )
        self._registration_status_label.setWordWrap(True)
        self._registration_status_label.setStyleSheet(_STATUS_STYLE + " color:#8b949e;")
        layout.addWidget(self._registration_status_label)

        target_controls = QHBoxLayout()
        self._register_target_button = QPushButton("물체 등록")
        if on_register_target is not None:
            self._register_target_button.clicked.connect(on_register_target)
        self._reregister_target_button = QPushButton("재등록")
        self._reregister_target_button.clicked.connect(self._emit_reregister)
        self._rename_target_button = QPushButton("이름 변경")
        self._rename_target_button.clicked.connect(self._emit_rename)
        self._delete_target_button = QPushButton("삭제")
        self._delete_target_button.clicked.connect(self._emit_delete)
        self._cancel_registration_button = QPushButton("등록 취소")
        self._cancel_registration_button.setEnabled(False)
        if on_cancel_registration is not None:
            self._cancel_registration_button.clicked.connect(on_cancel_registration)
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
        self._target_list.setMaximumHeight(110)
        layout.addWidget(self._target_list)

        self._hand_status = QLabel(
            "HAND  미검출  det 0%\npalm scale  —\ntilt —   pose —\n제어  TCN 판정 대기 · 실행 ON"
        )
        self._hand_status.setWordWrap(True)
        self._hand_status.setStyleSheet(
            _STATUS_STYLE + " font-family:Consolas,monospace; font-weight:600;"
        )
        layout.addWidget(self._hand_status)

        layout.addStretch(1)

        # --- 판정 로그 ------------------------------------------------------
        # 위젯만 여기서 만든다. 웹캠 밑(MainWindow._build_demo_tab)에 두므로 이
        # 패널의 레이아웃에는 넣지 않는다 — `log_widget`으로 밖에서 배치한다.
        self._log_list = QListWidget()
        self._log_list.setStyleSheet(
            "QListWidget{background:#0a0d12; border:1px solid #30363d;"
            " font-family:Consolas,monospace; font-size:12px; color:#c9d1d9;}"
        )

    # --- 조작 → 콜백 -------------------------------------------------------

    def _current_target_id(self) -> str | None:
        row = self._target_list.currentRow()
        return self._target_ids[row] if 0 <= row < len(self._target_ids) else None

    def _emit_reregister(self) -> None:
        target_id = self._current_target_id()
        if target_id is not None and self._on_reregister_target is not None:
            self._on_reregister_target(target_id)

    def _emit_rename(self) -> None:
        target_id = self._current_target_id()
        if target_id is not None and self._on_rename_target is not None:
            self._on_rename_target(target_id)

    def _emit_delete(self) -> None:
        target_id = self._current_target_id()
        if target_id is not None and self._on_delete_target is not None:
            self._on_delete_target(target_id)

    def _emit_fallback(self) -> None:
        device = self._fallback_combo.currentData() if self._fallback_toggle.isChecked() else None
        self._on_fallback_changed(device)

    def _emit_execution(self, enabled: bool) -> None:
        self._execution_status.setText(
            "실행 활성 · 확정된 제스처를 실제 컴퓨터/전구에 전달"
            if enabled
            else "판정 전용 · 실제 컴퓨터/전구 명령은 실행하지 않음"
        )
        self._execution_status.setStyleSheet(
            ("color:#3fb950;" if enabled else "color:#8b949e;") + " font-weight:600;"
        )
        self._on_execution_toggled(enabled)

    def _emit_preset(self, index: int) -> None:
        if 0 <= index < len(DEMO_PRESETS):
            self._on_preset_changed(DEMO_PRESETS[index])

    # --- MainWindow가 밀어 넣는 상태 ---------------------------------------

    def set_hand_status(self, snapshot: HandSnapshot, *, execution_enabled: bool) -> None:
        """웹캠에서 옮긴 손 검출·자세·TCN 상태를 한 카드에 실시간 표시한다."""
        execution = "ON" if execution_enabled else "OFF"
        if not snapshot.hand_detected:
            self._hand_status.setText(
                "HAND  미검출  det 0%\n"
                "palm scale  —\n"
                "tilt —   pose —\n"
                f"제어  TCN 판정 대기 · 실행 {execution}"
            )
            self._hand_status.setStyleSheet(
                _STATUS_STYLE + " color:#8b949e; font-family:Consolas,monospace; font-weight:600;"
            )
            return

        source = "스무딩 검출" if snapshot.smoothed else "raw 검출"
        handedness = snapshot.handedness or "?"
        tilt = "?" if snapshot.palm_tilt_degrees is None else f"{snapshot.palm_tilt_degrees:.0f}°"
        pose = snapshot.pose
        if pose is None or not pose.label:
            pose_text = "pose —"
        elif pose.trusted:
            pose_text = f"{pose.label} {pose.confidence:.0%}"
        else:
            pose_text = f"거부 {pose.label} {pose.confidence:.0%} · {pose.reason}"

        if snapshot.pose_events:
            control = "동작 " + ", ".join(event.kind for event in snapshot.pose_events)
        elif snapshot.pose_state:
            control = f"상태 {snapshot.pose_state}"
        else:
            control = "TCN 판정 대기"
        self._hand_status.setText(
            f"HAND  {handedness}  det {snapshot.detection_confidence:.0%}  [{source}]\n"
            f"palm scale  {snapshot.palm_scale:.3f}\n"
            f"tilt {tilt}   {pose_text}\n"
            f"제어  {control} · 실행 {execution}"
        )
        color = "#f85149" if pose is not None and not pose.trusted else "#58a6ff"
        self._hand_status.setStyleSheet(
            _STATUS_STYLE + f" color:{color}; font-family:Consolas,monospace; font-weight:600;"
        )

    def set_targets(self, choices: Sequence[TargetChoice]) -> None:
        """물체 관리 목록을 다시 그린다(물체 등록·삭제·이름 변경 후 호출).

        선택 상태는 target_id 기준으로 유지한다 — 재등록·이름변경으로 표시 텍스트가
        바뀌어도 같은 물체가 선택된 채로 남게 한다.
        """
        selected_id = self._current_target_id()
        self._target_list.clear()
        self._target_ids = [choice.target_id for choice in choices]
        for choice in choices:
            kind = f" · {choice.device_type}" if choice.device_type else ""
            self._target_list.addItem(f"{choice.name} ({choice.target_id}{kind})")
        if selected_id is not None and selected_id in self._target_ids:
            self._target_list.setCurrentRow(self._target_ids.index(selected_id))

    def set_registration_active(self, *, active: bool) -> None:
        """등록 진행 중에는 등록/재등록/이름변경/삭제를 잠그고 취소만 연다."""
        for button in (
            self._register_target_button,
            self._reregister_target_button,
            self._rename_target_button,
            self._delete_target_button,
        ):
            button.setEnabled(not active)
        self._cancel_registration_button.setEnabled(active)

    def set_registration_status(self, text: str, *, active: bool) -> None:
        self._registration_status_label.setText(text)
        self._registration_status_label.setStyleSheet(
            "background:#3d2a12; color:#f0b429; border:1px solid #7a5a1e;"
            " border-radius:6px; padding:8px; font-weight:700;"
            if active
            else _STATUS_STYLE + " color:#8b949e;"
        )

    def set_state(
        self,
        *,
        locked: str | None,
        candidate: str | None,
        phase: str,
        gesture: str,
        suppressed: bool,
        raw_target: str | None = None,
    ) -> None:
        raw_text = f"실시간 시선: {raw_target}" if raw_target is not None else "실시간 시선: 없음"
        raw_color = "#58a6ff" if raw_target is not None else "#8b949e"
        self._raw_target_label.setText(raw_text)
        self._raw_target_label.setStyleSheet(_STATUS_STYLE + f" color:{raw_color};")

        if locked is not None:
            target_text = f"바라보는 기기: {locked} (LOCKED)"
            color = "#3fb950"
        elif candidate is not None:
            target_text = f"바라보는 기기: {candidate} (후보 — 응시 중)"
            color = "#d29922"
        else:
            target_text = "바라보는 기기: 없음"
            color = "#8b949e"
        if suppressed:
            target_text += " · 커서 제어 정지"
        self._target_label.setText(target_text)
        self._target_label.setStyleSheet(_STATUS_STYLE + f" color:{color};")
        self._gesture_label.setText(f"제스처: {gesture}")
        self._phase_label.setText(f"Intent: {phase}")

    def set_bulb(self, state: VirtualBulbState, *, badge: str, ok: bool) -> None:
        """명령 기준 문구 + 배지를 갱신한다. 원(BulbView)은 `set_bulb_live`가 맡는다."""
        self._bulb_state_label.setText(state.describe())
        self._bulb_badge.setText(f"실물: {badge}")
        self._bulb_badge.setStyleSheet(
            ("color:#3fb950;" if ok else "color:#d29922;") + " font-weight:700;"
        )

    def set_bulb_live(self, pilot: Mapping[str, object] | None) -> None:
        """`BulbPoller`가 읽어온 실물 상태로 원을 갱신한다. 조회 실패면 ``None``."""
        self.bulb_view.set_live_state(pilot)

    def set_last_action(self, text: str, *, ok: bool) -> None:
        self._last_action.setText(f"마지막 실행: {text}")
        self._last_action.setStyleSheet(
            _STATUS_STYLE + (" color:#3fb950;" if ok else " color:#8b949e;")
        )

    def append_line(self, text: str, *, ok: bool) -> None:
        """판정·실행 한 줄. 차단도 반드시 남긴다 — 왜 실행되지 않았는지가 시연의 절반이다."""
        self._log_list.addItem(text)
        item = self._log_list.item(self._log_list.count() - 1)
        item.setForeground(QColor("#3fb950" if ok else "#8b949e"))
        while self._log_list.count() > 300:
            self._log_list.takeItem(0)
        self._log_list.scrollToBottom()

    # --- 테스트·MainWindow 편의 -------------------------------------------

    @property
    def execution_enabled(self) -> bool:
        return self._execution_toggle.isChecked()

    def set_execution_enabled(self, enabled: bool) -> None:
        """실행기를 쓸 수 없을 때 MainWindow가 스위치를 안전하게 되돌린다."""
        if self._execution_toggle.isChecked() == enabled:
            return
        self._execution_toggle.setChecked(enabled)

    @property
    def fallback_device(self) -> str | None:
        return self._fallback_combo.currentData() if self._fallback_toggle.isChecked() else None

    @property
    def log_widget(self) -> QListWidget:
        """판정·실행 로그 위젯. 이 패널의 레이아웃에는 넣지 않고 밖에서 배치한다
        (MainWindow._build_demo_tab이 웹캠 밑에 둔다)."""
        return self._log_list


def _section(title: str) -> QLabel:
    label = QLabel(title)
    label.setStyleSheet("font-weight:700; color:#8b949e; padding-top:6px;")
    label.setFrameShape(QFrame.Shape.NoFrame)
    return label


__all__ = ["BulbView", "DemoPanel", "TargetChoice"]
