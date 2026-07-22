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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.monitoring.demo_bridge import (
    DEMO_PRESETS,
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
from jarvis.runtime_protocol.adapters.wiz import hue_to_rgb

_NO_DEVICE = "(연결 안 함)"

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
    """매핑 UI 한 줄 — 등록된 물체 하나."""

    target_id: str
    name: str
    device_type: str = ""


class BulbView(QWidget):
    """가상 전구 — 전원·밝기·색온도를 원 하나의 색과 밝기로 그린다.

    실물 상태를 읽어오지 않는다. 그린 값은 지금까지 **보낸 명령의 누적**이다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(120, 120)
        self._state = VirtualBulbState()

    def set_state(self, state: VirtualBulbState) -> None:
        self._state = state
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height()) - 12
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        painter.setPen(QColor("#30363d"))
        painter.setBrush(self._bulb_color())
        painter.drawEllipse(x, y, size, size)
        painter.end()

    def _bulb_color(self) -> QColor:
        if not self._state.power:
            return QColor("#21262d")
        red, green, blue = self._tint()
        # 밝기는 명도로. 하한이 10이라 완전히 검어지지는 않는다(꺼짐과 구분).
        level = (self._state.brightness - BRIGHTNESS_MIN) / (BRIGHTNESS_MAX - BRIGHTNESS_MIN)
        scale = 0.35 + 0.65 * max(0.0, min(1.0, level))
        return QColor(int(red * scale), int(green * scale), int(blue * scale))

    def _tint(self) -> tuple[int, int, int]:
        """색조. 실물 WiZ와 같이 색상 모드와 색온도 모드 중 하나를 따른다."""
        if self._state.color_mode:
            # 색상 모드: adapter가 기기로 보내는 것과 **같은** 변환을 쓴다 — 화면과
            # 실물이 서로 다른 색을 내면 시연에서 바로 들통난다.
            return hue_to_rgb(self._state.hue)
        # 색온도 모드: 따뜻한 주황(2700K) ↔ 차가운 흰(6500K) 사이 선형 보간.
        span = COLOR_TEMPERATURE_MAX - COLOR_TEMPERATURE_MIN
        warmth = (self._state.color_temperature - COLOR_TEMPERATURE_MIN) / span
        warmth = max(0.0, min(1.0, warmth))
        return 255, int(170 + 70 * warmth), int(90 + 155 * warmth)


class DemoPanel(QWidget):
    """시연 탭 본문 — 상태 스트립 + 가상 전구 + 설정 + 판정 로그."""

    def __init__(
        self,
        *,
        on_mapping_changed: Callable[[str, str | None], None],
        on_fallback_changed: Callable[[str | None], None],
        on_preset_changed: Callable[[DemoPreset], None],
        on_execution_toggled: Callable[[bool], None],
        on_open_registration: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_mapping_changed = on_mapping_changed
        self._on_fallback_changed = on_fallback_changed
        self._on_preset_changed = on_preset_changed
        self._on_execution_toggled = on_execution_toggled
        self._mapping_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)

        # --- 상태 스트립 -------------------------------------------------
        self._target_label = QLabel("바라보는 기기: -")
        self._gesture_label = QLabel("제스처: -")
        self._phase_label = QLabel("Intent: IDLE")
        for label in (self._target_label, self._gesture_label, self._phase_label):
            label.setStyleSheet(_STATUS_STYLE)
            layout.addWidget(label)

        self._last_action = QLabel("마지막 실행: 없음")
        self._last_action.setWordWrap(True)
        self._last_action.setStyleSheet(_STATUS_STYLE + " color:#8b949e;")
        layout.addWidget(self._last_action)

        # --- 가상 전구 ----------------------------------------------------
        bulb_row = QHBoxLayout()
        self.bulb_view = BulbView()
        bulb_row.addWidget(self.bulb_view)
        bulb_text = QVBoxLayout()
        self._bulb_state_label = QLabel("밝기 60% · 색온도 4000K")
        self._bulb_state_label.setStyleSheet(f"font-weight:700; color:{_STATUS_TEXT};")
        bulb_text.addWidget(self._bulb_state_label)
        bulb_note = QLabel("위 값은 보낸 명령 기준이며 실물 응답이 아닙니다.")
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
        self._execution_toggle = QCheckBox(
            "기기 명령 실행 (TCN 동적 제스처 · 끄면 판정만 하고 실행하지 않음)"
        )
        self._execution_toggle.setChecked(False)
        self._execution_toggle.toggled.connect(self._emit_execution)
        layout.addWidget(self._execution_toggle)
        self._execution_status = QLabel(
            "판정 전용 · 실제 컴퓨터/전구 명령은 실행하지 않음"
        )
        self._execution_status.setStyleSheet("color:#8b949e; font-weight:600;")
        layout.addWidget(self._execution_status)

        # --- 폴백(타깃 고정) ----------------------------------------------
        fallback_row = QHBoxLayout()
        self._fallback_toggle = QCheckBox("타깃 고정")
        self._fallback_toggle.setToolTip(
            "시선 판정을 우회하고 아래 기기에 항상 lock한다. 등록·조명 조건이 나빠 "
            "lock이 안 걸릴 때의 안전 폴백."
        )
        self._fallback_toggle.toggled.connect(self._emit_fallback)
        fallback_row.addWidget(self._fallback_toggle)
        self._fallback_combo = QComboBox()
        for device_id in RUNTIME_DEVICE_IDS:
            self._fallback_combo.addItem(RUNTIME_DEVICE_LABELS[device_id], userData=device_id)
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

        # --- 물체 → 기기 매핑 ----------------------------------------------
        layout.addWidget(_section("등록 물체 → 런타임 기기"))
        mapping_note = QLabel(
            "물체 등록은 '시선 인식' 탭에서 진행합니다. 등록할 때 고른 computer / "
            "electric bulb 기종은 아래 laptop / room.bulb 실행 대상으로 자동 연결되며, "
            "필요할 때 여기서 바꿀 수 있습니다."
        )
        mapping_note.setWordWrap(True)
        mapping_note.setStyleSheet("color:#6e7681; font-size:11px;")
        layout.addWidget(mapping_note)
        self._registration_button = QPushButton("기기 등록·관리 → 시선 인식 탭")
        if on_open_registration is not None:
            self._registration_button.clicked.connect(on_open_registration)
        layout.addWidget(self._registration_button)
        self._mapping_container = QWidget()
        self._mapping_layout = QGridLayout(self._mapping_container)
        self._mapping_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._mapping_container)

        # --- 판정 로그 ------------------------------------------------------
        layout.addWidget(_section("판정·실행 로그"))
        self._log_list = QListWidget()
        self._log_list.setStyleSheet(
            "QListWidget{background:#0a0d12; border:1px solid #30363d;"
            " font-family:Consolas,monospace; font-size:12px; color:#c9d1d9;}"
        )
        layout.addWidget(self._log_list, 1)

    # --- 조작 → 콜백 -------------------------------------------------------

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

    def set_targets(self, choices: Sequence[TargetChoice], mapping: dict[str, str]) -> None:
        """매핑 표를 다시 그린다(물체 등록·삭제·이름 변경 후 호출)."""
        while self._mapping_layout.count():
            item = self._mapping_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._mapping_combos.clear()

        if not choices:
            self._mapping_layout.addWidget(
                QLabel("등록된 물체가 없습니다 — '시선 인식' 탭에서 먼저 등록하세요."), 0, 0
            )
            return

        for row, choice in enumerate(choices):
            kind = f" · {choice.device_type}" if choice.device_type else ""
            self._mapping_layout.addWidget(
                QLabel(f"{choice.name} ({choice.target_id}{kind})"), row, 0
            )
            combo = QComboBox()
            combo.addItem(_NO_DEVICE, userData=None)
            for device_id in RUNTIME_DEVICE_IDS:
                combo.addItem(RUNTIME_DEVICE_LABELS[device_id], userData=device_id)
            current = mapping.get(choice.target_id)
            if current is not None:
                index = combo.findData(current)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(
                lambda _index, target_id=choice.target_id, source=combo: self._on_mapping_changed(
                    target_id, source.currentData()
                )
            )
            self._mapping_combos[choice.target_id] = combo
            self._mapping_layout.addWidget(combo, row, 1)

    def set_state(
        self,
        *,
        locked: str | None,
        candidate: str | None,
        phase: str,
        gesture: str,
        suppressed: bool,
    ) -> None:
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
        self.bulb_view.set_state(state)
        self._bulb_state_label.setText(state.describe())
        self._bulb_badge.setText(f"실물: {badge}")
        self._bulb_badge.setStyleSheet(
            ("color:#3fb950;" if ok else "color:#d29922;") + " font-weight:700;"
        )

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


def _section(title: str) -> QLabel:
    label = QLabel(title)
    label.setStyleSheet("font-weight:700; color:#8b949e; padding-top:6px;")
    label.setFrameShape(QFrame.Shape.NoFrame)
    return label


__all__ = ["BulbView", "DemoPanel", "TargetChoice"]
