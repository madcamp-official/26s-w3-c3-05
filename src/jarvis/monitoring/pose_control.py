"""자세 상태기계 이벤트 → 실제 OS 입력. 디버깅 툴이 컴퓨터를 제어하는 지점.

`PoseStateMachine`은 순수 로직이라 아무것도 실행하지 않는다. 이 모듈이 그 이벤트를
`InputSink`(플랫폼별 실제 입력)로 옮긴다 — 판정과 실행을 분리해 두면 상태기계를
카메라·OS 없이 테스트할 수 있고, 실행을 끄고도 판정을 관찰할 수 있다.

이 경로는 사용자의 실제 데스크톱을 클릭하고 스크롤한다. 디버깅 툴에서 **기본 켜짐**이며
(사용자 지시), 양 탭(실시간·손 추적)의 토글이 상태를 공유한다. 분류 정확도 87.1%에
오발동 1.7%가 남아 있으므로, 끄고 싶을 때 바로 끌 수 있도록 토글을 눈에 띄게 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.gesture_fusion.pose_state import PoseEvent
from jarvis.runtime_protocol.adapters.windows import InputKey, InputSink, MouseButton

# 스크롤 이벤트는 매 프레임 나온다(30fps). 그대로 보내면 초당 30틱이라 너무 빠르므로
# 이 간격으로 솎아낸다. 값이 커지면 스크롤이 느려지고, 작아지면 걷잡을 수 없어진다.
SCROLL_INTERVAL_MS = 60
SCROLL_TICKS = 2  # 한 스크롤 스텝당 이동량. 2 = 기존 대비 스크롤 이동 속도 2배


def _transition_key() -> InputKey:
    """주먹→보 전이에 쓸 키. macOS는 Mission Control, Windows는 Task View(Win+Tab)."""
    import sys

    return InputKey.TASK_VIEW if sys.platform.startswith("win") else InputKey.MISSION_CONTROL


def default_input_sink() -> InputSink | None:
    """플랫폼에 맞는 입력 싱크. 지원하지 않는 OS에서는 None(제어를 흉내내지 않는다)."""
    import sys

    if sys.platform == "darwin":
        from jarvis.runtime_protocol.adapters.macos import MacOSInputSink

        return MacOSInputSink()
    if sys.platform.startswith("win"):
        from jarvis.runtime_protocol.adapters.windows import Win32InputSink

        return Win32InputSink()
    return None


@dataclass
class PoseControlBridge:
    """이벤트를 입력으로 실행하고, 무엇을 했는지 기록한다(UI 표시용)."""

    sink: InputSink | None = None
    enabled: bool = False
    last_action: str = ""
    _last_scroll_ms: int = 0
    _dragging: bool = False
    history: list[str] = field(default_factory=list)
    transition_key: InputKey = field(default_factory=_transition_key)

    def apply(self, events: list[PoseEvent]) -> None:
        """이벤트를 실행한다. 꺼져 있으면 기록만 남기고 아무것도 실행하지 않는다."""
        for event in events:
            # 스크롤은 30fps 그대로면 감당이 안 돼 솎아낸다 — 실행·기록 **양쪽**에
            # 적용해야 한다(예전엔 스로틀이 _describe에만 있어 실행 판단과 얽혔다).
            if event.kind == "scroll":
                if event.timestamp_ms - self._last_scroll_ms < SCROLL_INTERVAL_MS:
                    continue
                self._last_scroll_ms = event.timestamp_ms
            if self.enabled and self.sink is not None:
                self._execute(event)
            label = self._describe(event)
            if not label:
                continue  # move처럼 실행은 하되 UI 기록만 생략하는 경우
            self.last_action = label if self.enabled else f"{label} (실행 안 함)"
            self.history.append(self.last_action)
            del self.history[:-20]

    def release(self) -> None:
        """드래그 중 손을 놓치거나 제어를 끌 때 버튼이 눌린 채로 남지 않게 한다."""
        if self._dragging and self.sink is not None:
            self.sink.press(MouseButton.LEFT, down=False)
        self._dragging = False
        # macOS에서 커서가 하단에 머문 채 제어가 꺼졌을 때 Dock이 노출된 채 남지 않게
        # 원래(자동숨김)로 복구한다. Windows 등 다른 sink엔 없는 메서드라 있으면 부른다.
        restore_dock = getattr(self.sink, "restore_dock", None)
        if callable(restore_dock):
            restore_dock()

    def _describe(self, event: PoseEvent) -> str:
        if event.kind == "move":
            return ""  # 커서 이동은 매 프레임이라 기록/표시하지 않는다(로그가 뒤덮인다)
        if event.kind == "scroll":
            return f"스크롤 {'위' if event.value > 0 else '아래'}"
        if event.kind == "mouse_down":
            # down 시점엔 클릭/드래그가 아직 안 갈린다(누른 시간으로 결정) — 중립적으로
            # '버튼 ↓'로 적되, clickState=2면 더블클릭 눌림이므로 그렇게 표시한다.
            return "더블클릭 ↓" if event.value >= 2 else "버튼 ↓"
        if event.kind == "mouse_up":
            return "버튼 ↑"
        return {
            "right_click": "우클릭",
            # 주먹→보 전이. macOS=Mission Control, Windows=Task View.
            "media_toggle": (
                "미션 컨트롤" if self.transition_key is InputKey.MISSION_CONTROL
                else "태스크 뷰" if self.transition_key is InputKey.TASK_VIEW
                else "재생/일시정지"
            ),
            "close_tab": "탭 닫기",
            "play_pause": "재생/일시정지",
            "volume_up": "볼륨 +",
            "volume_down": "볼륨 −",
            # 두 손가락 좌↔우 전이 스와이프 → 가상 데스크톱 전환(정적 경로, 실험).
            "desktop_prev": "데스크톱 이전",
            "desktop_next": "데스크톱 다음",
        }.get(event.kind, "")

    def _execute(self, event: PoseEvent) -> None:
        assert self.sink is not None
        if event.kind == "move":
            # delta는 절대 픽셀 이동량이다 — 화면 해상도로 스케일하지 않는다. 같은 손동작은
            # 어떤 기기·해상도에서도 같은 픽셀 수만큼 커서를 옮긴다("절대 길이" 감도). 체감
            # 속도는 gain(CURSOR_BASE_GAIN) 하나로만 조절한다. 드래그 중이면 sink가 드래그
            # 이벤트를 보내 창이 실시간으로 따라오게 한다.
            self.sink.move_cursor(
                round(event.delta[0]), round(event.delta[1]), dragging=self._dragging
            )
        elif event.kind == "mouse_down":
            # 핀치 진입 → 버튼 down. clickState를 실어 더블클릭이면 OS가 합치게 한다.
            # 버튼이 눌린 동안(_dragging) move는 드래그로 나가고, release가 안전하게 뗀다.
            self.sink.press(MouseButton.LEFT, down=True, click_state=int(event.value) or 1)
            self._dragging = True
        elif event.kind == "mouse_up":
            self.sink.press(MouseButton.LEFT, down=False, click_state=int(event.value) or 1)
            self._dragging = False
        elif event.kind == "right_click":
            self.sink.click(MouseButton.RIGHT)
        elif event.kind == "media_toggle":
            self.sink.tap_key(self.transition_key)
        elif event.kind == "close_tab":
            self.sink.tap_key(InputKey.CLOSE_TAB)
        elif event.kind == "play_pause":
            self.sink.tap_key(InputKey.PLAY_PAUSE)
        elif event.kind == "volume_up":
            self.sink.tap_key(InputKey.VOLUME_UP)
        elif event.kind == "volume_down":
            self.sink.tap_key(InputKey.VOLUME_DOWN)
        elif event.kind == "scroll":
            self.sink.scroll(SCROLL_TICKS if event.value > 0 else -SCROLL_TICKS)
        elif event.kind == "desktop_prev":
            self.sink.switch_desktop(forward=False, repeat=1)
        elif event.kind == "desktop_next":
            self.sink.switch_desktop(forward=True, repeat=1)
